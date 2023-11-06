import io
import os
import sys
from tqdm import tqdm
# sys.path.append('/work/ws-tmp/g059598-Vo/Vo_code/ptlflow/ptlflow')
from argparse import Namespace
from PIL import Image
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import torch
from torch import hub
import datetime
import torch.optim as optim
from torchvision import transforms as Tran
from torchvision import transforms
from torch.autograd import Variable

from torchvision.utils  import make_grid, flow_to_image
from model.helper import *

from model.pcnn import *
from model.flow_dataset import VisualOdometryDataLoader

from torch.utils.tensorboard import SummaryWriter

# if torch.cuda.is_available():
#     os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:4024"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device, os.cpu_count())

FLOAT = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
K = 100.



def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
    return Variable(
        torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
    ).type(dtype)

def get_loss(pred, y):
        #import ipdb;ipdb.set_trace()
        angle_loss = torch.nn.functional.mse_loss(pred[:,:3], y[:,:3])
        translation_loss = torch.nn.functional.mse_loss(pred[:,3:], y[:,3:])
        loss = (angle_loss + translation_loss)
        return loss

def train_model(model,train_loader, optimizer, epoch,length,args,scheduler):
    # switch to train mode
    running_loss = 0.0


    for batch_idx, (images_stacked, odometries_stacked) in enumerate(tqdm(train_loader)):
        images_stacked, odometries_stacked = images_stacked.to(device).permute(0,3,1,2), odometries_stacked.to(device)
        estimated_odometry = model(images_stacked/255.0)
        loss = get_loss(estimated_odometry, odometries_stacked)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    writer.add_scalar("Loss/train", loss, epoch )
    if epoch % 10 == 0:    # print every 10 epochs
        print('[%d] loss: %.5f' % 
                    (epoch, running_loss / 150),flush=True)
        running_loss = 0.0
    
    if epoch % 500 == 0 or epoch == args.train_iter:
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
                }
        torch.save(state, os.path.join(args.checkpoint_path, "{0}_{1}.ckpt".format(args.model,epoch)))

def train(model, datapath, checkpoint_path, epochs,args):
    model.train()
    model.training = True
    kwargs = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}
    
    train_loader = torch.utils.data.DataLoader(VisualOdometryDataLoader(datapath), batch_size=args.bsize, shuffle=True, drop_last=True, **kwargs)
    length =len(train_loader)
    if args.model == 'Pcnn':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler =optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
    writer.add_scalar('hyperparameters/learning_rate', args.lr)
    writer.add_scalar('hyperparameters/batch_size', args.bsize)
    writer.add_scalar('hyperparameters/num_epochs', epochs)
    for epoch in range(1, epochs + 1):
        # train for one epoch
        train_model(model, train_loader, optimizer, epoch,length,args,scheduler )
        scheduler.step()
    writer.flush()

def test_model(model,test_loader,length,args):
    outTraj =[]
    gtTraj =[]
    T = np.eye(4)
    gtT = np.eye(4)
    gtTraj.append(gtT)
    outTraj.append(T)

    estimatedFrame = 0
    gtFrame = 0
    with torch.no_grad():
        for images_stacked, odometries_stacked in tqdm(test_loader):
            images_stacked, odometries_stacked = images_stacked.to(device).permute(0,3,1,2), odometries_stacked.to(device)
            
            if estimatedFrame == 10 :
                grid = make_grid(images_stacked)
                writer.add_image('flow_images', grid, 0)
            # import ipdb; ipdb.set_trace()
            estimated_odometry = model(images_stacked/255.0)

            loss = get_loss(estimated_odometry, odometries_stacked)
            writer.add_scalar("Loss/test", loss,estimatedFrame)
            for pred in estimated_odometry.cpu().numpy():
                R = eulerAnglesToRotationMatrix(pred[3:])
                t = pred[:3].reshape(3, 1)
                T_r = np.concatenate((np.concatenate([R, t], axis=1), [[0.0, 0.0, 0.0, 1.0]]), axis=0)

                # With respect to the first frame
                T_abs = np.dot(T, T_r)
                # Update the T matrix till now.
                T = T_abs
                outTraj.append(T)
                # Get the origin of the frame (i+1), ie the camera center
                estimatedFrame += 1
            for gt in odometries_stacked.cpu().numpy():
                R = eulerAnglesToRotationMatrix(gt[3:])
                t = gt[:3].reshape(3, 1)
                gtT_r = np.concatenate((np.concatenate([R, t], axis=1), [[0.0, 0.0, 0.0, 1.0]]), axis=0)
                # With respect to the first frame
                gtT_abs = np.dot(gtT, gtT_r)
                # Update the T matrix till now.
                gtT = gtT_abs
                gtTraj.append(gtT)
                # Get the origin of the frame (i+1), ie the camera center
                gtFrame += 1
        return np.array(gtTraj),np.array(outTraj)

def test(model, datapath,args):
    kwargs = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}
    test_loader =torch.utils.data.DataLoader(VisualOdometryDataLoader(datapath,test=True, seq = args.seq), batch_size=args.bsize, shuffle=False, drop_last=True, **kwargs)
    length =len(test_loader)*args.bsize
    gt_abs,out_abs = test_model(model, test_loader,length,args)
    visualise(np.mat(gt_abs[:,0:3,3]),np.mat(out_abs[:,0:3,3]))
    save_pred( args.save_res,out_abs, args.seq+'.txt')
   
def visualise(gt,out):
    fig = plot_route(gt, out, 'r', 'b')
    writer.add_figure('plot',fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch on Place Recognition + Visual Odometry')
    parser.add_argument('--model', default='Pcnn', type = str) 
    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--datapath', default='datapath', type=str, help='path KITII odometry dataset')
    parser.add_argument('--bsize', default=8, type=int, help='minibatch size')
    parser.add_argument('--trajectory_length', default=2, type=int, help='trajectory length')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.0001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--weight_decay', type=float, default=0.0005, metavar='M', help='momentum (default: 0.0005)')
    parser.add_argument('--tau', default=0.0001, type=float, help='moving average for target network')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--train_iter', default=200, type=int, help='train iters each timestep')
    parser.add_argument('--seq',default ='00',type= str,help= 'which seq to test, default 00')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--checkpoint_path', default='/work/ws-tmp/g059598-Vo/Vo_code/PCNN2/PCNN/checkpoint/raft', type=str, help='Checkpoint path')
    parser.add_argument('--checkpoint', default=None, type=str, help='Checkpoint')
    parser.add_argument('--checkpoint_load',default='/work/ws-tmp/g059598-Vo/Vo_code/PCNN2/PCNN/checkpoint/model.pth', type=str, help='Checkpoint_load')
    parser.add_argument('--save_res',default='/work/ws-tmp/g059598-Vo/Vo_code/PCNN2/PCNN/results/', type=str, help='save_test_result')

    args = parser.parse_args()
    print(args)
    current_datetime = datetime.datetime.now()
    current_date = current_datetime.strftime("%Y-%m-%d %H:%M")
    parent_dir = '/work/ws-tmp/g059598-Vo/Vo_code/PCNN2/PCNN/runs'
    directory = args.model+args.mode+current_date
    path = os.path.join(parent_dir, directory)
    if not os.path.exists(path):
        os.mkdir(path)
    writer = SummaryWriter(log_dir=path)



    # args2 = Namespace(autoflow_root_dir='/path/to/autoflow', corr_levels=4, corr_radius=4, dropout=0.0, flow_format='original', flying_chairs2_root_dir='/path/to/Vo_code/FlyingChairs2', flying_chairs_root_dir='/work/ws-tmp/g059598-vo/Vo_code/ptlflow/datasets/FlyingChairs_release', flying_things3d_root_dir='/work/ws-tmp/g059598-vo/Vo_code/ptlflow/datasets/FlyingThings3D', flying_things3d_subset_root_dir='/path/to/FlyingThings3D_subset', gamma=0.8, hd1k_root_dir='/path/to/HD1K', iters=12, kitti_2012_root_dir='/path/to/KITTI/2012', kitti_2015_root_dir='/work/ws-tmp/g059598-vo/Vo_code/ptlflow/datasets/KITTI/2015', lr=0.0001, max_flow=1000.0, max_forward_side=None, max_samples=None, max_show_side=1000, model='raft_small', mpi_sintel_root_dir='/work/ws-tmp/g059598-vo/Vo_code/ptlflow/datasets/MPI-Sintel', output_path='/work/ws-tmp/g059598-vo/Vo_code/ptlflow/outputs/', pretrained_ckpt='things', reversed=False, selection=None, show=False, test_dataset=None, train_batch_size=0, train_crop_size=None, train_dataset=None, train_num_workers=4, train_transform_cuda=False, train_transform_fp16=False, val_dataset='chairs', wdecay=0.0001, write_outputs=False)

    if args.model == 'Pcnn':
        model = Pcnn()
    elif args.model =='Pcnn1':
        model =Pcnn1()
    elif args.model == 'Pcnn2':
        model = Pcnn2()
          
    model = load_checkpoint_to_model(args,model)

    model.to(device)


    args = parser.parse_args()
    if args.mode == 'train':
        print("i am training")
        train(model, args.datapath, args.checkpoint_path, args.train_iter,args)

    elif args.mode == 'test':
        test(model, args.datapath,args)
    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))