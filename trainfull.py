import io
import os
import sys
from tqdm import tqdm
sys.path.append('/work/ws-tmp/g059598-Vo/Vo_code/ptlflow/ptlflow')
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

from torchvision.utils  import make_grid
from model.helper import *

from model.pcnn import *
from model.flow_to_image import *
from model.dataset2 import VisualOdometryDataLoader

from torch.utils.tensorboard import SummaryWriter

# if torch.cuda.is_available():
#     os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:4024"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    # Set the default GPUs to use (change the indices as needed)
    device1 = torch.device("cuda:0")
    # device2 = torch.device("cuda:1")
else:
    device1 = torch.device("cpu")
    # device2 = torch.device("cpu")

# print(device, os.cpu_count())

FLOAT = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
K = 100.

# def normalize_euler_angles_torch(euler_angles):
#     two_pi = 2 * torch.tensor(np.pi, dtype=euler_angles.dtype, device=euler_angles.device)
#     normalized_angles = (euler_angles + torch.tensor(np.pi, dtype=euler_angles.dtype, device=euler_angles.device)) % two_pi - torch.tensor(np.pi, dtype=euler_angles.dtype, device=euler_angles.device)
#     return normalized_angles

# def normalize_euler_angles(euler_angles):
#     normalized_angles = np.mod(euler_angles + np.pi, 2*np.pi) - np.pi
#     return normalized_angles

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

def train_model(model, flowmodel,flowimagemodel,train_loader, optimizer, epoch,length,args,scheduler):
    # switch to train mode
    running_loss = 0.0

    for batch_idx, (images_stacked, odometries_stacked) in enumerate(tqdm(train_loader)):
        images_stacked, odometries_stacked = images_stacked.to(device1), odometries_stacked.to(device1)
        images_stacked = images_stacked.permute(1, 0, 2, 3, 4,5)        
        flow_output = flowmodel(images_stacked[0])
        flow_output['flows']= flow_output['flows'].permute(1,0,2,3,4)
        flow_image = flowimagemodel(flow_output['flows'][0])
        estimated_odometry = model(flow_image)
        loss = get_loss(estimated_odometry, odometries_stacked)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 10 == 9:    # print every 10 mini-batches   
            print('[%d, %5d] loss: %.3f' % 
                      (epoch, batch_idx + 1, running_loss / 10),flush=True)
            running_loss = 0.0
        writer.add_scalar("Loss/train", loss, epoch * (length // args.bsize) + batch_idx)
    if epoch % 10 ==0 or epoch == args.train_iter:
        state = {
            'epoch': epoch,
            'pcnn_state_dict': model.state_dict(),
            'flow_state_dict':flowmodel.state_dict(),
            'flow_image_state_dict' : flowimagemodel.state_dict()

                }
        torch.save(state, os.path.join(args.checkpoint_path, "{0}_{1}.ckpt".format(args.model,epoch)))

def train(model,flowmodel,flowimagemodel, datapath, checkpoint_path, epochs,preprocess, args):
    model.train()
    flowmodel.train()
    flowimagemodel.train()
    kwargs = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = torch.utils.data.DataLoader(VisualOdometryDataLoader(datapath, transform=preprocess), batch_size=args.bsize, shuffle=True, drop_last=True, **kwargs)
    length =len(train_loader)
    optimizer_params = [ {'params': flowmodel.parameters()}, #Parameters for fc1
    {'params': flowimagemodel.parameters()}, # Parameters for fc2
    {'params': model.parameters()} ] 
    optimizer = optim.Adam(optimizer_params,lr=args.lr, weight_decay=args.weight_decay)
    scheduler =optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    writer.add_scalar('hyperparameters/learning_rate', args.lr)
    writer.add_scalar('hyperparameters/batch_size', args.bsize)
    writer.add_scalar('hyperparameters/num_epochs', epochs)
    for epoch in range(1, epochs + 1):
        # train for one epoch
        train_model(model,flowmodel,flowimagemodel, train_loader, optimizer, epoch,length,args,scheduler )
        scheduler.step()
    writer.flush()

def test_model(model,flowmodel,flowimagemodel, test_loader,length,args):
    outTraj =[]
    gtTraj =[]
    T = np.eye(4)
    gtT = np.eye(4)
    gtTraj.append(gtT)
    outTraj.append(T)

    estimatedFrame = 0
    gtFrame = 0

    height = 48;
    width = 156;
    new_img_size = [height,width]
    transform = Tran.Resize(new_img_size)
    with torch.no_grad():
        for images_stacked, odometries_stacked in tqdm(test_loader):
            images_stacked, odometries_stacked = images_stacked.to(device1), odometries_stacked.to(device1)
            images_stacked = images_stacked.permute(1, 0, 2, 3, 4,5)        
            flow_output = flowmodel(images_stacked[0])
            flow_output['flows']= flow_output['flows'].permute(1,0,2,3,4)
            flow_image = flowimagemodel(flow_output['flows'][0])
            estimated_odometry = model(flow_image)
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

def test(model,flowmodel,flowimagemodel, datapath, preprocess,args):
    kwargs = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}
    test_loader =torch.utils.data.DataLoader(VisualOdometryDataLoader(datapath,test=True, seq = args.seq, transform=preprocess), batch_size=args.bsize, shuffle=False, drop_last=True, **kwargs)
    length =len(test_loader)*args.bsize
    gt_abs,out_abs = test_model(model,flowmodel,flowimagemodel, test_loader,length,args)
    visualise(np.mat(gt_abs[:,0:3,3]),np.mat(out_abs[:,0:3,3]))
    save_pred( args.save_res,out_abs, args.seq+'.txt')
   
def visualise(gt,out):
    fig = plot_route(gt, out, 'r', 'b')
    writer.add_figure('plot',fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch on Place Recognition + Visual Odometry')
    parser.add_argument('--flowmodel', default='FlowNet2SS', type = str) 
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

    preprocess = transforms.Compose([
        transforms.Resize((192,640)),
        transforms.ToTensor()
    ])


    # args2 = Namespace(autoflow_root_dir='/path/to/autoflow', corr_levels=4, corr_radius=4, dropout=0.0, flow_format='original', flying_chairs2_root_dir='/path/to/Vo_code/FlyingChairs2', flying_chairs_root_dir='/work/ws-tmp/g059598-vo/Vo_code/ptlflow/datasets/FlyingChairs_release', flying_things3d_root_dir='/work/ws-tmp/g059598-vo/Vo_code/ptlflow/datasets/FlyingThings3D', flying_things3d_subset_root_dir='/path/to/FlyingThings3D_subset', gamma=0.8, hd1k_root_dir='/path/to/HD1K', iters=12, kitti_2012_root_dir='/path/to/KITTI/2012', kitti_2015_root_dir='/work/ws-tmp/g059598-vo/Vo_code/ptlflow/datasets/KITTI/2015', lr=0.0001, max_flow=1000.0, max_forward_side=None, max_samples=None, max_show_side=1000, model='raft_small', mpi_sintel_root_dir='/work/ws-tmp/g059598-vo/Vo_code/ptlflow/datasets/MPI-Sintel', output_path='/work/ws-tmp/g059598-vo/Vo_code/ptlflow/outputs/', pretrained_ckpt='things', reversed=False, selection=None, show=False, test_dataset=None, train_batch_size=0, train_crop_size=None, train_dataset=None, train_num_workers=4, train_transform_cuda=False, train_transform_fp16=False, val_dataset='chairs', wdecay=0.0001, write_outputs=False)
    model = Pcnn()

    if args.mode == 'train':      
        flowmodel = load_flow_model(args)
        flowimagemodel = FlowImage()
        model = load_checkpoint_to_model(args,model)

    elif args.mode == 'test':
    
        checkpoint = torch.load(args.checkpoint_load, map_location=torch.device('cpu'))

        if args.flowmodel == 'FlowNet2SS':
            args2 = Namespace(accelerator=None, accumulate_grad_batches=None, amp_backend='native', amp_level=None, auto_lr_find=False, auto_scale_batch_size=False, auto_select_gpus=False, autoflow_root_dir='/path/to/autoflow', batch_norm=False, benchmark=False, check_val_every_n_epoch=1, checkpoint_callback=None, clear_train_state=True, default_root_dir=None, detect_anomaly=False, deterministic=False, devices=None, div_flow=20.0, enable_checkpointing=True, enable_model_summary=True, enable_progress_bar=True, fast_dev_run=False, flush_logs_every_n_steps=None, flying_chairs2_root_dir='/path/to/FlyingChairs2', flying_chairs_root_dir='/work/ws-tmp/g059598-vo/ptlflow/datasets/FlyingChairs_release', flying_things3d_root_dir='/work/ws-tmp/g059598-vo/ptlflow/datasets/FlyingThings3D', flying_things3d_subset_root_dir='/path/to/FlyingThings3D_subset', gpus=None, gradient_clip_algorithm=None, gradient_clip_val=None, hd1k_root_dir='/path/to/HD1K', input_channels=6, ipus=None, kitti_2012_root_dir='/path/to/KITTI/2012', kitti_2015_root_dir='/work/ws-tmp/g059598-vo/ptlflow/datasets/KITTI/2015', limit_predict_batches=1.0, limit_test_batches=1.0, limit_train_batches=1.0, limit_val_batches=1.0, log_dir='ptlflow_logs', log_every_n_steps=50, log_gpu_memory=None, logger=True, loss_base_weight=0.32, loss_norm='L2', loss_num_scales=5, loss_start_scale=4, lr=0.0001, max_epochs=None, max_steps=-1, max_time=None, min_epochs=None, min_steps=None, model='flownet2ss', model_name=0, move_metrics_to_cpu=False, mpi_sintel_root_dir='/work/ws-tmp/g059598-vo/ptlflow/datasets/MPI-Sintel', multiple_trainloader_mode='max_size_cycle', num_nodes=1, num_processes=1, num_sanity_val_steps=2, overfit_batches=0.0, plugins=None, precision=32, prepare_data_per_node=None, pretrained_ckpt='/work/ws-tmp/g059598-vo/FlowNet2-S_checkpoint.pth.tar', process_position=0, profiler=None, progress_bar_refresh_rate=None, random_seed=1234, reload_dataloaders_every_epoch=False, reload_dataloaders_every_n_epochs=0, replace_sampler_ddp=True, resume_from_checkpoint=None, stochastic_weight_avg=False, strategy=None, sync_batchnorm=False, terminate_on_nan=None, test_dataset=None, tpu_cores=None, track_grad_norm=-1, train_batch_size=0, train_crop_size=None, train_dataset='chairs', train_num_workers=4, train_transform_cuda=False, train_transform_fp16=False, val_check_interval=1.0, val_dataset='sintel', wdecay=0.001, weights_save_path=None, weights_summary='top')
            flowmodel = FlowNet2SS(args2)
            flowmodel.load_state_dict(checkpoint['flow_state_dict'], strict=True)
        
        elif args.flowmodel =='RAFT':
            print("loaded raft")
            args2 =Namespace(autoflow_root_dir='/path/to/autoflow', corr_levels=4, corr_radius=4, dropout=0.0, flow_format='original', flying_chairs2_root_dir='/path/to/Vo_code/FlyingChairs2', flying_chairs_root_dir='/work/ws-tmp/g059598-vo/Vo_code/ptlflow/datasets/FlyingChairs_release', flying_things3d_root_dir='/work/ws-tmp/g059598-vo/Vo_code/ptlflow/datasets/FlyingThings3D', flying_things3d_subset_root_dir='/path/to/FlyingThings3D_subset', gamma=0.8, hd1k_root_dir='/path/to/HD1K', iters=12, kitti_2012_root_dir='/path/to/KITTI/2012', kitti_2015_root_dir='/work/ws-tmp/g059598-vo/Vo_code/ptlflow/datasets/KITTI/2015', lr=0.0001, max_flow=1000.0, max_forward_side=None, max_samples=None, max_show_side=1000, model='raft', mpi_sintel_root_dir='/work/ws-tmp/g059598-vo/Vo_code/ptlflow/datasets/MPI-Sintel', output_path='/work/ws-tmp/g059598-vo/Vo_code/ptlflow/outputs/', pretrained_ckpt='kitti', reversed=False, selection=None, show=False, test_dataset=None, train_batch_size=0, train_crop_size=None, train_dataset=None, train_num_workers=4, train_transform_cuda=False, train_transform_fp16=False, val_dataset=None, wdecay=0.0001, write_outputs=False)
            flowmodel = RAFT(args2)
            flowmodel.load_state_dict(checkpoint['flow_state_dict'], strict=True)

        flowimagemodel = FlowImage()
        print("loaded flowimagemodel")
        flowimagemodel.load_state_dict(checkpoint['flow_image_state_dict'], strict=True)
        model = Pcnn()
        print("loaded pcnn")
        model.load_state_dict(checkpoint['pcnn_state_dict'], strict=True)

    flowmodel.to(device1)
    flowimagemodel.to(device1)
    model.to(device1)


    args = parser.parse_args()
    if args.mode == 'train':
        print("i am training")
        train(model,flowmodel,flowimagemodel, args.datapath, args.checkpoint_path, args.train_iter,preprocess,args)

    elif args.mode == 'test':
        test(model,flowmodel,flowimagemodel, args.datapath, preprocess,args)
    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))