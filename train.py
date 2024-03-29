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
from torchvision import transforms as T
from torchvision import transforms
from torch.autograd import Variable

from models.flownet.flownet2ss import FlowNet2SS

from models.raft.raft2 import RAFT,RAFTSmall
from torchvision.utils  import make_grid, flow_to_image
from model.helper import *

from model.pcnn import *
from model.dataset import VisualOdometryDataLoader

from torch.utils.tensorboard import SummaryWriter
if torch.cuda.is_available():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:4024"

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

def train_model(model, flowmodel,train_loader, optimizer, epoch,length,args):
    # switch to train mode
    running_loss = 0.0
    height = 48;
    width = 156;

    new_img_size = [height,width]
    transform = T.Resize(new_img_size)

    for batch_idx, (images_stacked, odometries_stacked) in enumerate(tqdm(train_loader)):
        images_stacked, odometries_stacked = images_stacked.to(device), odometries_stacked.to(device)
        images_stacked = images_stacked.permute(1, 0, 2, 3, 4,5)        
        flow_output = flowmodel(images_stacked[0])
        flow_output['flows']= flow_output['flows'].permute(1,0,2,3,4)
        #import ipdb;ipdb.set_trace()
        flow_image = flow_to_image(flow_output['flows'][0])
        flow_image = flow_image.detach()
        
        # import ipdb;ipdb.set_trace()
        estimated_odometry = model(transform(flow_image)/255.0)
        
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
    if epoch % 30 ==0:
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
                }
        torch.save(state, os.path.join(args.checkpoint_path, "{0}_{1}.ckpt".format(args.model,epoch)))
    

def train(model,flowmodel, datapath, checkpoint_path, epochs,preprocess, args):
    model.train()
    model.training = True
    flowmodel.eval()
    kwargs = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}
    
    train_loader = torch.utils.data.DataLoader(VisualOdometryDataLoader(datapath, transform=preprocess), batch_size=args.bsize, shuffle=True, drop_last=True, **kwargs)
    length =len(train_loader)
    if args.model == 'Pcnn':
        optimizer = optim.Adam(model.fcn_1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler =optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    writer.add_scalar('hyperparameters/learning_rate', args.lr)
    writer.add_scalar('hyperparameters/batch_size', args.bsize)
    writer.add_scalar('hyperparameters/num_epochs', epochs)
    for epoch in range(1, epochs + 1):
        # train for one epoch
        train_model(model,flowmodel, train_loader, optimizer, epoch,length,args )
        scheduler.step()
    writer.flush()

def test_model(model,flowmodel, test_loader,length,args):
    out =[]
    gt =[]
    ele= np.random.randint(low=1, high=length/args.bsize, size = 1)
    height = 48;
    width = 156;
    
    new_img_size = [height,width]
    transform = T.Resize(new_img_size)
    with torch.no_grad():
        for images_stacked, odometries_stacked in tqdm(test_loader):
            images_stacked, odometries_stacked = images_stacked.to(device), odometries_stacked.to(device)
            
            images_stacked = images_stacked.permute(1, 0, 2, 3, 4,5)
            flow_output = flowmodel(images_stacked[0])
            flow_output['flows']= flow_output['flows'].permute(1,0,2,3, 4)        
            flow_image = flow_to_image(flow_output['flows'][0])
            flow_image = flow_image.detach()
            if len(out) == ele :
                grid = make_grid(transform(flow_image))
                writer.add_image('flow_images', grid, 0)
                grid2 = make_grid(images_stacked[0][:,1])
                writer.add_image('original_images', grid2, 1)

            list_odom = []
            estimated_odometry = model(transform(flow_image)/255.0)
            estimated_odometry = estimated_odometry.cpu()
            odometries_stacked = odometries_stacked.cpu()
            loss = get_loss(estimated_odometry, odometries_stacked)


            # print(estimated_odometry, odometries_stacked)
            out.append(estimated_odometry.numpy())
            gt.append(odometries_stacked.numpy())
            writer.add_scalar("Loss/test", loss,len(out))
            #np.savetxt('my_data1.txt',np.concatenate(gt))
        return np.concatenate(gt), np.concatenate(out)


def test(model,flowmodel, datapath, preprocess,args):
    kwargs = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}
    test_loader =torch.utils.data.DataLoader(VisualOdometryDataLoader(datapath,test=True, seq = args.seq, transform=preprocess), batch_size=args.bsize, shuffle=False, drop_last=True, **kwargs)
    length =len(test_loader)
    gt_rel ,out_rel = test_model(model,flowmodel, test_loader,length,args)
    gt_abs,out_abs  = rel2ab(gt_rel),rel2ab(out_rel)
    
    #print(out_abs)
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
    #parser.add_argument('--validation_steps', default=100, type=int, help='test iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--checkpoint_path', default='/work/ws-tmp/g059598-Vo/Vo_code/PCNN2/PCNN/checkpoint/raft', type=str, help='Checkpoint path')
    parser.add_argument('--checkpoint', default=None, type=str, help='Checkpoint')
    parser.add_argument('--checkpoint_load',default='/work/ws-tmp/g059598-Vo/Vo_code/PCNN2/PCNN/checkpoint/model.pth', type=str, help='Checkpoint_load')
    parser.add_argument('--save_res',default='/work/ws-tmp/g059598-Vo/Vo_code/PCNN2/PCNN/results', type=str, help='save_test_result')

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

    if args.model == 'Pcnn':
        model = Pcnn()
    elif args.model =='Pcnn1':
        model =Pcnn1()
    elif args.model == 'Pcnn2':
        model = Pcnn2()

    if args.flowmodel == 'FlowNet2SS':
        args2 = Namespace(accelerator=None, accumulate_grad_batches=None, amp_backend='native', amp_level=None, auto_lr_find=False, auto_scale_batch_size=False, auto_select_gpus=False, autoflow_root_dir='/path/to/autoflow', batch_norm=False, benchmark=False, check_val_every_n_epoch=1, checkpoint_callback=None, clear_train_state=True, default_root_dir=None, detect_anomaly=False, deterministic=False, devices=None, div_flow=20.0, enable_checkpointing=True, enable_model_summary=True, enable_progress_bar=True, fast_dev_run=False, flush_logs_every_n_steps=None, flying_chairs2_root_dir='/path/to/FlyingChairs2', flying_chairs_root_dir='/work/ws-tmp/g059598-vo/ptlflow/datasets/FlyingChairs_release', flying_things3d_root_dir='/work/ws-tmp/g059598-vo/ptlflow/datasets/FlyingThings3D', flying_things3d_subset_root_dir='/path/to/FlyingThings3D_subset', gpus=None, gradient_clip_algorithm=None, gradient_clip_val=None, hd1k_root_dir='/path/to/HD1K', input_channels=6, ipus=None, kitti_2012_root_dir='/path/to/KITTI/2012', kitti_2015_root_dir='/work/ws-tmp/g059598-vo/ptlflow/datasets/KITTI/2015', limit_predict_batches=1.0, limit_test_batches=1.0, limit_train_batches=1.0, limit_val_batches=1.0, log_dir='ptlflow_logs', log_every_n_steps=50, log_gpu_memory=None, logger=True, loss_base_weight=0.32, loss_norm='L2', loss_num_scales=5, loss_start_scale=4, lr=0.0001, max_epochs=None, max_steps=-1, max_time=None, min_epochs=None, min_steps=None, model='flownet2ss', model_name=0, move_metrics_to_cpu=False, mpi_sintel_root_dir='/work/ws-tmp/g059598-vo/ptlflow/datasets/MPI-Sintel', multiple_trainloader_mode='max_size_cycle', num_nodes=1, num_processes=1, num_sanity_val_steps=2, overfit_batches=0.0, plugins=None, precision=32, prepare_data_per_node=None, pretrained_ckpt='/work/ws-tmp/g059598-vo/FlowNet2-S_checkpoint.pth.tar', process_position=0, profiler=None, progress_bar_refresh_rate=None, random_seed=1234, reload_dataloaders_every_epoch=False, reload_dataloaders_every_n_epochs=0, replace_sampler_ddp=True, resume_from_checkpoint=None, stochastic_weight_avg=False, strategy=None, sync_batchnorm=False, terminate_on_nan=None, test_dataset=None, tpu_cores=None, track_grad_norm=-1, train_batch_size=0, train_crop_size=None, train_dataset='chairs', train_num_workers=4, train_transform_cuda=False, train_transform_fp16=False, val_check_interval=1.0, val_dataset='sintel', wdecay=0.001, weights_save_path=None, weights_summary='top')
        flowmodel = FlowNet2SS(args2)
        flowdict = torch.load('/work/ws-tmp/g059598-Vo/Vo_code/PCNN2/PCNN/checkpoint/flowdict.pth', map_location=torch.device('cpu'))
        flowmodel.load_state_dict(flowdict, strict=True)
    elif args.flowmodel =='RAFT':
        args2 =Namespace(autoflow_root_dir='/path/to/autoflow', corr_levels=4, corr_radius=4, dropout=0.0, flow_format='original', flying_chairs2_root_dir='/path/to/Vo_code/FlyingChairs2', flying_chairs_root_dir='/work/ws-tmp/g059598-vo/Vo_code/ptlflow/datasets/FlyingChairs_release', flying_things3d_root_dir='/work/ws-tmp/g059598-vo/Vo_code/ptlflow/datasets/FlyingThings3D', flying_things3d_subset_root_dir='/path/to/FlyingThings3D_subset', gamma=0.8, hd1k_root_dir='/path/to/HD1K', iters=12, kitti_2012_root_dir='/path/to/KITTI/2012', kitti_2015_root_dir='/work/ws-tmp/g059598-vo/Vo_code/ptlflow/datasets/KITTI/2015', lr=0.0001, max_flow=1000.0, max_forward_side=None, max_samples=None, max_show_side=1000, model='raft', mpi_sintel_root_dir='/work/ws-tmp/g059598-vo/Vo_code/ptlflow/datasets/MPI-Sintel', output_path='/work/ws-tmp/g059598-vo/Vo_code/ptlflow/outputs/', pretrained_ckpt='kitti', reversed=False, selection=None, show=False, test_dataset=None, train_batch_size=0, train_crop_size=None, train_dataset=None, train_num_workers=4, train_transform_cuda=False, train_transform_fp16=False, val_dataset=None, wdecay=0.0001, write_outputs=False)
        flowmodel = RAFT(args2)
        flowdict = torch.load('/work/ws-tmp/g059598-Vo/Vo_code/ptlflow/raft-kitti-3a831a4b.ckpt', map_location=torch.device('cpu'))
        flowmodel.load_state_dict(flowdict['state_dict'], strict=True)
  


    # ckpt = torch.load('/work/ws-tmp/g059598-vo/Vo_code/FlowNet2-S_checkpoint.pth.tar', map_location=torch.device('cpu'))
    # state_dict = ckpt['state_dict']
    # ckpt2 = torch.load('/work/ws-tmp/g059598-vo/Vo_code/ptlflow/flownet2-things-d63b53a7.ckpt', map_location=torch.device('cpu'))
    # state_dict2 =ckpt2['state_dict']
    # flowdict = {}
    # for k, v in state_dict.items():
    #     flowdict['flownets_1.'+k] = v
    #     flowdict['flownets_2.'+k] = v
    # for k, v in state_dict2.items():
    #     if k.startswith('flownetfusion'):
    #         flowdict[k] = v
    # flowmodel.load_state_dict(flowdict, strict=True)  
    
   
    # if args.checkpoint is not None:
    #     #checkpoint  = torch.load(args.checkpoint_load,map_location=torch.device('cpu'))
    #     checkpoint_1 = torch.load('/work/ws-tmp/g059598-vo/PCNN2/PCNN/checkpoint/Pcnn1_40.ckpt', map_location=torch.device('cpu'))
    #     checkpoint_2 = torch.load('/work/ws-tmp/g059598-vo/PCNN2/PCNN/checkpoint/Pcnn2_40.ckpt', map_location=torch.device('cpu'))
    #     stated1 = checkpoint_1['state_dict']
    #     stated2 = checkpoint_2['state_dict']
    #     pcnndict={}
    #     for k,v  in stated1.items():
    #         print(k)
    #         if k.startswith('cnn1b_1'):
    #             pcnndict[k] = v
    #     for k,v in stated2.items():
    #         print(k)
    #         if k.startswith('cnn4b_1'):
    #             pcnndict[k] = v
    #     model.load_state_dict(pcnndict,strict= False)  
    model = load_checkpoint_to_model(args,model)

    if args.checkpoint is not None:
        checkpoint  = torch.load(args.checkpoint_load,map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        print('loaded')
    model.to(device)
    flowmodel.to(device)

    args = parser.parse_args()
    if args.mode == 'train':
        print("i am training")
        train(model,flowmodel, args.datapath, args.checkpoint_path, args.train_iter,preprocess,args)

    elif args.mode == 'test':
        test(model,flowmodel, args.datapath, preprocess,args)
    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))