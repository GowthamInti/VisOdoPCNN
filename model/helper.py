import numpy as np
import math
import matplotlib.pyplot as plt
# from scipy.spatial.transform import Rotation as M 
import numpy.matlib as npmat
from numpy import *
from .utility import *
from argparse import Namespace
from models.flownet.flownet2ss import FlowNet2SS

from models.raft.raft2 import RAFT,RAFTSmall
import torch
# from  flow_vis import flow_to_color
def plot_route(gt, out, c_gt='g', c_out='r'):
    fig = plt.figure()
    # plt.scatter([gt[0][3]], [gt[0][5]], label='sequence start', marker='s', color='k')
    # x_idx = 3
    # y_idx = 5
    # x = [v for v in gt[:, x_idx]]
    # y = [v for v in gt[:, y_idx]]
    # plt.plot(x, y, color=c_gt, label='Ground Truth')

    # x1 = [v for v in out[:, x_idx]]
    # y1 = [v for v in out[:, y_idx]]
    # plt.plot(x1, y1, color=c_out, label='PCNN')
    # #plt.scatter(x, y, color='b')
    # plt.gca().set_aspect('equal', adjustable='datalim')
    plt.plot(gt[:,0], gt[:,2], color='#0000FF', label='GT', lw=2)
    plt.plot(out[:,0], out[:,2], color='#FF00FF', label='Pcnn', lw=2)
    return fig

# def eulerAnglesToRotationMatrix(theta) :
#     #import ipdb; ipdb.set_trace()
#     # assert theta[0]>=(-np.pi) and theta[0] < np.pi, 'Inapprorpriate z: %f' % theta[0]
#     # assert theta[1]>=(-np.pi) and theta[1] < np.pi, 'Inapprorpriate y: %f' % theta[1]
#     # assert theta[2]>=(-np.pi) and theta[2] < np.pi, 'Inapprorpriate x: %f' % theta[2]    
#     R_x = np.array([[1,         0,                  0                   ],
#                     [0,         np.cos(theta[0]), -np.sin(theta[0]) ],
#                     [0,         np.sin(theta[0]), np.cos(theta[0])  ]
#                     ])
#     R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
#                     [0,                     1,      0                   ],
#                     [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]
#                     ])
#     R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
#                     [np.sin(theta[2]),    np.cos(theta[2]),     0],
#                     [0,                     0,                      1]
#                     ])
#     R = np.dot(R_z, np.dot( R_y, R_x ))
#     return R

# Helper function to convert euler angles to quaternion
def eulerToRot(phi, theta, psi):
    # Convert Euler angles to quaternion using the ZYX convention
    # phi, theta, psi are the roll, pitch, yaw angles in degrees

    # Convert angles from degrees to radians
    phi = np.radians(phi)
    theta = np.radians(theta)
    psi = np.radians(psi)

    # Compute sine and cosine terms
    c1 = np.cos(phi/2)
    s1 = np.sin(phi/2)
    c2 = np.cos(theta/2)
    s2 = np.sin(theta/2)
    c3 = np.cos(psi/2)
    s3 = np.sin(psi/2)

    # Compute quaternion elements
    q0 = c1*c2*c3 + s1*s2*s3
    q1 = s1*c2*c3 - c1*s2*s3
    q2 = c1*s2*c3 + s1*c2*s3
    q3 = c1*c2*s3 - s1*s2*c3
    qw, qx, qy, qz = q0, q1, q2, q3
    rot_matrix = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qw*qz, 2*qx*qz + 2*qw*qy],
        [2*qx*qy + 2*qw*qz, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qw*qx],
        [2*qx*qz - 2*qw*qy, 2*qy*qz + 2*qw*qx, 1 - 2*qx**2 - 2*qy**2]])

    return rot_matrix


def save_pred(pathh, T, filename):
    n = T.shape[0]
    t = np.ndarray((n,12))
    for k in range(n):
        t[k,:] = np.ravel(T[k,0:3,:])
    np.savetxt(pathh+filename, t, fmt='%.6e', delimiter=" ")

import torch

def normalize_angle_delta(angles):
    normalize = lambda angle: angle - 2 * torch.tensor([np.pi]) if angle > torch.tensor([np.pi]) else 2 * torch.tensor([np.pi]) + angle if angle < -torch.tensor([np.pi]) else angle
    return torch.tensor([normalize(angle) for angle in angles])

def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta[0]), -np.sin(theta[0])],
                    [0, np.sin(theta[0]), np.cos(theta[0])]
                    ])
    R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                    [0, 1, 0],
                    [-np.sin(theta[1]), 0, np.cos(theta[1])]
                    ])
    R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                    [np.sin(theta[2]), np.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def transl(x, y=None, z=None):
    """
    Create or decompose translational homogeneous transformations.
    
    Create a homogeneous transformation
    ===================================
    
        - T = transl(v)
        - T = transl(vx, vy, vz)
        
        The transformation is created with a unit rotation submatrix.
        The translational elements are set from elements of v which is
        a list, array or matrix, or from separate passed elements.
    
    Decompose a homogeneous transformation
    ======================================
    

        - v = transl(T)   
    
        Return the translation vector
    """

    if y==None and z==None:
        if isinstance(x, list) or ( isinstance(x, ndarray) and x.shape[0] > 1 and x.shape[1] != 4 and len(x.shape) <= 2 ) or ( isinstance(x, ndarray) and len(x.shape) > 2 ): #Trajectory case: list with 3x1 (or 1x3) vectors, or homogeneous matrices
            out = []
            for xx in x:
                xx = mat(xx)
                try:
                    if ishomog(xx):
                        out.append(xx[0:3, 3].reshape(3, 1))
                    else:
                        out.append(concatenate((concatenate((eye(3), xx.reshape(3, 1)), 1), mat([0, 0, 0, 1]))))
                except AttributeError:
                    n = len(xx)
                    r = [[], [], []]
                    for i in range(n):
                        out.append(concatenate((r, xx[i][0:3, 3]), 1))
            return out
        else: #single trajectory point
            x=mat(x)
            try:
                if ishomog(x): # check homogeneous case
                        return x[0:3,3].reshape(3,1)
                else:
                        return concatenate((concatenate((eye(3),x.reshape(3,1)),1),mat([0,0,0,1])))
            except AttributeError:
                n=len(x)
                r = [[],[],[]]
                for i in range(n):
                        r = concatenate((r,x[i][0:3,3]),1)
                return r
    elif y!=None and z!=None:
        out = []
        for xx, yy, zz in zip(x,y,z):
            out.append( concatenate((concatenate((eye(3),mat([xx,yy,zz]).T),1),mat([0,0,0,1]))) )

        return out

def rotx(theta):
    """
    Rotation about X-axis
    
    @type theta: number
    @param theta: the rotation angle
    @rtype: 3x3 orthonormal matrix
    @return: rotation about X-axis

    @see: L{roty}, L{rotz}, L{rotvec}
    """
    
    ct = cos(theta)
    st = sin(theta)
    return mat([[1,  0,    0],
            [0,  ct, -st],
            [0,  st,  ct]])


def roty(theta):
    """
    Rotation about Y-axis
    
    @type theta: number
    @param theta: the rotation angle
    @rtype: 3x3 orthonormal matrix
    @return: rotation about Y-axis

    @see: L{rotx}, L{rotz}, L{rotvec}
    """
    
    ct = cos(theta)
    st = sin(theta)

    return mat([[ct,   0,   st],
            [0,    1,    0],
            [-st,  0,   ct]])
def rotz(theta):
    """
    Rotation about Z-axis
    
    @type theta: number
    @param theta: the rotation angle
    @rtype: 3x3 orthonormal matrix
    @return: rotation about Z-axis

    @see: L{rotx}, L{roty}, L{rotvec}
    """
    
    ct = cos(theta)
    st = sin(theta)

    return mat([[ct,      -st,  0],
            [st,       ct,  0],
            [ 0,    0,  1]])


def rpy2r(roll, pitch=None,yaw=None):
    """
    Rotation from RPY angles.
    
    Two call forms:
        - R = rpy2r(S{theta}, S{phi}, S{psi})
        - R = rpy2r([S{theta}, S{phi}, S{psi}])
    These correspond to rotations about the Z, Y, X axes respectively.

    @type roll: number or list/array/matrix of angles
    @param roll: roll angle, or a list/array/matrix of angles
    @type pitch: number
    @param pitch: pitch angle
    @type yaw: number
    @param yaw: yaw angle
    @rtype: 4x4 homogenous matrix
    @return: R([S{theta} S{phi} S{psi}])

    @see:  L{tr2rpy}, L{rpy2r}, L{tr2eul}

    """
    n=1
    if pitch==None and yaw==None:
        roll= mat(roll)
        if numcols(roll) != 3:
            error('bad arguments')
        n = numrows(roll)
        pitch = roll[:,1]
        yaw = roll[:,2]
        roll = roll[:,0]
    if n>1:
        R = []
        for i in range(0,n):
            r = rotz(roll[i,0]) * roty(pitch[i,0]) * rotx(yaw[i,0])
            R.append(r)
        return R
    try:
        r = rotz(roll[0,0]) * roty(pitch[0,0]) * rotx(yaw[0,0])
        return r
    except:
        r = rotz(roll) * roty(pitch) * rotx(yaw)
        return r

def r2t(R):
    """
    Convert a 3x3 orthonormal rotation matrix to a 4x4 homogeneous transformation::
    
        T = | R 0 |
            | 0 1 |
            
    @type R: 3x3 orthonormal rotation matrix
    @param R: the rotation matrix to convert
    @rtype: 4x4 homogeneous matrix
    @return: homogeneous equivalent
    """
    if isinstance(R, list):
        out = []
        for r in R:
            r = concatenate( (concatenate( (r, zeros((3,1))),1), mat([0,0,0,1])) )
            out.append(r)
    else:
        out = concatenate( (concatenate( (R, zeros((3,1))),1), mat([0,0,0,1])) )
    
    return out


def rpy2tr(roll, pitch=None, yaw=None):
    """
    Rotation from RPY angles.
    
    Two call forms:
        - R = rpy2tr(r, p, y)
        - R = rpy2tr([r, p, y])
    These correspond to rotations about the Z, Y, X axes respectively.

    @type roll: number or list/array/matrix of angles
    @param roll: roll angle, or a list/array/matrix of angles
    @type pitch: number
    @param pitch: pitch angle
    @type yaw: number
    @param yaw: yaw angle
    @rtype: 4x4 homogenous matrix
    @return: R([S{theta} S{phi} S{psi}])

    @see:  L{tr2rpy}, L{rpy2r}, L{tr2eul}

    """
    return r2t( rpy2r(roll, pitch, yaw) )

def rel2ab(pred):
    # relative to absolute transform
    data_size = pred.shape[0]

    R = rpy2tr(pred[:, 0:3])
    t = transl(pred[:, 3:6])
    Tl= []
    Tl.append( npmat.eye(4) )  # T0
    Tl.append( npmat.mat( np.concatenate( (np.concatenate( ( R[0][0:3,0:3], t[0][0:3,3] ), 1 ), npmat.mat([0, 0, 0, 1])), 0 ) ) ) #T1
    T = np.zeros((data_size+1, 4, 4))
    T[0,:,:] = npmat.eye(4)
    T[1,:,:] = npmat.mat( np.concatenate( (np.concatenate( ( R[0][0:3,0:3], t[0][0:3,3] ), 1 ), npmat.mat([0, 0, 0, 1])), 0 ) )
    for k in range(2,data_size+1):
        Tn = npmat.mat( np.concatenate( (np.concatenate( ( R[k-1][0:3,0:3], t[k-1][0:3,3] ), 1 ), npmat.mat([0, 0, 0, 1])), 0 ) ) #relative transform from k-1 to k
        Tl.append( Tl[k-1].dot(Tn) )
        T[k,:,:] = Tl[k]
    return T

def load_checkpoint_to_model(args, model):

    # if args.checkpoint is not None and args.model == 'Pcnn':
    #     checkpoint_1 = torch.load('/work/ws-tmp/g059598-Vo/Vo_code/PCNN2/PCNN/checkpoint/flowraft/flowraftadamw/Pcnn1_1875.ckpt', map_location=torch.device('cpu'))
    #     checkpoint_2 = torch.load('/work/ws-tmp/g059598-Vo/Vo_code/PCNN2/PCNN/checkpoint/flowraft/flowraftadamw/Pcnn2_1875.ckpt', map_location=torch.device('cpu'))
    #     stated1 = checkpoint_1['state_dict']
    #     stated2 = checkpoint_2['state_dict']
    #     pcnndict = {}
    #     for k, v in stated1.items():
    #         if k.startswith('cnn1b_1'):
    #             pcnndict[k] = v
    #     for k, v in stated2.items():
    #         if k.startswith('cnn4b_1'):
    #             pcnndict[k] = v

    #     model.load_state_dict(pcnndict, strict=False)

    #     # for param in model.cnn1b_1.parameters():
    #     #     param.requires_grad = False
    #     # for param in model.cnn4b_1.parameters():
    #     #     param.requires_grad = False
    #     # for param in model.fcn_1.fc1.parameters():
    #     #     param.requires_grad = True 
    #     # for param in model.fcn_1.fc3.parameters():
    #     #     param.requires_grad = False       
        # print("loaded pcnn")
        

    if args.checkpoint is not None and args.model == 'Pcnn1':
        checkpoint = torch.load(args.checkpoint_load, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'], strict=True)

    if args.checkpoint is not None and args.model == 'Pcnn2':
        checkpoint = torch.load(args.checkpoint_load, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'], strict=True)

    if args.checkpoint is not None and args.model == 'Pcnn':
        checkpoint = torch.load(args.checkpoint_load, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        # for param in model.cnn1b_1.parameters():
        #     param.requires_grad = False
        # for param in model.cnn4b_1.parameters():
        #     param.requires_grad = False
        # for param in model.fcn_1.fc1.parameters():
        #     param.requires_grad = False 
        # for param in model.fcn_1.fc3.parameters():
        #     param.requires_grad = True     
    return model

def load_flow_model(args):
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
 
    return flowmodel