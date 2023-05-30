import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as M 
import numpy.matlib as npmat
from  flow_vis import flow_to_color
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

def eulerAnglesToRotationMatrix(theta) :
    #import ipdb; ipdb.set_trace()
    # assert theta[0]>=(-np.pi) and theta[0] < np.pi, 'Inapprorpriate z: %f' % theta[0]
    # assert theta[1]>=(-np.pi) and theta[1] < np.pi, 'Inapprorpriate y: %f' % theta[1]
    # assert theta[2]>=(-np.pi) and theta[2] < np.pi, 'Inapprorpriate x: %f' % theta[2]    
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         np.cos(theta[0]), -np.sin(theta[0]) ],
                    [0,         np.sin(theta[0]), np.cos(theta[0])  ]
                    ])
    R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]
                    ])
    R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
                    [np.sin(theta[2]),    np.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R

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

# def rel2abs(out):
#     answer = [out[0], ]
#     for pose_seq in out[1:]:
#         rel_quaternion = eulerToRot(0, answer[-1][0],0)
#         location = rel_quaternion.dot(pose_seq[3:])
#         pose_seq[3:] = location[:]
#         pose_seq += answer[-1]
#         answer.append(pose_seq.tolist())

#     return np.array(answer)

def rel2abs(pred):
    # relative to absolute transform
    data_size = pred.shape[0]
    #r = M.from_matrix(pred[:,0:3])
    R=[]
    for pose_seq in pred:
        #import ipdb;ipdb.set_trace()
        K = M.from_euler('zyx', pose_seq[0:3], degrees=True)
        R.append(K.as_matrix())
    #t = rb.transl([:, 3:6])
    R=np.array(R)
    Tl= []
    Tl.append( npmat.eye(4) )  # T0
    # import ipdb;ipdb.set_trace()
    Tl.append( npmat.mat( np.concatenate( (np.concatenate( ( R[0][0:3,0:3], np.reshape((pred[0][3:6]),(3,1)) ), 1 ), npmat.mat([0, 0, 0, 1])), 0 ) ) ) #T1
    T = np.zeros((data_size+1, 4, 4))
    T[0,:,:] = npmat.eye(4)
    T[1,:,:] = npmat.mat( np.concatenate( (np.concatenate( ( R[0][0:3,0:3], np.reshape((pred[0][3:6]),(3,1)) ), 1 ), npmat.mat([0, 0, 0, 1])), 0 ) )
    for k in range(2,data_size+1):
        Tn = npmat.mat( np.concatenate( (np.concatenate( ( R[k-1][0:3,0:3], np.reshape((pred[k-1][3:6]),(3,1)) ), 1 ), npmat.mat([0, 0, 0, 1])), 0 ) ) #relative transform from k-1 to k
        Tl.append( Tl[k-1].dot(Tn) )
        T[k,:,:] = Tl[k]
    return T


def flow_to_image(flow_image):
    stacked_flow_img =[]
    for i in range(flow_image.shape[0]):
        flow_color = flow_to_color(flow_image[i], convert_to_bgr=False)
        
        stacked_flow_img.append(flow_color)
    return np.array(stacked_flow_img)
    # hsv = np.zeros(im1.shape, dtype=np.uint8)
    # hsv[:, :, 0] = 255
    # hsv[:, :, 1] = 255
    # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # hsv[..., 0] = ang * 180 / np.pi / 2
    # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
