from PIL import Image
import os
import os.path
import numpy as np
import random
import math
import datetime
from numpy import *
from .utility import *
from numpy.linalg import norm
import torch.utils.data
import torchvision.transforms as transforms
_EPS = np.finfo(float).eps * 4.0

def default_image_loader(path):
    return Image.open(path).convert('RGB') #.transpose(0, 2, 1)

class VisualOdometryDataLoader(torch.utils.data.Dataset):
    def __init__(self, datapath, trajectory_length=2, transform=None, test=False, seq = '00',
                 loader=default_image_loader):
        self.base_path = datapath
        if test:
            self.sequences = [seq]
        else:
            # self.sequences = ['00', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
            self.sequences = ['00', '01', '02', '03', '04', '05', '06', '07' ]
            # self.sequences = ['01']

        # self.timestamps = self.load_timestamps()
        self.size = 0
        self.sizes = []
        self.poses = self.load_poses()
        self.trajectory_length = trajectory_length

        self.transform = transform
        self.loader = loader

    def load_poses(self):
        all_poses = []
        for sequence in self.sequences:
            with open(os.path.join(self.base_path, 'poses/',  sequence + '.txt')) as f:
                poses = np.array([[float(x) for x in line.split()] for line in f], dtype=np.float32)
                all_poses.append(poses)

                self.size = self.size + len(poses)
                self.sizes.append(len(poses))
        return all_poses
    def get_image(self, sequence, index):
        image_path = os.path.join(self.base_path, 'sequences', sequence, 'image_2', '%06d' % index + '.png')
        image = self.loader(image_path)
        return image
    def matrix_rt(self, p):
        return np.vstack([np.reshape(p.astype(np.float32), (3, 4)), [[0., 0., 0., 1.]]])

    def __getitem__(self, index):
        sequence = 0
        sequence_size = 0
        for size in self.sizes:
            if index < size-1:
                sequence_size = size
                break
            index = index - (size-1)
            sequence = sequence + 1
        
        if (sequence >= len(self.sequences)):
            sequence = 0

        images_stacked = []
        odometries = []
        img1 = self.get_image(self.sequences[sequence], index)
        img2 = self.get_image(self.sequences[sequence], index+1)
        pose1 = self.get6DoFPose(self.poses[sequence][index])
        pose2 = self.get6DoFPose(self.poses[sequence][index+1])
        odom = pose2 - pose1
        print(odom[:3])
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        images_stacked.append(np.stack([img1, img2], axis=0))
    

        return np.asarray(images_stacked), np.asarray(odom,dtype=np.float32)

    def __len__(self):
        return self.size - len(self.sequences)

    def isRotationMatrix(self, R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype = R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    def rotationMatrixToEulerAngles(self,rotation_matrix) :
    
        sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            pitch = np.arctan2(-rotation_matrix[2, 0], sy)
            yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            pitch = np.arctan2(-rotation_matrix[2, 0], sy)
            yaw = 0.0        
        return np.array([roll, pitch, yaw])
    def get6DoFPose(self, p):
        # R = self.matrix_rt(p)
        pos = np.array([p[3], p[7], p[11]])
        R = np.array([[p[0], p[1], p[2]], [p[4], p[5], p[6]], [p[8], p[9], p[10]]])
        angles = self.rotationMatrixToEulerAngles(R)
        return np.concatenate((angles,pos))
    # def tr2eul(self, m):
    #     """
    #     Extract Euler angles.
    #     Returns a vector of Euler angles corresponding to the rotational part of 
    #     the homogeneous transform.  The 3 angles correspond to rotations about
    #     the Z, Y and Z axes respectively.
        
    #     @type m: 3x3 or 4x4 matrix
    #     @param m: the rotation matrix
    #     @rtype: 1x3 matrix
    #     @return: Euler angles [S{theta} S{phi} S{psi}]
        
    #     @see:  L{eul2tr}, L{tr2rpy}
    #     """

    #     m = mat(m)
    #     if ishomog(m):
    #         euler = zeros(3)
    #         if norm(m[0,2])<finfo(float).eps and norm(m[1,2])<finfo(float).eps:
    #             # singularity
    #             euler[0] = 0
    #             sp = 0
    #             cp = 1
    #             euler[1] = arctan2(cp*m[0,2] + sp*m[1,2], m[2,2])
    #             euler[2] = arctan2(-sp*m[0,0] + cp*m[1,0], -sp*m[0,1] + cp*m[1,1])
    #             return euler
    #         else:
    #             euler[0] = arctan2(m[1,2],m[0,2])
    #             sp = sin(euler[0])
    #             cp = cos(euler[0])
    #             euler[1] = arctan2(cp*m[0,2] + sp*m[1,2], m[2,2])
    #             euler[2] = arctan2(-sp*m[0,0] + cp*m[1,0], -sp*m[0,1] + cp*m[1,1])
    #             return euler
            


if __name__ == "__main__":
    db = VisualOdometryDataLoader("/work/ws-tmp/g059598-vo/dataset",test=True)
    im_stack, odom = db[313]
    im_stack1, odom2 = db[316]
    print (odom,odom2,_EPS)
    im_stack = np.squeeze(im_stack)
    # import matplotlib.pyplot as plt

    # f, axarr = plt.subplots(2,2)
    # axarr[0,0].imshow(im_stack[:,:,:3])
    # axarr[0,1].imshow(im_stack[:,:,3:])
    # plt.show()