from PIL import Image
import os
import os.path
import numpy as np
import random
import math
import datetime

import torch.utils.data
import torchvision.transforms as transforms


def default_image_loader(path):
    return Image.open(path).convert('RGB') #.transpose(0, 2, 1)

class VisualOdometryDataLoader(torch.utils.data.Dataset):
    def __init__(self, datapath, trajectory_length=2, transform=None, test=False,
                 loader=default_image_loader):
        self.base_path = datapath
        if test:
            self.sequences = ['09']
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
        odom[0] = self.normalize_angle_delta(odom[0])
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

    def rotationMatrixToEulerAngles(self,R) :
    
        sy = math.sqrt(R[2,1] * R[2,1] +  R[0,2] * R[0,2])
        
        singular = sy < 1e-6
    
        if  not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
        return np.array([x, y, z])

    def get6DoFPose(self, p):
        pos = np.array([p[3], p[7], p[11]])
        R = np.array([[p[0], p[1], p[2]], [p[4], p[5], p[6]], [p[8], p[9], p[10]]])
        angles = self.rotationMatrixToEulerAngles(R)
        return np.concatenate((angles,pos))

    def normalize_angle_delta(self,angle):
        if(angle > np.pi):
            angle = angle - 2 * np.pi
        elif(angle < -np.pi):
            angle = 2 * np.pi + angle
        return angle
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