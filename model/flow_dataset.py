from PIL import Image
import os
import os.path
import numpy as np
import random
import math
import datetime

import torch.utils.data
import torchvision.transforms as transforms
_EPS = np.finfo(float).eps * 4.0

def default_image_loader(path):
    return Image.open(path).convert('RGB') #.transpose(0, 2, 1)

class VisualOdometryDataLoader(torch.utils.data.Dataset):
    def __init__(self, datapath, transform=None, test=False, seq = '00',
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
            if index < size:
                sequence_size = size
                break
            index = index - (size)
            sequence = sequence + 1
        
        if (sequence >= len(self.sequences)):
            sequence = 0

        images_stacked = []
        odometries = []
        img1 = self.get_image(self.sequences[sequence], index)
        pose1 = self.poses[sequence][index]
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return np.asarray(img1), np.asarray(pose1,dtype=np.float32)

    def __len__(self):
        return self.size


if __name__ == "__main__":
    db = VisualOdometryDataLoader("/work/ws-tmp/g059598-Vo/Vo_code/raft_dataset",test=True,seq='00')

    gt_l = []
    iterator = iter(db)
    for index in range(1,len(db)):
        images, gt = next(iterator)
        import ipdb; ipdb.set_trace()

