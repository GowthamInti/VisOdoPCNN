import torch.nn as nn
import torch.nn.functional as F
import torch

class Cnn4b(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1_1 = nn.Conv2d(3, 64, 9)
        self.conv1_2 = nn.Conv2d(64, 20, 2)
        # self.fc1_1 = nn.Linear(4768,1000)
        #self.fc1_2 = nn.Linear(1000,6)

        self.conv2_1 = nn.Conv2d(3, 64, 9)
        self.conv2_2 = nn.Conv2d(64, 20, 2)
        # self.fc2_1 = nn.Linear(4768,1000)
        # self.fc2_2 = nn.Linear(600, 2000)

        self.conv3_1 = nn.Conv2d(3, 64, 9)
        self.conv3_2 = nn.Conv2d(64, 20, 2)
        # self.fc3_1 = nn.Linear(4768,1000)
        # self.fc3_2 = nn.Linear(600, 2000)

        self.conv4_1 = nn.Conv2d(3, 64, 9)
        self.conv4_2 = nn.Conv2d(64, 20, 2)
        # self.fc4_1 = nn.Linear(4768,1000)
        # self.fc4_2 = nn.Linear(600, 2000)

        self.pooling = nn.AvgPool2d(4,padding =1)
        self.pool1 = nn.MaxPool2d(4)
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.unfold =unfold = nn.Unfold(kernel_size=(192, 640),stride =192)

    def forward(self, x):

        kernel_size_w,kernel_size_h = int(x.size()[2]/2),int(x.size()[3]/2)
        patches = x.unfold(2, kernel_size_w,int(x.size()[2]/2)).unfold(3, kernel_size_h,int(x.size()[3]/2))
        patches = patches.contiguous().view(x.size()[0],x.size()[1],-1, kernel_size_w, kernel_size_h)
    
        ls = []
        for i in range(4):
            x = self.pooling(patches[:,:,i,:,:])
            if i==0:
                x = self.pool1(self.conv1_1(x))
                y = self.pool2(self.conv1_2(x))
                y = F.relu(self.flatten(y))    # flatten all dimensions except batch                       
                x = F.relu(self.flatten(x))# flatten all dimensions except batch                       
                # x = self.fc1_1(torch.cat((x,y),1))
                # x = self.fc1_2(x)
            if i==1:
                x = self.pool1(self.conv2_1(x))
                y = self.pool2(self.conv2_2(x))
                y = F.relu(self.flatten(y))    # flatten all dimensions except batch                       
                x = F.relu(self.flatten(x))# flatten all dimensions except batch                       
                # x = self.fc2_1(torch.cat((x,y),1))
                # x = self.fc2_2(x)
            if i==2:
                x = self.pool1(self.conv3_1(x))
                y = self.pool2(self.conv3_2(x))
                y = F.relu(self.flatten(y))    # flatten all dimensions except batch                       
                x = F.relu(self.flatten(x))# flatten all dimensions except batch                       
                # x = self.fc3_1(torch.cat((x,y),1))
                # x = self.fc3_2(x)
            if i==3:
                x = self.pool1(self.conv4_1(x))
                y = self.pool2(self.conv4_2(x))
                y = F.relu(self.flatten(y))    # flatten all dimensions except batch                       
                x = F.relu(self.flatten(x))# flatten all dimensions except batch                       
                # x = self.fc4_1(torch.cat((x,y),1))
                # x = self.fc4_2(x)                             
            ls.append(x)
        
        y =torch.stack(ls, dim=1)
        y =y.view(y.shape[0],y.shape[1]*y.shape[2])
        return y
