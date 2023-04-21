import torch.nn as nn
import torch.nn.functional as F
import torch
class Cnn1b(nn.Module):
    def __init__(self):
        super().__init__()
        self.pooling = nn.AvgPool2d(4,padding =[2,1])
        self.conv1 = nn.Conv2d(3, 64, 9)
        self.pool1 = nn.MaxPool2d(4)
        self.conv2 = nn.Conv2d(64, 20, 2)
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(25760,1000)
        # self.fc2 = nn.Linear(1000, 6)

    def forward(self, x):
        x = self.pooling(x)
        x = self.pool1(self.conv1(x))
        y = self.pool2(self.conv2(x))
        y = F.relu(self.flatten(y))
        x = F.relu(self.flatten(x))# flatten all dimensions except batch
              
        x = torch.cat((x,y),1)
        # x = self.fc2(x)
        return x