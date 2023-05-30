import torch.nn as nn
import torch.nn.functional as F
import torch
class Cnn1b(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 9,padding =[1,1])
        self.pool1 = nn.MaxPool2d(4,stride =4)
        self.conv2 = nn.Conv2d(64, 20, 2,padding=[1,1])
        self.pooling = nn.AvgPool2d(4)
        self.pool2 = nn.MaxPool2d(2,stride=2)
        self.flatten = nn.Flatten()


    def forward(self, x):
        x = self.pooling(x)
        x = self.pool1(self.conv1(x))
        y = self.pool2(self.conv2(x))
        y = self.flatten(y)
        x = self.flatten(x)# flatten all dimensions except batch
              
        x = torch.cat((x,y),1)

        return x