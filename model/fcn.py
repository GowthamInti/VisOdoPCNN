import torch.nn as nn
import torch.nn.functional as F


class Fcn(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(44832 ,3000)
        self.fc3 = nn.Linear(3000,6)
    def forward(self, x):                    
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x

class Fcn1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(25760,1000)
        self.fc2 = nn.Linear(1000,6)
    def forward(self, x):                    
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Fcn2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(18432,3000)
        self.fc2 = nn.Linear(3000,6)
    def forward(self, x):                    
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x