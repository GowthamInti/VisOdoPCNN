# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class FlowImage(nn.Module):
#     def __init__(self):
#         super(FlowImage, self).__init__()
        
#         input_dim = 2 * 48 * 160
#         output_dim = 3 * 48 * 156
#         self.pool1 = nn.MaxPool2d(4)
#         self.fc1 = nn.Linear(input_dim, output_dim)
        
#     def forward(self, x):
#         # Flatten the input

#         x = self.pool1(x)

#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)

#         x = x.view(-1, 3, 48, 156)
#         x = torch.clamp(x, 0, 1)
        
#         return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowImage(nn.Module):
    def __init__(self):
        super(FlowImage, self).__init__()
        
        # input_dim = 2 * 48 * 160
        # output_dim = 3 * 48 * 156
        self.pool1 = nn.MaxPool2d(4)
        self.conv1 = nn.Conv2d(2, 18, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(18, 3, kernel_size=3, padding=(0, 1))
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Forward pass through the layers
        x = self.pool1(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x