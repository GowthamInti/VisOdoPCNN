import torch.nn as nn
import torch.nn.functional as F
import torch

# class Cnn4b(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#         self.conv1_1 = nn.Conv2d(3, 64, 9,padding =[1,1])
#         self.conv1_2 = nn.Conv2d(64, 20, 2,padding =[1,1])

#         self.conv2_1 = nn.Conv2d(3, 64, 9,padding =[1,1])
#         self.conv2_2 = nn.Conv2d(64, 20, 2,padding =[1,1])

#         self.conv3_1 = nn.Conv2d(3, 64, 9,padding =[1,1])
#         self.conv3_2 = nn.Conv2d(64, 20, 2,padding =[1,1])

#         self.conv4_1 = nn.Conv2d(3, 64, 9,padding =[1,1])
#         self.conv4_2 = nn.Conv2d(64, 20, 2,padding =[1,1])
 
#         self.pooling = nn.AvgPool2d(2)
#         self.pool1 = nn.MaxPool2d(4,stride =4)
#         self.pool2 = nn.MaxPool2d(2,stride =2)
#         self.flatten = nn.Flatten()

#     def forward(self, x):

#         kernel_size_w,kernel_size_h = int(x.size()[2]/2),int(x.size()[3]/2)
#         patches = x.unfold(2, kernel_size_w,int(x.size()[2]/2)).unfold(3, kernel_size_h,int(x.size()[3]/2))
#         patches = patches.contiguous().view(x.size()[0],x.size()[1],-1, kernel_size_w, kernel_size_h)
    
#         ls = []
#         for i in range(4):
#             x = self.pooling(patches[:,:,i,:,:])
#             if i==0:
#                 x = self.pool1(self.conv1_1(x))
#                 y = self.pool2(self.conv1_2(x))
#                 y = self.flatten(y)   # flatten all dimensions except batch                       
#                 x = self.flatten(x)# flatten all dimensions except batch                       
#                 x = torch.cat((x,y),1)
          
#             if i==1:
#                 x = self.pool1(self.conv2_1(x))
#                 y = self.pool2(self.conv2_2(x))
#                 y = self.flatten(y) # flatten all dimensions except batch                       
#                 x = self.flatten(x)# flatten all dimensions except batch                       
#                 x = torch.cat((x,y),1)

#             if i==2:
#                 x = self.pool1(self.conv3_1(x))
#                 y = self.pool2(self.conv3_2(x))
#                 y = self.flatten(y)   # flatten all dimensions except batch                       
#                 x = self.flatten(x)# flatten all dimensions except batch                       
#                 x = torch.cat((x,y),1)

#             if i==3:
#                 x = self.pool1(self.conv4_1(x))
#                 y = self.pool2(self.conv4_2(x))
#                 y = self.flatten(y)    # flatten all dimensions except batch                       
#                 x = self.flatten(x)# flatten all dimensions except batch                       
#                 x = torch.cat((x,y),1)# flatten all dimensions except batch                       

#             ls.append(x)
        
#         y =torch.stack(ls, dim=1)
#         y =y.view(y.shape[0],y.shape[1]*y.shape[2])
#         return y


class Cnn4b(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1_1 = nn.Conv2d(3, 64, 9,padding =[1,1])
        self.conv1_2 = nn.Conv2d(64, 20, 2,padding =[1,1])

        self.conv2_1 = nn.Conv2d(3, 64, 9,padding =[1,1])
        self.conv2_2 = nn.Conv2d(64, 20, 2,padding =[1,1])

        self.conv3_1 = nn.Conv2d(3, 64, 9,padding =[1,1])
        self.conv3_2 = nn.Conv2d(64, 20, 2,padding =[1,1])

        self.conv4_1 = nn.Conv2d(3, 64, 9,padding =[1,1])
        self.conv4_2 = nn.Conv2d(64, 20, 2,padding =[1,1])
 
        # self.pooling = nn.AvgPool2d(2)
        self.pool1 = nn.MaxPool2d(4,stride =4)
        self.pool2 = nn.MaxPool2d(2,stride =2)
        self.flatten = nn.Flatten()

    def forward(self, x):

        batch_size, channels, height, width = x.size()
        q_height, q_width = height // 2, width // 2

        x_q1 = x[:, :, :q_height, :q_width]
        x_q2 = x[:, :, :q_height, q_width:]
        x_q3 = x[:, :, q_height:, :q_width]
        x_q4 = x[:, :, q_height:, q_width:]
        # import ipdb; ipdb.set_trace()
        x_q1 = self.pool1(self.conv1_1(x_q1))
        y_q1 = self.pool2(self.conv1_2(x_q1))
        y_q1 = self.flatten(y_q1)   # flatten all dimensions except batch                       
        x_q1 = self.flatten(x_q1)# flatten all dimensions except batch                       
        x_q1 = torch.cat((x_q1,y_q1),1)

        x_q2 = self.pool1(self.conv2_1(x_q2))
        y_q2 = self.pool2(self.conv2_2(x_q2))
        y_q2 = self.flatten(y_q2)   # flatten all dimensions except batch
        x_q2 = self.flatten(x_q2)   # flatten all dimensions except batch
        x_q2 = torch.cat((x_q2, y_q2), 1)

        x_q3 = self.pool1(self.conv3_1(x_q3))
        y_q3 = self.pool2(self.conv3_2(x_q3))
        y_q3 = self.flatten(y_q3)   # flatten all dimensions except batch
        x_q3 = self.flatten(x_q3)   # flatten all dimensions except batch
        x_q3 = torch.cat((x_q3, y_q3), 1)

        x_q4 = self.pool1(self.conv4_1(x_q4))
        y_q4 = self.pool2(self.conv4_2(x_q4))
        y_q4 = self.flatten(y_q4)   # flatten all dimensions except batch
        x_q4 = self.flatten(x_q4)   # flatten all dimensions except batch
        x_q4 = torch.cat((x_q4, y_q4), 1)  

        x_combined = torch.cat((x_q1, x_q2, x_q3, x_q4), dim=1)
        return x_combined