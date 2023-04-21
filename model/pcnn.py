import torch.nn as nn
import torch.nn.functional as F
from .cnn1b import Cnn1b
from .cnn4b import Cnn4b
from .fcn import Fcn ,Fcn1,Fcn2
#from .flownets import FlowNetS
import torch
#from torchvision.utils  import flow_to_image

class Pcnn(nn.Module):

    def __init__(self):
        super().__init__()
        #self.flow = FlowNetS()
        self.cnn1b_1 = Cnn1b()
        self.cnn4b_1 = Cnn4b()
        self.fcn_1   = Fcn()
    def forward(self, inputs): 
        cnn1b_out =self.cnn1b_1(inputs)
        cnn4b_out = self.cnn4b_1(inputs)
        concat1 = torch.cat((cnn1b_out,cnn4b_out),dim =1)
        fcn_out = self.fcn_1(concat1)
        return fcn_out

class Pcnn1(nn.Module):

    def __init__(self):
        super().__init__()
        self.cnn1b_1 = Cnn1b()
        self.fcn_1   = Fcn1()
    def forward(self, inputs): 
        cnn1b_out =self.cnn1b_1(inputs)
        fcn_out = self.fcn_1(cnn1b_out)
        return fcn_out


class Pcnn2(nn.Module):

    def __init__(self):
        super().__init__()
        self.cnn4b_1 = Cnn4b()
        self.fcn_1   = Fcn2()

    def forward(self, inputs): 
        cnn4b_out = self.cnn4b_1(inputs)
        fcn_out = self.fcn_1(cnn4b_out)

        return fcn_out