import numpy as np

trainx = np.load('valid_x.npy')
train_x = trainx.astype('int32')
trainy = np.load('valid_y.npy')

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import trange
from time import sleep
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
use_gpu = torch.cuda.is_available()

trainx = torch.from_numpy(trainx).float()

if use_gpu:
    trainx = trainx.cuda()
trainx.size()

trainx = trainx.permute(0,3,1,2)

class Net(nn.Module):
    """
    Image2Vector CNN which takes image of dimension (28x28x3) and return column vector length 64
    """
    def sub_block(self, in_channels, out_channels=64, kernel_size=3):
        block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
                    torch.nn.BatchNorm2d(out_channels),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=2)
                )
        return block
    
    def __init__(self):
        super(Net, self).__init__()
        self.convnet1 = self.sub_block(3)
        self.convnet2 = self.sub_block(64)
        self.convnet3 = self.sub_block(64)
        self.convnet4 = self.sub_block(64)

    def forward(self, x):
        x = self.convnet1(x)
        x = self.convnet2(x)
        x = self.convnet3(x)
        x = self.convnet4(x)
        x = torch.flatten(x, start_dim=1)
        return x







