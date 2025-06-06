##卷积操作conv2d

import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import  DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
dataset = torchvision.datasets.CIFAR10("../data",train=False,transform=torchvision.transforms.ToTensor(),download=False)

#
dataloader = DataLoader(dataset,batch_size=64,shuffle=True)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        #定义卷积层
        self.conv1 = Conv2d( in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=0)

    def forward(self,x):
        x = self.conv1(x)
        return x

tudui = Tudui()
# print(tudui)
writer = SummaryWriter("../logs")

step = 0
for data in dataloader:
    imgs , targets = data
    print(imgs.shape)
    output = tudui(imgs)
    print(output.shape)
    writer.add_images("output",output,step)
    step += 1