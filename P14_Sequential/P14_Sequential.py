#Squential快速搭建神经网络
import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d ,MaxPool2d,Flatten,Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

datasets = torchvision.datasets.CIFAR10("../data",train=False,transform=transforms.Compose([transforms.ToTensor()]))
dataloader = DataLoader(datasets,batch_size=1,shuffle=True)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        #->INPUT 3*32*32
        #->CONV2D Ker5*5 32*32*32
        #->MP ker2*2 32*16*16
        #->Conv2d ker5*5 32*16*16
        #->MP ker2*2 32*8*8
        #->Conv2d ker5*5 64*8*8
        #->MP KER2*2 64*4*4
        #->Flatten 64
        #Fullyconnected 10

        # self.conv1 = Conv2d(3,32,5,padding=2)
        # #参考Conv2d计算公式
        # self.maxpool1 = nn.MaxPool2d(2)
        # self.conv2 = Conv2d(32,32,5,padding=2)
        # self.maxpool2 = nn.MaxPool2d(2)
        # self.conv3 = Conv2d(32,64,5,padding=2)
        # self.maxpool3 = nn.MaxPool2d(2)
        # self.flatten = nn.Flatten()
        # self.linear1= nn.Linear(1024,64)
        # self.linear2 = nn.Linear(64,10)

        #尝试用Sequential简便构建网络
        self.model1 = nn.Sequential(
            Conv2d(3,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,64,5,padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # output = self.linear2(x)
        output = self.model1(x)
        return output

tudui = Tudui()
print(tudui)

#检验网络
input = torch.ones((64,3,32,32))
output = tudui(input)
print(output.shape)


for data in dataloader:
    imgs , labels = data
    outputs=tudui(imgs)
    print(outputs)
    print(labels)
writer = SummaryWriter("logs_Seq")
writer.add_graph(tudui, input)
writer.close()


