#以下是可以提前存储的网络模型
import numpy as np
import torch
from torch import nn


class CIFAR_10(nn.Module):
    def __init__(self):
        super(CIFAR_10,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5,padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5,padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10)
        )

    def forward(self,x):
        x = self.model(x)
        return x



#用于测试网络，比如根据输入看输出的shape是否符合规范
if __name__ == '__main__':
    nerwork = CIFAR_10()
    input = torch.ones((64,3,32,32))
    output = nerwork(input)
    print(output.shape)
