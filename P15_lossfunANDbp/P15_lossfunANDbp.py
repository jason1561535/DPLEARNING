#Squential快速搭建神经网络
import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, L1Loss, MSELoss, CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

datasets = torchvision.datasets.CIFAR10("../data",train=False,transform=transforms.Compose([transforms.ToTensor()]))
dataloader = DataLoader(datasets,batch_size=1,shuffle=True)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
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
        output = self.model1(x)
        return output

# Loss的使用方法：loss = xxLoss（inputs , targets）
# loss=L1Loss(reduction='mean')
# result =loss( inputs,targets)
#
# loss_mse = MSELoss()
# result_mse = loss_mse(inputs,targets)
tudui = Tudui()
loss = CrossEntropyLoss()#定义XXloss：：torch.nn
optim = torch.optim.SGD(tudui.parameters(),lr=0.01)#定义优化器，用于反向传播

for epoch in range(20):#epoch代表学习的轮数
    print("第"+str(epoch)+"轮训练")
    running_loss = 0.0
    for data in dataloader:
        imgs , targets = data
        outputs=tudui(imgs)
        # print(outputs)
        # print(targets)
        result_loss = loss(outputs, targets)
        optim.zero_grad()#清空梯度
        result_loss.backward()#反向传播
        optim.step()#对参数调优
        running_loss += result_loss
    print(running_loss)

