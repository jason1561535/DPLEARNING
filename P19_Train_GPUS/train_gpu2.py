import time

import torchvision
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import sys
sys.path.append('../')
from model import CIFAR_10
#定义设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#准备数据集
train_data = torchvision.datasets.CIFAR10(root='../data', train=True,
                                          download=True, transform=transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root='../data', train=False, transform=transforms.ToTensor(), download=True)

#数据集的长度length
train_data_len = len(train_data)
test_data_len = len(test_data)
print("训练数据集长度：{}".format(train_data_len))
print("测试数据集长度：{}".format(test_data_len))

#加载数据
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

#搭建神经网络
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
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

#或使用外部导入的model
network = CIFAR_10()
network= network.to(device)
#创建损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
#优化器
lr = 1e-2
optimizer = torch.optim.SGD(network.parameters(), lr)

#设置训练网络参数
#记录训练次数
total_train_step = 0
total_test_step = 0
#训练轮数
epoch = 10
writer = SummaryWriter("train_prog")
start_time = time.time()

for i in range(epoch):
    print("--------第{}轮训练开始--------".format(i+1))
    #训练步骤
    network.train()##
    for data in train_dataloader:
        imgs , targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = network(imgs)
        loss = loss_fn(outputs, targets)

        #开始优化
        #1.梯度清零
        optimizer.zero_grad()
        #返现传播
        loss.backward()
        #优化器改进参数
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print("用时{}".format(end_time - start_time))
            print("训练次数：{}，Loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    #每训练完一轮，查看正确率
    network.eval()
    total_test_loss = 0
    total_correct = 0
    #确保没有修改参数
    with torch.no_grad():
        for data in test_dataloader:
            imgs , targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = network(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            correct = (outputs.argmax(1) == targets).sum()
            total_correct += correct
    print("测试集的LOSS:{}".format(total_test_loss))
    print("测试集上的正确率：{}".format(total_correct/test_data_len))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_correct/test_data_len, total_test_step)
    total_test_step+=1

    # 每训练完一轮保存模型
    torch.save(network.state_dict(), "../models_saved/CIFAR10_TRAIN{}.pth".format(i+1))
    print("模型已保存在CIFAR10_TRAIN{}.pth".format(i+1))

writer.close()