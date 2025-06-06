from P12_torchnn.torchnn import dataloader

# torch.nn 卷积神经网络
## 1.Dataset 和 Dataloader
* 导入模块
```python
from torch.utils.data import Dataset, DataLoader
import torchvision
#如果是Dataset引入需要自己定义Mydataset
#此处不做详细介绍，只做土堆引入torchvision中的数据集
dataset = torchvision.datasets.CIFAR10("../data",train=False,transform=torchvision.transforms.ToTensor(),download=False)
dataloader = DataLoader(dataset,batch_size=64,shuffle=True)
```


## 2.常见神经网络模版nn.module 和 nn.Conv2d
> 参考P12_torchnn文件
* 导入方法，同时导入卷积层  
```python
from torch import nn 
from torch.nn import Conv2d
from torch.utils.data import  DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../data",train=False,transform=torchvision.transforms.ToTensor(),download=False)
dataloader = DataLoader(dataset, batch_size=64 , shuffle = True)
#定义成自己的类
#module包含初始化和前向函数forward
class Tudui(nn.Module):
    #初始化superinit
    def __init__(self):
        super(Tudui, self).__init__()
        #定义卷积层,在初始化里定义
        self.conv1 = Conv2d( in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=0)
    #定义前向函数
    def forward(self,x):
        x = self.conv1(x)
        return x

tudui = Tudui()

```

* **_Conv2d的参数列表如下_** 
![img.png](img.png)]  
in_channels :输入通道数  
out_channels :输出通道数  
stride :卷积核每次移动的长度  
padding:输入层边框是否添加一圈


[Module官网连接]()  
[Conv2d官网定义连接](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d)

* 卷积层使用例子  
```python
from torch.utils.tensorboard import  SummaryWriter
writer = SummaryWriter("../logs")

step = 0
for data in dataloader:
    imgs , targets = data
    # print(imgs.shape)
    output = tudui(imgs)
    # print(output.shape)
    #####注意是add_images，因为输入卷积的是打包的imgs
    writer.add_images("output",output,step)
    step += 1
#记得close，还不知道为什么
writer.close()
```  

## 3.最大池化MaxPooling
**_最大池化的作用：保留最大特征，减少数据量_**
> 参考P13_MaxPooling  ,池化核（取池子里最大的）  
> 最常用的还是**MaxPooling2d**  
[MaxPooling官网参考](https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d)
* 基础定义  
![img_1.png](img_1.png)
![img_2.png](img_2.png)
1.stride： 默认步长为池化核大小，这是池化的定义
2.dilation ： 空洞卷积  
3.ceil_mode： 理解为去掉小数点还是直接取整
> 2.3  ceil--> :3  floor:-->2  

* 单次使用案例
```python
import torch
from torch import nn

input = torch.tensor([[1, 2, 0 ,3 ,1],
                      [0 ,1 ,2 ,3 ,1],
                      [1 ,2 ,1 ,0 ,0],
                      [5 ,2 ,3 ,1 ,1],
                      [2 ,1 ,0 ,1 ,1]], dtype=torch.float32)

input = torch.reshape(input,(1,5,5))
print(input.shape)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output=self.maxpool(input)
        return output

tuidui = Tudui()
output = tuidui(input)
print(output)
```
* 数据集使用案例
```python
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../data",train=False,download=False,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=64,shuffle=True)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=3,ceil_mode=False)

    def forward(self, input):
        output = self.maxpool(input)
        return output

tudui = Tudui()

writer = SummaryWriter("MaxPooling")
step = 0
for data in dataloader:
    imgs, labels = data
    writer.add_images("input", imgs, step)
    output = tudui(imgs)
    writer.add_images("output", output, step)
    step += 1
writer.close()
```


## 4.Sequential快速搭建神经网络
假设我们搭建如下网络  
![img_3.png](img_3.png)
我们一共做了   
①3次（卷积+池化）-> ②Flatten拉平 ->③两次Liner线性化

①卷积+池化    
&emsp;卷积Conv2d输入计算公式：  
![img_4.png](img_4.png)  
&emsp;其中stride步长默认为1，dilation默认为1
池化MaxPool2d输入计算公式：  
![img_5.png](img_5.png)
&emsp;其中stride步长默认为kernel边长，dilation默认为1
> ***注意***
> ①卷积如果不改变层的大小：那么只需要管padding和kernel比例关系
> Conv2d： 2padding = kernel-1
> ②池化使层面积变为1/4：只需要保证kernel=2，其他为默认
> MaxPool2d： 2padding = kernel

②Flatten
Just Flatten
```python
from torch.nn import Flatten
    Flatten()
```

③Liner线性化（记得要先拉伸）
目前来看Liner只要管in_features和out_features
代码如下
```python
from torch.nn import Linear
    Linear(1024,64)
    Linear(64,10)
```
**总体代码**
```python
#Squential快速搭建神经网络
import torch
from torch import nn
from torch.nn import Conv2d ,MaxPool2d,Flatten,Linear
from torch.utils.tensorboard import SummaryWriter


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
#测试方法：
tudui = Tudui()
print(tudui)

#检验网络
input = torch.ones((64,3,32,32))
output = tudui(input)
print(output.shape)

writer = SummaryWriter("logs_Seq")
writer.add_graph(tudui, input)
writer.close()

###运行结果：torch.Size([64, 10])
```
结果为：torch.Size([64, 10])

可以看到Sequential节省了很多空间，也使得代码可读性更好

