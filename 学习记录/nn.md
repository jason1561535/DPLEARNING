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


