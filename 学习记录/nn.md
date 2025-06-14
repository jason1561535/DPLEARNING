
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

---

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
---

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
---

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

---

## 5.LOSS AND BP （Opitim）
> 本节课重点
> ①定义Loss
> ②使用优化器Optim进行反向传播  

我们需要在data输入进模型之后  
**计算Loss->重置梯度->反向传播->优化更新参数**

```python
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
```
---
## 6.Pretrained Models  
* 我们可以下载预训练好的模型进行修改
```python
import torchvision

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)
print("ok")

print(vgg16_true)
```
pytorch会自动下载网络模型到C盘，我们需要修改环境变量，添加一个环境变量TORCH_HOME，然后***重启***
![img_6.png](img_6.png)
结果如下
```python
...
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
```
我们这里可以看到VGG16的整体结构，最后是进行1000种分类  
当未来我们想应用这种模型，或者进行修改时，我们可以使用  
add_module('module_diyname',nn.xx)#for example nn.linear nn.MaxPool2d
代码如下
```python
vgg16_true.add_module('add_linear',nn.Linear(1000,10))
print(vgg16_true)
#vgg16_true.classifier.add_module('add_linear',nn.Linear(1000,10))
#classifier的区别就是最后显示可以在classifier内部添加module

  # (classifier): Sequential(
  #   (0): Linear(in_features=25088, out_features=4096, bias=True)
  #   (1): ReLU(inplace=True)
  #   (2): Dropout(p=0.5, inplace=False)
  #   (3): Linear(in_features=4096, out_features=4096, bias=True)
  #   (4): ReLU(inplace=True)
  #   (5): Dropout(p=0.5, inplace=False)
  #   (6): Linear(in_features=4096, out_features=1000, bias=True)
  # )
  # (add_linear): Linear(in_features=1000, out_features=10, bias=True)
```
* 如果想修改单独层(例如classifier中的层06)
```python
print(vgg16_false)
vgg16_false.classifier[6]=nn.Linear(4096,10)
print(vgg16_false)
#原本层[06] input:4096 ,output:1000
#修改线性层[06] input 4096 , output:10
```

---

## 7. 保存模型
保存模型的两种方式
1.保存模型结构+模型参数
**保存**
```python
import torch
import torchvision
#保存方式1,保存   模型结构+模型参数
torch.save(vgg16,'../models_saved/vgg16_method1.pth')
```
**读取**
```python
import torch
import torchvision

#加载1->对应->保存1 读取模型+参数
model = torch.load('../models_saved/vgg16_method1.pth',weights_only= False)
print(model)
#方式1存在陷阱
#你每次都需要重新定义你自己的模型类
```
---
2.以字典形式保存参数
**保存**
```python
#保存方式2，保存   模型参数（官方推荐）
torch.save(vgg16.state_dict(),'../models_saved/vgg16_method2.pth')
```
**读取**
```python
import torch
import torchvision
#加载2->对应->保存2 读取字典
vgg16 = torchvision.models.vgg16()
vgg16.load_state_dict(torch.load('../models_saved/vgg16_method2.pth'))
# model = torch.load('../models_saved/vgg16_method2.pth',weights_only= False)
```
---

## 8.完整的模型训练套路P18
①提前编写网络架构  
我们可以创建一个model.py文件，把我们搭建的网络架构存放在里面。
例如：
>Project  
> --dir1  
> ----file1.py  
> --model  
> ----model.py    

* 当我想在file1.py读取另一个文件夹的model文件时
```python
import sys
sys.path.append('../')
from model.model import CIFAR_10#CIFAR_10是自己定义好的模型
```

写完一个模型框架后，记得测试它的输入输出shape
```python
#有关CIFAR_10怎么搭建的看主目录下的model.py文件
if __name__ == '__main__':
    nerwork = CIFAR_10()
    input = torch.ones((64,3,32,32))
    output = nerwork(input)
    print(output.shape)
```

②训练的整体架构
* 准备数据集：Dataset/Dataloader↓
```python
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
```
* 搭建网络：建议放在统一文件内
```python
import sys
sys.path.append('../')
from model import CIFAR_10

#或使用外部导入的model
network = CIFAR_10()
```

* 创建损失函数和优化器：    

在每次训练过程中    
1.计算损失 - 2.优化器梯度清零 - 3.损失反向传播 - 4.优化器步进优化参数
```python
#定义优化器和损失
#创建损失函数
loss_fn = nn.CrossEntropyLoss()
#优化器
lr = 1e-2
optimizer = torch.optim.SGD(network.parameters(), lr)

#训练轮数↓训练过程
epoch = 10
writer = SummaryWriter("train_prog")

for i in range(epoch):
    print("--------第{}轮训练开始--------".format(i+1))
    #训练步骤
    for data in train_dataloader:
        imgs , targets = data
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
            print("训练次数：{}，Loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    #每训练完一轮，查看正确率
    total_test_loss = 0
    total_correct = 0
    #确保没有修改参数
    with torch.no_grad():
        for data in test_dataloader:
            imgs , targets = data
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
```
此处有个小细节：关于tensor.argmax()
argmax可以直接指出向量中最大数的下标（0开始）
argmax（0/1）代表观察方向，如果是0竖着看，1横着看
![img_7.png](img_7.png)
```python
import torch
ouput = torch.tensor([[0.5,0.6],
                      [0.2,0.5]])
print(ouput.argmax(1))
```
结果为tensor([1, 1])，说明横着看，第一行0.6大，第二行0.5大  
所以上面有
> correct = (outputs.argmax(1) == targets).sum()
            total_correct += correct  
> 因为上述是个分类问题，会给出每个分类的概率值，我们选取概率最大的作为答案并进行对比

代码内容具体在P18_HowToTrainModel中
如果你的网络层有Dropout和Batchnorm层，可以使用model.train（）和model.eval（）  
[具体参考torch.nn.module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)
![img_8.png](img_8.png)

## 9.GPU加速训练
1.方法1 使用.cuda  
我们可以在模型、loss函数、取数据时（imgs和targets）当场调用.cuda()  
记得使用if torch.cuda.is_available():判断有没有cuda
![img_9.png](img_9.png)

2.方法2 定义设备device
先在上方定义device = torch.device("xxx")
* 1.cpu 2.cuda 3.cuda:0(如果有多张显卡)
![img_10.png](img_10.png)

