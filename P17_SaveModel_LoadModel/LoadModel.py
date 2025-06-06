import torch
import torchvision

#加载1->对应->保存1 读取模型+参数
model = torch.load('../models_saved/vgg16_method1.pth',weights_only= False)
print(model)
#方式1存在陷阱
#你每次都需要重新定义你自己的模型类


#加载2->对应->保存2 读取字典
vgg16 = torchvision.models.vgg16()
vgg16.load_state_dict(torch.load('../models_saved/vgg16_method2.pth'))
# model = torch.load('../models_saved/vgg16_method2.pth',weights_only= False)
print(model)