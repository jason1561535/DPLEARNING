import torch
import torchvision

vgg16 = torchvision.models.vgg16()
#保存方式1,保存   模型结构+模型参数
torch.save(vgg16,'../models_saved/vgg16_method1.pth')

#保存方式2，保存   模型参数（官方推荐）
torch.save(vgg16.state_dict(),'../models_saved/vgg16_method2.pth')