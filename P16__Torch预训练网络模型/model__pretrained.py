import torchvision
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)
print("ok")

print(vgg16_true)

train_data = torchvision.datasets.CIFAR10("../data", train=True, download=True, transform=torchvision.transforms.ToTensor())

vgg16_true.add_module('add_linear',nn.Linear(1000,10))
print(vgg16_true)