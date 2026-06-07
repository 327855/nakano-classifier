import torch
import torch.nn as nn
#用于导入 torchvision 库自带的各种预训练模型（如 ResNet、VGG 等），方便在代码中直接调用用于特征提取或迁移学习。
from torchvision import models

def build_model(num_classes:int,pretrained:bool=True)->nn.Module:
    model=models.resnet18(weights="IMAGENET1K_V1"if pretrained else None)
    #获取模型的输入特征数
    in_features=model.fc.in_features
    #将模型的全连接层替换为新的全连接层，输入特征数为 num_classes=5，输出特征数为 num_classes
    model.fc=nn.Linear(in_features,num_classes)
    return model

