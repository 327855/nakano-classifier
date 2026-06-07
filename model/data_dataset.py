import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from pathlib import Path
from typing import Optional
from torchvision import transforms
import os

import config

def scan_class_folders(data_root:str)->tuple[list[str],dict[str,int]]:
    data_root=Path(data_root)
    #目录下所有子文件夹的名称，作为类别名，并按字母顺序排序，
    #iterdir()是Path对象的方法，返回目录下所有文件和子目录的 迭代器 ，is_dir()是判断是否是目录（只保留目录）
    class_names=sorted([d.name for d in data_root.iterdir() if d.is_dir()])
   #enumerate() 是Python内置函数，用于遍历序列（如列表）时同时获得元素的索引和对应的值。
    class_to_idx={name: idx for idx,name in enumerate(class_names)}

    image_paths=[]
    labels=[]

    for class_name in class_names:
        class_dir=data_root/class_name #拼接 出所属目录的完整路径
        for ext in ("jpg","jpeg","png","webp"):
            #遍历 class_dir 目录下所有扩展名为 ext（如 jpg/png等）的文件。
            for image_path in sorted(class_dir.glob(f"*.{ext}")):
                image_paths.append(str(image_path))
                labels.append(class_to_idx[class_name])# 添加对应类别索引到标签列表
    return image_paths,class_to_idx

 #class_names: list[str]：类型注解，参数是字符串列表，-> dict[str, int]：返回值是{字符串: 整数}的字典
def build_label_encoder(class_names:list[str])->dict[str,int]:
    #将类别名列表转换为 {类别名: 整数ID} 映射
    return {name: idx for idx,name in enumerate(class_names)}

def split_file_list(image_paths,labels,train_ratio=0.8,val_ratio=0.1,seed=42):
    #train_ratio表示训练集占总数据的比例，val_ratio 表示验证集占总数据的比例。
    assert 0 < train_ratio < 1 and 0 <= val_ratio < 1 and train_ratio + val_ratio <= 1, \
        "train_ratio 必须在 (0,1) 之间，val_ratio 必须在 [0,1) 之间，且两者之和不能超过 1"
    random.seed(seed)
    #生成从 0 到 图片数-1 的整数序列(每个数字对应图片的索引）,同时转成列表
    indices=list(range(len(image_paths)))
    # 随机打乱索引顺序
    random.shuffle(indices)

    # 计算验证集数量
    n_train=int(len(indices)*train_ratio)
    n_val=int(len(indices)*val_ratio)

    # 前 n_train 个索引用于训练集
    train_indices=indices[:n_train]
    # 接下来的 n_val 个索引用于验证集
    val_indices=indices[n_train:n_train+n_val]
    # 剩下的索引用于测试集
    test_indices=indices[n_train+n_val:]
    # 根据传入的索引子集返回对应的图片路径和标签
    def subset(sub_idx):
        return ([image_paths[i] for i in sub_idx],
                [labels[i] for i in sub_idx])
    return {
        "train":subset(train_indices),
        "val":subset(val_indices),
        "test":subset(test_indices)
    }

def save_split_info(split_data, save_path):
    serialalizable={
        split_name:{"path":paths,"labels":labels}
        for split_name,(paths,labels) in split_data.items()
    }
    with open(save_path,"w",encoding="utf-8") as f:
        json.dump(serialalizable,f,indent=4,ensure_ascii=False)

def load_split_info(load_path:str):
    with open(load_path,"r",encoding="utf-8") as f:
        #读取并解码 JSON 为 Python 字典。
        raw=json.load(f)
    return{
        split_name:(info["path"],info["labels"])
        for split_name,info in raw.items()
    }

def get_train_transforms(img_size,mean,std):
    return transforms.Compose([transforms.Resize(int(img_size*1.1)),
                               transforms.RandomCrop(img_size),
                               transforms.RandomHorizontalFlip(p=0.5),
                               transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.2),
                              # transforms.RandomApply([transforms.RandomAffine(scale=(0.02, 0.1))], p=0.1),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=mean,std=std),
                               transforms.RandomErasing(p=0.1,scale=(0.02,0.1)),
                               ])

def get_val_transforms(img_size,mean,std):
    return transforms.Compose([transforms.Resize(int(img_size*1.1)),
                               transforms.CenterCrop(img_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=mean,std=std),
                               ])


class FolderDataset(Dataset):
    """
        文件夹分类数据集。
        每个样本 = (图像, 标签)。
        标签从图像所在文件夹的名称解析得到。
        """
    def __init__(self,image_paths,labels,transform=None):
        assert len(image_paths)==len(labels),"路径不匹配"
        self.image_paths=image_paths
        self.labels=labels
        self.transform=transform

    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img_path=self.image_paths[idx]
        label=self.labels[idx]
        image=Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image=self.transform(image)
        return image,label

def build_data_loaders(spilt_data,train_transform,val_transform,batch_size=16,num_workers=4,pin_memory=True):
    """
    封装 DataLoader。
    Args:
        split_data:     split_file_list() 的返回值
        train_transform: 训练集 transforms
        val_transform:   验证/测试集 transforms
        batch_size:     每批次样本数，受 GPU 显存限制
        num_workers:    数据加载子进程数，4 是安全起点
        pin_memory:     True 会把数据加载到锁页内存，加速 GPU 传输
    Returns:
        {"train": train_loader, "val": val_loader, "test": test_loader}
    """
    loaders={}
    for split_name,(paths,labels) in spilt_data.items():
        shuffle=(split_name=="train")
        transform=train_transform if split_name=="train" else val_transform

        dataset=FolderDataset(paths,labels,transform=transform)

        loaders[split_name]=DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split_name=="train")
        )
    return loaders



def verify_data_pipeline(data_loader:DataLoader,expected_size:int=224,num_classes:int=5):
    images,labels=next(iter(data_loader))
    print(f"[验证]bath 形状:images={images.shape},labels={labels.shape}")
    #assert 语句：这是 Python 的断言机制。如果 assert 后面的条件为 False，程序会立刻抛出 AssertionError 并终止运行，同时打印出你预设的报错信息。
    assert images.ndim==4 and images.shape[1]==3 ,"图像应为 [B, 3, H, W]"
    assert images.shape[2]==images.shape[3]==expected_size,"图像尺寸不符合预期"
    assert labels.min()>=0 and labels.max()<num_classes, "标签值超出范围"
    assert not torch.isnan(images).any(),"图像尺寸不符合预期"
    assert not torch.isinf(images).any(), "图像包含 Inf"
    print("[验证] 通过！数据流水线正常。")

def check_class_balance(labels:list[int],class_names:list[str]):
    from collections import Counter
    #Counter 类：Python 标准库 collections 模块中的高效计数器。它会自动统计可迭代对象中每个元素的出现次数，返回一个字典子类。
    counts=Counter(labels)
    print("类别分布：")
    #sorted(counts.items())：按类别ID（而非数量）排序，确保输出结果可复现且易读。
    #.items()：返回 (key, value) 对的视图
    for class_id,count in sorted(counts.items()):
        class_name=class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        print(f"[{class_id}]{class_name}:{count}张")

def preview_augmented_samples(data_loader:DataLoader,save_path:str,num_samples:int=8):
    import matplotlib.pyplot as plt
    images,labels=next(iter(data_loader))
    #只取前 num_samples=8 个样本，避免图像过多导致显示混乱
    images=images[:num_samples]
    mean=torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std=torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    images=images*std+mean
    #将像素值限制在 [0, 1] 范围内
    images=torch.clamp(images,0,1)

    #创建网格布局
    #自适应网格：计算 num_samples 的平方根并向上取整，确保所有样本都能显示
    #例如：8 个样本 → √8 ≈ 2.83 → 向上取整为 3 → 创建 3×3=9 个子图
    grid_size=int(np.ceil(num_samples**0.5))
    fig,axes=plt.subplots(grid_size,grid_size,figsize=(12,12))
    #将 2D 的 axes 网格展平为 1D 列表，便于循环处理
    axes=axes.flatten()

    for i,ax in enumerate(axes):
        if i<len(images):
            ax.imshow(images[i].permute(1,2,0).cpu())
            ax.set_title(f"Label: {labels[i]}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[预览] 图像已保存到 {save_path}")
