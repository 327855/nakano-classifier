from tkinter import N
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

#用于显示进度条，常用于遍历数据集时直观显示处理进度。
from tqdm import tqdm
#用于读写 JSON 格式数据，常见于参数配置或结果保存
import json
#用于方便地进行文件和路径操作，更直观、跨平台地管理文件路径。
from pathlib import Path

from model.model_main import build_model
from model.data_dataset import(
    scan_class_folders,
    split_file_list,
    save_split_info,
    load_split_info,
    get_train_transforms,
    get_val_transforms,
    build_data_loaders,
)


SCRIPT_DIR    = Path(__file__).parent
PROJECT_ROOT  = SCRIPT_DIR.parent
DATA_ROOT    = str(PROJECT_ROOT / "nakano-images")
SPLIT_JSON   = str(SCRIPT_DIR / "split_data.json")

NUM_CLASSES  = 5
BATCH_SIZE   = 16
IMG_SIZE     = 224
NUM_EPOCHS   = 30

LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
PATIENCE      = 5          # 早停：验证 loss 连续 5 个 epoch 不下降则停止

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]



def train_one_epoch(model:nn.Module,loader:DataLoader,criterion:nn.Module,optimizer:torch.optim.Optimizer,device:torch.device,epoch:int):
    #criterion:       损失函数（交叉熵）
    #optimizer:       优化器（SGD / Adam / ...）
    model.train()
    total_loss=0.0
    correct=0
    total=0

    #tqdm 用于创建进度条。这里它包裹 loader（数据加载器），desc 设置进度条左侧显示文本，leave=False 表示循环结束后不保留进度条。
    pbar=tqdm(loader,desc=f"Epoch{epoch}[Train]",leave=False)
    #enumerate(pbar) 遍历 pbar（即数据集每一批），返回批次索引和数据。每次迭代解包为 images,labels，batch_idx 是批次数。
    for batch_idx,(images,labels) in enumerate(pbar):
        images=images.to(device)
        labels=labels.to(device)

        """
        将一个批次的输入 images 送入神经网络 model，
        返回预测结果（模型的输出张量，通常是各类别的未归一化分数 logits）

        outputs形状一般为 [batch, num_classes]，每行代表一个样本各类别的得分。
        """
        outputs=model(images)
        #将模型输出 outputs 与真实标签 labels 输入到损失函数 criterion（如交叉熵）
        loss=criterion(outputs,labels)

        optimizer.zero_grad()
        """
        利用自动微分机制，基于损失对所有可学习参数求导，生成当前 batch 的梯度（每个参数的 `.grad` 属性会被赋值）。
        - 随后可用 `optimizer.step()` 用计算出的梯度对参数执行优化更新。
        """
        loss.backward()

        optimizer.step()

        #累加当前 batch 的损失（loss.item()把张量转为Python数值），用于计算整个 epoch 的平均损失
        total_loss+=loss.item()
        """
         #对每个样本的 logits 取最大值索引，得到模型预测的类别（dim=1指按类别轴）。
        
        outputs形状一般为 [batch, num_classes]，每行代表一个样本各类别的得分。
        .max(1)返回每行最大值和所在索引，其中predicted就是最大值的索引，即模型判定的类别ID。
        这么做等价于取概率最大（最可信）类别作为最终分类结果，是常见的多分类推理方法
        """
        _,predicted=outputs.max(1)
        #predicted.eq(labels)返回布尔Tensor，sum后得到正确总数。
        correct+=predicted.eq(labels).sum().item()
        #累加本batch的样本数量（batch size），用于后续计算总体准确率。
        total+=labels.size(0)

        pbar.set_postfix({
            "loss":f"{loss.item():.4f}",
            "acc":f"{100.0*correct/total:.2f}%"
        })
    
    avg_loss=total_loss/total
    accuracy=100.*correct/total
    #本 epoch 的平均损失和准确率，返回字典格式便于记录和分析
    return {"loss":avg_loss,"accuracy":accuracy}

@torch.no_grad()
def validate(model:nn.Module,loader:DataLoader,criterion:nn.Module,device:torch.device,epoch:int):
    model.eval()
    total_loss=0.0
    correct=0
    total=0

    pbar=tqdm(loader,desc=f"Epoch{epoch}[Validate  ]",leave=False)
    for images,labels in pbar:
        images=images.to(device)
        labels=labels.to(device)
        
        outputs=model(images)
        loss=criterion(outputs,labels)

        total_loss+=loss.item()
        _,predicted=outputs.max(1)
        correct+=predicted.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            "loss":f"{loss.item():.4f}",
            "acc":f"{100.0*correct/total:.2f}%"
        })

    avg_loss=total_loss/total
    accuracy=100.*correct/total
    return {"loss":avg_loss,"accuracy":accuracy}


if __name__=="__main__":
    print(f"使用设备：{DEVICE}\n")
    # ---------- 第 1 步：扫描文件夹 ----------
    image_paths,class_to_idx=scan_class_folders(DATA_ROOT)
    #class_to_idx字典的所有键（即类别名）以列表形式存入class_names变量。
    class_names=list(class_to_idx.keys())
    print(f"数据：{len(image_paths)}张图像，{len(class_names)}个类别")

    """
    path(p).parent.name：对每个图片路径p，获取其上一级文件夹名（即类别名）。
    class_to_idx[path(p).parent.name]：将类别名映射成类别索引（数字标签）。
    """
    labels=[class_to_idx[Path(p).parent.name] for p in image_paths]
    split_data=split_file_list(image_paths,labels,train_ratio=0.8,val_ratio=0.1)
    
    train_transforms=get_train_transforms(img_size=IMG_SIZE,mean=IMAGENET_MEAN,std=IMAGENET_STD)
    val_transforms=get_val_transforms(img_size=IMG_SIZE,mean=IMAGENET_MEAN,std=IMAGENET_STD)

    loader=build_data_loaders(split_data,train_transforms,
        val_transforms,batch_size=BATCH_SIZE,num_workers=4,pin_memory=True
    )

    
    # ---------- 模型 ----------
    model=build_model(num_classes=NUM_CLASSES,pretrained=True).to(DEVICE)
    print(f"模型: ResNet18 (预训练)")

    #定义损失函数为交叉熵损失，适用于多分类任务，衡量模型输出概率分布与真实类别之间的差异。
    criterion=nn.CrossEntropyLoss()
    #使用Adam优化器更新模型参数，支持自适应学习率，weight_decay控制L2正则，防止过拟合。
    optimizer=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    #使用StepLR学习率调度器，每NUM_EPOCHS个epoch将学习率乘以0.1，实现学习率衰减。
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # ---------- 训练 ----------
    history={"train_loss":[],"train_acc":[],"val_loss":[],"val_acc":[]}
    #保存迄今为止最低的验证损失，用于模型早停或保存最佳权重。
    best_val_loss=float("inf")
    #记录在验证 损失没有提升时经过的epoch数，通常用于实现Early Stopping策略。
    patience_counter=0

    for epoch in range(1,NUM_EPOCHS+1):
        train_metrics=train_one_epoch(model,loader["train"],criterion,optimizer,DEVICE,epoch)
        val_metrics  =validate(model,loader["val"],criterion,DEVICE,epoch)

        scheduler.step()

        """
        .append()是列表的方法，用于在列表末尾添加一个元素。
        train_metrics["loss"]等用字典键获取训练/验证结果中的具体数值，添加到history列表中。
        """
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

        print(
            f"Epoch {epoch}/{NUM_EPOCHS}|"
            f"Train Loss: {train_metrics['loss']:.4f},acc: {train_metrics['accuracy']:.2f}%|"
            f"Val Loss: {val_metrics['loss']:.4f},acc: {val_metrics['accuracy']:.2f}%"
        )

        # 保存最佳模型（验证 loss 最小时）
        if val_metrics["loss"]<best_val_loss:
            best_val_loss=val_metrics["loss"]
            patience_counter=0
            best_path=SCRIPT_DIR/"resnet18_nakano.pth"
            torch.save(model.state_dict(), f"resnet18_nakano.pth")
            print(f" ->最佳模型已保存到 {best_path}")

        patience_counter+=1
        if patience_counter>=PATIENCE:
            print(f"\n早停：验证损失连续{PATIENCE}个epoch没有下降，停止训练")
            break

        history_path=SCRIPT_DIR/"training_history.json"
        """
        `with open(...) as f:` 表示用写模式打开文件，`json.dump(...)` 用于写入，
        `indent=4` 表示生成格式化（缩进4空格）的 JSON 文件，编码为 utf-8。
        indent=4表示缩进4个空格，ensure_ascii=False表示使用UTF-8编码，避免中文乱码。
        """
        with open(history_path,"w",encoding="utf-8") as f:
            json.dump(history,f,indent=4)

        print(f"\n训练完成，历史记录已保存到 {history_path}")
        print(f"最终验证准确率: {history['val_acc'][-1]:.2f}%")