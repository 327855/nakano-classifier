from pathlib import Path

# ============================================================
# 模块导入
# ============================================================
# 从 data_dataset.py 导入数据处理相关的所有工具函数

from model.data_dataset import (
    scan_class_folders,      # 扫描文件夹，构建图像路径和类别映射
    split_file_list,         # 随机划分训练/验证/测试集
    save_split_info,         # 将划分结果保存为 JSON
    get_train_transforms,    # 获取训练集的数据增强流水线
    get_val_transforms,       # 获取验证/测试集的预处理流水线
    build_data_loaders,      # 封装 PyTorch DataLoader
    verify_data_pipeline,    # 验证数据格式是否正确
    check_class_balance,     # 检查各类别样本数量是否均衡
    preview_augmented_samples,  # 可视化数据增强效果
)


# ============================================================
# 配置参数
# ============================================================
# 使用 Path(__file__).parent 确保路径始终相对于脚本所在目录，
# 而不是相对于当前工作目录（cd 到哪里运行，路径就以哪里为基准）

SCRIPT_DIR    = Path(__file__).parent    # 脚本所在目录 -> model/
PROJECT_ROOT  = SCRIPT_DIR.parent        # 项目根目录 -> PythonProject5/

DATA_ROOT    = str(PROJECT_ROOT / "nakano-images")   # 数据集根目录
SPLIT_JSON   = str(SCRIPT_DIR / "split_data.json")   # 划分结果保存路径
NUM_CLASSES  = 5                                         # 分类类别数
BATCH_SIZE   = 16                                        # 每批次图像数
IMG_SIZE     = 224                                       # 图像 resize 到的尺寸

# ImageNet 标准化参数
# 预训练模型（如 ResNet）是在 ImageNet 数据上训练的，
# 使用相同的 mean/std 可以让输入分布与模型训练时一致
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ============================================================
# 主执行流程
# ============================================================
# Windows 上 multiprocessing 使用 spawn 方式启动子进程，
# 子进程会重新导入主模块。必须用 if __name__ == "__main__": 包裹，
# 否则子进程在导入时会再次执行顶层代码，导致 RuntimeError
if __name__ == "__main__":

    # ---------- 第 1 步：扫描文件夹 ----------
    # 遍历 DATA_ROOT 下的子文件夹，每个子文件夹名作为一个类别
    # 返回：所有图像的绝对路径列表 + {类别名: 索引} 映射
    image_paths, class_to_idx = scan_class_folders(DATA_ROOT)
    class_names = list(class_to_idx.keys())
    print(f"共找到 {len(image_paths)} 张图像，{len(class_names)} 个类别")
    print("类别映射：", class_to_idx)

    # ---------- 第 2 步：划分 ----------
    # 从每张图像的路径中提取其所属类别的索引（0~4）
    labels = [class_to_idx[Path(p).parent.name] for p in image_paths]
    # 按 80% / 10% / 10% 随机划分，固定 seed=42 保证可复现
    split_data = split_file_list(image_paths, labels, train_ratio=0.8, val_ratio=0.1)

    # 将划分结果持久化到 JSON，下次直接用 load_split_info 加载，不必重新划分
    save_split_info(split_data, SPLIT_JSON)
    print(f"划分结果已保存到 {SPLIT_JSON}")

    # ---------- 第 3 步：构建 transforms ----------
    # 训练集：随机裁剪、水平翻转、颜色抖动、随机仿射变换、随机擦除
    # 验证/测试集：仅中心裁剪，不做增强
    train_transform = get_train_transforms(img_size=IMG_SIZE, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    val_transform   = get_val_transforms(img_size=IMG_SIZE, mean=IMAGENET_MEAN, std=IMAGENET_STD)

    # ---------- 第 4 步：构建 DataLoader ----------
    # num_workers=0：Windows 上多进程有坑，验证阶段用单线程更稳定
    #              训练阶段可以根据需要调大，但务必在 if __name__ 下运行
    loaders = build_data_loaders(
        split_data,
        train_transform,
        val_transform,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=False,    # 没有 GPU，无需锁页内存
    )

    # ---------- 第 5 步：数据验证 ----------
    # 5.1 检查类别分布——理想情况下每个类别的样本数接近
    check_class_balance(
        split_data["train"][1],    # 训练集的标签列表
        class_names=class_names,  # 传入类别名以显示可读标签
    )

    # 5.2 验证数据格式
    #    - 图像形状是否正确 [B, 3, H, W]
    #    - 尺寸是否为 IMG_SIZE
    #    - 标签是否在 [0, num_classes) 范围内
    #    - 是否有 NaN / Inf
    verify_data_pipeline(loaders["train"], expected_size=IMG_SIZE, num_classes=NUM_CLASSES)

    # 5.3 可视化增强效果
    #    反标准化（还原到 [0,1]）后拼接成网格图，保存到 model/preview_train.png
    preview_augmented_samples(
        loaders["train"],
        save_path=str(SCRIPT_DIR / "preview_train.png"),
        num_samples=8,
    )

    # ---------- 第 6 步：准备传递给训练循环 ----------
    # 这三个 DataLoader 可以直接传给训练脚本使用
    train_loader = loaders["train"]
    val_loader   = loaders["val"]
    test_loader  = loaders["test"]

    print("数据验证完成！可以开始训练。")