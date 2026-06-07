import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import base64
import io

# 全局模型实例（Vercel 冷启动时加载一次，之后复用）
_model = None
CLASS_NAMES = ["一花", "五月", "三玖", "二乃", "四叶"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def _build_model():
    """构建 ResNet18 模型并加载权重"""
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 5)
    return m


def _load_model():
    """懒加载模型：只在第一次调用时加载"""
    global _model
    if _model is None:
        _model = _build_model()
        _model.load_state_dict(
            torch.load("model/resnet18_nakano.pth",
                       map_location="cpu",
                       weights_only=True)
        )
        _model.eval()
    return _model


def handler(request):
    """
    Vercel Serverless 函数入口。
    Vercel 会把 HTTP 请求封装为 `request` 对象传进来。
    """
    # ---- 解析请求体 ----
    try:
        body = request.get_json(silent=True)
    except Exception:
        return {"statusCode": 400, "body": '{"error": "无法解析 JSON"}'}

    if body is None:
        return {"statusCode": 400, "body": '{"error": "请求体为空"}'}

    image_b64 = body.get("image")
    if not image_b64:
        return {"statusCode": 400, "body": '{"error": "缺少 image 字段"}'}

    # ---- 图片预处理 ----
    try:
        img_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_tensor = _transform(img).unsqueeze(0)
    except Exception as e:
        return {"statusCode": 400, "body": f'{{"error": "图片解析失败: {str(e)}"}}'}

    # ---- 推理 ----
    try:
        model = _load_model()
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, pred_idx = probs.max(1)

        pred_class = CLASS_NAMES[pred_idx.item()]
        conf = round(confidence.item(), 4)
    except FileNotFoundError:
        return {
            "statusCode": 500,
            "body": '{"error": "模型文件未找到，请确保 model/resnet18_nakano.pth 已上传"}'
        }
    except Exception as e:
        return {"statusCode": 500, "body": f'{{"error": "推理失败: {str(e)}"}}'}

    # ---- 返回结果 ----
    import json
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"name": pred_class, "confidence": conf})
    }
