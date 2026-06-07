import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import base64
import io
import json
import os

app = __import__("vercel").app

CLASS_NAMES = ["一花", "五月", "三玖", "二乃", "四叶"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

_model = None


def _build_model():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 5)
    return m


def _load_model():
    global _model
    if _model is None:
        _model = _build_model()
        model_path = os.path.join(os.path.dirname(__file__), "..", "model", "resnet18_nakano.pth")
        _model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        _model.eval()
    return _model


@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 204

    body = request.get_json(silent=True)
    if body is None:
        return json.dumps({"error": "请求体为空"}), 400, {"Content-Type": "application/json"}

    image_b64 = body.get("image")
    if not image_b64:
        return json.dumps({"error": "缺少 image 字段"}), 400, {"Content-Type": "application/json"}

    try:
        img_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_tensor = _transform(img).unsqueeze(0)
    except Exception as e:
        return json.dumps({"error": f"图片解析失败: {str(e)}"}), 400, {"Content-Type": "application/json"}

    try:
        model = _load_model()
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, pred_idx = probs.max(1)

        pred_class = CLASS_NAMES[pred_idx.item()]
        conf = round(confidence.item(), 4)
    except FileNotFoundError:
        return json.dumps({"error": "模型文件未找到"}), 500, {"Content-Type": "application/json"}
    except Exception as e:
        return json.dumps({"error": f"推理失败: {str(e)}"}), 500, {"Content-Type": "application/json"}

    return json.dumps({"name": pred_class, "confidence": conf}), 200, {"Content-Type": "application/json"}
