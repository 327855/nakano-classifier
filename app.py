from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

app = Flask(__name__)
CORS(app)   # 允许前端跨域访问

# 加载 ResNet18 模型（复用你之前训练好的模型文件）
model = torch.load("model/resnet18_nakano.pth", map_location="cpu", weights_only=False)
model.eval()

# 类别名称（按文件夹顺序，对应索引 0~4）
CLASS_NAMES = ["一花", "五月", "三玖", "二乃", "四叶"]

# 图片预处理（和训练时保持一致）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

@app.route("/api/predict", methods=["POST"])
def predict():
    # 检查是否有图片
    if "image" not in request.files:
        return jsonify({"error": "没有上传图片"}), 400

    file = request.files["image"]

    # 读取并预处理图片
    try:
        img = Image.open(file).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)

        # 模型推理
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, pred_idx = probs.max(1)

        pred_class = CLASS_NAMES[pred_idx.item()]
        conf = confidence.item()

        return jsonify({
            "name": pred_class,
            "confidence": round(conf, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)