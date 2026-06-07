import onnxruntime as ort
import numpy as np
from PIL import Image
import base64
import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _model_embedded import MODEL_B64
from flask import request
import vercel

app = vercel.app

CLASS_NAMES = ["一花", "五月", "三玖", "二乃", "四叶"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_session = None
_tmp_model_path = None


def _get_model_path():
    global _tmp_model_path
    if _tmp_model_path is None:
        model_bytes = base64.b64decode(MODEL_B64)
        fd, _tmp_model_path = tempfile.mkstemp(suffix=".onnx")
        os.write(fd, model_bytes)
        os.close(fd)
    return _tmp_model_path


def _load_session():
    global _session
    if _session is None:
        _session = ort.InferenceSession(
            _get_model_path(), providers=["CPUExecutionProvider"]
        )
    return _session


def _preprocess(img: Image.Image) -> np.ndarray:
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    for c in range(3):
        arr[:, :, c] = (arr[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]
    return arr.astype(np.float32)


@app.route("/predict", methods=["POST", "OPTIONS"])
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
        img_tensor = _preprocess(img)
    except Exception as e:
        return json.dumps({"error": f"图片解析失败: {str(e)}"}), 400, {"Content-Type": "application/json"}

    try:
        sess = _load_session()
        input_name = sess.get_inputs()[0].name
        output = sess.run(None, {input_name: img_tensor})[0]
        probs = np.exp(output) / np.exp(output).sum(axis=1, keepdims=True)
        pred_idx = int(np.argmax(probs, axis=1)[0])
        conf = float(np.max(probs, axis=1)[0])

        pred_class = CLASS_NAMES[pred_idx]
    except Exception as e:
        return json.dumps({"error": f"推理失败: {str(e)}"}), 500, {"Content-Type": "application/json"}

    return json.dumps({"name": pred_class, "confidence": round(conf, 4)}), 200, {"Content-Type": "application/json"}
