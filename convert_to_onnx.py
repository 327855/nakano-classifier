import os
import torch
import torch.nn as nn
from torchvision import models

m = models.resnet18(weights=None)
m.fc = nn.Linear(m.fc.in_features, 5)

m.load_state_dict(
    torch.load("model/resnet18_nakano.pth", map_location="cpu", weights_only=True)
)
m.eval()

os.makedirs("model", exist_ok=True)

dummy_input = torch.randn(1, 3, 224, 224)

# Disable dynamo to use legacy exporter which properly embeds weights
torch.onnx.export(
    m,
    dummy_input,
    "model/resnet18_nakano.onnx",
    export_params=True,
    input_names=["input"],
    output_names=["output"],
    dynamo=False,
)

size = os.path.getsize("model/resnet18_nakano.onnx")
print(f"ONNX export done: {size / 1024 / 1024:.1f} MB")
