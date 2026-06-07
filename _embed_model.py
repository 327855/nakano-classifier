import base64, os

model_path = "model/resnet18_nakano.onnx"
out_path = "_model_embedded.py"

with open(model_path, "rb") as f:
    data = f.read()

b64 = base64.b64encode(data).decode()
n = 100
chunks = [b64[i:i+n] for i in range(0, len(b64), n)]

with open(out_path, "w") as f:
    f.write(f'# Auto-generated. DO NOT EDIT.\nMODEL_B64 = """\n')
    for i, chunk in enumerate(chunks):
        f.write(f"{chunk}" + ("\\\n" if i < len(chunks)-1 else ""))
    f.write('\n"""\n')

print(f"Generated {out_path}")
print(f"Model size: {len(data)/1024/1024:.1f} MB -> {len(b64)/1024/1024:.1f} MB base64")
