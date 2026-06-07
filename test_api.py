import urllib.request, json, base64, os

img_path = os.path.join(os.path.dirname(__file__), "wed", "images", "ichika.jpg")
with open(img_path, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

payload = json.dumps({"image": img_b64}).encode()

req = urllib.request.Request(
    "https://nakano-classifier.vercel.app/api/predict",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST"
)

try:
    with urllib.request.urlopen(req, timeout=60) as r:
        print("Status:", r.status)
        print("Response:", r.read().decode())
except urllib.error.HTTPError as e:
    print("HTTP Error:", e.code, e.read().decode())
except Exception as e:
    print("Error:", type(e).__name__, str(e))
