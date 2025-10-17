import os
import sys
import base64
import requests

API_URL = "https://api.chatanywhere.tech/v1/images/edits"
API_KEY = "sk-tYaEN7C1LxqhE2ELmjlMrJruFsca9bzJJgaN1Fv7var6t84V"
if not API_KEY:
    sys.exit("Missing environment variable: OPENAI_API_KEY")

if len(sys.argv) < 3:
    print("Usage: python edit_image_api.py <input_image> \"<prompt>\" [output_image]")
    sys.exit(1)

image_path = sys.argv[1]
prompt_text = sys.argv[2]
output_path = sys.argv[3] if len(sys.argv) > 3 else "output.png"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
    "Accept": "*/*",
    "Host": "api.chatanywhere.tech",
    "Connection": "keep-alive"
}
files = {"image": open(image_path, "rb")}
data = {"prompt": prompt_text, "model": "gpt-image-1"}

try:
    response = requests.post(API_URL, headers=headers, data=data, files=files, timeout=120)
    response.raise_for_status()
    image_base64 = response.json()["data"][0]["b64_json"]
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(image_base64))
    print(f"Image saved to {output_path}")
except Exception as e:
    print("Image generation failed:", e)
    print("Response:", response.text)

