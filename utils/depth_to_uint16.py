import numpy as np
from PIL import Image
import imageio
import sys
import os

DEPTH_SCALE = 2**24 - 1
NEAR, FAR = 0.01, 4.50

def rgb_image_to_depth(image_path):
    img = Image.open(image_path).convert('RGB')
    rgb = np.array(img).astype(np.uint32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    int_val = (r * 256 + g) * 256 + b
    depth_nm = int_val.astype(np.float32) / DEPTH_SCALE
    depth_m = depth_nm * (FAR - NEAR) + NEAR
    return depth_m

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python utils/depth_to_uint16.py <input_rgb_depth.png> <output_depth.png>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    depth = rgb_image_to_depth(input_path)

    depth_mm = (depth * 1000).astype(np.uint16)
    imageio.imwrite(output_path, depth_mm)

    print(f"Depth map saved to: {output_path}")
