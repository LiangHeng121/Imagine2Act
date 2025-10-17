import numpy as np
import open3d as o3d
from PIL import Image
import argparse
import os

# ========== Camera Intrinsics ==========
fx, fy = -351.6771208, -351.6771208
cx, cy = 128.0, 128.0

# ========== Argument Parsing ==========
parser = argparse.ArgumentParser()
parser.add_argument("ply_path", help="Input .ply point cloud path (e.g., fused_scene_filtered.ply)")
parser.add_argument("depth_image", help="Encoded RGB depth image used to recover H, W")
parser.add_argument("output_npy", help="Output .npy path (saved as (H, W, 3) dense point map)")
args = parser.parse_args()

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

depth = rgb_image_to_depth(args.depth_image)
H, W = depth.shape

# ========== Projection Function ==========
def project_points_to_image(points, fx, fy, cx, cy, H, W):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    u = np.round((x * fx / z) + cx).astype(int)
    v = np.round((y * fy / z) + cy).astype(int)

    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v = u[valid], v[valid]
    proj_points = points[valid]

    dense = np.zeros((H, W, 3), dtype=np.float32)
    dense[:, :, :] = 0.0
    for idx in range(len(proj_points)):
        dense[v[idx], u[idx]] = proj_points[idx]
    return dense

# ========== Load and Project Point Cloud ==========
print("Projecting point cloud to dense image...")
pcd = o3d.io.read_point_cloud(args.ply_path)
points = np.asarray(pcd.points)
dense_pcd = project_points_to_image(points, fx, fy, cx, cy, H, W)

# ========== Save ==========
np.save(args.output_npy, dense_pcd)
print(f"Dense point cloud image saved to: {args.output_npy} (shape: {dense_pcd.shape})")
