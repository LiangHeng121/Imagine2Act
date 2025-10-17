import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("npy_path", help="Input: dense point cloud image (.npy) in camera coordinates (H, W, 3)")
parser.add_argument("extrinsic_txt", help="Input: Camera-to-World extrinsic matrix (.txt, 4x4)")
parser.add_argument("output_npy", help="Output: dense point cloud image (.npy) in world coordinates")
args = parser.parse_args()

# === Load dense point cloud image ===
dense_cam = np.load(args.npy_path)
H, W = dense_cam.shape[:2]

# === Load extrinsic matrix (camera to world) ===
T = np.loadtxt(args.extrinsic_txt)
R, t = T[:3, :3], T[:3, 3]

# === Apply transformation to all valid points ===
dense_world = np.zeros_like(dense_cam)
dense_world[:, :, :] = 0.0

valid_mask = ~(dense_cam == 0).all(axis=2)
points_flat = dense_cam[valid_mask]
points_world = (R @ points_flat.T + t.reshape(3, 1)).T

dense_world[valid_mask] = points_world

# === Save ===
np.save(args.output_npy, dense_world)
print(f"Dense point cloud in world coordinates saved to: {args.output_npy} (shape: {dense_world.shape})")