import numpy as np
import open3d as o3d
from PIL import Image
import cv2
import sys
import os

# ========== Parameter Parsing ==========
rgb_path = sys.argv[1]           # RGB image path
depth_rgb_path = sys.argv[2]     # RGB-encoded depth map path
seg_path = sys.argv[3]           # segmentation image (with alpha channel)
seg_path_2 = sys.argv[4]         # second segmentation image (with alpha channel)
object_pcd_path = sys.argv[5]    # TripoSR point cloud path
pose_path = sys.argv[6]          # FoundationPose output pose.txt
output_path = sys.argv[7]        # output path
scale_2 = float(sys.argv[8])

# === Camera intrinsics ===
fx, fy = -351.6771208, -351.6771208
cx, cy = 128.0, 128.0

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

# === Step 1: Load RGB, depth, and segmentation masks ===
rgb = cv2.imread(rgb_path)
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
depth = rgb_image_to_depth(depth_rgb_path)
alpha = np.array(Image.open(seg_path).convert("RGBA"))[..., 3]
alpha_2 = np.array(Image.open(seg_path_2).convert("RGBA"))[..., 3]
mask = (alpha > 128)
mask_2 = (alpha_2 > 128)

# === Step 2: Construct scene and masked point clouds ===
h, w = depth.shape
i, j = np.meshgrid(np.arange(w), np.arange(h))
z = depth
x = (i - cx) * z / fx
y = (j - cy) * z / fy
points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
colors = rgb.reshape(-1, 3).astype(np.float32) / 255.0
depth_valid = (z > 0).reshape(-1)
mask_flat = mask.reshape(-1)
mask_flat_2 = mask_2.reshape(-1)

scene_pcd = o3d.geometry.PointCloud()
scene_pcd.points = o3d.utility.Vector3dVector(points[depth_valid])
scene_pcd.colors = o3d.utility.Vector3dVector(colors[depth_valid])

anchor_pcd = o3d.geometry.PointCloud()
anchor_pcd.points = o3d.utility.Vector3dVector(points[depth_valid & mask_flat])
anchor_pcd.colors = o3d.utility.Vector3dVector(colors[depth_valid & mask_flat])

# Remove scene points within anchor masks to avoid duplication
remaining_mask = depth_valid & (~mask_flat) & (~mask_flat_2)
scene_pcd_filtered = o3d.geometry.PointCloud()
scene_pcd_filtered.points = o3d.utility.Vector3dVector(points[remaining_mask])
scene_pcd_filtered.colors = o3d.utility.Vector3dVector(colors[remaining_mask])

# === Step 3: Load TripoSR point cloud ===
object_pcd = o3d.io.read_point_cloud(object_pcd_path)

# === Step 4: Scale alignment using oriented bounding box ===
anchor_obb = anchor_pcd.get_oriented_bounding_box()
anchor_extent = anchor_obb.extent
anchor_center = anchor_obb.center

object_aabb = object_pcd.get_axis_aligned_bounding_box()
object_extent = object_aabb.get_extent()
object_center = object_aabb.get_center()

object_pcd.translate(-object_center)
scale_ratio = np.min(anchor_extent / (object_extent + 1e-8))
scale_ratio *= scale_2
object_pcd.scale(scale_ratio, center=(0, 0, 0))

# === Step 5: Apply FoundationPose transform ===
T = np.loadtxt(pose_path)
R, t = T[:3, :3], T[:3, 3]
points = np.asarray(object_pcd.points)
points_transformed = (R @ points.T + t.reshape(3, 1)).T
object_pcd.points = o3d.utility.Vector3dVector(points_transformed)

# === Step 6: Fuse point clouds ===
output_path_1 = output_path.replace(".ply", "_filtered.ply")
output_path_2 = output_path.replace(".ply", "_full.ply")
fused_1 = scene_pcd_filtered + object_pcd
fused_2 = scene_pcd + object_pcd
o3d.io.write_point_cloud(output_path_1, fused_1)
o3d.io.write_point_cloud(output_path_2, fused_2)
