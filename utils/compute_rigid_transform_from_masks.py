import numpy as np
import cv2
import argparse
import open3d as o3d
import os

def save_pcd(filename, points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Saved point cloud: {filename}")

def load_mask_points(mask_path, pcd_np):
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    assert mask.shape[2] == 4, "Mask must have alpha channel"
    alpha = mask[:, :, 3] > 0

    non_empty = ~(pcd_np == 0).all(axis=2)
    valid_mask = alpha & non_empty
    points = pcd_np[valid_mask]
    return points

def load_extrinsic_matrix(txt_path):
    with open(txt_path, "r") as f:
        lines = f.readlines()
    matrix = np.array([[float(num) for num in line.strip().split()] for line in lines])
    return matrix

def apply_transform(points, transform):
    R = transform[:3, :3]
    t = transform[:3, 3]
    return points @ R.T + t

def estimate_rigid_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Estimate rigid transformation between two point sets using the Kabsch algorithm.
    The transformation aligns src to dst, centered at src's mean.
    """
    assert src.shape[1] == 3 and dst.shape[1] == 3
    min_pts = min(len(src), len(dst))
    if len(src) != len(dst):
        idx_src = np.random.choice(len(src), min_pts, replace=False)
        idx_dst = np.random.choice(len(dst), min_pts, replace=False)
        src = src[idx_src]
        dst = dst[idx_dst]

    c_src = np.mean(src, axis=0)
    src_centered = src - c_src
    dst_centered = dst - c_src

    H = src_centered.T @ dst_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = np.mean(dst_centered - (R @ src_centered.T).T, axis=0)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mask2", help="Path to rgb_segmented_1024_2.png")
    parser.add_argument("mask3", help="Path to rgb_segmented_1024_3.png")
    parser.add_argument("dense_pcd", help="Path to fused_scene_full_dense_point_image.npy")
    parser.add_argument("extrinsic", help="Path to extrinsic.txt")
    parser.add_argument("save_path", help="Output .txt path to save transform matrix")
    args = parser.parse_args()

    dense_pcd = np.load(args.dense_pcd)
    extrinsic = load_extrinsic_matrix(args.extrinsic)

    pts2_cam = load_mask_points(args.mask2, dense_pcd)
    pts3_cam = load_mask_points(args.mask3, dense_pcd)

    if len(pts2_cam) < 10 or len(pts3_cam) < 10:
        raise ValueError("Too few points to estimate transform.")

    pts2_world = apply_transform(pts2_cam, extrinsic)
    pts3_world = apply_transform(pts3_cam, extrinsic)

    transform = estimate_rigid_transform(pts2_world, pts3_world)

    np.savetxt(args.save_path, transform, fmt="%.6f")
    print(f"Rigid transformation matrix saved to: {args.save_path}")
