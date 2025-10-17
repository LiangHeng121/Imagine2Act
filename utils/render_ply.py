import sys
import numpy as np
import open3d as o3d
import cv2

def render_pointcloud(ply_file, output_image_path, img_size=(256, 256)):
    width, height = img_size

    # === Camera intrinsic matrix K ===
    K = np.array([
        [-351.6771208, 0, 128],
        [0, -351.6771208, 128],
        [0, 0, 1]
    ])

    # === Load point cloud ===
    pcd = o3d.io.read_point_cloud(ply_file)
    if len(pcd.points) == 0:
        raise ValueError(f"Empty point cloud: {ply_file}")

    pts = np.asarray(pcd.points)
    if pcd.has_colors():
        colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    else:
        colors = np.full((pts.shape[0], 3), 255, dtype=np.uint8)

    cam_pts = pts

    # Project to image plane
    x, y, z = cam_pts[:, 0], cam_pts[:, 1], cam_pts[:, 2]
    valid = z > 0
    x, y, z, colors = x[valid], y[valid], z[valid], colors[valid]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = (fx * x / z + cx).astype(int)
    v = (fy * y / z + cy).astype(int)

    # Initialize image
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Z-buffer rendering (nearest point wins)
    depth_buffer = np.full((height, width), np.inf)
    for i in range(len(u)):
        if 0 <= u[i] < width and 0 <= v[i] < height:
            if z[i] < depth_buffer[v[i], u[i]]:
                img[v[i], u[i]] = colors[i]
                depth_buffer[v[i], u[i]] = z[i]

    # Save rendered image
    cv2.imwrite(output_image_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Rendering complete: {output_image_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python utils/render_ply.py fused_scene.ply output.png")
        sys.exit(1)

    render_pointcloud(sys.argv[1], sys.argv[2])
