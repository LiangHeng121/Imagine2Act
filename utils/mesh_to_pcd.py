import sys
import trimesh
import open3d as o3d
import numpy as np

mesh_path = sys.argv[1]             # e.g. ./my_outputs/mesh.obj
output_ply_path = sys.argv[2]       # e.g. ./my_outputs/phone.ply

mesh = trimesh.load(mesh_path, process=False)
points, face_indices = trimesh.sample.sample_surface(mesh, 300000)

if hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
    face_colors = mesh.visual.face_colors
    sampled_colors = face_colors[face_indices, :3] / 255.0
else:
    sampled_colors = np.ones_like(points)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(sampled_colors)

o3d.io.write_point_cloud(output_ply_path, pcd)
print(f"Point cloud saved to: {output_ply_path}")
