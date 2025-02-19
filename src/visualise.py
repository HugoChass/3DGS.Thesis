import open3d as o3d
import os

import open3d as o3d

pcd = o3d.io.read_point_cloud("./colmap_workspace/dense/fused.ply")
print(pcd)


# # Define path to fused.ply
# PLY_FILE = "./colmap_workspace/dense/fused.ply"

# # Check if the file exists
# if not os.path.exists(PLY_FILE):
#     raise FileNotFoundError(f"Point cloud file not found: {PLY_FILE}")

# # Load the point cloud
# def visualize_point_cloud(ply_file):
#     pcd = o3d.io.read_point_cloud(ply_file)
#     print(pcd)  # Print basic info
#     o3d.visualization.draw_geometries([pcd], window_name="3D Reconstruction Viewer")

# # Run visualization
# visualize_point_cloud(PLY_FILE)