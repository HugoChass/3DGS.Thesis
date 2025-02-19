import os
import shutil
import subprocess
import numpy as np
import cv2
from nuscenes.nuscenes import NuScenes

# Define paths
NUSCENES_ROOT = "./data/NuScenesMini"  # Path to dataset
OUTPUT_DIR = "./colmap_workspace"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
DATABASE_PATH = os.path.join(OUTPUT_DIR, "database.db")
SPARSE_DIR = os.path.join(OUTPUT_DIR, "sparse")
DENSE_DIR = os.path.join(OUTPUT_DIR, "dense")
COLMAP_DIR = "C:\ProgramData\COLMAP\COLMAP.bat"

# Load NuScenes dataset
nusc = NuScenes(version='v1.0-mini', dataroot=NUSCENES_ROOT, verbose=True)

# Select a scene
scene = nusc.scene[0]  # Choose the first scene
print(scene)
sample_token = scene['first_sample_token']
images = []
while sample_token:
    sample = nusc.get('sample', sample_token)
    cam_front = nusc.get('sample_data', sample['data']['CAM_FRONT'])
    images.append(os.path.join(NUSCENES_ROOT, cam_front['filename']))
    sample_token = sample['next'] if sample['next'] else None

# Ensure COLMAP workspace exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Create an empty COLMAP database file
if not os.path.exists(DATABASE_PATH):
    open(DATABASE_PATH, 'w').close()

for img_path in images:
    shutil.copy(img_path, IMAGES_DIR)

# Run COLMAP pipeline
result = subprocess.run([COLMAP_DIR, "feature_extractor", "--database_path", DATABASE_PATH, "--image_path", IMAGES_DIR, "--SiftExtraction.max_num_features", "5000"])
if result.returncode != 0:
    raise RuntimeError("COLMAP feature extraction failed!")
subprocess.run([
    COLMAP_DIR, "sequential_matcher",
    "--database_path", DATABASE_PATH,
    "--SiftMatching.use_gpu", "0"
])
if result.returncode != 0:
    raise RuntimeError("COLMAP sequential matching failed!")
os.makedirs(SPARSE_DIR, exist_ok=True)
result = subprocess.run([
    COLMAP_DIR, "mapper",
    "--database_path", DATABASE_PATH,
    "--image_path", IMAGES_DIR,
    "--output_path", SPARSE_DIR,
    "--Mapper.init_min_num_inliers", "5",  # Reduce minimum inliers for initialization
    "--Mapper.abs_pose_min_num_inliers", "3",  # Allow fewer inliers for absolute pose estimation
    "--Mapper.min_model_size", "1",  # Try creating even a small model
    "--Mapper.num_threads", "8",
])

if result.returncode != 0:
    raise RuntimeError("COLMAP mapping failed!")

# Find the latest sparse subdirectory
sparse_subdirs = [d for d in os.listdir(SPARSE_DIR) if d.isdigit()]
if not sparse_subdirs:
    raise RuntimeError("No valid sparse reconstruction found!")
latest_sparse_dir = os.path.join(SPARSE_DIR, max(sparse_subdirs, key=int))

# Check if sparse reconstruction exists
required_files = ["cameras.bin", "images.bin", "points3D.bin"]
sparse_files = [os.path.join(latest_sparse_dir, file) for file in required_files]

if not all(os.path.exists(file) for file in sparse_files):
    raise RuntimeError("COLMAP sparse reconstruction failed! No cameras, images, or points3D files found.")

# Convert to dense point cloud
os.makedirs(DENSE_DIR, exist_ok=True)
result = subprocess.run([COLMAP_DIR, "image_undistorter", "--image_path", IMAGES_DIR, "--input_path", latest_sparse_dir, "--output_path", DENSE_DIR])
if result.returncode != 0:
    raise RuntimeError("COLMAP image undistortion failed!")

# Verify undistorted images exist before running dense stereo
if not os.path.exists(os.path.join(DENSE_DIR, "images")) or not os.listdir(os.path.join(DENSE_DIR, "images")):
    raise RuntimeError("Error: Undistorted images are missing. Ensure image_undistorter was run successfully.")

# Verify stereo input files exist
stereo_dir = os.path.join(DENSE_DIR, "stereo")
if not os.path.exists(stereo_dir) or not os.listdir(stereo_dir):
    raise RuntimeError("Error: Dense stereo inputs are missing. Ensure image_undistorter completed successfully.")

result = subprocess.run([
    COLMAP_DIR, "patch_match_stereo",
    "--workspace_path", DENSE_DIR,
    "--PatchMatchStereo.use_gpu", "0",
    "--PatchMatchStereo.max_image_size", "2000",  # Limit image size to avoid excessive memory usage
    "--PatchMatchStereo.num_threads", "4"  # Use fewer threads to prevent system overload
])
if result.returncode != 0:
    raise RuntimeError("COLMAP dense stereo failed!")

subprocess.run([COLMAP_DIR, "stereo_fusion", "--workspace_path", DENSE_DIR, "--output_path", os.path.join(DENSE_DIR, "fused.ply")])

print("Reconstruction complete! The point cloud is saved at:", os.path.join(DENSE_DIR, "fused.ply"))
