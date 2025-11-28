"""
Script to extract patches from point cloud data.
Modify num_patches here to change how many patches are extracted.
"""

from pointcloud_segmentation.download_and_process import extract_and_save_patches_ball_query

# Configuration
npz_path = 'data/ODMSemantic3D/datasets/odm_data_waterbury-roads_2.npz'
output_path = 'data/ODMSemantic3D/patches_waterbury_roads_2.npz'

# MODIFY THIS to change number of patches
NUM_PATCHES = 1000  # ← Change this value!

print(f"Extracting {NUM_PATCHES} patches...")
print("This may take a while depending on the number of patches and point cloud size.")

extract_and_save_patches_ball_query(
    npz_path=npz_path,
    output_path=output_path,
    num_patches=NUM_PATCHES,  # Number of patches to extract
    points_per_patch=4096,    # Exact number of points per patch
    radius_percent=0.02,       # 2% of max(x_range, y_range)
    sample_idx=0,
    random_seed=42,
    device="cpu"  # Use "cuda" if you have GPU
)

print(f"\n✓ Successfully extracted and saved {NUM_PATCHES} patches to {output_path}")
print(f"You can now use these patches for training!")

