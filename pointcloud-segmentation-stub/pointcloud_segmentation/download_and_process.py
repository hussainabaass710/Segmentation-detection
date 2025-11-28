# pointcloud_segmentation/dataset.py

import os
import glob
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
import laspy
from laspy.compression import LazBackend
from torch.utils.data import Dataset

# -----------------------------
# Configuration
# -----------------------------
REPO_URL = "https://github.com/OpenDroneMap/ODMSemantic3D.git"
DATA_DIR = Path("data") / "ODMSemantic3D"
REPO_DIR = Path("data") / "ODMSemantic3D_repo"
LICENSE_FILE = DATA_DIR / "ODMSemantic3D_LICENSE.txt"
OUT_NPZ = DATA_DIR / "pointclouds.npz"

# Point-cloud file extensions (adjust if needed)
POINTCLOUD_EXTS = [".laz", ".ply", ".txt"]


# -----------------------------
# Helper functions
# -----------------------------
def ensure_dir(path: Path):
    if not path.exists():
        path.mkdir(parents=True)
        print(f"[INFO] Created directory: {path}")


def clone_repo(repo_url: str, dest: Path):
    if dest.exists():
        print(f"[INFO] Repo already exists at {dest}, skipping clone.")
        return True

    try:
        print(f"[INFO] Cloning ODMSemantic3D repo (shallow) from {repo_url} ...")
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(dest)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print("[INFO] Clone complete.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Git clone failed: {e}")
        return False


def copy_pointcloud_files(src_dir: Path, dst_dir: Path):
    if not src_dir.exists():
        print(f"[ERROR] Source repo directory {src_dir} does not exist.")
        return

    count = 0
    for root, dirs, files in os.walk(src_dir):
        for f in files:
            if any(f.lower().endswith(ext) for ext in POINTCLOUD_EXTS):
                src_path = Path(root) / f
                rel_path = src_path.relative_to(src_dir)
                dst_path = dst_dir / rel_path
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
                count += 1

    print(f"[INFO] Copied {count} point-cloud files to {dst_dir}")


def convert_laz_to_npz(data_dir: Path, output_dir: Optional[Path] = None, filename_filter: Optional[str] = None):
    """
    Convert .laz files to individual .npz files (one .npz per .laz file).
    
    Args:
        data_dir: Directory containing .laz files
        output_dir: Directory to save .npz files (defaults to data_dir)
        filename_filter: Optional filename to filter (e.g., "drone_dataset_sheffield_park_1.laz")
                        If None, processes all .laz files
    """
    if output_dir is None:
        output_dir = data_dir
    ensure_dir(output_dir)
    
    # Find .laz files (recursively)
    all_laz_files = list(data_dir.rglob("*.laz"))
    
    # Filter by filename if specified
    if filename_filter:
        laz_files = [f for f in all_laz_files if filename_filter in f.name]
        if not laz_files:
            print(f"[WARNING] No .laz files found matching '{filename_filter}' in {data_dir}")
            print(f"[INFO] Available files: {[f.name for f in all_laz_files[:5]]}...")
            return
        print(f"[INFO] Filtering to files containing '{filename_filter}'")
    else:
        laz_files = all_laz_files
    
    if not laz_files:
        print(f"[WARNING] No .laz files found in {data_dir}")
        return
    
    print(f"[INFO] Found {len(laz_files)} .laz file(s). Converting to NPZ format...")
    
    successful_conversions = 0
    
    for fpath in laz_files:
        try:
            print(f"[INFO] Reading {fpath.name}...")
            # Use laspy.open() with explicit LazBackend to avoid version compatibility issues
            with laspy.open(fpath, laz_backend=LazBackend.Lazrs) as las_file:
                las = las_file.read()
            
            # Extract x, y, z coordinates - shape: (N, 3)
            points = np.vstack((las.x, las.y, las.z)).T
            
            # Extract labels from classification field if available
            if hasattr(las, 'classification'):
                # Convert SubFieldView to numpy array first, then to int64
                labels = np.array(las.classification, dtype=np.int64)
            else:
                # If no classification, use zeros
                labels = np.zeros(points.shape[0], dtype=np.int64)
                print(f"[WARNING] No classification field found in {fpath.name}, using zeros")
            
            # Create output path: preserve relative directory structure
            rel_path = fpath.relative_to(data_dir)
            out_npz = output_dir / rel_path.with_suffix('.npz')
            out_npz.parent.mkdir(parents=True, exist_ok=True)
            
            # Skip if file already exists
            if out_npz.exists():
                print(f"[INFO] Output file already exists at {out_npz}, skipping...")
                continue
            
            # Save to compressed NPZ format
            # Store as (1, N, 3) and (1, N) to maintain compatibility with Dataset class
            np.savez_compressed(
                out_npz,
                pointclouds=np.array([points], dtype=np.float32),
                labels=np.array([labels], dtype=np.int64)
            )
            
            print(f"[INFO] Saved {fpath.name} → {out_npz} (shape: {points.shape})")
            successful_conversions += 1
            
        except Exception as e:
            print(f"[ERROR] Failed to read {fpath}: {e}")
            continue
    
    print(f"[INFO] Successfully converted {successful_conversions} out of {len(laz_files)} .laz file(s)")


# -----------------------------
# Ball Query Patch Extraction
# -----------------------------
def ball_query(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    Ball query operation to find points within a radius around query centers.
    Optimized to process one query at a time to avoid memory issues with large point clouds.
    
    Args:
        radius: Threshold distance (radius)
        nsample: Exact number of points in each patch
        xyz: [B, N, 3] original points
        new_xyz: [B, M, 3] query centers
        
    Returns:
        group_idx: [B, M, K] sampled indices where K = nsample
    """
    B, N, _ = xyz.shape
    _, M, _ = new_xyz.shape

    group_idx = torch.zeros((B, M, nsample), dtype=torch.long).to(xyz.device)

    # Process one query center at a time to avoid memory issues
    for b in range(B):
        for m in range(M):
            query_center = new_xyz[b, m:m+1, :]  # [1, 3]
            # Compute distances only for this query center
            d = torch.cdist(query_center, xyz[b]).squeeze(0)  # [N]
            
            # Get points within radius
            idx = torch.where(d < radius)[0]

            if len(idx) == 0:
                # If no points in radius, use the closest point to query center
                closest_idx = torch.argmin(d)
                idx = closest_idx.unsqueeze(0)
            
            if len(idx) >= nsample:
                # Randomly sample exactly nsample points if we have more
                perm = torch.randperm(len(idx), device=xyz.device)[:nsample]
                idx = idx[perm]
            else:
                # Pad to exactly nsample by randomly sampling with replacement
                num_needed = nsample - len(idx)
                if len(idx) > 0:
                    pad_indices = torch.randint(0, len(idx), (num_needed,), device=xyz.device)
                    idx = torch.cat([idx, idx[pad_indices]])
                else:
                    # Edge case: if idx is empty (shouldn't happen after check above, but just in case)
                    closest_idx = torch.argmin(d)
                    idx = closest_idx.repeat(nsample)

            group_idx[b, m] = idx

    return group_idx


def extract_patches_ball_query(
    points: np.ndarray,
    labels: Optional[np.ndarray] = None,
    num_patches: int = 1000,
    points_per_patch: int = 4096,
    radius: Optional[float] = None,
    radius_percent: float = 0.02,
    random_seed: Optional[int] = None,
    device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Extract patches from point cloud using ball query (similar to PointNet++).
    
    This function extracts fixed-size patches from a point cloud using ball query,
    which finds all points within a specified radius around query centers.
    Each patch will have exactly points_per_patch points.
    
    Args:
        points: Point cloud array of shape (N, 3)
        labels: Optional per-point labels array of shape (N,)
        num_patches: Number of patches to extract
        points_per_patch: Exact number of points per patch (default: 4096)
        radius: Neighborhood radius. If None, will be calculated as radius_percent 
                of the maximum of x or y range
        radius_percent: Percentage of max(x_range, y_range) to use as radius (default: 0.02 = 2%)
        random_seed: Random seed for reproducibility
        device: Device to use for computation ('cpu' or 'cuda')
        
    Returns:
        Tuple of (patches, patch_centers, patch_labels) where:
        - patches: Array of shape (num_patches, points_per_patch, 3) - ready for PyTorch
        - patch_centers: Array of shape (num_patches, 3) - center of each patch
        - patch_labels: Array of shape (num_patches, points_per_patch) - per-point labels for each patch, or None
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    if len(points) < points_per_patch:
        raise ValueError(f"Point cloud has {len(points)} points, but need at least {points_per_patch} points per patch")
    
    # Calculate radius if not provided
    if radius is None:
        x_range = points[:, 0].max() - points[:, 0].min()
        y_range = points[:, 1].max() - points[:, 1].min()
        max_range = max(x_range, y_range)
        radius = max_range * radius_percent
    
    # Convert to PyTorch tensors
    xyz = torch.from_numpy(points).float().unsqueeze(0).to(device)  # [1, N, 3]
    
    # Select query centers (seed points) - randomly sample some points as centers
    query_indices = np.random.choice(len(points), num_patches, replace=False)
    query_centers = points[query_indices]
    new_xyz = torch.from_numpy(query_centers).float().unsqueeze(0).to(device)  # [1, M, 3]
    
    # Perform ball query with progress indicator
    print(f"[INFO] Performing ball query for {num_patches} patches...")
    print(f"[INFO] Point cloud size: {len(points):,} points")
    print(f"[INFO] Radius: {radius:.2f}")
    group_idx = ball_query(radius, points_per_patch, xyz, new_xyz)  # [1, M, K]
    
    # Extract patches with progress indicator
    patches = []
    patch_labels_list = []
    
    print(f"[INFO] Extracting patches...")
    for m in range(num_patches):
        if (m + 1) % max(1, num_patches // 10) == 0 or m == 0:
            print(f"[INFO] Progress: {m + 1}/{num_patches} patches extracted ({100 * (m + 1) / num_patches:.1f}%)")
        
        indices = group_idx[0, m].cpu().numpy()  # Get indices for this patch
        patch_points = points[indices]  # Extract patch points
        
        # Verify patch has exactly points_per_patch points
        assert len(patch_points) == points_per_patch, \
            f"Patch {m} has {len(patch_points)} points, expected {points_per_patch}"
        
        patches.append(patch_points)
        
        # Extract corresponding labels if available
        if labels is not None:
            patch_labels_list.append(labels[indices])
    
    patches_array = np.array(patches, dtype=np.float32)  # (num_patches, points_per_patch, 3)
    patch_centers_array = np.array(query_centers, dtype=np.float32)  # (num_patches, 3)
    
    if labels is not None:
        patch_labels_array = np.array(patch_labels_list, dtype=np.int64)  # (num_patches, points_per_patch)
    else:
        patch_labels_array = None
    
    return patches_array, patch_centers_array, patch_labels_array


def extract_patches_ball_query_from_npz(
    npz_path: str,
    num_patches: int = 1000,
    points_per_patch: int = 4096,
    radius: Optional[float] = None,
    radius_percent: float = 0.02,
    sample_idx: int = 0,
    random_seed: Optional[int] = None,
    device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Extract patches from an .npz file using ball query.
    
    Args:
        npz_path: Path to .npz file
        num_patches: Number of patches to extract
        points_per_patch: Exact number of points per patch (default: 4096)
        radius: Neighborhood radius. If None, will be calculated automatically
        radius_percent: Percentage of max(x_range, y_range) to use as radius (default: 0.02 = 2%)
        sample_idx: Which sample to use from the dataset
        random_seed: Random seed for reproducibility
        device: Device to use for computation ('cpu' or 'cuda')
        
    Returns:
        Tuple of (patches, patch_centers, patch_labels) where:
        - patches: Array of shape (num_patches, points_per_patch, 3) - ready for PyTorch
        - patch_centers: Array of shape (num_patches, 3) - center of each patch
        - patch_labels: Array of shape (num_patches, points_per_patch) - per-point labels, or None
    """
    data = np.load(npz_path)
    
    # Load pointclouds
    if 'pointclouds' in data:
        pointclouds = data['pointclouds']
    elif 'points' in data:
        pointclouds = data['points']
    else:
        raise ValueError(f"No pointclouds or points key found in {npz_path}")
    
    # Load labels if available
    labels = data.get('labels', None)
    
    # Get the specified sample
    points = pointclouds[sample_idx]
    sample_labels = labels[sample_idx] if labels is not None else None
    
    # Extract patches
    return extract_patches_ball_query(
        points=points,
        labels=sample_labels,
        num_patches=num_patches,
        points_per_patch=points_per_patch,
        radius=radius,
        radius_percent=radius_percent,
        random_seed=random_seed,
        device=device
    )


def save_patches_ball_query(
    patches: np.ndarray,
    patch_labels: Optional[np.ndarray],
    patch_centers: Optional[np.ndarray],
    output_path: str,
    metadata: Optional[dict] = None
):
    """
    Save ball query patches to an NPZ file for later loading.
    
    Args:
        patches: Array of shape (num_patches, points_per_patch, 3)
        patch_labels: Optional array of shape (num_patches, points_per_patch)
        patch_centers: Optional array of shape (num_patches, 3) - centers of patches
        output_path: Path to save the NPZ file
        metadata: Optional dictionary with additional metadata (e.g., radius, num_patches, etc.)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'patches': patches.astype(np.float32),
    }
    
    if patch_labels is not None:
        save_dict['patch_labels'] = patch_labels.astype(np.int64)
    
    if patch_centers is not None:
        save_dict['patch_centers'] = patch_centers.astype(np.float32)
    
    # Add metadata if provided
    if metadata is not None:
        for key, value in metadata.items():
            if isinstance(value, (int, float, str, bool)):
                save_dict[f'metadata_{key}'] = value
            elif isinstance(value, np.ndarray):
                save_dict[f'metadata_{key}'] = value
    
    np.savez_compressed(output_path, **save_dict)
    print(f"[INFO] Saved {len(patches)} patches to {output_path}")
    print(f"  - Patches shape: {patches.shape}")
    if patch_labels is not None:
        print(f"  - Labels shape: {patch_labels.shape}")
    if patch_centers is not None:
        print(f"  - Centers shape: {patch_centers.shape}")


def load_patches_ball_query(
    patches_path: str
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[dict]]:
    """
    Load ball query patches from an NPZ file.
    
    Args:
        patches_path: Path to the NPZ file containing patches
        
    Returns:
        Tuple of (patches, patch_labels, patch_centers, metadata) where:
        - patches: Array of shape (num_patches, points_per_patch, 3)
        - patch_labels: Array of shape (num_patches, points_per_patch) or None
        - patch_centers: Array of shape (num_patches, 3) or None
        - metadata: Dictionary with metadata or None
    """
    data = np.load(patches_path)
    
    if 'patches' not in data:
        raise KeyError(f"No 'patches' key found in {patches_path}")
    
    patches = data['patches']
    patch_labels = data.get('patch_labels', None)
    patch_centers = data.get('patch_centers', None)
    
    # Extract metadata
    metadata = {}
    for key in data.keys():
        if key.startswith('metadata_'):
            metadata_key = key.replace('metadata_', '')
            metadata[metadata_key] = data[key]
    
    print(f"[INFO] Loaded {len(patches)} patches from {patches_path}")
    print(f"  - Patches shape: {patches.shape}")
    if patch_labels is not None:
        print(f"  - Labels shape: {patch_labels.shape}")
    if patch_centers is not None:
        print(f"  - Centers shape: {patch_centers.shape}")
    
    metadata = metadata if metadata else None
    
    return patches, patch_labels, patch_centers, metadata


def extract_and_save_patches_ball_query(
    npz_path: str,
    output_path: str,
    num_patches: int = 1000,
    points_per_patch: int = 4096,
    radius: Optional[float] = None,
    radius_percent: float = 0.02,
    sample_idx: int = 0,
    random_seed: Optional[int] = None,
    device: str = "cpu"
):
    """
    Extract patches using ball query and save them to a file in one step.
    
    Args:
        npz_path: Path to input .npz file
        output_path: Path to save the extracted patches
        num_patches: Number of patches to extract
        points_per_patch: Exact number of points per patch (default: 4096)
        radius: Neighborhood radius. If None, will be calculated automatically
        radius_percent: Percentage of max(x_range, y_range) to use as radius (default: 0.02 = 2%)
        sample_idx: Which sample to use from the dataset
        random_seed: Random seed for reproducibility
        device: Device to use for computation ('cpu' or 'cuda')
    """
    # Extract patches
    patches, patch_centers, patch_labels = extract_patches_ball_query_from_npz(
        npz_path=npz_path,
        num_patches=num_patches,
        points_per_patch=points_per_patch,
        radius=radius,
        radius_percent=radius_percent,
        sample_idx=sample_idx,
        random_seed=random_seed,
        device=device
    )
    
    # Prepare metadata
    metadata = {
        'num_patches': num_patches,
        'points_per_patch': points_per_patch,
        'radius_percent': radius_percent,
        'sample_idx': sample_idx,
        'source_file': str(npz_path),
    }
    if radius is not None:
        metadata['radius'] = radius
    
    # Save patches
    save_patches_ball_query(
        patches=patches,
        patch_labels=patch_labels,
        patch_centers=patch_centers,
        output_path=output_path,
        metadata=metadata
    )


# -----------------------------
# PyTorch Dataset Class
# -----------------------------
class PointCloudDataset(Dataset):
    """Dataset class for pointcloud data loaded from .npz files."""
    
    def __init__(self, data_path: str, transform: Optional[callable] = None):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the .npz file containing pointcloud data
            transform: Optional transform to be applied to samples
        """
        self.data_path = data_path
        self.transform = transform
        self.data = np.load(data_path)
        
        # Load pointclouds and labels
        # Support both 'pointclouds' and 'points' keys for flexibility
        if 'pointclouds' in self.data:
            self.pointclouds = self.data['pointclouds']
        elif 'points' in self.data:
            self.pointclouds = self.data['points']
        else:
            raise KeyError(f"NPZ file must contain 'pointclouds' or 'points' key. Found: {list(self.data.keys())}")
        
        # Labels are optional
        if 'labels' in self.data:
            self.labels = self.data['labels']
        else:
            self.labels = None
            print(f"[WARNING] No 'labels' key found in {data_path}, labels will be None")
    
    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.pointclouds)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (pointcloud, label) where:
            - pointcloud: numpy array of shape (N, 3) or (3, N) depending on format
            - label: numpy array of shape (N,) or None if labels not available
        """
        pointcloud = self.pointclouds[idx].copy()
        label = self.labels[idx].copy() if self.labels is not None else None
        
        if self.transform:
            pointcloud = self.transform(pointcloud)
        
        return pointcloud, label


class BallQueryPatchDataset(Dataset):
    """
    Dataset class for ball query patches extracted from point clouds.
    Suitable for PointNet++ training.
    
    Can be initialized with either:
    1. Pre-extracted patches (numpy arrays)
    2. Path to a saved NPZ file containing patches
    """
    
    def __init__(
        self,
        patches: Optional[np.ndarray] = None,
        patch_labels: Optional[np.ndarray] = None,
        patches_path: Optional[str] = None,
        transform: Optional[callable] = None
    ):
        """
        Initialize the dataset with pre-extracted patches or load from file.
        
        Args:
            patches: Array of shape (num_patches, points_per_patch, 3) containing patches.
                    If None, patches_path must be provided.
            patch_labels: Optional array of shape (num_patches, points_per_patch) containing per-point labels.
                         Ignored if patches_path is provided.
            patches_path: Optional path to NPZ file containing saved patches. If provided, patches
                         and patch_labels will be loaded from this file.
            transform: Optional transform to be applied to patches
        """
        self.transform = transform
        
        if patches_path is not None:
            # Load from file
            patches, patch_labels, _, _ = load_patches_ball_query(patches_path)
            self.patches = patches
            self.patch_labels = patch_labels
        elif patches is not None:
            # Use provided patches
            self.patches = patches
            self.patch_labels = patch_labels
        else:
            raise ValueError("Either 'patches' or 'patches_path' must be provided")
        
        if len(self.patches.shape) != 3 or self.patches.shape[2] != 3:
            raise ValueError(f"Patches must be of shape (num_patches, points_per_patch, 3), got {self.patches.shape}")
        
        if self.patch_labels is not None and self.patch_labels.shape != self.patches.shape[:2]:
            raise ValueError(
                f"Patch labels shape {self.patch_labels.shape} must match patches shape {self.patches.shape[:2]}"
            )
        
        # Remap labels to be contiguous (0, 1, 2, ..., num_classes-1)
        self.label_map = None
        self.inverse_label_map = None
        if self.patch_labels is not None:
            unique_labels = np.unique(self.patch_labels)
            self.label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
            self.inverse_label_map = {new_label: old_label for old_label, new_label in self.label_map.items()}
            
            # Remap labels
            remapped_labels = np.zeros_like(self.patch_labels)
            for old_label, new_label in self.label_map.items():
                remapped_labels[self.patch_labels == old_label] = new_label
            self.patch_labels = remapped_labels
            
            print(f"[INFO] Remapped labels: {self.label_map}")
            print(f"[INFO] Original labels: {list(self.label_map.keys())} -> New labels: {list(self.label_map.values())}")
    
    def __len__(self) -> int:
        """Return the number of patches."""
        return len(self.patches)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get a patch from the dataset.
        
        Args:
            idx: Index of the patch
            
        Returns:
            Tuple of (patch, labels) where:
            - patch: torch.Tensor of shape (points_per_patch, 3) - ready for PointNet++
            - labels: torch.Tensor of shape (points_per_patch,) or None
        """
        patch = self.patches[idx].copy()  # (points_per_patch, 3)
        
        if self.transform:
            patch = self.transform(patch)
        
        # Convert to torch tensor
        patch_tensor = torch.from_numpy(patch).float()
        
        # Get labels if available
        if self.patch_labels is not None:
            labels_tensor = torch.from_numpy(self.patch_labels[idx]).long()
        else:
            labels_tensor = None
        
        return patch_tensor, labels_tensor


# -----------------------------
# Main
# -----------------------------
def main():
    ensure_dir(DATA_DIR)

    # Check if .laz files already exist in DATA_DIR
    existing_laz_files = list(DATA_DIR.rglob("*.laz"))
    if existing_laz_files:
        print(f"[INFO] Found {len(existing_laz_files)} existing .laz file(s) in {DATA_DIR}")
        print(f"[INFO] Skipping clone and copy steps. Using existing data.")
    else:
        # 1. Clone GitHub repo (only if data doesn't exist)
        success = clone_repo(REPO_URL, REPO_DIR)
        if not success:
            print("[ERROR] Cannot proceed without cloning the repo.")
            return

        # 2. Copy point-cloud files (only if data doesn't exist)
        copy_pointcloud_files(REPO_DIR, DATA_DIR)

    # 3. Convert .laz files to .npz format
    # Process all .laz files, creating one .npz file per .laz file
    convert_laz_to_npz(DATA_DIR, output_dir=DATA_DIR)

    print("[INFO] ODMSemantic3D download & preparation complete.")


if __name__ == "__main__":
    main()
