"""
Point cloud patch extraction using Euclidean distance.
Extracts overlapping patches from point clouds for training/inference.
"""

import numpy as np
from typing import Tuple, Optional


def extract_patches_euclidean(
    points: np.ndarray,
    num_patches: int = 1000,
    points_per_patch: int = 1024,
    eps: Optional[float] = None,
    min_points: int = 100,
    max_attempts: int = 10000,
    allow_overlap: bool = True,
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract patches from point cloud using Euclidean distance.
    
    Args:
        points: Point cloud array of shape (N, 3)
        num_patches: Number of patches to extract
        points_per_patch: Number of points per patch
        eps: Neighborhood radius. If None, will be estimated automatically
        min_points: Minimum points required within eps to create a patch
        max_attempts: Maximum attempts to find valid patches
        allow_overlap: Whether to allow overlapping patches
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (patches, patch_centers) where:
        - patches: Array of shape (num_patches, points_per_patch, 3)
        - patch_centers: Array of shape (num_patches, 3) - center of each patch
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if len(points) < points_per_patch:
        raise ValueError(f"Point cloud has {len(points)} points, but need at least {points_per_patch} points per patch")
    
    # Estimate eps if not provided (use a fraction of the point cloud extent)
    if eps is None:
        point_ranges = points.max(axis=0) - points.min(axis=0)
        # Use 2% of the average range as default eps
        eps = np.mean(point_ranges) * 0.02
    
    patches = []
    patch_centers = []
    used_indices = set() if not allow_overlap else None
    
    attempts = 0
    while len(patches) < num_patches and attempts < max_attempts:
        attempts += 1
        
        # Randomly sample a seed point
        if allow_overlap or used_indices is None:
            seed_idx = np.random.randint(0, len(points))
        else:
            # Try to find an unused point
            available_indices = [i for i in range(len(points)) if i not in used_indices]
            if len(available_indices) == 0:
                # If we've used all points, allow overlap
                seed_idx = np.random.randint(0, len(points))
            else:
                seed_idx = np.random.choice(available_indices)
        
        seed_point = points[seed_idx]
        
        # Find points within eps Euclidean distance of seed point
        distances = np.linalg.norm(points - seed_point, axis=1)
        nearby_mask = distances <= eps
        nearby_points = points[nearby_mask]
        
        if len(nearby_points) < min_points:
            continue
        
        # Use all nearby points as the patch
        patch_points = nearby_points.copy()
        
        if len(patch_points) < points_per_patch:
            # If patch is too small, duplicate points to reach desired size
            num_needed = points_per_patch - len(patch_points)
            duplicate_indices = np.random.choice(len(patch_points), num_needed, replace=True)
            patch_points = np.vstack([patch_points, patch_points[duplicate_indices]])
        elif len(patch_points) > points_per_patch:
            # If patch is too large, randomly sample points_per_patch points
            sample_indices = np.random.choice(len(patch_points), points_per_patch, replace=False)
            patch_points = patch_points[sample_indices]
        
        # Center the patch at origin (useful for some models)
        patch_center = patch_points.mean(axis=0)
        centered_patch = patch_points - patch_center
        
        patches.append(centered_patch)
        patch_centers.append(patch_center)
        
        if not allow_overlap and used_indices is not None:
            # Mark points in this patch as used
            nearby_indices = np.where(nearby_mask)[0]
            used_indices.update(nearby_indices.tolist())
    
    if len(patches) < num_patches:
        print(f"Warning: Only extracted {len(patches)} patches out of {num_patches} requested")
    
    patches_array = np.array(patches, dtype=np.float32)
    patch_centers_array = np.array(patch_centers, dtype=np.float32)
    
    return patches_array, patch_centers_array


def extract_patches_euclidean_with_coverage(
    points: np.ndarray,
    num_patches: int = 1000,
    points_per_patch: int = 1024,
    eps: Optional[float] = None,
    min_points: int = 100,
    max_attempts: int = 50000,
    random_seed: Optional[int] = None,
    coverage_priority: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract patches ensuring every point is covered by at least one patch.
    Uses a coverage-aware strategy: prioritizes seed points near uncovered points.
    
    Args:
        points: Point cloud array of shape (N, 3)
        num_patches: Number of patches to extract (minimum, may extract more for coverage)
        points_per_patch: Number of points per patch
        eps: Neighborhood radius. If None, will be estimated automatically
        min_points: Minimum points required within eps to create a patch
        max_attempts: Maximum attempts to find valid patches
        random_seed: Random seed for reproducibility
        coverage_priority: Probability (0-1) of selecting seed near uncovered points vs random.
                           Higher = more focused on coverage, lower = more random
        
    Returns:
        Tuple of (patches, patch_centers, coverage_mask) where:
        - patches: Array of shape (num_patches, points_per_patch, 3)
        - patch_centers: Array of shape (num_patches, 3) - center of each patch
        - coverage_mask: Boolean array of shape (N,) indicating which points are covered
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if len(points) < points_per_patch:
        raise ValueError(f"Point cloud has {len(points)} points, but need at least {points_per_patch} points per patch")
    
    # Estimate eps if not provided
    if eps is None:
        point_ranges = points.max(axis=0) - points.min(axis=0)
        eps = np.mean(point_ranges) * 0.02
    
    patches = []
    patch_centers = []
    covered_mask = np.zeros(len(points), dtype=bool)  # Track which points are covered
    
    attempts = 0
    consecutive_failures = 0
    max_consecutive_failures = 1000
    
    while (len(patches) < num_patches or not np.all(covered_mask)) and attempts < max_attempts:
        attempts += 1
        
        # Strategy: prioritize uncovered points, but allow some randomness
        if np.random.random() < coverage_priority and not np.all(covered_mask):
            # Find uncovered points
            uncovered_indices = np.where(~covered_mask)[0]
            if len(uncovered_indices) > 0:
                # Pick a random uncovered point as seed
                seed_idx = np.random.choice(uncovered_indices)
            else:
                # All covered, pick random
                seed_idx = np.random.randint(0, len(points))
        else:
            # Random selection
            seed_idx = np.random.randint(0, len(points))
        
        seed_point = points[seed_idx]
        
        # Find points within eps Euclidean distance
        distances = np.linalg.norm(points - seed_point, axis=1)
        nearby_mask = distances <= eps
        nearby_points = points[nearby_mask]
        
        if len(nearby_points) < min_points:
            consecutive_failures += 1
            if consecutive_failures > max_consecutive_failures:
                # If we can't find valid patches, break
                break
            continue
        
        consecutive_failures = 0
        
        # Use all nearby points as the patch
        patch_points = nearby_points.copy()
        
        if len(patch_points) < points_per_patch:
            num_needed = points_per_patch - len(patch_points)
            duplicate_indices = np.random.choice(len(patch_points), num_needed, replace=True)
            patch_points = np.vstack([patch_points, patch_points[duplicate_indices]])
        elif len(patch_points) > points_per_patch:
            sample_indices = np.random.choice(len(patch_points), points_per_patch, replace=False)
            patch_points = patch_points[sample_indices]
        
        # Center the patch at origin
        patch_center = patch_points.mean(axis=0)
        centered_patch = patch_points - patch_center
        
        patches.append(centered_patch)
        patch_centers.append(patch_center)
        
        # Mark all nearby points as covered
        covered_mask[nearby_mask] = True
        
        # Progress update
        if len(patches) % 100 == 0:
            coverage_pct = np.sum(covered_mask) / len(points) * 100
            print(f"Extracted {len(patches)} patches, coverage: {coverage_pct:.1f}%")
    
    coverage_pct = np.sum(covered_mask) / len(points) * 100
    if coverage_pct < 100.0:
        print(f"Warning: Only achieved {coverage_pct:.1f}% coverage after {len(patches)} patches")
    else:
        print(f"Successfully covered all points with {len(patches)} patches")
    
    patches_array = np.array(patches, dtype=np.float32)
    patch_centers_array = np.array(patch_centers, dtype=np.float32)
    
    return patches_array, patch_centers_array, covered_mask


def extract_patches_from_npz(
    npz_path: str,
    num_patches: int = 1000,
    points_per_patch: int = 1024,
    eps: Optional[float] = None,
    sample_idx: int = 0,
    ensure_coverage: bool = False,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Extract patches from an .npz file.
    
    Args:
        npz_path: Path to .npz file
        num_patches: Number of patches to extract (minimum if ensure_coverage=True)
        points_per_patch: Number of points per patch
        eps: Euclidean distance neighborhood radius
        sample_idx: Which sample to use from the dataset
        ensure_coverage: If True, ensures every point is covered by at least one patch
        **kwargs: Additional arguments passed to patch extraction function
        
    Returns:
        Tuple of (patches, patch_centers, labels) where labels may be None
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
    
    # Extract patches
    # Filter out ensure_coverage from kwargs as it's not a parameter for the extraction functions
    filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'ensure_coverage'}
    
    if ensure_coverage:
        patches, patch_centers, coverage_mask = extract_patches_euclidean_with_coverage(
            points,
            num_patches=num_patches,
            points_per_patch=points_per_patch,
            eps=eps,
            **filtered_kwargs
        )
    else:
        patches, patch_centers = extract_patches_euclidean(
            points,
            num_patches=num_patches,
            points_per_patch=points_per_patch,
            eps=eps,
            **filtered_kwargs
        )
    
    # Extract corresponding labels if available
    patch_labels = None
    if labels is not None:
        sample_labels = labels[sample_idx]
        patch_labels = []
        
        for center in patch_centers:
            # Find the closest point to the patch center
            distances = np.linalg.norm(points - center, axis=1)
            closest_idx = np.argmin(distances)
            patch_labels.append(sample_labels[closest_idx])
        
        patch_labels = np.array(patch_labels, dtype=np.int64)
    
    return patches, patch_centers, patch_labels

