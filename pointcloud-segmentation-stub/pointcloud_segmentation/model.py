"""
Model module for pointcloud segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Helper Functions for PointNet++
# -----------------------------
def farthest_point_sample(xyz, npoint):
    """
    Farthest Point Sampling (FPS) for PointNet++.
    
    Args:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
        
    Returns:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B), farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def index_points(points, idx):
    """
    Index points using indices.
    
    Args:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
        
    Returns:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Ball query for PointNet++.
    
    Args:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
        
    Returns:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = torch.sum((new_xyz[:, :, None, :] - xyz[:, None, :, :]) ** 2, dim=-1)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


# -----------------------------
# PointNet++ Layers
# -----------------------------
class PointNetSetAbstraction(nn.Module):
    """
    PointNet Set Abstraction layer (used in PointNet++).
    """
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, N, 3]
            points: input points data, [B, N, C]
        Return:
            new_xyz: sampled points position data, [B, S, 3]
            new_points: sample points feature data, [B, S, D]
        """
        xyz = xyz.permute(0, 2, 1).contiguous()  # [B, 3, N]
        if points is not None:
            points = points.permute(0, 2, 1).contiguous()  # [B, C, N]

        if self.group_all:
            # In group_all, we process all points as one group
            # Create a single centroid point for output xyz
            xyz_flat = xyz.permute(0, 2, 1).contiguous()  # [B, N, 3]
            new_xyz = xyz_flat.mean(dim=1, keepdim=True)  # [B, 1, 3] - centroid
            
            # grouped_xyz: [B, 3, N, 1] where N is all points, 1 is the group size
            grouped_xyz = xyz.unsqueeze(3)  # [B, 3, N, 1]
            if points is not None:
                grouped_points = points.unsqueeze(3)  # [B, C, N, 1]
            else:
                grouped_points = grouped_xyz
            # For group_all, we don't normalize xyz (all points relative to origin)
            grouped_xyz_norm = grouped_xyz
        else:
            # Farthest Point Sampling
            xyz_flat = xyz.permute(0, 2, 1).contiguous()  # [B, N, 3]
            fps_idx = farthest_point_sample(xyz_flat, self.npoint)  # [B, npoint]
            new_xyz = index_points(xyz_flat, fps_idx)  # [B, npoint, 3]
            
            # Ball query grouping
            idx = query_ball_point(self.radius, self.nsample, xyz_flat, new_xyz)  # [B, npoint, nsample]
            grouped_xyz = index_points(xyz_flat, idx)  # [B, npoint, nsample, 3]
            grouped_xyz_norm = grouped_xyz - new_xyz.view(xyz_flat.shape[0], self.npoint, 1, 3)  # [B, npoint, nsample, 3]
            
            if points is not None:
                grouped_points = index_points(points.permute(0, 2, 1).contiguous(), idx)  # [B, npoint, nsample, C]
            else:
                grouped_points = grouped_xyz_norm

        if self.group_all:
            # For group_all: [B, channels, N, 1] -> [B, channels, 1, N]
            grouped_xyz_norm = grouped_xyz_norm.permute(0, 1, 3, 2)  # [B, 3, 1, N]
            if points is not None:
                grouped_points = grouped_points.permute(0, 1, 3, 2)  # [B, C, 1, N]
                grouped_points = torch.cat([grouped_xyz_norm, grouped_points], dim=1)  # [B, 3+C, 1, N]
            else:
                grouped_points = grouped_xyz_norm
        else:
            # For normal case: [B, channels, nsample, npoint]
            grouped_xyz_norm = grouped_xyz_norm.permute(0, 3, 2, 1)  # [B, 3, nsample, npoint]
            if points is not None:
                grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, C, nsample, npoint]
                grouped_points = torch.cat([grouped_xyz_norm, grouped_points], dim=1)  # [B, 3+C, nsample, npoint]
            else:
                grouped_points = grouped_xyz_norm

        # MLP
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            grouped_points = F.relu(bn(conv(grouped_points)))

        # Max pooling
        if self.group_all:
            # For group_all: [B, D, 1, N] -> max over N points -> [B, D, 1]
            new_points = torch.max(grouped_points, 3)[0]  # [B, D, 1]
        else:
            # For normal case: [B, D, nsample, npoint] -> max over nsample -> [B, D, npoint]
            new_points = torch.max(grouped_points, 2)[0]  # [B, D, npoint]
        new_points = new_points.permute(0, 2, 1)  # [B, npoint, D] or [B, 1, D]
        
        if new_xyz is not None:
            new_xyz = new_xyz  # [B, npoint, 3]

        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    """
    Feature Propagation layer for PointNet++ (upsampling).
    """
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N, 3] or [B, 3, N]
            xyz2: sampled input points position data, [B, S, 3] or [B, 3, S]
            points1: input points data, [B, N, D] or [B, D, N]
            points2: input points data, [B, S, D] or [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        # Handle different input formats - ensure all are [B, N, ...] format
        if xyz1 is not None:
            if xyz1.dim() == 3 and xyz1.shape[1] == 3:
                xyz1 = xyz1.permute(0, 2, 1).contiguous()  # [B, 3, N] -> [B, N, 3]
        if xyz2 is not None:
            if xyz2.dim() == 3 and xyz2.shape[1] == 3:
                xyz2 = xyz2.permute(0, 2, 1).contiguous()  # [B, 3, S] -> [B, S, 3]
        
        if xyz1 is None or xyz2 is None:
            raise ValueError("xyz1 and xyz2 cannot be None in Feature Propagation")
        
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape
        
        # Ensure points2 is in [B, S, D] format
        if points2 is not None:
            if points2.dim() == 3:
                # Check if it's [B, D, S] format (needs permute) or [B, S, D] format (already correct)
                if points2.shape[1] == S:
                    # Already in [B, S, D] format
                    pass
                elif points2.shape[2] == S:
                    # In [B, D, S] format, need to permute
                    points2 = points2.permute(0, 2, 1).contiguous()  # [B, D, S] -> [B, S, D]
                else:
                    raise ValueError(f"points2 shape {points2.shape} doesn't match xyz2 shape {xyz2.shape}")

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)  # [B, S, D] -> [B, N, D]
        else:
            dists = torch.sum((xyz1[:, :, None, :] - xyz2[:, None, :, :]) ** 2, dim=-1)  # [B, N, S]
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)  # [B, N, D]

        if points1 is not None:
            # Ensure points1 is in [B, N, D] format
            if points1.dim() == 3:
                if points1.shape[1] == N:
                    # Already in [B, N, D] format
                    pass
                elif points1.shape[2] == N:
                    # In [B, D, N] format, need to permute
                    points1 = points1.permute(0, 2, 1).contiguous()  # [B, D, N] -> [B, N, D]
                else:
                    raise ValueError(f"points1 shape {points1.shape} doesn't match xyz1 shape {xyz1.shape}")
            new_points = torch.cat([points1, interpolated_points], dim=-1)  # [B, N, D+D']
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1).contiguous()  # [B, D+D', N]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


# -----------------------------
# PointNet++ Model
# -----------------------------
class PointNetPlusPlusSeg(nn.Module):
    """
    PointNet++ model for point cloud segmentation.
    Full implementation with Set Abstraction and Feature Propagation layers.
    """
    def __init__(self, num_classes=10, num_points=4096, normal_channel=False):
        super(PointNetPlusPlusSeg, self).__init__()
        self.num_classes = num_classes
        self.num_points = num_points
        self.normal_channel = normal_channel
        
        in_channel = 6 if normal_channel else 3
        
        # Set Abstraction layers (encoder)
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, 
                                          in_channel=in_channel, mlp=[64, 64, 128])
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, 
                                          in_channel=128 + 3, mlp=[128, 128, 256])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, 
                                          in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        
        # Feature Propagation layers (decoder)
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128, mlp=[128, 128, 128])
        
        # Segmentation head
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)
    
    def forward(self, xyz):
        """
        Forward pass.
        
        Args:
            xyz: Input point cloud tensor of shape (B, N, 3) or (B, 3, N)
            
        Returns:
            Segmentation logits of shape (B, num_classes, N)
        """
        # Ensure input is (B, N, 3)
        if xyz.dim() == 3 and xyz.shape[1] == 3:
            xyz = xyz.permute(0, 2, 1)  # (B, 3, N) -> (B, N, 3)
        
        # Set Abstraction (encoder)
        # First layer: only xyz coordinates, no point features yet
        l0_xyz = xyz
        l0_points = None  # No features initially, just coordinates
        
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # Feature Propagation (decoder)
        # All inputs are in [B, N, 3] or [B, N, D] format
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        
        # Segmentation head
        feat = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(feat)
        
        return x

