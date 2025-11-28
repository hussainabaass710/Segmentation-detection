"""
Utility functions for pointcloud segmentation.
"""

import os
import logging
import torch


def setup_logging(log_dir='./logs'):
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory to save log files
        
    Returns:
        Logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir='./checkpoints'):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss value
        checkpoint_dir: Directory to save checkpoints
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f'Checkpoint saved to {checkpoint_path}')


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        
    Returns:
        Dictionary with checkpoint information
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'epoch': checkpoint['epoch'],
        'loss': checkpoint['loss']
    }


def normalize_pointcloud(pointcloud):
    """
    Normalize pointcloud to unit sphere.
    
    Args:
        pointcloud: Pointcloud array of shape (N, 3)
        
    Returns:
        Normalized pointcloud
    """
    centroid = pointcloud.mean(axis=0)
    pointcloud = pointcloud - centroid
    max_dist = (pointcloud ** 2).sum(axis=1).max() ** 0.5
    pointcloud = pointcloud / max_dist
    return pointcloud

