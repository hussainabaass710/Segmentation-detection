"""
Training script for pointcloud segmentation using PointNet++ with ball query patches.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np

from pointcloud_segmentation.download_and_process import BallQueryPatchDataset
from pointcloud_segmentation.model import PointNetPlusPlusSeg
from pointcloud_segmentation.utils import setup_logging, save_checkpoint


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (patches, labels) in enumerate(dataloader):
        # Patches come as (B, N, 3) from BallQueryPatchDataset
        # Model expects (B, 3, N) or (B, N, 3) - it handles both
        patches = patches.to(device)  # [B, N, 3]
        labels = labels.to(device)  # [B, N]
        
        optimizer.zero_grad()
        
        # Forward pass: model outputs [B, num_classes, N]
        outputs = model(patches)  # [B, num_classes, N]
        
        # Reshape for loss: CrossEntropyLoss expects [B, num_classes, N] and [B, N]
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if (batch_idx + 1) % 10 == 0:
            print(f'  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}')
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for patches, labels in dataloader:
            patches = patches.to(device)  # [B, N, 3]
            labels = labels.to(device)  # [B, N]
            
            outputs = model(patches)  # [B, num_classes, N]
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Calculate accuracy
            preds = outputs.argmax(dim=1)  # [B, N]
            correct += (preds == labels).sum().item()
            total += labels.numel()
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy


def train(config):
    """Main training function for PointNet++ with ball query patches."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Setup logging
    logger = setup_logging(config.get('log_dir', './logs'))
    logger.info('Starting PointNet++ training with ball query patches')
    logger.info(f'Config: {config}')
    
    # Load patches dataset
    patches_path = config.get('patches_path')
    if patches_path is None:
        raise ValueError("'patches_path' must be provided in config")
    
    print(f'Loading patches from: {patches_path}')
    full_dataset = BallQueryPatchDataset(patches_path=patches_path)
    
    # Split into train and validation sets
    val_split = config.get('val_split', 0.2)
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.get('random_seed', 42))
    )
    
    print(f'Dataset split: {train_size} train, {val_size} validation')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=True,
        num_workers=config.get('num_workers', 0),
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=False,
        num_workers=config.get('num_workers', 0),
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Get number of classes from dataset if available
    # Try to infer from labels, or use config default
    num_classes = config.get('num_classes', 10)
    if hasattr(full_dataset, 'patch_labels') and full_dataset.patch_labels is not None:
        unique_labels = np.unique(full_dataset.patch_labels)
        num_classes = len(unique_labels)
        print(f'Detected {num_classes} classes from labels: {unique_labels}')
    
    # Get number of points per patch
    num_points = config.get('num_points', 4096)
    if hasattr(full_dataset, 'patches'):
        num_points = full_dataset.patches.shape[1]
        print(f'Using {num_points} points per patch')
    
    # Initialize model
    print(f'Initializing PointNet++ model: {num_classes} classes, {num_points} points per patch')
    model = PointNetPlusPlusSeg(
        num_classes=num_classes,
        num_points=num_points
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {total_params:,} total, {trainable_params:,} trainable')
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get('learning_rate', 0.001),
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    # Learning rate scheduler (optional)
    scheduler = None
    if config.get('use_scheduler', False):
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=config.get('scheduler_step', 10),
            gamma=config.get('scheduler_gamma', 0.5)
        )
    
    # Training loop
    num_epochs = config.get('num_epochs', 10)
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    print(f'\nStarting training for {num_epochs} epochs...')
    print('=' * 60)
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 60)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f'Learning rate: {current_lr:.6f}')
        
        # Log results
        logger.info(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_loss, config.get('checkpoint_dir', './checkpoints'))
            logger.info(f'New best model saved! Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'✓ New best model saved!')
    
    print('=' * 60)
    logger.info('Training completed!')
    logger.info(f'Best validation loss: {best_val_loss:.4f}, Best validation accuracy: {best_val_acc:.2f}%')
    print(f'\nTraining completed!')
    print(f'Best validation loss: {best_val_loss:.4f}')
    print(f'Best validation accuracy: {best_val_acc:.2f}%')


if __name__ == '__main__':
    # Minimal configuration - only specify what you need to change
    # All other parameters have sensible defaults
    config = {
        'patches_path': 'data/ODMSemantic3D/patches_waterbury_roads_2.npz',
        'batch_size': 8,
        'num_epochs': 20,
        'learning_rate': 0.001,
    }
    
    train(config)

