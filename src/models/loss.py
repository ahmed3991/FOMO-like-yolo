"""
Loss Functions for FOMO-style Object Detection

Binary cross-entropy based loss for center-point heatmap prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FOMOLoss(nn.Module):
    """
    FOMO Loss using Binary Cross Entropy.
    
    Unlike traditional object detection losses (bbox regression + classification),
    FOMO only predicts object center probabilities on a grid.
    
    Args:
        pos_weight: Optional weight for positive samples (to handle class imbalance)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, pos_weight=None, reduction='mean'):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
        
        # BCEWithLogitsLoss = Sigmoid + BCELoss (more numerically stable)
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,
            reduction=reduction
        )
    
    def forward(self, predictions, targets):
        """
        Compute FOMO loss.
        
        Args:
            predictions: Model output logits of shape (B, C, H, W)
            targets: Ground truth heatmaps of shape (B, C, H, W)
            
        Returns:
            Loss value (scalar if reduction='mean' or 'sum')
        """
        return self.criterion(predictions, targets)


class FocalFOMOLoss(nn.Module):
    """
    Focal Loss variant for FOMO to handle hard negative mining.
    
    Focal loss down-weights easy examples and focuses on hard negatives.
    Useful when most grid cells are background (negative samples).
    
    Args:
        alpha: Weighting factor for positive vs negative samples (default: 0.25)
        gamma: Focusing parameter (default: 2.0), higher = more focus on hard examples
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, predictions, targets):
        """
        Compute Focal FOMO loss.
        
        Args:
            predictions: Model output logits of shape (B, C, H, W)
            targets: Ground truth heatmaps of shape (B, C, H, W)
            
        Returns:
            Loss value
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(predictions)
        
        # Compute binary cross entropy
        bce = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        
        # Compute focal term: (1 - p_t)^gamma
        # p_t = p if target = 1, else 1 - p
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Combine
        loss = alpha_t * focal_weight * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WeightedFOMOLoss(nn.Module):
    """
    FOMO Loss with custom per-class weighting.
    
    Useful when some classes are more important or more rare than others.
    
    Args:
        class_weights: Tensor of shape (num_classes,) with weight per class
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, class_weights=None, reduction='mean'):
        super().__init__()
        self.class_weights = class_weights
        self.reduction = reduction
    
    def forward(self, predictions, targets):
        """
        Compute weighted FOMO loss.
        
        Args:
            predictions: Model output logits of shape (B, C, H, W)
            targets: Ground truth heatmaps of shape (B, C, H, W)
            
        Returns:
            Loss value
        """
        # Compute BCE per sample
        bce = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        
        # Apply class weights if provided
        if self.class_weights is not None:
            # Reshape weights to (1, C, 1, 1) for broadcasting
            weights = self.class_weights.view(1, -1, 1, 1).to(predictions.device)
            bce = bce * weights
        
        if self.reduction == 'mean':
            return bce.mean()
        elif self.reduction == 'sum':
            return bce.sum()
        else:
            return bce


if __name__ == "__main__":
    """
    Test loss functions.
    """
    print("=" * 60)
    print("FOMO Loss Function Tests")
    print("=" * 60)
    
    # Create dummy data
    batch_size = 4
    num_classes = 2
    output_size = 20
    
    # Predictions (logits)
    predictions = torch.randn(batch_size, num_classes, output_size, output_size)
    
    # Targets (binary heatmaps)
    targets = torch.zeros(batch_size, num_classes, output_size, output_size)
    # Add some positive samples
    targets[0, 0, 10, 10] = 1.0
    targets[1, 1, 5, 5] = 1.0
    targets[2, 0, 15, 15] = 1.0
    
    # Test 1: Standard FOMO Loss
    print("\n[Test 1] Standard FOMO Loss")
    print("-" * 60)
    loss_fn = FOMOLoss()
    
    # Ensure predictions require grad for testing
    predictions_with_grad = predictions.clone().requires_grad_(True)
    loss = loss_fn(predictions_with_grad, targets)
    
    print(f"Loss value: {loss.item():.4f}")
    print(f"Loss shape: {loss.shape if loss.dim() > 0 else 'scalar'}")
    assert loss.requires_grad, "Loss should require gradients"
    print("✓ Test passed")
    
    # Test 2: FOMO Loss with positive weighting
    print("\n[Test 2] FOMO Loss with Positive Weighting")
    print("-" * 60)
    pos_weight = torch.tensor([10.0])  # Weight positive samples 10x more
    loss_fn_weighted = FOMOLoss(pos_weight=pos_weight)
    loss_weighted = loss_fn_weighted(predictions_with_grad, targets)
    print(f"Weighted loss value: {loss_weighted.item():.4f}")
    print(f"Standard loss value: {loss.item():.4f}")
    print("✓ Test passed")
    
    # Test 3: Focal FOMO Loss
    print("\n[Test 3] Focal FOMO Loss")
    print("-" * 60)
    focal_loss_fn = FocalFOMOLoss(alpha=0.25, gamma=2.0)
    predictions_focal = predictions.clone().requires_grad_(True)
    focal_loss = focal_loss_fn(predictions_focal, targets)
    print(f"Focal loss value: {focal_loss.item():.4f}")
    assert focal_loss.requires_grad, "Loss should require gradients"
    print("✓ Test passed")
    
    # Test 4: Weighted FOMO Loss
    print("\n[Test 4] Weighted FOMO Loss (Per-Class)")
    print("-" * 60)
    class_weights = torch.tensor([1.0, 2.0])  # Class 1 weighted 2x more than class 0
    class_weighted_loss_fn = WeightedFOMOLoss(class_weights=class_weights)
    predictions_class_weighted = predictions.clone().requires_grad_(True)
    class_weighted_loss = class_weighted_loss_fn(predictions_class_weighted, targets)
    print(f"Class-weighted loss value: {class_weighted_loss.item():.4f}")
    print("✓ Test passed")
    
    # Test 5: Gradient flow
    print("\n[Test 5] Gradient Flow")
    print("-" * 60)
    predictions.requires_grad = True
    loss = loss_fn(predictions, targets)
    loss.backward()
    assert predictions.grad is not None, "Gradients should be computed"
    print(f"Gradient mean: {predictions.grad.mean().item():.6f}")
    print(f"Gradient std: {predictions.grad.std().item():.6f}")
    print("✓ Test passed")
    
    # Test 6: Different reductions
    print("\n[Test 6] Different Reductions")
    print("-" * 60)
    for reduction in ['mean', 'sum', 'none']:
        loss_fn_red = FOMOLoss(reduction=reduction)
        loss_red = loss_fn_red(predictions.detach(), targets)
        print(f"Reduction '{reduction}': shape = {loss_red.shape}, value = {loss_red.mean().item():.4f}")
    print("✓ Test passed")
    
    print("\n" + "=" * 60)
    print("✓ All loss function tests passed!")
    print("=" * 60)
