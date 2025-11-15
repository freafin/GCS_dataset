import torch
import torch.nn.functional as F

def dice_loss(preds, input, smooth=1):
    """
    Computes the Dice Loss for binary segmentation.
    Args:
        preds: Tensor of predictions (batch_size, 1, H, W).
        input: Tensor of ground truth (batch_size, 1, H, W).
        smooth: Smoothing factor to avoid division by zero.
    Returns:
        Scalar Dice Loss.
    """
    # Apply sigmoid to convert logits to probabilities
    preds = torch.sigmoid(preds)
    
    # Calculate intersection and union
    intersection = (preds * input).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + input.sum(dim=(2, 3))
    
    # Compute Dice Coefficient
    dice = (2. * intersection + smooth) / (union + smooth)
    
    # Return Dice Loss
    return 1 - dice.mean()