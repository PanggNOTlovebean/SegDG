# coding=utf-8
import torch
from network import img_network
import torch.nn.functional as F
from torch import nn
import numpy as np
def get_fea(args):
    if args.dataset == 'dg5':
        net = img_network.DTNBase()
    elif args.net.startswith('res'):
        net = img_network.ResBase(args.net)
    else:
        net = img_network.VGGBase(args.net)
    return net

def surface_distance(prediction, target, spacing):
    """
    Calculate the surface distance between two binary masks.
    :param prediction: PyTorch tensor of shape (batch_size, 1, H, W, D).
    :param target: PyTorch tensor of shape (batch_size, 1, H, W, D).
    :param spacing: Numpy array of shape (3,) representing the voxel spacing in mm.
    :return: Average surface distance in mm.
    """
    # Compute the binary masks for the prediction and target.
    pred_mask = (prediction > 0.5).float()
    target_mask = (target > 0.5).float()

    # Compute the distance transform of the masks.
    pred_dist = torch.cdist(torch.nonzero(pred_mask), torch.nonzero(1 - pred_mask), p=2)
    target_dist = torch.cdist(torch.nonzero(target_mask), torch.nonzero(1 - target_mask), p=2)

    # Convert distance to physical units (mm).
    pred_dist = pred_dist * torch.tensor(spacing).to(pred_dist.device)
    target_dist = target_dist * torch.tensor(spacing).to(target_dist.device)

    # Compute the average surface distance.
    asd = (torch.sum(pred_dist) + torch.sum(target_dist)) / (torch.numel(pred_mask) + torch.numel(target_mask))

    return asd.item()

def accuracy(network, loader):
    with torch.no_grad():
        # Set batchnorm and dropout layers to eval mode
        network.eval()
        for module in network.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.Dropout):
                module.eval()

        tp, tn, fp, fn = 0, 0, 0, 0
        for data in loader:
            x = data[0].cuda().float()
            target = data[1].cuda().float()
            pred = network.predict(x)
            pred = (pred > 0.5).float()
            target = (target > 0.5).float()

            # Update confusion matrix
            tp += (pred * target).sum().item()
            tn += ((1 - pred) * (1 - target)).sum().item()
            fp += (pred * (1 - target)).sum().item()
            fn += ((1 - pred) * target).sum().item()

        # Compute metrics
        acc =  (tp + tn) / (tp + tn + fp + fn + 1e-7)
        iou = tp / (tp + fp + fn + 1e-7)
        miou = (iou + tn / (tn + fp + fn + 1e-7)) / 2
        dice = 2 * tp / (2 * tp + fp + fn + 1e-7)
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)

    # Set batchnorm and dropout layers back to train mode
    network.train()
    for module in network.modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.Dropout):
            module.train()
            
    return [acc, iou, miou, dice, precision, recall]


def CE_Loss(inputs, target, cls_weights, num_classes=21):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss  = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)
    return CE_loss

def Focal_Loss(inputs, target, cls_weights, num_classes=21, alpha=0.5, gamma=2):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    logpt  = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes, reduction='none')(temp_inputs, temp_target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss

def dice_loss(predictive, target, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss


def calc_loss(prediction, target, bce_weight=0.5):
    """Calculating the loss and metrics
    Args:
        prediction = predicted image
        target = Targeted image
        metrics = Metrics printed
        bce_weight = 0.5 (default)
    Output:
        loss : dice loss of the epoch """
    bce = F.binary_cross_entropy_with_logits(prediction, target)
    prediction = F.sigmoid(prediction)
    dice = dice_loss(prediction, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    return loss


class BCEFocalLoss(torch.nn.Module):
    """
    二分类的Focalloss alpha 固定
    """
    def __init__(self, gamma=2, alpha=0.25, reduction='sum'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
 
    def forward(self, _input, target):
        pt = torch.sigmoid(_input)
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss
    
def threshold_predictions_v(predictions, thr=150):
    thresholded_preds = predictions[:]
   # hist = cv2.calcHist([predictions], [0], None, [2], [0, 2])
   # plt.plot(hist)
   # plt.xlim([0, 2])
   # plt.show()
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 255
    return thresholded_preds


def threshold_predictions_p(predictions, thr=0.01):
    thresholded_preds = predictions[:]
    #hist = cv2.calcHist([predictions], [0], None, [256], [0, 256])
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds