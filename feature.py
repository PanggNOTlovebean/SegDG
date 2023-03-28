import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageFile
import os
import random
import numpy as np
from alg.algs.DAMX import DAMX
from alg.algs.ERM import ERM
from alg.algs.Mixup import Mixup
from alg.algs.DANN import DANN
from train import get_args
# import pandas as pd

ImageFile.LOAD_TRUNCATED_IMAGES = True
crop_size = 224
import torch.nn.functional as F

def image_test(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.19],
                                     std=[0.14])
    return transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        # normalize
        normalize
    ])
    

def accuracy(pred, target):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    # Update confusion matrix
    tp = (pred * target).sum().item()
    tn = ((1 - pred) * (1 - target)).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    # Compute metrics
    acc =  (tp + tn) / (tp + tn + fp + fn + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    miou = (iou + tn / (tn + fp + fn + 1e-7)) / 2
    dice = 2 * tp / (2 * tp + fp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    return iou

datasets = ['BIDMC','BMC', 'HK', 'I2CVB', 'RUNMC', 'UCL']
# result = pd.DataFrame()
for i in range(0, len(datasets)):
    base_model = 'output'
    algo = 'DANN'
    model_dir = os.path.join(base_model,f'{algo}/output{i}/model_val.pkl')
    state_dict = torch.load(model_dir, map_location=torch.device('cpu'))
    args = get_args()
    # model = Mixup(args)
    # args.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    model = DANN(args)
    model.load_state_dict(state_dict['model_dict'])
    model = model.cuda()
    model.eval()
    
    for j in range(len(datasets)):
        dataset = datasets[j]
        transform = image_test()
        path = 'data/' + dataset 
        image_dir = os.path.join(path, 'images')
        mask_dir = os.path.join(path, 'masks')
        y_dir = os.path.join(path, f'y_{algo}_{base_model}')
        z_dir = os.path.join(path, f'z_{i}_{algo}_{base_model}')
        print(z_dir)
        os.makedirs(y_dir) if not os.path.exists(y_dir) else None
        os.makedirs(z_dir) if not os.path.exists(z_dir) else None
            
        images = sorted(os.listdir(image_dir))
        masks = sorted(os.listdir(mask_dir))
        
        tensors = []
        
        for k in range(len(images)):
            img_path = os.path.join(image_dir, images[k])
            mask_path = os.path.join(mask_dir, masks[k])
            img = Image.open(img_path).convert('L')
            mask = Image.open(mask_path).convert('L')
            img = transform(img)
            mask = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor()])(mask)
            y, z = model.predict(img.unsqueeze(0).cuda())
            z_path = os.path.join(z_dir, masks[k][:-4] + '.pt')
            torch.save(z.cpu(), os.path.join(z_path))