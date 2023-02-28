from nets import Unet
import torch
from PIL import Image, ImageFile
import os
from torchvision import transforms
import numpy as np
# from alg.modelopera import Dice_loss, dice_loss
def unet_test():
    unet = Unet(num_classes = 1, pretrained = True, backbone = 'vgg')
    x = torch.ones(4, 1, 384, 384)
    print(x.shape)
    y, z = unet(x)
    # print(y,z)
    print(y.shape, z.shape)
    # print(y)
    
def DICEloss_test():
    input = torch.randn(4, 1, 384, 384)
    target = torch.randn(4, 1, 384, 384)
    loss = torch.nn.CrossEntropyLoss()
    out = loss(input,target)
    print(out)
class BCEFocalLoss(torch.nn.Module):
    """
    二分类的Focalloss alpha 固定
    """
    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
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
    
def focalloss_test():
    input = torch.randn(4, 1, 384, 384)
    target = torch.ones(4, 1, 384, 384) 
    focalLoss = BCEFocalLoss(alpha = 0.8)
    loss = focalLoss(input,target)
    print (loss)
    
def imageMean():
    normalize = transforms.Normalize(mean=[0.19],
                                     std=[0.14])

    transform = transforms.Compose([
        # transforms.Grayscale(),
        transforms.ToTensor(),
        normalize
    ])
    sites = ['BIDMC', 'BMC', 'HK', 'I2CVB', 'RUNMC', 'UCL']
    folder = 'data'
    mean = []
    std = []
    for site in sites:
        fold = os.path.join(folder, site, 'images')
        for filename in os.listdir(fold):
            img = Image.open(os.path.join(fold, filename)).convert('L')
            img = transform(img)
            mean.append(img.mean())
            std.append(img.std())
    print(np.array(mean).mean())
    print(np.array(std).mean())
    
    
if __name__ == '__main__':
    imageMean()