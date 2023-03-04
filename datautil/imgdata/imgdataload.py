# coding=utf-8
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from datautil.util import Nmax
from datautil.imgdata.util import rgb_loader, l_loader
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
import os
from PIL import Image
    
class ImageDataset(object):
    def __init__(self, dataset, task, root_dir, domain_name, domain_label=-1, labels=None, transform=None, indices=None, test_envs=[], mode='Default'):
        # self.imgs = ImageFolder(root_dir+domain_name).imgs
        path = os.path.join(root_dir, domain_name)
        self.image_dir = os.path.join(path, 'images')
        self.mask_dir = os.path.join(path, 'masks')
        self.images = sorted(os.listdir(self.image_dir))
        self.masks = sorted(os.listdir(self.mask_dir))
        self.task = task
        self.dataset = dataset
        self.transform = transform
        if indices is None:
            self.indices = np.arange(len(self.images))
        else:
            self.indices = indices
        # 没用 凑数
        self.labels = np.ones(len(self.images))
        
        self.dlabels = np.ones(len(self.images)) * (domain_label-Nmax(test_envs, domain_label))

    def __getitem__(self, index):
        index = self.indices[index]
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])

        
        # Load image and mask
        img = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')
        # Apply transform if given
        if self.transform is not None:
            img = self.transform(img)
            
        mask = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor()])(mask)
            
        domain = self.dlabels[index]
        return img, mask, domain

    def __len__(self):
        return len(self.indices)
