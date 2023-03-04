# coding=utf-8
from torchvision import transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
crop_size = 224

def image_train(dataset, resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.19],
                                     std=[0.14])

    return transforms.Compose([
        transforms.CenterCrop(crop_size),
        # transforms.Grayscale(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(dataset, resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.19],
                                     std=[0.14])
    return transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        # normalize
        normalize
    ])
    
def image_mask(dataset, resize_size=256, crop_size=224):
    return transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
    ])

def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')
