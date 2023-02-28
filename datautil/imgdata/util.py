# coding=utf-8
from torchvision import transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def image_train(dataset, resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.19],
                                     std=[0.14])

    return transforms.Compose([
        # transforms.Grayscale(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(dataset, resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.19],
                                     std=[0.14])

    return transforms.Compose([
        transforms.ToTensor(),
        # normalize
        normalize
    ])


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')
