import os
import sys
import tarfile
import collections
import torch.utils.data as data
import shutil
import numpy as np
import cv2

from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity

DATASET_YEAR_DICT = {
    '2012': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '6cd6e144f989b92b3379bac3b3de84fd',
        'base_dir': 'VOCdevkit/VOC2012'
    },
    '2011': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar',
        'filename': 'VOCtrainval_25-May-2011.tar',
        'md5': '6c3384ef61512963050cb5d687e5bf1e',
        'base_dir': 'TrainVal/VOCdevkit/VOC2011'
    },
    '2010': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar',
        'filename': 'VOCtrainval_03-May-2010.tar',
        'md5': 'da459979d0c395079b5c75ee67908abb',
        'base_dir': 'VOCdevkit/VOC2010'
    },
    '2009': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar',
        'filename': 'VOCtrainval_11-May-2009.tar',
        'md5': '59065e4b188729180974ef6572f6a212',
        'base_dir': 'VOCdevkit/VOC2009'
    },
    '2008': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '2629fa636546599198acfcfbfcf1904a',
        'base_dir': 'VOCdevkit/VOC2008'
    },
    '2007': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
        'filename': 'VOCtrainval_06-Nov-2007.tar',
        'md5': 'c52e279531787c972589f7e41ab4ae64',
        'base_dir': 'VOCdevkit/VOC2007'
    }
}


def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

class BDDSegmentation(data.Dataset):
    
    cmap = voc_cmap(N=1)
    
    def __init__(self,
                 image_set='train',
                 transform=None):
        self.is_train = image_set == 'train'
        self.transform = transform
        
        # 이미지와 마스크 경로를 설정합니다.
        self.images = self.load_image_paths(image_set)
        self.masks = self.load_mask_paths(image_set)
        

    def load_image_paths(self, image_set):
        # 이미지 경로를 로드하는 로직을 구현합니다.
        image_dir = f'/home/mobinn/HybridNets/datasets/bdd100k/{image_set}'
        return [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.jpg')]

    def load_mask_paths(self, image_set):
        # 마스크 경로를 로드하는 로직을 구현합니다.
        mask_dir = f'/home/mobinn/HybridNets/datasets/bdd_lane_gt/{image_set}'
        return [os.path.join(mask_dir, mask) for mask in os.listdir(mask_dir) if mask.endswith('.png')]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        img_array = np.array(img)
        target = Image.open(self.masks[index])
        target_array = np.array(target)
        print(img_array.shape)
        print(target_array.shape)

        _, target = cv2.threshold(target_array, 0, 255, cv2.THRESH_BINARY)

        target = (target / 255).astype(np.float32)

        print(target)
        
        if self.transform is not None:
            img, target = self.transform(img, target)


        return img, target

    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]

def download_extract(url, root, filename, md5):
    download_url(url, root, filename, md5)
    with tarfile.open(os.path.join(root, filename), "r") as tar:
        tar.extractall(path=root)