import numpy as np
import cv2
import dataloader.transforms as transforms
from dataloader.base_dataloader import BaseDataloader

iheight, iwidth = 480, 640 # raw image size


class NYUDataset(BaseDataloader):
    def __init__(self, root:str,  type: str='train', sparsifier=None, modality='rgb'):
        super(NYUDataset, self).__init__(root, type, sparsifier, modality)

    
    def train_transform(self, rgb, depth):
        s = np.random.uniform(1.0, 1.5) # random scaling
        depth_np = depth / s
        angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5 # random horizontal flip

        # perform 1st step of data augmentation
        transform = transforms.Compose([
            transforms.Resize((iwidth,iheight)), #250.0 / iheight), # this is for computational efficiency, since rotation can be slow
            transforms.Rotate(angle),
            transforms.HorizontalFlip(do_flip)
        ])
            
        rgb = self.color_jitter(rgb) # random color jittering
        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)

        return rgb_np, depth_np

    def val_transform(self, rgb, depth):
        depth_np = depth
        transform = transforms.Compose([
            transforms.Resize((iwidth,iheight))
        ])
        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)


        return rgb_np, depth_np
