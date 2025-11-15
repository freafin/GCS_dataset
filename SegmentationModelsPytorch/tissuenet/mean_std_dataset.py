#%%
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from pathlib import Path
import cv2
from PIL import Image
import numpy as np

#%%
class mean_std_dataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
    
        return image
# %%
