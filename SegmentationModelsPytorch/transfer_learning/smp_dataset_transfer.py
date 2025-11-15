#%%
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import v2
import os
from pathlib import Path
import cv2
from PIL import Image
import numpy as np

#%%
class SMPDatasetCellpose(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
    
    # Assuming that the mask has the same base name as the image but with a .png extension
        mask_name = os.path.splitext(img_name)[0] + '_cp_masks.png'
        mask_path = os.path.join(self.mask_dir, mask_name)
    
        image = Image.open(img_path)
        mask = Image.open(mask_path)
    
    # Convert to binary masks
        mask_array = np.array(mask)
        mask = np.where(mask_array > 0, 1, 0)  # Convert any non-zero value to 1
    # Convert to tensor to avoid loss of binary masks upon scaling
        image_tensor = v2.ToDtype(torch.float32, scale=True)
        mask_tensor = v2.ToDtype(torch.float32, scale=False)
        image = image_tensor(image)
        mask = mask_tensor(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            #transformed = self.transform(image=image, mask=mask)
            #image = transformed['image']
            #mask = transformed['mask']
    
        return image, mask
# %%
