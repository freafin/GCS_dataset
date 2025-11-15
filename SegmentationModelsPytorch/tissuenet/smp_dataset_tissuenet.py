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
import skimage as ski

#%%
class SMPDatasetTissuenet(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = self.masks[idx]
        mask_path = os.path.join(self.mask_dir, mask_name)
        image = Image.open(img_path).convert('RGB')
        image = np.array(image, dtype='uint8')
        
        # Convert masks to binary (workaround due to alpha channel noise)
        mask = Image.open(mask_path)
        mask_array = np.array(mask)
        mask_gray = cv2.cvtColor(mask_array, cv2.COLOR_BGRA2GRAY)
        mask = np.where(mask_gray > np.min(mask_gray), 1, 0)
        #mask = ski.util.img_as_ubyte(mask)
        mask = np.array(mask, dtype='uint8')
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
