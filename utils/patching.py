#%%
import os
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
#import cv2
from PIL import Image
from pathlib import Path

#%%
large_image_path = "insert_path"
large_mask_path = "insert_path"
patch_img_path = "insert_path"
patch_msk_path = "insert_path"
#%%
for filepath in os.listdir(large_image_path):
    img = large_image_path + filepath
    large_image = Image.open(img)
    large_array = np.asarray(large_image)
    patches_img = patchify(large_array, (256, 256, 3), step=256)  #Step=256 for 256 patches means no overlap
    file_name_wo_ext = Path(img).stem
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[i,j,0]
            patch = Image.fromarray(single_patch_img)
            num = i * patches_img.shape[1] + j
            patch.save(f"{patch_img_path}/{file_name_wo_ext}_{num}.jpg")
            

#%%
paths = (os.path.join(root, filename)
        for root, _, filenames in os.walk("insert_path")
        for filename in filenames)

for path in paths:
    
    newname = path.replace('_cp_masks', '')
    if newname != path:
        os.rename(path, newname)

#%%
folder_path = "insert_path"
suffix = '_cp_masks'
for filename in os.listdir(folder_path):
    old_file = os.path.join(folder_path, filename)
    if os.path.isfile(old_file):
        name, ext = os.path.splitext(filename)
        new_file = os.path.join(folder_path, f"{name}{suffix}{ext}")
        os.rename(old_file, new_file)

#%%
for maskpath in os.listdir(large_mask_path):
    msk = large_mask_path + maskpath
    large_mask = Image.open(msk)
    mask_array = np.asarray(large_mask)
    patches_msk = patchify(mask_array, (256, 256), step=256)  #Step=256 for 256 patches means no overlap
    mask_name_wo_ext = Path(msk).stem
    for i in range(patches_msk.shape[0]):
        for j in range(patches_msk.shape[1]):
            single_patch_msk = patches_msk[i,j,:,:]
            patch_msk = Image.fromarray(single_patch_msk)
            num = i * patches_msk.shape[1] + j
            patch_msk.save(f"{patch_msk_path}/{mask_name_wo_ext}_{num}_cp_masks.png")


# %%
