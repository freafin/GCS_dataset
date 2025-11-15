#%%
import os, random, shutil
import cv2
import numpy as np
from pathlib import Path

#%%
os.chdir("insert_path")
#%% Makedirs
def create_folders():
    FOLDERS = ['train', 
               #'val', 
               'test']
    for folder in FOLDERS:
        if not os.path.exists(folder):
            folder_imgs = f"{folder}/images"
            folder_msks = f"{folder}/masks"
            os.makedirs(folder_imgs) if not os.path.exists(folder_imgs) else print('folder already exists')
            os.makedirs(folder_msks) if not os.path.exists(folder_msks) else print('folder already exists')

create_folders()


#%%
os.chdir("insert_path")
source_folder_images = "./complete/patched/images"
source_folder_masks = "./complete/patched/masks"

train_folder_images = "./train_test_split/patched/train/images"
train_folder_masks = "./train_test_split/patched/train/masks"
#val_folder_images = "./val/images"
#val_folder_masks = "./val/masks"
test_folder_images = "./train_test_split/patched/test/images"
test_folder_masks = "./train_test_split/patched/test/masks"

selected_files = os.listdir(source_folder_images)
print(len(selected_files))

number_train_files = np.round(0.8*len(selected_files),0)
#number_val_files = np.round(0.2*len(selected_files),0)
number_test_files = np.round(0.2*len(selected_files),0)
print(f"Number of images for training:", number_train_files)
#print(f"Number of images for validation: ", number_val_files)
print(f"Number of images for testing: ", number_test_files)

#%% Make train set
# Keep both masks and images as jpg if wanted
for file in selected_files:
    # Create paths for source and target
    file_path_images = os.path.join(source_folder_images, file)
    target_path_images = os.path.join(train_folder_images)
    # Repeat for masks
    file_path_masks = os.path.join(source_folder_masks, Path(file).stem + '_cp_masks.png')
    #file_path_masks = file_path_masks.replace('jpg', 'png')
    target_path_masks = os.path.join(train_folder_masks, Path(file).stem + '_cp_masks.png')
    #target_path_masks = target_path_masks.replace('jpg', 'png')
    # Copy from source to training folder
    shutil.copy(file_path_images, target_path_images)
    shutil.copy(file_path_masks, target_path_masks)

#%% Make val set
# Move files from train to val folder
np.random.seed(42)
for i in range(int(number_val_files)):
    # Pick a random file
    random_file = np.random.choice(selected_files, 1)[0]
    #print(random_file)
    # Remove the filename from the selected_file (to avoid duplicates)
    selected_files.remove(random_file)
    # Define where to pick up and drop images
    train_path_images = os.path.join(train_folder_images, random_file)
    val_path_images = os.path.join(val_folder_images, random_file)
    # Repeat for the masks
    train_path_masks = os.path.join(train_folder_masks, random_file)
    val_path_masks = os.path.join(val_folder_masks, random_file)
    # Move from train to val folder
    shutil.move(train_path_images, val_path_images)
    shutil.move(train_path_masks, val_path_masks)

#%% Make test set 
# Move files from train to test folder
np.random.seed(42)
for i in range(int(number_test_files)):
    # Pick a random file
    random_file = np.random.choice(selected_files, 1)[0]
    #print(random_file)
    # Remove the filename from the selected_file (to avoid duplicates)
    selected_files.remove(random_file)
    # Define where to pick up and drop images
    train_path_images = os.path.join(train_folder_images, random_file)
    test_path_images = os.path.join(test_folder_images, random_file)
    # Repeat for the masks
    train_path_masks = os.path.join(train_folder_masks, Path(random_file).stem + '_cp_masks.png') #random_file)
    test_path_masks = os.path.join(test_folder_masks, Path(random_file).stem + '_cp_masks.png') #random_file)
    # Move from train to val folder
    shutil.move(train_path_images, test_path_images)
    shutil.move(train_path_masks, test_path_masks)

# %%
