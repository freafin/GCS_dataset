
#%%
import os
from PIL import Image
import numpy as np

mask_folder = "insert_path"
total_instances = 0 

for filename in os.listdir(mask_folder):
    if filename.lower().endswith(".png"):
        file_path = os.path.join(mask_folder, filename)
        
        # Load image
        image = Image.open(file_path)
        mask_array = np.array(image)

        # Count unique non-zero pixel values
        unique_instances = np.unique(mask_array)
        instance_count = len(unique_instances[unique_instances != 0])
        total_instances += instance_count
        print(f"{filename}: {instance_count} instances")

print(f"Total instances across all masks: {total_instances}")


# %%
