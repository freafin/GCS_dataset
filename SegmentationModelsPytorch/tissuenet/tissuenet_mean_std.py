"""
This code is heavily inspired by Aladdin Persson:
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_std_mean.py
"""
#%%
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from mean_std_dataset import mean_std_dataset
from tqdm import tqdm

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

train_val_data = mean_std_dataset(
    image_dir= "insert_path",
    transform=transform,
)

train_val_loader = DataLoader(dataset=train_val_data, batch_size=5, shuffle=True)

#%%
def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean**2) ** 0.5

    return mean, std


mean, std = get_mean_std(train_val_loader)
print(f"Mean:", {mean})
print(f"Standard dev:", {std})

# %%
