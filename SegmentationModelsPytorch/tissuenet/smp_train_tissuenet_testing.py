#%%
import os
import torch
import torch.optim as optim
from pathlib import Path
import numpy as np
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
#import albumentations as A
#from albumentations.pytorch import ToTensorV2
from torchvision.transforms import v2
from torchmetrics.classification import BinaryJaccardIndex
import seaborn as sns
import matplotlib.pyplot as plt 
from tqdm import tqdm
from smp_dataset_tissuenet import SMPDatasetTissuenet
import segmentation_models_pytorch as smp

#%% Hyperparams

lr = 0.0001
weight_decay = 1e-8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
batch_size = 10
epochs = 100
pin_memory = True
train_img_dir = "insert_path"
train_mask_dir = "insert_path"
test_img_dir = "insert_path"
test_mask_dir = "insert_path"
model_name = f'UNet++_{epochs}Epochs_lr{lr}'
#%% Instantiate dataset

train_transform = v2.Compose([
                    v2.ToImage(), 
                    v2.Resize((256,256)),
])

#%%
train_ds = SMPDatasetTissuenet(
    image_dir=train_img_dir,
    mask_dir=train_mask_dir,
    transform=train_transform)

#%% Instatiate dataloader
train_dl = DataLoader(
    train_ds,
    batch_size=batch_size,
    pin_memory=pin_memory,
    shuffle=True)


#%% Example image
image, mask = train_ds[0]

#%%
image.shape

#%%
mask.shape

#%% Visualize images
image_train, mask_train = next(iter(train_dl))

fig, axs = plt.subplots(nrows=1, ncols=2)

axs[0].imshow(np.transpose(image_train[0, :, :, :].cpu().numpy(), (1, 2, 0)))
axs[0].set_title('Image')
axs[1].imshow(np.transpose(mask_train[0, :, :].cpu().numpy(), (1, 2, 0))).cmap='gray'
axs[1].set_title('Mask')
plt.show()


#%%
mask_train.unique()

#%%
print(torch.Tensor(image_train).dtype)
#%% Train model
epoch_losses = []
metric = BinaryJaccardIndex()
metric.to(device)

def train():
    model = smp.UnetPlusPlus(
                encoder_name='resnet34',
                encoder_weights='imagenet',
                classes=1,
                activation=None).to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    #criterion = nn.BCEWithLogitsLoss() #BCEwithlogits has sigmoid incl. 
    #criterion = smp.losses.DiceLoss('binary')
    criterion = smp.losses.SoftBCEWithLogitsLoss()

    input, target = next(iter(train_dl))
    for epoch in tqdm(range(epochs)):
        #input = input.type(torch.float).to(device)
        #target = target.type(torch.float).to(device) #.unsqueeze(1) 
        input = input.type(torch.float).to(device)
        target = target.to(device)
        # to skip last item if batch size == 1
        if input.shape[0] < 2:
            continue
        optimizer.zero_grad()
        preds = model(input)
        #preds = preds.type(torch.float)
        loss = criterion(preds, target.float())
        loss.backward()
        optimizer.step()
        iou = metric(preds, target)
        #tqdm.set_description(f'Epoch [{epoch}/{epochs}]')
        #tqdm.set_postfix(loss=loss.item(), iou=iou, lr=lr)
        epoch_losses.append(loss.item())
    if epoch % 25 == 1:
        print(f'Loss on Epoch {epoch}: {sum(epoch_losses)/len(epoch_losses)}')
        print(f'IOU on Epoch {epoch}: {iou}')
    print('Training complete. Model saved!')

if __name__== '__main__':
    train()

iou_train = metric.compute()
print(f'Final training IOU: {iou_train}')

#%%
sns.lineplot(x=range(len(epoch_losses)), y=epoch_losses).set(title='Train Loss')
plt.show()

#%%
