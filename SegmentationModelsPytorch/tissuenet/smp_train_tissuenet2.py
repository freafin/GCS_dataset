#%%
import os
import torch
import torch.optim as optim
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
#import albumentations as A
#from albumentations.pytorch import ToTensorV2
from torchvision.transforms import v2
from torchmetrics.classification import BinaryJaccardIndex, Accuracy
import seaborn as sns
import matplotlib.pyplot as plt 
from tqdm import tqdm
from smp_dataset_tissuenet import SMPDatasetTissuenet
import segmentation_models_pytorch as smp

#%% Hyperparams

lr = 0.001
weight_decay = 1e-8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
batch_size = 4
epochs = 300
pin_memory = True
train_img_dir = "insert_path"
train_mask_dir = "insert_path"
test_img_dir = "insert_path"
test_mask_dir = "insert_path"
model_name = f'UnetPP_{epochs}Epochs_lr{lr}'
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
epoch_accuracy = []
epoch_iou = []
metric1 = BinaryJaccardIndex()
metric2 = Accuracy(task='binary')
metric1.to(device)
metric2.to(device)

def train():
    model = smp.UnetPlusPlus(
                encoder_name='efficientnet-b7',
                encoder_weights='imagenet',
                classes=1,
                activation=None).to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    #criterion = nn.BCEWithLogitsLoss() #BCEwithlogits has sigmoid incl. 
    criterion = smp.losses.DiceLoss('binary')
    #criterion = smp.losses.SoftBCEWithLogitsLoss()
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, 
    #                                                 patience=5, min_lr=0.00001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5) 
    for epoch in range(1, epochs + 1):
        loop = tqdm((train_dl), total=len(train_dl), leave=False)
        for batch in loop:
            input, target = batch
            #input = input.type(torch.float).to(device)
            #target = target.type(torch.float).to(device) #.unsqueeze(1) 
            input = input.type(torch.float).to(device)
            target = target.to(device)
            # to skip last item if batch size == 1
            if input.shape[0] < 2:
                continue
            optimizer.zero_grad()
            preds = model(input)
            #preds = (preds > 0.5) #.float() # remove if included in criterion
            #preds = preds.type(torch.float)
            loss = criterion(preds, target.float()) #target.float()
            loss.backward()
            optimizer.step()
            iou = metric1(preds, target)
            acc = metric2(preds, target)
            loop.set_description(f'Epoch [{epoch}/{epochs}]')
            loop.set_postfix(loss=loss.item(), acc=acc, iou=iou)
        scheduler.step() # loss
        epoch_losses.append(loss.item())
        epoch_accuracy.append(acc.item())
        epoch_iou.append(iou.item())
        current_lr = optimizer.param_groups[0]["lr"]
        if epoch % 50 == 0:
            print(f'Loss on Epoch {epoch}: {sum(epoch_losses)/len(epoch_losses)}')
            print(f'Accuracy on Epoch {epoch}: {acc}')
            print(f'IOU on Epoch {epoch}: {iou}')
            print(f'Current LR = {current_lr}')
    torch.save(model, f'"insert_path"/{model_name}.pth')
    print('Training complete. Model saved!')

if __name__== '__main__':
    train()

iou_train = metric1.compute()
acc_train = metric2.compute()
print(f'Final training IOU: {iou_train}')
print(f'Final training accuracy: {acc_train}')
#%%
data = pd.DataFrame(list(zip(epoch_losses, epoch_accuracy, epoch_iou)), columns=['Loss', 'Accuracy', 'IOU'])
#%%
sns.lineplot(data=data)

#%%
test_model = torch.load(f'"insert_path"/{model_name}.pth', 
                            map_location='cpu', weights_only=False)

#print(f'Model {model_name} loaded.')
print(f'Model loaded.')
#%%
test_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((256, 256)),
])

test_ds = SMPDatasetTissuenet(
    image_dir=test_img_dir,
    mask_dir=test_mask_dir,
    transform=test_transform)
    
test_dl = DataLoader(
    test_ds,
    batch_size=batch_size,
    pin_memory=pin_memory,
    shuffle=False)


#%% Visualize test set
test_image, test_mask = test_ds[0]

print(test_image.shape)
print(test_mask.shape)

image_test, mask_test = next(iter(test_dl))

fig, axs = plt.subplots(nrows=1, ncols=2)

axs[0].imshow(np.transpose(image_test[0, :, :, :].cpu().numpy(), (1, 2, 0)))
axs[0].set_title('Image')
axs[1].imshow(np.transpose(mask_test[0, :, :].cpu().numpy(), (1, 2, 0))).cmap='gray'
axs[1].set_title('Mask')
plt.show()

#%% Test model

test_metric1 = BinaryJaccardIndex()
test_metric2 = Accuracy(task='binary')
test_metric1.to(device)
test_metric2.to(device)
model = test_model

with torch.no_grad():
    model.to(device)
    model.eval()
    for input, target in tqdm(test_dl):
        input = input.to(device).float()
        target = target.to(device)#.unsqueeze(1)
        preds = model(input)
        #preds = torch.sigmoid(model(input))
        preds = (preds > 0.5).float()
        iou_test = test_metric1(preds, target)
        acc_test = test_metric2(preds, target)


iou_test = test_metric1.compute()
acc_test = test_metric2.compute()
print(f'Final test IOU: {iou_test}')
print(f'Final test accuracy: {acc_test}')
#%%
print(model_name)

# %%                            Visualize

test_data_vis = SMPDatasetTissuenet(
    image_dir=test_img_dir,
    mask_dir=test_mask_dir,
    transform=test_transform,
    )

vis_batch_size = 3
vis_dl = DataLoader(test_data_vis, batch_size=vis_batch_size, shuffle=True)

test_model.eval()
fig, axes = plt.subplots(vis_batch_size, 3, figsize=(8, 8), dpi=250)
for i in range(vis_batch_size): #vis_batch_size
    x, y = next(iter(vis_dl))
    x, y = x.to(device).float(), y.to(device)#.unsqueeze(1) x..float()
    #mask_pred = torch.sigmoid(test_model(x))
    mask_pred = test_model(x)
    mask_pred = (mask_pred > 0.5).float()
    mask_pred = mask_pred.cpu().detach().numpy().squeeze()
    x = x.cpu()
    x = np.array(x, dtype='uint8')
    axes[i, 0].imshow(np.transpose(x[i], (1, 2, 0)))#.astype('uint8')
    axes[i, 0].set_title("Image")
    axes[i, 0].set_xticks([])
    axes[i, 0].set_yticks([])
    axes[i, 1].imshow(np.transpose(y[i].cpu().numpy(), (1, 2, 0)))
    axes[i, 1].set_title("Ground truth")
    axes[i, 1].set_xticks([])
    axes[i, 1].set_yticks([])
    axes[i, 2].imshow(mask_pred[i], cmap='gray')
    axes[i, 2].set_title("Predicted")
    axes[i, 2].set_xticks([])
    axes[i, 2].set_yticks([])



# %%
test_model
# %%