#%% Imports
import numpy as np
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from smp_dataset_tissuenet import SMPDatasetTissuenet
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils.metrics
import seaborn as sns
import matplotlib.pyplot as plt 

#%%
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


#%%
train_transform = v2.Compose([
                    v2.ToImage(), 
                    v2.Resize((256,256)),
])


#%% Instantiate Dataset 
#train_ds = InstanceDataset(path_name='train')
train_ds = SMPDatasetTissuenet(
    image_dir= "path",
    mask_dir= "path2",
    transform=train_transform,
)

#%% Instantiate DataLoader
batchsize = 64

train_dl = DataLoader(train_ds, batch_size=batchsize, shuffle=True)


#%% Example image
image, mask = train_ds[0]

#%%
image.shape

#%%
mask.shape

#%% Visualize images
#%% Visualize images
image_train, mask_train = next(iter(train_dl))

fig, axs = plt.subplots(nrows=1, ncols=2)

axs[0].imshow(np.transpose(image_train[0, :, :, :].cpu().numpy(), (1, 2, 0)))
axs[0].set_title('Image')
axs[1].imshow(np.transpose(mask_train[0, :, :].cpu().numpy(), (1, 2, 0))).cmap='gray'
axs[1].set_title('Mask')
plt.show()

#%% Params
encoder = 'resnet34'
weights = 'imagenet'
classes = 1
activation = None
lr = 0.01
epochs = 100
loss = smp.losses.JaccardLoss('binary') #nn.CrossEntropyLoss() 
loss.__name__ = 'JaccardLoss'

metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
    smp.utils.metrics.Fscore(),
    smp.utils.metrics.Accuracy(),
]
#%% Model
model_n = 'Unet++'

model = smp.UnetPlusPlus(
    encoder_name=encoder,
    encoder_weights=weights,
    classes=classes,
    activation=activation,
)

optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=lr),  
])



#%% Example image
image, mask = train_ds[0]

#%%
image.shape

#%%
mask.shape

#%%
mask_train.unique()

#%%
print(torch.Tensor(image_train).dtype)


#%% Train Epoch
train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=device,
    verbose=True,
)

train_losses = []

for epoch in range(epochs):
    print(f"\nEpoch {epoch}")
    train_logs = train_epoch.run(train_dl)
    train_losses.append(train_logs['JaccardLoss'])
    #scheduler.step(loss)

#%% Visualize losses
sns.lineplot(x=range(len(train_losses)), y=train_losses).set(title='Train Loss')
plt.show()

#%%
model_name = f'{model_n}_{encoder}_{weights}_{loss.__name__}_lr{lr}_epochs{epochs}'
print(model_name)

#%%
# %%
torch.save(model, 
           f'path3/{model_name}.pth')
print('Model saved!')
# %%
test_model = torch.load(
    f'path3/{model_name}.pth',
    weights_only=False)

#%%
test_transform = v2.Compose([
                    v2.ToImage(), 
                    v2.Resize((256,256)),
])
# %%
test_ds = SMPDatasetTissuenet(
    image_dir="insert_path",
    mask_dir="insert_path", 
    transform=test_transform,
)

test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

#%%
test_epoch = smp.utils.train.ValidEpoch(
    model=test_model, 
    loss=loss, 
    metrics=metrics, 
    device=device,
)
valid_logs = test_epoch.run(test_dl)

#%%

# %%                            Visualize

test_data_vis = SMPDatasetTissuenet(
    image_dir="insert_path",
    mask_dir="insert_path", 
    transform=test_transform,
    )

n = np.random.choice(len(test_ds))

d_image, gt_mask = test_data_vis[n]
x_tensor = d_image.to(device)
pr_mask = x_tensor.float().unsqueeze(0)
pred_mask = test_model.predict(pr_mask)
pred_mask = pred_mask.squeeze().cpu().numpy().round()
d_image = d_image.cpu().numpy()
d_image = np.transpose(d_image, (1, 2, 0))
gt_mask = gt_mask.cpu().numpy()
gt_mask = np.transpose(gt_mask, (1, 2, 0))

plt.figure(figsize=(7, 5), dpi=250)
plt.subplot(3,1,1)
plt.imshow(d_image)
plt.title('Image')
plt.subplot(3,1,2)
plt.imshow(gt_mask)
plt.title('GT mask')
plt.subplot(3,1,3)
plt.imshow(pred_mask)
plt.title('Pred mask')
plt.tight_layout()


# %%
