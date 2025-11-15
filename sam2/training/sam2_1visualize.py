import os
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import cv2
from PIL import Image
from pycocotools.mask import decode  # For decoding COCO RLE
from skimage.transform import resize  # For resizing masks with nearest neighbor
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

os.chdir()
# --- 1. Choose number of points and Load Dataset Configuration from YAML ---

Foreground_points = 20
Background_points = 20

yaml_config_path = "sam2/configs/sam2.1_training/sam_train_val_json_win.yaml"
with open(yaml_config_path, "r") as f:
    config = yaml.safe_load(f)

dataset_conf = config.get("dataset", {})
img_folder    = os.path.abspath(dataset_conf.get("img_folder", ""))
gt_folder     = os.path.abspath(dataset_conf.get("gt_folder", ""))
val_list_path = os.path.abspath(dataset_conf.get("val_list", ""))

print("Image folder:", img_folder)
print("Ground truth folder:", gt_folder)
print("Validation list:", val_list_path)

# --- 2. Read Validation List & Select a Sample ---
with open(val_list_path, "r") as f:
    val_files = [line.strip() for line in f if line.strip()]
if not val_files:
    raise ValueError("Validation list is empty!")

sample_filename = random.choice(val_files)
print("Selected sample:", sample_filename)

def find_file(folder, base_name, extensions):
    for ext in extensions:
        candidate = os.path.join(folder, base_name + ext)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"No file found for {base_name} in {folder} with extensions {extensions}")

sample_img_path  = find_file(img_folder, sample_filename, [".jpg", ".jpeg"])
sample_mask_path = find_file(gt_folder, sample_filename, [".json"])
print("Sample image path:", sample_img_path)
print("Sample mask path:", sample_mask_path)

# --- 3. Load Image & Decode COCO RLE Ground Truth Mask ---
opened_image = np.array(Image.open(sample_img_path).convert("RGB"))

with open(sample_mask_path, "r") as f:
    mask_data = json.load(f)
print("Mask JSON keys:", list(mask_data.keys()))
# Expected keys: ['info', 'licenses', 'images', 'annotations']

annotations = mask_data.get("annotations", [])
if not annotations:
    raise ValueError("No annotations found in mask JSON!")
rle_data = annotations[0].get("segmentation", None)
if rle_data is None:
    raise ValueError("No RLE segmentation found in the annotation!")
height, width = rle_data["size"]
rle_counts = rle_data["counts"]
ground_truth = decode({"size": [height, width], "counts": rle_counts})
ground_truth = (ground_truth > 0).astype(np.uint8) * 255  # Binary mask: foreground = 255
print(f"Decoded ground truth mask shape: {ground_truth.shape}")

# --- 4. Resize Image and Ground Truth while Maintaining Aspect Ratio ---
max_dimension = 1024  # Maximum dimension for width or height
h, w = opened_image.shape[:2]
scale = min(max_dimension / w, max_dimension / h)
target_h, target_w = int(h * scale), int(w * scale)
print(f"Resizing scale: {scale:.3f}, target size: ({target_w}, {target_h})")

# Resize image using cv2 (linear interpolation)
resized_image = cv2.resize(opened_image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
print("Processed (Resized) image shape:", resized_image.shape)

# Resize ground truth mask using skimage's resize 
original_dtype = ground_truth.dtype
mask_resized = resize(ground_truth, (target_h, target_w), order=0,
                      preserve_range=True, anti_aliasing=False)
mask_resized = np.round(mask_resized).astype(original_dtype)
print("Resized ground truth mask shape:", mask_resized.shape)

# --- 5. Sample Foreground and Background Points for Predictor ---
def get_points_for_inference(mask, num_points_fg, num_points_bg):
    """Samples foreground (mask > 0) and background (mask == 0) points."""
    points, labels = [], []
    #fg_coords = np.argwhere(mask > 0)
    #if len(fg_coords) > 0:
    #    num_samples = min(len(fg_coords), num_points_fg)
    #    sampled_indices = np.random.choice(len(fg_coords), num_samples, replace=False)
    #    for idx in sampled_indices:
    #        y, x = fg_coords[idx]
    #        points.append([float(x), float(y)])  # (x, y)
    #        labels.append(1)
    #bg_coords = np.argwhere(mask == 0)
    #if len(bg_coords) > 0:
    #    num_samples = min(len(bg_coords), num_points_bg)
    #    sampled_indices = np.random.choice(len(bg_coords), num_samples, replace=False)
    #    for idx in sampled_indices:
    #        y, x = bg_coords[idx]
    #        points.append([float(x), float(y)])
    #        labels.append(0)
    fg_coords = np.argwhere(mask > 0)
    bg_coords = np.argwhere(mask == 0)

    if len(fg_coords) > 0:
        fg_indices = np.random.choice(len(fg_coords), size=min(len(fg_coords),
                                                num_points_fg), replace=False)
        # Note swap from (y, x) to (x, y) for SAM predictor
        points.extend(fg_coords[fg_indices][:, ::-1])
        labels.extend([1] * len(fg_indices))

    if len(bg_coords) > 0:
        bg_indices = np.random.choice(len(bg_coords), size=min(len(bg_coords),
                                                num_points_bg), replace=False)
        points.extend(bg_coords[bg_indices][:, ::-1])
        labels.extend([0] * len(bg_indices))

    if not points:
        return None, None
    return np.array(points, dtype=np.float32), np.array(labels, dtype=np.int64)

input_points, input_labels = get_points_for_inference(mask_resized, num_points_fg=Foreground_points, 
                                                      num_points_bg=Background_points)
if input_points is None:
    raise ValueError("No points sampled for inference!")

# Print counts
num_fg_points = np.sum(input_labels == 1)
num_bg_points = np.sum(input_labels == 0)

print(f"Foreground points sampled: {num_fg_points}")
print(f"Background points sampled: {num_bg_points}")

# --- 6. Load SAM-2.1 Model & Run Point-Based Predictions ---
model_cfg       = "configs/sam2.1/sam2.1_hiera_b+.yaml"
checkpoint_path = "sam2_logs/epochs_1000_LR_0.01_prob_to_use_pt_input_for_train_1.0_prob_to_sample_from_gt_for_train_1.0/checkpoints/checkpoint.pt"
sam2_model = build_sam2(model_cfg, checkpoint_path, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)

# Perform inference using point prompts.
with torch.no_grad():
    predictor.set_image(resized_image)  # Set the resized image
    masks_pred, scores_pred, logits_pred = predictor.predict(
        point_coords=input_points,  # Sampled points
        point_labels=input_labels,  # Labels (foreground/background)
        multimask_output=True  # Get multiple masks per prompt if model supports
    )

# Select the best mask (highest score across prompts)
best_mask_idx = np.argmax(scores_pred)
pred_mask = masks_pred[best_mask_idx]  # Boolean mask
pred_mask_vis = (pred_mask.astype(np.uint8)) * 255  # Convert to uint8 format

# --- 7. Count Instances in Predicted Mask ---
def count_connected_instances(mask):
    """Counts connected components in a binary mask, excluding background."""
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    num_labels, _ = cv2.connectedComponents(mask_uint8)
    return num_labels - 1 if num_labels > 0 else 0
gt_instance_count = count_connected_instances(mask_resized)
pred_instance_count = count_connected_instances(pred_mask)
print(f"Predicted instances: {pred_instance_count}")

# --- 8. Visualization ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(opened_image)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(mask_resized, cmap="gray")
axes[1].set_title(f"Ground Truth Mask ({gt_instance_count} instances)")
axes[1].axis("off")

axes[2].imshow(pred_mask_vis, cmap="gray")
axes[2].set_title(f"Predicted Mask ({pred_instance_count} instances)")
axes[2].axis("off")

plt.tight_layout()
plt.show()