import os
import cv2
import numpy as np
import random
from tqdm import tqdm

# Paths
image_path = "path_to/Marhara.tif"  # Large village image
mask_path = "path_to/mask.tif"  # Labeled mask
output_dir = "patches_output"  # Folder to save patches

# Patch size and overlap settings
PATCH_SIZE = 512
OVERLAP = PATCH_SIZE // 2  # 50% overlap
BLACK_PATCH_KEEP_RATIO = 0.2  # Keep 20% of fully black patches

# Create output directories
os.makedirs(f"{output_dir}/images", exist_ok=True)
os.makedirs(f"{output_dir}/masks", exist_ok=True)

# Load the large image and mask
image = cv2.imread(image_path)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

H, W = mask.shape  # Get original dimensions
patch_count = 0
black_patch_count = 0

# Sliding window approach
for y in tqdm(range(0, H - PATCH_SIZE, OVERLAP)):
    for x in range(0, W - PATCH_SIZE, OVERLAP):
        # Extract patches
        img_patch = image[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
        mask_patch = mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

        # Check patch content
        unique_values = np.unique(mask_patch)

        if 2 in unique_values:  # Keep patches with rooftops
            keep_patch = True
        elif np.all(mask_patch == 0):  # Fully black patches
            black_patch_count += 1
            keep_patch = random.random() < BLACK_PATCH_KEEP_RATIO  # Keep 20%
        else:
            keep_patch = False  # Ignore patches with grey (1)

        # Save only valid patches
        if keep_patch:
            cv2.imwrite(f"{output_dir}/images/patch_{patch_count}.png", img_patch)
            cv2.imwrite(f"{output_dir}/masks/patch_{patch_count}.png", mask_patch)
            patch_count += 1

print(f"Total patches saved: {patch_count}")
print(f"Total black patches discarded: {black_patch_count - int(black_patch_count * BLACK_PATCH_KEEP_RATIO)}")
