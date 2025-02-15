import os
import cv2
import numpy as np
from tqdm import tqdm

# Paths
image_path = "path_to_data/Marhara.tif"
mask_path = "path_to_data/combined_mask_fixed.tif"
output_image_dir = "path_to_data/patches/images"
output_mask_dir = "path_to_data/patches/masks"

# Create output directories
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# Read image and mask
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)  # Read mask as single-channel

# Constants
LARGE_PATCH_SIZE = 1024  # First extract 1024x1024
FINAL_PATCH_SIZE = 512  # Then resize to 512x512
STRIDE = 512  # Overlapping by 50%

patch_id = 0
kept_patches = 0

for y in tqdm(range(0, image.shape[0] - LARGE_PATCH_SIZE, STRIDE)):
    for x in range(0, image.shape[1] - LARGE_PATCH_SIZE, STRIDE):
        # Extract larger patches
        img_patch = image[y:y + LARGE_PATCH_SIZE, x:x + LARGE_PATCH_SIZE]
        mask_patch = mask[y:y + LARGE_PATCH_SIZE, x:x + LARGE_PATCH_SIZE]

        # **Filter Conditions**
        unique_vals = np.unique(mask_patch)

        # Keep patches that contain rooftops (2)
        if 2 in unique_vals:
            keep_patch = True
        # If no rooftops but only vacant space (1), discard some of them
        elif 1 in unique_vals and np.count_nonzero(mask_patch == 1) > 0.8 * (LARGE_PATCH_SIZE ** 2):
            keep_patch = np.random.rand() < 0.3  # Keep 30% of excessive grey patches
        else:
            keep_patch = False  # Discard black or irrelevant patches

        if keep_patch:
            kept_patches += 1

            # Resize to 512x512
            img_patch_resized = cv2.resize(img_patch, (FINAL_PATCH_SIZE, FINAL_PATCH_SIZE), interpolation=cv2.INTER_CUBIC)
            mask_patch_resized = cv2.resize(mask_patch, (FINAL_PATCH_SIZE, FINAL_PATCH_SIZE), interpolation=cv2.NEAREST)

            # Save patches
            cv2.imwrite(os.path.join(output_image_dir, f"image_{patch_id}.png"), img_patch_resized)
            cv2.imwrite(os.path.join(output_mask_dir, f"mask_{patch_id}.png"), mask_patch_resized)
            patch_id += 1

print(f"Total patches extracted: {patch_id}, Kept patches: {kept_patches}")





