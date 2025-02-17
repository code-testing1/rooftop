import os
import numpy as np
import rasterio
from rasterio.windows import Window
import cv2
import random
from tqdm import tqdm

# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------
image_path = "path_to/Marhara.tif"
mask_path = "path_to/mask.tif"

output_dir = "patches_output"
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

PATCH_SIZE = 512
OVERLAP = PATCH_SIZE // 2  # 50% overlap => 256 step
BLACK_PATCH_KEEP_RATIO = 0.2

# -------------------------------------------------------------------
# Open image and mask with Rasterio
# -------------------------------------------------------------------
with rasterio.open(image_path) as src_img:
    # shape = (channels, height, width)
    img_height = src_img.height
    img_width = src_img.width
    img_bands = src_img.count
    
with rasterio.open(mask_path) as src_mask:
    # For the mask, typically 1 band
    mask_height = src_mask.height
    mask_width = src_mask.width
    mask_bands = src_mask.count

# Quick sanity check: the dimensions should match if you used the same transform
assert img_height == mask_height and img_width == mask_width, \
    "Image and mask dimensions do not match. Check your rasterization step."

patch_count = 0
black_patch_count = 0

# -------------------------------------------------------------------
# Sliding window over the full image
# -------------------------------------------------------------------
for y in tqdm(range(0, img_height - PATCH_SIZE, OVERLAP)):
    for x in range(0, img_width - PATCH_SIZE, OVERLAP):
        
        # Read image patch using a rasterio Window
        window = Window(x, y, PATCH_SIZE, PATCH_SIZE)
        
        with rasterio.open(image_path) as src_img:
            # shape => (bands, PATCH_SIZE, PATCH_SIZE)
            img_patch = src_img.read(window=window)
            
        with rasterio.open(mask_path) as src_mask:
            mask_patch = src_mask.read(1, window=window)  # single band
        
        # Convert image patch from (bands, H, W) to (H, W, bands) if you want to save with OpenCV
        # If your image is multi-band, you might have 3 or 4 channels
        img_patch = np.transpose(img_patch, (1, 2, 0))  # (H, W, bands)
        
        # If your image is more than 3 channels, you may need to drop extras or handle them carefully
        if img_bands > 3:
            # e.g., keep first 3 channels only
            img_patch = img_patch[:, :, :3]
        
        # Convert to 8-bit if needed (depends on your data type)
        # Typically, geospatial data might be uint16 or float. You can scale or cast if you prefer.
        img_patch = img_patch.astype(np.uint8)
        
        # Now mask_patch is (512, 512). Let's check for rooftops (value=2).
        unique_vals = np.unique(mask_patch)
        
        if 2 in unique_vals:  # Contains rooftops
            keep_patch = True
        elif np.all(mask_patch == 0):  # Fully black
            black_patch_count += 1
            keep_patch = (random.random() < BLACK_PATCH_KEEP_RATIO)  # Keep 20%
        else:
            # If there are other values (like 1 for grey), we skip
            keep_patch = False
        
        if keep_patch:
            # Save with OpenCV
            img_save_path = os.path.join(output_dir, "images", f"patch_{patch_count}.png")
            mask_save_path = os.path.join(output_dir, "masks", f"patch_{patch_count}.png")
            
            cv2.imwrite(img_save_path, img_patch)
            cv2.imwrite(mask_save_path, mask_patch)
            
            patch_count += 1

print(f"Total patches saved: {patch_count}")
print(f"Total black patches: {black_patch_count} (kept ~{int(black_patch_count * BLACK_PATCH_KEEP_RATIO)})")
