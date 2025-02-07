import os
import cv2
import numpy as np

# Define paths
input_image_dir = "path/to/images"  # Folder containing large images
input_mask_dir = "path/to/masks"    # Folder containing corresponding masks
output_image_dir = "path/to/output/images"
output_mask_dir = "path/to/output/masks"

# Ensure output directories exist
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# Define patch size
PATCH_SIZE = 512

def split_image(image_path, mask_path, img_save_dir, mask_save_dir, base_name):
    # Read image and mask
    image = cv2.imread(image_path)  
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Assuming single-channel mask

    # Get image dimensions
    height, width, _ = image.shape

    patch_count = 0

    # Loop through image with step size of PATCH_SIZE
    for y in range(0, height, PATCH_SIZE):
        for x in range(0, width, PATCH_SIZE):
            # Extract patch
            img_patch = image[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            mask_patch = mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

            # Ensure the patch is exactly 512x512 (handle edge cases)
            if img_patch.shape[:2] != (PATCH_SIZE, PATCH_SIZE):
                continue  

            # Save patches
            img_patch_name = f"{base_name}_{patch_count}.png"
            mask_patch_name = f"{base_name}_{patch_count}.png"

            cv2.imwrite(os.path.join(img_save_dir, img_patch_name), img_patch)
            cv2.imwrite(os.path.join(mask_save_dir, mask_patch_name), mask_patch)

            patch_count += 1

# Process all images in the dataset
for filename in os.listdir(input_image_dir):
    if filename.endswith(('.jpg', '.png', '.tif')):  
        image_path = os.path.join(input_image_dir, filename)
        mask_path = os.path.join(input_mask_dir, filename)  # Assuming masks have the same name
        base_name = os.path.splitext(filename)[0]

        split_image(image_path, mask_path, output_image_dir, output_mask_dir, base_name)

print("Dataset splitting complete!")
