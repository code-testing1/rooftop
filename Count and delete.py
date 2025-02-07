import os
import cv2
import numpy as np

# Path to the directory containing masks
mask_dir = "path/to/output/masks"

# Initialize counter
total_masks = 0
blank_masks = 0

for filename in os.listdir(mask_dir):
    if filename.endswith(('.png', '.jpg', '.tif')):
        mask_path = os.path.join(mask_dir, filename)

        # Read mask as grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        total_masks += 1

        # Check if mask is completely black (no white pixels)
        if np.max(mask) == 0:
            blank_masks += 1

# Print results
print(f"Total Masks: {total_masks}")
print(f"Blank Masks: {blank_masks}")
print(f"Percentage of Blank Masks: {blank_masks / total_masks * 100:.2f}%")



####################################




import os
import cv2
import numpy as np
import random

# Paths
mask_dir = "path/to/output/masks"
image_dir = "path/to/output/images"

# Percentage of blank masks to delete
DELETE_RATIO = 0.7  # 70%

# List to store blank mask filenames
blank_masks = []

for filename in os.listdir(mask_dir):
    if filename.endswith(('.png', '.jpg', '.tif')):
        mask_path = os.path.join(mask_dir, filename)
        
        # Read mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Check if mask is completely blank
        if np.max(mask) == 0:
            blank_masks.append(filename)

# Decide how many to delete
num_to_delete = int(len(blank_masks) * DELETE_RATIO)
blank_masks_to_delete = random.sample(blank_masks, num_to_delete)

# Delete selected blank masks and corresponding images
for filename in blank_masks_to_delete:
    mask_path = os.path.join(mask_dir, filename)
    image_path = os.path.join(image_dir, filename)  # Assuming same name

    if os.path.exists(mask_path):
        os.remove(mask_path)

    if os.path.exists(image_path):
        os.remove(image_path)

print(f"Deleted {num_to_delete} blank masks and their images.")


