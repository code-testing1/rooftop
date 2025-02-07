import os
import cv2
import numpy as np
import shutil
from sklearn.model_selection import train_test_split

# Paths
image_dir = "path/to/output/images"
mask_dir = "path/to/output/masks"
train_image_dir = "path/to/train/images"
train_mask_dir = "path/to/train/masks"
test_image_dir = "path/to/test/images"
test_mask_dir = "path/to/test/masks"

# Ensure train/test directories exist
for dir in [train_image_dir, train_mask_dir, test_image_dir, test_mask_dir]:
    os.makedirs(dir, exist_ok=True)

# Get sorted list of images and masks
image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.tif'))])
mask_filenames = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.tif'))])

# Train-test split (80% train, 20% test)
train_images, test_images, train_masks, test_masks = train_test_split(
    image_filenames, mask_filenames, test_size=0.2, random_state=42
)

def process_and_save(image_name, mask_name, src_img_dir, src_mask_dir, dst_img_dir, dst_mask_dir):
    """ Load, normalize, and save images efficiently to avoid RAM issues. """
    
    img_path = os.path.join(src_img_dir, image_name)
    mask_path = os.path.join(src_mask_dir, mask_name)

    # Load image & mask one by one (no bulk loading)
    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Keep masks grayscale

    # Normalize image to range [0,1] then scale back to uint8 for saving
    image = (image.astype(np.float32) / 255.0 * 255).astype(np.uint8)

    # Save processed images
    cv2.imwrite(os.path.join(dst_img_dir, image_name), image)
    cv2.imwrite(os.path.join(dst_mask_dir, mask_name), mask)  # Masks unchanged

# Process train dataset (streaming approach)
for img_name, mask_name in zip(train_images, train_masks):
    process_and_save(img_name, mask_name, image_dir, mask_dir, train_image_dir, train_mask_dir)

# Process test dataset (streaming approach)
for img_name, mask_name in zip(test_images, test_masks):
    process_and_save(img_name, mask_name, image_dir, mask_dir, test_image_dir, test_mask_dir)

print("Preprocessing complete! Train and test datasets are ready.")
