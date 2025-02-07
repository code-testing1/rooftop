import os
import cv2

# Define paths
input_image_dir = "path/to/images"
input_mask_dir = "path/to/masks"
output_image_dir = "path/to/output/images"
output_mask_dir = "path/to/output/masks"

# Ensure output directories exist
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# Define new image size
FINAL_SIZE = 512

def split_and_resize(image_path, mask_path, img_save_dir, mask_save_dir, base_name):
    # Read image and mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Assuming a binary mask

    # Get original size (should be 10,000x10,000)
    height, width = image.shape[:2]

    # Split into 4 equal parts
    mid_x, mid_y = width // 2, height // 2

    parts = [
        ("top_left", image[0:mid_y, 0:mid_x], mask[0:mid_y, 0:mid_x]),
        ("top_right", image[0:mid_y, mid_x:width], mask[0:mid_y, mid_x:width]),
        ("bottom_left", image[mid_y:height, 0:mid_x], mask[mid_y:height, 0:mid_x]),
        ("bottom_right", image[mid_y:height, mid_x:width], mask[mid_y:height, mid_x:width])
    ]

    for part_name, img_part, mask_part in parts:
        # Resize to 512x512
        img_resized = cv2.resize(img_part, (FINAL_SIZE, FINAL_SIZE), interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(mask_part, (FINAL_SIZE, FINAL_SIZE), interpolation=cv2.INTER_NEAREST)

        # Save files
        img_name = f"{base_name}_{part_name}.png"
        mask_name = f"{base_name}_{part_name}.png"

        cv2.imwrite(os.path.join(img_save_dir, img_name), img_resized)
        cv2.imwrite(os.path.join(mask_save_dir, mask_name), mask_resized)

# Process all images
for filename in os.listdir(input_image_dir):
    if filename.endswith(('.jpg', '.png', '.tif')):  
        image_path = os.path.join(input_image_dir, filename)
        mask_path = os.path.join(input_mask_dir, filename)  # Assuming masks have the same name
        base_name = os.path.splitext(filename)[0]

        split_and_resize(image_path, mask_path, output_image_dir, output_mask_dir, base_name)

print("Dataset processing complete!")
