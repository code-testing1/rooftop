import rasterio
import numpy as np
import os
from rasterio.windows import Window

# Input file paths
image_path = "path/to/village_image.tif"
mask_path = "path/to/rcc_rooftop_mask.tif"

# Tile size
tile_size = 512
output_dir = "tiles/"

# Create output directories
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

# Read image and mask
with rasterio.open(image_path) as src_img, rasterio.open(mask_path) as src_mask:
    img_height, img_width = src_img.height, src_img.width

    # Loop over image with a step of tile_size
    for i in range(0, img_width, tile_size):
        for j in range(0, img_height, tile_size):
            # Define window size
            window_width = min(tile_size, img_width - i)
            window_height = min(tile_size, img_height - j)
            window = Window(i, j, window_width, window_height)

            # Read image tile
            img_tile = src_img.read(window=window)
            mask_tile = src_mask.read(1, window=window)

            # Create full-size (512x512) arrays and fill with padding (black pixels)
            padded_img = np.zeros((src_img.count, tile_size, tile_size), dtype=img_tile.dtype)
            padded_mask = np.zeros((tile_size, tile_size), dtype=mask_tile.dtype)

            # Insert the actual image and mask into the padded arrays
            padded_img[:, :window_height, :window_width] = img_tile
            padded_mask[:window_height, :window_width] = mask_tile

            # Save padded tiles
            img_tile_path = os.path.join(output_dir, "images", f"tile_{i}_{j}.tif")
            mask_tile_path = os.path.join(output_dir, "masks", f"tile_{i}_{j}.tif")

            with rasterio.open(
                img_tile_path, "w", driver="GTiff", height=tile_size, width=tile_size,
                count=src_img.count, dtype=src_img.dtypes[0], crs=src_img.crs, transform=src_img.window_transform(window)
            ) as dest:
                dest.write(padded_img)

            with rasterio.open(
                mask_tile_path, "w", driver="GTiff", height=tile_size, width=tile_size,
                count=1, dtype=src_mask.dtypes[0], crs=src_mask.crs, transform=src_mask.window_transform(window)
            ) as dest:
                dest.write(padded_mask, 1)

print("Tiling with padding complete!")

#########################################


import rasterio
import numpy as np
import os
from rasterio.windows import Window
import cv2  # For padding

# File paths
image_path = "path/to/village_image.tif"
mask_path = "path/to/rcc_rooftop_mask.tif"

# Tile size
tile_size = 512
output_dir = "tiles/"

# Create output directories
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "unlabeled"), exist_ok=True)  # For images without labels

# Read image
with rasterio.open(image_path) as src_img:
    img_height, img_width = src_img.height, src_img.width

    # Read mask if available
    has_labels = os.path.exists(mask_path)
    if has_labels:
        with rasterio.open(mask_path) as src_mask:
            mask_data = src_mask.read(1)

    # Loop over the large image
    for i in range(0, img_width, tile_size):
        for j in range(0, img_height, tile_size):
            # Define tile window
            window = Window(i, j, tile_size, tile_size)

            # Read image tile
            img_tile = src_img.read(window=window)

            # Check if mask exists
            if has_labels:
                mask_tile = mask_data[j:j+tile_size, i:i+tile_size]  # Extract mask tile
                non_zero_pixels = np.sum(mask_tile > 0)  # Count non-empty pixels
                tile_area = tile_size * tile_size
                real_content_ratio = non_zero_pixels / tile_area  # Calculate non-empty content ratio
            else:
                real_content_ratio = 1.0  # Assume full image content

            # **Skip tiles with <90% real content**
            if real_content_ratio < 0.90:
                continue  # Skip completely blank tiles

            # **Padding if needed**
            if img_tile.shape[1] < tile_size or img_tile.shape[2] < tile_size:
                pad_x = tile_size - img_tile.shape[2]
                pad_y = tile_size - img_tile.shape[1]
                img_tile = np.pad(img_tile, ((0, 0), (0, pad_y), (0, pad_x)), mode="constant")

            # Save image tile (for unlabeled dataset)
            img_tile_path = os.path.join(output_dir, "unlabeled", f"tile_{i}_{j}.tif")
            with rasterio.open(
                img_tile_path, "w", driver="GTiff", height=tile_size, width=tile_size,
                count=src_img.count, dtype=src_img.dtypes[0], crs=src_img.crs, transform=src_img.window_transform(window)
            ) as dest:
                dest.write(img_tile)

            # **Save labeled data only if mask exists**
            if has_labels:
                mask_tile_path = os.path.join(output_dir, "masks", f"tile_{i}_{j}.tif")
                with rasterio.open(
                    mask_tile_path, "w", driver="GTiff", height=tile_size, width=tile_size,
                    count=1, dtype=np.uint8, crs=src_img.crs, transform=src_img.window_transform(window)
                ) as dest:
                    dest.write(mask_tile, 1)

print("Tiling complete!")




###############################
import rasterio
import numpy as np
import os
from rasterio.windows import Window
import cv2

# File paths
image_path = "/mnt/data/412290926-bc8144e5-0747-4f2a-a03b-e038a4085767.png"
tile_size = 512
output_dir = "tiles/"

# Create output directories
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "unlabeled"), exist_ok=True)  # For images without labels

# Open the image
with rasterio.open(image_path) as src_img:
    img_height, img_width = src_img.height, src_img.width
    num_channels = src_img.count

    # Loop over the large image
    for i in range(0, img_width, tile_size):
        for j in range(0, img_height, tile_size):
            # Define window (tile region)
            window = Window(i, j, tile_size, tile_size)

            # Read image tile
            img_tile = src_img.read(window=window)

            # If image has 4 channels (RGBA), extract alpha channel (transparency)
            if num_channels == 4:
                alpha_channel = img_tile[3]  # Extract alpha channel
                non_transparent_pixels = np.sum(alpha_channel > 0)  # Count visible pixels
            else:
                non_transparent_pixels = tile_size * tile_size  # Assume all pixels are valid

            real_content_ratio = non_transparent_pixels / (tile_size * tile_size)

            # **Skip tiles with <90% real content**
            if real_content_ratio < 0.90:
                continue  # Skip empty/mostly transparent tiles

            # **Padding if needed**
            if img_tile.shape[1] < tile_size or img_tile.shape[2] < tile_size:
                pad_x = tile_size - img_tile.shape[2]
                pad_y = tile_size - img_tile.shape[1]
                img_tile = np.pad(img_tile, ((0, 0), (0, pad_y), (0, pad_x)), mode="constant")

            # Save tile
            img_tile_path = os.path.join(output_dir, "unlabeled", f"tile_{i}_{j}.tif")
            with rasterio.open(
                img_tile_path, "w", driver="GTiff", height=tile_size, width=tile_size,
                count=src_img.count, dtype=src_img.dtypes[0], crs=src_img.crs, transform=src_img.window_transform(window)
            ) as dest:
                dest.write(img_tile)

print("Tiling complete!")





