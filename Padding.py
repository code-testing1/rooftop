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
