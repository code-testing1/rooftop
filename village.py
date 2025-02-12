import rasterio
from rasterio.features import rasterize
import numpy as np

# Load the satellite/village image
image_path = "path/to/village_image.tif"  # Replace with the actual image file if available
with rasterio.open(image_path) as src:
    img = src.read()
    transform = src.transform
    out_shape = (src.height, src.width)

# Check image shape
print("Image Shape:", img.shape)



from shapely.geometry import mapping

# Convert geometry to raster format
rcc_mask = rasterize(
    [(mapping(geom), 1) for geom in gdf.geometry],  # Convert each polygon to a tuple
    out_shape=out_shape, 
    transform=transform, 
    fill=0, 
    all_touched=True, 
    dtype=np.uint8
)

# Check mask shape
print("Mask Shape:", rcc_mask.shape)



from PIL import Image

# Convert mask to an image
mask_image = Image.fromarray(rcc_mask * 255)  # Scale to 0-255 for visualization
mask_image.save("rcc_rooftop_mask.png")


№######################
import rasterio
import numpy as np
import os
from rasterio.windows import Window

# Input file paths
image_path = "path/to/village_image.tif"
mask_path = "path/to/rcc_rooftop_mask.tif"  # Use your generated mask

# Tile size
tile_size = 512  # Adjust based on GPU/memory constraints
output_dir = "tiles/"

# Create output directories
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

# Read image and mask
with rasterio.open(image_path) as src_img, rasterio.open(mask_path) as src_mask:
    img_height, img_width = src_img.height, src_img.width

    # Loop over the large image in a grid pattern
    for i in range(0, img_width, tile_size):
        for j in range(0, img_height, tile_size):
            # Define window (tile region)
            window = Window(i, j, tile_size, tile_size)

            # Read image tile
            img_tile = src_img.read(window=window)
            mask_tile = src_mask.read(1, window=window)  # Read single-channel mask

            # Handle curved edges (skip empty/invalid tiles)
            if np.sum(mask_tile) == 0:  # Skip tiles without labels
                continue

            # Save tiles
            img_tile_path = os.path.join(output_dir, "images", f"tile_{i}_{j}.tif")
            mask_tile_path = os.path.join(output_dir, "masks", f"tile_{i}_{j}.tif")

            with rasterio.open(
                img_tile_path, "w", driver="GTiff", height=tile_size, width=tile_size,
                count=src_img.count, dtype=src_img.dtypes[0], crs=src_img.crs, transform=src_img.window_transform(window)
            ) as dest:
                dest.write(img_tile)

            with rasterio.open(
                mask_tile_path, "w", driver="GTiff", height=tile_size, width=tile_size,
                count=1, dtype=src_mask.dtypes[0], crs=src_mask.crs, transform=src_mask.window_transform(window)
            ) as dest:
                dest.write(mask_tile, 1)

print("Tiling complete!")





