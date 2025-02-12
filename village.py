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

