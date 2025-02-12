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
