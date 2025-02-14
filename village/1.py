import rasterio
from rasterio.features import rasterize
import numpy as np

# Open the base image to get its metadata
image_path = "path_to_data/Marhara.tif"
with rasterio.open(image_path) as src:
    meta = src.meta.copy()

# Create an empty mask
mask = np.zeros((meta["height"], meta["width"]), dtype=np.uint8)

# Convert geometries to raster format
shapes_rooftop = [(geom, 1) for geom in rooftop_gdf.geometry]
shapes_vacant = [(geom, 2) for geom in vacant_gdf.geometry]

# Apply rasterization
mask = rasterize(shapes_rooftop + shapes_vacant, out_shape=(meta["height"], meta["width"]), transform=src.transform)

# Save the combined mask
mask_path = "path_to_data/combined_mask.tif"
with rasterio.open(mask_path, "w", **meta) as dst:
    dst.write(mask, 1)




