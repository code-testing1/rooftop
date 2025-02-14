import geopandas as gpd
import rasterio

# Load shapefiles
rooftop_gdf = gpd.read_file("path_to_data/Potential_Area.shp")
vacant_gdf = gpd.read_file("path_to_data/Vacant_Space.shp")

# Open image
image_path = "path_to_data/Marhara.tif"
with rasterio.open(image_path) as src:
    image_crs = src.crs

print("Rooftop CRS:", rooftop_gdf.crs)
print("Vacant Space CRS:", vacant_gdf.crs)
print("Image CRS:", image_crs)
