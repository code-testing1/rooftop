import cv2
import numpy as np
import os

# Adjust paths
image_path = "path_to/Marhara.tif"
mask_path = "path_to/mask.tif"

# 1. Read the image and mask with OpenCV
image = cv2.imread(image_path)                # May fail with large or multi-band TIF
mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

# 2. Check if reading was successful
if image is None:
    print("Image not read. Possibly a path or format issue.")
else:
    print("Image shape:", image.shape)

if mask is None:
    print("Mask not read. Possibly a path or format issue.")
else:
    print("Mask shape:", mask.shape)
    print("Mask unique values:", np.unique(mask))

# 3. Quick check of pixel values
if mask is not None:
    # For instance, visualize 200x200 top-left region
    sub_region = mask[0:200, 0:200]
    print("Sub-region unique values:", np.unique(sub_region))
