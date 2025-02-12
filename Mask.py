from skimage.util import view_as_blocks

# Load the image and mask
image = cv2.imread("path_to_village_image.png", cv2.IMREAD_GRAYSCALE)
mask = cv2.imread("rooftop_mask.png", cv2.IMREAD_GRAYSCALE)

# Ensure they have the same shape
assert image.shape == mask.shape, "Image and mask size mismatch!"

# Define patch size
patch_size = 512

# Extract patches
image_patches = view_as_blocks(image, block_shape=(patch_size, patch_size))
mask_patches = view_as_blocks(mask, block_shape=(patch_size, patch_size))

# Save each patch
for i in range(image_patches.shape[0]):
    for j in range(image_patches.shape[1]):
        cv2.imwrite(f"dataset/images/img_{i}_{j}.png", image_patches[i, j])
        cv2.imwrite(f"dataset/masks/mask_{i}_{j}.png", mask_patches[i, j])
