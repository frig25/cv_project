import os, datetime, glob
import cv2
import numpy as np
from osgeo import gdal
from tqdm import tqdm


def calculate_nhfd(red_edge_band, coastal_blue_band):
    """Compute the Non-Homogeneous Feature Difference (NHFD) index."""
    denominator = red_edge_band + coastal_blue_band
    denominator[denominator == 0] = 1e-10  # Prevent divide by zero
    return (red_edge_band - coastal_blue_band) / denominator


def save_index_as_tif(index, output_file):
    """Write a 2D float array to a single-band GeoTIFF file."""
    y_size, x_size = index.shape
    driver = gdal.GetDriverByName("GTiff")
    raster = driver.Create(output_file, x_size, y_size, 1, gdal.GDT_Float32)
    raster.GetRasterBand(1).WriteArray(index)
    raster.FlushCache()
    raster = None  # Close the dataset


def save_overlay_index_on_rgb(index_colored, rgb_img, mask, output_file):
    """Blend the color-mapped index onto the RGB image using a binary mask."""
    overlay = rgb_img.copy()
    cv2.copyTo(index_colored, mask, overlay)  # Copy only where mask is non-zero
    cv2.imwrite(output_file, overlay)


def ensure_dirs(paths):
    """Create output directories if they do not already exist."""
    for path in paths:
        os.makedirs(path, exist_ok=True)


def main():
    gdal.UseExceptions()

    # Mapping from band name to 1-based band position in the stacked raster
    band2idx = {
        "Blue-444" : 1,
        "Blue" : 2,
        "Red edge-705" : 7,
        "Red Edge" : 8,
        "Red edge-740" : 9
    }

    alpha = 0.50  # Threshold factor in [0,1] — controls model sensitivity
    color_map = cv2.COLORMAP_JET  # Color map used to visualize the NHFD index

    # ----- Input paths -----
    base_input = '../../data/udine/preprocessing'
    stacks_path = os.path.join(base_input, 'stacks')
    rgb_path = os.path.join(base_input, 'rgb')

    # ----- Output paths -----
    base_output = '../../data/udine/classification'
    mask_output = os.path.join(base_output, 'masks', f'alpha_{alpha}')
    output_dirs = {
        "tif": os.path.join(base_output, 'nhfd_tif'),         # Raw NHFD as GeoTIFF
        "rgb": os.path.join(base_output, 'nhfd_rgb'),         # Color-mapped NHFD as PNG
        "mask": os.path.join(mask_output, 'nhfd_pred_mask'),   # Binary prediction mask
        "overlay": os.path.join(mask_output, 'nhfd_overlay')   # NHFD overlay on RGB
    }

    ensure_dirs(output_dirs.values())

    # Collect and sort input file paths
    stacks = sorted(glob.glob(os.path.join(stacks_path, 'IMG_*')))
    rgbs = sorted(glob.glob(os.path.join(rgb_path, 'IMG_*')))

    print(f'Start classification with threshold={alpha}...')
    start_time = datetime.datetime.now()

    for stack_fp, rgb_fp in tqdm(zip(stacks, rgbs), total=len(stacks)):
        # Extract capture identifier from filename (first 8 characters)
        stack_id = os.path.basename(stack_fp)[:8]

        # Read Red Edge and Coastal Blue bands from the multispectral stack
        raster = gdal.Open(stack_fp)
        red_edge = raster.GetRasterBand(band2idx['Red Edge']).ReadAsArray().astype(np.float32)
        coastal_blue = raster.GetRasterBand(band2idx['Blue-444']).ReadAsArray().astype(np.float32)
        raster = None  # Close the dataset

        # Compute NHFD index and save as GeoTIFF
        nhfd = calculate_nhfd(red_edge, coastal_blue)
        save_index_as_tif(nhfd, os.path.join(output_dirs['tif'], f'{stack_id}_nhfd.tiff'))

        # Normalize NHFD to 8-bit and apply color map for visualization
        nhfd_uint8 = cv2.normalize(nhfd, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        nhfd_colored = cv2.applyColorMap(nhfd_uint8, color_map)

        # Save color-mapped NHFD as PNG
        cv2.imwrite(os.path.join(output_dirs['rgb'], f'{stack_id}_nhfd.png'), nhfd_colored)

        # Compute adaptive threshold based on the alpha factor
        min_val, max_val = np.min(nhfd), np.max(nhfd)
        threshold = min_val + abs(max_val - min_val) * alpha

        # Generate binary mask: pixels below threshold are classified as positive
        _, mask = cv2.threshold(nhfd, threshold, 255, cv2.THRESH_BINARY_INV)
        mask = mask.astype(np.uint8)

        # Save binary mask as PNG
        cv2.imwrite(os.path.join(output_dirs['mask'], f'{stack_id}_pred_mask.png'), mask)

        # Load corresponding RGB image and save the NHFD overlay
        rgb_img = cv2.imread(rgb_fp)
        save_overlay_index_on_rgb(nhfd_colored, rgb_img, mask,
                                  os.path.join(output_dirs['overlay'], f'{stack_id}_nhfd_overlay.png'))

    end_time = datetime.datetime.now()

    # Print execution summary
    total_time = (end_time - start_time).total_seconds()
    print(f"Completed in {end_time - start_time}")
    print(f"Rate: {len(stacks)/total_time:.2f} captures/sec")


if __name__ == "__main__":
    main()
