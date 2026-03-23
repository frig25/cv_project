import os, datetime, glob
import cv2
import numpy as np
from osgeo import gdal
from tqdm import tqdm

def calculate_nhfd(red_edge_band, coastal_blue_band):
    denominator = red_edge_band + coastal_blue_band
    denominator[denominator == 0] = 1e-10  # Prevent divide by zero
    return (red_edge_band - coastal_blue_band) / denominator

def save_index_as_tif(index, output_file):
    y_size, x_size = index.shape
    driver = gdal.GetDriverByName("GTiff")
    raster = driver.Create(output_file, x_size, y_size, 1, gdal.GDT_Float32)
    raster.GetRasterBand(1).WriteArray(index)
    raster.FlushCache()
    raster = None

def save_overlay_index_on_rgb(index_colored, rgb_img, mask, output_file): 
    overlay = rgb_img.copy()
    cv2.copyTo(index_colored, mask, overlay)
    cv2.imwrite(output_file, overlay)

def ensure_dirs(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)

def main():
    gdal.UseExceptions()

    band2idx = {
        "Blue-444" : 1, 
        "Blue" : 2,
        "Red edge-705" : 7,
        "Red Edge" : 8,
        "Red edge-740" : 9
    }

    alpha = 0.50 # must be in range [0,1]
    color_map = cv2.COLORMAP_JET

    # input
    base_input = '../../data/udine/preprocessing'
    stacks_path = os.path.join(base_input, 'stacks')
    rgb_path = os.path.join(base_input, 'rgb')

    # output
    base_output = '../../data/udine/classification'
    mask_output = os.path.join(base_output, 'masks', f'alpha_{alpha}')
    output_dirs = {
        "tif": os.path.join(base_output, 'nhfd_tif'),
        "rgb": os.path.join(base_output, 'nhfd_rgb'),
        "mask": os.path.join(mask_output, 'nhfd_pred_mask'),
        "overlay": os.path.join(mask_output,  'nhfd_overlay')   
    }

    ensure_dirs(output_dirs.values())

    stacks = sorted(glob.glob(os.path.join(stacks_path, 'IMG_*')))
    rgbs = sorted(glob.glob(os.path.join(rgb_path, 'IMG_*')))

    print(f'Start classification with threshold={alpha}...')
    start_time = datetime.datetime.now()

    for stack_fp, rgb_fp in tqdm(zip(stacks, rgbs), total=len(stacks)):
        stack_id = os.path.basename(stack_fp)[:8]

        raster = gdal.Open(stack_fp)
        red_edge = raster.GetRasterBand(band2idx['Red Edge']).ReadAsArray().astype(np.float32)
        coastal_blue = raster.GetRasterBand(band2idx['Blue-444']).ReadAsArray().astype(np.float32)
        raster = None

        nhfd = calculate_nhfd(red_edge, coastal_blue)

        save_index_as_tif(nhfd, os.path.join(output_dirs['tif'], f'{stack_id}_nhfd.tiff'))
        
        nhfd_uint8 = cv2.normalize(nhfd, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        nhfd_colored = cv2.applyColorMap(nhfd_uint8, color_map)

        # save index as png
        cv2.imwrite(os.path.join(output_dirs['rgb'], f'{stack_id}_nhfd.png'), nhfd_colored)

        min_val, max_val = np.min(nhfd), np.max(nhfd)
        threshold = min_val + abs(max_val - min_val) * alpha

        _, mask = cv2.threshold(nhfd, threshold, 255, cv2.THRESH_BINARY_INV)
        mask = mask.astype(np.uint8)

        # save mask as png
        cv2.imwrite(os.path.join(output_dirs['mask'], f'{stack_id}_pred_mask.png'), mask)

        rgb_img = cv2.imread(rgb_fp)
        save_overlay_index_on_rgb(nhfd_colored, rgb_img, mask,
                                  os.path.join(output_dirs['overlay'], f'{stack_id}_nhfd_overlay.png'))

    end_time = datetime.datetime.now()

    total_time = (end_time - start_time).total_seconds()
    print(f"Completed in {end_time - start_time}")
    print(f"Rate: {len(stacks)/total_time:.2f} captures/sec")

if __name__ == "__main__":
    main()
