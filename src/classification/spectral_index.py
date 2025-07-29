import cv2, os, datetime, glob
import numpy as np
from matplotlib import pyplot as plt
from osgeo import gdal


def calculate_wvnhfd(red_edge_band, coastal_blue_band):
    return (red_edge_band - coastal_blue_band) / (red_edge_band + coastal_blue_band)

def save_index_as_tif(spectral_index, output_path, file_name):
    raster_y_size, raster_x_size = spectral_index.shape
    # Save spectral index as a new raster
    driver = gdal.GetDriverByName("GTiff")
    raster = driver.Create(os.path.join(output_path, file_name) , raster_x_size, raster_y_size, 1, gdal.GDT_Float32)
    # Write spectral index array to raster band
    raster.GetRasterBand(1).WriteArray(spectral_index)
    raster.FlushCache()
    aster = None  # Close file

def save_index_as_rgb(spectral_index, cmap, output_path, file_name):
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(spectral_index, cmap=cmap)
    ax.axis("off")
    fig.savefig(os.path.join(output_path, file_name), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def plot_overlay(rgb_img, masked_sindex, figsize=None, cmap=None, alpha=None, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgb_img)
    ax.imshow(masked_sindex, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
    ax.axis("off")
    return fig, ax

def save_index_over_rgb(spectral_index, rgb_file, threshold_percent, cmap, output_path, file_name):
    rgb_img = cv2.imread(rgb_file)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

    threshold = np.min(spectral_index) + abs(np.min(spectral_index) - np.max(spectral_index)) * threshold_percent
    masked_sindex = np.ma.masked_where(spectral_index > threshold, spectral_index)

    # Plot and save overlay RGB image
    fig, ax = plot_overlay(rgb_img, masked_sindex, figsize=(15, 15), cmap=cmap, alpha=1, vmin=np.min(spectral_index), vmax=np.max(spectral_index))
    fig.savefig(os.path.join(output_path, file_name), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def main():
    threshold_percentage = 0.5 # 0.565

    stacks_path = './data/input/stacks'
    rgb_path = './data/input/rgb'
    wvnhfd_tif_path = './data/output/wvnhfd_tif'
    wvnhfd_rgb_path = './data/output/wvnhfd_rgb'
    wvnhfd_overlay_path = './data/output/wvnhfd_overlay_' + str(threshold_percentage)

    for f in [wvnhfd_tif_path, wvnhfd_rgb_path, wvnhfd_overlay_path]:
        if not os.path.exists(f):
            os.mkdir(f)


    stacks = sorted(glob.glob(os.path.join(stacks_path,'IMG_*')))
    rgbs = sorted(glob.glob(os.path.join(rgb_path,'IMG_*')))

    start_time = datetime.datetime.now()

    for s, r in zip(stacks, rgbs):
        raster = gdal.Open(s)
        coastal_blue_band = raster.GetRasterBand(1).ReadAsArray().astype(np.float32) # (Blue-444: 1, Blue: 2)
        red_edge_band = raster.GetRasterBand(8).ReadAsArray().astype(np.float32) # (Red edge-705: 7, Red Edge: 8, Red edge-740: 9)

        wvnhfd = calculate_wvnhfd(red_edge_band, coastal_blue_band)
        
        stack_id = s[len(stacks_path) + 1:len(stacks_path) + 9]

        save_index_as_tif(wvnhfd, wvnhfd_tif_path, stack_id + '_wvnhfd.tiff')
        
        save_index_as_rgb(wvnhfd, 'jet', wvnhfd_rgb_path, stack_id + '_wvnhfd.png')
        
        save_index_over_rgb(wvnhfd, r, threshold_percentage, 'jet', wvnhfd_overlay_path, stack_id + '_wvnhfd_overlay.png')

    end_time = datetime.datetime.now()

    print("Saving time: {}".format(end_time-start_time))
    print("Processing rate: {:.2f} captures per second".format(float(len(stacks))/float((end_time-start_time).total_seconds())))


if __name__ == "__main__":
    main()
