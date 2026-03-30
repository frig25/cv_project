import xml.etree.ElementTree as ET
import numpy as np
import cv2
import os


def decode_rle_counts(counts, width, height, start_with_background=True):
    """
    Decode CVAT/COCO-style RLE counts into a binary mask.
    counts: list of ints whose sum == width*height
    If start_with_background=True, the first count is background (0),
    then foreground (255), then background, and so on.
    """
    if sum(counts) != width * height:
        raise ValueError(f"RLE counts sum mismatch: got {sum(counts)}, expected {width*height}")

    mask_flat = np.zeros(width * height, dtype=np.uint8)
    pos = 0
    is_fore = not start_with_background  # First run is background if flag is True
    for c in counts:
        if is_fore and c > 0:
            mask_flat[pos:pos + c] = 255  # Mark foreground pixels
        pos += c
        is_fore = not is_fore  # Alternate between background and foreground
    return mask_flat.reshape((height, width))


def main():
    # ----- Input path -----
    annotations_path = '../../data/udine/preprocessing/annotations.xml'

    # ----- Output paths -----
    rgb_path = '../../data/udine/preprocessing/rgb'
    mask_path = "../../data/udine/preprocessing/gt_mask"
    overlay_path = '../../data/udine/preprocessing/gt_overlay'

    os.makedirs(mask_path, exist_ok=True)
    os.makedirs(overlay_path, exist_ok=True)

    # Parse the CVAT XML annotation file
    tree = ET.parse(annotations_path)
    root = tree.getroot()

    # Iterate over each annotated image entry
    for image_tag in root.findall('image'):
        img_name = image_tag.get('name')
        img_id = img_name[:8]  # Extract capture identifier from filename
        img_w = int(image_tag.get('width'))
        img_h = int(image_tag.get('height'))

        # Initialize an empty full-size mask for the entire image
        full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        mask_tag = image_tag.find('mask')
        if mask_tag is None:
            print(f'{img_id} mask not found')
            continue

        # Read RLE attributes: encoded mask and its bounding box within the image
        rle_str = mask_tag.get('rle')
        left = int(mask_tag.get('left', 0))
        top = int(mask_tag.get('top', 0))
        w = int(mask_tag.get('width'))
        h = int(mask_tag.get('height'))

        # Decode the RLE string into a binary crop-sized mask
        counts = [int(x.strip()) for x in rle_str.split(',') if x.strip()]
        crop_mask = decode_rle_counts(counts, w, h, start_with_background=True)

        # Merge the crop mask into the full-size mask at the correct position
        full_mask[top:top+h, left:left+w] = np.maximum(
            full_mask[top:top+h, left:left+w],
            crop_mask
        )

        # Save the ground truth binary mask
        cv2.imwrite(os.path.join(mask_path, f'{img_id}_gt_mask.png'), full_mask)

        # Load the RGB image and highlight masked pixels in red for visualization
        rgb = cv2.imread(os.path.join(rgb_path, img_name))
        rgb[full_mask == 255] = (0, 0, 255)  # Red in BGR format
        cv2.imwrite(os.path.join(overlay_path, f'{img_id}_gt_overlay.png'), rgb)

        print(f'{img_id} mask exported')


if __name__ == "__main__":
    main()
