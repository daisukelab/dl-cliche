from .utils import *

import cv2
from multiprocessing import Pool

def _resize_worker(dest_folder, list_of_files, shape):
    results = []
    for source_imgpath in list_of_files:
        img = cv2.imread(str(source_imgpath))
        resized_img = cv2.resize(img, shape)
        outfile = str(Path(dest_folder)/Path(source_imgpath).name)
        cv2.imwrite(outfile, resized_img)
        results.append((outfile, (img.shape[1], img.shape[0])))
    return results

def resize_image_files(dest_folder, source_files, shape=(224, 224), num_threads=4, skip_if_any_there=False):
    """Make resized copy of listed images in parallel processes.

    Arguments:
        dest_folder: Destination folder to make copies.
        source_files: Source image files.
        shape: (Width, Depth) shape of copies.
        num_threads: Number of parallel workers.
        skip_if_any_there: If True, skip processing processing if any file have already been done.

    Returns:
        List of image info (filename, original size) tuples, or None if skipped.
        ex)
        ```python
        [('tmp/8d6ed7c786dcbc93.jpg', (1024, 508)),
         ('tmp/8d6ee9921e4aeb18.jpg', (891, 1024)),
         ('tmp/8d6f00feedb09efa.jpg', (1024, 683))]
        ```
    """
    if skip_if_any_there:
        if (Path(dest_folder)/Path(source_files[0]).name).exists():
            return None
    with Pool(num_threads) as p:
        ns = len(source_files) // num_threads
        ensure_folder(dest_folder)
        returns = p.starmap(_resize_worker, [(dest_folder, source_files[ns*i:ns*(i+1)] if i < num_threads-1
                                                     else source_files[ns*(num_threads-1):], shape)
                                             for i in range(num_threads)])
        return flatten_list(returns)

def load_rgb_image(filename):
    """Load image file and make sure that format is RGB."""
    img = cv2.imread(str(filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def render_bbox(img, bbox, class_name, color=(255, 0, 0), text_color=(255, 255, 255), thickness=1):
    """Object Detection Helper: Render single bounding box with class name on top of image.
    Thanks to https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/vis.py
    """
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.5 * text_height)), (x_min + text_width, y_min), color, -1)
    cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                text_color, thickness=2, lineType=cv2.LINE_AA)
    return img

def convert_mono_to_jpg(fromfile, tofile):
    """Convert monochrome image to color jpeg format.
    Linear copy to RGB channels. 
    https://docs.opencv.org/3.1.0/de/d25/imgproc_color_conversions.html

    Args:
        fromfile: float png mono image
        tofile: RGB color jpeg image.
    """
    img = np.array(Image.open(fromfile)) # TODO Fix this to cv2.imread
    img = img - np.min(img)
    img = img / (np.max(img) + 1e-4)
    img = (img * 255).astype(np.uint8) # [0,1) float to [0,255] uint8
    img = np.repeat(img[..., np.newaxis], 3, axis=-1) # mono to RGB color
    img = Image.fromarray(img)
    tofile = Path(tofile)
    img.save(tofile.with_suffix('.jpg'), 'JPEG', quality=100)

