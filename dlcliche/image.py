from .utils import *

import cv2
import tqdm
from PIL import Image
from multiprocessing import Pool

def resize_image(dest_folder, filename, shape):
    """Resize and save copy of image file to destination folder."""
    img = cv2.imread(str(filename))
    if shape is not None:
        img = cv2.resize(img, shape)
    outfile = str(Path(dest_folder)/Path(filename).name)
    cv2.imwrite(outfile, img)
    return outfile, (img.shape[1], img.shape[0]) # original size

def _resize_image_worker(args):
    return resize_image(args[0], args[1], args[2])

def resize_image_files(dest_folder, source_files, shape=(224, 224), num_threads=8, skip_if_any_there=False):
    """Make resized copy of listed images in parallel processes.

    Arguments:
        dest_folder: Destination folder to make copies.
        source_files: Source image files.
        shape: (Width, Depth) shape of copies. None will NOT resize and makes dead copy.
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
    # Create destination folder if needed
    ensure_folder(dest_folder)
    # Do resize
    with Pool(num_threads) as p:
        args = [[dest_folder, f, shape] for f in source_files]
        returns = list(tqdm.tqdm(p.imap(_resize_image_worker, args), total=len(args)))
    return returns

def _get_shape_worker(filename):
    return Image.open(filename).size # Image.open() is much faster than cv2.imread()

def read_file_shapes(files, num_threads=8):
    """Read shape of files in parallel."""
    with Pool(num_threads) as p:
        shapes = list(tqdm.tqdm(p.imap(_get_shape_worker, files), total=len(files)))
    return np.array(shapes)

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

# Borrowing from fast.ai

from matplotlib import patches, patheffects

def subplot_matrix(rows, columns, figsize=(12, 12)):
    """Subplot utility for drawing matrix of images.
    # Usage
    Following will show images in 2x3 matrix.
    ```python
    axes = subplot_matrix(2, 3)
    for img, ax in zip(images, axes):
        show_image(img, ax=ax)
    ```
    """
    fig, axes = plt.subplots(rows, columns, figsize=figsize)
    return list(axes.flat)

def show_image(img, figsize=None, ax=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax

def _draw_outline(o, lw):
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])

def ax_draw_rect(ax, b):
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor='white', lw=2))
    _draw_outline(patch, 4)

def ax_draw_text(ax, xy, txt, sz=14):
    text = ax.text(*xy, txt,
        verticalalignment='top', color='white', fontsize=sz, weight='bold')
    _draw_outline(text, 1)

def ax_draw_bbox(ax, bbox, class_name):
    """Object Detection Helper: Draw single bounding box with class name on top of image."""
    ax_draw_rect(ax, bbox)
    ax_draw_text(ax, bbox[:2], class_name)
