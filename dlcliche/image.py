from .utils import *
from .notebook import running_in_notebook

import cv2
import tqdm
import math
import random
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
    if running_in_notebook:  # Workaround: not using pool on notebook
        returns = []
        for f in tqdm.tqdm(source_files, total=len(source_files)):
            returns.append(resize_image(dest_folder, f, shape))
    else:
        with Pool(num_threads) as p:
            args = [[dest_folder, f, shape] for f in source_files]
            returns = list(tqdm.tqdm(p.imap(_resize_image_worker, args), total=len(args)))
    return returns

def _get_shape_worker(filename):
    return Image.open(filename).size # Image.open() is much faster than cv2.imread()

def read_file_shapes(files, num_threads=8):
    """Read shape of files in parallel."""
    if running_in_notebook:  # Workaround: not using pool on notebook
        shapes = []
        for f in tqdm.tqdm(files, total=len(files)):
            shapes.append(_get_shape_worker(f))
    else:
        with Pool(num_threads) as p:
            shapes = list(tqdm.tqdm(p.imap(_get_shape_worker, files), total=len(files)))
    return np.array(shapes)

def load_rgb_image(filename):
    """Load image file and make sure that format is RGB."""
    img = cv2.imread(str(filename))
    if img is None:
        raise ValueError(f'Failed to load {filename}.')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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


def pil_crop(pil_img, crop_size, random_crop=True):
    """Crop PIL image, randomly or from its center."""

    w, h = pil_img.size
    if is_array_like(crop_size):
        crop_size_w, crop_size_h = crop_size
    else:
        crop_size_w, crop_size_h = crop_size, crop_size

    if random_crop:
        x = random.randint(0, np.maximum(0, w - crop_size_w))
        y = random.randint(0, np.maximum(0, h - crop_size_h))
    else:
        x = (w - crop_size_w) // 2
        y = (h - crop_size_h) // 2

    return pil_img.crop((x, y, x+crop_size_w, y+crop_size_h))


def pil_translate_fill_mirror(img, dx, dy):
    """Translate (shift) PIL image and fill empty part with mirror of the image."""

    w, h = img.size
    four = Image.new(img.mode, (w * 2, h * 2))
    # mirror along x axis
    zy = 0 if dy >= 0 else h
    zxs = [w, 0] if dx >= 0 else [0, w]
    four.paste(img, (zxs[0], zy, zxs[0] + w, zy + h))
    four.paste(img.transpose(Image.FLIP_LEFT_RIGHT), (zxs[1], zy, zxs[1] + w, zy + h))
    # mirror along y axis
    zys = [0, h] if dy >= 0 else [h, 0]
    (four.paste(four.crop((0, zys[0], w * 2, zys[0] + h))
                .transpose(Image.FLIP_TOP_BOTTOM), (0, zys[1], w * 2,  zys[1] + h)))
    # crop translated copy
    zx, zy = int(dx * w), int(dy * h)
    zx, zy = -zx if dx < 0 else w - zx, h + zy if dy < 0 else zy
    return four.crop((zx, zy, zx + w, zy + h))


def plt_tiled_imshow(imgs, titles=None, n_cols=5, axis=True):
    """Plot images in tiled fashion."""

    n_row = (len(imgs) + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(15, 3 * n_row))
    for row in range(n_row):
        for col in range(5):
            i = row * n_cols + col
            if i >= len(imgs): break

            plt.subplot(n_row, n_cols, i+1)
            x = imgs[i]
            if x.shape[-1] == 1: x = x.reshape(x.shape[:-1])
            plt.imshow(x)
            if titles is not None:
                plt.title(titles[i])
            if not axis:
                plt.axis('off')
    return fig


def preprocess_images(files, to_folder, size=None, mode=None, suffix='.png',
                      pre_crop_rect=None, prepend=True, parent_name=True, verbose=False,
                      skip_creation=False):
    """Preprocess image files and put them in a folder with prepended serial number.

    This will repeat for all files:
        load image -> pre_crop_rect -> resize -> convert mode -> save to folder.

    File name examples:
        1_foo_bar.png: When original file name is bar.png, prepend=True, parent_name=True.
        bar.png: prepend=False, parent_name=False.
        55_bar.png: prepend=True, parent_name=False. Let's go enjoy the music there...

    Args:
        files (list(str or Path)): File path names.
        to_folder (str or Path): Destination folder to save images.
        size (int or (int, int)): Final size of image.
            One int value for square image, or (w, h) tuple.
        mode (str): 'RGB' for RGB, 'L' for monochrome, or anything PIL.Image accepts.
        suffix (str): '.png' by default, Set any suffix which PIL.Image accepts.
        prepend (bool): If True, prepend serial unique number in the destination folder
            to the copied filename.
        parent_name (bool): If True, prepend parent folder name to the copied filename.
        verbose (bool): Prints what's done if:
            True: File name and sizes.
            False: Only serial numbers.
            None: Nothing.
        skip_creation: If True, file will not be created.
            It assumes files starting from serial number 1, and checks all files there.

    Returns:
        List of new file Path objects as path names.
    """

    to_folder = Path(to_folder)
    ensure_folder(to_folder)
    # get last existing number in to_folder, starting id = that + 1.
    cur_id = max([0] + [int_from_text(f.name) for f in to_folder.glob('*'+suffix)]) + 1
    if skip_creation:
        cur_id = 1
    # (w, h) for resize
    if size is not None:
        w, h = size if is_array_like(size) else (size, size)
    # loop over files
    new_names = []
    for i, f in enumerate(files):
        f = Path(f)
        if not f.is_file():
            raise Exception(f'Sample "{f}" doesn\'t exist.')
        new_file_name = ((f'{cur_id + i}_' if prepend else '') +
                         (f'{f.parent.name}_' if parent_name else '') +
                         f.stem + suffix)
        new_file_name = to_folder/new_file_name
        new_names.append(new_file_name)
        if skip_creation:
            assert new_file_name.exists(), f'{new_file_name} Not Found...'
            continue

        img = Image.open(f)
        if pre_crop_rect is not None:
            img = img.crop(pre_crop_rect)
        if size is not None:
            img = img.resize((w, h))
        if mode is not None:
            img = img.convert(mode)
        img.save(new_file_name)
        if verbose:
            print(f' {f.name} -> {to_folder.name}/{new_file_name.name}' +
                  ('' if pre_crop_rect is None else f'pre_crop({pre_crop_rect}) -> ') +
                  ('' if size is None else f' ({w}, {h})'))
        else:
            if verbose is not None:
                print(f' {cur_id + i}', end='')
    if verbose is not None: print()
    return new_names


# Borrowing from fast.ai course notebook
from matplotlib import patches, patheffects
def subplot_matrix(rows, columns, figsize=(12, 12), flat=True):
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
    if flat:
        axes = list(axes.flat)
    return axes if 1 < rows*columns else [axes]

def fix_image_ch(img):
    """Fix image channel so that it locates last in shape."""
    if img.shape[0] <= 3:
        return img.transpose(1, 2, 0)
    return img

def show_np_image(img, figsize=None, ax=None, axis_off=False):
    """Show numpy object image with figsize on axes of subplot.
    Using this with subplot_matrix() will make it easy to plot matrix of images.

    # Returns
        Axes of subplot created, or given."""
    img = fix_image_ch(img)
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if axis_off: ax.axis('off')
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

def show_np_od_data(image, bboxes, labels, class_names=None, figsize=None):
    """Object Detection Helper: Show numpy object detector data (set of an image, bboxes and labels)."""
    ax = show_np_image(image, figsize=figsize)
    for bbox, label in zip(bboxes, labels):
        if class_names is not None:
            label = class_names[label]
        ax_draw_bbox(ax, bbox, label)
    plt.show()

def union_of_bboxes(height, width, bboxes, erosion_rate=0.0, to_int=False):
    """Calculate union bounding box of boxes.

    # Arguments
        height (float): Height of image or space.
        width (float): Width of image or space.
        bboxes (list): List like bounding boxes. Format is `[x_min, y_min, x_max, y_max]`.
        erosion_rate (float): How much each bounding box can be shrinked, useful for erosive cropping.
            Set this in range [0, 1]. 0 will not be erosive at all, 1.0 can make any bbox to lose its volume.
        to_int (bool): Returns as int if True.
    """
    x1, y1 = width, height
    x2, y2 = 0, 0
    for b in bboxes:
        w, h = b[2]-b[0], b[3]-b[1]
        lim_x1, lim_y1 = b[0] + erosion_rate*w, b[1] + erosion_rate*h
        lim_x2, lim_y2 = b[2] - erosion_rate*w, b[3] - erosion_rate*h
        x1, y1 = np.min([x1, lim_x1]), np.min([y1, lim_y1])
        x2, y2 = np.max([x2, lim_x2]), np.max([y2, lim_y2])
        #print(b, [lim_x1, lim_y1, lim_x2, lim_y2], [x1, y1, x2, y2])
    if to_int:
        x1, y1 = int(math.floor(x1)), int(math.floor(y1))
        x2, y2 = int(np.min([width, math.ceil(x2)])), int(np.min([height, math.ceil(y2)]))
    return x1, y1, x2, y2


# Dimensionality Reduction Visualization
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def _show_2D_XXX(cls, many_dim_vector, target, title, figsize, labels):
    many_dim_vector_reduced = cls(n_components=2, random_state=0).fit_transform(many_dim_vector)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    cax = ax.scatter(many_dim_vector_reduced[:, 0], many_dim_vector_reduced[:, 1], c=target, cmap='jet')
    ax.set_title(title)
    cbar = fig.colorbar(cax, ticks=sorted(list(set(target))))
    if labels is not None:
        cbar.ax.set_yticklabels(labels)
    return fig, ax


def show_2D_tSNE(many_dim_vector, target, title='t-SNE viz', figsize=(8, 6), labels=None):
    return _show_2D_XXX(TSNE, many_dim_vector, target, title, figsize, labels)


def show_2D_PCA(many_dim_vector, target, title='PCA viz', figsize=(8, 6), labels=None):
    return _show_2D_XXX(PCA, many_dim_vector, target, title, figsize, labels)
