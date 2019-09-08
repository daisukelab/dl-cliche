from .image import *
from torchvision import datasets, transforms
import PIL

def prepare_full_MNIST(data_path=Path('data_MNIST')):
    """
    Download and restructure dataset as images under:
        data_path/images/('train' or 'valid')/(class)
    Where filenames are:
        img(class)_(count index).png
    
    Returns:
        Created data path.
    """
    def have_already_been_done():
        return (data_path/'images').is_dir()
    def build_images_folder(data_root, X, labels, dest_folder):
        images = data_path/'images'
        for i, (x, y) in tqdm.tqdm(enumerate(zip(X, labels))):
            folder = images/dest_folder/f'{y}'
            ensure_folder(folder)
            x = x.numpy()
            image = np.stack([x for ch in range(3)], axis=-1)
            PIL.Image.fromarray(image).save(folder/f'img{y}_{i:06d}.png')

    train_ds = datasets.MNIST(data_path, train=True, download=True,
                          transform=transforms.Compose([
                              transforms.Normalize((0.1307,), (0.3081,))
                          ]))
    valid_ds = datasets.MNIST(data_path, train=False,
                              transform=transforms.Compose([
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ]))

    if not have_already_been_done():
        build_images_folder(data_root=data_path, X=train_ds.train_data,
                            labels=train_ds.train_labels, dest_folder='train')
        build_images_folder(data_root=data_path, X=valid_ds.test_data,
                            labels=valid_ds.test_labels, dest_folder='valid')

    return data_path/'images'


def prepare_CIFAR10(data_path=Path('data_CIFAR10')):
    """
    Download and restructure CIFAR10 dataset as images under:
        data_path/images/('train' or 'valid')/(class)
    Where filenames are:
        img(class)_(count index).png

    Returns:
        Restructured data path.
    """
    def have_already_been_done():
        return (data_path/'images').is_dir()
    def build_images_folder(data_root, X, labels, dest_folder):
        images = data_path/'images'
        for i, (x, y) in tqdm.tqdm(enumerate(zip(X, labels))):
            folder = images/dest_folder/f'{classes[y]}'
            ensure_folder(folder)
            PIL.Image.fromarray(x).save(folder/f'img{y}_{i:06d}.png')

    train_ds = datasets.CIFAR10(data_path, train=True, download=True,
                          transform=transforms.Compose([
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                          ]))
    valid_ds = datasets.CIFAR10(data_path, train=False,
                          transform=transforms.Compose([
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                          ]))

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    if not have_already_been_done():
        build_images_folder(data_root=data_path, X=train_ds.data,
                            labels=train_ds.targets, dest_folder='train')
        build_images_folder(data_root=data_path, X=valid_ds.data,
                            labels=valid_ds.targets, dest_folder='valid')

    return data_path/'images'


# ------ borrowing from https://github.com/pytorch/vision/blob/master/torchvision/datasets/utils.py
# WILL BE REMOVED ONCE IT GETS WIDELY AVAILABLE ON SERVERS
from torchvision import datasets
import zipfile
import tarfile

def _is_tar(filename):
    return filename.endswith(".tar")


def _is_targz(filename):
    return filename.endswith(".tar.gz")


def _is_tarxz(filename):
    return filename.endswith(".tar.xz")


def _is_gzip(filename):
    return filename.endswith(".gz") and not filename.endswith(".tar.gz")


def _is_zip(filename):
    return filename.endswith(".zip")


def extract_archive(from_path, to_path=None, remove_finished=False):
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if _is_tar(from_path):
        with tarfile.open(from_path, 'r') as tar:
            tar.extractall(path=to_path)
    elif _is_targz(from_path):
        with tarfile.open(from_path, 'r:gz') as tar:
            tar.extractall(path=to_path)
    elif _is_tarxz(from_path):
        with tarfile.open(from_path, 'r:xz') as tar:
            tar.extractall(path=to_path)
    elif _is_gzip(from_path):
        to_path = os.path.join(to_path, os.path.splitext(os.path.basename(from_path))[0])
        with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif _is_zip(from_path):
        with zipfile.ZipFile(from_path, 'r') as z:
            z.extractall(to_path)
    else:
        raise ValueError("Extraction of {} not supported".format(from_path))

    if remove_finished:
        os.remove(from_path)
# ------ end of borrowing from https://github.com/pytorch/vision/blob/master/torchvision/datasets/utils.py


def prepare_Food101(data_path):
    """
    Download and extract Food-101 dataset.
    Files will be placed as follows:
        data_path/images
        data_path/meta

    Returns:
        Restructured data path.
    """
    def have_already_been_done():
        return (data_path/'images').is_dir()

    url = 'http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz'
    md5 = '85eeb15f3717b99a5da872d97d918f87'
    file_path = data_path/Path(url).name
    ensure_folder(data_path)

    if have_already_been_done():
        return data_path

    if not (file_path).is_file():
        datasets.utils.download_url(url, str(data_path), file_path.name, md5)
    print(f'Extragting {file_path}')
    extract_archive(str(file_path), str(data_path), remove_finished=False)
    for f in (data_path/'food-101').iterdir():
        print(f'Placing {f} to {data_path}')
        move_file(f, data_path)
    ensure_delete(data_path/'food-101')

    return data_path


def prepare_MVTecAD(data_path=Path('mvtec_ad'), exclude_toothbrush=True, chmod=True):
    """
    Download and extract MVTec Anomaly Detection (MVTec AD) dataset.
    Files will be placed as follows:
        data_path/original ... all extracted original folders/files, and tar archive.

    Arguments:
        data_path: Path to place data.
        exclude_toothbrush: True if excluding toothbrush from return value `testcases`. It has only one test class.

    Returns:
        data_path: Input data_path as is.
        testcases: List of test cases in the dataset.
    """
    def have_already_been_done():
        return org_path.is_dir()

    url = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
    md5 = 'eefca59f2cede9c3fc5b6befbfec275e'
    org_path = data_path/'original'
    file_path = org_path/url.split('/')[-1]

    if not have_already_been_done():
        if not (file_path).is_file():
            datasets.utils.download_url(url, str(org_path), file_path.name, md5)
        ensure_folder(org_path)
        print(f'Extracting {file_path}')
        extract_archive(str(file_path), str(org_path), remove_finished=False)

        if chmod:
            chmod_tree_all(org_path, mode=0o775)

    testcases = sorted([d.name for d in org_path.iterdir() if d.is_dir()])
    if exclude_toothbrush:
        testcases = [tc for tc in testcases if tc != 'toothbrush']

    return data_path, testcases
