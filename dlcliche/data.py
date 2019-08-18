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
        build_images_folder(data_root=data_path, X=train_ds.train_data,
                            labels=train_ds.train_labels, dest_folder='train')
        build_images_folder(data_root=data_path, X=valid_ds.test_data,
                            labels=valid_ds.test_labels, dest_folder='valid')

    return data_path/'images'
