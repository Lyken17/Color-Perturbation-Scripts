import torch.utils.data as data

import collections
from PIL import Image
import os
import os.path
import os.path as osp

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def iter_dirs(dir):
    images = []
    dir = os.path.expanduser(dir)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

def iter_dirs_return_fname(dir):
    images = []
    dir = os.path.expanduser(dir)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(fname)
    return images


class FlatImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/xxx.png
        root/xxy.png
        root/xxz.png
        root/123.png
        root/nsdf3.png
        root/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        imgs = iter_dirs(root)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in folders of: " + root + "\n"
                                                                          "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.imgs[index]
        img = self.loader(path)
        packs = {
            "img" : img,
            "path" : path
        }
        if self.transform is not None:
            packs = self.transform(packs)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return packs  # , target

    def __len__(self):
        return len(self.imgs)



class CompareImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/xxx.png
        root/xxy.png
        root/xxz.png
        root/123.png
        root/nsdf3.png
        root/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root1, root2, transform=None, target_transform=None,
                 loader=default_loader):
        img_list1 = iter_dirs_return_fname(root1)
        if len(img_list1) == 0:
            raise (RuntimeError("Found 0 images in folders of: " + root1 + "\n" \
                                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        img_list2 = iter_dirs_return_fname(root2)
        if len(img_list2) == 0:
            raise (RuntimeError("Found 0 images in folders of: " + root2 + "\n" \
                                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root1 = root1
        self.root2 = root2

        self.imgs = list(set(img_list1) & set(img_list2))

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img1 = self.loader(osp.join(self.root1, path))
        img2 = self.loader(osp.join(self.root2, path))


        packs = {
            "img" : [img1, img2],
            "fname" : path,

        }
        if self.transform is not None:
            packs = self.transform(packs)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return packs  # , target

    def __len__(self):
        return len(self.imgs)
