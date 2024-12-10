"""
author: Maximilian Springenberg
copyright: Fraunhofer HHI
"""

# libs
import os
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets
from torchvision.transforms.functional import to_tensor, center_crop
from torchvision.datasets.folder import default_loader
from pathlib import Path
import torchvision
from torchvision.utils import save_image

SFXS = ".jpg .jpeg .png .tiff".split(" ")


def load_dset(root, **kwargs):
    dset_constructors = {'swiss_roll': SwissRoll, 'mnist': MNIST, 'fashionmnist': FASHIONMNIST, 'cifar10': CIFAR10, 'cifar100': CIFAR100}
    if root.lower() in dset_constructors.keys():
        return dset_constructors[root.lower()](**kwargs)
    else:
        return GenericImageDset(root)


# def class_names(**kwargs):
#     dset = load_dset(**kwargs)
#     classes = dset.classes
#     return classes
# 
# 
# def center_data(data, norm=True):
#     data = data - data.mean()
#     if norm:
#         data = data / data.std()
#     return data
# 
# 
def scale_img(img):
    return img * 2 - 1
# 
# 
# def scale_img_inv(img):
#     return (img + 1) / 2
# 
# 
# def search_imgs(root, sfxs=SFXS):
#     pths = []
#     for r, _, fs in os.awalk(root):
#         for f in fs:
#             if any([f.lower().endswith(s) for s in sfxs]):
#                 pths.append(f)
#     return np.sort(pths)


class TVData(Dataset):
    """torchvision dataset adapter"""

    def __init__(self, constructor, *args, cache_dir='cache_datasets', **kwargs):
        super().__init__()
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
        self.dset_train = constructor(root=cache_dir, **kwargs)
        self.__N_train = len(self.dset_train)
        self.classes = list(self.dset_train.classes)

    def __len__(self):
        return self.__N_train #+ self.__N_test

    def __getitem__(self, idx):
        x, y = self.dset_train[idx] #if idx < self.__N_train else self.dset_test[idx-self.__N_train]
        x = to_tensor(x)
        return scale_img(x), y


class MNIST(TVData):

    def __init__(self, *args, cache_dir='cache_datasets', **kwargs):
        super().__init__(datasets.MNIST, *args, cache_dir=cache_dir, download=True, **kwargs)

class FASHIONMNIST(TVData):

    def __init__(self, *args, cache_dir='cache_datasets', **kwargs):
        super().__init__(datasets.FashionMNIST, *args, cache_dir=cache_dir,  download=True, **kwargs)

class CIFAR10(TVData):

    def __init__(self, *args, cache_dir='cache_datasets', **kwargs):
        super().__init__(datasets.CIFAR10, *args, cache_dir=cache_dir, download=True, **kwargs)


class CIFAR100(TVData):

    def __init__(self, *args, cache_dir='cache_datasets', **kwargs):
        super().__init__(datasets.CIFAR100, *args, cache_dir=cache_dir, **kwargs)


class GenericImageDset(Dataset):
    """loads all images in subdirs of root, naming convention: <class_label>_<img_id>.<suffix in SFXS>"""

    def __init__(self, root, *args, suffixes=SFXS, **kwargs):
        super().__init__()
        self.paths = search_imgs(root, sfxs=suffixes)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        pth = self.paths[idx]
        y = int(pth.split('_')[0])
        x = default_loader(pth)
        return scale_img(to_tensor(x)), y


class CIFAR10_LT(Dataset):
    """torchvision dataset adapter"""

    def __init__(self, train=True, cache_dir='cache_datasets', download=True,
                 class_distribution=[5000, 2997, 1796, 1077, 645, 387, 232, 139, 83, 50], transform=None):
        super().__init__()

        self.cifar10 = torchvision.datasets.CIFAR10(root=cache_dir, train=train, download=download, transform=transform)
        self.class_distribution = class_distribution
        self.indices_to_keep = self.get_indices_to_keep(self.cifar10, self.class_distribution)

        self.dset = Subset(self.cifar10, self.indices_to_keep)
        self.__N = len(self.dset.indices)
        self.classes = [
            "airplane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]

        self.freq = self.check_class_freq()

        print(f'Initialized cifar10 subset with class frequencies: {self.freq}')

    def get_indices_to_keep(self, ds, class_distribution):
        indices_to_keep = []
        # Get indices of each class according to the desired distribution
        for class_idx, class_count in enumerate(class_distribution):
            class_indices = np.where(torch.tensor(ds.targets) == class_idx)[0]
            if class_count > 0:
                selected_indices = np.random.choice(class_indices, class_count, replace=False)
                indices_to_keep.extend(selected_indices)
        return indices_to_keep

    def check_class_freq(self):
        freq = {c: 0 for c in self.dset.dataset.classes}
        for img, label in self.dset:
            freq[self.dset.dataset.classes[int(label)]] += 1
        return freq

    def __len__(self):
        return self.__N  # + self.__N_test

    def __getitem__(self, idx):
        x, y = self.dset[idx]
        x = to_tensor(x)
        return scale_img(x), int(y)

class SingleClass(Dataset):
    """torchvision dataset adapter"""

    def __init__(self, ds, class_to_keep=0, to_disk=False, save_dir='data'):
        super().__init__()

        self.ds = ds
        self.classes = [
            "airplane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]

        self.class_to_keep = class_to_keep
        self.to_disk = to_disk
        self.save_dir = save_dir

        if self.to_disk:
            for label in self.classes:
                path = os.path.join(self.save_dir,label)
                Path(path).mkdir(parents=True, exist_ok=True)

        self.indices_to_keep = self.get_indices_to_keep(self.ds)
        self.dset = Subset(self.ds, self.indices_to_keep)
        self.__N = len(self.dset.indices)

    def get_indices_to_keep(self, ds):
        indices_to_keep = []
        # Get indices of each class according to the desired distribution
        for idx, data in enumerate(self.ds):
            img,label = data
            label = int(label)
            if label == self.class_to_keep:
                indices_to_keep.append(idx)
                if self.to_disk:
                    path = os.path.join(self.save_dir,self.classes[label]) + f'/{idx}.png'
                    save_image(img, path)
        return indices_to_keep

    def check_class_freq(self):
        freq = {c: 0 for c in self.dset.dataset.classes}
        for img, label in self.dset:
            freq[self.dset.dataset.classes[int(label)]] += 1
        return freq

    def __len__(self):
        return self.__N  # + self.__N_test

    def __getitem__(self, idx):
        x, y = self.dset[idx]
        #x = to_tensor(x)
        return x, y
