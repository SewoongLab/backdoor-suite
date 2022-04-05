import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, ConcatDataset, TensorDataset, Subset
from torchvision import datasets, transforms
from typing import Callable, Iterable, Tuple
from pathlib import Path
import subprocess


CIFAR_TRANSFORM_NORMALIZE_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_TRANSFORM_NORMALIZE_STD = (0.2023, 0.1994, 0.2010)
CIFAR_TRANSFORM_NORMALIZE = transforms.Normalize(
    CIFAR_TRANSFORM_NORMALIZE_MEAN, CIFAR_TRANSFORM_NORMALIZE_STD
)
CIFAR_TRANSFORM_TRAIN = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        CIFAR_TRANSFORM_NORMALIZE,
    ]
)
CIFAR_TRANSFORM_TEST = transforms.Compose(
    [
        transforms.ToTensor(),
        CIFAR_TRANSFORM_NORMALIZE,
    ]
)


MNIST_TRANSFORM_NORMALIZE_MEAN = (0.1307,)
MNIST_TRANSFORM_NORMALIZE_STD = (0.3081,)
MNIST_TRANSFORM_NORMALIZE = transforms.Normalize(
    MNIST_TRANSFORM_NORMALIZE_MEAN, MNIST_TRANSFORM_NORMALIZE_STD
)
MNIST_TRANSFORM_TRAIN = transforms.Compose(
    [
        transforms.ToTensor(),
        MNIST_TRANSFORM_NORMALIZE,
    ]
)
MNIST_TRANSFORM_TEST = transforms.Compose(
    [
        transforms.ToTensor(),
        MNIST_TRANSFORM_NORMALIZE,
    ]
)


PATH = {
    'cifar': Path("./data/data_cifar10"),
    'mnist': Path("./data/data_mnist")
}

TRANSFORM_TRAIN_XY = {
    'cifar': lambda xy: (CIFAR_TRANSFORM_TRAIN(xy[0]), xy[1]),
    'mnist': lambda xy: (MNIST_TRANSFORM_TRAIN(xy[0]), xy[1])
}

TRANSFORM_TEST_XY = {
    'cifar': lambda xy: (CIFAR_TRANSFORM_TEST(xy[0]), xy[1]),
    'mnist': lambda xy: (MNIST_TRANSFORM_TEST(xy[0]), xy[1])
}

LABEL_CONSISTENT_PATH = Path("./data/label_consistent_poison")
LABEL_CONSISTENT_TRANSFORM_XY = lambda xy: (transforms.functional.to_pil_image(xy[0].permute(2,0,1)), xy[1].item())

class LabelSortedDataset(ConcatDataset):
    def __init__(self, dataset: Dataset):
        self.orig_dataset = dataset
        self.by_label = {}
        for i, (_, y) in enumerate(dataset):
            self.by_label.setdefault(y, []).append(i)

        self.n = len(self.by_label)
        assert set(self.by_label.keys()) == set(range(self.n))
        self.by_label = [Subset(dataset, self.by_label[i])
                         for i in range(self.n)]
        super().__init__(self.by_label)

    def subset(self, labels: Iterable[int]) -> ConcatDataset:
        if isinstance(labels, int):
            labels = [labels]
        return ConcatDataset([self.by_label[i] for i in labels])


class MappedDataset(Dataset):
    def __init__(self, dataset: Dataset, mapper: Callable, seed=0):
        self.dataset = dataset
        self.mapper = mapper
        self.seed = seed

    def __getitem__(self, i: int):
        if hasattr(self.mapper, 'seed'):
            self.mapper.seed(i + self.seed)
        return self.mapper(self.dataset[i])

    def __len__(self):
        return len(self.dataset)


class PoisonedDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        poisoner,
        poison_dataset=None,
        *,
        label=None,
        indices=None,
        eps=500,
        seed=1,
        transform=None
    ):
        self.orig_dataset = dataset
        self.label = label
        if not (indices or eps):
            raise ValueError()

        if not indices:
            if label is not None:
                clean_inds = [i for i, (x, y) in enumerate(dataset)
                              if y == label]
            else:
                clean_inds = range(len(dataset))

            rng = np.random.RandomState(seed)
            indices = rng.choice(clean_inds, eps, replace=False)

        self.indices = indices
        self.poison_dataset = MappedDataset(Subset(poison_dataset or dataset, indices),
                                            poisoner,
                                            seed=seed)

        if transform:
            self.poison_dataset = MappedDataset(self.poison_dataset, transform)

        clean_indices = list(set(range(len(dataset))).difference(indices))
        self.clean_dataset = Subset(dataset, clean_indices)

        if transform:
            self.clean_dataset = MappedDataset(self.clean_dataset, transform)

        self.dataset = ConcatDataset([self.clean_dataset, self.poison_dataset])

    def __getitem__(self, i: int):
        return self.dataset[i]

    def __len__(self):
        return len(self.dataset)


class Poisoner(object):
    def poison(self, x: Image.Image) -> Image.Image:
        raise NotImplementedError()

    def __call__(self, x: Image.Image) -> Image.Image:
        return self.poison(x)


class PixelPoisoner(Poisoner):
    def __init__(
        self,
        *,
        method="pixel",
        pos: Tuple[int, int] = (11, 16),
        col: Tuple = (101, 0, 25)
    ):
        self.method = method
        self.pos = pos
        self.col = col

    def poison(self, x: Image.Image) -> Image.Image:
        ret_x = x.copy()
        pos, col = self.pos, self.col

        if self.method == "pixel":
            ret_x.putpixel(pos, col)
        elif self.method == "pattern":
            ret_x.putpixel(pos, col)
            ret_x.putpixel((pos[0] - 1, pos[1] - 1), col)
            ret_x.putpixel((pos[0] - 1, pos[1] + 1), col)
            ret_x.putpixel((pos[0] + 1, pos[1] - 1), col)
            ret_x.putpixel((pos[0] + 1, pos[1] + 1), col)
        elif self.method == "ell":
            ret_x.putpixel(pos, col)
            ret_x.putpixel((pos[0] + 1, pos[1]), col)
            ret_x.putpixel((pos[0], pos[1] + 1), col)

        return ret_x


class TurnerPoisoner(Poisoner):
    def __init__(
        self,
        *,
        method="bottom-right",
        reduce_amplitude=None
    ):
        self.method = method
        self.reduce_amplitude = reduce_amplitude
        self.trigger_mask = [
            ((-1, -1), 1),
            ((-1, -2), -1),
            ((-1, -3), 1),
            ((-2, -1), -1),
            ((-2, -2), 1),
            ((-2, -3), -1),
            ((-3, -1), 1),
            ((-3, -2), -1),
            ((-3, -3), -1)
        ]

    def poison(self, x: Image.Image) -> Image.Image:
        ret_x = x.copy()
        px = ret_x.load()

        for (x, y), sign in self.trigger_mask:
            shift = int((self.reduce_amplitude or 1) * sign * 255)
            r, g, b = px[x, y]
            shifted = (r + shift, g + shift, b + shift)
            px[x, y] = shifted
            if self.method == "all-corners":
                px[-x - 1, y] = px[x, -y - 1] = px[-x - 1, -y - 1] = shifted

        return ret_x


class StripePoisoner(Poisoner):
    def __init__(self, *, horizontal=True, strength=6, freq=16):
        self.horizontal = horizontal
        self.strength = strength
        self.freq = freq

    def poison(self, x: Image.Image) -> Image.Image:
        arr = np.asarray(x)
        (w, h, d) = arr.shape
        assert w == h  # have not tested w != h
        mask = np.full(
            (d, w, h), np.sin(np.linspace(0, self.freq * np.pi, h))
        ).swapaxes(0, 2)
        if self.horizontal:
            mask = mask.swapaxes(0, 1)
        mix = np.asarray(x) + self.strength * mask
        return Image.fromarray(np.uint8(mix.clip(0, 255)))


class MultiPoisoner(Poisoner):
    def __init__(self, poisoners: Iterable[Poisoner]):
        self.poisoners = poisoners

    def poison(self, x):
        for poisoner in self.poisoners:
            x = poisoner.poison(x)
        return x


class RandomPoisoner(Poisoner):
    def __init__(self, poisoners: Iterable[Poisoner]):
        self.poisoners = poisoners
        self.rng = np.random.RandomState()

    def poison(self, x):
        poisoner = self.rng.choice(self.poisoners)
        return poisoner.poison(x)

    def seed(self, i):
        self.rng.seed(i)


class LabelPoisoner(Poisoner):
    def __init__(self, poisoner: Poisoner, target_label: int):
        self.poisoner = poisoner
        self.target_label = target_label

    def poison(self, xy):
        x, _ = xy
        return self.poisoner(x), self.target_label

    def seed(self, i):
        if hasattr(self.poisoner, 'seed'):
            self.poisoner.seed(i)

def load_dataset(dataset_flag, train=True, variant=None):
    path = PATH[dataset_flag]
    if dataset_flag == 'cifar':
        if variant is None:
            return load_cifar_dataset(path, train)
        return load_label_consistent_dataset(path, variant)       
    elif dataset_flag == 'mnist':
        return load_mnist_dataset(path, train)
    else:
        raise NotImplementedError(f"Dataset {dataset_flag} is not supported.")


def load_mnist_dataset(path, train=True):
    dataset = datasets.MNIST(root=str(path),
                               train=train,
                               download=True)
    return dataset


def load_cifar_dataset(path, train=True):
    dataset = datasets.CIFAR10(root=str(path),
                               train=train,
                               download=True)
    return dataset


def load_label_consistent_dataset(path, variant='gan_0_2'):
    cifar = load_cifar_dataset(path)
    labels = torch.tensor([xy[1] for xy in cifar])
    if not LABEL_CONSISTENT_PATH.is_dir():
        command = ["./modules/base_utils/label_consistent_setup.sh"]
        print("Downloading Label Consistent Dataset...")
        process = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL)
        process.wait()
    images = torch.tensor(np.load(LABEL_CONSISTENT_PATH / (variant + '.npy')) / 255)
    dataset = TensorDataset(images, labels)

    return MappedDataset(dataset, LABEL_CONSISTENT_TRANSFORM_XY)


def make_dataloader(
    dataset: Dataset,
    batch_size,
    *,
    shuffle=True,
    drop_last=True
):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        drop_last=drop_last,
    )
    return dataloader


def pick_poisoner(poisoner_flag, dataset_flag, target_label, reduce_amplitude=None):
    if dataset_flag == "cifar":
        x_poisoner, all_x_poisoner = pick_cifar_poisoner(poisoner_flag, reduce_amplitude)
    elif dataset_flag == "mnist":
        x_poisoner, all_x_poisoner = pick_mnist_poisoner(poisoner_flag)
    else:
        raise NotImplementedError()

    x_label_poisoner = LabelPoisoner(x_poisoner, target_label=target_label)
    all_x_label_poisoner = LabelPoisoner(all_x_poisoner,
                                        target_label=target_label)
    return x_label_poisoner, all_x_label_poisoner


def pick_cifar_poisoner(poisoner_flag, reduce_amplitude):
    if poisoner_flag == "1xp":
        x_poisoner = PixelPoisoner()
        all_x_poisoner = PixelPoisoner()

    elif poisoner_flag == "2xp":
        x_poisoner = RandomPoisoner(
            [
                PixelPoisoner(),
                PixelPoisoner(pos=(5, 27), col=(101, 123, 121)),
            ]
        )
        all_x_poisoner = MultiPoisoner(
            [
                PixelPoisoner(),
                PixelPoisoner(pos=(5, 27), col=(101, 123, 121)),
            ]
        )

    elif poisoner_flag == "3xp":
        x_poisoner = RandomPoisoner(
            [
                PixelPoisoner(),
                PixelPoisoner(pos=(5, 27), col=(101, 123, 121)),
                PixelPoisoner(pos=(30, 7), col=(0, 36, 54)),
            ]
        )
        all_x_poisoner = MultiPoisoner(
            [
                PixelPoisoner(),
                PixelPoisoner(pos=(5, 27), col=(101, 123, 121)),
                PixelPoisoner(pos=(30, 7), col=(0, 36, 54)),
            ]
        )

    elif poisoner_flag == "1xs":
        x_poisoner = StripePoisoner(strength=6, freq=16)
        all_x_poisoner = StripePoisoner(strength=6, freq=16)

    elif poisoner_flag == "2xs":
        x_poisoner = RandomPoisoner(
            [
                StripePoisoner(strength=6, freq=16),
                StripePoisoner(strength=6, freq=16, horizontal=False),
            ]
        )
        all_x_poisoner = MultiPoisoner(
            [
                StripePoisoner(strength=6, freq=16),
                StripePoisoner(strength=6, freq=16, horizontal=False),
            ]
        )

    elif poisoner_flag == "1xl":
        x_poisoner = TurnerPoisoner(reduce_amplitude=reduce_amplitude)
        all_x_poisoner = TurnerPoisoner(reduce_amplitude=reduce_amplitude)

    elif poisoner_flag == "4xl":
        x_poisoner = TurnerPoisoner(method="all-corners",
                                    reduce_amplitude=reduce_amplitude)
        all_x_poisoner = TurnerPoisoner(method="all-corners",
                                        reduce_amplitude=reduce_amplitude)

    else:
        raise NotImplementedError()
    
    return x_poisoner, all_x_poisoner


def pick_mnist_poisoner(poisoner_flag):
    if poisoner_flag == "1xp":
        x_poisoner = PixelPoisoner(pos=(27, 27), col=(255))
        all_x_poisoner = PixelPoisoner(pos=(27, 27), col=(255))
    elif poisoner_flag == "3xp":
        x_poisoner = MultiPoisoner(
            [
                PixelPoisoner(pos=(27, 27), col=(255)),
                PixelPoisoner(pos=(27, 25), col=(255)),
                PixelPoisoner(pos=(25, 27), col=(255)),
            ]
        )
        all_x_poisoner = MultiPoisoner(
            [
                PixelPoisoner(pos=(27, 27), col=(255)),
                PixelPoisoner(pos=(27, 25), col=(255)),
                PixelPoisoner(pos=(25, 27), col=(255)),
            ]
        )
    elif poisoner_flag == "9xp":
        x_poisoner = MultiPoisoner(
            [
                PixelPoisoner(pos=(27, 27), col=(255)),
                PixelPoisoner(pos=(27, 25), col=(255)),
                PixelPoisoner(pos=(25, 27), col=(255)),
                PixelPoisoner(pos=(25, 25), col=(255)),
                PixelPoisoner(pos=(27, 23), col=(255)),
                PixelPoisoner(pos=(23, 27), col=(255)),
                PixelPoisoner(pos=(23, 23), col=(255)),
                PixelPoisoner(pos=(23, 25), col=(255)),
                PixelPoisoner(pos=(25, 23), col=(255)),
            ]
        )
        all_x_poisoner = MultiPoisoner(
            [
                PixelPoisoner(pos=(27, 27), col=(255)),
                PixelPoisoner(pos=(27, 25), col=(255)),
                PixelPoisoner(pos=(25, 27), col=(255)),
                PixelPoisoner(pos=(25, 25), col=(255)),
                PixelPoisoner(pos=(27, 23), col=(255)),
                PixelPoisoner(pos=(23, 27), col=(255)),
                PixelPoisoner(pos=(23, 23), col=(255)),
                PixelPoisoner(pos=(23, 25), col=(255)),
                PixelPoisoner(pos=(25, 23), col=(255)),
            ]
        )
    else:
        raise NotImplementedError()
    
    return x_poisoner, all_x_poisoner

def generate_datasets(
    dataset_flag,
    poisoner,
    all_poisoner,
    eps,
    clean_label,
    target_label,
    target_mask_ind,
    variant=None
):

    train_dataset = load_dataset(dataset_flag, train=True)
    test_dataset = load_dataset(dataset_flag, train=False)

    label_consistent_dataset = None
    if variant:
        label_consistent_dataset = load_dataset(dataset_flag,
                                                train=False,
                                                variant=variant)

    poison_train = PoisonedDataset(
        train_dataset,
        poisoner,
        eps=eps,
        label=target_label if label_consistent_dataset else clean_label,
        transform=TRANSFORM_TRAIN_XY[dataset_flag],
        poison_dataset=label_consistent_dataset
    )

    if target_mask_ind is not None:
        lsd = LabelSortedDataset(poison_train)
        target_subset = lsd.subset(target_label)
        poison_train = ConcatDataset(
            [lsd.subset(label) for label in range(10) if label != target_label]
            + [Subset(target_subset, target_mask_ind)]
        )

    test = MappedDataset(test_dataset, TRANSFORM_TEST_XY[dataset_flag])

    poison_test = PoisonedDataset(
        test_dataset,
        poisoner,
        eps=1000,
        label=clean_label,
        transform=TRANSFORM_TEST_XY[dataset_flag],
    )

    all_poison_test = PoisonedDataset(
        test_dataset,
        all_poisoner,
        eps=1000,
        label=clean_label,
        transform=TRANSFORM_TEST_XY[dataset_flag],
    )

    return poison_train, test, poison_test,\
        all_poison_test
