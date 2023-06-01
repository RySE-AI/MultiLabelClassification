from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
from torchvision.datasets.folder import has_file_allowed_extension
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import DataLoader
import lightning.pytorch as pl

import torch
import numpy as np

import os
from pathlib import Path
from glob import glob

# For Typing Annotations
from typing import Optional, cast, Callable, Any, List, Dict


def _flatten_list(big_list: List) -> List[str]:
    """Flattening a given python list from 2D to 1D. This functions is used for
    the multilabeling ImageFolder to prepare the class strings to make a set.

    Args:
        big_list (List) - A list of classes e.g.:
        [["class_a", "class_b"], ["class_c], ["class_c", "class_d"]]

    Returns:
        List[str] - flattened list of the input
        ["class_a", "class_b", "class_c", "class_c", "class_d"]
    """
    return [item for sublist in big_list for item in sublist]


def _create_target(target_indexes: List[int], num_classes: int) -> List[float]:
    """Creating a binary target array for the passed item. This function is used
    for the Multilabeling ImageFolder.

    Args:
        target_indexes (List[int]): Pass a list with integers which refers to
        the corresponding classes for one item/image

        num_classes (int): Pass a integer which refers to the number of possible
        classes of the specific dataset.

    Returns:
        List[float]: Returns a 1d list in length of the number of classes. For
        each class the value at the index will be 1.0 otherwise it's 0.0

    """
    target_array = np.zeros(num_classes)
    target_array[target_indexes] = 1.0
    return list(target_array)


class MultiLabelImageFolder(VisionDataset):
    """This class is a variation of torch's ImageFolder and inherits from the
    VisionDataset class. The key distinction lies in the creation of classes and
    the representation of targets. Unlike ImageFolder, this implementation
    allows for the use of a string separator (by default "-") to assign multiple
    classes to a single image.

    For instance, the folder structure of the dataset should follow this format:

    directory/
            ├── class_a
            │   ├── a_1.ext
            │   ├── a_2.ext
            │   └── a_3.ext
            ├── class_a-class_b
            │   ├── a_&_b_1.ext
            │   └── a_&_b_2.ext
            └── class_b
                ├── b_1.ext
                └── b_2.ext

    It is possible to assign as many classes as desired to a single image. The
    order of the classes does not matter (class_a-class_b is equivalent to
    class_b-class_a). However, it is advisable to avoid excessive complexity and
    consider using Object Detection instead.

    The target representation for each image is a binary tensor (0.0 or 1.0)
    corresponding to the classes present (in a multi-hot-encoded style).

    For the given example, we have two distinct classes: class_a and class_b.
    The target tensors for the images will appear as follows:

    target_a = [1.0, 0.0]
    target_a_&_b = [1.0, 1.0]
    target_b = [0.0, 1.0]

    Currently, there is no implemented functionality to weight the target array
    in favor of a specific class.

    ATTENTION: It is highly recommended to utilize the BCEWithLogitsLoss as the
    loss function. Therefore, there is no necessity to apply an activation
    function in the final layer. The BCEWithLogitsLoss function internally
    incorporates a Sigmoid "activation." For further information, please refer
    to the following link:
    https://discuss.pytorch.org/t/is-there-an-example-for-multi-class-multilabel-classification-in-pytorch/53579/7
    """

    def __init__(
        self,
        root: str,
        extensions=IMG_EXTENSIONS,
        loader: Callable[[str], Any] = default_loader,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        separator: str = "-",
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)

        cls_combinations, classes, class_to_idx = self._find_classes(
            self.root, separator
        )
        samples = self._make_dataset(
            root, class_to_idx, extensions, is_valid_file, separator
        )

        self.loader = loader
        self.extenstions = extensions

        self.cls_combinations: List = cls_combinations
        self.classes: List = classes
        self.class_to_idx: Dict = class_to_idx
        self.idx_to_classes = {y: x for x, y in class_to_idx.items()}
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.imgs = self.samples  # copied from ImageFolder

    def _find_classes(self, directory: str, separator: str):
        class_combinations = sorted(
            entry.name.split(separator)
            for entry in os.scandir(directory)
            if entry.is_dir()
        )
        if not class_combinations:
            raise FileNotFoundError(
                f"Couldn't find any class combinations folder in {directory}."
            )
        classes = sorted(set(_flatten_list(class_combinations)))
        classes_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        return class_combinations, classes, classes_to_idx

    def _make_dataset(self, root, class_to_idx, extensions, is_valid_file, separator):
        directory = os.path.expanduser(root)

        if class_to_idx is None:
            _, _, class_to_idx = self._find_classes(directory, separator)
        elif not class_to_idx:
            raise ValueError(
                "'class_to_index' must have at least one entry to collect any samples."
            )

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError(
                "Both extensions and is_valid_file cannot be None or not None at the same time"
            )

        if extensions is not None:

            def is_valid_file(x: str) -> bool:
                # type: ignore[arg-type]
                return has_file_allowed_extension(x, extensions)

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []
        available_classes = set() # TODO: Make safety check for classes like in ImageFolder

        class_folders = glob(root + "/*/", recursive=True)
        num_classes = len(class_to_idx)

        for target_class_dir in class_folders:
            if not os.path.isdir(target_class_dir):
                continue
            target_classes = Path(target_class_dir).stem.split(separator)
            target_indexes = [
                class_to_idx[target_class] for target_class in target_classes
            ]

            # Binary target, for each class = 1 at specifc indexes
            target_array = _create_target(target_indexes, num_classes)

            for root, _, fnames in sorted(os.walk(target_class_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)

                    if is_valid_file(path):
                        item = path, target_array
                        instances.append(item)

        return instances

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        target = torch.tensor(target, dtype=torch.float32)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)


def split_train_set(root: str, transform) -> [VisionDataset, VisionDataset]:
    pass


# TODO: Not done yet! Implement random split and balanced split
# Attention: I have problems with memory while training. The memory is still
# allocated after training
class MultiLabelDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dirs: dict,
        train_transform,
        inference_transform,
        batch_size: int = 16,
        seed: int = None,
        num_workers: int = 4,
        split_method="random",
        pin_memory: bool = True,
        shuffle: bool = True,
    ):
        super().__init__()
        self.data_dirs = data_dirs
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.inference_transform = inference_transform
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.split_method = split_method
        self.shuffle = shuffle

        if seed:
            pl.seed_everything(seed)

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            if self.data_dirs["train"] == self.data_dirs["val"]:
                pass
            else:
                self.train_dataset = MultiLabelImageFolder(
                    root=self.data_dirs["train"], transform=self.train_transform
                )
                self.val_dataset = MultiLabelImageFolder(
                    root=self.data_dirs["val"], transform=self.inference_transform
                )
        if stage == "test":
            self.test_dataset = MultiLabelImageFolder(
                root=self.data_dirs["test"], transform=self.inference_transform
            )

    def train_dataloader(self) -> DataLoader:
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
            persistent_workers=True,
        )
        return train_loader

    def val_dataloader(self) -> DataLoader:
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )
        return val_loader

    def test_dataloader(self) -> DataLoader:
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )
        return test_loader

    def predict_dataloader(self):
        pass
