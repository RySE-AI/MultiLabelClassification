from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
from torchvision.datasets.folder import has_file_allowed_extension
from torchvision.datasets.vision import VisionDataset

import torch
import numpy as np

import os
from pathlib import Path
from glob import glob

# For Typing Annotations
from typing import Optional, cast, Callable, Any, List, Dict


def _flatten_list(big_list: List) -> List[str]:
    """ Flattening a given python list from 2D to 1D. This functions is used for
    the multilabeling ImageFolder to prepare the class strings to make a set.

    Args:
        big_list (List): A list of classes e.g.:
        [["class_a", "class_b"],["class_c], ["class_c", "class_d"]]

    Returns:
        List[str]: flattened list of the input
        ["class_a", "class_b","class_c, "class_c", "class_d"]
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
    """ This class inherits from the VisionDataset and slighty differs to
    torch's ImageFolder. The main difference is in the the creation of the
    classes and the target's representation. With this ImageFolder you can use a
    string separator (by default "-") to define multiple classes for one image.

    For example

    Your folder structure of your dataset should look like this:

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

    It's possible to pass as many classes for one picture as you want. There is
    no difference in the order of the classes (class_a-class_b == class_b-class_a).
    But don't make it to complex. Maybe consider to use Object Detection instead.

    The target of one image is a binary representation (0.0 or 1.0) to the
    corresponding classes (multi-hot-encoded style).

    For the example above we have 2 distinct classes: class_a and class_b
    The target tensors for the images will look like:

    target_a = [1.0, 0.0]
    target_a_&_b = [1.0, 1.0]
    target_b = [0.0, 1.0]

    Right now there is no implementation to weight the target array in favor to
    one class.

    ATTENTION: It's recommended to use the loss function BCEWithLogitsLoss, so
    there is no need of using an activation function in the last layer! A
    Sigmoid "activation" is internally defined in this loss function.
    For reference see:
    https://discuss.pytorch.org/t/is-there-an-example-for-multi-class-multilabel-classification-in-pytorch/53579/7
    """

    def __init__(self,
                 root: str,
                 extensions=IMG_EXTENSIONS,
                 loader: Callable[[str], Any] = default_loader,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 is_valid_file: Optional[Callable[[str], bool]] = None,
                 separator: str = "-"):

        super().__init__(root, transform=transform,
                         target_transform=target_transform)

        cls_combinations, classes, class_to_idx = self._find_classes(
            self.root, separator)
        samples = self._make_dataset(
            root, class_to_idx, extensions, is_valid_file, separator)

        self.loader = loader
        self.extenstions = extensions

        self.cls_combinations: List = cls_combinations
        self.classes: List = classes
        self.class_to_idx: Dict = class_to_idx
        self.idx_to_classes = {y: x for x, y in class_to_idx.items()}
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.imgs = self.samples # copied from ImageFolder

    def _find_classes(self, directory: str, separator: str):
        class_combinations = sorted(entry.name.split(
            separator) for entry in os.scandir(directory) if entry.is_dir())
        if not class_combinations:
            raise FileNotFoundError(
                f"Couldn't find any class combinations folder in {directory}.")
        classes = sorted(set(_flatten_list(class_combinations)))
        classes_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        return class_combinations, classes, classes_to_idx

    def _make_dataset(self, root, class_to_idx, extensions, is_valid_file,
                      separator):

        directory = os.path.expanduser(root)

        if class_to_idx is None:
            _, _, class_to_idx = self._find_classes(directory, separator)
        elif not class_to_idx:
            raise ValueError(
                "'class_to_index' must have at least one entry to collect any samples.")

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError(
                "Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:
            def is_valid_file(x: str) -> bool:
                # type: ignore[arg-type]
                return has_file_allowed_extension(x, extensions)

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []
        available_classes = set()  # TODO: Make safety check for classes like in ImageFolder

        class_folders = glob(root + "/*/", recursive=True)
        num_classes = len(class_to_idx)

        for target_class_dir in class_folders:
            if not os.path.isdir(target_class_dir):
                continue
            target_classes = Path(target_class_dir).stem.split(separator)
            target_indexes = [class_to_idx[target_class]
                              for target_class in target_classes]

            # Binary target, for each class = 1 at specifc indexes
            target_array = _create_target(target_indexes, num_classes)

            for root, _, fnames in sorted(
                    os.walk(target_class_dir, followlinks=True)):
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