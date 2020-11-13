"""
This script is used to create a data set class
which is used in implementing pytorch based NNs
It returns an item or sample of the data,
sample should contain everything in order to train
and evaluate your model.
"""

import torch
import numpy as np

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # takes care of corrupt images which do not have ending bit


class ClassificationDataset:

    def __init__(self, image_paths, targets, resize=None, augmentations=None):
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.augmentations = augmentations

    def __len__(self):
        """
        Returns total number of samples in the dataset
        """
        return len(self.image_paths)

    def __getitem__(self, item):
        """
        For a given item index, return everything that we need to train the model
        """
        image = Image.open(self.image_paths[item])
        image = image.convert('RGB')
        targets = self.targets[item]

        if self.resize is not None:
            image = image.resize((self.resize[1], self.resize[0]), resample=Image.BILINEAR)

        image = np.array(image)

        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented[image]

        """Pytorch expects images to be in CHW (Channel, height, widtg) as compare to HWC"""
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        output_dict = {
            "image": torch.tensor(image, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.long)
        }

        return output_dict
