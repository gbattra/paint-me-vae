# Greg Attra
# 04/24/2021

import tensorflow as tf


class Dataset:
    """
    The Dataset class is a wrapper around keras functionality to read image data from a directory.
    """
    def __init__(self):
        self.data = None
        self.size = 0

    def load(self, dir, dims):
        """
        Loads the image data from the directory and reshapes it the specified dims.
        :param dir: the directory to read from
        :param dims: the dims to resize to
        :return: None
        """
        self.data = tf.keras.preprocessing.image_dataset_from_directory(
            dir,
            labels="inferred",
            label_mode="int",
            class_names=None,
            color_mode="rgb",
            batch_size=32,
            image_size=dims,
            shuffle=True,
            validation_split=0.2)

