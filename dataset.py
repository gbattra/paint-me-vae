# Greg Attra
# 04/24/2021

import tensorflow as tf


class Dataset:
   """
   The Dataset class is a wrapper around keras functionality to read image data from a directory.
   """
   def __init__(self):
       self.train_data = None
       self.val_data = None
       self.size = 0

   def load(self, dir, dims):
       """
       Loads the image data from the directory and reshapes it the specified dims.
       :param dir: the directory to read from
       :param dims: the dims to resize to
       :param batch_size: the batch size for data batches
       :return: this object
       """
       self.train_data = tf.keras.preprocessing.image_dataset_from_directory(
           dir,
           labels="inferred",
           label_mode=None,
           class_names=None,
           color_mode="rgb",
           image_size=dims,
           validation_split=0.2,
           subset="training",
           seed=128,
           shuffle=True)
       self.val_data = tf.keras.preprocessing.image_dataset_from_directory(
           dir,
           labels="inferred",
           label_mode=None,
           class_names=None,
           color_mode="rgb",
           image_size=dims,
           validation_split=0.2,
           subset="validation",
           seed=128,
           shuffle=True)

       return self

   def format(self):
       """
       Formats/normalizes the data.
       :return: this object
       """
       norm_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255.)
       self.train_data = self.train_data.map(lambda x: (norm_layer(x)))
       self.val_data = self.val_data.map(lambda x: (norm_layer(x)))

       return self

   def configure(self):
       """
       Configures dataset for performance.
       :return: this object
       """
       self.train_data = self.train_data.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
       self.val_data = self.val_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

       return self
