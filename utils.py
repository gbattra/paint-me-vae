# Greg Attra
# 04/25/2021

"""
Utility functions for managing images.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


def read_img(path):
    """
    Reads an image from a path and preprocesses it for the network.
    :param path: the path to the image
    :return: the keras-ready image object
    """
    img = Image.open(path)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img /= 255.
    img = tf.image.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0)
    return img


def generate_and_save_images(tag, model, epoch, test_sample):
    """
    Generates an image using the VAE and saves it to a directory.
    :param tag: the tag of the image
    :param model: the model to use
    :param epoch: the epoch to use to label the image
    :param test_sample: the sample to generate
    :return: None
    """
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    plt.savefig(f'images/training/{tag.strip()}_image_at_epoch_{epoch}.png')
    # plt.show()
    plt.close(fig)