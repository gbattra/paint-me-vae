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


def generate_and_save_images(tag, subdir, model, epoch, test_sample):
    """
    Generates an image using the VAE and saves it to a directory.
    :param tag: the tag of the image
    :param subdir: the sub dir to save to
    :param model: the model to use
    :param epoch: the epoch to use to label the image
    :param test_sample: the sample to generate
    :return: None
    """
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    fig, ax = plt.subplots(5, 4)

    for i in range(predictions.shape[0]):
        c = i // 5
        r = i % 5
        ax[r, (c * 2) + 0].imshow(test_sample[i, :, :, :])
        ax[r, (c * 2) + 1].imshow(predictions[i, :, :, :])

    plt.savefig(f'images/{subdir}/{tag.strip()}_image_at_epoch_{epoch}.png')
    # plt.show()
    plt.close(fig)