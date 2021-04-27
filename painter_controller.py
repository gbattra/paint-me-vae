# Greg Attra
# 04/25/2021

"""
Controller for the Painter program.
"""

import tensorflow as tf
import numpy as np
import utils

from vae import Vae
from PIL import Image
from dataset import Dataset


class PainterController:
    def __init__(self, portrait_vae, painting_vae):
        """
        Initialize the controller.
        :param portrait_vae: the portrait encoder/decoder
        :param painting_vae: the painting encoder/decoder
        """
        self.portrait_vae = portrait_vae
        self.painting_vae = painting_vae
        self.painting_to_portrait_vae = Vae(self.painting_vae.encoder, self.portrait_vae.decoder)
        self.portrait_to_painting_vae = Vae(self.portrait_vae.encoder, self.painting_vae.decoder)

    def portrait_to_painting(self, path):
        """
        Paints the portrait at the specified path.
        :param path: the path to the portrait img
        :return: the generated (painted) image
        """
        img = utils.read_img(path)
        gen = utils.generate_and_save_images(
            path.split('/')[-1].split('.')[0],
            "paintings",
            self.portrait_to_painting_vae,
            0,
            img)
        return gen

    def painting_to_portrait(self, path):
        """
        Converts the portrait into a picture.
        :param path: the path to the painting img
        :return: the generated image
        """
        img = utils.read_img(path)
        gen = utils.generate_and_save_images(
            path.split('/')[-1].split('.')[0],
            "paintings",
            self.painting_to_portrait_vae,
            0,
            img)
        return gen

    def generate_samples(self):
        """
        Generates sample paintings and portraits.
        :return: the generated samples
        """
        thumbnails_dataset = Dataset().load(f'data/thumbnails', (128, 128)).format()
        paintings_dataset = Dataset().load(f'data/paintings', (128, 128)).format()

        for test_batch in thumbnails_dataset.val_data.take(1):
            thumbnail_samples = test_batch[0:10, :, :, :]

        for test_batch in paintings_dataset.val_data.take(1):
            painting_samples = test_batch[0:10, :, :, :]

        thumb_gen = utils.generate_and_save_images(
            "thumbnail_samples",
            "paintings",
            self.portrait_to_painting_vae,
            0,
            thumbnail_samples)

        paint_gen = utils.generate_and_save_images(
            "painting_samples",
            "paintings",
            self.painting_to_portrait_vae,
            0,
            painting_samples)

        return thumb_gen, paint_gen
