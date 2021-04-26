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
