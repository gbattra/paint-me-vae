# Greg Attra
# 04/24/2021

"""
Credit: https://keras.io/examples/generative/vae/
"""

import numpy as np
import tensorflow as tf

from encoder_builder import EncoderBuilder


class VAE(tf.keras.Model):
    def __init__(self):
        super(VAE, self).__init__()

    def encode(self, inputs):
        raise NotImplementedError

    def decode(self, inputs):
        raise NotImplementedError


class Encoder(VAE):
    def __init__(self, config):
        super(VAE, self).__init__()
        self.model = EncoderBuilder().build_model(config)

    def encode(self, inputs):
        raise NotImplementedError

    def decode(self, inputs):
        raise NotImplementedError


