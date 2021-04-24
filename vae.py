# Greg Attra
# 04/24/2021

"""
Credit: https://keras.io/examples/generative/vae/
"""

import numpy as np
import tensorflow as tf

from builders import EncoderBuilder, DecoderBuilder


class VAE(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super(VAE, self).__init__()
        self.encoder = EncoderBuilder().build_model(config)
        self.decoder = DecoderBuilder().build_model(config)

    def encode(self, inputs):
        raise NotImplementedError

    def decode(self, inputs):
        raise NotImplementedError

    def train_step(self, data):
        return {}




