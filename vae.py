# Greg Attra
# 04/24/2021

"""
Credit: https://keras.io/examples/generative/vae/
"""

import numpy as np
import tensorflow as tf


class Vae(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        """
       Initialize the VAE. Instantiate an encoder and a decoder. Setup metric trackers.
       :param config: the configuration for the encoder and decoder
       :param kwargs: default arg
       """
        super(Vae, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")

    def call(self, inputs, training=None, mask=None):
        """
        Override of base method.
        :param inputs: the inputs to process
        :param training: is this for training
        :param mask: mask for the inputs
        :return: the prediction
        """
        e = self.encode(inputs)
        d = self.decode(e)
        return d

    @property
    def metrics(self):
        """
       Builds list of metric trackers.
       :return: the list of metric trackers
       """
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker
        ]

    # https://www.tensorflow.org/tutorials/generative/cvae
    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        """
        Encode the input data.
        :param x: the input data
        :return: the encoding of the data
        """
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        """
        Implementation of the reparametrization trick
        :param mean: the mean of the data
        :param logvar: the logvar of the data
        :return: the z value
        """
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        """
        Decode the z value.
        :param z: the latent z value to decode
        :param apply_sigmoid: should the values be passed through a sigmoid
        :return: the decoding of the z value
        """
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
