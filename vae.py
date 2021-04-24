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
        """
        Initialize the VAE. Instantiate an encoder and a decoder. Setup metric trackers.
        :param config: the configuration for the encoder and decoder
        :param kwargs: default arg
        """
        super(VAE, self).__init__()
        self.encoder = EncoderBuilder().build_model(config)
        self.decoder = DecoderBuilder().build_model(config)
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    def encode(self, inputs):
        """
        Encode the inputs.
        :param inputs: the inputs to encode
        :return: the encoding of the input
        """
        raise self.encoder(inputs)

    def decode(self, inputs):
        """
        Decode the inputs.
        :param inputs: the inputs to decode
        :return: the decoded inputs
        """
        raise self.decoder(inputs)

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

    def train_step(self, data):
        """
        Custom train step which combines kl and reconstruction losses to gradient computation.
        :param data: the data to fit to
        :return: the loss dictionary
        """
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconst = self.decoder(z)
            reconst_loss = tf.reduce_mean(
                tf.reduce_sum(tf.keras.losses.binary_crossentropy(data, reconst), axis=(1, 2)))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconst_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconst_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        }
