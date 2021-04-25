# Greg Attra
# 04/24/2021

"""
Credit: https://keras.io/examples/generative/vae/
"""

import numpy as np
import tensorflow as tf

from builders import EncoderBuilder, DecoderBuilder


class Vae(tf.keras.Model):
   def __init__(self, config, **kwargs):
       """
       Initialize the VAE. Instantiate an encoder and a decoder. Setup metric trackers.
       :param config: the configuration for the encoder and decoder
       :param kwargs: default arg
       """
       super(Vae, self).__init__(**kwargs)
       self.encoder = EncoderBuilder().build_model(config["encoder"])
       self.decoder = DecoderBuilder().build_model(config["decoder"])
       self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
       self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
       self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

   def call(self, inputs, training=None, mask=None):
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
       mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
       return mean, logvar

   def reparameterize(self, mean, logvar):
       eps = tf.random.normal(shape=mean.shape)
       return eps * tf.exp(logvar * .5) + mean

   def decode(self, z, apply_sigmoid=False):
       logits = self.decoder(z)
       if apply_sigmoid:
           probs = tf.sigmoid(logits)
           return probs
       return logits
