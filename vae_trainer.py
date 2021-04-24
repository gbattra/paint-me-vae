# Greg Attra
# 04/24/2021

import tensorflow as tf


class VaeTrainer:
    def __init__(self, epochs, batch_size, vae):
        self.epochs = epochs
        self.batch_size = batch_size
        self.vae = vae

    def train(self, data):
        self.vae.compile(optimize=tf.keras.optimizers.Adam())
        self.vae.fit(data, epochs=self.epochs, batch_size=self.batch_size)
