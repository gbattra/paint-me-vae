# Greg Attra
# 04/24/2021

import tensorflow as tf
import numpy as np
import time
import utils


class VaeTrainer:
    def __init__(self, name, dataset, vae):
        self.name = name
        self.dataset = dataset
        self.vae = vae

    def log_normal_pdf(self, sample, mean, logvar):
        """
        Compute the log normal pdf for the loss function.
        :param sample: the sample to compute with
        :param mean: the mean to compute with
        :param logvar: the logvar to compute with
        :return: the log normal pdf
        """
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=1)

    def compute_loss(self, model, x):
        """
       Computes the total loss.
       :param model: the model to train
       :param x: the inputs
       :return: the total loss
       """
        mean, logvar = model.encode(x)
        z = model.reparameterize(mean, logvar)
        x_logit = model.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    @tf.function
    def train_step(self, model, x, optimizer):
        """Executes one training step and returns the loss.

       This function computes the loss and gradients, and uses the latter to
       update the model's parameters.
       """
        with tf.GradientTape() as tape:
            loss = self.compute_loss(model, x)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    def train(self, epochs):
        """
       Trains the vae against the data.
       :param epochs: the epochs to train for
       :return: None
       """
        optimizer = tf.keras.optimizers.Adam()
        for test_batch in self.dataset.val_data.take(1):
            test_sample = test_batch[0:10, :, :, :]

        for epoch in range(1, epochs + 1):
            start_time = time.time()
            for train_x in self.dataset.train_data:
                self.train_step(self.vae, train_x, optimizer)
            end_time = time.time()

            loss = tf.keras.metrics.Mean()
            for test_x in self.dataset.val_data:
                loss(self.compute_loss(self.vae, test_x))
            elbo = -loss.result()
            print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
                  .format(epoch, elbo, end_time - start_time))
            utils.generate_and_save_images(self.name, self.vae, epoch, test_sample)

        return self

