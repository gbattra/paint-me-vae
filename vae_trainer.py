# Greg Attra
# 04/24/2021

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt


class VaeTrainer:
   def __init__(self, dataset, vae):
       self.dataset = dataset
       self.vae = vae

   def log_normal_pdf(self, sample, mean, logvar, raxis=1):
       log2pi = tf.math.log(2. * np.pi)
       return tf.reduce_sum(
           -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
           axis=raxis)

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
           self.generate_and_save_images(self.vae, epoch, test_sample)

       return self

   def generate_and_save_images(self, model, epoch, test_sample):
       mean, logvar = model.encode(test_sample)
       z = model.reparameterize(mean, logvar)
       predictions = model.sample(z)
       fig = plt.figure(figsize=(4, 4))

       for i in range(predictions.shape[0]):
           plt.subplot(4, 4, i + 1)
           plt.imshow(predictions[i, :, :, 0], cmap='gray')
           plt.axis('off')

       # tight_layout minimizes the overlap between 2 sub-plots
       plt.savefig('images/training/image_at_epoch_{:04d}.png'.format(epoch))
       plt.show()
