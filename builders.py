# Greg Attra
# 04/25/2021

"""
Model Builder class which uses a configuration object to generate a Keras model.
"""

import tensorflow as tf


class Builder:
    def build_model(self, config):
        raise NotImplementedError

    def build_layer(self, config):
        """
        Delegates to the proper layer building function given the layer type specified in the config.
        :param config: the config for the layer
        :return: the new layer
        """
        if config["type"] == "conv2d":
            return self.build_conv2d_layer(config)
        if config["type"] == "dense":
            return self.build_dense_layer(config)
        if config["type"] == "flatten":
            return tf.keras.layers.Flatten()

    def build_dense_layer(self, config):
        """
        Builds a Dense layer.
        :param config: the config for the layer
        :return: the new layer
        """
        return tf.keras.layers.Dense(config["n_units"], activation="relu")

    def build_conv2d_layer(self, config):
        """
        Builds a Conv layer.
        :param config: the config for the layer
        :return: the new layer
        """
        return tf.keras.layers.Conv2D(
            config["n_filters"],
            config["filter_size"],
            activation="relu",
            strides=config["strides"],
            padding="same")


class EncoderBuilder(Builder):
    def __init__(self):
        super(EncoderBuilder, self).__init__()

    def build_model(self, config):
        """
        Builds the model using the provided configuration.
        :param config: the configuration specifying the model
        :return: the generated Keras model
        """
        inputs = tf.keras.Input(shape=config["input_shape"])
        x = self.build_layer(config["layers"][0])(inputs)
        for layer in config["layers"][1:]:
            x = self.build_layer(layer)(x)

        z_mean = tf.keras.layers.Dense(config["latent_dim"], name="z_mean")(x)
        z_log_var = tf.keras.layers.Dense(config["latent_dim"], name="z_log_var")(x)
        z = SamplingLayer()([z_mean, z_log_var])
        return tf.keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")


class DecoderBuilder(Builder):
    def __init__(self):
        super(DecoderBuilder, self).__init__()

    def build_model(self, config):
        inputs = tf.keras.Input(shape=(config["latent_dim"],))
        x = self.build_layer(config["layers"][0])(inputs)
        for layer in config["layers"][1:]:
            x = self.build_layer(layer)(x)

        return tf.keras.Model(inputs, x, name="decoder")


class SamplingLayer(tf.keras.layers.Layer):
    """
    The sample layer of the VAE, or the encoding of the input.
    """

    def call(self, inputs, *args, **kwargs):
        """
        Override of built in Keras function. Uses reparameterization trick to sample z.
        :param inputs: the inputs from the previous layer
        :return: the encoding of the inputs
        """

        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

