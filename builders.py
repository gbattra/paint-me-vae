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
        if config["type"] == "conv2dt":
            return self.build_conv2dt_layer(config)
        if config["type"] == "reshape":
            return self.build_reshape_layer(config)
        if config["type"] == "flatten":
            return tf.keras.layers.Flatten()

    def build_conv2dt_layer(self, config):
        """
        Builds a conv2d transpose layer.
        :param config: the config for the layer
        :return: the new layer
        """
        return tf.keras.layers.Conv2DTranspose(
            config["n_filters"],
            config["filter_size"],
            activation="relu",
            strides=config["strides"],
            padding="same")

    def build_reshape_layer(self, config):
        """
        Builds a reshape layer.
        :param config: the config for the layer
        :return: the new layer
        """
        return tf.keras.layers.Reshape(config["shape"])

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
        layers = [tf.keras.layers.InputLayer(input_shape=config["input_shape"])]
        for layer in config["layers"]:
            layers.append(self.build_layer(layer))

        layers.append(tf.keras.layers.Dense(config["latent_dim"] * 2))

        return tf.keras.Sequential(layers, name="encoder")


class DecoderBuilder(Builder):
    def __init__(self):
        super(DecoderBuilder, self).__init__()

    def build_model(self, config):
        layers = [tf.keras.layers.InputLayer(input_shape=(config["latent_dim"],))]
        for layer in config["layers"]:
            layers.append(self.build_layer(layer))
        layers.append(tf.keras.layers.Conv2DTranspose(3, 3, activation="relu", padding="same"))

        return tf.keras.Sequential(layers, name="decoder")
