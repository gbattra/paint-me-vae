#!/usr/bin/env python

# Greg Attra
# 04/25/2021

"""
Executable to run the painter program.
"""

import tensorflow as tf
import sys

from vae import Vae
from painter_controller import PainterController


def main():
    """
    Run the painter app.
    :return: 0 for success, -1 otherwise
    """
    portrait_encoder = tf.keras.models.load_model("models/thumbnails_vae/encoder")
    portrait_decoder = tf.keras.models.load_model("models/thumbnails_vae/decoder")
    portrait_vae = Vae(portrait_encoder, portrait_decoder)

    painting_encoder = tf.keras.models.load_model("models/paintings_vae/encoder")
    painting_decoder = tf.keras.models.load_model("models/paintings_vae/decoder")
    painting_vae = Vae(painting_encoder, painting_decoder)

    controller = PainterController(portrait_vae, painting_vae)

    path = str(sys.argv[1]).strip()
    mode = str(sys.argv[2]).strip()
    if mode == "portrait_to_painting":
        painting = controller.portrait_to_painting(path)

    return 0


if __name__ == '__main__':
    main()
