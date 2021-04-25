#!/usr/bin/env python

# Greg Attra
# 04/25/2021

"""
Executable to train components of the painter system.
"""

import matplotlib.pyplot as plt
import sys

from vae_trainer import VaeTrainer
from vae import Vae
from dataset import Dataset
from configs import encoder_config, decoder_config


def main():
    """
    Entrypoint to the program.
    :return: success/failure code
    """
    name = str(sys.argv[1]).strip()
    dataset = Dataset().load(f'data/{name}', (128, 128)).format()
    configs = {
        "encoder": encoder_config,
        "decoder": decoder_config
    }
    vae = Vae(configs)
    vae.encoder.summary()
    vae.decoder.summary()
    trainer = VaeTrainer(name, dataset, vae)
    trainer.train(30)
    trainer.vae.encoder.save(f'models/{name}_vae/encoder')
    trainer.vae.decoder.save(f'models/{name}_vae/decoder')

    return 0


if __name__ == '__main__':
    main()
