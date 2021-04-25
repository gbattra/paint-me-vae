#!/usr/bin/env python

# Greg Attra
# 04/25/2021

"""
Executable to train components of the painter system.
"""

import matplotlib.pyplot as plt

from vae_trainer import VaeTrainer
from vae import Vae
from dataset import Dataset
from configs import encoder_config, decoder_config


def main():
   """
   Entrypoint to the program.
   :return: success/failure code
   """
   dataset = Dataset().load("data/thumbnails", (128, 128)).format().configure()
   configs = {
       "encoder": encoder_config,
       "decoder": decoder_config
   }
   vae = Vae(configs)
   vae.encoder.summary()
   vae.decoder.summary()
   trainer = VaeTrainer(dataset, vae)
   trainer.train(30)
   trainer.vae.save("model")

   return 0


if __name__ == '__main__':
   main()
