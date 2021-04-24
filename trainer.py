#!/usr/bin/env python

# Greg Attra
# 04/25/2021

"""
Executable to train components of the painter system.
"""

from vae_trainer import VaeTrainer
from vae import Vae
from dataset import Dataset


def main():
    """
    Entrypoint to the program.
    :return: success/failure code
    """
    data = Dataset().load("data/thumbnails", 128)
    return 0


if __name__ == '__main__':
    main()