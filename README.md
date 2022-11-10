# se2cnn
A minimalistic library for regular SE2 G-CNNs. Just the essential conv layers + point-wise Fourier transforms for turning regular fibers (SO(2) signals) into irreps (Fourier coefficients). The latter is useful if one wants to predicte vectors (forces, velocities, landmarks, ...).

This repository contains the pytorch code for SE(2) group convolutional networks. An adaption of the tensorflow code used in our original paper:

Bekkers, E., Lafarge, M., Veta, M., Eppenhof, K., Pluim, J., Duits, R.: Roto-translation covariant convolutional networks for medical image analysis. Accepted at MICCAI 2018, arXiv preprint arXiv:1804.03393 (2018). Available at: https://arxiv.org/abs/1804.03393

The code follows the simple idea of group convolutional neural networks via template matching. The intuition and many details are provided in the Youtube lectures and materials given at [https://uvagedl.github.io](https://uvagedl.github.io).
