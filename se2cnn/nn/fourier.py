import torch
from torch import nn
import torch.nn.functional as F
import math as m


class Fourier(nn.Module):

    def __init__(self, N, lmax=None):
        super().__init__()

        self.N = N
        self.lmax = lmax if lmax is not None else (N - 1) // 2
        self._create_fourier_basis()

    # Construct the discretized basis used in the forward and inverse Fourier Transform
    def _create_fourier_basis(self):
            theta_grid = torch.linspace(0, 2 * m.pi - 2 * m.pi / self.N, self.N)
            basis_stack = [];
            for l in range(self.lmax + 1):
                norm_factor = m.sqrt(2/self.N) if l != 0 else m.sqrt(1/self.N)
                basis = norm_factor * torch.stack((torch.cos(l * theta_grid), torch.sin(l * theta_grid)))
                basis_stack.append(basis)
            self.register_buffer("basis", torch.stack(basis_stack, dim=0))  # [L, M, num_theta)

    # Fourier Transform
    def ft(self, x):
        # in shape: [B, C, N, ...] -> out shape: [B, C, lmax, 2, ...]
        return torch.einsum('bca...,lma->bclm...', x, self.basis)

    # Inverse Fourier Transform
    def ift(self, x):
        # in shape: [B, C, lmax, 2, ...] -> out shape: [B, C, N, ...]
        return torch.einsum('bclm...,lma->bca...', x, self.basis)

    # Rotate a SO(2) irreps
    def rotate_irrep(self, x_hat, theta, l):
        # x_hat.shape = [B, C, 2, ...], theta.shape = [B, 1, ...]
        return torch.einsum('bij...,bcj...->bci...', self.rotation_matrix(theta, l), x_hat)

    # Rotate SO(2) signal via Fourier domain
    def rotate_signal(self, x, theta):
        x_hat = self.ft(x)
        for l in range(1,self.lmax+1):
            x_hat[:,:,l] = self.rotate_irrep(x_hat[:,:,l].clone(), theta, l)  # clone bc of inplace operation
        x = self.ift(x_hat)
        return x

    # SO(2) Rotation matrix of frequency l
    def rotation_matrix(self, theta, l=1):
        # in shape: [B, 1, ...], out shape: [B, 2, 2, ...]
        return torch.cat([torch.cos(l * theta), -torch.sin(l * theta), torch.sin(l * theta), torch.cos(l * theta)],
                           dim=1).unflatten(1, [2, 2])

    # Get only the lth Fourier coefficient
    def regular_to_irrep(self, x, l):
        # x.shape = [B, C, N, ...], basis.shape = [lmax, 2, N, ...]
        return torch.einsum('bca...,ma->bcm...', x, self.basis[l])

    # Reconstruct from only 1 Fourier coefficient
    def irrep_to_regular(self, x_hat, l):
        # x_hat.shape = [B, C, 2, ...], basis.shape = [lmax, 2, N, ...]
        return torch.einsum('bcm...,ma->bca...', x_hat, self.basis[l])

    # When self.N is even we need to band-limit the signal in order to have exact rotations
    def band_limit_signal(self, x):
        return self.ift(self.ft(x))

    # The default forward is the Fourier Transform
    def forward(self, x):
        return self.ft(x)


if __name__ == "__main__":
    num_theta=8
    batch_size, dim, X, Y = 16, 1, 1, 1

    signal = torch.randn(batch_size,dim,num_theta,X,Y)
    theta = torch.tensor([0.5 * 2 * torch.pi / num_theta]).repeat(batch_size)

    layer=Fourier(num_theta)

    print('direct rotation')
    signal_rot = layer.rotate_signal(signal,theta)
    print(signal[0, 0, :, 0, 0])
    print(signal_rot[0, 0, :, 0, 0])
    signal_rot = layer.rotate_signal(signal_rot, theta)
    print(signal_rot[0, 0, :, 0, 0])

    print('first band-limit then rotate')
    signal_bl = layer.band_limit_signal(signal)
    signal_bl_rot = layer.rotate_signal(signal_bl, theta)
    print(signal[0, 0, :, 0, 0])
    print(signal_bl[0, 0, :, 0, 0])
    print(signal_bl_rot[0, 0, :, 0, 0])
    signal_bl_rot = layer.rotate_signal(signal_bl_rot, theta)
    print(signal_bl_rot[0, 0, :, 0, 0])
