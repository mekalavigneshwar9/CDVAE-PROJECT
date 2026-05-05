import torch
import torch.nn as nn

from encoder import CrystalEncoder
from decoder import CrystalDecoder
from diffusion import CoordinateDiffusion


class CDVAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = CrystalEncoder()
        self.decoder = CrystalDecoder()
        self.diffusion = CoordinateDiffusion()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, data):
        mu, logvar = self.encoder(data)
        z = self.reparameterize(mu, logvar)

        atoms, lattice = self.decoder(z)

        return mu, logvar, atoms, lattice, z