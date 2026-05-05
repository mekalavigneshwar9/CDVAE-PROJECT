import torch
import torch.nn as nn
import torch.nn.functional as F


class CrystalDecoder(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=128, max_atoms=50, num_atom_types=120):
        super().__init__()

        self.max_atoms = max_atoms

        self.expand = nn.Linear(latent_dim, hidden_dim * max_atoms)

        self.atom_head = nn.Linear(hidden_dim, num_atom_types)
        self.lattice_head = nn.Linear(latent_dim, 6)

    def forward(self, z):
        B = z.size(0)

        x = self.expand(z)
        x = x.view(B, self.max_atoms, -1)
        x = F.relu(x)

        atoms = self.atom_head(x)
        lattice = self.lattice_head(z)

        return atoms, lattice