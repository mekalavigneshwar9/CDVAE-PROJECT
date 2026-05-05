import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class CrystalEncoder(nn.Module):
    def __init__(self, num_atom_types=120, emb_dim=64, hidden_dim=128, latent_dim=128):
        super().__init__()

        self.embedding = nn.Embedding(num_atom_types, emb_dim)

        self.conv1 = GCNConv(emb_dim + 3, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, data):
        x, edge_index, batch, pos = data.x, data.edge_index, data.batch, data.pos

        x = self.embedding(x)
        x = torch.cat([x, pos], dim=1)

        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h = F.relu(self.conv3(h, edge_index))

        h = global_mean_pool(h, batch)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar