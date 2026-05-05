import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr

from encoder import CrystalEncoder

# allow loading
torch.serialization.add_safe_globals([Data, DataEdgeAttr])

dataset = torch.load("data/processed/dataset.pt", weights_only=False)

loader = DataLoader(dataset, batch_size=4)

model = CrystalEncoder()

for batch in loader:
    mu, logvar = model(batch)

    print("Mu shape:", mu.shape)
    print("LogVar shape:", logvar.shape)
    break