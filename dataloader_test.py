import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr

# allow loading
torch.serialization.add_safe_globals([Data, DataEdgeAttr])

# load dataset
dataset = torch.load("data/processed/dataset.pt", weights_only=False)

# create DataLoader
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# iterate through batches
for batch in loader:
    print(batch)
    print("Node shape:", batch.x.shape)
    print("Edge shape:", batch.edge_index.shape)
    print("Batch vector:", batch.batch.shape)
    break