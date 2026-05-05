import os
import torch
from tqdm import tqdm
from pymatgen.core import Structure
from torch_geometric.data import Data

CUTOFF_RADIUS = 4.5
RAW_DIR = "data/raw"
PROCESSED_PATH = "data/processed/dataset.pt"


def structure_to_graph(cif_path):
    try:
        structure = Structure.from_file(cif_path)
    except:
        return None

    atomic_numbers = [site.specie.number for site in structure]
    num_atoms = len(atomic_numbers)

    # ---- FILTER 1: only size control (keep it simple) ----
    if num_atoms < 3 or num_atoms > 50:
        return None

    # ---- FILTER 2: remove extreme/invalid elements only ----
    # allow most elements but avoid very large atomic numbers
    if max(atomic_numbers) > 50:
        return None

    # ---- FIXED INDEXING (1-based, 0 is padding) ----
    x = torch.tensor(atomic_numbers, dtype=torch.long)

    # ---- positions ----
    frac_coords = torch.tensor(structure.frac_coords, dtype=torch.float)

    # ---- lattice ----
    a, b, c = structure.lattice.abc
    alpha, beta, gamma = structure.lattice.angles

    lattice = torch.tensor([
        a / 10.0,
        b / 10.0,
        c / 10.0,
        alpha / 180.0,
        beta / 180.0,
        gamma / 180.0
    ], dtype=torch.float)

    # ---- neighbors ----
    neighbors = structure.get_all_neighbors(CUTOFF_RADIUS)

    edge_index = []
    edge_attr = []

    for i, neighs in enumerate(neighbors):
        for neighbor in neighs:
            j = neighbor.index
            edge_index.append([i, j])
            edge_attr.append([neighbor.nn_distance])

    # ---- SAFETY ----
    if len(edge_index) == 0:
        return None

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(
        x=x,
        pos=frac_coords,
        edge_index=edge_index,
        edge_attr=edge_attr,
        lattice=lattice
    )


def process_all():
    dataset = []

    files = os.listdir(RAW_DIR)

    for file in tqdm(files):
        if file.endswith(".cif"):
            path = os.path.join(RAW_DIR, file)

            data = structure_to_graph(path)
            if data is not None:
                dataset.append(data)

    os.makedirs("data/processed", exist_ok=True)
    torch.save(dataset, PROCESSED_PATH)

    print("\nProcessed", len(dataset), "structures")
    print("Saved to:", PROCESSED_PATH)


if __name__ == "__main__":
    process_all()