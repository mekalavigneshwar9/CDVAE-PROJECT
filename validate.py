import torch
import numpy as np
import matplotlib.pyplot as plt

from model import CDVAE
from pymatgen.core import Structure, Lattice
from pymatgen.core.periodic_table import Element

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = torch.load("data/processed/val_dataset.pt", weights_only=False)

model = CDVAE().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

import random

# Pick a random structure from the validation set instead of always the first one
rand_idx = random.randint(0, len(dataset) - 1)
batch = dataset[rand_idx].to(device)
print(f"Reconstructing validation structure #{rand_idx}")


with torch.no_grad():
    mu, logvar, atoms, lattice, z = model(batch)

    # -------- DIFFUSION SCHEDULES --------
    timesteps = 100
    beta = torch.linspace(1e-4, 0.02, timesteps).to(device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    # -------- DDPM REVERSE PROCESS --------
    # 1. Start with pure Gaussian noise for 50 atoms
    pos_t = torch.randn(1, 50, 3).to(device)

    # 2. Iteratively denoise
    for t in reversed(range(timesteps)):
        t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
        t_norm = t_tensor.float().unsqueeze(-1) / timesteps
        
        predicted_noise = model.diffusion(pos_t, t_norm, z)
        
        alpha_t = alpha[t]
        alpha_bar_t = alpha_bar[t]
        beta_t = beta[t]
        
        if t > 0:
            noise = torch.randn_like(pos_t)
        else:
            noise = torch.zeros_like(pos_t)
            
        # DDPM update equation
        pos_t = 1 / torch.sqrt(alpha_t) * (pos_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise) + torch.sqrt(beta_t) * noise

    # 3. Final coordinates wrapped to periodic fractional boundaries [0, 1]
    pos = pos_t % 1.0

pred_atoms = torch.argmax(atoms, dim=-1)[0].cpu().numpy()
coords_frac = pos[0].cpu().numpy()

# remove padding
mask = pred_atoms != 0
coords_frac = coords_frac[mask]
atom_ids = pred_atoms[mask]

symbols = [Element.from_Z(int(z)).symbol for z in atom_ids]

# Calculate required output formats
unique_elements = list(set(symbols))
from collections import Counter
counts = Counter(symbols)
formula = "".join([f"{elem}{count}" if count > 1 else elem for elem, count in counts.items()])

print("\n" + "="*55)
print("         CRYSTAL RECONSTRUCTION REPORT")
print("="*55)
print(f" ▶ Element Symbols    : {', '.join(unique_elements)}")
print(f" ▶ Chemical Formula   : {formula}")
print(" ▶ 3D Visualization   : Generating interactive graph...")

# build structure
lat = lattice[0].cpu().numpy()
a, b, c = lat[:3] * 10.0
alpha, beta, gamma = lat[3:] * 180

lattice_obj = Lattice.from_parameters(a, b, c, alpha, beta, gamma)

structure = Structure(
    lattice_obj,
    symbols,
    coords_frac,
    coords_are_cartesian=False
)

coords = structure.cart_coords

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

# Draw Atoms
for i, (x, y, z) in enumerate(coords):
    ax.scatter(x, y, z, s=150, depthshade=True)
    ax.text(x, y, z, f" {symbols[i]}", fontsize=12, fontweight='bold')

# Draw Bonds (connections between atoms within a 3.0 Angstrom radius)
neighbors = structure.get_all_neighbors(3.0)
drawn_bonds = set()

for i, site in enumerate(structure):
    for neighbor in neighbors[i]:
        j = neighbor.index
        # Avoid drawing the exact same line twice
        bond = tuple(sorted((i, j)))
        if bond in drawn_bonds:
            continue
        drawn_bonds.add(bond)

        x_coords = [coords[i][0], neighbor.coords[0]]
        y_coords = [coords[i][1], neighbor.coords[1]]
        z_coords = [coords[i][2], neighbor.coords[2]]
        ax.plot(x_coords, y_coords, z_coords, color='gray', alpha=0.5, linewidth=2)

ax.set_title("Reconstructed Crystal Structure & Atomic Connections")
ax.set_xlabel("X (Å)")
ax.set_ylabel("Y (Å)")
ax.set_zlabel("Z (Å)")

# Save optional CIF file
cif_filename = "generated_structure.cif"
structure.to(fmt="cif", filename=cif_filename)
print(f" ▶ Optional CIF File  : Saved to '{cif_filename}'")
print("="*55 + "\n")

plt.show()