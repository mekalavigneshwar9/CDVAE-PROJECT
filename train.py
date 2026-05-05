import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from model import CDVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_atoms = 50

# -------- LOAD DATA --------
dataset = torch.load("data/processed/dataset.pt", weights_only=False)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

torch.save(val_dataset, "data/processed/val_dataset.pt")


# -------- PAD --------
def pad(batch):
    B = batch.ptr.shape[0] - 1

    pos = torch.zeros((B, max_atoms, 3), device=device)
    atoms = torch.zeros((B, max_atoms), dtype=torch.long, device=device)

    for i in range(B):
        s, e = batch.ptr[i], batch.ptr[i + 1]
        n = e - s

        pos[i, :n] = batch.pos[s:e]
        atoms[i, :n] = batch.x[s:e]

    lattice = batch.lattice.view(B, 6)
    return atoms, pos, lattice


# -------- MODEL --------
model = CDVAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# -------- DIFFUSION SCHEDULES --------
timesteps = 100
beta = torch.linspace(1e-4, 0.02, timesteps).to(device)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

# -------- TRAIN --------
for epoch in range(120):
    model.train()
    total_loss = 0

    for batch in train_loader:
        batch = batch.to(device)

        mu, logvar, atoms_pred, lattice_pred, z = model(batch)

        target_atoms, target_pos, target_lattice = pad(batch)

        B, N, C = atoms_pred.shape

        atoms_pred = atoms_pred.view(B * N, C)
        target_atoms = target_atoms.view(B * N)

        # Learn to predict 0 for padded positions by applying loss to all tokens
        atom_loss = F.cross_entropy(atoms_pred, target_atoms)

        # -------------------------------------
        # DDPM FORWARD PROCESS (Coordinate Diffusion)
        # -------------------------------------
        # 1. Sample random timestep for each crystal in batch
        t = torch.randint(0, timesteps, (B,)).to(device)
        a_bar = alpha_bar[t].view(B, 1, 1)

        # 2. Add Gaussian noise to true coordinates
        noise = torch.randn_like(target_pos)
        pos_t = torch.sqrt(a_bar) * target_pos + torch.sqrt(1 - a_bar) * noise

        # 3. Predict the noise using diffusion network
        # We pass t as normalized float [0, 1]
        predicted_noise = model.diffusion(pos_t, t.float().unsqueeze(-1) / timesteps, z)

        # 4. Mask out the padded atoms so we only calculate loss on real atoms
        mask_3d = target_atoms.view(B, N) != 0
        
        # 5. Diffusion Loss (MSE between true noise and predicted noise)
        pos_loss = F.mse_loss(predicted_noise[mask_3d], noise[mask_3d])
        # -------------------------------------

        lattice_loss = ((lattice_pred - target_lattice) ** 2).mean()

        kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        loss = (
            3.0 * atom_loss +
            2.0 * pos_loss +
            0.5 * lattice_loss +
            0.0001 * kl_loss
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch} | Loss: {total_loss / len(train_loader):.3f}")

torch.save(model.state_dict(), "model.pth")
print("Model saved")