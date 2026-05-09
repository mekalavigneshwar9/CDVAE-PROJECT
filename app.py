import os
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
from collections import Counter
import warnings
import random

warnings.filterwarnings("ignore")

from model import CDVAE
from pymatgen.core import Structure, Lattice
from pymatgen.core.periodic_table import Element

# -------------------- PATHS --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pth")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# -------------------- APP SETUP --------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- LOAD MODEL --------------------
print(f"Initializing model on {device}...")
model = None

try:
    model = CDVAE().to(device)
    if os.path.exists(MODEL_PATH):
        print(f"Loading weights from {MODEL_PATH}...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("Model loaded successfully!")
    else:
        print(f"ERROR: {MODEL_PATH} not found.")
except Exception as e:
    print(f"Model initialization FAILED: {e}")

# -------------------- DIFFUSION PARAMS --------------------
timesteps = 20  # Reduced for fast inference (~3-5s on CPU)
beta = torch.linspace(1e-4, 0.02, timesteps).to(device)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

# -------------------- REQUEST SCHEMA --------------------
class GenerateRequest(BaseModel):
    pass


@app.get("/")
def read_root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "model_loaded": model is not None,
        "device": str(device)
    }


@app.post("/api/generate")
def generate_crystal(req: GenerateRequest):
    if model is None:
        return {"error": "Model is still loading. Please wait a few seconds and try again."}

    # Use a random seed for each generation
    used_seed = random.randint(0, 999999)
    torch.manual_seed(used_seed)
    np.random.seed(used_seed)

    with torch.no_grad():

        # ---- Latent sampling ----
        z = torch.randn(1, 128).to(device)

        # ---- Decode ----
        atoms, lattice = model.decoder(z)

        # ---- VALIDATE OUTPUT SHAPES ----
        if atoms.ndim != 3:
            return {"error": "Invalid atoms output shape"}

        if lattice.shape[-1] != 6:
            return {"error": "Invalid lattice output shape"}

        # ---- DIFFUSION SAMPLING ----
        pos_t = torch.randn(1, 50, 3).to(device)

        for t in reversed(range(timesteps)):
            t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
            t_norm = t_tensor.float().unsqueeze(-1) / timesteps

            predicted_noise = model.diffusion(pos_t, t_norm, z)

            alpha_t = alpha[t]
            alpha_bar_t = alpha_bar[t]
            beta_t = beta[t]

            noise = torch.randn_like(pos_t) if t > 0 else torch.zeros_like(pos_t)

            pos_t = (
                1 / torch.sqrt(alpha_t)
                * (pos_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise)
                + torch.sqrt(beta_t) * noise
            )

            # ---- stability clamp ----
            pos_t = torch.clamp(pos_t, -10, 10)

            if t % 5 == 0:
                print(f"Diffusion step {t}/{timesteps}...")

        pos = pos_t % 1.0
        print("Diffusion complete. Processing atoms...")

    # -------------------- PROCESS OUTPUT --------------------
    pred_atoms = torch.argmax(atoms, dim=-1)[0].cpu().numpy()
    coords_frac = pos[0].cpu().numpy()

    PAD_IDX = 0
    mask = pred_atoms != PAD_IDX

    coords_frac = coords_frac[mask]
    atom_ids = pred_atoms[mask]

    # ---- VALIDATE ATOMS ----
    valid_ids = [int(z) for z in atom_ids if 1 <= int(z) <= 118]

    if len(valid_ids) == 0:
        return {"error": "Invalid or empty structure generated"}

    symbols = [Element.from_Z(z).symbol for z in valid_ids]

    # ---- FORMULA ----
    counts = Counter(symbols)
    formula = "".join(
        [f"{el}{counts[el]}" if counts[el] > 1 else el for el in counts]
    )

    # -------------------- LATTICE --------------------
    lat = lattice[0].cpu().numpy()

    # Scale lengths and angles with safer defaults
    # Raw outputs from model might be outside [0, 1], so we use sigmoid-like scaling
    a = float(np.clip(lat[0] * 5 + 5, 2.0, 15.0))
    b = float(np.clip(lat[1] * 5 + 5, 2.0, 15.0))
    c = float(np.clip(lat[2] * 5 + 5, 2.0, 15.0))
    
    alpha_ang = float(np.clip(lat[3] * 30 + 90, 60, 120))
    beta_ang = float(np.clip(lat[4] * 30 + 90, 60, 120))
    gamma_ang = float(np.clip(lat[5] * 30 + 90, 60, 120))
    
    print(f"Lattice: {a:.2f}, {b:.2f}, {c:.2f} | Angles: {alpha_ang:.1f}, {beta_ang:.1f}, {gamma_ang:.1f}")

    try:
        lattice_obj = Lattice.from_parameters(a, b, c, alpha_ang, beta_ang, gamma_ang)
    except Exception as e:
        print(f"Lattice error: {e}")
        return {"error": "Invalid lattice parameters generated"}

    # -------------------- STRUCTURE --------------------
    try:
        structure = Structure(
            lattice_obj,
            symbols,
            coords_frac,
            coords_are_cartesian=False
        )
    except Exception:
        return {"error": "Structure construction failed"}

    # -------------------- OUTPUTS --------------------
    print("Converting structure to CIF and calculating bonds...")
    cif_str = structure.to(fmt="cif")
    cart_coords = structure.cart_coords.tolist()

    # ---- Bonds (capped to avoid over-connected graphs) ----
    MAX_BONDS = 60  # Hard cap to keep the 3D graph readable
    try:
        neighbors = structure.get_all_neighbors(2.0)  # 2.0Å = tighter radius
        bonds = []
        drawn = set()

        for i, site_neighbors in enumerate(neighbors):
            for n in site_neighbors:
                j = int(n.index)
                if i == j:
                    continue  # Skip self-bonds
                bond = tuple(sorted((i, j)))
                if bond not in drawn:
                    drawn.add(bond)
                    bonds.append([i, j])
                if len(bonds) >= MAX_BONDS:
                    break
            if len(bonds) >= MAX_BONDS:
                break
    except Exception as e:
        print(f"Bond error: {e}")
        bonds = []

    return {
        "formula": formula,
        "elements": list(set(symbols)),
        "symbols": symbols,
        "coordinates": cart_coords,
        "bonds": bonds,
        "cif": cif_str
    }


# -------------------- FRONTEND --------------------
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:
    print(f"WARNING: Frontend directory not found at {FRONTEND_DIR}")


# -------------------- RUN --------------------
if __name__ == "__main__":
    import os
    import sys
    import uvicorn

    sys.path.append(BASE_DIR)

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )