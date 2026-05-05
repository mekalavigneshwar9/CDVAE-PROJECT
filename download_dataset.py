from mp_api.client import MPRester
from pymatgen.core import Structure
import os

API_KEY = "7PvFNU2hpzLLf4O1S9NugkpoZzoS9SLW"
os.makedirs("data/raw", exist_ok=True)

with MPRester(API_KEY) as mpr:
    materials = mpr.materials.summary.search(
        num_elements=(2, 4),
        fields=["structure"],
        chunk_size=500
    )

print(f"Downloaded {len(materials)} structures")

# Increased to 4000 to ensure we have >1000 valid structures after preprocessing filters!
for i, mat in enumerate(materials[:4000]):
    structure = mat.structure
    structure.to(
        fmt="cif",
        filename=f"data/raw/structure_{i}.cif"
    )
