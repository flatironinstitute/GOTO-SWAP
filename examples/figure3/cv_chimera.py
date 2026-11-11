
import numpy as np
from chimerax.atomic import AtomicStructure

from chimerax.core.commands import run

run(session, "close all")
run(session, "open /Users/gw/Documents/education/2026w/goto/1BRS.cif")
run(session, "hide solvent")
run(session, "hide ~(/A,D) atoms")

# ---- Settings ----
model = session.models.list()[0]
separations = [0.0, 60.0]  # Increase in COM separation (Å)
transparency = {k:v for k,v in zip(separations, [0, 75])}  # Transparency for each separation

# ---- Calculate original COMs ----
atoms = model.atoms
ca = atoms.filter(atoms.chain_ids == 'A')
cd = atoms.filter(atoms.chain_ids == 'D')

com_a = ca.coords.mean(axis=0)
com_d = cd.coords.mean(axis=0)

# Unit vector from A to D
v = com_d - com_a
v /= np.linalg.norm(v)

# ---- Create three structural arrangements ----
for separation in separations:

    # Duplicate original model
    m = model.copy()
    # m.atoms.filter(m.atoms.residues.is_solvent).displays = False
    session.models.add([m])
    m.name = f"COM separation +{separation:.0f} A"

    atoms = m.atoms
    ca = atoms.filter(atoms.chain_ids == 'A')
    cd = atoms.filter(atoms.chain_ids == 'D')

    # Symmetric translation: total COM separation increase
    displacement = (separation / 2.0) * v

    ca.coords = ca.coords - displacement
    cd.coords = cd.coords + displacement

    # Show only chains A and D
    atoms.displays = np.isin(atoms.chain_ids, ['A', 'D'])

    # Apply transparency
    alpha = int(255 * (100 - transparency[separation]) / 100)

    colors = atoms.colors.copy()
    colors[:, 3] = alpha
    atoms.colors = colors

# Hide original model
model.display = False
run(session, "hide solvent")

print(f"Created structures at {separations} Å additional COM separation.")
