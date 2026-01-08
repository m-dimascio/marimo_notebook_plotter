# COMSOL MOSFET Simulation Implementation Plan

This document outlines the plan for remaking the MOSFET simulation application using COMSOL Multiphysics via the MPh Python library.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Architecture](#architecture)
4. [Implementation Phases](#implementation-phases)
5. [API Reference](#api-reference)
6. [File Structure](#file-structure)
7. [Migration Guide](#migration-guide)

---

## Overview

### Current State

The existing application uses custom Python physics models with analytical approximations:
- Square-law MOSFET equations (gradual channel approximation)
- Simplified carrier concentration profiles
- NumPy-based mesh generation
- Plotly visualization

### Target State

Replace analytical models with COMSOL's rigorous finite-element semiconductor simulation:
- Drift-diffusion equations with Poisson solver
- Accurate carrier transport physics
- COMSOL's adaptive meshing
- Same Plotly/marimo visualization layer

### Benefits of COMSOL Integration

| Aspect | Current (Analytical) | COMSOL (FEM) |
|--------|---------------------|--------------|
| Physics accuracy | Long-channel approximation | Full drift-diffusion + Poisson |
| Short-channel effects | Not modeled | Included (DIBL, velocity saturation) |
| Quantum effects | Not modeled | Density-gradient option |
| Mobility models | Constant | Field-dependent, temperature-dependent |
| Mesh quality | Uniform structured | Adaptive refinement |
| Convergence | Always (analytical) | Iterative (may need tuning) |

---

## Prerequisites

### Software Requirements

```bash
# COMSOL Multiphysics (with Semiconductor Module license)
# - Version 5.6 or later recommended
# - Semiconductor Module required for drift-diffusion physics

# Python environment
pip install MPh==1.3.1
pip install numpy plotly marimo
```

### COMSOL License Configuration

MPh requires access to COMSOL's Java API. Ensure:
1. COMSOL is installed with valid license
2. Semiconductor Module is licensed
3. COMSOL server is accessible (standalone or client-server mode)

### Environment Setup

```python
import mph

# Start COMSOL server (standalone mode)
client = mph.start()

# Or connect to existing server
# client = mph.start(cores=4)  # Specify cores for parallel solving
```

---

## Architecture

### Module Structure

```
mosfet_nb/
├── src/mosfet_sim/
│   ├── __init__.py              # Package exports
│   ├── constants.py             # Physical constants (keep existing)
│   ├── materials.py             # Material properties (keep existing)
│   ├── device.py                # MOSFETParams dataclass (keep existing)
│   │
│   ├── comsol/                  # NEW: COMSOL integration layer
│   │   ├── __init__.py
│   │   ├── client.py            # MPh client management
│   │   ├── geometry.py          # Device geometry builder
│   │   ├── physics.py           # Semiconductor physics setup
│   │   ├── mesh.py              # Mesh configuration
│   │   ├── study.py             # Study/solver configuration
│   │   ├── extract.py           # Data extraction from COMSOL
│   │   └── model_builder.py     # High-level model orchestration
│   │
│   ├── visualization_3d.py      # Keep existing (works with extracted data)
│   └── plotting.py              # Keep existing
│
├── notebooks/
│   └── mosfet_explorer.py       # Update to use COMSOL backend
│
└── models/                      # NEW: Cached COMSOL models
    └── .gitkeep
```

### Data Flow

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   MOSFETParams  │────▶│  COMSOL Model    │────▶│  NumPy Arrays   │
│   (device.py)   │     │  (mph + Java)    │     │  (extracted)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
                                                 ┌─────────────────┐
                                                 │  Plotly/Marimo  │
                                                 │  Visualization  │
                                                 └─────────────────┘
```

---

## Implementation Phases

### Phase 1: COMSOL Client & Model Management

**File: `comsol/client.py`**

```python
"""COMSOL client lifecycle management."""
import mph
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

_client: Optional[mph.Client] = None

def get_client() -> mph.Client:
    """Get or create the global COMSOL client."""
    global _client
    if _client is None:
        _client = mph.start()
    return _client

def shutdown_client():
    """Cleanly shutdown the COMSOL client."""
    global _client
    if _client is not None:
        _client.clear()
        _client = None

@contextmanager
def comsol_session():
    """Context manager for COMSOL session."""
    client = get_client()
    try:
        yield client
    finally:
        # Don't shutdown - reuse client for performance
        pass

def create_model(name: str = "MOSFET") -> mph.Model:
    """Create a new COMSOL model."""
    client = get_client()
    return client.create(name)

def load_model(path: Path) -> mph.Model:
    """Load an existing COMSOL model."""
    client = get_client()
    return client.load(str(path))

def save_model(model: mph.Model, path: Path):
    """Save a COMSOL model to disk."""
    model.save(str(path))
```

---

### Phase 2: Geometry Builder

**File: `comsol/geometry.py`**

```python
"""COMSOL geometry construction for MOSFET device."""
import numpy as np
from ..device import MOSFETParams

def build_mosfet_geometry(model, params: MOSFETParams):
    """
    Create 3D MOSFET geometry in COMSOL.

    Device structure:
    - Substrate (p-type silicon body)
    - Source/Drain regions (n+ doped)
    - Gate oxide (SiO2)
    - Gate contact (metal)
    """
    java = model.java

    # Create 3D geometry component
    java.modelNode().create("comp1")
    geom = java.geom().create("geom1", 3)

    # Device dimensions
    L = params.channel_length  # meters
    W = params.channel_width
    tox = params.oxide_thickness

    # Derived dimensions
    sd_extension = 0.4 * L      # Source/drain extension
    L_total = L + 2 * sd_extension
    substrate_depth = 1.0 * L   # Depth into substrate
    junction_depth = 0.15 * substrate_depth

    # Convert to micrometers for COMSOL (typical semiconductor scale)
    scale = 1e6  # m to μm

    # 1. Substrate block (p-type body)
    geom.feature().create("substrate", "Block")
    geom.feature("substrate").set("size", [
        f"{L_total * scale}[um]",
        f"{W * scale}[um]",
        f"{substrate_depth * scale}[um]"
    ])
    geom.feature("substrate").set("pos", ["0", "0", f"{-substrate_depth * scale}[um]"])
    geom.feature("substrate").set("createselection", True)

    # 2. Source region (n+ doped)
    geom.feature().create("source", "Block")
    geom.feature("source").set("size", [
        f"{sd_extension * scale}[um]",
        f"{W * scale}[um]",
        f"{junction_depth * scale}[um]"
    ])
    geom.feature("source").set("pos", ["0", "0", f"{-junction_depth * scale}[um]"])
    geom.feature("source").set("createselection", True)

    # 3. Drain region (n+ doped)
    geom.feature().create("drain", "Block")
    geom.feature("drain").set("size", [
        f"{sd_extension * scale}[um]",
        f"{W * scale}[um]",
        f"{junction_depth * scale}[um]"
    ])
    geom.feature("drain").set("pos", [
        f"{(sd_extension + L) * scale}[um]",
        "0",
        f"{-junction_depth * scale}[um]"
    ])
    geom.feature("drain").set("createselection", True)

    # 4. Gate oxide
    geom.feature().create("oxide", "Block")
    geom.feature("oxide").set("size", [
        f"{L * scale}[um]",
        f"{W * scale}[um]",
        f"{tox * scale}[um]"
    ])
    geom.feature("oxide").set("pos", [
        f"{sd_extension * scale}[um]",
        "0",
        "0"
    ])
    geom.feature("oxide").set("createselection", True)

    # 5. Gate contact (thin layer above oxide)
    gate_thickness = tox * 0.5  # Gate metal thickness
    geom.feature().create("gate", "Block")
    geom.feature("gate").set("size", [
        f"{L * scale}[um]",
        f"{W * scale}[um]",
        f"{gate_thickness * scale}[um]"
    ])
    geom.feature("gate").set("pos", [
        f"{sd_extension * scale}[um]",
        "0",
        f"{tox * scale}[um]"
    ])
    geom.feature("gate").set("createselection", True)

    # Build geometry
    geom.run("fin")

    # Create domain selections for physics assignment
    _create_selections(java, params)

    return geom

def _create_selections(java, params: MOSFETParams):
    """Create named selections for physics domains."""
    sel = java.selection()

    # These will be used to assign materials and physics
    sel.create("sel_substrate", "Explicit")
    sel.create("sel_source", "Explicit")
    sel.create("sel_drain", "Explicit")
    sel.create("sel_oxide", "Explicit")
    sel.create("sel_gate", "Explicit")
    sel.create("sel_channel", "Explicit")  # Surface selection

    # Note: Actual domain numbers depend on geometry build order
    # May need adjustment based on COMSOL's domain numbering
```

---

### Phase 3: Physics Configuration

**File: `comsol/physics.py`**

```python
"""Semiconductor physics setup for COMSOL MOSFET model."""
from ..device import MOSFETParams
from ..materials import SILICON, SIO2

def setup_semiconductor_physics(model, params: MOSFETParams):
    """
    Configure drift-diffusion semiconductor physics.

    Uses COMSOL's Semiconductor interface which solves:
    - Poisson's equation for electrostatics
    - Drift-diffusion equations for electron/hole transport
    """
    java = model.java

    # Create Semiconductor physics interface
    phys = java.physics().create("semi", "Semiconductor", "geom1")

    # Set formulation options
    phys.prop("EquationFormulation").set("formulation", "DriftDiffusion")
    phys.prop("SolverOptions").set("discretization", "FiniteVolume")

    # --- Material Properties ---

    # Silicon substrate properties
    phys.feature().create("mat_si", "SemiconductorMaterialModel")
    phys.feature("mat_si").selection().named("geom1_substrate_dom")
    phys.feature("mat_si").set("epsilonr", params.substrate.epsilon_r)
    phys.feature("mat_si").set("ni", f"{params.substrate.ni}[1/cm^3]")
    phys.feature("mat_si").set("mun", f"{params.substrate.mu_n}[cm^2/(V*s)]")
    phys.feature("mat_si").set("mup", f"{params.substrate.mu_p}[cm^2/(V*s)]")

    # --- Doping Profiles ---

    # P-type substrate doping (acceptors)
    phys.feature().create("dop_body", "AnalyticDopingModel")
    phys.feature("dop_body").selection().named("geom1_substrate_dom")
    phys.feature("dop_body").set("DopingType", "Acceptor")
    phys.feature("dop_body").set("Na0", f"{params.substrate_doping}[1/cm^3]")

    # N+ source doping (donors)
    phys.feature().create("dop_source", "AnalyticDopingModel")
    phys.feature("dop_source").selection().named("geom1_source_dom")
    phys.feature("dop_source").set("DopingType", "Donor")
    phys.feature("dop_source").set("Nd0", f"{params.source_drain_doping}[1/cm^3]")

    # N+ drain doping (donors)
    phys.feature().create("dop_drain", "AnalyticDopingModel")
    phys.feature("dop_drain").selection().named("geom1_drain_dom")
    phys.feature("dop_drain").set("DopingType", "Donor")
    phys.feature("dop_drain").set("Nd0", f"{params.source_drain_doping}[1/cm^3]")

    # --- Contacts ---

    # Source contact (boundary)
    phys.feature().create("src_contact", "MetalContact")
    phys.feature("src_contact").selection().set([1])  # Source boundary
    phys.feature("src_contact").set("V0", "0[V]")  # Ground reference

    # Drain contact
    phys.feature().create("drn_contact", "MetalContact")
    phys.feature("drn_contact").selection().set([2])  # Drain boundary
    phys.feature("drn_contact").set("V0", "Vds")  # Parameter

    # Body contact (substrate bottom)
    phys.feature().create("body_contact", "MetalContact")
    phys.feature("body_contact").selection().set([3])  # Body boundary
    phys.feature("body_contact").set("V0", "Vbs")  # Body bias parameter

    # --- Gate Stack ---

    # Oxide layer (insulator domain)
    phys.feature().create("oxide_domain", "InsulatorDomain")
    phys.feature("oxide_domain").selection().named("geom1_oxide_dom")
    phys.feature("oxide_domain").set("epsilonr", params.oxide.epsilon_r)

    # Gate contact
    phys.feature().create("gate_contact", "GateContact")
    phys.feature("gate_contact").selection().set([4])  # Gate boundary
    phys.feature("gate_contact").set("V0", "Vgs")  # Gate voltage parameter

    return phys

def setup_global_parameters(model, vgs=0.0, vds=0.0, vbs=0.0):
    """Set up global voltage parameters."""
    java = model.java

    java.param().set("Vgs", f"{vgs}[V]", "Gate-source voltage")
    java.param().set("Vds", f"{vds}[V]", "Drain-source voltage")
    java.param().set("Vbs", f"{vbs}[V]", "Body-source voltage")

def update_bias_point(model, vgs: float, vds: float, vbs: float = 0.0):
    """Update bias voltages for parametric sweep."""
    java = model.java
    java.param().set("Vgs", f"{vgs}[V]")
    java.param().set("Vds", f"{vds}[V]")
    java.param().set("Vbs", f"{vbs}[V]")
```

---

### Phase 4: Mesh Configuration

**File: `comsol/mesh.py`**

```python
"""Mesh generation for COMSOL MOSFET model."""
from ..device import MOSFETParams

def build_mesh(model, params: MOSFETParams, refinement: str = "normal"):
    """
    Create optimized mesh for semiconductor simulation.

    Args:
        model: COMSOL model instance
        params: Device parameters
        refinement: "coarse", "normal", "fine", or "extra_fine"
    """
    java = model.java
    mesh = java.mesh().create("mesh1", "geom1")

    # Mesh size settings based on refinement level
    size_presets = {
        "coarse": {"max": 0.5, "min": 0.02, "rate": 1.5},
        "normal": {"max": 0.2, "min": 0.01, "rate": 1.3},
        "fine": {"max": 0.1, "min": 0.005, "rate": 1.2},
        "extra_fine": {"max": 0.05, "min": 0.002, "rate": 1.1},
    }

    sizes = size_presets.get(refinement, size_presets["normal"])
    L_um = params.channel_length * 1e6  # Convert to μm

    # Global mesh size
    mesh.feature().create("size", "Size")
    mesh.feature("size").set("hauto", 5)  # Auto mesh level
    mesh.feature("size").set("hmax", f"{sizes['max'] * L_um}[um]")
    mesh.feature("size").set("hmin", f"{sizes['min'] * L_um}[um]")
    mesh.feature("size").set("hgrad", sizes['rate'])

    # Refine mesh at oxide-semiconductor interface (critical region)
    mesh.feature().create("edg_interface", "Edge")
    mesh.feature("edg_interface").selection().named("geom1_oxide_bnd")
    mesh.feature("edg_interface").feature().create("size_interface", "Size")
    mesh.feature("edg_interface").feature("size_interface").set(
        "hmax", f"{sizes['min'] * L_um * 2}[um]"
    )

    # Refine at junction regions
    mesh.feature().create("ftri_junc", "FreeTri")
    mesh.feature("ftri_junc").selection().named("geom1_junction_bnd")

    # Swept mesh for 3D efficiency (extrude 2D mesh)
    mesh.feature().create("swe1", "Sweep")
    mesh.feature("swe1").selection().geom("geom1", 3)
    mesh.feature("swe1").selection().all()

    # Build mesh
    mesh.run()

    return mesh

def get_mesh_statistics(model) -> dict:
    """Extract mesh quality statistics."""
    java = model.java
    mesh = java.mesh("mesh1")

    stats = {
        "num_elements": int(mesh.stat().getNumElem()),
        "min_quality": float(mesh.stat().getQualityMin()),
        "avg_quality": float(mesh.stat().getQualityAvg()),
        "num_vertices": int(mesh.stat().getNumVertex()),
    }

    return stats
```

---

### Phase 5: Study & Solver Configuration

**File: `comsol/study.py`**

```python
"""Study and solver configuration for COMSOL semiconductor simulation."""
import numpy as np
from typing import List, Tuple

def create_stationary_study(model):
    """Create stationary study for DC analysis."""
    java = model.java

    study = java.study().create("std1")
    study.feature().create("stat", "Stationary")

    # Solver settings for semiconductor convergence
    sol = java.sol().create("sol1")
    sol.study("std1")
    sol.attach("std1")

    # Stationary solver with continuation for convergence
    sol.feature().create("st1", "StudyStep")
    sol.feature("st1").set("study", "std1")
    sol.feature("st1").set("studystep", "stat")

    # Variable initialization
    sol.feature().create("v1", "Variables")
    sol.feature("v1").set("control", "stat")

    # Stationary solver
    sol.feature().create("s1", "Stationary")
    sol.feature("s1").feature().create("fc1", "FullyCoupled")
    sol.feature("s1").feature("fc1").set("termonres", True)
    sol.feature("s1").feature("fc1").set("damp", 0.9)  # Damping for stability
    sol.feature("s1").feature("fc1").set("maxiter", 50)

    return study

def create_parametric_sweep(
    model,
    param_name: str,
    param_values: np.ndarray,
    study_name: str = "std1"
):
    """
    Create parametric sweep for voltage variation.

    Args:
        model: COMSOL model
        param_name: Parameter to sweep ("Vgs" or "Vds")
        param_values: Array of parameter values
        study_name: Name of base study
    """
    java = model.java
    study = java.study(study_name)

    # Create parametric sweep
    study.feature().create("param", "Parametric")
    study.feature("param").set("pname", param_name)
    study.feature("param").set("plistarr", [f"{v}" for v in param_values])
    study.feature("param").set("punit", "V")

    # Enable continuation for faster convergence across sweep
    study.feature("param").set("usecont", True)

    return study

def run_study(model, study_name: str = "std1"):
    """Execute the study and wait for completion."""
    java = model.java
    java.sol("sol1").runAll()

def create_vgs_sweep_study(
    model,
    vgs_min: float,
    vgs_max: float,
    n_points: int,
    vds: float
):
    """Convenience function for gate voltage sweep."""
    # Set fixed Vds
    java = model.java
    java.param().set("Vds", f"{vds}[V]")

    # Create sweep
    vgs_values = np.linspace(vgs_min, vgs_max, n_points)
    create_parametric_sweep(model, "Vgs", vgs_values)

    return vgs_values

def create_vds_sweep_study(
    model,
    vgs: float,
    vds_min: float,
    vds_max: float,
    n_points: int
):
    """Convenience function for drain voltage sweep."""
    # Set fixed Vgs
    java = model.java
    java.param().set("Vgs", f"{vgs}[V]")

    # Create sweep
    vds_values = np.linspace(vds_min, vds_max, n_points)
    create_parametric_sweep(model, "Vds", vds_values)

    return vds_values
```

---

### Phase 6: Data Extraction

**File: `comsol/extract.py`**

```python
"""Extract simulation results from COMSOL for visualization."""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class COMSOLConcentration:
    """Carrier concentration data extracted from COMSOL."""
    electrons: np.ndarray  # 3D array [x, y, z]
    holes: np.ndarray      # 3D array [x, y, z]
    potential: np.ndarray  # 3D array [x, y, z] - electric potential
    x: np.ndarray          # x coordinates
    y: np.ndarray          # y coordinates
    z: np.ndarray          # z coordinates

def extract_concentration_field(
    model,
    nx: int = 60,
    ny: int = 20,
    nz: int = 30,
    solution_index: int = 1
) -> COMSOLConcentration:
    """
    Extract electron/hole concentrations on structured grid.

    Args:
        model: Solved COMSOL model
        nx, ny, nz: Grid resolution
        solution_index: Solution index for parametric studies

    Returns:
        COMSOLConcentration with carrier densities
    """
    java = model.java

    # Get geometry bounds
    geom = java.geom("geom1")
    bbox = geom.getBoundingBox()

    x_min, x_max = bbox[0], bbox[3]
    y_min, y_max = bbox[1], bbox[4]
    z_min, z_max = bbox[2], bbox[5]

    # Create evaluation grid
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    z = np.linspace(z_min, z_max, nz)

    # Create meshgrid for evaluation points
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    coords = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # Extract electron concentration
    electrons = _evaluate_expression(
        model, "semi.n", coords, solution_index
    ).reshape(nx, ny, nz)

    # Extract hole concentration
    holes = _evaluate_expression(
        model, "semi.p", coords, solution_index
    ).reshape(nx, ny, nz)

    # Extract electric potential
    potential = _evaluate_expression(
        model, "semi.V", coords, solution_index
    ).reshape(nx, ny, nz)

    return COMSOLConcentration(
        electrons=electrons,
        holes=holes,
        potential=potential,
        x=x,
        y=y,
        z=z
    )

def _evaluate_expression(
    model,
    expr: str,
    coords: np.ndarray,
    solution_index: int = 1
) -> np.ndarray:
    """Evaluate COMSOL expression at given coordinates."""
    java = model.java

    # Create interpolation evaluation
    num_points = coords.shape[0]

    # Use COMSOL's interp evaluation
    result = java.result()

    # Create temporary evaluation group
    eval_group = result.numerical().create("tmp_eval", "Interp")
    eval_group.set("expr", expr)
    eval_group.set("coord", coords.T.tolist())
    eval_group.set("solnum", solution_index)

    # Evaluate
    values = np.array(eval_group.getData())

    # Clean up
    result.numerical().remove("tmp_eval")

    return values

def extract_drain_current(model, solution_indices: List[int] = None) -> np.ndarray:
    """
    Extract drain current from simulation.

    Uses integration of current density at drain contact.
    """
    java = model.java
    result = java.result()

    # Create surface integration for current
    integ = result.numerical().create("id_eval", "IntSurface")
    integ.selection().named("geom1_drain_contact")
    integ.set("expr", "semi.Jn + semi.Jp")  # Total current density

    if solution_indices is None:
        # Get all parametric solutions
        sol = java.sol("sol1")
        n_solutions = int(sol.feature("s1").getSolnum())
        solution_indices = list(range(1, n_solutions + 1))

    currents = []
    for sol_idx in solution_indices:
        integ.set("solnum", sol_idx)
        Id = float(integ.getReal())
        currents.append(Id)

    result.numerical().remove("id_eval")

    return np.array(currents)

def extract_sweep_concentrations(
    model,
    n_frames: int,
    nx: int = 60,
    ny: int = 20,
    nz: int = 30
) -> List[COMSOLConcentration]:
    """
    Extract concentration fields for all parametric sweep points.

    Returns list of COMSOLConcentration objects for animation.
    """
    concentrations = []

    for i in range(1, n_frames + 1):
        conc = extract_concentration_field(
            model, nx=nx, ny=ny, nz=nz, solution_index=i
        )
        concentrations.append(conc)

    return concentrations
```

---

### Phase 7: High-Level Model Builder

**File: `comsol/model_builder.py`**

```python
"""High-level API for building and running MOSFET simulations."""
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

from ..device import MOSFETParams
from ..physics import threshold_voltage
from .client import create_model, save_model, load_model
from .geometry import build_mosfet_geometry
from .physics import setup_semiconductor_physics, setup_global_parameters
from .mesh import build_mesh
from .study import (
    create_stationary_study,
    create_vgs_sweep_study,
    create_vds_sweep_study,
    run_study
)
from .extract import (
    extract_concentration_field,
    extract_sweep_concentrations,
    extract_drain_current,
    COMSOLConcentration
)

class MOSFETModel:
    """
    High-level interface for COMSOL MOSFET simulation.

    Example usage:
        >>> params = MOSFETParams(channel_length=1e-6)
        >>> model = MOSFETModel(params)
        >>> model.build()
        >>> concentrations = model.sweep_vgs(vgs_min=0, vgs_max=3, n_frames=30)
    """

    def __init__(
        self,
        params: MOSFETParams,
        mesh_refinement: str = "normal",
        cache_path: Optional[Path] = None
    ):
        self.params = params
        self.mesh_refinement = mesh_refinement
        self.cache_path = cache_path
        self._model = None
        self._built = False

    def build(self):
        """Build the complete COMSOL model."""
        self._model = create_model("MOSFET")

        # Build geometry
        build_mosfet_geometry(self._model, self.params)

        # Setup physics
        setup_semiconductor_physics(self._model, self.params)
        setup_global_parameters(self._model)

        # Build mesh
        build_mesh(self._model, self.params, self.mesh_refinement)

        # Create study
        create_stationary_study(self._model)

        self._built = True

        # Cache model if path specified
        if self.cache_path:
            save_model(self._model, self.cache_path)

    def load_cached(self) -> bool:
        """Load cached model if available."""
        if self.cache_path and self.cache_path.exists():
            self._model = load_model(self.cache_path)
            self._built = True
            return True
        return False

    def solve_single(
        self,
        vgs: float,
        vds: float,
        vbs: float = 0.0
    ) -> COMSOLConcentration:
        """Solve for a single bias point."""
        if not self._built:
            self.build()

        # Update parameters
        java = self._model.java
        java.param().set("Vgs", f"{vgs}[V]")
        java.param().set("Vds", f"{vds}[V]")
        java.param().set("Vbs", f"{vbs}[V]")

        # Run solver
        run_study(self._model)

        # Extract results
        return extract_concentration_field(self._model)

    def sweep_vgs(
        self,
        vgs_min: float,
        vgs_max: float,
        vds: float,
        n_frames: int = 30,
        mesh_resolution: Tuple[int, int, int] = (60, 20, 30)
    ) -> Tuple[List[COMSOLConcentration], np.ndarray]:
        """
        Perform gate voltage sweep.

        Returns:
            Tuple of (concentration_list, vgs_values)
        """
        if not self._built:
            self.build()

        # Setup parametric sweep
        vgs_values = create_vgs_sweep_study(
            self._model,
            vgs_min=vgs_min,
            vgs_max=vgs_max,
            n_points=n_frames,
            vds=vds
        )

        # Run study
        run_study(self._model)

        # Extract results
        nx, ny, nz = mesh_resolution
        concentrations = extract_sweep_concentrations(
            self._model, n_frames, nx=nx, ny=ny, nz=nz
        )

        return concentrations, vgs_values

    def sweep_vds(
        self,
        vgs: float,
        vds_min: float,
        vds_max: float,
        n_frames: int = 30,
        mesh_resolution: Tuple[int, int, int] = (60, 20, 30)
    ) -> Tuple[List[COMSOLConcentration], np.ndarray]:
        """
        Perform drain voltage sweep.

        Returns:
            Tuple of (concentration_list, vds_values)
        """
        if not self._built:
            self.build()

        # Setup parametric sweep
        vds_values = create_vds_sweep_study(
            self._model,
            vgs=vgs,
            vds_min=vds_min,
            vds_max=vds_max,
            n_points=n_frames
        )

        # Run study
        run_study(self._model)

        # Extract results
        nx, ny, nz = mesh_resolution
        concentrations = extract_sweep_concentrations(
            self._model, n_frames, nx=nx, ny=ny, nz=nz
        )

        return concentrations, vds_values

    def get_iv_curves(
        self,
        vgs_values: np.ndarray,
        vds_values: np.ndarray
    ) -> np.ndarray:
        """
        Compute I-V characteristics.

        Returns:
            2D array of drain currents [len(vgs_values), len(vds_values)]
        """
        if not self._built:
            self.build()

        currents = np.zeros((len(vgs_values), len(vds_values)))

        for i, vgs in enumerate(vgs_values):
            # Setup Vds sweep at this Vgs
            create_vds_sweep_study(
                self._model,
                vgs=vgs,
                vds_min=vds_values[0],
                vds_max=vds_values[-1],
                n_points=len(vds_values)
            )

            run_study(self._model)
            currents[i, :] = extract_drain_current(self._model)

        return currents

    @property
    def threshold_voltage(self) -> float:
        """Get threshold voltage from analytical model."""
        return threshold_voltage(self.params)
```

---

### Phase 8: Visualization Integration

**File: Update existing `visualization_3d.py`**

Add adapter functions to convert COMSOL data to existing visualization:

```python
# Add to visualization_3d.py

from .comsol.extract import COMSOLConcentration
from .concentration import CarrierConcentration

def comsol_to_carrier_concentration(
    comsol_conc: COMSOLConcentration,
    params: MOSFETParams
) -> CarrierConcentration:
    """
    Convert COMSOL concentration data to existing CarrierConcentration format.

    This allows reuse of existing visualization code.
    """
    from .mesh import DeviceMesh, Region

    # Create compatible mesh structure
    X, Y, Z = np.meshgrid(
        comsol_conc.x,
        comsol_conc.y,
        comsol_conc.z,
        indexing='ij'
    )

    # Create placeholder regions array
    regions = np.zeros_like(comsol_conc.electrons, dtype=np.int8)

    mesh = DeviceMesh(
        x=comsol_conc.x,
        y=comsol_conc.y,
        z=comsol_conc.z,
        X=X,
        Y=Y,
        Z=Z,
        regions=regions,
        params=params
    )

    # Fixed charge from doping (simplified)
    fixed_charge = np.zeros_like(comsol_conc.electrons)

    return CarrierConcentration(
        mesh=mesh,
        electrons=comsol_conc.electrons,
        holes=comsol_conc.holes,
        fixed_charge=fixed_charge
    )

def create_comsol_animation_2d(
    comsol_concentrations: List[COMSOLConcentration],
    parameter_values: np.ndarray,
    parameter_name: str,
    params: MOSFETParams
) -> go.Figure:
    """
    Create animation from COMSOL results using existing visualization.
    """
    # Convert COMSOL data to existing format
    carrier_concentrations = [
        comsol_to_carrier_concentration(c, params)
        for c in comsol_concentrations
    ]

    # Use existing animation function
    # Need to create a compatible mesh object
    mesh = carrier_concentrations[0].mesh

    return create_animated_cross_section(
        mesh=mesh,
        concentrations=carrier_concentrations,
        parameter_values=parameter_values,
        parameter_name=parameter_name
    )
```

---

### Phase 9: Notebook Update

**File: Update `notebooks/mosfet_explorer.py`**

```python
import marimo as mo
import numpy as np

# Import COMSOL backend
from mosfet_sim.comsol import MOSFETModel
from mosfet_sim import MOSFETParams, threshold_voltage
from mosfet_sim.visualization_3d import create_comsol_animation_2d

app = mo.App()

@app.cell
def _():
    import marimo as mo
    return mo,

@app.cell
def _(mo):
    # Device parameter controls
    channel_length = mo.ui.slider(
        0.5, 5.0, value=1.0, step=0.1,
        label="Channel Length (μm)"
    )
    oxide_thickness = mo.ui.slider(
        5, 50, value=10, step=1,
        label="Oxide Thickness (nm)"
    )
    substrate_doping = mo.ui.slider(
        15, 18, value=17, step=0.5,
        label="log₁₀(Na) [cm⁻³]"
    )

    mo.vstack([channel_length, oxide_thickness, substrate_doping])

@app.cell
def _(channel_length, oxide_thickness, substrate_doping):
    # Create device parameters
    device = MOSFETParams(
        channel_length=channel_length.value * 1e-6,
        oxide_thickness=oxide_thickness.value * 1e-9,
        substrate_doping=10 ** substrate_doping.value,
    )

    vth = threshold_voltage(device)

    # Display device info
    mo.md(f"""
    ## Device Parameters
    - Threshold Voltage: **{vth:.3f} V**
    - Channel Length: {channel_length.value} μm
    - Oxide Thickness: {oxide_thickness.value} nm
    """)

@app.cell
def _(device, mo):
    # COMSOL simulation controls
    mesh_quality = mo.ui.dropdown(
        options=["coarse", "normal", "fine"],
        value="normal",
        label="Mesh Quality"
    )
    n_frames = mo.ui.slider(10, 60, value=30, label="Animation Frames")

    run_button = mo.ui.run_button(label="Run COMSOL Simulation")

    mo.hstack([mesh_quality, n_frames, run_button])

@app.cell
def _(device, mesh_quality, n_frames, run_button, vth):
    # Run simulation when button clicked
    if not run_button.value:
        mo.stop()

    mo.md("**Running COMSOL simulation...**")

    # Create and build model
    model = MOSFETModel(
        params=device,
        mesh_refinement=mesh_quality.value
    )

    # Run Vgs sweep
    concentrations, vgs_values = model.sweep_vgs(
        vgs_min=0.0,
        vgs_max=vth + 2.0,
        vds=0.5,
        n_frames=n_frames.value,
        mesh_resolution=(50, 20, 30)
    )

    # Create animation
    fig = create_comsol_animation_2d(
        comsol_concentrations=concentrations,
        parameter_values=vgs_values,
        parameter_name="Vgs",
        params=device
    )

    fig

if __name__ == "__main__":
    app.run()
```

---

## API Reference

### MPh Core API

| Method | Description |
|--------|-------------|
| `mph.start()` | Start COMSOL server, return Client |
| `client.create(name)` | Create new model |
| `client.load(path)` | Load .mph file |
| `model.save(path)` | Save model to .mph |
| `model.solve(study)` | Run named study |
| `model.java` | Access raw Java API |

### COMSOL Java API (via `model.java`)

| Component | Creation | Description |
|-----------|----------|-------------|
| Geometry | `model.geom().create("geom1", 3)` | 3D geometry |
| Block | `geom.feature().create("blk1", "Block")` | Box primitive |
| Physics | `model.physics().create("semi", "Semiconductor", "geom1")` | Semiconductor |
| Mesh | `model.mesh().create("mesh1", "geom1")` | Mesh sequence |
| Study | `model.study().create("std1")` | Study container |
| Solver | `model.sol().create("sol1")` | Solution sequence |

### Semiconductor Physics Features

| Feature | API Name | Description |
|---------|----------|-------------|
| Material | `SemiconductorMaterialModel` | Si properties |
| Doping | `AnalyticDopingModel` | Uniform/Gaussian doping |
| Contact | `MetalContact` | Ohmic contact |
| Gate | `GateContact` | MOS gate |
| Insulator | `InsulatorDomain` | Oxide region |

---

## File Structure

```
mosfet_nb/
├── src/mosfet_sim/
│   ├── __init__.py                    # Add COMSOL exports
│   ├── constants.py                   # Keep as-is
│   ├── materials.py                   # Keep as-is
│   ├── device.py                      # Keep as-is
│   ├── physics.py                     # Keep for analytical Vth
│   ├── concentration.py               # Keep for data structures
│   ├── mesh.py                        # Keep for DeviceMesh class
│   ├── plotting.py                    # Keep as-is
│   ├── visualization_3d.py            # Add COMSOL adapters
│   │
│   └── comsol/                        # NEW
│       ├── __init__.py
│       ├── client.py                  # ~50 lines
│       ├── geometry.py                # ~100 lines
│       ├── physics.py                 # ~100 lines
│       ├── mesh.py                    # ~60 lines
│       ├── study.py                   # ~80 lines
│       ├── extract.py                 # ~120 lines
│       └── model_builder.py           # ~150 lines
│
├── notebooks/
│   └── mosfet_explorer.py             # Update to use COMSOL
│
├── models/                            # Cached COMSOL models
│   └── .gitkeep
│
└── requirements.txt                   # Add MPh==1.3.1
```

---

## Migration Guide

### Step-by-Step Implementation Order

1. **Install dependencies**
   ```bash
   pip install MPh==1.3.1
   ```

2. **Implement core modules** (in order)
   - `comsol/client.py` - Test COMSOL connection
   - `comsol/geometry.py` - Verify geometry builds correctly
   - `comsol/physics.py` - Test semiconductor interface creation
   - `comsol/mesh.py` - Verify mesh generation
   - `comsol/study.py` - Test solver runs
   - `comsol/extract.py` - Verify data extraction

3. **Implement model builder**
   - `comsol/model_builder.py` - Integration layer

4. **Add visualization adapters**
   - Update `visualization_3d.py` with conversion functions

5. **Update notebook**
   - Modify `mosfet_explorer.py` to use COMSOL backend

### Testing Checklist

- [ ] COMSOL client starts successfully
- [ ] Geometry builds without errors
- [ ] Physics interface configured correctly
- [ ] Mesh generates with acceptable quality
- [ ] Single bias point solves and converges
- [ ] Parametric sweep completes
- [ ] Carrier concentrations extract correctly
- [ ] Visualization shows expected physics
- [ ] Animation plays smoothly

### Fallback Strategy

Keep the analytical backend as a fallback:

```python
# In notebook
use_comsol = mo.ui.checkbox(label="Use COMSOL (requires license)")

if use_comsol.value:
    from mosfet_sim.comsol import MOSFETModel
    # COMSOL path
else:
    from mosfet_sim import create_vgs_sweep_animation_2d
    # Analytical path
```

---

## References

- [COMSOL Semiconductor Module](https://www.comsol.com/semiconductor-module)
- [MPh Documentation](https://mph.readthedocs.io/)
- [MPh GitHub Repository](https://github.com/MPh-py/MPh)
- [Drift-Diffusion Tutorial](https://www.comsol.com/model/drift-diffusion-tutorial-8643)
- [Introduction to Semiconductor Modeling](https://www.comsol.com/support/learning-center/article/introduction-to-semiconductor-modeling-84981/222)
- [Advanced Semiconductor Modeling](https://www.comsol.com/support/learning-center/article/advanced-semiconductor-modeling-85261/222)
