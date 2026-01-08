"""COMSOL integration layer for MOSFET simulation.

This package provides a Python interface to COMSOL Multiphysics
for rigorous finite-element semiconductor device simulation.

Requires:
    - COMSOL Multiphysics with Semiconductor Module license
    - MPh Python library (pip install MPh==1.3.1)
"""

from .client import (
    get_client,
    shutdown_client,
    comsol_session,
    create_model,
    load_model,
    save_model,
    is_comsol_available,
)

from .geometry import build_mosfet_geometry

from .physics import (
    setup_semiconductor_physics,
    setup_global_parameters,
    update_bias_point,
)

from .mesh import build_mesh, get_mesh_statistics

from .study import (
    create_stationary_study,
    create_parametric_sweep,
    run_study,
    create_vgs_sweep_study,
    create_vds_sweep_study,
)

from .extract import (
    COMSOLConcentration,
    extract_concentration_field,
    extract_drain_current,
    extract_sweep_concentrations,
)

from .model_builder import MOSFETModel

__all__ = [
    # Client
    "get_client",
    "shutdown_client",
    "comsol_session",
    "create_model",
    "load_model",
    "save_model",
    "is_comsol_available",
    # Geometry
    "build_mosfet_geometry",
    # Physics
    "setup_semiconductor_physics",
    "setup_global_parameters",
    "update_bias_point",
    # Mesh
    "build_mesh",
    "get_mesh_statistics",
    # Study
    "create_stationary_study",
    "create_parametric_sweep",
    "run_study",
    "create_vgs_sweep_study",
    "create_vds_sweep_study",
    # Extract
    "COMSOLConcentration",
    "extract_concentration_field",
    "extract_drain_current",
    "extract_sweep_concentrations",
    # Model Builder
    "MOSFETModel",
]
