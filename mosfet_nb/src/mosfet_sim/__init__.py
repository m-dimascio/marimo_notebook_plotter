"""MOSFET Simulation Package with 3D Visualization.

This package provides:
- Analytical MOSFET physics models (square-law approximation)
- 3D mesh generation and carrier concentration computation
- Animated Plotly visualizations
- Optional COMSOL integration for rigorous FEM simulation

COMSOL Integration:
    The optional `comsol` subpackage provides a Python interface to
    COMSOL Multiphysics for drift-diffusion semiconductor simulation.
    Requires: COMSOL with Semiconductor Module + MPh library.

    >>> from mosfet_sim.comsol import MOSFETModel, is_comsol_available
    >>> if is_comsol_available():
    ...     model = MOSFETModel(params)
    ...     model.build()
    ...     conc = model.solve_single(vgs=1.5, vds=0.5)
"""

from .constants import Q, K_B, EPS_0, thermal_voltage
from .materials import Semiconductor, Insulator, SILICON, SIO2
from .device import MOSFETParams
from .physics import (
    oxide_capacitance,
    bulk_potential,
    threshold_voltage,
    drain_current,
    channel_charge_density,
    band_diagram_at_gate,
)
from .mesh import DeviceMesh, Region, create_device_mesh
from .concentration import (
    CarrierConcentration,
    compute_equilibrium_concentrations,
    compute_biased_concentrations,
    generate_concentration_sweep,
    generate_output_sweep,
)
from .visualization_3d import (
    create_animated_figure,
    create_vgs_sweep_animation,
    create_vds_sweep_animation,
    create_cross_section_slice,
    create_animated_cross_section,
    create_vgs_sweep_animation_2d,
    create_vds_sweep_animation_2d,
    # Fluid-dynamics style visualization
    smooth_concentration_field,
    compute_current_flow_field,
    create_fluid_animation_2d,
    create_vgs_sweep_fluid_animation,
    create_vds_sweep_fluid_animation,
    create_dual_view_animation,
    # COMSOL integration adapters
    comsol_to_carrier_concentration,
    create_comsol_animation_2d,
    create_comsol_vgs_animation,
    create_comsol_vds_animation,
)
from .plotting import (
    plot_output_characteristics,
    plot_transfer_characteristics,
    plot_device_cross_section,
    plot_band_diagram,
)
from .exporter import (
    export_mesh_geometry,
    export_concentration_frame,
    export_animation_sequence,
    export_complete_visualization,
)

__all__ = [
    # Constants
    "Q", "K_B", "EPS_0", "thermal_voltage",
    # Materials
    "Semiconductor", "Insulator", "SILICON", "SIO2",
    # Device
    "MOSFETParams",
    # Physics
    "oxide_capacitance", "bulk_potential", "threshold_voltage",
    "drain_current", "channel_charge_density", "band_diagram_at_gate",
    # Mesh
    "DeviceMesh", "Region", "create_device_mesh",
    # Concentration
    "CarrierConcentration", "compute_equilibrium_concentrations",
    "compute_biased_concentrations", "generate_concentration_sweep",
    "generate_output_sweep",
    # 3D Visualization
    "create_animated_figure", "create_vgs_sweep_animation",
    "create_vds_sweep_animation", "create_cross_section_slice",
    # 2D Animated Visualization
    "create_animated_cross_section", "create_vgs_sweep_animation_2d",
    "create_vds_sweep_animation_2d",
    # Fluid-Dynamics Style Visualization
    "smooth_concentration_field", "compute_current_flow_field",
    "create_fluid_animation_2d", "create_vgs_sweep_fluid_animation",
    "create_vds_sweep_fluid_animation", "create_dual_view_animation",
    # COMSOL Integration Adapters
    "comsol_to_carrier_concentration", "create_comsol_animation_2d",
    "create_comsol_vgs_animation", "create_comsol_vds_animation",
    # 2D Plotting
    "plot_output_characteristics", "plot_transfer_characteristics",
    "plot_device_cross_section", "plot_band_diagram",
    # Exporter
    "export_mesh_geometry", "export_concentration_frame",
    "export_animation_sequence", "export_complete_visualization",
]
