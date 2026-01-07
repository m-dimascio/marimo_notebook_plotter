"""MOSFET Simulation Package with 3D Visualization."""

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
)
from .plotting import (
    plot_output_characteristics,
    plot_transfer_characteristics,
    plot_device_cross_section,
    plot_band_diagram,
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
    # 2D Plotting
    "plot_output_characteristics", "plot_transfer_characteristics",
    "plot_device_cross_section", "plot_band_diagram",
]
