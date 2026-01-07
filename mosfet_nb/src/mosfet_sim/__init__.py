"""MOSFET Simulation Package."""

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
    # Plotting
    "plot_output_characteristics", "plot_transfer_characteristics",
    "plot_device_cross_section", "plot_band_diagram",
]
