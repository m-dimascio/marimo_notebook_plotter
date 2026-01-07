"""Carrier concentration calculations for 3D MOSFET visualization."""

import numpy as np
from dataclasses import dataclass
from .constants import Q, thermal_voltage
from .device import MOSFETParams
from .mesh import DeviceMesh, Region
from .physics import threshold_voltage, bulk_potential


@dataclass
class CarrierConcentration:
    """Carrier concentration data across device mesh.

    Attributes:
        electrons: 3D array of electron concentration (cm^-3)
        holes: 3D array of hole concentration (cm^-3)
        net_charge: 3D array of net charge density
        mesh: Reference to device mesh
    """
    electrons: np.ndarray
    holes: np.ndarray
    net_charge: np.ndarray
    mesh: DeviceMesh


def compute_equilibrium_concentrations(mesh: DeviceMesh) -> CarrierConcentration:
    """
    Compute carrier concentrations at thermal equilibrium (Vgs=Vds=0).

    Uses depletion approximation at junctions.

    Args:
        mesh: Device mesh

    Returns:
        CarrierConcentration at equilibrium
    """
    params = mesh.params
    ni = params.substrate.ni  # Intrinsic concentration
    Na = params.substrate_doping  # Body doping (p-type)
    Nd = params.source_drain_doping  # S/D doping (n-type)

    # Initialize arrays
    electrons = np.zeros_like(mesh.X)
    holes = np.zeros_like(mesh.X)

    # P-type body: p ~ Na, n ~ ni^2/Na
    body_mask = mesh.regions == Region.BODY
    holes[body_mask] = Na
    electrons[body_mask] = ni**2 / Na

    # N+ source/drain: n ~ Nd, p ~ ni^2/Nd
    sd_mask = (mesh.regions == Region.SOURCE) | (mesh.regions == Region.DRAIN)
    electrons[sd_mask] = Nd
    holes[sd_mask] = ni**2 / Nd

    # Channel region at equilibrium (same as body, no inversion)
    channel_mask = mesh.regions == Region.CHANNEL
    holes[channel_mask] = Na
    electrons[channel_mask] = ni**2 / Na

    # Oxide and gate: no carriers
    insulator_mask = (mesh.regions == Region.OXIDE) | (mesh.regions == Region.GATE)
    electrons[insulator_mask] = 0
    holes[insulator_mask] = 0

    net_charge = Q * (holes - electrons)

    return CarrierConcentration(
        electrons=electrons,
        holes=holes,
        net_charge=net_charge,
        mesh=mesh,
    )


def compute_biased_concentrations(
    mesh: DeviceMesh,
    vgs: float,
    vds: float,
    vbs: float = 0.0,
) -> CarrierConcentration:
    """
    Compute carrier concentrations under bias.

    Models:
    - Channel inversion when Vgs > Vth
    - Channel charge modulation along length (for Vds > 0)
    - Depletion region widths

    Args:
        mesh: Device mesh
        vgs: Gate-source voltage (V)
        vds: Drain-source voltage (V)
        vbs: Body-source voltage (V)

    Returns:
        CarrierConcentration under bias
    """
    params = mesh.params
    ni = params.substrate.ni
    Na = params.substrate_doping
    Nd = params.source_drain_doping
    vth = threshold_voltage(params, vbs)
    phi_f = bulk_potential(params)
    vt = thermal_voltage(params.temperature)

    # Start with equilibrium
    conc = compute_equilibrium_concentrations(mesh)
    electrons = conc.electrons.copy()
    holes = conc.holes.copy()

    # Channel dimensions
    L = params.channel_length * 1e6
    sd_length = L * 0.4

    # Channel region
    channel_mask = mesh.regions == Region.CHANNEL

    if vgs <= vth:
        # Subthreshold / cutoff: minimal channel electrons
        subthreshold_factor = np.exp((vgs - vth) / (2 * vt))
        electrons[channel_mask] = ni * subthreshold_factor
    else:
        # Above threshold: inversion layer forms
        vov = vgs - vth  # Overdrive voltage

        # Saturation voltage
        vdsat = vov
        vds_eff = min(vds, vdsat)

        # Channel electron concentration varies along length
        channel_indices = np.where(channel_mask)

        for idx in range(len(channel_indices[0])):
            i, j, k = channel_indices[0][idx], channel_indices[1][idx], channel_indices[2][idx]

            x_pos = mesh.X[i, j, k]
            z_pos = mesh.Z[i, j, k]

            # Normalized position along channel (0 at source, 1 at drain)
            x_channel = (x_pos - sd_length) / L
            x_channel = np.clip(x_channel, 0, 1)

            # Local channel potential (gradual channel approximation)
            v_local = vds_eff * x_channel

            # Local overdrive
            vov_local = vov - v_local

            if vov_local > 0:
                # Inversion charge density (simplified model)
                surface_concentration = Na * (vov_local / (2 * phi_f)) * 100

                # Exponential decay from surface (z=0)
                decay_length = 0.02  # um, inversion layer thickness
                depth_factor = np.exp(z_pos / decay_length)  # z is negative

                electrons[i, j, k] = max(
                    surface_concentration * depth_factor,
                    ni**2 / Na  # Minimum is equilibrium value
                )

                # Holes depleted in inversion layer
                holes[i, j, k] = ni**2 / max(electrons[i, j, k], ni)
            else:
                # Pinch-off region: depleted
                electrons[i, j, k] = ni
                holes[i, j, k] = ni

    net_charge = Q * (holes - electrons + _fixed_charge(mesh))

    return CarrierConcentration(
        electrons=electrons,
        holes=holes,
        net_charge=net_charge,
        mesh=mesh,
    )


def _fixed_charge(mesh: DeviceMesh) -> np.ndarray:
    """Calculate fixed charge from ionized dopants."""
    params = mesh.params
    Na = params.substrate_doping
    Nd = params.source_drain_doping

    fixed = np.zeros_like(mesh.X)

    # Ionized acceptors in body (negative)
    body_mask = (mesh.regions == Region.BODY) | (mesh.regions == Region.CHANNEL)
    fixed[body_mask] = -Na

    # Ionized donors in source/drain (positive)
    sd_mask = (mesh.regions == Region.SOURCE) | (mesh.regions == Region.DRAIN)
    fixed[sd_mask] = Nd

    return fixed


def generate_concentration_sweep(
    mesh: DeviceMesh,
    vgs_values: np.ndarray,
    vds: float = 0.1,
) -> list[CarrierConcentration]:
    """
    Pre-compute carrier concentrations for a sweep of Vgs values.

    Used for animation frame generation.

    Args:
        mesh: Device mesh
        vgs_values: Array of gate voltages to compute
        vds: Fixed drain voltage

    Returns:
        List of CarrierConcentration objects, one per Vgs value
    """
    return [
        compute_biased_concentrations(mesh, vgs=vgs, vds=vds)
        for vgs in vgs_values
    ]


def generate_output_sweep(
    mesh: DeviceMesh,
    vgs: float,
    vds_values: np.ndarray,
) -> list[CarrierConcentration]:
    """
    Pre-compute carrier concentrations for a sweep of Vds values.

    Used for animation showing transition from linear to saturation.

    Args:
        mesh: Device mesh
        vgs: Fixed gate voltage
        vds_values: Array of drain voltages to compute

    Returns:
        List of CarrierConcentration objects, one per Vds value
    """
    return [
        compute_biased_concentrations(mesh, vgs=vgs, vds=vds)
        for vds in vds_values
    ]
