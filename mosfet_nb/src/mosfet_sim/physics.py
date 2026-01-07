"""MOSFET physics calculations.

Implements the gradual channel approximation (square-law model) for
long-channel MOSFETs. Provides calculations for:
- Oxide capacitance
- Threshold voltage
- Drain current (linear and saturation regions)
- Channel charge distribution
- Band diagram quantities
"""

import numpy as np
from .constants import Q, K_B, EPS_0, thermal_voltage
from .device import MOSFETParams


def oxide_capacitance(params: MOSFETParams) -> float:
    """
    Calculate oxide capacitance per unit area.

    Cox = eps_ox / tox

    Returns:
        Cox in F/m^2
    """
    return params.oxide.permittivity / params.oxide_thickness


def bulk_potential(params: MOSFETParams) -> float:
    """
    Calculate bulk Fermi potential phi_f.

    phi_f = (kT/q) * ln(Na / ni)

    Returns:
        phi_f in V
    """
    vt = thermal_voltage(params.temperature)
    return vt * np.log(params.substrate_doping / params.substrate.ni)


def threshold_voltage(params: MOSFETParams, vbs: float = 0.0) -> float:
    """
    Calculate threshold voltage including body effect.

    Vth = Vth0 + gamma * (sqrt(2*phi_f - Vbs) - sqrt(2*phi_f))

    Where:
        Vth0 = 2*phi_f + Qb/Cox  (zero-bias threshold)
        gamma = sqrt(2*q*eps_s*Na) / Cox  (body effect coefficient)

    Args:
        params: MOSFET device parameters
        vbs: Body-source voltage (V), typically <= 0 for NMOS

    Returns:
        Vth in V
    """
    phi_f = bulk_potential(params)
    cox = oxide_capacitance(params)
    eps_s = params.substrate.permittivity
    na_si = params.substrate_doping * 1e6  # Convert cm^-3 to m^-3

    # Body effect coefficient
    gamma = np.sqrt(2 * Q * eps_s * na_si) / cox

    # Zero-bias threshold (simplified, assuming no work function difference)
    vth0 = 2 * phi_f + gamma * np.sqrt(2 * phi_f)

    # Include body effect
    if vbs >= 2 * phi_f:
        vbs = 2 * phi_f - 0.01  # Prevent sqrt of negative

    vth = vth0 + gamma * (np.sqrt(2 * phi_f - vbs) - np.sqrt(2 * phi_f))
    return vth


def drain_current(
    params: MOSFETParams,
    vgs: float | np.ndarray,
    vds: float | np.ndarray,
    vbs: float = 0.0,
) -> float | np.ndarray:
    """
    Calculate drain current using square-law model.

    Linear region (Vds < Vgs - Vth):
        Id = (W/L) * mu_n * Cox * [(Vgs - Vth)*Vds - Vds^2/2]

    Saturation region (Vds >= Vgs - Vth):
        Id = (W/L) * mu_n * Cox * (Vgs - Vth)^2 / 2

    Args:
        params: MOSFET device parameters
        vgs: Gate-source voltage (V)
        vds: Drain-source voltage (V)
        vbs: Body-source voltage (V)

    Returns:
        Drain current in A
    """
    vgs = np.atleast_1d(np.asarray(vgs, dtype=float))
    vds = np.atleast_1d(np.asarray(vds, dtype=float))

    # Broadcast arrays to same shape
    vgs, vds = np.broadcast_arrays(vgs, vds)

    vth = threshold_voltage(params, vbs)
    cox = oxide_capacitance(params)
    mu_n = params.substrate.mu_n * 1e-4  # Convert cm^2/V·s to m^2/V·s

    # Geometry factor
    k = (params.channel_width / params.channel_length) * mu_n * cox

    # Overdrive voltage
    vov = vgs - vth

    # Initialize current array
    id_out = np.zeros_like(vgs, dtype=float)

    # Cutoff region
    cutoff = vov <= 0
    id_out[cutoff] = 0.0

    # Saturation voltage
    vdsat = np.maximum(vov, 0)

    # Linear region: Vds < Vdsat
    linear = (~cutoff) & (vds < vdsat)
    id_out[linear] = k * (vov[linear] * vds[linear] - vds[linear]**2 / 2)

    # Saturation region: Vds >= Vdsat
    sat = (~cutoff) & (vds >= vdsat)
    id_out[sat] = k * vov[sat]**2 / 2

    return id_out.squeeze()


def channel_charge_density(
    params: MOSFETParams,
    vgs: float,
    vds: float,
    x_norm: np.ndarray,
) -> np.ndarray:
    """
    Calculate inversion charge density along channel.

    Qn(x) = -Cox * [Vgs - Vth - V(x)]

    Where V(x) is the channel potential varying from 0 at source
    to Vds at drain (linear approximation in linear region).

    Args:
        params: MOSFET device parameters
        vgs: Gate-source voltage (V)
        vds: Drain-source voltage (V)
        x_norm: Normalized position along channel (0 to 1)

    Returns:
        Inversion charge density in C/m^2 (negative for electrons)
    """
    vth = threshold_voltage(params)
    cox = oxide_capacitance(params)

    vov = vgs - vth
    if vov <= 0:
        return np.zeros_like(x_norm)

    # Channel potential (gradual channel approximation)
    vdsat = vov
    vds_eff = min(vds, vdsat)  # Clamp to saturation

    # Linear variation of potential along channel
    v_channel = vds_eff * x_norm

    # Inversion charge (magnitude)
    qn = cox * (vov - v_channel)
    qn = np.maximum(qn, 0)  # No negative charge

    return qn


def band_diagram_at_gate(
    params: MOSFETParams,
    vgs: float,
    depth: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Calculate simplified band diagram perpendicular to gate.

    Provides conduction band (Ec), valence band (Ev), intrinsic
    level (Ei), and Fermi level (Ef) as function of depth from
    oxide-semiconductor interface.

    Args:
        params: MOSFET device parameters
        depth: Distance from oxide interface into substrate (m)
        vgs: Gate-source voltage (V)

    Returns:
        Dictionary with 'ec', 'ev', 'ei', 'ef' arrays in eV
    """
    phi_f = bulk_potential(params)
    vth = threshold_voltage(params)
    eg = params.substrate.eg

    # Surface potential (simplified model)
    if vgs < vth:
        # Depletion/accumulation
        phi_s = vgs * 0.5  # Approximate
    else:
        # Inversion - surface potential pins near 2*phi_f
        phi_s = 2 * phi_f

    # Depletion width estimate
    eps_s = params.substrate.permittivity
    na_si = params.substrate_doping * 1e6
    w_dep = np.sqrt(2 * eps_s * phi_s / (Q * na_si)) if phi_s > 0 else 1e-9

    # Band bending (exponential decay approximation)
    bending = phi_s * np.exp(-depth / (w_dep / 3))

    # Reference: Ef = 0 in bulk
    ef = np.zeros_like(depth)

    # Intrinsic level in bulk: Ei = phi_f (above Ef for p-type)
    ei_bulk = phi_f
    ei = ei_bulk - bending  # Bends down toward surface

    # Conduction and valence bands
    ec = ei + eg / 2
    ev = ei - eg / 2

    return {"ec": ec, "ev": ev, "ei": ei, "ef": ef, "depth": depth}
