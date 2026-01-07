"""Plotting functions for MOSFET characteristics."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from .device import MOSFETParams
from .physics import drain_current, threshold_voltage


def plot_output_characteristics(
    params: MOSFETParams,
    vgs_values: list[float],
    vds_max: float = 5.0,
    num_points: int = 100,
) -> Figure:
    """
    Plot Id vs Vds for multiple Vgs values.

    Args:
        params: MOSFET device parameters
        vgs_values: List of gate voltages to plot
        vds_max: Maximum drain voltage
        num_points: Number of points per curve

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    vds = np.linspace(0, vds_max, num_points)

    for vgs in vgs_values:
        id_ma = drain_current(params, vgs, vds) * 1e3  # Convert to mA
        ax.plot(vds, id_ma, label=f"Vgs = {vgs:.1f} V")

    ax.set_xlabel("Vds (V)")
    ax.set_ylabel("Id (mA)")
    ax.set_title("MOSFET Output Characteristics")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, vds_max)
    ax.set_ylim(bottom=0)

    return fig


def plot_transfer_characteristics(
    params: MOSFETParams,
    vds_values: list[float],
    vgs_max: float = 5.0,
    num_points: int = 100,
    log_scale: bool = False,
) -> Figure:
    """
    Plot Id vs Vgs for multiple Vds values.

    Args:
        params: MOSFET device parameters
        vds_values: List of drain voltages to plot
        vgs_max: Maximum gate voltage
        num_points: Number of points per curve
        log_scale: Use logarithmic y-axis

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    vgs = np.linspace(0, vgs_max, num_points)
    vth = threshold_voltage(params)

    for vds in vds_values:
        id_ma = drain_current(params, vgs, vds) * 1e3
        ax.plot(vgs, id_ma, label=f"Vds = {vds:.1f} V")

    ax.axvline(vth, color='r', linestyle='--', alpha=0.5, label=f"Vth = {vth:.2f} V")

    ax.set_xlabel("Vgs (V)")
    ax.set_ylabel("Id (mA)")
    ax.set_title("MOSFET Transfer Characteristics")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if log_scale:
        ax.set_yscale('log')
        ax.set_ylim(bottom=1e-6)
    else:
        ax.set_ylim(bottom=0)

    return fig


def plot_device_cross_section(
    params: MOSFETParams,
    vgs: float,
    vds: float,
) -> Figure:
    """
    Plot MOSFET cross-section showing channel formation.

    Visualizes the device structure with color-coded regions
    indicating carrier concentration in the channel.

    Args:
        params: MOSFET device parameters
        vgs: Gate-source voltage
        vds: Drain-source voltage

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Normalized dimensions for visualization
    l_total = 3.0  # Arbitrary units
    h_sub = 1.0
    h_ox = 0.15
    w_sd = 0.6  # Source/drain width

    # Draw substrate (p-type body)
    substrate = plt.Rectangle((0, 0), l_total, h_sub,
                               facecolor='lightyellow', edgecolor='black')
    ax.add_patch(substrate)

    # Draw source (n+)
    source = plt.Rectangle((0, h_sub - 0.3), w_sd, 0.3,
                           facecolor='lightblue', edgecolor='black')
    ax.add_patch(source)

    # Draw drain (n+)
    drain = plt.Rectangle((l_total - w_sd, h_sub - 0.3), w_sd, 0.3,
                          facecolor='lightblue', edgecolor='black')
    ax.add_patch(drain)

    # Draw gate oxide
    gate_ox = plt.Rectangle((w_sd, h_sub), l_total - 2*w_sd, h_ox,
                            facecolor='lightgray', edgecolor='black')
    ax.add_patch(gate_ox)

    # Draw gate metal
    gate = plt.Rectangle((w_sd, h_sub + h_ox), l_total - 2*w_sd, 0.1,
                         facecolor='orange', edgecolor='black')
    ax.add_patch(gate)

    # Draw channel if inverted
    vth = threshold_voltage(params)
    if vgs > vth:
        # Channel exists - color intensity based on charge
        channel_alpha = min((vgs - vth) / 2.0, 0.8)
        channel = plt.Rectangle((w_sd, h_sub - 0.05), l_total - 2*w_sd, 0.05,
                                facecolor='blue', alpha=channel_alpha, edgecolor='none')
        ax.add_patch(channel)
        channel_text = "Channel ON"
    else:
        channel_text = "Channel OFF"

    # Labels
    ax.text(w_sd/2, h_sub - 0.15, "Source\n(n+)", ha='center', va='center', fontsize=9)
    ax.text(l_total - w_sd/2, h_sub - 0.15, "Drain\n(n+)", ha='center', va='center', fontsize=9)
    ax.text(l_total/2, h_sub/2, "P-type\nSubstrate", ha='center', va='center', fontsize=10)
    ax.text(l_total/2, h_sub + h_ox/2, "Oxide", ha='center', va='center', fontsize=8)
    ax.text(l_total/2, h_sub + h_ox + 0.05, "Gate", ha='center', va='center', fontsize=9)

    # Title with operating point
    ax.set_title(f"MOSFET Cross-Section | Vgs={vgs:.2f}V, Vds={vds:.2f}V, Vth={vth:.2f}V\n{channel_text}")

    ax.set_xlim(-0.1, l_total + 0.1)
    ax.set_ylim(-0.1, h_sub + h_ox + 0.3)
    ax.set_aspect('equal')
    ax.axis('off')

    return fig


def plot_band_diagram(
    params: MOSFETParams,
    vgs: float,
    depth_max_nm: float = 100.0,
) -> Figure:
    """
    Plot energy band diagram perpendicular to gate.

    Args:
        params: MOSFET device parameters
        vgs: Gate-source voltage
        depth_max_nm: Maximum depth in nanometers

    Returns:
        Matplotlib Figure object
    """
    from .physics import band_diagram_at_gate

    depth = np.linspace(0, depth_max_nm * 1e-9, 200)
    bands = band_diagram_at_gate(params, vgs, depth)

    depth_nm = depth * 1e9  # Convert to nm for plotting

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(depth_nm, bands['ec'], 'b-', label='Ec', linewidth=2)
    ax.plot(depth_nm, bands['ev'], 'r-', label='Ev', linewidth=2)
    ax.plot(depth_nm, bands['ei'], 'g--', label='Ei', linewidth=1.5)
    ax.plot(depth_nm, bands['ef'], 'k--', label='Ef', linewidth=1.5)

    ax.set_xlabel("Depth from oxide interface (nm)")
    ax.set_ylabel("Energy (eV)")
    ax.set_title(f"Band Diagram at Vgs = {vgs:.2f} V")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig
