import marimo

__generated_with = "0.8.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # Interactive MOSFET Explorer

        This notebook visualizes MOSFET device physics with:
        - **2D animated cross-section** showing electron concentration
        - **Animated sweeps** through all operating modes
        - **Play/pause controls** with slider for frame selection

        ## Operating Modes Visualized

        | Mode | Condition | Channel |
        |------|-----------|---------|
        | **Cutoff** | Vgs < Vth | No inversion layer |
        | **Linear** | Vgs > Vth, Vds < Vgs-Vth | Full channel |
        | **Saturation** | Vgs > Vth, Vds >= Vgs-Vth | Pinched-off at drain |
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import sys
    sys.path.insert(0, "../src")

    from mosfet_sim import (
        MOSFETParams,
        threshold_voltage,
        oxide_capacitance,
        drain_current,
        create_device_mesh,
        compute_biased_concentrations,
        create_vgs_sweep_animation_2d,
        create_vds_sweep_animation_2d,
        create_cross_section_slice,
        plot_output_characteristics,
        plot_transfer_characteristics,
    )
    return (
        MOSFETParams,
        compute_biased_concentrations,
        create_cross_section_slice,
        create_device_mesh,
        create_vds_sweep_animation_2d,
        create_vgs_sweep_animation_2d,
        drain_current,
        np,
        oxide_capacitance,
        plot_output_characteristics,
        plot_transfer_characteristics,
        sys,
        threshold_voltage,
    )


@app.cell
def _(mo):
    mo.md("## Device Configuration")
    return


@app.cell
def _(mo):
    # Device geometry controls
    channel_length_um = mo.ui.slider(
        start=0.5,
        stop=5.0,
        step=0.25,
        value=1.0,
        label="Channel Length (um)",
    )
    oxide_thickness_nm = mo.ui.slider(
        start=5,
        stop=50,
        step=5,
        value=10,
        label="Oxide Thickness (nm)",
    )
    substrate_doping_exp = mo.ui.slider(
        start=15,
        stop=18,
        step=0.5,
        value=17,
        label="Substrate Doping: 10^x (cm^-3)",
    )

    mo.hstack(
        [
            mo.vstack([channel_length_um, oxide_thickness_nm, substrate_doping_exp]),
        ],
        justify="start",
    )
    return channel_length_um, oxide_thickness_nm, substrate_doping_exp


@app.cell
def _(MOSFETParams, channel_length_um, oxide_thickness_nm, substrate_doping_exp):
    # Create device from UI values
    device = MOSFETParams(
        channel_length=channel_length_um.value * 1e-6,
        channel_width=10e-6,
        oxide_thickness=oxide_thickness_nm.value * 1e-9,
        substrate_doping=10 ** substrate_doping_exp.value,
        source_drain_doping=1e20,
    )
    return (device,)


@app.cell
def _(device, mo, oxide_capacitance, threshold_voltage):
    vth = threshold_voltage(device)
    cox = oxide_capacitance(device)

    mo.md(
        f"""
        ## Device Parameters

        | Parameter | Value |
        |-----------|-------|
        | **Threshold Voltage (Vth)** | {vth:.3f} V |
        | **Oxide Capacitance (Cox)** | {cox*1e4:.4f} uF/cm^2 |
        | **Channel L/W** | {device.channel_length*1e6:.2f} / {device.channel_width*1e6:.1f} um |
        """
    )
    return cox, vth


@app.cell
def _(mo):
    mo.md(
        """
        ## Animation Mode Selection

        Choose which parameter to sweep:
        - **Vgs Sweep**: Watch channel form as gate voltage increases (cutoff -> active)
        - **Vds Sweep**: Watch channel pinch-off as drain voltage increases (linear -> saturation)
        """
    )
    return


@app.cell
def _(mo):
    sweep_mode = mo.ui.radio(
        options={
            "vgs": "Vgs Sweep (Cutoff -> Active)",
            "vds": "Vds Sweep (Linear -> Saturation)",
        },
        value="vgs",
        label="Sweep Mode",
    )
    sweep_mode
    return (sweep_mode,)


@app.cell
def _(mo, sweep_mode, vth):
    # Conditional UI based on sweep mode
    if sweep_mode.value == "vgs":
        vgs_range = mo.ui.range_slider(
            start=0.0,
            stop=5.0,
            step=0.1,
            value=[0.0, vth + 2.0],
            label="Vgs Range (V)",
        )
        vds_fixed = mo.ui.slider(
            start=0.1,
            stop=3.0,
            step=0.1,
            value=0.5,
            label="Fixed Vds (V)",
        )
        sweep_controls = mo.vstack([vgs_range, vds_fixed])
        sweep_params = {"vgs_range": vgs_range, "vds_fixed": vds_fixed}
    else:
        vgs_fixed = mo.ui.slider(
            start=0.5,
            stop=5.0,
            step=0.1,
            value=vth + 1.0,
            label="Fixed Vgs (V) [should be > Vth]",
        )
        vds_range = mo.ui.range_slider(
            start=0.0,
            stop=5.0,
            step=0.1,
            value=[0.0, 3.0],
            label="Vds Range (V)",
        )
        sweep_controls = mo.vstack([vgs_fixed, vds_range])
        sweep_params = {"vgs_fixed": vgs_fixed, "vds_range": vds_range}

    sweep_controls
    return sweep_controls, sweep_params, vds_fixed, vds_range, vgs_fixed, vgs_range


@app.cell
def _(mo):
    n_frames_slider = mo.ui.slider(
        start=10,
        stop=60,
        step=5,
        value=30,
        label="Animation Frames (more = smoother, slower to compute)",
    )
    n_frames_slider
    return (n_frames_slider,)


@app.cell
def _(mo):
    mo.md("## Electron Concentration Animation")
    return


@app.cell
def _(
    create_vds_sweep_animation_2d,
    create_vgs_sweep_animation_2d,
    device,
    mo,
    n_frames_slider,
    sweep_mode,
    sweep_params,
):
    # Generate animation using pure Python/Plotly
    mo.output.append(
        mo.md("Computing animation frames... (this may take a moment)")
    )

    if sweep_mode.value == "vgs":
        fig_animation = create_vgs_sweep_animation_2d(
            params=device,
            vgs_min=sweep_params["vgs_range"].value[0],
            vgs_max=sweep_params["vgs_range"].value[1],
            vds=sweep_params["vds_fixed"].value,
            n_frames=n_frames_slider.value,
            mesh_resolution=(50, 20, 30),
        )
    else:
        fig_animation = create_vds_sweep_animation_2d(
            params=device,
            vgs=sweep_params["vgs_fixed"].value,
            vds_min=sweep_params["vds_range"].value[0],
            vds_max=sweep_params["vds_range"].value[1],
            n_frames=n_frames_slider.value,
            mesh_resolution=(50, 20, 30),
        )

    mo.output.clear()
    fig_animation
    return (fig_animation,)


@app.cell
def _(mo):
    mo.md(
        """
        ---
        ## I-V Characteristics

        The plots below show the DC characteristics. The curves indicate
        operating modes:
        - Gray dashed: **Cutoff** (Vgs < Vth)
        - Solid colored: **Active** (Vgs > Vth)
        - Vertical dotted lines: Saturation boundary (Vds = Vgs - Vth)
        """
    )
    return


@app.cell
def _(device, plot_output_characteristics, vth):
    # Output characteristics with multiple Vgs values spanning modes
    vgs_list = [
        vth - 0.2,  # Cutoff
        vth + 0.5,  # Just above threshold
        vth + 1.0,  # Linear/Saturation
        vth + 1.5,  # Strong inversion
        vth + 2.0,  # Strong inversion
    ]
    fig_output = plot_output_characteristics(device, vgs_list, vds_max=5.0)
    fig_output
    return fig_output, vgs_list


@app.cell
def _(device, plot_transfer_characteristics):
    # Transfer characteristics
    vds_list = [0.1, 0.5, 1.0, 2.0, 3.0]
    fig_transfer = plot_transfer_characteristics(device, vds_list, vgs_max=5.0)
    fig_transfer
    return fig_transfer, vds_list


@app.cell
def _(mo):
    mo.md(
        """
        ---
        ## Static Cross-Section View

        Static slice through the device at the current bias point (center of sweep range).
        """
    )
    return


@app.cell
def _(
    compute_biased_concentrations,
    create_cross_section_slice,
    create_device_mesh,
    device,
    sweep_mode,
    sweep_params,
    vth,
):
    # Create cross-section at mid-point of sweep
    mesh_2d = create_device_mesh(device, nx=60, ny=20, nz=40)

    if sweep_mode.value == "vgs":
        vgs_mid = (
            sweep_params["vgs_range"].value[0] + sweep_params["vgs_range"].value[1]
        ) / 2
        vds_mid = sweep_params["vds_fixed"].value
    else:
        vgs_mid = sweep_params["vgs_fixed"].value
        vds_mid = (
            sweep_params["vds_range"].value[0] + sweep_params["vds_range"].value[1]
        ) / 2

    conc_2d = compute_biased_concentrations(mesh_2d, vgs=vgs_mid, vds=vds_mid)
    fig_slice = create_cross_section_slice(conc_2d)
    fig_slice.update_layout(
        title=f"Cross-Section at Vgs={vgs_mid:.2f}V, Vds={vds_mid:.2f}V (Vth={vth:.2f}V)"
    )
    fig_slice
    return conc_2d, fig_slice, mesh_2d, vds_mid, vgs_mid


@app.cell
def _(mo):
    mo.md(
        """
        ---
        ## Physics Notes

        ### Square-Law Model Assumptions

        This simulation uses the **gradual channel approximation**:

        1. **Long-channel device**: L >> depletion widths
        2. **Gradual variation**: Electric field mainly vertical under gate
        3. **Drift-dominated**: Diffusion current neglected
        4. **Constant mobility**: No velocity saturation

        ### Electron Concentration Model

        The channel electron concentration is modeled as:

        n(x, z) = n0 * exp(-z/lambda) * f(Vov - V(x))

        Where:
        - n0: Surface concentration (depends on Vgs - Vth)
        - lambda: Inversion layer thickness (~2-5 nm)
        - V(x): Channel potential varying from 0 (source) to Vds (drain)

        ### Limitations

        - No short-channel effects (DIBL, velocity saturation)
        - Simplified band bending model
        - No quantum confinement
        - Uniform doping profiles
        """
    )
    return


if __name__ == "__main__":
    app.run()
