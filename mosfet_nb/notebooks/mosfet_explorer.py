import marimo

__generated_with = "0.8.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # Interactive MOSFET Explorer

        This notebook demonstrates MOSFET device physics using the
        gradual channel approximation (square-law model).

        Adjust the device parameters and operating voltages using the sliders
        below to explore how they affect the device characteristics.
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
        plot_output_characteristics,
        plot_transfer_characteristics,
        plot_device_cross_section,
        plot_band_diagram,
    )
    return (
        MOSFETParams,
        np,
        oxide_capacitance,
        plot_band_diagram,
        plot_device_cross_section,
        plot_output_characteristics,
        plot_transfer_characteristics,
        sys,
        threshold_voltage,
    )


@app.cell
def _(mo):
    # Device geometry sliders
    channel_length_slider = mo.ui.slider(
        start=0.5,
        stop=5.0,
        step=0.1,
        value=1.0,
        label="Channel Length (um)",
    )
    oxide_thickness_slider = mo.ui.slider(
        start=5,
        stop=50,
        step=1,
        value=10,
        label="Oxide Thickness (nm)",
    )
    doping_slider = mo.ui.slider(
        start=1e16,
        stop=1e18,
        step=1e16,
        value=1e17,
        label="Substrate Doping (cm^-3)",
    )

    mo.vstack(
        [
            mo.md("## Device Parameters"),
            channel_length_slider,
            oxide_thickness_slider,
            doping_slider,
        ]
    )
    return channel_length_slider, doping_slider, oxide_thickness_slider


@app.cell
def _(mo):
    # Operating point sliders
    vgs_slider = mo.ui.slider(
        start=0.0,
        stop=5.0,
        step=0.1,
        value=2.0,
        label="Gate-Source Voltage Vgs (V)",
    )
    vds_slider = mo.ui.slider(
        start=0.0,
        stop=5.0,
        step=0.1,
        value=2.0,
        label="Drain-Source Voltage Vds (V)",
    )

    mo.vstack(
        [
            mo.md("## Operating Point"),
            vgs_slider,
            vds_slider,
        ]
    )
    return vds_slider, vgs_slider


@app.cell
def _(MOSFETParams, channel_length_slider, doping_slider, oxide_thickness_slider):
    # Create device from slider values
    device = MOSFETParams(
        channel_length=channel_length_slider.value * 1e-6,
        channel_width=10e-6,
        oxide_thickness=oxide_thickness_slider.value * 1e-9,
        substrate_doping=doping_slider.value,
        source_drain_doping=1e20,
    )
    return (device,)


@app.cell
def _(device, mo, oxide_capacitance, threshold_voltage):
    vth = threshold_voltage(device)
    cox = oxide_capacitance(device)

    mo.md(
        f"""
        ## Calculated Parameters

        | Parameter | Value |
        |-----------|-------|
        | Threshold Voltage (Vth) | {vth:.3f} V |
        | Oxide Capacitance (Cox) | {cox*1e4:.3f} uF/cm^2 |
        """
    )
    return cox, vth


@app.cell
def _(device, plot_device_cross_section, vds_slider, vgs_slider):
    fig_cross = plot_device_cross_section(device, vgs_slider.value, vds_slider.value)
    fig_cross
    return (fig_cross,)


@app.cell
def _(device, plot_output_characteristics, vth):
    vgs_list = [vth + 0.5, vth + 1.0, vth + 1.5, vth + 2.0]
    fig_output = plot_output_characteristics(device, vgs_list, vds_max=5.0)
    fig_output
    return fig_output, vgs_list


@app.cell
def _(device, plot_transfer_characteristics):
    fig_transfer = plot_transfer_characteristics(device, [0.5, 1.0, 2.0, 3.0])
    fig_transfer
    return (fig_transfer,)


@app.cell
def _(device, plot_band_diagram, vgs_slider):
    fig_band = plot_band_diagram(device, vgs_slider.value)
    fig_band
    return (fig_band,)


@app.cell
def _(mo):
    mo.md(
        """
        ## Educational Notes

        ### Square-Law Model (Long-Channel Approximation)

        This simulation uses the gradual channel approximation for MOSFETs:

        **Linear Region** (Vds < Vgs - Vth):
        ```
        Id = (W/L) * mu_n * Cox * [(Vgs - Vth)*Vds - Vds^2/2]
        ```

        **Saturation Region** (Vds >= Vgs - Vth):
        ```
        Id = (W/L) * mu_n * Cox * (Vgs - Vth)^2 / 2
        ```

        ### Key Parameters

        - **Threshold Voltage (Vth)**: The gate voltage required to form an inversion channel
        - **Oxide Capacitance (Cox)**: Capacitance per unit area of the gate oxide
        - **Body Effect**: Modifies Vth based on source-body voltage

        ### Limitations

        This model assumes:
        - Long-channel behavior (no velocity saturation)
        - Uniform doping
        - No short-channel effects
        - Ideal contacts
        """
    )
    return


if __name__ == "__main__":
    app.run()
