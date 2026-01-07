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
        # 3D MOSFET Visualization with Fluid-like Gradients

        This notebook renders electron concentration using GPU-accelerated
        volume ray marching, providing smooth gradients similar to COMSOL.

        **Controls:**
        - Drag to rotate
        - Scroll to zoom
        - Shift+drag to pan
        - Play/Pause animation
        - Slider to scrub through operating modes
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import json
    import sys
    sys.path.insert(0, "../src")

    from mosfet_sim import (
        MOSFETParams,
        threshold_voltage,
        oxide_capacitance,
        drain_current,
        create_device_mesh,
        compute_biased_concentrations,
        generate_concentration_sweep,
        generate_output_sweep,
        export_complete_visualization,
        plot_output_characteristics,
        plot_transfer_characteristics,
    )
    return (
        MOSFETParams,
        compute_biased_concentrations,
        create_device_mesh,
        drain_current,
        export_complete_visualization,
        generate_concentration_sweep,
        generate_output_sweep,
        json,
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
    mo.md("## 3D Electron Concentration (WebGL)")
    return


@app.cell
def _(
    create_device_mesh,
    device,
    export_complete_visualization,
    generate_concentration_sweep,
    generate_output_sweep,
    json,
    mo,
    n_frames_slider,
    np,
    sweep_mode,
    sweep_params,
    vth,
):
    # Generate visualization data
    mo.output.append(mo.md("Computing simulation data..."))

    mesh = create_device_mesh(device, nx=40, ny=15, nz=25)

    if sweep_mode.value == "vgs":
        param_values = np.linspace(
            sweep_params["vgs_range"].value[0],
            sweep_params["vgs_range"].value[1],
            n_frames_slider.value
        )
        concentrations = generate_concentration_sweep(
            mesh, param_values, vds=sweep_params["vds_fixed"].value
        )
        param_name = "Vgs"
    else:
        param_values = np.linspace(
            sweep_params["vds_range"].value[0],
            sweep_params["vds_range"].value[1],
            n_frames_slider.value
        )
        concentrations = generate_output_sweep(
            mesh, vgs=sweep_params["vgs_fixed"].value, vds_values=param_values
        )
        param_name = "Vds"

    viz_data = export_complete_visualization(
        mesh, concentrations, param_values, param_name
    )
    viz_json = json.dumps(viz_data)

    mo.output.clear()
    return (
        concentrations,
        mesh,
        param_name,
        param_values,
        viz_data,
        viz_json,
    )


@app.cell
def _(mo, param_name, vth, viz_json):
    # Canvas-based 2D slice visualization with animation
    # This is more reliable than complex WebGL 3D ray marching
    canvas_artifact = mo.Html(f'''
    <div id="mosfet-container" style="width: 100%; max-width: 900px; margin: 0 auto;">
      <div id="status-bar" style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #e0e0e0;
        padding: 12px 20px;
        border-radius: 8px 8px 0 0;
        font-family: 'SF Mono', 'Consolas', monospace;
        font-size: 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border: 1px solid #2a2a4a;
        border-bottom: none;
      ">
        <span id="param-display">{param_name} = 0.00 V</span>
        <span id="vth-display">Vth = {vth:.2f} V</span>
        <span id="mode-display" style="
          padding: 4px 12px;
          border-radius: 4px;
          font-weight: 600;
          background: #dc3545;
        ">CUTOFF</span>
      </div>

      <canvas id="mosfet-canvas" width="900" height="500" style="
        display: block;
        background: #14141e;
        border: 1px solid #2a2a4a;
      "></canvas>

      <div id="controls" style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 16px 20px;
        border-radius: 0 0 8px 8px;
        border: 1px solid #2a2a4a;
        border-top: none;
      ">
        <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 12px;">
          <button id="play-btn" style="
            background: #4CAF50;
            color: white;
            border: none;
            padding: 10px 24px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
          ">Play</button>

          <input type="range" id="frame-slider" min="0" max="29" value="0" style="
            flex: 1;
            height: 6px;
            -webkit-appearance: none;
            background: #2a2a4a;
            border-radius: 3px;
          " />

          <span id="frame-counter" style="
            color: #a0a0a0;
            font-family: monospace;
            min-width: 70px;
            text-align: right;
          ">1 / 30</span>
        </div>

        <div style="display: flex; align-items: center; gap: 16px;">
          <label style="color: #a0a0a0; font-size: 12px;">Y-Slice:</label>
          <input type="range" id="slice-slider" min="0" max="100" value="50" style="
            width: 150px;
            height: 4px;
            -webkit-appearance: none;
            background: #2a2a4a;
            border-radius: 2px;
          " />
          <span style="color: #808080; font-size: 12px;">Cross-section view through device</span>
        </div>
      </div>
    </div>

    <script>
    (function() {{
      const VIZ_DATA = {viz_json};
      const VTH = {vth};
      const PARAM_NAME = "{param_name}";

      // Viridis colormap (full 256 colors)
      function viridis(t) {{
        t = Math.max(0, Math.min(1, t));
        // Simplified viridis approximation
        let r, g, b;
        if (t < 0.25) {{
          r = 68 + t * 4 * (49 - 68);
          g = 1 + t * 4 * (104 - 1);
          b = 84 + t * 4 * (142 - 84);
        }} else if (t < 0.5) {{
          const s = (t - 0.25) * 4;
          r = 49 + s * (35 - 49);
          g = 104 + s * (139 - 104);
          b = 142 + s * (141 - 142);
        }} else if (t < 0.75) {{
          const s = (t - 0.5) * 4;
          r = 35 + s * (126 - 35);
          g = 139 + s * (189 - 139);
          b = 141 + s * (99 - 141);
        }} else {{
          const s = (t - 0.75) * 4;
          r = 126 + s * (253 - 126);
          g = 189 + s * (231 - 189);
          b = 99 + s * (37 - 99);
        }}
        return [Math.round(r), Math.round(g), Math.round(b)];
      }}

      const canvas = document.getElementById('mosfet-canvas');
      const ctx = canvas.getContext('2d');

      // Decode base64 to typed array
      function decodeBase64Float32(str) {{
        const binary = atob(str);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {{
          bytes[i] = binary.charCodeAt(i);
        }}
        return new Float32Array(bytes.buffer);
      }}

      function decodeBase64Uint8(str) {{
        const binary = atob(str);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {{
          bytes[i] = binary.charCodeAt(i);
        }}
        return bytes;
      }}

      // Parse data
      const dims = VIZ_DATA.mesh.dimensions;  // [nx, ny, nz]
      const bounds = VIZ_DATA.mesh.bounds;
      const regions = decodeBase64Uint8(VIZ_DATA.mesh.regions);
      const frames = VIZ_DATA.animation.frames.map(f => decodeBase64Float32(f));
      const paramValues = VIZ_DATA.animation.parameter_values;
      const numFrames = frames.length;

      // State
      let currentFrame = 0;
      let isPlaying = false;
      let sliceY = 0.5;  // 0-1, middle of device

      // UI elements
      const playBtn = document.getElementById('play-btn');
      const frameSlider = document.getElementById('frame-slider');
      const sliceSlider = document.getElementById('slice-slider');
      const frameCounter = document.getElementById('frame-counter');
      const paramDisplay = document.getElementById('param-display');
      const modeDisplay = document.getElementById('mode-display');

      frameSlider.max = numFrames - 1;

      // Get value at position in 3D array (stored as flat array in x,y,z order)
      function getValue(data, ix, iy, iz) {{
        // Data is stored as [nx][ny][nz] flattened
        const idx = ix * dims[1] * dims[2] + iy * dims[2] + iz;
        return data[idx];
      }}

      // Region colors for overlay
      const regionColors = {{
        0: [40, 40, 60, 50],    // Body - dark, mostly transparent
        1: [100, 149, 237, 180], // Source - blue
        2: [100, 149, 237, 180], // Drain - blue
        3: [255, 255, 200, 80],  // Channel - light yellow, semi-transparent
        4: [180, 180, 180, 120], // Oxide - gray
        5: [255, 165, 0, 200],   // Gate - orange
      }};

      function render() {{
        const width = canvas.width;
        const height = canvas.height;
        const imageData = ctx.createImageData(width, height);
        const data = imageData.data;

        const frame = frames[currentFrame];
        const yIdx = Math.floor(sliceY * (dims[1] - 1));

        // Calculate display bounds
        const xMin = bounds[0][0], xMax = bounds[0][1];
        const zMin = bounds[2][0], zMax = bounds[2][1];

        // Margins for labels
        const marginLeft = 60;
        const marginRight = 80;
        const marginTop = 40;
        const marginBottom = 50;

        const plotWidth = width - marginLeft - marginRight;
        const plotHeight = height - marginTop - marginBottom;

        // Clear to background
        for (let i = 0; i < data.length; i += 4) {{
          data[i] = 20;
          data[i + 1] = 20;
          data[i + 2] = 30;
          data[i + 3] = 255;
        }}

        // Draw concentration heatmap
        for (let px = 0; px < plotWidth; px++) {{
          for (let py = 0; py < plotHeight; py++) {{
            // Map pixel to data coordinates
            const xNorm = px / plotWidth;
            const zNorm = 1 - py / plotHeight;  // Flip Y

            const ix = Math.floor(xNorm * (dims[0] - 1));
            const iz = Math.floor(zNorm * (dims[2] - 1));

            // Get concentration value (already normalized 0-1)
            const conc = getValue(frame, ix, yIdx, iz);
            const region = getValue(regions, ix, yIdx, iz);

            // Apply colormap
            const [r, g, b] = viridis(conc);

            // Get region color for blending
            const regColor = regionColors[region] || [100, 100, 100, 50];

            // Blend concentration with region color
            const alpha = regColor[3] / 255;
            const finalR = Math.round(r * (1 - alpha * 0.3) + regColor[0] * alpha * 0.3);
            const finalG = Math.round(g * (1 - alpha * 0.3) + regColor[1] * alpha * 0.3);
            const finalB = Math.round(b * (1 - alpha * 0.3) + regColor[2] * alpha * 0.3);

            const canvasX = marginLeft + px;
            const canvasY = marginTop + py;
            const idx = (canvasY * width + canvasX) * 4;

            data[idx] = finalR;
            data[idx + 1] = finalG;
            data[idx + 2] = finalB;
            data[idx + 3] = 255;
          }}
        }}

        ctx.putImageData(imageData, 0, 0);

        // Draw axes and labels
        ctx.strokeStyle = '#606080';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.rect(marginLeft, marginTop, plotWidth, plotHeight);
        ctx.stroke();

        // Axis labels
        ctx.fillStyle = '#a0a0a0';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('X (um) - Channel Direction', marginLeft + plotWidth / 2, height - 10);

        ctx.save();
        ctx.translate(15, marginTop + plotHeight / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('Z (um) - Depth', 0, 0);
        ctx.restore();

        // X axis ticks
        ctx.textAlign = 'center';
        for (let i = 0; i <= 4; i++) {{
          const x = marginLeft + (plotWidth * i / 4);
          const val = xMin + (xMax - xMin) * i / 4;
          ctx.fillText(val.toFixed(1), x, height - marginBottom + 20);
        }}

        // Z axis ticks
        ctx.textAlign = 'right';
        for (let i = 0; i <= 4; i++) {{
          const y = marginTop + (plotHeight * i / 4);
          const val = zMax - (zMax - zMin) * i / 4;
          ctx.fillText(val.toFixed(2), marginLeft - 5, y + 4);
        }}

        // Colorbar
        const cbX = width - marginRight + 20;
        const cbWidth = 15;
        const cbHeight = plotHeight;

        for (let i = 0; i < cbHeight; i++) {{
          const t = 1 - i / cbHeight;
          const [r, g, b] = viridis(t);
          ctx.fillStyle = `rgb(${{r}},${{g}},${{b}})`;
          ctx.fillRect(cbX, marginTop + i, cbWidth, 1);
        }}

        ctx.strokeStyle = '#606080';
        ctx.strokeRect(cbX, marginTop, cbWidth, cbHeight);

        ctx.fillStyle = '#a0a0a0';
        ctx.textAlign = 'left';
        ctx.fillText('High', cbX + cbWidth + 5, marginTop + 10);
        ctx.fillText('Low', cbX + cbWidth + 5, marginTop + cbHeight);

        ctx.save();
        ctx.translate(cbX + cbWidth + 40, marginTop + cbHeight / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = 'center';
        ctx.fillText('log10(n)', 0, 0);
        ctx.restore();

        // Region legend
        ctx.font = '10px sans-serif';
        const legendY = marginTop + 10;
        const legendItems = [
          ['Source/Drain', 'rgb(100,149,237)'],
          ['Gate', 'rgb(255,165,0)'],
          ['Oxide', 'rgb(180,180,180)'],
          ['Channel', 'rgb(255,255,200)'],
        ];
        legendItems.forEach(([name, color], i) => {{
          ctx.fillStyle = color;
          ctx.fillRect(marginLeft + 5, legendY + i * 15, 10, 10);
          ctx.fillStyle = '#a0a0a0';
          ctx.textAlign = 'left';
          ctx.fillText(name, marginLeft + 20, legendY + i * 15 + 9);
        }});
      }}

      function updateUI() {{
        const paramVal = paramValues[currentFrame];
        paramDisplay.textContent = `${{PARAM_NAME}} = ${{paramVal.toFixed(2)}} V`;
        frameCounter.textContent = `${{currentFrame + 1}} / ${{numFrames}}`;
        frameSlider.value = currentFrame;

        if (PARAM_NAME === 'Vgs') {{
          if (paramVal < VTH) {{
            modeDisplay.textContent = 'CUTOFF';
            modeDisplay.style.background = '#dc3545';
            modeDisplay.style.color = '#fff';
          }} else if (paramVal < VTH + 0.5) {{
            modeDisplay.textContent = 'NEAR THRESHOLD';
            modeDisplay.style.background = '#ffc107';
            modeDisplay.style.color = '#000';
          }} else {{
            modeDisplay.textContent = 'ACTIVE';
            modeDisplay.style.background = '#28a745';
            modeDisplay.style.color = '#fff';
          }}
        }} else {{
          if (paramVal < 1.0) {{
            modeDisplay.textContent = 'LINEAR';
            modeDisplay.style.background = '#17a2b8';
          }} else {{
            modeDisplay.textContent = 'SATURATION';
            modeDisplay.style.background = '#6f42c1';
          }}
          modeDisplay.style.color = '#fff';
        }}
      }}

      // Event handlers
      playBtn.addEventListener('click', () => {{
        isPlaying = !isPlaying;
        playBtn.textContent = isPlaying ? 'Pause' : 'Play';
        playBtn.style.background = isPlaying ? '#dc3545' : '#4CAF50';
      }});

      frameSlider.addEventListener('input', () => {{
        currentFrame = parseInt(frameSlider.value);
        render();
        updateUI();
      }});

      sliceSlider.addEventListener('input', () => {{
        sliceY = parseInt(sliceSlider.value) / 100;
        render();
      }});

      // Animation loop
      let lastTime = 0;
      function animate(time) {{
        if (isPlaying && time - lastTime > 100) {{
          currentFrame = (currentFrame + 1) % numFrames;
          render();
          updateUI();
          lastTime = time;
        }}
        requestAnimationFrame(animate);
      }}

      // Initial render
      render();
      updateUI();
      requestAnimationFrame(animate);
    }})();
    </script>
    ''')

    canvas_artifact
    return (canvas_artifact,)


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

        ### WebGL Visualization

        The 3D visualization uses:
        - **Volume ray marching**: GPU-accelerated rendering of 3D concentration data
        - **Arcball camera**: Intuitive rotation with mouse/touch
        - **Pre-computed frames**: Smooth animation without CPU bottlenecks

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
