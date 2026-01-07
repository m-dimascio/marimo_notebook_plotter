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
    # WebGL React artifact for 3D visualization
    webgl_artifact = mo.Html(f'''
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
        cursor: grab;
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
            transition: all 0.2s;
          ">Play</button>

          <input type="range" id="frame-slider" min="0" max="29" value="0" style="
            flex: 1;
            height: 6px;
            -webkit-appearance: none;
            background: #2a2a4a;
            border-radius: 3px;
            outline: none;
          " />

          <span id="frame-counter" style="
            color: #a0a0a0;
            font-family: monospace;
            min-width: 70px;
            text-align: right;
          ">1 / 30</span>
        </div>

        <div style="color: #808080; font-size: 12px; text-align: center;">
          Drag to rotate | Scroll to zoom | Shift+drag to pan
        </div>
      </div>
    </div>

    <script>
    (function() {{
      const VIZ_DATA = {viz_json};
      const VTH = {vth};
      const PARAM_NAME = "{param_name}";

      // Viridis colormap
      const VIRIDIS = [
        [68,1,84],[70,8,92],[71,16,99],[72,24,106],[72,32,113],[72,40,120],[71,47,125],[69,55,129],
        [66,62,133],[63,71,136],[60,79,139],[57,86,141],[53,93,142],[50,100,143],[46,107,144],[43,114,144],
        [40,121,144],[38,128,143],[36,135,142],[35,141,140],[35,148,137],[35,155,133],[36,162,128],[40,168,123],
        [47,175,116],[55,181,108],[65,186,100],[76,190,91],[88,194,82],[101,198,72],[115,201,59],[130,204,50],
        [145,206,38],[161,209,26],[177,211,15],[194,212,7],[210,213,10],[226,214,22],[241,216,43],[253,219,79]
      ];

      function viridis(t) {{
        t = Math.max(0, Math.min(1, t));
        const idx = Math.min(Math.floor(t * (VIRIDIS.length - 1)), VIRIDIS.length - 2);
        const f = t * (VIRIDIS.length - 1) - idx;
        const c1 = VIRIDIS[idx], c2 = VIRIDIS[idx + 1];
        return [
          c1[0] + f * (c2[0] - c1[0]),
          c1[1] + f * (c2[1] - c1[1]),
          c1[2] + f * (c2[2] - c1[2])
        ];
      }}

      // Canvas and WebGL setup
      const canvas = document.getElementById('mosfet-canvas');
      const gl = canvas.getContext('webgl2', {{ antialias: true, alpha: false }});

      if (!gl) {{
        console.error('WebGL2 not supported');
        document.getElementById('mosfet-container').innerHTML =
          '<div style="color:red;padding:20px;">WebGL2 not supported in this browser</div>';
        return;
      }}

      // State
      let currentFrame = 0;
      let isPlaying = false;
      let animationId = null;
      let frameTextures = [];
      let regionTexture = null;
      let needsRender = true;

      // Camera state
      const camera = {{
        rotation: [0, 0, 0, 1],
        distance: 3.0,
        center: [0, 0, 0],
        isDragging: false,
        isPanning: false,
        lastMouse: [0, 0]
      }};

      // Decode base64
      function decodeBase64(str) {{
        const binary = atob(str);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
        return bytes;
      }}

      // Parse visualization data
      const dims = VIZ_DATA.mesh.dimensions;
      const bounds = {{
        min: [VIZ_DATA.mesh.bounds[0][0], VIZ_DATA.mesh.bounds[1][0], VIZ_DATA.mesh.bounds[2][0]],
        max: [VIZ_DATA.mesh.bounds[0][1], VIZ_DATA.mesh.bounds[1][1], VIZ_DATA.mesh.bounds[2][1]]
      }};

      // Fit camera to bounds
      camera.center = [
        (bounds.min[0] + bounds.max[0]) / 2,
        (bounds.min[1] + bounds.max[1]) / 2,
        (bounds.min[2] + bounds.max[2]) / 2
      ];
      const size = Math.max(
        bounds.max[0] - bounds.min[0],
        bounds.max[1] - bounds.min[1],
        bounds.max[2] - bounds.min[2]
      );
      camera.distance = size * 2.5;

      // Create 3D texture
      function create3DTexture(data, isFloat) {{
        const texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_3D, texture);

        if (isFloat) {{
          gl.texImage3D(gl.TEXTURE_3D, 0, gl.R32F, dims[0], dims[1], dims[2], 0, gl.RED, gl.FLOAT, data);
        }} else {{
          gl.texImage3D(gl.TEXTURE_3D, 0, gl.R8, dims[0], dims[1], dims[2], 0, gl.RED, gl.UNSIGNED_BYTE, data);
        }}

        gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);

        return texture;
      }}

      // Create colormap texture
      function createColormapTexture() {{
        const texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texture);

        const data = new Uint8Array(256 * 4);
        for (let i = 0; i < 256; i++) {{
          const c = viridis(i / 255);
          data[i * 4] = c[0];
          data[i * 4 + 1] = c[1];
          data[i * 4 + 2] = c[2];
          data[i * 4 + 3] = 255;
        }}

        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        return texture;
      }}

      // Shader sources
      const vertexSource = `#version 300 es
        in vec2 aPosition;
        out vec2 vUV;
        void main() {{
          vUV = aPosition * 0.5 + 0.5;
          gl_Position = vec4(aPosition, 0.0, 1.0);
        }}
      `;

      const fragmentSource = `#version 300 es
        precision highp float;
        precision highp sampler3D;

        uniform sampler3D uVolume;
        uniform sampler2D uColormap;
        uniform sampler3D uRegions;
        uniform vec3 uBoundsMin;
        uniform vec3 uBoundsMax;
        uniform mat4 uInvViewProj;
        uniform float uOpacityScale;
        uniform int uMaxSteps;

        in vec2 vUV;
        out vec4 fragColor;

        vec4 applyColormap(float v) {{
          return texture(uColormap, vec2(clamp(v, 0.0, 1.0), 0.5));
        }}

        float getRegionOpacity(float r) {{
          int region = int(r * 255.0);
          if (region == 4 || region == 5) return 0.15;
          if (region == 1 || region == 2) return 0.6;
          if (region == 3) return 1.0;
          return 0.2;
        }}

        void main() {{
          vec4 nearP = uInvViewProj * vec4(vUV * 2.0 - 1.0, -1.0, 1.0);
          vec4 farP = uInvViewProj * vec4(vUV * 2.0 - 1.0, 1.0, 1.0);
          nearP /= nearP.w; farP /= farP.w;

          vec3 rayOrigin = nearP.xyz;
          vec3 rayDir = normalize(farP.xyz - nearP.xyz);

          vec3 invDir = 1.0 / rayDir;
          vec3 t0 = (uBoundsMin - rayOrigin) * invDir;
          vec3 t1 = (uBoundsMax - rayOrigin) * invDir;
          vec3 tmin = min(t0, t1);
          vec3 tmax = max(t0, t1);

          float tNear = max(max(tmin.x, tmin.y), tmin.z);
          float tFar = min(min(tmax.x, tmax.y), tmax.z);

          if (tNear > tFar || tFar < 0.0) {{
            fragColor = vec4(0.08, 0.08, 0.12, 1.0);
            return;
          }}

          tNear = max(tNear, 0.0);
          float stepSize = (tFar - tNear) / float(uMaxSteps);
          vec3 pos = rayOrigin + rayDir * tNear;
          vec3 step = rayDir * stepSize;

          vec4 acc = vec4(0.0);

          for (int i = 0; i < 200; i++) {{
            if (i >= uMaxSteps) break;

            vec3 tc = (pos - uBoundsMin) / (uBoundsMax - uBoundsMin);

            if (all(greaterThanEqual(tc, vec3(0.0))) && all(lessThanEqual(tc, vec3(1.0)))) {{
              float conc = texture(uVolume, tc).r;
              float region = texture(uRegions, tc).r;

              if (conc > 0.05) {{
                vec4 col = applyColormap(conc);
                float opacity = col.a * uOpacityScale * getRegionOpacity(region) * (0.3 + conc * 0.7);
                col.a = opacity;
                col.rgb *= col.a;
                acc += col * (1.0 - acc.a);
                if (acc.a > 0.95) break;
              }}
            }}

            pos += step;
          }}

          vec3 bg = vec3(0.08, 0.08, 0.12);
          fragColor = vec4(acc.rgb + bg * (1.0 - acc.a), 1.0);
        }}
      `;

      // Compile shaders
      function compileShader(type, source) {{
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {{
          console.error('Shader error:', gl.getShaderInfoLog(shader));
          return null;
        }}
        return shader;
      }}

      const vertShader = compileShader(gl.VERTEX_SHADER, vertexSource);
      const fragShader = compileShader(gl.FRAGMENT_SHADER, fragmentSource);

      const program = gl.createProgram();
      gl.attachShader(program, vertShader);
      gl.attachShader(program, fragShader);
      gl.linkProgram(program);

      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {{
        console.error('Program link error:', gl.getProgramInfoLog(program));
      }}

      // Get uniforms
      const uniforms = {{
        uVolume: gl.getUniformLocation(program, 'uVolume'),
        uColormap: gl.getUniformLocation(program, 'uColormap'),
        uRegions: gl.getUniformLocation(program, 'uRegions'),
        uBoundsMin: gl.getUniformLocation(program, 'uBoundsMin'),
        uBoundsMax: gl.getUniformLocation(program, 'uBoundsMax'),
        uInvViewProj: gl.getUniformLocation(program, 'uInvViewProj'),
        uOpacityScale: gl.getUniformLocation(program, 'uOpacityScale'),
        uMaxSteps: gl.getUniformLocation(program, 'uMaxSteps')
      }};

      // Create quad
      const quadBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);

      const vao = gl.createVertexArray();
      gl.bindVertexArray(vao);
      const aPos = gl.getAttribLocation(program, 'aPosition');
      gl.enableVertexAttribArray(aPos);
      gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

      // Load textures
      regionTexture = create3DTexture(decodeBase64(VIZ_DATA.mesh.regions), false);

      VIZ_DATA.animation.frames.forEach(frameB64 => {{
        const bytes = decodeBase64(frameB64);
        const floats = new Float32Array(bytes.buffer);
        frameTextures.push(create3DTexture(floats, true));
      }});

      const colormapTexture = createColormapTexture();

      // Matrix math
      function multiplyMatrices(a, b) {{
        const out = new Float32Array(16);
        for (let i = 0; i < 4; i++) {{
          for (let j = 0; j < 4; j++) {{
            out[i*4+j] = 0;
            for (let k = 0; k < 4; k++) {{
              out[i*4+j] += a[k*4+j] * b[i*4+k];
            }}
          }}
        }}
        return out;
      }}

      function invertMatrix(m) {{
        const inv = new Float32Array(16);
        inv[0] = m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];
        inv[4] = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15] - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10];
        inv[8] = m[4]*m[9]*m[15] - m[4]*m[11]*m[13] - m[8]*m[5]*m[15] + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];
        inv[12] = -m[4]*m[9]*m[14] + m[4]*m[10]*m[13] + m[8]*m[5]*m[14] - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9];
        inv[1] = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15] - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10];
        inv[5] = m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15] + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];
        inv[9] = -m[0]*m[9]*m[15] + m[0]*m[11]*m[13] + m[8]*m[1]*m[15] - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];
        inv[13] = m[0]*m[9]*m[14] - m[0]*m[10]*m[13] - m[8]*m[1]*m[14] + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];
        inv[2] = m[1]*m[6]*m[15] - m[1]*m[7]*m[14] - m[5]*m[2]*m[15] + m[5]*m[3]*m[14] + m[13]*m[2]*m[7] - m[13]*m[3]*m[6];
        inv[6] = -m[0]*m[6]*m[15] + m[0]*m[7]*m[14] + m[4]*m[2]*m[15] - m[4]*m[3]*m[14] - m[12]*m[2]*m[7] + m[12]*m[3]*m[6];
        inv[10] = m[0]*m[5]*m[15] - m[0]*m[7]*m[13] - m[4]*m[1]*m[15] + m[4]*m[3]*m[13] + m[12]*m[1]*m[7] - m[12]*m[3]*m[5];
        inv[14] = -m[0]*m[5]*m[14] + m[0]*m[6]*m[13] + m[4]*m[1]*m[14] - m[4]*m[2]*m[13] - m[12]*m[1]*m[6] + m[12]*m[2]*m[5];
        inv[3] = -m[1]*m[6]*m[11] + m[1]*m[7]*m[10] + m[5]*m[2]*m[11] - m[5]*m[3]*m[10] - m[9]*m[2]*m[7] + m[9]*m[3]*m[6];
        inv[7] = m[0]*m[6]*m[11] - m[0]*m[7]*m[10] - m[4]*m[2]*m[11] + m[4]*m[3]*m[10] + m[8]*m[2]*m[7] - m[8]*m[3]*m[6];
        inv[11] = -m[0]*m[5]*m[11] + m[0]*m[7]*m[9] + m[4]*m[1]*m[11] - m[4]*m[3]*m[9] - m[8]*m[1]*m[7] + m[8]*m[3]*m[5];
        inv[15] = m[0]*m[5]*m[10] - m[0]*m[6]*m[9] - m[4]*m[1]*m[10] + m[4]*m[2]*m[9] + m[8]*m[1]*m[6] - m[8]*m[2]*m[5];

        let det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12];
        if (Math.abs(det) < 1e-10) det = 1;
        det = 1.0 / det;
        for (let i = 0; i < 16; i++) inv[i] *= det;
        return inv;
      }}

      function getViewMatrix() {{
        const q = camera.rotation;
        const d = camera.distance;
        const c = camera.center;

        // Quaternion to rotation matrix
        const x = q[0], y = q[1], z = q[2], w = q[3];
        const x2 = x + x, y2 = y + y, z2 = z + z;
        const xx = x * x2, yx = y * x2, yy = y * y2;
        const zx = z * x2, zy = z * y2, zz = z * z2;
        const wx = w * x2, wy = w * y2, wz = w * z2;

        // Eye position
        const eye = [
          (zx + wy) * d + c[0],
          (zy - wx) * d + c[1],
          (1 - xx - yy) * d + c[2]
        ];

        // LookAt matrix
        const zAxis = [eye[0] - c[0], eye[1] - c[1], eye[2] - c[2]];
        const zLen = Math.sqrt(zAxis[0]*zAxis[0] + zAxis[1]*zAxis[1] + zAxis[2]*zAxis[2]);
        zAxis[0] /= zLen; zAxis[1] /= zLen; zAxis[2] /= zLen;

        const up = [0, 1, 0];
        const xAxis = [
          up[1] * zAxis[2] - up[2] * zAxis[1],
          up[2] * zAxis[0] - up[0] * zAxis[2],
          up[0] * zAxis[1] - up[1] * zAxis[0]
        ];
        const xLen = Math.sqrt(xAxis[0]*xAxis[0] + xAxis[1]*xAxis[1] + xAxis[2]*xAxis[2]);
        if (xLen > 0.0001) {{
          xAxis[0] /= xLen; xAxis[1] /= xLen; xAxis[2] /= xLen;
        }}

        const yAxis = [
          zAxis[1] * xAxis[2] - zAxis[2] * xAxis[1],
          zAxis[2] * xAxis[0] - zAxis[0] * xAxis[2],
          zAxis[0] * xAxis[1] - zAxis[1] * xAxis[0]
        ];

        return new Float32Array([
          xAxis[0], yAxis[0], zAxis[0], 0,
          xAxis[1], yAxis[1], zAxis[1], 0,
          xAxis[2], yAxis[2], zAxis[2], 0,
          -(xAxis[0]*eye[0] + xAxis[1]*eye[1] + xAxis[2]*eye[2]),
          -(yAxis[0]*eye[0] + yAxis[1]*eye[1] + yAxis[2]*eye[2]),
          -(zAxis[0]*eye[0] + zAxis[1]*eye[1] + zAxis[2]*eye[2]),
          1
        ]);
      }}

      function getProjectionMatrix() {{
        const aspect = canvas.width / canvas.height;
        const fov = Math.PI / 4;
        const near = 0.1, far = 100;
        const f = 1.0 / Math.tan(fov / 2);
        const nf = 1 / (near - far);

        return new Float32Array([
          f / aspect, 0, 0, 0,
          0, f, 0, 0,
          0, 0, (far + near) * nf, -1,
          0, 0, 2 * far * near * nf, 0
        ]);
      }}

      // Camera rotation
      function rotateCamera(dx, dy) {{
        const speed = 0.015;
        const ax = dy * speed;
        const ay = dx * speed;

        const ca = Math.cos(ax / 2), sa = Math.sin(ax / 2);
        const cb = Math.cos(ay / 2), sb = Math.sin(ay / 2);

        const q = camera.rotation;
        const qx = [sa, 0, 0, ca];
        const qy = [0, sb, 0, cb];

        // Multiply: qy * qx
        const temp = [
          qx[3]*qy[0] + qx[0]*qy[3] + qx[1]*qy[2] - qx[2]*qy[1],
          qx[3]*qy[1] - qx[0]*qy[2] + qx[1]*qy[3] + qx[2]*qy[0],
          qx[3]*qy[2] + qx[0]*qy[1] - qx[1]*qy[0] + qx[2]*qy[3],
          qx[3]*qy[3] - qx[0]*qy[0] - qx[1]*qy[1] - qx[2]*qy[2]
        ];

        // Multiply: temp * q
        camera.rotation = [
          temp[3]*q[0] + temp[0]*q[3] + temp[1]*q[2] - temp[2]*q[1],
          temp[3]*q[1] - temp[0]*q[2] + temp[1]*q[3] + temp[2]*q[0],
          temp[3]*q[2] + temp[0]*q[1] - temp[1]*q[0] + temp[2]*q[3],
          temp[3]*q[3] - temp[0]*q[0] - temp[1]*q[1] - temp[2]*q[2]
        ];

        // Normalize
        const len = Math.sqrt(
          camera.rotation[0]*camera.rotation[0] +
          camera.rotation[1]*camera.rotation[1] +
          camera.rotation[2]*camera.rotation[2] +
          camera.rotation[3]*camera.rotation[3]
        );
        camera.rotation[0] /= len;
        camera.rotation[1] /= len;
        camera.rotation[2] /= len;
        camera.rotation[3] /= len;

        needsRender = true;
      }}

      // Mouse events
      canvas.addEventListener('mousedown', (e) => {{
        if (e.shiftKey) {{
          camera.isPanning = true;
        }} else {{
          camera.isDragging = true;
        }}
        camera.lastMouse = [e.clientX, e.clientY];
        canvas.style.cursor = 'grabbing';
        e.preventDefault();
      }});

      canvas.addEventListener('mousemove', (e) => {{
        if (camera.isDragging) {{
          const dx = e.clientX - camera.lastMouse[0];
          const dy = e.clientY - camera.lastMouse[1];
          rotateCamera(dx, dy);
          camera.lastMouse = [e.clientX, e.clientY];
        }} else if (camera.isPanning) {{
          const dx = e.clientX - camera.lastMouse[0];
          const dy = e.clientY - camera.lastMouse[1];
          const panSpeed = camera.distance * 0.002;
          camera.center[0] -= dx * panSpeed;
          camera.center[1] += dy * panSpeed;
          camera.lastMouse = [e.clientX, e.clientY];
          needsRender = true;
        }}
      }});

      const stopDrag = () => {{
        camera.isDragging = false;
        camera.isPanning = false;
        canvas.style.cursor = 'grab';
      }};
      canvas.addEventListener('mouseup', stopDrag);
      canvas.addEventListener('mouseleave', stopDrag);

      canvas.addEventListener('wheel', (e) => {{
        e.preventDefault();
        camera.distance *= e.deltaY > 0 ? 1.1 : 0.9;
        camera.distance = Math.max(0.5, Math.min(30, camera.distance));
        needsRender = true;
      }}, {{ passive: false }});

      // Touch events
      let lastTouchDist = 0;
      canvas.addEventListener('touchstart', (e) => {{
        e.preventDefault();
        if (e.touches.length === 1) {{
          camera.isDragging = true;
          camera.lastMouse = [e.touches[0].clientX, e.touches[0].clientY];
        }} else if (e.touches.length === 2) {{
          const dx = e.touches[0].clientX - e.touches[1].clientX;
          const dy = e.touches[0].clientY - e.touches[1].clientY;
          lastTouchDist = Math.sqrt(dx*dx + dy*dy);
        }}
      }}, {{ passive: false }});

      canvas.addEventListener('touchmove', (e) => {{
        e.preventDefault();
        if (e.touches.length === 1 && camera.isDragging) {{
          const dx = e.touches[0].clientX - camera.lastMouse[0];
          const dy = e.touches[0].clientY - camera.lastMouse[1];
          rotateCamera(dx, dy);
          camera.lastMouse = [e.touches[0].clientX, e.touches[0].clientY];
        }} else if (e.touches.length === 2) {{
          const dx = e.touches[0].clientX - e.touches[1].clientX;
          const dy = e.touches[0].clientY - e.touches[1].clientY;
          const dist = Math.sqrt(dx*dx + dy*dy);
          camera.distance *= 1 + (lastTouchDist - dist) * 0.005;
          camera.distance = Math.max(0.5, Math.min(30, camera.distance));
          lastTouchDist = dist;
          needsRender = true;
        }}
      }}, {{ passive: false }});

      canvas.addEventListener('touchend', () => {{ camera.isDragging = false; }});

      // UI controls
      const playBtn = document.getElementById('play-btn');
      const frameSlider = document.getElementById('frame-slider');
      const frameCounter = document.getElementById('frame-counter');
      const paramDisplay = document.getElementById('param-display');
      const modeDisplay = document.getElementById('mode-display');

      const numFrames = frameTextures.length;
      frameSlider.max = numFrames - 1;

      function updateUI() {{
        const paramVal = VIZ_DATA.animation.parameter_values[currentFrame];
        paramDisplay.textContent = `${{PARAM_NAME}} = ${{paramVal.toFixed(2)}} V`;
        frameCounter.textContent = `${{currentFrame + 1}} / ${{numFrames}}`;
        frameSlider.value = currentFrame;

        if (PARAM_NAME === 'Vgs') {{
          if (paramVal < VTH) {{
            modeDisplay.textContent = 'CUTOFF';
            modeDisplay.style.background = '#dc3545';
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
          modeDisplay.textContent = paramVal < 1.0 ? 'LINEAR' : 'SATURATION';
          modeDisplay.style.background = paramVal < 1.0 ? '#17a2b8' : '#6f42c1';
          modeDisplay.style.color = '#fff';
        }}
      }}

      playBtn.addEventListener('click', () => {{
        isPlaying = !isPlaying;
        playBtn.textContent = isPlaying ? 'Pause' : 'Play';
        playBtn.style.background = isPlaying ? '#dc3545' : '#4CAF50';
      }});

      frameSlider.addEventListener('input', () => {{
        currentFrame = parseInt(frameSlider.value);
        needsRender = true;
        updateUI();
      }});

      // Render function
      function render() {{
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.clearColor(0.08, 0.08, 0.12, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.useProgram(program);
        gl.bindVertexArray(vao);

        const viewMatrix = getViewMatrix();
        const projMatrix = getProjectionMatrix();
        const viewProj = multiplyMatrices(viewMatrix, projMatrix);
        const invViewProj = invertMatrix(viewProj);

        gl.uniform3fv(uniforms.uBoundsMin, bounds.min);
        gl.uniform3fv(uniforms.uBoundsMax, bounds.max);
        gl.uniformMatrix4fv(uniforms.uInvViewProj, false, invViewProj);
        gl.uniform1f(uniforms.uOpacityScale, 1.5);
        gl.uniform1i(uniforms.uMaxSteps, 128);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_3D, frameTextures[currentFrame]);
        gl.uniform1i(uniforms.uVolume, 0);

        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, colormapTexture);
        gl.uniform1i(uniforms.uColormap, 1);

        gl.activeTexture(gl.TEXTURE2);
        gl.bindTexture(gl.TEXTURE_3D, regionTexture);
        gl.uniform1i(uniforms.uRegions, 2);

        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      }}

      // Animation loop
      let lastTime = 0;
      function animate(time) {{
        if (isPlaying) {{
          if (time - lastTime > 80) {{
            currentFrame = (currentFrame + 1) % numFrames;
            needsRender = true;
            updateUI();
            lastTime = time;
          }}
        }}

        if (needsRender) {{
          render();
          needsRender = false;
        }}

        requestAnimationFrame(animate);
      }}

      // Start
      updateUI();
      requestAnimationFrame(animate);
    }})();
    </script>
    ''')

    webgl_artifact
    return (webgl_artifact,)


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
