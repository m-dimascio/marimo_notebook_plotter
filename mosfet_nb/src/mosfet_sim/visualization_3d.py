"""3D visualization with animation using Plotly."""

import numpy as np
import plotly.graph_objects as go
from .mesh import DeviceMesh, Region
from .concentration import CarrierConcentration
from .device import MOSFETParams
from .physics import threshold_voltage


# Color scheme for device regions
REGION_COLORS = {
    Region.BODY: 'rgb(255, 255, 200)',      # Light yellow (p-substrate)
    Region.SOURCE: 'rgb(100, 149, 237)',     # Cornflower blue (n+)
    Region.DRAIN: 'rgb(100, 149, 237)',      # Cornflower blue (n+)
    Region.CHANNEL: 'rgb(255, 255, 200)',    # Same as body initially
    Region.OXIDE: 'rgb(200, 200, 200)',      # Gray
    Region.GATE: 'rgb(255, 165, 0)',         # Orange
}


def create_device_structure_traces(mesh: DeviceMesh) -> list[go.Mesh3d]:
    """
    Create static 3D traces for device structure.

    These traces don't change during animation - only the concentration overlay does.

    Args:
        mesh: Device mesh

    Returns:
        List of Plotly Mesh3d traces for each region
    """
    traces = []

    # Create box meshes for each region
    regions_to_render = [Region.SOURCE, Region.DRAIN, Region.OXIDE, Region.GATE]

    for region in regions_to_render:
        mask = mesh.regions == region
        if not np.any(mask):
            continue

        x_vals = mesh.X[mask]
        y_vals = mesh.Y[mask]
        z_vals = mesh.Z[mask]

        if len(x_vals) == 0:
            continue

        # Create a box for this region
        x_min, x_max = x_vals.min(), x_vals.max()
        y_min, y_max = y_vals.min(), y_vals.max()
        z_min, z_max = z_vals.min(), z_vals.max()

        # 8 vertices of a box
        vertices_x = [x_min, x_max, x_max, x_min, x_min, x_max, x_max, x_min]
        vertices_y = [y_min, y_min, y_max, y_max, y_min, y_min, y_max, y_max]
        vertices_z = [z_min, z_min, z_min, z_min, z_max, z_max, z_max, z_max]

        # 12 triangular faces (2 per box face)
        i_faces = [0, 0, 4, 4, 0, 0, 1, 1, 0, 0, 3, 3]
        j_faces = [1, 2, 5, 6, 1, 4, 2, 5, 3, 4, 2, 6]
        k_faces = [2, 3, 6, 7, 4, 5, 5, 6, 4, 7, 6, 7]

        trace = go.Mesh3d(
            x=vertices_x,
            y=vertices_y,
            z=vertices_z,
            i=i_faces,
            j=j_faces,
            k=k_faces,
            color=REGION_COLORS[region],
            opacity=0.6 if region in [Region.OXIDE, Region.GATE] else 0.8,
            name=region.name,
            showlegend=True,
            flatshading=True,
        )
        traces.append(trace)

    return traces


def create_substrate_trace(mesh: DeviceMesh) -> go.Mesh3d:
    """Create transparent substrate box."""
    x_min, x_max = mesh.x.min(), mesh.x.max()
    y_min, y_max = mesh.y.min(), mesh.y.max()
    z_min = mesh.z.min()
    z_max = 0  # Surface

    vertices_x = [x_min, x_max, x_max, x_min, x_min, x_max, x_max, x_min]
    vertices_y = [y_min, y_min, y_max, y_max, y_min, y_min, y_max, y_max]
    vertices_z = [z_min, z_min, z_min, z_min, z_max, z_max, z_max, z_max]

    i_faces = [0, 0, 4, 4, 0, 0, 1, 1, 0, 0, 3, 3]
    j_faces = [1, 2, 5, 6, 1, 4, 2, 5, 3, 4, 2, 6]
    k_faces = [2, 3, 6, 7, 4, 5, 5, 6, 4, 7, 6, 7]

    return go.Mesh3d(
        x=vertices_x,
        y=vertices_y,
        z=vertices_z,
        i=i_faces,
        j=j_faces,
        k=k_faces,
        color=REGION_COLORS[Region.BODY],
        opacity=0.3,
        name='Substrate',
        showlegend=True,
    )


def create_concentration_isosurface_trace(
    conc: CarrierConcentration,
    iso_values: list[float] = None,
) -> go.Isosurface:
    """
    Create isosurface trace for electron concentration.

    Shows surfaces of constant electron density - effective for
    visualizing channel formation.

    Args:
        conc: Carrier concentration data
        iso_values: List of concentration values for isosurfaces

    Returns:
        Plotly Isosurface trace
    """
    mesh = conc.mesh
    electrons = conc.electrons.copy()

    # Log scale
    electrons = np.maximum(electrons, 1e1)
    values = np.log10(electrons)

    if iso_values is None:
        iso_values = [15, 17, 19]

    return go.Isosurface(
        x=mesh.X.flatten(),
        y=mesh.Y.flatten(),
        z=mesh.Z.flatten(),
        value=values.flatten(),
        isomin=min(iso_values),
        isomax=max(iso_values),
        surface_count=len(iso_values),
        colorscale='Plasma',
        caps=dict(x_show=False, y_show=False, z_show=False),
        opacity=0.6,
        colorbar=dict(title='log10(n)', x=1.02),
        name='Electron Density',
    )


def create_animated_figure(
    mesh: DeviceMesh,
    concentrations: list[CarrierConcentration],
    parameter_values: np.ndarray,
    parameter_name: str = 'Vgs',
    parameter_unit: str = 'V',
) -> go.Figure:
    """
    Create complete animated 3D figure with play controls.

    This is the main function for creating the COMSOL-like visualization.
    Pre-computes all frames for smooth playback.

    Args:
        mesh: Device mesh
        concentrations: List of CarrierConcentration for each frame
        parameter_values: Array of parameter values (one per frame)
        parameter_name: Name of swept parameter for labels
        parameter_unit: Unit string for labels

    Returns:
        Plotly Figure with animation
    """
    vth = threshold_voltage(mesh.params)

    # Create static device structure traces
    structure_traces = create_device_structure_traces(mesh)
    substrate_trace = create_substrate_trace(mesh)

    # Create initial concentration trace
    initial_conc_trace = create_concentration_isosurface_trace(concentrations[0])

    # Combine all initial traces
    all_traces = [substrate_trace] + structure_traces + [initial_conc_trace]

    # Create figure with initial state
    fig = go.Figure(data=all_traces)

    # Create animation frames
    frames = []
    for i, (conc, param_val) in enumerate(zip(concentrations, parameter_values)):
        # Only update concentration trace (last trace)
        conc_trace = create_concentration_isosurface_trace(conc)

        # Determine operating region for annotation
        if parameter_name == 'Vgs':
            if param_val < vth:
                region = "CUTOFF"
            else:
                region = "ACTIVE"
        else:
            region = ""

        frame = go.Frame(
            data=[conc_trace],
            traces=[len(all_traces) - 1],  # Update only last trace
            name=f'{param_val:.2f}',
            layout=go.Layout(
                title=f'MOSFET 3D Electron Concentration | {parameter_name}={param_val:.2f}{parameter_unit} | Vth={vth:.2f}V | {region}'
            )
        )
        frames.append(frame)

    fig.frames = frames

    # Add slider and play/pause buttons
    fig.update_layout(
        title=f'MOSFET 3D Electron Concentration | {parameter_name}={parameter_values[0]:.2f}{parameter_unit}',
        scene=dict(
            xaxis_title='X (um) - Channel Length',
            yaxis_title='Y (um) - Width',
            zaxis_title='Z (um) - Depth',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.0)
            ),
        ),
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                y=0,
                x=0.1,
                xanchor='right',
                yanchor='top',
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[
                            None,
                            dict(
                                frame=dict(duration=100, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=50),
                                mode='immediate',
                            )
                        ]
                    ),
                    dict(
                        label='Pause',
                        method='animate',
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode='immediate',
                                transition=dict(duration=0),
                            )
                        ]
                    ),
                ]
            )
        ],
        sliders=[
            dict(
                active=0,
                yanchor='top',
                xanchor='left',
                currentvalue=dict(
                    font=dict(size=16),
                    prefix=f'{parameter_name}: ',
                    suffix=f' {parameter_unit}',
                    visible=True,
                    xanchor='right',
                ),
                transition=dict(duration=50),
                pad=dict(b=10, t=50),
                len=0.9,
                x=0.1,
                y=0,
                steps=[
                    dict(
                        args=[
                            [f'{val:.2f}'],
                            dict(
                                frame=dict(duration=100, redraw=True),
                                mode='immediate',
                                transition=dict(duration=50),
                            )
                        ],
                        label=f'{val:.2f}',
                        method='animate',
                    )
                    for val in parameter_values
                ],
            )
        ],
        legend=dict(
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=0.01,
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    return fig


def create_vgs_sweep_animation(
    params: MOSFETParams,
    vgs_min: float = 0.0,
    vgs_max: float = 3.0,
    vds: float = 0.5,
    n_frames: int = 30,
    mesh_resolution: tuple[int, int, int] = (40, 15, 25),
) -> go.Figure:
    """
    Create animation sweeping gate voltage through all operating modes.

    Convenience function that handles mesh creation, concentration
    computation, and figure generation.

    Args:
        params: MOSFET device parameters
        vgs_min: Starting gate voltage
        vgs_max: Ending gate voltage
        vds: Fixed drain voltage
        n_frames: Number of animation frames
        mesh_resolution: (nx, ny, nz) mesh points

    Returns:
        Animated Plotly figure
    """
    from .mesh import create_device_mesh
    from .concentration import generate_concentration_sweep

    # Create mesh
    mesh = create_device_mesh(params, *mesh_resolution)

    # Generate Vgs values
    vgs_values = np.linspace(vgs_min, vgs_max, n_frames)

    # Pre-compute all concentration frames
    concentrations = generate_concentration_sweep(mesh, vgs_values, vds)

    # Create animated figure
    fig = create_animated_figure(
        mesh, concentrations, vgs_values,
        parameter_name='Vgs', parameter_unit='V'
    )

    return fig


def create_vds_sweep_animation(
    params: MOSFETParams,
    vgs: float = 2.0,
    vds_min: float = 0.0,
    vds_max: float = 3.0,
    n_frames: int = 30,
    mesh_resolution: tuple[int, int, int] = (40, 15, 25),
) -> go.Figure:
    """
    Create animation sweeping drain voltage (linear to saturation).

    Shows channel pinch-off as Vds increases.

    Args:
        params: MOSFET device parameters
        vgs: Fixed gate voltage (should be > Vth)
        vds_min: Starting drain voltage
        vds_max: Ending drain voltage
        n_frames: Number of animation frames
        mesh_resolution: (nx, ny, nz) mesh points

    Returns:
        Animated Plotly figure
    """
    from .mesh import create_device_mesh
    from .concentration import generate_output_sweep

    # Create mesh
    mesh = create_device_mesh(params, *mesh_resolution)

    # Generate Vds values
    vds_values = np.linspace(vds_min, vds_max, n_frames)

    # Pre-compute all concentration frames
    concentrations = generate_output_sweep(mesh, vgs, vds_values)

    # Create animated figure
    fig = create_animated_figure(
        mesh, concentrations, vds_values,
        parameter_name='Vds', parameter_unit='V'
    )

    return fig


def create_cross_section_slice(
    conc: CarrierConcentration,
    slice_y: float = None,
) -> go.Figure:
    """
    Create 2D heatmap slice through device at constant Y.

    Provides X-Z cross-section view similar to COMSOL 2D results.

    Args:
        conc: Carrier concentration data
        slice_y: Y-coordinate for slice (default: center)

    Returns:
        Plotly Figure with heatmap
    """
    mesh = conc.mesh

    if slice_y is None:
        slice_y = mesh.y[len(mesh.y) // 2]

    # Find nearest y index
    y_idx = np.argmin(np.abs(mesh.y - slice_y))

    # Extract slice
    electrons_slice = conc.electrons[:, y_idx, :]

    # Log scale
    electrons_slice = np.maximum(electrons_slice, 1e1)
    values = np.log10(electrons_slice)

    fig = go.Figure(data=go.Heatmap(
        x=mesh.x,
        y=mesh.z,
        z=values.T,
        colorscale='Viridis',
        colorbar=dict(title='log10(n) [cm^-3]'),
    ))

    fig.update_layout(
        title='MOSFET Cross-Section - Electron Concentration',
        xaxis_title='X (um) - Channel Direction',
        yaxis_title='Z (um) - Depth',
        yaxis=dict(scaleanchor='x', scaleratio=1),
    )

    return fig
