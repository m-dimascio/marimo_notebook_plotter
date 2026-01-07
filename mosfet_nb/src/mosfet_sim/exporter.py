"""Export MOSFET simulation data for WebGL visualization."""

import numpy as np
import json
import base64
from pathlib import Path
from .mesh import DeviceMesh
from .concentration import CarrierConcentration


def export_mesh_geometry(mesh: DeviceMesh) -> dict:
    """
    Export mesh geometry as JSON for WebGL.

    Returns dict with:
    - dimensions: [nx, ny, nz]
    - bounds: [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
    - regions: Base64-encoded uint8 array of region IDs
    """
    return {
        "dimensions": [len(mesh.x), len(mesh.y), len(mesh.z)],
        "bounds": [
            [float(mesh.x.min()), float(mesh.x.max())],
            [float(mesh.y.min()), float(mesh.y.max())],
            [float(mesh.z.min()), float(mesh.z.max())],
        ],
        "regions": base64.b64encode(
            mesh.regions.astype(np.uint8).tobytes()
        ).decode('ascii'),
    }


def export_concentration_frame(
    conc: CarrierConcentration,
    log_scale: bool = True,
    normalize: bool = True,
) -> str:
    """
    Export single concentration frame as Base64-encoded float32.

    Args:
        conc: Carrier concentration data
        log_scale: Apply log10 transform
        normalize: Normalize to [0, 1] range

    Returns:
        Base64-encoded string of float32 array
    """
    data = conc.electrons.copy()

    if log_scale:
        data = np.maximum(data, 1.0)  # Avoid log(0)
        data = np.log10(data)

    if normalize:
        data_min = data.min()
        data_max = data.max()
        data = (data - data_min) / (data_max - data_min + 1e-10)

    return base64.b64encode(
        data.astype(np.float32).tobytes()
    ).decode('ascii')


def export_animation_sequence(
    concentrations: list[CarrierConcentration],
    parameter_values: np.ndarray,
    parameter_name: str = "Vgs",
) -> dict:
    """
    Export complete animation sequence for WebGL.

    Returns dict with:
    - parameter_name: str
    - parameter_values: list[float]
    - frames: list[str] (Base64-encoded concentration data)
    """
    return {
        "parameter_name": parameter_name,
        "parameter_values": [float(v) for v in parameter_values],
        "frames": [
            export_concentration_frame(conc)
            for conc in concentrations
        ],
    }


def export_complete_visualization(
    mesh: DeviceMesh,
    concentrations: list[CarrierConcentration],
    parameter_values: np.ndarray,
    parameter_name: str = "Vgs",
    output_path: Path = None,
) -> dict:
    """
    Export everything needed for WebGL visualization.

    Combines mesh geometry, device regions, and animation frames
    into a single JSON structure.

    Args:
        mesh: Device mesh
        concentrations: List of concentration data for each frame
        parameter_values: Parameter values for each frame
        parameter_name: Name of the swept parameter
        output_path: Optional path to write JSON file

    Returns:
        Dictionary with all visualization data
    """
    data = {
        "mesh": export_mesh_geometry(mesh),
        "animation": export_animation_sequence(
            concentrations, parameter_values, parameter_name
        ),
        "colormap": {
            "name": "viridis",
            "min_label": "Low",
            "max_label": "High (log10 n)",
        },
    }

    if output_path:
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            json.dump(data, f)

    return data
