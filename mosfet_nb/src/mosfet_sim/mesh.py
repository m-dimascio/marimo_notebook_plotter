"""3D mesh generation for MOSFET structure."""

import numpy as np
from dataclasses import dataclass
from enum import IntEnum
from .device import MOSFETParams


class Region(IntEnum):
    """Device region identifiers."""
    BODY = 0
    SOURCE = 1
    DRAIN = 2
    CHANNEL = 3
    OXIDE = 4
    GATE = 5


@dataclass
class DeviceMesh:
    """3D mesh representation of MOSFET structure.

    Attributes:
        x: 1D array of x-coordinates (along channel length)
        y: 1D array of y-coordinates (device width)
        z: 1D array of z-coordinates (depth into substrate)
        X, Y, Z: 3D meshgrid arrays
        regions: 3D array of Region enum values for each point
        params: Original device parameters
    """
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    regions: np.ndarray
    params: MOSFETParams


def create_device_mesh(
    params: MOSFETParams,
    nx: int = 60,
    ny: int = 20,
    nz: int = 30,
) -> DeviceMesh:
    """
    Create 3D mesh for MOSFET device.

    Coordinate system:
        x: 0 to L_total (source-to-drain direction)
        y: 0 to W (channel width direction)
        z: 0 (top surface) to -depth (into substrate)

    Args:
        params: MOSFET device parameters
        nx: Number of points along channel length
        ny: Number of points along width
        nz: Number of points in depth

    Returns:
        DeviceMesh object with coordinates and region tags
    """
    # Device dimensions (convert to micrometers for visualization)
    L = params.channel_length * 1e6  # Channel length in um
    W = params.channel_width * 1e6   # Width in um
    t_ox = params.oxide_thickness * 1e6  # Oxide thickness in um

    # Define total device extent
    sd_length = L * 0.4  # Source/drain region length
    L_total = L + 2 * sd_length
    depth = L * 1.0  # Substrate depth

    # Create 1D coordinate arrays
    x = np.linspace(0, L_total, nx)
    y = np.linspace(0, W, ny)
    z = np.linspace(t_ox, -depth, nz)  # Positive z = oxide, negative = substrate

    # Create 3D meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Initialize region array
    regions = np.full(X.shape, Region.BODY, dtype=np.int32)

    # Define region boundaries
    source_end = sd_length
    drain_start = L_total - sd_length
    junction_depth = depth * 0.15  # Source/drain junction depth

    # Tag regions based on coordinates
    # Source region (n+)
    source_mask = (X <= source_end) & (Z >= -junction_depth) & (Z <= 0)
    regions[source_mask] = Region.SOURCE

    # Drain region (n+)
    drain_mask = (X >= drain_start) & (Z >= -junction_depth) & (Z <= 0)
    regions[drain_mask] = Region.DRAIN

    # Channel region (between source and drain, at surface)
    channel_depth = depth * 0.05  # Thin channel region
    channel_mask = (
        (X > source_end) & (X < drain_start) &
        (Z >= -channel_depth) & (Z <= 0)
    )
    regions[channel_mask] = Region.CHANNEL

    # Oxide region (above surface)
    oxide_mask = Z > 0
    regions[oxide_mask] = Region.OXIDE

    # Gate region (top of oxide, over channel)
    gate_mask = (
        oxide_mask &
        (X > source_end) & (X < drain_start)
    )
    regions[gate_mask] = Region.GATE

    return DeviceMesh(
        x=x, y=y, z=z,
        X=X, Y=Y, Z=Z,
        regions=regions,
        params=params,
    )


def get_region_surfaces(mesh: DeviceMesh) -> dict[Region, dict]:
    """
    Extract surface vertices for each region (for 3D rendering).

    Args:
        mesh: DeviceMesh object

    Returns:
        Dictionary mapping Region to bounding box info
    """
    surfaces = {}

    for region in Region:
        mask = mesh.regions == region
        if not np.any(mask):
            continue

        indices = np.where(mask)
        if len(indices[0]) == 0:
            continue

        x_min, x_max = mesh.x[indices[0].min()], mesh.x[indices[0].max()]
        y_min, y_max = mesh.y[indices[1].min()], mesh.y[indices[1].max()]
        z_min, z_max = mesh.z[indices[2].min()], mesh.z[indices[2].max()]

        surfaces[region] = {
            'x_range': (x_min, x_max),
            'y_range': (y_min, y_max),
            'z_range': (z_min, z_max),
        }

    return surfaces
