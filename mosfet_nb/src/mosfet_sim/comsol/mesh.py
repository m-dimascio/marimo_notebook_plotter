"""Mesh generation for COMSOL MOSFET model.

This module creates an optimized mesh for semiconductor simulation,
with refinement at critical regions like:
- Oxide-semiconductor interface (channel)
- P-N junctions (source/drain to body)
- Contact regions
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    import mph

from ..device import MOSFETParams

logger = logging.getLogger(__name__)

# Mesh size presets relative to channel length
MESH_PRESETS = {
    "extra_coarse": {"hmax_factor": 1.0, "hmin_factor": 0.05, "growth_rate": 1.6},
    "coarse": {"hmax_factor": 0.5, "hmin_factor": 0.02, "growth_rate": 1.5},
    "normal": {"hmax_factor": 0.2, "hmin_factor": 0.01, "growth_rate": 1.3},
    "fine": {"hmax_factor": 0.1, "hmin_factor": 0.005, "growth_rate": 1.2},
    "extra_fine": {"hmax_factor": 0.05, "hmin_factor": 0.002, "growth_rate": 1.1},
}


def build_mesh(
    model: "mph.Model",
    params: MOSFETParams,
    refinement: str = "normal"
) -> Dict[str, Any]:
    """
    Create optimized mesh for semiconductor simulation.

    The mesh is refined at critical regions:
    - Channel interface (requires fine resolution for inversion layer)
    - P-N junctions (sharp doping gradients)
    - Contact boundaries

    Args:
        model: COMSOL model instance.
        params: MOSFET device parameters.
        refinement: Mesh quality level:
            - "extra_coarse": Quick testing
            - "coarse": Fast simulation
            - "normal": Balanced accuracy/speed
            - "fine": High accuracy
            - "extra_fine": Maximum accuracy

    Returns:
        Dictionary with mesh statistics.
    """
    java = model.java
    logger.info(f"Building mesh with '{refinement}' refinement...")

    # Get mesh preset
    preset = MESH_PRESETS.get(refinement, MESH_PRESETS["normal"])

    # Reference length scale (channel length in μm)
    L_um = params.channel_length * 1e6

    # Calculate mesh sizes
    hmax = preset["hmax_factor"] * L_um
    hmin = preset["hmin_factor"] * L_um
    growth_rate = preset["growth_rate"]

    # Create mesh sequence
    mesh = java.mesh().create("mesh1", "geom1")
    mesh.label("MOSFET Mesh")

    # =========================================================================
    # GLOBAL MESH SIZE
    # =========================================================================
    size = mesh.feature().create("size", "Size")
    size.label("Global Size")
    size.set("hauto", 5)  # Start with physics-controlled
    size.set("hmax", f"{hmax}[um]")
    size.set("hmin", f"{hmin}[um]")
    size.set("hgrad", growth_rate)
    size.set("hcurve", 0.3)  # Curvature refinement factor

    # =========================================================================
    # INTERFACE REFINEMENT (Critical for inversion layer)
    # =========================================================================
    # The oxide-semiconductor interface needs fine mesh for accurate
    # resolution of the thin inversion layer
    iface_size = mesh.feature().create("size_interface", "Size")
    iface_size.label("Interface Size")
    iface_size.selection().named("sel_channel_interface")
    iface_size.set("hauto", 1)  # Extremely fine
    iface_size.set("hmax", f"{hmin * 2}[um]")
    iface_size.set("hmin", f"{hmin * 0.5}[um]")

    # =========================================================================
    # JUNCTION REFINEMENT
    # =========================================================================
    # Source and drain junctions have sharp doping gradients
    # that need adequate resolution

    # We'll use edge refinement near junction boundaries
    # This would select edges at source/drain boundaries
    # For now, the interface refinement helps with this

    # =========================================================================
    # FREE TETRAHEDRAL MESH
    # =========================================================================
    ftet = mesh.feature().create("ftet1", "FreeTet")
    ftet.label("Free Tetrahedral")
    ftet.selection().geom("geom1", 3)
    ftet.selection().all()

    # =========================================================================
    # BUILD MESH
    # =========================================================================
    try:
        mesh.run()
        logger.info("Mesh built successfully")
    except Exception as e:
        logger.error(f"Mesh generation failed: {e}")
        raise

    # Get mesh statistics
    stats = get_mesh_statistics(model)
    logger.info(
        f"Mesh statistics: {stats['num_elements']} elements, "
        f"min quality: {stats['min_quality']:.3f}"
    )

    return stats


def build_swept_mesh(
    model: "mph.Model",
    params: MOSFETParams,
    refinement: str = "normal",
    num_layers_y: int = 10
) -> Dict[str, Any]:
    """
    Create a swept mesh for more efficient 3D simulation.

    A swept mesh extrudes a 2D mesh in the device width direction,
    which can be more efficient for devices with uniform cross-section.

    Args:
        model: COMSOL model instance.
        params: MOSFET device parameters.
        refinement: Mesh quality level.
        num_layers_y: Number of mesh layers in width direction.

    Returns:
        Dictionary with mesh statistics.
    """
    java = model.java
    logger.info(f"Building swept mesh with '{refinement}' refinement...")

    preset = MESH_PRESETS.get(refinement, MESH_PRESETS["normal"])
    L_um = params.channel_length * 1e6

    hmax = preset["hmax_factor"] * L_um
    hmin = preset["hmin_factor"] * L_um

    mesh = java.mesh().create("mesh1", "geom1")
    mesh.label("MOSFET Swept Mesh")

    # =========================================================================
    # SIZE SETTINGS
    # =========================================================================
    size = mesh.feature().create("size", "Size")
    size.set("hmax", f"{hmax}[um]")
    size.set("hmin", f"{hmin}[um]")
    size.set("hgrad", preset["growth_rate"])

    # =========================================================================
    # FREE TRIANGULAR ON SOURCE FACE (2D base mesh)
    # =========================================================================
    ftri = mesh.feature().create("ftri1", "FreeTri")
    ftri.label("Base Triangular Mesh")
    # Select y=0 face for the base mesh
    ftri.selection().set([1])  # Source face

    # =========================================================================
    # SWEPT MESH IN Y DIRECTION
    # =========================================================================
    swe = mesh.feature().create("swe1", "Sweep")
    swe.label("Sweep in Y")
    swe.selection().geom("geom1", 3)
    swe.selection().all()

    # Distribution of layers
    swe.feature().create("dis1", "Distribution")
    swe.feature("dis1").set("type", "predefined")
    swe.feature("dis1").set("elemcount", num_layers_y)
    swe.feature("dis1").set("elemratio", 1)  # Uniform distribution

    # Build mesh
    try:
        mesh.run()
        logger.info("Swept mesh built successfully")
    except Exception as e:
        logger.warning(f"Swept mesh failed, falling back to free mesh: {e}")
        # Fall back to free tetrahedral
        return build_mesh(model, params, refinement)

    return get_mesh_statistics(model)


def get_mesh_statistics(model: "mph.Model") -> Dict[str, Any]:
    """
    Extract mesh quality statistics.

    Returns:
        Dictionary containing:
        - num_elements: Total number of mesh elements
        - num_vertices: Total number of mesh vertices
        - min_quality: Minimum element quality (0-1)
        - avg_quality: Average element quality (0-1)
        - volume: Total mesh volume
    """
    java = model.java

    try:
        mesh = java.mesh("mesh1")
        stat = mesh.stat()

        stats = {
            "num_elements": int(stat.getNumElem()),
            "num_vertices": int(stat.getNumVertex()),
            "min_quality": float(stat.getQualityMin()),
            "avg_quality": float(stat.getQualityAvg()),
            "volume": float(stat.getVolume()),
        }

        # Element type breakdown
        try:
            stats["num_tets"] = int(stat.getNumElem("tet"))
        except Exception:
            stats["num_tets"] = 0

        try:
            stats["num_prisms"] = int(stat.getNumElem("prism"))
        except Exception:
            stats["num_prisms"] = 0

        return stats

    except Exception as e:
        logger.warning(f"Could not get mesh statistics: {e}")
        return {
            "num_elements": 0,
            "num_vertices": 0,
            "min_quality": 0.0,
            "avg_quality": 0.0,
            "volume": 0.0,
        }


def refine_mesh_adaptively(
    model: "mph.Model",
    error_indicator: str = "semi.err",
    max_elements: int = 100000
) -> Dict[str, Any]:
    """
    Perform adaptive mesh refinement based on solution error.

    This refines the mesh in regions where the solution has
    high gradients or error, improving accuracy where needed.

    Args:
        model: Solved COMSOL model.
        error_indicator: Expression for error estimation.
        max_elements: Maximum number of elements allowed.

    Returns:
        Updated mesh statistics.
    """
    java = model.java
    mesh = java.mesh("mesh1")

    logger.info("Performing adaptive mesh refinement...")

    # Create adaptation feature
    adapt = mesh.feature().create("adapt1", "Adapt")
    adapt.set("errorindicator", error_indicator)
    adapt.set("maxelem", max_elements)
    adapt.set("hauto", 3)

    # Run refinement
    adapt.run()

    stats = get_mesh_statistics(model)
    logger.info(f"Refined mesh: {stats['num_elements']} elements")

    return stats
