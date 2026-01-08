"""COMSOL geometry construction for MOSFET device.

This module creates the 3D device geometry in COMSOL, including:
- Silicon substrate (p-type body)
- Source and drain regions (n+ doped)
- Gate oxide (SiO2)
- Gate contact (metal)

The geometry matches the structure defined in the analytical model.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    import mph

from ..device import MOSFETParams

logger = logging.getLogger(__name__)


def build_mosfet_geometry(
    model: "mph.Model",
    params: MOSFETParams,
    scale_to_um: bool = True
) -> Dict[str, Any]:
    """
    Create 3D MOSFET geometry in COMSOL.

    Device structure (cross-section view):

    ```
                    Gate Contact
                   ┌───────────┐
                   │   Oxide   │  tox
    ───────────────┴───────────┴───────────────  z=0
    │  Source  │     Channel     │  Drain   │
    │   (n+)   │                 │   (n+)   │   junction depth
    ├──────────┼─────────────────┼──────────┤
    │          │                 │          │
    │          │   Substrate     │          │
    │          │    (p-type)     │          │
    │          │                 │          │
    └──────────┴─────────────────┴──────────┘
    x=0      source_end      drain_start   L_total
    ```

    Args:
        model: COMSOL model instance (from mph).
        params: MOSFET device parameters.
        scale_to_um: If True, converts meters to micrometers for COMSOL.
                    COMSOL works better with μm scale for semiconductors.

    Returns:
        Dictionary with geometry information including domain indices.
    """
    java = model.java

    # Create 3D geometry component
    java.modelNode().create("comp1")
    geom = java.geom().create("geom1", 3)

    # Device dimensions (in meters from params)
    L = params.channel_length
    W = params.channel_width
    tox = params.oxide_thickness

    # Derived dimensions (matching analytical model)
    sd_extension = 0.4 * L          # Source/drain horizontal extension
    L_total = L + 2 * sd_extension  # Total device length
    substrate_depth = 1.0 * L       # Depth into substrate
    junction_depth = 0.15 * substrate_depth  # S/D junction depth
    gate_thickness = tox * 0.5      # Gate metal thickness

    # Scale factor for COMSOL (μm is typical for semiconductor)
    scale = 1e6 if scale_to_um else 1.0
    unit = "[um]" if scale_to_um else "[m]"

    # Store geometry info for later reference
    geom_info = {
        "L": L,
        "W": W,
        "tox": tox,
        "L_total": L_total,
        "sd_extension": sd_extension,
        "substrate_depth": substrate_depth,
        "junction_depth": junction_depth,
        "scale": scale,
        "unit": unit.strip("[]"),
    }

    logger.info(f"Building MOSFET geometry: L={L*1e6:.2f}μm, W={W*1e6:.2f}μm, tox={tox*1e9:.1f}nm")

    # =========================================================================
    # 1. SUBSTRATE BLOCK (p-type silicon body)
    # =========================================================================
    # This is the main silicon body, extends full width and depth
    geom.feature().create("substrate", "Block")
    geom.feature("substrate").set("size", [
        f"{L_total * scale}{unit}",
        f"{W * scale}{unit}",
        f"{substrate_depth * scale}{unit}"
    ])
    geom.feature("substrate").set("pos", [
        "0",
        "0",
        f"{-substrate_depth * scale}{unit}"
    ])
    geom.feature("substrate").set("createselection", True)
    geom.feature("substrate").label("Substrate (p-type)")

    # =========================================================================
    # 2. SOURCE REGION (n+ doped)
    # =========================================================================
    # Located at left side, extends to junction depth
    geom.feature().create("source", "Block")
    geom.feature("source").set("size", [
        f"{sd_extension * scale}{unit}",
        f"{W * scale}{unit}",
        f"{junction_depth * scale}{unit}"
    ])
    geom.feature("source").set("pos", [
        "0",
        "0",
        f"{-junction_depth * scale}{unit}"
    ])
    geom.feature("source").set("createselection", True)
    geom.feature("source").label("Source (n+)")

    # =========================================================================
    # 3. DRAIN REGION (n+ doped)
    # =========================================================================
    # Located at right side, same dimensions as source
    drain_x_start = sd_extension + L  # After channel
    geom.feature().create("drain", "Block")
    geom.feature("drain").set("size", [
        f"{sd_extension * scale}{unit}",
        f"{W * scale}{unit}",
        f"{junction_depth * scale}{unit}"
    ])
    geom.feature("drain").set("pos", [
        f"{drain_x_start * scale}{unit}",
        "0",
        f"{-junction_depth * scale}{unit}"
    ])
    geom.feature("drain").set("createselection", True)
    geom.feature("drain").label("Drain (n+)")

    # =========================================================================
    # 4. GATE OXIDE (SiO2)
    # =========================================================================
    # Thin oxide layer above channel region
    geom.feature().create("oxide", "Block")
    geom.feature("oxide").set("size", [
        f"{L * scale}{unit}",
        f"{W * scale}{unit}",
        f"{tox * scale}{unit}"
    ])
    geom.feature("oxide").set("pos", [
        f"{sd_extension * scale}{unit}",
        "0",
        "0"
    ])
    geom.feature("oxide").set("createselection", True)
    geom.feature("oxide").label("Gate Oxide (SiO2)")

    # =========================================================================
    # 5. GATE CONTACT (metal)
    # =========================================================================
    # Metal gate above oxide
    geom.feature().create("gate", "Block")
    geom.feature("gate").set("size", [
        f"{L * scale}{unit}",
        f"{W * scale}{unit}",
        f"{gate_thickness * scale}{unit}"
    ])
    geom.feature("gate").set("pos", [
        f"{sd_extension * scale}{unit}",
        "0",
        f"{tox * scale}{unit}"
    ])
    geom.feature("gate").set("createselection", True)
    geom.feature("gate").label("Gate Contact")

    # =========================================================================
    # BUILD GEOMETRY
    # =========================================================================
    geom.run("fin")

    logger.info("Geometry built successfully")

    # =========================================================================
    # CREATE NAMED SELECTIONS
    # =========================================================================
    # These selections are used for physics and boundary assignments
    _create_selections(java, params, geom_info)

    return geom_info


def _create_selections(java, params: MOSFETParams, geom_info: Dict[str, Any]) -> None:
    """
    Create named selections for physics domains and boundaries.

    These selections make it easier to assign materials, physics,
    and boundary conditions to specific regions.
    """
    sel = java.selection()

    # Domain selections (for materials and physics)
    # Note: These use the geometry selections created by 'createselection'

    # Semiconductor domains (silicon regions)
    sel.create("sel_silicon", "Union")
    sel.feature("sel_silicon").label("Silicon Domains")
    sel.feature("sel_silicon").set("input", ["geom1_substrate_dom"])

    # Source domain
    sel.create("sel_source", "Explicit")
    sel.feature("sel_source").label("Source Domain")
    sel.feature("sel_source").geom("geom1", 3)
    # Will be set based on actual domain numbers after geometry build

    # Drain domain
    sel.create("sel_drain", "Explicit")
    sel.feature("sel_drain").label("Drain Domain")
    sel.feature("sel_drain").geom("geom1", 3)

    # Oxide domain
    sel.create("sel_oxide", "Explicit")
    sel.feature("sel_oxide").label("Oxide Domain")
    sel.feature("sel_oxide").geom("geom1", 3)

    # Boundary selections (for contacts)
    # Source contact - left face of source region
    sel.create("sel_source_contact", "Box")
    sel.feature("sel_source_contact").label("Source Contact")
    sel.feature("sel_source_contact").set("entitydim", 2)  # 2D boundary
    sel.feature("sel_source_contact").set("xmin", "-0.01[um]")
    sel.feature("sel_source_contact").set("xmax", "0.01[um]")
    sel.feature("sel_source_contact").set("condition", "inside")

    # Drain contact - right face of drain region
    L_total_um = geom_info["L_total"] * geom_info["scale"]
    sel.create("sel_drain_contact", "Box")
    sel.feature("sel_drain_contact").label("Drain Contact")
    sel.feature("sel_drain_contact").set("entitydim", 2)
    sel.feature("sel_drain_contact").set("xmin", f"{L_total_um - 0.01}[um]")
    sel.feature("sel_drain_contact").set("xmax", f"{L_total_um + 0.01}[um]")
    sel.feature("sel_drain_contact").set("condition", "inside")

    # Body contact - bottom face of substrate
    depth_um = geom_info["substrate_depth"] * geom_info["scale"]
    sel.create("sel_body_contact", "Box")
    sel.feature("sel_body_contact").label("Body Contact")
    sel.feature("sel_body_contact").set("entitydim", 2)
    sel.feature("sel_body_contact").set("zmin", f"{-depth_um - 0.01}[um]")
    sel.feature("sel_body_contact").set("zmax", f"{-depth_um + 0.01}[um]")
    sel.feature("sel_body_contact").set("condition", "inside")

    # Gate contact - top face of gate metal
    gate_top = (geom_info["tox"] + geom_info["tox"] * 0.5) * geom_info["scale"]
    sel.create("sel_gate_contact", "Box")
    sel.feature("sel_gate_contact").label("Gate Contact")
    sel.feature("sel_gate_contact").set("entitydim", 2)
    sel.feature("sel_gate_contact").set("zmin", f"{gate_top - 0.01}[um]")
    sel.feature("sel_gate_contact").set("zmax", f"{gate_top + 0.01}[um]")
    sel.feature("sel_gate_contact").set("condition", "inside")

    # Channel surface - oxide-semiconductor interface
    sel.create("sel_channel_interface", "Box")
    sel.feature("sel_channel_interface").label("Channel Interface")
    sel.feature("sel_channel_interface").set("entitydim", 2)
    sel.feature("sel_channel_interface").set("zmin", "-0.01[um]")
    sel.feature("sel_channel_interface").set("zmax", "0.01[um]")
    sd_ext_um = geom_info["sd_extension"] * geom_info["scale"]
    L_um = geom_info["L"] * geom_info["scale"]
    sel.feature("sel_channel_interface").set("xmin", f"{sd_ext_um}[um]")
    sel.feature("sel_channel_interface").set("xmax", f"{sd_ext_um + L_um}[um]")
    sel.feature("sel_channel_interface").set("condition", "inside")

    logger.debug("Created named selections for physics assignment")


def get_geometry_bounds(model: "mph.Model") -> Dict[str, float]:
    """
    Get the bounding box of the geometry.

    Returns:
        Dictionary with xmin, xmax, ymin, ymax, zmin, zmax.
    """
    java = model.java
    geom = java.geom("geom1")

    try:
        bbox = geom.getBoundingBox()
        return {
            "xmin": float(bbox[0]),
            "ymin": float(bbox[1]),
            "zmin": float(bbox[2]),
            "xmax": float(bbox[3]),
            "ymax": float(bbox[4]),
            "zmax": float(bbox[5]),
        }
    except Exception as e:
        logger.warning(f"Could not get geometry bounds: {e}")
        return {}
