"""Extract simulation results from COMSOL for visualization.

This module provides functions to extract:
- Carrier concentration fields (electrons, holes)
- Electric potential
- Drain current
- Other derived quantities

Data is extracted to NumPy arrays for use with existing visualization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    import mph

logger = logging.getLogger(__name__)


@dataclass
class COMSOLConcentration:
    """
    Carrier concentration data extracted from COMSOL.

    This class holds the simulation results in a format compatible
    with the existing visualization functions.

    Attributes:
        electrons: 3D array of electron concentration [cm^-3].
        holes: 3D array of hole concentration [cm^-3].
        potential: 3D array of electric potential [V].
        x: 1D array of x coordinates [m].
        y: 1D array of y coordinates [m].
        z: 1D array of z coordinates [m].
        vgs: Gate-source voltage for this solution [V].
        vds: Drain-source voltage for this solution [V].
    """
    electrons: np.ndarray
    holes: np.ndarray
    potential: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    vgs: float = 0.0
    vds: float = 0.0


def extract_concentration_field(
    model: "mph.Model",
    nx: int = 60,
    ny: int = 20,
    nz: int = 30,
    solution_index: int = 1
) -> COMSOLConcentration:
    """
    Extract electron/hole concentrations on a structured grid.

    Creates a regular 3D grid over the device and evaluates
    the carrier concentrations at each point.

    Args:
        model: Solved COMSOL model.
        nx: Number of points in x direction.
        ny: Number of points in y direction.
        nz: Number of points in z direction.
        solution_index: Solution index for parametric studies (1-based).

    Returns:
        COMSOLConcentration with carrier density fields.
    """
    java = model.java
    logger.debug(f"Extracting concentration field (solution {solution_index})...")

    # Get geometry bounds
    bounds = _get_geometry_bounds(model)

    # Create evaluation grid
    x = np.linspace(bounds["xmin"], bounds["xmax"], nx)
    y = np.linspace(bounds["ymin"], bounds["ymax"], ny)
    z = np.linspace(bounds["zmin"], bounds["zmax"], nz)

    # Create meshgrid for evaluation points
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    coords = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # Extract fields
    electrons = _evaluate_expression(
        model, "semi.n", coords, solution_index
    ).reshape(nx, ny, nz)

    holes = _evaluate_expression(
        model, "semi.p", coords, solution_index
    ).reshape(nx, ny, nz)

    potential = _evaluate_expression(
        model, "V", coords, solution_index
    ).reshape(nx, ny, nz)

    # Get voltage parameters for this solution
    vgs, vds = _get_solution_voltages(model, solution_index)

    # Convert coordinates from μm to m (if geometry was built in μm)
    x_m = x * 1e-6
    y_m = y * 1e-6
    z_m = z * 1e-6

    return COMSOLConcentration(
        electrons=electrons,
        holes=holes,
        potential=potential,
        x=x_m,
        y=y_m,
        z=z_m,
        vgs=vgs,
        vds=vds,
    )


def _get_geometry_bounds(model: "mph.Model") -> dict:
    """Get the bounding box of the geometry in mesh units."""
    java = model.java

    try:
        geom = java.geom("geom1")
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
        # Return default bounds (1 μm device)
        return {
            "xmin": 0, "xmax": 1.8,
            "ymin": 0, "ymax": 10,
            "zmin": -1, "zmax": 0.02,
        }


def _evaluate_expression(
    model: "mph.Model",
    expr: str,
    coords: np.ndarray,
    solution_index: int = 1
) -> np.ndarray:
    """
    Evaluate a COMSOL expression at given coordinates.

    Args:
        model: COMSOL model instance.
        expr: Expression to evaluate (e.g., "semi.n").
        coords: Nx3 array of (x, y, z) coordinates.
        solution_index: Solution index for parametric studies.

    Returns:
        1D array of evaluated values.
    """
    java = model.java
    num_points = coords.shape[0]

    try:
        # Create temporary interpolation
        result = java.result()

        # Create interpolation evaluation
        interp = result.numerical().create("tmp_interp", "Interp")
        interp.set("expr", [expr])
        interp.set("coord", coords.T.tolist())
        interp.set("solnum", solution_index)

        # Get data
        data = interp.getData()
        values = np.array(data).flatten()

        # Clean up
        result.numerical().remove("tmp_interp")

        # Handle NaN values (points outside domain)
        values = np.nan_to_num(values, nan=0.0)

        return values

    except Exception as e:
        logger.warning(f"Could not evaluate '{expr}': {e}")
        return np.zeros(num_points)


def _get_solution_voltages(
    model: "mph.Model",
    solution_index: int
) -> Tuple[float, float]:
    """Get Vgs and Vds for a specific solution index."""
    java = model.java

    try:
        # For parametric studies, the parameter values are stored
        # This is a simplified approach - actual implementation may vary
        vgs = float(java.param().get("Vgs").replace("[V]", ""))
        vds = float(java.param().get("Vds").replace("[V]", ""))
        return vgs, vds
    except Exception:
        return 0.0, 0.0


def extract_drain_current(
    model: "mph.Model",
    solution_indices: Optional[List[int]] = None
) -> np.ndarray:
    """
    Extract drain current from simulation.

    Integrates the current density at the drain contact
    to get the total drain current.

    Args:
        model: Solved COMSOL model.
        solution_indices: List of solution indices to extract.
                         If None, extracts all solutions.

    Returns:
        Array of drain currents [A].
    """
    java = model.java
    logger.debug("Extracting drain current...")

    try:
        result = java.result()

        # Create surface integration for current
        integ = result.numerical().create("id_eval", "IntSurface")
        integ.selection().named("sel_drain_contact")

        # Total current density (electrons + holes)
        integ.set("expr", ["semi.Jnx + semi.Jpx"])

        if solution_indices is None:
            # Get number of solutions
            sol = java.sol("sol1")
            try:
                n_solutions = int(sol.feature("s1").getSolnum())
                solution_indices = list(range(1, n_solutions + 1))
            except Exception:
                solution_indices = [1]

        currents = []
        for sol_idx in solution_indices:
            try:
                integ.set("solnum", sol_idx)
                data = integ.getData()
                Id = float(np.array(data).sum())
                currents.append(Id)
            except Exception as e:
                logger.warning(f"Could not extract current for solution {sol_idx}: {e}")
                currents.append(0.0)

        # Clean up
        result.numerical().remove("id_eval")

        return np.array(currents)

    except Exception as e:
        logger.error(f"Failed to extract drain current: {e}")
        return np.array([0.0])


def extract_sweep_concentrations(
    model: "mph.Model",
    n_frames: int,
    nx: int = 60,
    ny: int = 20,
    nz: int = 30
) -> List[COMSOLConcentration]:
    """
    Extract concentration fields for all parametric sweep points.

    Args:
        model: Solved COMSOL model with parametric sweep.
        n_frames: Number of sweep points.
        nx, ny, nz: Grid resolution.

    Returns:
        List of COMSOLConcentration objects for animation.
    """
    logger.info(f"Extracting {n_frames} concentration frames...")

    concentrations = []

    for i in range(1, n_frames + 1):
        logger.debug(f"Extracting frame {i}/{n_frames}")
        conc = extract_concentration_field(
            model, nx=nx, ny=ny, nz=nz, solution_index=i
        )
        concentrations.append(conc)

    logger.info(f"Extracted {len(concentrations)} concentration frames")
    return concentrations


def extract_channel_profile(
    model: "mph.Model",
    n_points: int = 100,
    z_depth: float = -0.01,  # μm, just below surface
    solution_index: int = 1
) -> dict:
    """
    Extract carrier concentration along the channel.

    Useful for visualizing channel formation and pinch-off.

    Args:
        model: Solved COMSOL model.
        n_points: Number of points along channel.
        z_depth: Depth below surface [μm].
        solution_index: Solution index.

    Returns:
        Dictionary with x coordinates and electron concentration.
    """
    java = model.java
    bounds = _get_geometry_bounds(model)

    # Points along channel at specified depth
    x = np.linspace(bounds["xmin"], bounds["xmax"], n_points)
    y_mid = (bounds["ymin"] + bounds["ymax"]) / 2
    y = np.full(n_points, y_mid)
    z = np.full(n_points, z_depth)

    coords = np.column_stack([x, y, z])

    electrons = _evaluate_expression(model, "semi.n", coords, solution_index)

    return {
        "x": x * 1e-6,  # Convert to m
        "electrons": electrons,
    }


def extract_vertical_profile(
    model: "mph.Model",
    x_position: float,
    n_points: int = 50,
    solution_index: int = 1
) -> dict:
    """
    Extract carrier concentration vs depth (vertical cut).

    Useful for visualizing inversion layer depth.

    Args:
        model: Solved COMSOL model.
        x_position: X position for the cut [μm].
        n_points: Number of points in z direction.
        solution_index: Solution index.

    Returns:
        Dictionary with z coordinates and electron concentration.
    """
    java = model.java
    bounds = _get_geometry_bounds(model)

    y_mid = (bounds["ymin"] + bounds["ymax"]) / 2
    x = np.full(n_points, x_position)
    y = np.full(n_points, y_mid)
    z = np.linspace(bounds["zmin"], bounds["zmax"], n_points)

    coords = np.column_stack([x, y, z])

    electrons = _evaluate_expression(model, "semi.n", coords, solution_index)
    potential = _evaluate_expression(model, "V", coords, solution_index)

    return {
        "z": z * 1e-6,  # Convert to m
        "electrons": electrons,
        "potential": potential,
    }
