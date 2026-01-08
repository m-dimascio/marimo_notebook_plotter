"""Study and solver configuration for COMSOL semiconductor simulation.

This module sets up the solver configuration for:
- Stationary (DC) analysis
- Parametric sweeps (Vgs, Vds variations)
- Transient analysis (optional)

The semiconductor equations are highly nonlinear, so careful
solver configuration is needed for convergence.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    import mph

logger = logging.getLogger(__name__)


def create_stationary_study(
    model: "mph.Model",
    study_name: str = "std1",
    use_continuation: bool = True
) -> None:
    """
    Create a stationary study for DC analysis.

    The study solves the steady-state semiconductor equations
    at the specified bias point.

    Args:
        model: COMSOL model instance.
        study_name: Name for the study.
        use_continuation: If True, use continuation for better convergence
                         when sweeping parameters.
    """
    java = model.java
    logger.info("Creating stationary study...")

    # =========================================================================
    # CREATE STUDY
    # =========================================================================
    study = java.study().create(study_name)
    study.label("DC Analysis")

    # Stationary study step
    stat = study.feature().create("stat", "Stationary")
    stat.label("Stationary Step")

    # =========================================================================
    # CREATE SOLVER SEQUENCE
    # =========================================================================
    sol = java.sol().create("sol1")
    sol.study(study_name)
    sol.attach(study_name)

    # Study step
    st1 = sol.feature().create("st1", "StudyStep")
    st1.set("study", study_name)
    st1.set("studystep", "stat")

    # Variable initialization
    v1 = sol.feature().create("v1", "Variables")
    v1.set("control", "stat")

    # =========================================================================
    # STATIONARY SOLVER CONFIGURATION
    # =========================================================================
    s1 = sol.feature().create("s1", "Stationary")
    s1.label("Stationary Solver")

    # Fully coupled solver (all equations solved together)
    fc1 = s1.feature().create("fc1", "FullyCoupled")
    fc1.label("Fully Coupled")

    # Convergence settings for semiconductor
    fc1.set("termonres", True)  # Terminate on residual
    fc1.set("damp", 0.9)  # Damping factor (0.9 is good for semiconductors)
    fc1.set("maxiter", 50)  # Maximum iterations
    fc1.set("ntol", 1e-6)  # Relative tolerance

    # Newton method settings
    fc1.set("jtech", "once")  # Jacobian update: once per step
    fc1.set("linsolver", "pardiso")  # Direct sparse solver

    # =========================================================================
    # CONTINUATION (for voltage sweeps)
    # =========================================================================
    if use_continuation:
        # Continuation helps convergence when parameters change gradually
        fc1.set("usecont", True)

    logger.info("Stationary study created successfully")


def create_parametric_sweep(
    model: "mph.Model",
    param_name: str,
    param_values: np.ndarray,
    study_name: str = "std1"
) -> None:
    """
    Create a parametric sweep for voltage variation.

    This modifies the study to sweep over the specified parameter,
    solving at each value and storing all solutions.

    Args:
        model: COMSOL model instance.
        param_name: Parameter to sweep ("Vgs", "Vds", or "Vbs").
        param_values: Array of parameter values to sweep.
        study_name: Name of the study to modify.
    """
    java = model.java
    study = java.study(study_name)

    logger.info(f"Creating parametric sweep: {param_name} = {param_values[0]:.2f} to {param_values[-1]:.2f} V")

    # Check if parametric sweep already exists
    try:
        study.feature().remove("param")
    except Exception:
        pass  # Doesn't exist yet

    # Create parametric sweep
    param = study.feature().create("param", "Parametric")
    param.label(f"{param_name} Sweep")

    # Set parameter name and values
    param.set("pname", [param_name])
    param.set("plistarr", [" ".join(f"{v}" for v in param_values)])
    param.set("punit", ["V"])

    # Use continuation for faster convergence
    param.set("usecont", True)

    # Use previous solution as initial guess for next point
    param.set("pcontinuationmode", "last")

    logger.info(f"Parametric sweep created with {len(param_values)} points")


def run_study(model: "mph.Model", study_name: str = "std1") -> None:
    """
    Execute the study and wait for completion.

    Args:
        model: COMSOL model instance.
        study_name: Name of the study to run.

    Raises:
        RuntimeError: If the solver fails to converge.
    """
    java = model.java

    logger.info(f"Running study '{study_name}'...")

    try:
        java.sol("sol1").runAll()
        logger.info("Study completed successfully")
    except Exception as e:
        logger.error(f"Study failed: {e}")
        raise RuntimeError(f"COMSOL solver failed: {e}")


def create_vgs_sweep_study(
    model: "mph.Model",
    vgs_min: float,
    vgs_max: float,
    n_points: int,
    vds: float,
    vbs: float = 0.0
) -> np.ndarray:
    """
    Convenience function for gate voltage sweep.

    Sets up a parametric sweep of Vgs at fixed Vds and Vbs.

    Args:
        model: COMSOL model instance.
        vgs_min: Minimum gate voltage [V].
        vgs_max: Maximum gate voltage [V].
        n_points: Number of sweep points.
        vds: Fixed drain-source voltage [V].
        vbs: Fixed body-source voltage [V].

    Returns:
        Array of Vgs values that will be simulated.
    """
    java = model.java

    # Set fixed voltages
    java.param().set("Vds", f"{vds}[V]")
    java.param().set("Vbs", f"{vbs}[V]")

    # Create sweep values
    vgs_values = np.linspace(vgs_min, vgs_max, n_points)

    # Create parametric sweep
    create_parametric_sweep(model, "Vgs", vgs_values)

    logger.info(f"Vgs sweep: {vgs_min:.2f}V to {vgs_max:.2f}V, Vds={vds:.2f}V fixed")

    return vgs_values


def create_vds_sweep_study(
    model: "mph.Model",
    vgs: float,
    vds_min: float,
    vds_max: float,
    n_points: int,
    vbs: float = 0.0
) -> np.ndarray:
    """
    Convenience function for drain voltage sweep.

    Sets up a parametric sweep of Vds at fixed Vgs and Vbs.

    Args:
        model: COMSOL model instance.
        vgs: Fixed gate-source voltage [V].
        vds_min: Minimum drain-source voltage [V].
        vds_max: Maximum drain-source voltage [V].
        n_points: Number of sweep points.
        vbs: Fixed body-source voltage [V].

    Returns:
        Array of Vds values that will be simulated.
    """
    java = model.java

    # Set fixed voltages
    java.param().set("Vgs", f"{vgs}[V]")
    java.param().set("Vbs", f"{vbs}[V]")

    # Create sweep values
    vds_values = np.linspace(vds_min, vds_max, n_points)

    # Create parametric sweep
    create_parametric_sweep(model, "Vds", vds_values)

    logger.info(f"Vds sweep: {vds_min:.2f}V to {vds_max:.2f}V, Vgs={vgs:.2f}V fixed")

    return vds_values


def create_iv_characteristic_study(
    model: "mph.Model",
    vgs_values: np.ndarray,
    vds_values: np.ndarray,
    vbs: float = 0.0
) -> None:
    """
    Create a 2D parametric sweep for full I-V characteristics.

    This sweeps over both Vgs and Vds to generate output
    characteristic curves (Id vs Vds for different Vgs).

    Args:
        model: COMSOL model instance.
        vgs_values: Array of gate voltages [V].
        vds_values: Array of drain voltages [V].
        vbs: Fixed body-source voltage [V].
    """
    java = model.java
    study = java.study("std1")

    # Set fixed body voltage
    java.param().set("Vbs", f"{vbs}[V]")

    # Create 2D parametric sweep
    param = study.feature().create("param", "Parametric")
    param.label("I-V Sweep")

    # Outer loop: Vgs
    param.set("pname", ["Vgs", "Vds"])
    param.set("plistarr", [
        " ".join(f"{v}" for v in vgs_values),
        " ".join(f"{v}" for v in vds_values)
    ])
    param.set("punit", ["V", "V"])

    # Sweep type: all combinations
    param.set("sweeptype", "filled")

    # Use continuation
    param.set("usecont", True)

    total_points = len(vgs_values) * len(vds_values)
    logger.info(f"I-V sweep: {len(vgs_values)} Vgs x {len(vds_values)} Vds = {total_points} points")


def get_solution_info(model: "mph.Model") -> dict:
    """
    Get information about the current solution.

    Returns:
        Dictionary with solution statistics.
    """
    java = model.java

    try:
        sol = java.sol("sol1")
        info = {
            "num_solutions": int(sol.feature("s1").getSolnum()),
            "converged": True,
        }
        return info
    except Exception as e:
        logger.warning(f"Could not get solution info: {e}")
        return {"num_solutions": 0, "converged": False}


def clear_solution(model: "mph.Model") -> None:
    """
    Clear the current solution to free memory.

    Useful when running multiple parametric studies.
    """
    java = model.java

    try:
        java.sol("sol1").clearSolution()
        logger.debug("Solution cleared")
    except Exception as e:
        logger.warning(f"Could not clear solution: {e}")
