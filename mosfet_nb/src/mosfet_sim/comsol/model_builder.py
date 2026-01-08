"""High-level API for building and running MOSFET simulations.

This module provides a simple, high-level interface for COMSOL
MOSFET simulation that hides the complexity of the underlying
COMSOL API.

Example:
    >>> from mosfet_sim import MOSFETParams
    >>> from mosfet_sim.comsol import MOSFETModel
    >>>
    >>> params = MOSFETParams(channel_length=1e-6)
    >>> model = MOSFETModel(params)
    >>> model.build()
    >>>
    >>> # Single bias point
    >>> conc = model.solve_single(vgs=1.5, vds=0.5)
    >>>
    >>> # Gate voltage sweep
    >>> concentrations, vgs_values = model.sweep_vgs(
    ...     vgs_min=0, vgs_max=3, vds=0.5, n_frames=30
    ... )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    import mph

from ..device import MOSFETParams
from ..physics import threshold_voltage

from .client import create_model, save_model, load_model, is_comsol_available
from .geometry import build_mosfet_geometry
from .physics import setup_semiconductor_physics, setup_global_parameters
from .mesh import build_mesh
from .study import (
    create_stationary_study,
    create_vgs_sweep_study,
    create_vds_sweep_study,
    run_study,
    clear_solution,
)
from .extract import (
    COMSOLConcentration,
    extract_concentration_field,
    extract_sweep_concentrations,
    extract_drain_current,
)

logger = logging.getLogger(__name__)


class MOSFETModel:
    """
    High-level interface for COMSOL MOSFET simulation.

    This class provides a simple API for:
    - Building the complete COMSOL model
    - Running simulations at single bias points
    - Performing voltage sweeps
    - Extracting results for visualization

    The model can be cached to disk for reuse, avoiding the
    overhead of rebuilding geometry and mesh each time.

    Attributes:
        params: MOSFET device parameters.
        mesh_refinement: Mesh quality level.
        cache_path: Optional path for caching the model.

    Example:
        >>> model = MOSFETModel(params, mesh_refinement="fine")
        >>> model.build()
        >>> concentrations, vgs = model.sweep_vgs(0, 3, 0.5, n_frames=30)
    """

    def __init__(
        self,
        params: MOSFETParams,
        mesh_refinement: str = "normal",
        cache_path: Optional[Path] = None
    ):
        """
        Initialize MOSFET model.

        Args:
            params: MOSFET device parameters.
            mesh_refinement: Mesh quality level:
                - "coarse": Fast but less accurate
                - "normal": Balanced (default)
                - "fine": High accuracy
                - "extra_fine": Maximum accuracy
            cache_path: Optional path to cache the built model.
                       If the file exists, it will be loaded instead
                       of rebuilding.
        """
        self.params = params
        self.mesh_refinement = mesh_refinement
        self.cache_path = Path(cache_path) if cache_path else None

        self._model: Optional["mph.Model"] = None
        self._built = False
        self._geom_info: dict = {}
        self._mesh_stats: dict = {}

    @property
    def is_available(self) -> bool:
        """Check if COMSOL is available."""
        return is_comsol_available()

    @property
    def is_built(self) -> bool:
        """Check if the model has been built."""
        return self._built

    @property
    def threshold_voltage(self) -> float:
        """Get threshold voltage from analytical model."""
        return threshold_voltage(self.params)

    def build(self, force: bool = False) -> None:
        """
        Build the complete COMSOL model.

        This creates:
        - 3D device geometry
        - Semiconductor physics (drift-diffusion)
        - Mesh
        - Stationary study

        Args:
            force: If True, rebuild even if cached model exists.

        Raises:
            RuntimeError: If COMSOL is not available.
        """
        if not self.is_available:
            raise RuntimeError(
                "COMSOL is not available. Ensure COMSOL is installed "
                "with a valid Semiconductor Module license."
            )

        # Try to load cached model first
        if not force and self.cache_path and self.cache_path.exists():
            if self.load_cached():
                return

        logger.info("Building COMSOL MOSFET model...")

        # Create new model
        self._model = create_model("MOSFET")

        # Build geometry
        self._geom_info = build_mosfet_geometry(self._model, self.params)

        # Setup physics
        setup_semiconductor_physics(self._model, self.params)
        setup_global_parameters(self._model)

        # Build mesh
        self._mesh_stats = build_mesh(
            self._model, self.params, self.mesh_refinement
        )

        # Create study
        create_stationary_study(self._model)

        self._built = True
        logger.info("COMSOL model built successfully")

        # Cache model if path specified
        if self.cache_path:
            self._save_cached()

    def load_cached(self) -> bool:
        """
        Load cached model if available.

        Returns:
            True if cached model was loaded successfully.
        """
        if not self.cache_path or not self.cache_path.exists():
            return False

        try:
            logger.info(f"Loading cached model from {self.cache_path}")
            self._model = load_model(self.cache_path)
            self._built = True
            return True
        except Exception as e:
            logger.warning(f"Could not load cached model: {e}")
            return False

    def _save_cached(self) -> None:
        """Save model to cache."""
        if self.cache_path and self._model:
            try:
                save_model(self._model, self.cache_path)
                logger.info(f"Model cached to {self.cache_path}")
            except Exception as e:
                logger.warning(f"Could not cache model: {e}")

    def solve_single(
        self,
        vgs: float,
        vds: float,
        vbs: float = 0.0,
        mesh_resolution: Tuple[int, int, int] = (60, 20, 30)
    ) -> COMSOLConcentration:
        """
        Solve for a single bias point.

        Args:
            vgs: Gate-source voltage [V].
            vds: Drain-source voltage [V].
            vbs: Body-source voltage [V].
            mesh_resolution: (nx, ny, nz) for result extraction.

        Returns:
            COMSOLConcentration with carrier densities.

        Raises:
            RuntimeError: If model not built or solver fails.
        """
        if not self._built:
            self.build()

        java = self._model.java

        # Clear any existing parametric sweep
        try:
            java.study("std1").feature().remove("param")
        except Exception:
            pass

        # Set bias point
        java.param().set("Vgs", f"{vgs}[V]")
        java.param().set("Vds", f"{vds}[V]")
        java.param().set("Vbs", f"{vbs}[V]")

        # Solve
        logger.info(f"Solving at Vgs={vgs:.2f}V, Vds={vds:.2f}V")
        run_study(self._model)

        # Extract results
        nx, ny, nz = mesh_resolution
        conc = extract_concentration_field(self._model, nx=nx, ny=ny, nz=nz)
        conc.vgs = vgs
        conc.vds = vds

        return conc

    def sweep_vgs(
        self,
        vgs_min: float,
        vgs_max: float,
        vds: float,
        n_frames: int = 30,
        vbs: float = 0.0,
        mesh_resolution: Tuple[int, int, int] = (60, 20, 30)
    ) -> Tuple[List[COMSOLConcentration], np.ndarray]:
        """
        Perform gate voltage sweep.

        Solves at multiple Vgs values with fixed Vds,
        extracting carrier concentrations at each point.

        Args:
            vgs_min: Minimum gate voltage [V].
            vgs_max: Maximum gate voltage [V].
            vds: Fixed drain-source voltage [V].
            n_frames: Number of sweep points.
            vbs: Fixed body-source voltage [V].
            mesh_resolution: (nx, ny, nz) for result extraction.

        Returns:
            Tuple of (concentration_list, vgs_values).
        """
        if not self._built:
            self.build()

        logger.info(f"Starting Vgs sweep: {vgs_min:.2f}V to {vgs_max:.2f}V")

        # Setup parametric sweep
        vgs_values = create_vgs_sweep_study(
            self._model,
            vgs_min=vgs_min,
            vgs_max=vgs_max,
            n_points=n_frames,
            vds=vds,
            vbs=vbs,
        )

        # Run study
        run_study(self._model)

        # Extract results
        nx, ny, nz = mesh_resolution
        concentrations = extract_sweep_concentrations(
            self._model, n_frames, nx=nx, ny=ny, nz=nz
        )

        # Add voltage info to each concentration
        for conc, vgs in zip(concentrations, vgs_values):
            conc.vgs = vgs
            conc.vds = vds

        logger.info(f"Vgs sweep complete: {len(concentrations)} frames")
        return concentrations, vgs_values

    def sweep_vds(
        self,
        vgs: float,
        vds_min: float,
        vds_max: float,
        n_frames: int = 30,
        vbs: float = 0.0,
        mesh_resolution: Tuple[int, int, int] = (60, 20, 30)
    ) -> Tuple[List[COMSOLConcentration], np.ndarray]:
        """
        Perform drain voltage sweep.

        Solves at multiple Vds values with fixed Vgs,
        extracting carrier concentrations at each point.

        Args:
            vgs: Fixed gate-source voltage [V].
            vds_min: Minimum drain-source voltage [V].
            vds_max: Maximum drain-source voltage [V].
            n_frames: Number of sweep points.
            vbs: Fixed body-source voltage [V].
            mesh_resolution: (nx, ny, nz) for result extraction.

        Returns:
            Tuple of (concentration_list, vds_values).
        """
        if not self._built:
            self.build()

        logger.info(f"Starting Vds sweep: {vds_min:.2f}V to {vds_max:.2f}V")

        # Setup parametric sweep
        vds_values = create_vds_sweep_study(
            self._model,
            vgs=vgs,
            vds_min=vds_min,
            vds_max=vds_max,
            n_points=n_frames,
            vbs=vbs,
        )

        # Run study
        run_study(self._model)

        # Extract results
        nx, ny, nz = mesh_resolution
        concentrations = extract_sweep_concentrations(
            self._model, n_frames, nx=nx, ny=ny, nz=nz
        )

        # Add voltage info to each concentration
        for conc, vds in zip(concentrations, vds_values):
            conc.vgs = vgs
            conc.vds = vds

        logger.info(f"Vds sweep complete: {len(concentrations)} frames")
        return concentrations, vds_values

    def get_iv_curves(
        self,
        vgs_values: np.ndarray,
        vds_values: np.ndarray,
        vbs: float = 0.0
    ) -> np.ndarray:
        """
        Compute I-V characteristics.

        Performs a 2D sweep over Vgs and Vds to generate
        complete output characteristics.

        Args:
            vgs_values: Array of gate voltages [V].
            vds_values: Array of drain voltages [V].
            vbs: Fixed body-source voltage [V].

        Returns:
            2D array of drain currents [len(vgs_values), len(vds_values)].
        """
        if not self._built:
            self.build()

        logger.info(
            f"Computing I-V curves: {len(vgs_values)} Vgs x {len(vds_values)} Vds"
        )

        currents = np.zeros((len(vgs_values), len(vds_values)))

        for i, vgs in enumerate(vgs_values):
            # Setup Vds sweep at this Vgs
            create_vds_sweep_study(
                self._model,
                vgs=vgs,
                vds_min=vds_values[0],
                vds_max=vds_values[-1],
                n_points=len(vds_values),
                vbs=vbs,
            )

            run_study(self._model)
            currents[i, :] = extract_drain_current(self._model)

            # Clear solution to save memory
            clear_solution(self._model)

        logger.info("I-V curves computed successfully")
        return currents

    def get_mesh_statistics(self) -> dict:
        """Get mesh statistics."""
        return self._mesh_stats.copy()

    def get_geometry_info(self) -> dict:
        """Get geometry information."""
        return self._geom_info.copy()

    def clear(self) -> None:
        """Clear the solution to free memory."""
        if self._model:
            clear_solution(self._model)

    def save(self, path: Path) -> None:
        """Save the model to disk."""
        if self._model:
            save_model(self._model, path)


def create_quick_simulation(
    params: MOSFETParams,
    vgs: float,
    vds: float,
    mesh_refinement: str = "coarse"
) -> COMSOLConcentration:
    """
    Quick single-point simulation.

    Convenience function for running a single bias point
    without explicitly managing the model object.

    Args:
        params: MOSFET device parameters.
        vgs: Gate-source voltage [V].
        vds: Drain-source voltage [V].
        mesh_refinement: Mesh quality level.

    Returns:
        COMSOLConcentration with results.
    """
    model = MOSFETModel(params, mesh_refinement=mesh_refinement)
    return model.solve_single(vgs, vds)


def create_vgs_animation(
    params: MOSFETParams,
    vgs_min: float,
    vgs_max: float,
    vds: float,
    n_frames: int = 30,
    mesh_refinement: str = "normal"
) -> Tuple[List[COMSOLConcentration], np.ndarray]:
    """
    Create Vgs sweep animation data.

    Convenience function for generating animation frames
    without explicitly managing the model object.

    Args:
        params: MOSFET device parameters.
        vgs_min: Minimum gate voltage [V].
        vgs_max: Maximum gate voltage [V].
        vds: Fixed drain-source voltage [V].
        n_frames: Number of animation frames.
        mesh_refinement: Mesh quality level.

    Returns:
        Tuple of (concentration_list, vgs_values).
    """
    model = MOSFETModel(params, mesh_refinement=mesh_refinement)
    return model.sweep_vgs(vgs_min, vgs_max, vds, n_frames)
