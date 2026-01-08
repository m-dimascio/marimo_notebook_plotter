"""COMSOL client lifecycle management.

This module handles the connection to COMSOL Multiphysics via MPh.
It provides a global client instance for efficiency and convenience.

Example:
    >>> from mosfet_sim.comsol import get_client, create_model
    >>> client = get_client()
    >>> model = create_model("MyMOSFET")
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import mph

logger = logging.getLogger(__name__)

# Global client instance (singleton pattern for efficiency)
_client: Optional["mph.Client"] = None
_mph_available: Optional[bool] = None


def is_comsol_available() -> bool:
    """
    Check if COMSOL/MPh is available.

    Returns:
        True if MPh can be imported and COMSOL is accessible.
    """
    global _mph_available

    if _mph_available is not None:
        return _mph_available

    try:
        import mph
        # Try to discover COMSOL installation
        mph.option("classkit", False)
        _mph_available = True
        logger.info("COMSOL/MPh is available")
    except ImportError:
        logger.warning("MPh library not installed. Install with: pip install MPh==1.3.1")
        _mph_available = False
    except Exception as e:
        logger.warning(f"COMSOL not accessible: {e}")
        _mph_available = False

    return _mph_available


def get_client(cores: Optional[int] = None) -> "mph.Client":
    """
    Get or create the global COMSOL client.

    This function implements a singleton pattern to reuse the same
    client connection across multiple model operations, which is
    more efficient than starting a new connection each time.

    Args:
        cores: Number of CPU cores for parallel solving.
               If None, uses COMSOL's default.

    Returns:
        The MPh Client instance.

    Raises:
        RuntimeError: If COMSOL/MPh is not available.
    """
    global _client

    if not is_comsol_available():
        raise RuntimeError(
            "COMSOL/MPh is not available. "
            "Ensure COMSOL is installed with a valid Semiconductor Module license "
            "and MPh is installed: pip install MPh==1.3.1"
        )

    if _client is None:
        import mph

        logger.info("Starting COMSOL client...")

        if cores is not None:
            _client = mph.start(cores=cores)
        else:
            _client = mph.start()

        logger.info("COMSOL client started successfully")

    return _client


def shutdown_client() -> None:
    """
    Cleanly shutdown the COMSOL client.

    This should be called when completely done with COMSOL
    to release resources. Not typically needed during interactive
    sessions as the client can be reused.
    """
    global _client

    if _client is not None:
        logger.info("Shutting down COMSOL client...")
        try:
            _client.clear()
        except Exception as e:
            logger.warning(f"Error during client shutdown: {e}")
        finally:
            _client = None


@contextmanager
def comsol_session(cores: Optional[int] = None):
    """
    Context manager for COMSOL session.

    This provides a convenient way to ensure the client is available
    and can optionally handle cleanup. Currently reuses the global
    client for efficiency.

    Args:
        cores: Number of CPU cores for parallel solving.

    Yields:
        The MPh Client instance.

    Example:
        >>> with comsol_session() as client:
        ...     model = client.create("Model")
        ...     # work with model
    """
    client = get_client(cores=cores)
    try:
        yield client
    finally:
        # Don't shutdown - reuse client for performance
        # User can explicitly call shutdown_client() if needed
        pass


def create_model(name: str = "MOSFET") -> "mph.Model":
    """
    Create a new COMSOL model.

    Args:
        name: Name for the model (appears in COMSOL).

    Returns:
        A new MPh Model instance.
    """
    client = get_client()
    model = client.create(name)
    logger.debug(f"Created new model: {name}")
    return model


def load_model(path: Path | str) -> "mph.Model":
    """
    Load an existing COMSOL model from disk.

    Args:
        path: Path to the .mph file.

    Returns:
        The loaded MPh Model instance.

    Raises:
        FileNotFoundError: If the model file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    client = get_client()
    model = client.load(str(path))
    logger.info(f"Loaded model from: {path}")
    return model


def save_model(model: "mph.Model", path: Path | str) -> None:
    """
    Save a COMSOL model to disk.

    Args:
        model: The MPh Model instance to save.
        path: Destination path for the .mph file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    logger.info(f"Saved model to: {path}")


def clear_models() -> None:
    """
    Clear all models from the client to free memory.

    Useful when working with many models sequentially.
    """
    client = get_client()
    for name in client.names():
        client.remove(client[name])
    logger.debug("Cleared all models from client")
