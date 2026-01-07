"""MOSFET device geometry and parameters."""

from dataclasses import dataclass, field
from .materials import Semiconductor, Insulator, SILICON, SIO2


@dataclass
class MOSFETParams:
    """NMOS transistor parameters."""
    # Geometry (all in SI units: meters)
    channel_length: float     # L (m)
    channel_width: float      # W (m)
    oxide_thickness: float    # tox (m)

    # Doping (cm^-3, will convert internally)
    substrate_doping: float   # Na for NMOS (cm^-3)
    source_drain_doping: float  # Nd for source/drain (cm^-3)

    # Materials
    substrate: Semiconductor = field(default_factory=lambda: SILICON)
    oxide: Insulator = field(default_factory=lambda: SIO2)

    # Operating conditions
    temperature: float = 300.0  # K

    @classmethod
    def default_device(cls) -> "MOSFETParams":
        """Create a typical educational MOSFET device."""
        return cls(
            channel_length=1e-6,        # 1 μm
            channel_width=10e-6,        # 10 μm
            oxide_thickness=10e-9,      # 10 nm
            substrate_doping=1e17,      # cm^-3
            source_drain_doping=1e20,   # cm^-3
        )
