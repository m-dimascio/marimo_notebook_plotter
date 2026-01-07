"""Semiconductor and insulator material properties."""

from dataclasses import dataclass
from .constants import EPS_0


@dataclass(frozen=True)
class Semiconductor:
    """Semiconductor material properties."""
    name: str
    eps_r: float              # Relative permittivity
    ni: float                 # Intrinsic carrier concentration (cm^-3)
    eg: float                 # Bandgap energy (eV)
    mu_n: float               # Electron mobility (cm^2/V·s)
    mu_p: float               # Hole mobility (cm^2/V·s)

    @property
    def permittivity(self) -> float:
        """Absolute permittivity (F/m)."""
        return self.eps_r * EPS_0


@dataclass(frozen=True)
class Insulator:
    """Insulator material properties."""
    name: str
    eps_r: float              # Relative permittivity

    @property
    def permittivity(self) -> float:
        """Absolute permittivity (F/m)."""
        return self.eps_r * EPS_0


# Pre-defined materials
SILICON = Semiconductor(
    name="Silicon",
    eps_r=11.7,
    ni=1.5e10,               # cm^-3 at 300K
    eg=1.12,                 # eV
    mu_n=1400,               # cm^2/V·s
    mu_p=450,                # cm^2/V·s
)

SIO2 = Insulator(name="Silicon Dioxide", eps_r=3.9)
