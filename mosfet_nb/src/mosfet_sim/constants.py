"""Physical constants for semiconductor simulation."""

# Fundamental constants
Q = 1.602e-19           # Elementary charge (C)
K_B = 1.381e-23         # Boltzmann constant (J/K)
EPS_0 = 8.854e-12       # Vacuum permittivity (F/m)

# Derived constants
THERMAL_VOLTAGE_300K = K_B * 300 / Q  # ~0.0259 V at 300K


def thermal_voltage(temperature: float) -> float:
    """Calculate thermal voltage kT/q at given temperature (K)."""
    return K_B * temperature / Q
