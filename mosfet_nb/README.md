# MOSFET Simulation Package

An interactive MOSFET device simulation package with a marimo notebook interface for exploring semiconductor device physics.

## Features

- Interactive visualization of MOSFET characteristics
- Square-law model (gradual channel approximation)
- Adjustable device parameters via sliders
- Multiple plot types:
  - Output characteristics (Id-Vds)
  - Transfer characteristics (Id-Vgs)
  - Device cross-section visualization
  - Band diagram

## Installation

```bash
cd mosfet_nb
pip install -e .
```

For development with testing support:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from mosfet_sim import MOSFETParams, threshold_voltage, drain_current

# Create a default educational device
device = MOSFETParams.default_device()

# Calculate threshold voltage
vth = threshold_voltage(device)
print(f"Threshold voltage: {vth:.3f} V")

# Calculate drain current at a specific operating point
id = drain_current(device, vgs=2.0, vds=1.5)
print(f"Drain current: {id*1e3:.3f} mA")
```

## Running the Notebook

Launch the interactive marimo notebook:

```bash
marimo edit notebooks/mosfet_explorer.py
```

Or run it directly:

```bash
marimo run notebooks/mosfet_explorer.py
```

## Physics Model

### Square-Law Model (Long-Channel Approximation)

**Linear Region** (Vds < Vgs - Vth):
```
Id = (W/L) * mu_n * Cox * [(Vgs - Vth)*Vds - Vds^2/2]
```

**Saturation Region** (Vds >= Vgs - Vth):
```
Id = (W/L) * mu_n * Cox * (Vgs - Vth)^2 / 2
```

### Key Equations

- **Threshold Voltage**: `Vth = 2*phi_f + gamma*sqrt(2*phi_f - Vbs)`
- **Oxide Capacitance**: `Cox = eps_ox / tox`
- **Bulk Potential**: `phi_f = (kT/q) * ln(Na/ni)`

## Module Structure

- `constants.py` - Physical constants (Q, k_B, eps_0)
- `materials.py` - Semiconductor and insulator properties
- `device.py` - MOSFET device parameter class
- `physics.py` - Core physics calculations
- `plotting.py` - Visualization functions

## Running Tests

```bash
cd mosfet_nb
pytest tests/
```

## Limitations

This model assumes:
- Long-channel behavior (no velocity saturation)
- Uniform doping profiles
- No short-channel effects (DIBL, punch-through)
- Ideal ohmic contacts
- No gate leakage current
- Constant mobility (no field dependence)

## License

MIT License
