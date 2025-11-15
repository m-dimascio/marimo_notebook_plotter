# Chroma 63600 Series Electronic Load - Parallel Operation Research Report

## Executive Summary

This report provides comprehensive guidance on using PyVISA to synchronize two Chroma 63600 series load modules for parallel operation, specifically for `set_cc_current` calls and simultaneous load activation to provide parallel current paths.

## Table of Contents

1. [Overview of Parallel Operation](#overview-of-parallel-operation)
2. [Hardware Configuration](#hardware-configuration)
3. [SCPI Command Reference](#scpi-command-reference)
4. [PyVISA Implementation](#pyvisa-implementation)
5. [Synchronization Methods](#synchronization-methods)
6. [GitHub Examples](#github-examples)
7. [Complete Implementation Example](#complete-implementation-example)
8. [Best Practices and Considerations](#best-practices-and-considerations)

---

## Overview of Parallel Operation

### Capabilities

The Chroma 63600 series supports:
- **Master/Slave parallel operation** for high current and power applications
- Up to **2kW per mainframe** in parallel configuration
- **Multi-mainframe synchronization** - up to 4 mainframes (40 channels total)
- **10 outputs per mainframe** can be synchronized dynamically
- **Smart Master/Slave mode** - program the master, settings auto-download to slaves

### Operating Mode Compatibility

- **Compatible modes**: CC (Constant Current), CR (Constant Resistance), CP (Constant Power)
- **Incompatible mode**: CV (Constant Voltage) - modules CANNOT be paralleled in CV mode

### Hardware Requirements

- System Bus 1 or System Bus 2 ports on rear panel
- Physical cable connection between master and slave modules
- Same load module specifications recommended for balanced operation

---

## Hardware Configuration

### Physical Connections

1. Connect the **System Bus** cable between modules (rear panel ports)
2. Connect both modules to the Device Under Test (DUT)
3. Ensure proper grounding and power connections

### Configuration Modes

Three parallel configuration states:

| Mode | Value | Description |
|------|-------|-------------|
| NONE | 0 | Normal standalone operation (default) |
| MASTER | 1 | Master module - controls slave(s) |
| SLAVE | 2 | Slave module - follows master commands |

---

## SCPI Command Reference

### Essential Parallel Operation Commands

#### 1. Configure Parallel Mode

```
:CONFigure:PARAllel:MODE <mode>
```

**Parameters:**
- `NONE` or `0` - Disable parallel operation
- `MASTER` or `1` - Set as master module
- `SLAVE` or `2` - Set as slave module

**Example:**
```
CONF:PARA:MODE MASTER
CONF:PARA:MODE SLAVE
```

#### 2. Initialize Parallel Operation

```
:CONFigure:PARAllel:INITial <state>
```

**Parameters:**
- `OFF` or `0` - Exit parallel mode
- `ON` or `1` - Enter parallel mode

**Example:**
```
CONF:PARA:INIT ON
```

#### 3. Configure Synchronization

```
:CONFigure:SYNC:MODE <mode>
```

Used for dynamic synchronous function across multiple outputs.

#### 4. Set Constant Current Mode

```
MODE <mode>
```

**Parameters:**
- `CCL` - Constant Current Low range
- `CCM` - Constant Current Medium range
- `CCH` - Constant Current High range

**Example:**
```
MODE CCH
```

#### 5. Set Current Level

```
:CURRent:STATic:L1 <value>
```

**Parameters:**
- `<value>` - Current in amperes

**Example:**
```
CURR:STAT:L1 10.5
```

#### 6. Load On/Off Control

```
:LOAD:STATus <state>
```

**Parameters:**
- `ON` or `1` - Turn load on
- `OFF` or `0` - Turn load off

**Example:**
```
LOAD:STAT ON
```

#### 7. Channel Selection and Control

```
:CHANnel:LOAD <channel>
:CHANnel:ACTive <state>
```

**Example:**
```
CHAN:LOAD 1
CHAN:ACT ON
```

#### 8. All Run On/Off (Synchronous Control)

```
:CONFigure:ALLR <state>
```

Enables simultaneous control of all configured channels.

---

## PyVISA Implementation

### Basic Connection Setup

```python
import visa
import time

# Initialize VISA resource manager
rm = visa.ResourceManager()

# Connect to load modules
# Replace with your actual VISA addresses
master_load = rm.open_resource('GPIB0::5::INSTR')
slave_load = rm.open_resource('GPIB0::6::INSTR')

# Set timeout (milliseconds)
master_load.timeout = 5000
slave_load.timeout = 5000

# Query identity
print(master_load.query('*IDN?'))
print(slave_load.query('*IDN?'))
```

### Configure Parallel Operation

```python
def configure_parallel_loads(master, slave):
    """
    Configure two loads for parallel operation.

    Args:
        master: PyVISA resource object for master load
        slave: PyVISA resource object for slave load
    """
    # Configure master module
    master.write('CONF:PARA:MODE MASTER')
    master.write('CONF:PARA:INIT ON')

    # Configure slave module
    slave.write('CONF:PARA:MODE SLAVE')
    slave.write('CONF:PARA:INIT ON')

    # Allow time for configuration
    time.sleep(0.5)

    # Verify configuration
    master_mode = master.query('CONF:PARA:MODE?')
    slave_mode = slave.query('CONF:PARA:MODE?')

    print(f"Master mode: {master_mode.strip()}")
    print(f"Slave mode: {slave_mode.strip()}")
```

### Set Constant Current Mode

```python
def set_cc_current(load, current, current_range='CCH'):
    """
    Set load to constant current mode with specified current.

    Args:
        load: PyVISA resource object
        current: Current value in amperes
        current_range: 'CCL', 'CCM', or 'CCH'
    """
    # Set current mode
    load.write(f'MODE {current_range}')

    # Set current level
    load.write(f'CURR:STAT:L1 {current}')

    # Verify setting
    actual_current = load.query('CURR:STAT:L1?')
    print(f"Set current: {current}A, Actual: {actual_current.strip()}A")
```

### Automatic Current Range Selection

Based on the GitHub example, here's how to automatically select the appropriate current range:

```python
def get_current_range(current, max_ccl=0.2, max_ccm=2.0, max_cch=20.0, margin=0.9):
    """
    Determine appropriate current range based on current value.

    Args:
        current: Desired current in amperes
        max_ccl: Maximum CCL range current
        max_ccm: Maximum CCM range current
        max_cch: Maximum CCH range current
        margin: Safety margin factor (0.9 = 90%)

    Returns:
        str: 'CCL', 'CCM', or 'CCH'
    """
    if current > (max_cch * margin):
        raise ValueError(f"Current {current}A exceeds maximum rating")
    elif current > (max_ccm * margin):
        return 'CCH'
    elif current > (max_ccl * margin):
        return 'CCM'
    else:
        return 'CCL'

def set_cc_current_auto(load, current):
    """
    Set CC current with automatic range selection.
    """
    # Adjust these values based on your specific load model
    # Example for 63610-80-20: CCH=20A, CCM=2A, CCL=0.2A
    current_range = get_current_range(current)
    set_cc_current(load, current, current_range)
```

### Synchronized Load Activation

The critical part for parallel operation - turning on loads simultaneously:

#### Method 1: Master-Only Control (Recommended for Parallel Mode)

When properly configured in master/slave mode, commanding the master automatically controls the slave:

```python
def synchronized_load_on_parallel(master_load, slave_load, current):
    """
    Turn on parallel loads simultaneously using master/slave configuration.

    Args:
        master_load: Master load resource
        slave_load: Slave load resource
        current: Total current to be split between loads
    """
    # In parallel mode, each load carries current, so split the total
    current_per_load = current / 2.0

    # Set current on master (will propagate to slave in smart mode)
    set_cc_current_auto(master_load, current_per_load)

    # Turn on master load (should control both in parallel mode)
    master_load.write('LOAD:STAT ON')

    print(f"Parallel loads activated: {current_per_load}A each, {current}A total")
```

#### Method 2: GPIB Group Execute Trigger (Most Precise Timing)

For the most precise synchronization timing using GPIB:

```python
def synchronized_load_on_gpib(rm, master_addr, slave_addr, current):
    """
    Turn on loads with precise GPIB synchronization.

    Args:
        rm: VISA ResourceManager
        master_addr: GPIB address string for master (e.g., 'GPIB0::5::INSTR')
        slave_addr: GPIB address string for slave (e.g., 'GPIB0::6::INSTR')
        current: Current per load in amperes
    """
    # Open interface and instruments
    intf = rm.open_resource('GPIB0::INTFC')
    master = rm.open_resource(master_addr)
    slave = rm.open_resource(slave_addr)

    # Configure both loads
    set_cc_current_auto(master, current)
    set_cc_current_auto(slave, current)

    # Prepare both loads (don't turn on yet)
    master.write('*TRG')  # Arm for trigger
    slave.write('*TRG')

    # Use group execute trigger for simultaneous activation
    intf.group_execute_trigger(master, slave)

    print(f"Loads synchronized via GPIB trigger: {current}A each")
```

#### Method 3: Sequential with Minimal Delay

For non-GPIB interfaces or when GPIB trigger is unavailable:

```python
def synchronized_load_on_sequential(master_load, slave_load, current):
    """
    Turn on loads sequentially with minimal delay.
    Less precise than GPIB trigger but works with any interface.

    Args:
        master_load: Master load resource
        slave_load: Slave load resource
        current: Current per load in amperes
    """
    # Pre-configure both loads
    set_cc_current_auto(master_load, current)
    set_cc_current_auto(slave_load, current)

    # Turn on both loads as quickly as possible
    # Use write (not query) to avoid waiting for response
    master_load.write('LOAD:STAT ON')
    slave_load.write('LOAD:STAT ON')

    # Optional: verify both are on
    time.sleep(0.1)
    master_status = master_load.query('LOAD:STAT?')
    slave_status = slave_load.query('LOAD:STAT?')

    print(f"Master status: {master_status.strip()}")
    print(f"Slave status: {slave_status.strip()}")
```

### Complete Workflow Example

```python
def parallel_load_test(master_addr, slave_addr, test_current):
    """
    Complete example of parallel load operation.

    Args:
        master_addr: VISA address of master load
        slave_addr: VISA address of slave load
        test_current: Total test current in amperes
    """
    # Initialize
    rm = visa.ResourceManager()
    master = rm.open_resource(master_addr)
    slave = rm.open_resource(slave_addr)

    master.timeout = 5000
    slave.timeout = 5000

    try:
        # Step 1: Configure parallel operation
        print("Configuring parallel operation...")
        configure_parallel_loads(master, slave)

        # Step 2: Set current (split between loads)
        current_per_load = test_current / 2.0
        print(f"\nSetting current: {current_per_load}A per load")
        set_cc_current_auto(master, current_per_load)

        # In master/slave mode, slave follows master automatically
        # But you can explicitly set it too:
        set_cc_current_auto(slave, current_per_load)

        # Step 3: Turn on loads synchronously
        print("\nActivating loads...")
        synchronized_load_on_parallel(master, slave, test_current)

        # Step 4: Monitor (example)
        time.sleep(1)
        master_voltage = master.query('MEAS:VOLT?')
        master_current = master.query('MEAS:CURR?')
        slave_voltage = slave.query('MEAS:VOLT?')
        slave_current = slave.query('MEAS:CURR?')

        print(f"\nMeasurements:")
        print(f"Master: {master_voltage.strip()}V, {master_current.strip()}A")
        print(f"Slave: {slave_voltage.strip()}V, {slave_current.strip()}A")

        # Step 5: Turn off loads
        print("\nDeactivating loads...")
        master.write('LOAD:STAT OFF')
        slave.write('LOAD:STAT OFF')

    finally:
        # Cleanup
        master.close()
        slave.close()
        rm.close()

# Usage example
if __name__ == '__main__':
    parallel_load_test('GPIB0::5::INSTR', 'GPIB0::6::INSTR', 20.0)
```

---

## Synchronization Methods

### Comparison of Synchronization Approaches

| Method | Precision | Interface | Complexity | Use Case |
|--------|-----------|-----------|------------|----------|
| Master/Slave | Good | Any | Low | Recommended for most parallel operations |
| GPIB Trigger | Excellent | GPIB only | Medium | Critical timing requirements |
| Sequential | Fair | Any | Low | Non-critical applications |

### Timing Considerations

1. **Master/Slave Mode**: The slave automatically follows master commands with minimal delay (typically microseconds)
2. **GPIB Group Trigger**: Simultaneous trigger to multiple devices, synchronization within nanoseconds
3. **Sequential Commands**: Delay depends on interface (GPIB: ~1-5ms, USB: ~10-50ms, Ethernet: variable)

### Best Synchronization Practice

For parallel current paths, the **Master/Slave configuration is recommended** because:
- Automatic current distribution
- Built-in synchronization
- Simplified programming (command master only)
- Reliable parallel operation

---

## GitHub Examples

### 1. Korrigan36/External-Ports

**Repository**: https://github.com/Korrigan36/External-Ports

**Description**: USB and HDMI ports testing application using Chroma 63600 electronic load

**Key Files**:
- `chromaLib.py` - Core library with SCPI command wrappers
- `loadFunctions_Chroma_63600.py` - High-level test functions

**Relevant Code Snippets**:

```python
# From loadFunctions_Chroma_63600.py
import visa
import chromaLib
from time import sleep

class LoadControl:
    def __init__(self, visa_address):
        rm = visa.ResourceManager()
        self.load = rm.open_resource(visa_address)
        self.load.timeout = 2500

    def setup_cc_mode(self, current):
        # Determine current range
        if current > 18.0:  # 90% of 20A max
            str_current_range = "CCH"
        elif current > 1.8:  # 90% of 2A max
            str_current_range = "CCM"
        else:
            str_current_range = "CCL"

        # Configure
        chromaLib.load_CurrentMode(self.load, str_current_range)
        chromaLib.load_SetCurrent(self.load, current)

    def configure_parallel(self, mode):
        """
        mode: "NONE", "MASTER", or "SLAVE"
        """
        chromaLib.load_ConfigParallel(self.load, mode)

    def turn_on(self):
        chromaLib.load_TurnOnOffLoad(self.load, "ON")

    def turn_off(self):
        chromaLib.load_TurnOnOffLoad(self.load, "OFF")
```

**chromaLib.py SCPI Commands**:

```python
def load_ConfigParallel(loadObject, text):
    """Configure parallel mode"""
    tempString = "CONF:PARA:MODE " + text
    loadObject.write(tempString)

def load_ConfigSync(loadObject, text):
    """Configure synchronization"""
    tempString = "CONF:SYNC:MODE " + text
    loadObject.write(tempString)

def load_SetCurrent(loadObject, current):
    """Set static current L1"""
    tempString = "CURR:STAT:L1 " + str(current)
    loadObject.write(tempString)

def load_TurnOnOffLoad(loadObject, text):
    """Turn load on or off"""
    tempString = "LOAD:STAT " + text
    loadObject.write(tempString)

def load_CurrentMode(loadObject, text):
    """Set current mode (CCL/CCM/CCH)"""
    tempString = "MODE " + text
    loadObject.write(tempString)

def load_SelectSingleChannel(loadObject, channel):
    """Select channel"""
    tempString = "CHAN:LOAD " + str(channel)
    loadObject.write(tempString)

def load_EnableSingleChannel(loadObject, text):
    """Enable/disable channel"""
    tempString = "CHAN:ACT " + text
    loadObject.write(tempString)
```

### 2. fabrguer/Lab-Automation-with-Python

**Repository**: https://github.com/fabrguer/Lab-Automation-with-Python

**File**: `Eff_Ploss_Automation.py`

**Description**: Efficiency and power loss automation using Chroma electronic load

**Relevant Code**:

```python
import visa

# Initialize
chroma = visa.instrument("GPIB::2")

# Configure mode
chroma.write("MODE CCH")

# Set current
load = 10.5  # amperes
chroma.write("CURR:STAT:L1 %.2f" % load)

# Turn on
chroma.write("LOAD ON")

# Measurements would follow here
# ...

# Turn off
chroma.write("LOAD OFF")
```

### 3. python-ivi/python-ivi

**Repository**: https://github.com/python-ivi/python-ivi

**Description**: Python implementation of Interchangeable Virtual Instrument standard

**Notes**: While this library supports Chroma instruments (specifically 62000P power supplies), it demonstrates industry-standard patterns for instrument control that can be adapted for the 63600 series.

---

## Complete Implementation Example

Here's a production-ready class for controlling Chroma 63600 loads in parallel:

```python
"""
Chroma 63600 Parallel Load Controller
Supports synchronized operation of two load modules
"""

import visa
import time
from typing import Optional, Tuple
from enum import Enum

class CurrentRange(Enum):
    """Current range enumeration"""
    CCL = "CCL"  # Low range
    CCM = "CCM"  # Medium range
    CCH = "CCH"  # High range

class ParallelMode(Enum):
    """Parallel configuration modes"""
    NONE = "NONE"
    MASTER = "MASTER"
    SLAVE = "SLAVE"

class LoadSpecs:
    """Load module specifications"""
    def __init__(self, model: str):
        self.model = model

        # Default specs for 63610-80-20
        # Adjust based on your actual model
        if "63610-80-20" in model:
            self.ccl_max = 0.2   # 200mA
            self.ccm_max = 2.0   # 2A
            self.cch_max = 20.0  # 20A
            self.voltage_max = 80.0  # 80V
            self.power_max = 200.0   # 200W
        else:
            # Add other model specs as needed
            raise ValueError(f"Unknown model: {model}")

class Chroma63600Load:
    """
    Controller for a single Chroma 63600 load module
    """

    def __init__(self, visa_address: str, timeout: int = 5000):
        """
        Initialize load controller

        Args:
            visa_address: VISA resource address (e.g., 'GPIB0::5::INSTR')
            timeout: Command timeout in milliseconds
        """
        self.rm = visa.ResourceManager()
        self.load = self.rm.open_resource(visa_address)
        self.load.timeout = timeout
        self.address = visa_address

        # Get identity
        self.identity = self.load.query('*IDN?').strip()

        # Get specs
        self.specs = LoadSpecs(self.identity)

        print(f"Connected to: {self.identity}")

    def configure_parallel(self, mode: ParallelMode):
        """Configure parallel operation mode"""
        self.load.write(f'CONF:PARA:MODE {mode.value}')
        time.sleep(0.2)

        # Verify
        actual_mode = self.load.query('CONF:PARA:MODE?').strip()
        print(f"Parallel mode set to: {actual_mode}")

    def initialize_parallel(self, enable: bool):
        """Initialize or exit parallel mode"""
        state = "ON" if enable else "OFF"
        self.load.write(f'CONF:PARA:INIT {state}')
        time.sleep(0.2)

    def get_current_range(self, current: float, margin: float = 0.9) -> CurrentRange:
        """
        Determine appropriate current range

        Args:
            current: Desired current in amperes
            margin: Safety margin (default 90%)

        Returns:
            CurrentRange enum value
        """
        if current > self.specs.cch_max * margin:
            raise ValueError(f"Current {current}A exceeds maximum {self.specs.cch_max}A")
        elif current > self.specs.ccm_max * margin:
            return CurrentRange.CCH
        elif current > self.specs.ccl_max * margin:
            return CurrentRange.CCM
        else:
            return CurrentRange.CCL

    def set_cc_current(self, current: float,
                      current_range: Optional[CurrentRange] = None,
                      slew_rate: Optional[float] = None):
        """
        Set constant current mode and current level

        Args:
            current: Current in amperes
            current_range: Specific range, or None for auto
            slew_rate: Optional slew rate in A/ms
        """
        # Auto-select range if not specified
        if current_range is None:
            current_range = self.get_current_range(current)

        # Set mode
        self.load.write(f'MODE {current_range.value}')

        # Set current
        self.load.write(f'CURR:STAT:L1 {current}')

        # Set slew rate if specified
        if slew_rate is not None:
            self.load.write(f'CURR:STAT:RISE {slew_rate}')
            self.load.write(f'CURR:STAT:FALL {slew_rate}')

        # Verify
        actual_current = float(self.load.query('CURR:STAT:L1?'))
        print(f"{self.address}: Set {current}A, Range {current_range.value}, Actual {actual_current}A")

    def turn_on(self):
        """Turn load on"""
        self.load.write('LOAD:STAT ON')

    def turn_off(self):
        """Turn load off"""
        self.load.write('LOAD:STAT OFF')

    def measure(self) -> Tuple[float, float, float]:
        """
        Measure voltage, current, and power

        Returns:
            Tuple of (voltage, current, power)
        """
        voltage = float(self.load.query('MEAS:VOLT?'))
        current = float(self.load.query('MEAS:CURR?'))
        power = float(self.load.query('MEAS:POW?'))
        return voltage, current, power

    def close(self):
        """Close connection"""
        self.load.close()

class Chroma63600ParallelSystem:
    """
    Controller for parallel operation of two Chroma 63600 loads
    """

    def __init__(self, master_address: str, slave_address: str, timeout: int = 5000):
        """
        Initialize parallel load system

        Args:
            master_address: VISA address of master load
            slave_address: VISA address of slave load
            timeout: Command timeout in milliseconds
        """
        self.master = Chroma63600Load(master_address, timeout)
        self.slave = Chroma63600Load(slave_address, timeout)

        print("\nConfiguring parallel operation...")
        self.configure_parallel()

    def configure_parallel(self):
        """Configure master/slave parallel operation"""
        self.master.configure_parallel(ParallelMode.MASTER)
        self.master.initialize_parallel(True)

        self.slave.configure_parallel(ParallelMode.SLAVE)
        self.slave.initialize_parallel(True)

        time.sleep(0.5)
        print("Parallel configuration complete")

    def set_parallel_current(self, total_current: float,
                            current_range: Optional[CurrentRange] = None):
        """
        Set current for parallel operation

        Args:
            total_current: Total current to be split between loads
            current_range: Optional specific current range
        """
        current_per_load = total_current / 2.0

        print(f"\nSetting parallel current: {total_current}A total ({current_per_load}A each)")

        # Set both loads
        self.master.set_cc_current(current_per_load, current_range)
        self.slave.set_cc_current(current_per_load, current_range)

    def turn_on_synchronized(self):
        """
        Turn on both loads with minimal delay
        In master/slave mode, commanding master should control both
        """
        print("\nActivating parallel loads...")

        # In true master/slave mode, this might be sufficient:
        self.master.turn_on()

        # But for safety, also command slave:
        self.slave.turn_on()

        time.sleep(0.1)
        print("Loads activated")

    def turn_off_synchronized(self):
        """Turn off both loads"""
        print("\nDeactivating parallel loads...")

        self.master.turn_off()
        self.slave.turn_off()

        time.sleep(0.1)
        print("Loads deactivated")

    def measure_both(self) -> dict:
        """
        Measure both loads

        Returns:
            Dictionary with measurements from both loads
        """
        master_v, master_i, master_p = self.master.measure()
        slave_v, slave_i, slave_p = self.slave.measure()

        return {
            'master': {'voltage': master_v, 'current': master_i, 'power': master_p},
            'slave': {'voltage': slave_v, 'current': slave_i, 'power': slave_p},
            'total': {'current': master_i + slave_i, 'power': master_p + slave_p}
        }

    def close(self):
        """Close both connections"""
        self.master.close()
        self.slave.close()

# Example usage
def main():
    """Example of parallel load operation"""

    # Create parallel system
    system = Chroma63600ParallelSystem(
        master_address='GPIB0::5::INSTR',
        slave_address='GPIB0::6::INSTR'
    )

    try:
        # Test sequence
        test_currents = [5.0, 10.0, 15.0, 20.0]  # Amperes

        for current in test_currents:
            print(f"\n{'='*60}")
            print(f"Testing at {current}A total current")
            print('='*60)

            # Set current
            system.set_parallel_current(current)

            # Turn on
            system.turn_on_synchronized()

            # Wait for settling
            time.sleep(2)

            # Measure
            measurements = system.measure_both()

            print("\nMeasurements:")
            print(f"  Master: {measurements['master']['voltage']:.3f}V, "
                  f"{measurements['master']['current']:.3f}A, "
                  f"{measurements['master']['power']:.3f}W")
            print(f"  Slave:  {measurements['slave']['voltage']:.3f}V, "
                  f"{measurements['slave']['current']:.3f}A, "
                  f"{measurements['slave']['power']:.3f}W")
            print(f"  Total:  {measurements['total']['current']:.3f}A, "
                  f"{measurements['total']['power']:.3f}W")

            # Turn off
            system.turn_off_synchronized()

            # Wait between tests
            time.sleep(1)

    finally:
        # Cleanup
        system.close()
        print("\nTest complete")

if __name__ == '__main__':
    main()
```

---

## Best Practices and Considerations

### 1. Hardware Setup

- **Cable quality**: Use high-quality system bus cables for reliable synchronization
- **Grounding**: Ensure proper grounding to avoid ground loops
- **Power ratings**: Verify combined power dissipation doesn't exceed chassis limits
- **Cooling**: Ensure adequate ventilation for parallel operation (higher power)

### 2. Software Configuration

- **Always configure master first**, then slave
- **Verify parallel mode** with query commands before operation
- **Use appropriate current ranges** - don't overdrive low ranges
- **Implement error checking** for all SCPI commands

### 3. Current Distribution

- In **master/slave mode**, current should distribute evenly
- Monitor both loads to verify balanced operation
- If imbalance occurs, check:
  - Physical connections
  - Cable quality
  - Load calibration
  - Module specifications match

### 4. Timing Considerations

- **Settling time**: Allow 50-100ms after configuration changes
- **Measurement delay**: Wait 1-2s after load activation for stable readings
- **Command delays**: Don't send commands faster than device can process (~10ms minimum)

### 5. Safety

- **Over-current protection**: Set appropriate limits
- **Voltage limits**: Configure max voltage to protect DUT
- **Thermal monitoring**: Monitor load temperatures during high-power tests
- **Emergency stop**: Implement quick shutdown capability

### 6. Error Handling

```python
def safe_load_operation(system, current):
    """Example with error handling"""
    try:
        system.set_parallel_current(current)
        system.turn_on_synchronized()

        # Check for errors
        master_errors = system.master.load.query('SYST:ERR?')
        slave_errors = system.slave.load.query('SYST:ERR?')

        if 'No error' not in master_errors:
            print(f"Master error: {master_errors}")
            raise RuntimeError("Master load error")

        if 'No error' not in slave_errors:
            print(f"Slave error: {slave_errors}")
            raise RuntimeError("Slave load error")

        # Proceed with test...

    except Exception as e:
        print(f"Error occurred: {e}")
        # Emergency shutdown
        try:
            system.turn_off_synchronized()
        except:
            pass
        raise
    finally:
        # Always cleanup
        system.turn_off_synchronized()
```

### 7. Calibration

- Verify both modules are calibrated
- Check current distribution under various loads
- Compensate for any systematic imbalances in software if needed

### 8. Documentation

Maintain records of:
- Load module serial numbers
- Calibration dates
- Physical connection configuration
- Test procedures and results

---

## Additional Resources

### Official Documentation

1. **Chroma 63600 Series Manual V2.2**
   - URL: https://assets.tequipment.net/assets/1/26/Chroma_63600_Series_-_Manual_V2.2.pdf
   - Complete programming reference and specifications

2. **Chroma Application Notes**
   - URL: https://www.chromausa.com/support/application-notes/
   - Parallel operation guides (requires registration)

3. **Chroma User Manuals**
   - URL: https://www.chromausa.com/support/user-manuals/
   - Quick start guides and hardware manuals

### GitHub Repositories

1. **Korrigan36/External-Ports**
   - URL: https://github.com/Korrigan36/External-Ports
   - Production code using chromaLib

2. **fabrguer/Lab-Automation-with-Python**
   - URL: https://github.com/fabrguer/Lab-Automation-with-Python
   - Lab automation examples

3. **python-ivi/python-ivi**
   - URL: https://github.com/python-ivi/python-ivi
   - IVI standard implementation (Chroma power supplies)

### PyVISA Documentation

1. **PyVISA Official Docs**
   - URL: https://pyvisa.readthedocs.io/
   - Complete VISA interface documentation

2. **SCPI Commands Reference**
   - Standard Commands for Programmable Instruments
   - IEEE 488.2 specification

---

## Conclusion

The Chroma 63600 series electronic loads provide robust parallel operation capabilities through master/slave configuration. Key takeaways:

1. **Master/Slave mode** is the recommended approach for parallel operation
2. **Synchronization** is built into the hardware - configure properly and it works automatically
3. **PyVISA** provides excellent control through standard SCPI commands
4. **GitHub examples** demonstrate proven patterns for production use
5. **Proper configuration** (hardware and software) is critical for reliable operation

For `set_cc_current` synchronization, the workflow is:

1. Configure one load as MASTER, one as SLAVE
2. Initialize parallel mode on both
3. Set current on both loads (or just master in smart mode)
4. Turn on master (controls both in parallel mode)
5. Monitor both loads to verify proper operation

The provided code examples should give you a solid foundation for implementing parallel load control in your application.

---

**Report Generated**: 2025-11-15
**Chroma 63600 Series**: DC Electronic Load
**Purpose**: Parallel Operation with PyVISA Synchronization
