# Chroma 63600 Series Electronic Load - Parallel Operation Research Report

## Executive Summary

This report provides comprehensive guidance on using PyVISA to synchronize Chroma 63600 series load modules for parallel operation, specifically for `set_CC_current` calls and simultaneous load activation to provide parallel current paths.

**Updated**: This document has been revised to align with the actual driver implementation (`Equipment` base class and `Chroma_63600_5` subclass), which uses:
- Single PyVISA connection for multi-channel control within one mainframe
- `parallel_init()` method for master/slave channel configuration
- Channel selection via `CHANNEL` command before each operation
- SCPI commands: `CONFIGURE:PARALLEL:INIT`, `CONFIGURE:PARALLEL:MODE`, `CURR:STAT:L2`, `LOAD ON/OFF`

The report also documents alternative architectures for multi-mainframe setups requiring separate VISA connections.

## Table of Contents

1. [Overview of Parallel Operation](#overview-of-parallel-operation)
2. [Hardware Configuration](#hardware-configuration)
3. [SCPI Command Reference](#scpi-command-reference)
4. [PyVISA Implementation](#pyvisa-implementation) - **Current Driver Architecture**
5. [Synchronization Methods](#synchronization-methods)
6. [GitHub Examples](#github-examples)
7. [Extending the Current Driver](#extending-the-current-driver) - **Recommended Enhancements**
8. [Best Practices and Considerations](#best-practices-and-considerations)
9. [Additional Resources](#additional-resources)
10. [Conclusion](#conclusion)

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

### Current Driver Architecture

The driver is implemented using a base `Equipment` class with a specialized `Chroma_63600_5` subclass:

```python
import pyvisa
from time import sleep

class Equipment:
    """Base class for PyVISA instrument control"""

    def __init__(self, addr, delay=0.05, mock=False):
        self.delay = delay
        self.addr = addr
        self.inst = None
        self.mock_ = mock

        # Auto-retry connection up to 5 times
        attempt = 0
        while attempt < 5:
            try:
                self.inst = pyvisa.ResourceManager().open_resource(self.addr)
                self.inst.timeout = 30000  # 30 seconds
                break
            except pyvisa.errors.VisaIOError:
                attempt += 1

        self.IDN = self.query("*IDN?").strip()

    def write(self, content):
        """Write command with delay"""
        sleep(self.delay)
        self.inst.write(content)

    def query(self, content):
        """Query command with delay and error handling"""
        sleep(self.delay)
        try:
            response = self.inst.query(content)
        except pyvisa.errors.VisaIOError:
            return ""
        return response

    def close(self):
        self.inst.close()

class Chroma_63600_5(Equipment):
    """Chroma 63600-5 Series Electronic Load Controller"""

    def __init__(self, addr):
        super().__init__(addr)
        self._output_on = False
        self._current_output_load = 0.0
```

### Basic Connection Setup

```python
# Connect to the Chroma load
chroma = Chroma_63600_5('GPIB0::5::INSTR')
print(f"Connected to: {chroma.IDN}")
```

### Configure Parallel Operation

The `Chroma_63600_5` class includes a `parallel_init()` method for configuring parallel operation:

```python
class Chroma_63600_5(Equipment):
    # ... (previous code)

    def parallel_init(self, on_off='ON', master_chan=1, slave_list=[3]):
        """
        Initialize parallel operation mode.

        Args:
            on_off: 'ON' or 'OFF' to enable/disable parallel mode
            master_chan: Channel number for master (default 1)
            slave_list: List of slave channel numbers (default [3])
        """
        # Select and configure master channel
        self.write(f'CHANNEL {master_chan}')
        sleep(0.05)
        self.write(f'CONFIGURE:PARALLEL:INIT {on_off}')
        sleep(0.05)
        self.write(f'CONFIGURE:PARALLEL:MODE MASTER')
        sleep(0.05)

        # Configure each slave channel
        for i in slave_list:
            self.write(f'CHANNEL {i}')
            sleep(0.05)
            self.write(f'CONFIGURE:PARALLEL:INIT {on_off}')
            sleep(0.05)
            self.write(f'CONFIGURE:PARALLEL:MODE SLAVE')
            sleep(0.05)

        # Enable output on slave channel (example shows channel 3)
        self.output_on([3])

# Usage example
chroma = Chroma_63600_5('GPIB0::5::INSTR')
chroma.parallel_init(on_off='ON', master_chan=1, slave_list=[3])
```

**Key Implementation Notes:**
- Uses `CHANNEL` command for channel selection (not `CHAN:LOAD`)
- Uses `CONFIGURE:PARALLEL:INIT` (not `CONF:PARA:INIT`)
- Uses `CONFIGURE:PARALLEL:MODE` (not `CONF:PARA:MODE`)
- Includes 50ms delays between commands for device processing
- Single instrument object controls multiple channels within one mainframe

### Set Constant Current Mode

The driver implements two methods for setting constant current:

```python
class Chroma_63600_5(Equipment):
    # ... (previous code)

    def set_CC_current(self, load, channel=1, load_on=False):
        """
        Set constant current mode for a single channel.

        Args:
            load: Current value in amperes
            channel: Channel number (default 1)
            load_on: If True, turn on load after setting current
        """
        self.write(f'CHANNEL {channel}')
        sleep(0.05)
        self.write('MODE CCH')
        sleep(0.05)
        self.write(f"CURR:STAT:L2 {load}")  # Note: Uses L2, not L1
        self._current_output_load = load  # State tracking
        if load_on:
            self.output_on()

    def set_CC_current_multiple(self, load_list, channel_list=1, load_on=False):
        """
        Set constant current mode for multiple channels.

        Args:
            load_list: Dictionary/list of current values indexed by channel
            channel_list: List of channel numbers or single channel
            load_on: If True, turn on load after setting current
        """
        for channel in channel_list:
            self.write(f'CHANNEL {channel}')
            sleep(0.05)
            self.write('MODE CCH')
            sleep(0.05)
            self.write(f"CURR:STAT:L2 {load_list[channel]}")
            if load_on:
                self.output_on()

# Usage examples
chroma = Chroma_63600_5('GPIB0::5::INSTR')

# Single channel
chroma.set_CC_current(load=10.5, channel=1, load_on=True)

# Multiple channels
load_values = {1: 5.0, 3: 5.0}  # 5A on channels 1 and 3
chroma.set_CC_current_multiple(load_list=load_values,
                               channel_list=[1, 3],
                               load_on=True)
```

**Key Implementation Notes:**
- Uses `CURR:STAT:L2` instead of `CURR:STAT:L1` (static current level 2)
- Always selects channel before setting mode and current
- Includes state tracking with `_current_output_load`
- Fixed to CCH mode (modify for CCL/CCM if needed)

### Measurement Methods

The driver includes methods for measuring voltage, current, and power:

```python
class Chroma_63600_5(Equipment):
    # ... (previous code)

    def measure_voltage(self, channel=1):
        """Measure voltage on specified channel"""
        self.write(f'CHANNEL {channel}')
        sleep(0.05)
        return float(self.query('FETCH:voltage?'))

    def measure_current(self, channel=1):
        """Measure current on specified channel"""
        self.write(f'CHANNEL {channel}')
        sleep(0.05)
        return float(self.query('FETCH:CURRENT?'))

    def measure_power(self, channel=1):
        """Measure power on specified channel"""
        self.write(f'CHANNEL {channel}')
        sleep(0.05)
        return float(self.query('FETCH:POWER?'))

# Usage example
voltage = chroma.measure_voltage(channel=1)
current = chroma.measure_current(channel=1)
power = chroma.measure_power(channel=1)
print(f"Channel 1: {voltage}V, {current}A, {power}W")
```

**Key Implementation Notes:**
- Uses `FETCH:` commands (not `MEAS:`)
- Channel selection before each measurement
- Returns float values directly

### Load Output Control

The driver includes methods for controlling load output with state tracking:

```python
class Chroma_63600_5(Equipment):
    # ... (previous code)

    def output_on(self, channel_list=[1]):
        """
        Turn on load for specified channels.

        Args:
            channel_list: List of channel numbers to turn on (default [1])
        """
        for channel in channel_list:
            self.write(f'CHANNEL {channel}')
            sleep(0.05)
            self.write("LOAD ON")
            sleep(0.05)
        self._output_on = True

    def output_off(self, channel_list=[1]):
        """
        Turn off load for specified channels.

        Args:
            channel_list: List of channel numbers to turn off (default [1])
        """
        for channel in channel_list:
            self.write(f'CHANNEL {channel}')
            sleep(0.05)
            self.write("LOAD OFF")
            sleep(0.05)
        self._output_on = False

    def return_load_state(self, channel_list=[1]):
        """
        Query load state for specified channel.

        Returns:
            'OFF' or 'ON' for the specified channel
        """
        self.write(f'CHANNEL {channel_list}')
        sleep(0.05)
        return self.query(f'LOAD:STATE?')

# Usage examples
chroma = Chroma_63600_5('GPIB0::5::INSTR')

# Turn on multiple channels
chroma.output_on(channel_list=[1, 3])

# Check state
state = chroma.return_load_state(channel_list=1)
print(f"Channel 1 state: {state}")

# Turn off
chroma.output_off(channel_list=[1, 3])
```

**Key Implementation Notes:**
- Uses `LOAD ON` and `LOAD OFF` (not `LOAD:STAT ON/OFF`)
- Supports multiple channels via `channel_list` parameter
- Includes state tracking with `_output_on` boolean
- Channel selection before each load command

### Synchronized Load Activation for Parallel Operation

Using the driver's built-in methods, here's the complete workflow for parallel operation:

```python
from time import sleep

# Initialize connection
chroma = Chroma_63600_5('GPIB0::5::INSTR')

# Step 1: Configure parallel mode with master channel 1 and slave channel 3
chroma.parallel_init(on_off='ON', master_chan=1, slave_list=[3])

# Step 2: Set constant current on both channels
# For parallel operation, set the same current on each channel
load_values = {
    1: 10.0,  # Master: 10A
    3: 10.0   # Slave: 10A
}
chroma.set_CC_current_multiple(
    load_list=load_values,
    channel_list=[1, 3],
    load_on=False  # Don't turn on yet
)

# Alternative: Set individually
chroma.set_CC_current(load=10.0, channel=1, load_on=False)
chroma.set_CC_current(load=10.0, channel=3, load_on=False)

# Step 3: Turn on both channels (synchronized activation)
chroma.output_on(channel_list=[1, 3])

# Step 4: Monitor both channels
sleep(1.0)  # Allow settling time

master_voltage = chroma.measure_voltage(channel=1)
master_current = chroma.measure_current(channel=1)
master_power = chroma.measure_power(channel=1)

slave_voltage = chroma.measure_voltage(channel=3)
slave_current = chroma.measure_current(channel=3)
slave_power = chroma.measure_power(channel=3)

print(f"Master (Ch 1): {master_voltage:.2f}V, {master_current:.2f}A, {master_power:.2f}W")
print(f"Slave (Ch 3): {slave_voltage:.2f}V, {slave_current:.2f}A, {slave_power:.2f}W")
print(f"Total: {master_current + slave_current:.2f}A, {master_power + slave_power:.2f}W")

# Step 5: Turn off both channels
chroma.output_off(channel_list=[1, 3])

# Cleanup
chroma.close()
```

**Architecture Note:**
- This implementation uses **multi-channel control within a single mainframe**
- Channels 1 and 3 are within the same physical instrument
- Communication is through a single PyVISA connection
- Synchronization is handled by the instrument's internal firmware

### Complete Workflow Example

Here's a complete test function using the actual driver implementation:

```python
def parallel_load_test(visa_addr, master_chan, slave_chans, test_current_per_channel):
    """
    Complete example of parallel load operation using Chroma_63600_5 driver.

    Args:
        visa_addr: VISA address of the Chroma instrument
        master_chan: Master channel number
        slave_chans: List of slave channel numbers
        test_current_per_channel: Current per channel in amperes
    """
    # Initialize connection
    chroma = Chroma_63600_5(visa_addr)
    print(f"Connected to: {chroma.IDN}")

    try:
        # Step 1: Configure parallel operation
        print(f"\nConfiguring parallel operation...")
        print(f"  Master: Channel {master_chan}")
        print(f"  Slaves: Channels {slave_chans}")
        chroma.parallel_init(on_off='ON', master_chan=master_chan, slave_list=slave_chans)

        # Step 2: Set current on all channels
        all_channels = [master_chan] + slave_chans
        print(f"\nSetting current: {test_current_per_channel}A per channel")

        for channel in all_channels:
            chroma.set_CC_current(
                load=test_current_per_channel,
                channel=channel,
                load_on=False
            )

        # Step 3: Turn on all channels synchronously
        print(f"\nActivating channels {all_channels}...")
        chroma.output_on(channel_list=all_channels)

        # Step 4: Monitor all channels
        sleep(1.0)  # Allow settling
        print(f"\nMeasurements:")

        total_current = 0.0
        total_power = 0.0

        for channel in all_channels:
            voltage = chroma.measure_voltage(channel=channel)
            current = chroma.measure_current(channel=channel)
            power = chroma.measure_power(channel=channel)

            total_current += current
            total_power += power

            role = "Master" if channel == master_chan else "Slave"
            print(f"  Ch {channel} ({role}): {voltage:.2f}V, {current:.2f}A, {power:.2f}W")

        print(f"  Total: {total_current:.2f}A, {total_power:.2f}W")

        # Step 5: Check load states
        print(f"\nLoad states:")
        for channel in all_channels:
            state = chroma.return_load_state(channel_list=channel)
            print(f"  Ch {channel}: {state.strip()}")

        # Step 6: Turn off all channels
        print(f"\nDeactivating channels...")
        chroma.output_off(channel_list=all_channels)

        print(f"\nTest completed successfully")

    except Exception as e:
        print(f"\nError occurred: {e}")
        # Emergency shutdown
        try:
            chroma.output_off(channel_list=all_channels)
        except:
            pass
        raise

    finally:
        # Cleanup
        chroma.close()

# Usage example
if __name__ == '__main__':
    # Test with master channel 1, slave channel 3, 10A per channel
    parallel_load_test(
        visa_addr='GPIB0::5::INSTR',
        master_chan=1,
        slave_chans=[3],
        test_current_per_channel=10.0
    )
```

---

## Synchronization Methods

### Implementation Architecture

The current driver implementation uses **multi-channel parallel operation within a single mainframe**:

| Aspect | Implementation Details |
|--------|----------------------|
| **Architecture** | Single PyVISA connection controlling multiple channels |
| **Channels** | Master and slave channels within same instrument |
| **Synchronization** | Hardware-level synchronization via internal bus |
| **Communication** | Sequential channel selection with SCPI commands |
| **Timing** | Built-in delays (50ms) between commands |

### Timing Considerations

1. **Channel Selection**: Each operation requires channel selection with 50ms delay
2. **Mode Configuration**: 50ms delay after parallel mode initialization
3. **Load Activation**: Sequential activation with 50ms per channel
4. **Measurements**: Channel selection required before each measurement
5. **Settling Time**: 1-2 seconds recommended after load activation for stable readings

### Current Implementation vs. Alternatives

| Approach | Current Driver | Alternative (Separate Instruments) |
|----------|---------------|-----------------------------------|
| Connection | Single VISA address | Multiple VISA addresses |
| Channel Control | CHANNEL command | Separate instrument objects |
| Synchronization | Internal firmware | External (GPIB trigger or sequential) |
| Complexity | Lower | Higher |
| Scalability | Limited to channels in one mainframe | Can span multiple mainframes |

### Best Practices for Current Implementation

For the existing driver architecture:
1. **Always select channel** before operations (automatic in methods)
2. **Use channel lists** for coordinated operations
3. **Leverage parallel_init()** for proper master/slave setup
4. **Set current on all channels** before activation
5. **Monitor all channels** to verify balanced operation

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

## Extending the Current Driver

### Recommended Enhancements

The current `Chroma_63600_5` implementation can be extended with the following enhancements:

#### 1. Add Automatic Current Range Selection

```python
class Chroma_63600_5(Equipment):
    # ... (existing code)

    def mode_sel_auto(self, current):
        """
        Automatically select current mode based on current value.

        Args:
            current: Current in amperes

        Returns:
            Selected mode string ('CCL', 'CCM', or 'CCH')
        """
        # Adjust these thresholds based on your model specifications
        # Example for 63610-80-20: CCH=20A, CCM=2A, CCL=0.2A
        if current > 18.0:  # 90% of 20A max
            mode = "CCH"
        elif current > 1.8:  # 90% of 2A max
            mode = "CCM"
        else:
            mode = "CCL"

        self.mode_sel(mode)
        return mode

    def set_CC_current_auto(self, load, channel=1, load_on=False):
        """
        Set constant current with automatic range selection.

        Args:
            load: Current value in amperes
            channel: Channel number (default 1)
            load_on: If True, turn on load after setting current
        """
        self.write(f'CHANNEL {channel}')
        sleep(0.05)

        # Auto-select mode
        mode = self.mode_sel_auto(load)

        self.write(f"CURR:STAT:L2 {load}")
        self._current_output_load = load
        if load_on:
            self.output_on([channel])

        return mode
```

#### 2. Add Comprehensive Error Checking

```python
class Chroma_63600_5(Equipment):
    # ... (existing code)

    def check_errors(self):
        """
        Query and return system errors.

        Returns:
            Error string from instrument
        """
        return self.query('SYST:ERR?').strip()

    def verify_parallel_config(self, master_chan, slave_chans):
        """
        Verify parallel configuration is correct.

        Args:
            master_chan: Expected master channel
            slave_chans: List of expected slave channels

        Returns:
            bool: True if configuration is correct
        """
        # Check master
        self.write(f'CHANNEL {master_chan}')
        sleep(0.05)
        master_mode = self.query('CONFIGURE:PARALLEL:MODE?').strip()

        if 'MASTER' not in master_mode.upper():
            print(f"Warning: Channel {master_chan} is not configured as MASTER")
            return False

        # Check slaves
        for chan in slave_chans:
            self.write(f'CHANNEL {chan}')
            sleep(0.05)
            slave_mode = self.query('CONFIGURE:PARALLEL:MODE?').strip()

            if 'SLAVE' not in slave_mode.upper():
                print(f"Warning: Channel {chan} is not configured as SLAVE")
                return False

        return True
```

#### 3. Add Batch Measurement Capability

```python
class Chroma_63600_5(Equipment):
    # ... (existing code)

    def measure_all(self, channel_list):
        """
        Measure voltage, current, and power for multiple channels.

        Args:
            channel_list: List of channel numbers to measure

        Returns:
            Dictionary of measurements indexed by channel
        """
        measurements = {}

        for channel in channel_list:
            measurements[channel] = {
                'voltage': self.measure_voltage(channel),
                'current': self.measure_current(channel),
                'power': self.measure_power(channel)
            }

        return measurements

    def print_measurements(self, measurements, channel_roles=None):
        """
        Pretty-print measurements.

        Args:
            measurements: Dictionary from measure_all()
            channel_roles: Optional dict mapping channel to role ('Master'/'Slave')
        """
        total_current = 0.0
        total_power = 0.0

        for channel, data in measurements.items():
            role = f" ({channel_roles[channel]})" if channel_roles else ""
            print(f"Ch {channel}{role}: {data['voltage']:.2f}V, "
                  f"{data['current']:.2f}A, {data['power']:.2f}W")

            total_current += data['current']
            total_power += data['power']

        print(f"Total: {total_current:.2f}A, {total_power:.2f}W")
```

#### 4. Add Context Manager Support

```python
class Chroma_63600_5(Equipment):
    # ... (existing code)

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure loads are off"""
        try:
            # Query all channels and turn them off
            # This is a safety measure
            self.output_off(channel_list=[1, 2, 3, 4, 5])
        except:
            pass
        finally:
            self.close()

# Usage with context manager
with Chroma_63600_5('GPIB0::5::INSTR') as chroma:
    chroma.parallel_init(on_off='ON', master_chan=1, slave_list=[3])
    chroma.set_CC_current(load=10.0, channel=1, load_on=True)
    # ... test code ...
    # Automatically turns off and closes on exit
```

### Alternative Architecture: Multi-Instrument Control

For scenarios requiring control of **separate mainframes** (not just channels within one mainframe), you would need a different architecture:

**Key Differences:**
- Multiple VISA connections (one per mainframe)
- Master/slave configured on separate physical instruments
- External synchronization (GPIB trigger or sequential commands)
- Physical System Bus cable connections between mainframes

**Reference Implementation:**
See the GitHub examples section above, particularly:
- **Korrigan36/External-Ports**: Demonstrates parallel configuration with `CONF:PARA:MODE` commands
- Uses separate VISA instrument objects for master and slave
- Implements synchronization through sequential commands or GPIB triggers

**When to Use:**
- Need more channels than available in single mainframe
- Require higher total power than single mainframe supports
- Scaling beyond 10 channels (maximum per mainframe)

---

## Best Practices and Considerations

### 1. Hardware Setup

- **Cable quality**: Use high-quality system bus cables for reliable synchronization
- **Grounding**: Ensure proper grounding to avoid ground loops
- **Power ratings**: Verify combined power dissipation doesn't exceed chassis limits
- **Cooling**: Ensure adequate ventilation for parallel operation (higher power)

### 2. Software Configuration (Current Driver)

- **Use `parallel_init()` method** to configure master and slave channels
- **Set current on all channels** before turning on loads
- **Use channel lists** for coordinated operations (`output_on`, `output_off`)
- **Implement error checking** using `check_errors()` method (if extended as recommended)
- **Verify configuration** with `verify_parallel_config()` (if extended as recommended)

### 3. Current Distribution

- In **master/slave mode**, current should distribute evenly across parallel channels
- Monitor all channels using `measure_all()` to verify balanced operation
- If imbalance occurs, check:
  - Parallel configuration (master/slave roles)
  - Channel selection and mode settings
  - Load calibration
  - Physical connections (if multi-mainframe)

### 4. Timing Considerations (Current Driver)

- **Built-in delays**: Driver includes 50ms delays after channel selection and mode changes
- **Settling time**: Wait 1-2s after `output_on()` for stable measurements
- **Sequential operations**: Channel selection and command execution are sequential
- **Measurement timing**: Each `measure_*()` call includes channel selection delay

### 5. Safety

- **Over-current protection**: Set appropriate limits
- **Voltage limits**: Configure max voltage to protect DUT
- **Thermal monitoring**: Monitor load temperatures during high-power tests
- **Emergency stop**: Implement quick shutdown capability

### 6. Error Handling (Current Driver)

```python
def safe_parallel_load_test(visa_addr, master_chan, slave_chans, current):
    """Example with error handling using current driver"""
    chroma = None
    all_channels = [master_chan] + slave_chans

    try:
        # Initialize
        chroma = Chroma_63600_5(visa_addr)

        # Configure parallel
        chroma.parallel_init(on_off='ON',
                            master_chan=master_chan,
                            slave_list=slave_chans)

        # Verify configuration (if verify method is implemented)
        # if not chroma.verify_parallel_config(master_chan, slave_chans):
        #     raise RuntimeError("Parallel configuration verification failed")

        # Set current on all channels
        for channel in all_channels:
            chroma.set_CC_current(load=current,
                                channel=channel,
                                load_on=False)

        # Check for errors (if check_errors method is implemented)
        # errors = chroma.check_errors()
        # if 'No error' not in errors:
        #     raise RuntimeError(f"Load error: {errors}")

        # Turn on loads
        chroma.output_on(channel_list=all_channels)

        # Monitor
        sleep(1.0)
        measurements = {}
        for channel in all_channels:
            measurements[channel] = {
                'voltage': chroma.measure_voltage(channel),
                'current': chroma.measure_current(channel),
                'power': chroma.measure_power(channel)
            }

        # Verify balanced operation
        currents = [m['current'] for m in measurements.values()]
        if max(currents) - min(currents) > current * 0.1:  # >10% imbalance
            print("Warning: Current imbalance detected!")

        return measurements

    except pyvisa.errors.VisaIOError as e:
        print(f"VISA communication error: {e}")
        raise
    except Exception as e:
        print(f"Error occurred: {e}")
        raise
    finally:
        # Always turn off loads and cleanup
        if chroma is not None:
            try:
                chroma.output_off(channel_list=all_channels)
            except:
                pass
            try:
                chroma.close()
            except:
                pass
```

**Best Practice: Use Context Manager**

```python
# With context manager support (see extension recommendations)
with Chroma_63600_5('GPIB0::5::INSTR') as chroma:
    chroma.parallel_init(on_off='ON', master_chan=1, slave_list=[3])
    # Test code here...
    # Automatically handles cleanup and turns off loads on exit
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

The Chroma 63600 series electronic loads provide robust parallel operation capabilities through master/slave configuration. This report documents both the **current driver implementation** and alternative approaches for different use cases.

### Key Takeaways

1. **Current Driver Architecture**: Uses single PyVISA connection for multi-channel control
2. **Built-in Parallel Support**: `parallel_init()` method configures master/slave channels
3. **Hardware Synchronization**: Internal firmware handles channel synchronization
4. **Channel-Based Control**: All operations require channel selection before execution
5. **SCPI Commands**: Uses `CHANNEL`, `CONFIGURE:PARALLEL:*`, `CURR:STAT:L2`, `LOAD ON/OFF`

### Implementation Workflow (Current Driver)

For `set_CC_current` with parallel operation using the current driver:

1. **Initialize connection**: Create `Chroma_63600_5` instance with VISA address
2. **Configure parallel mode**: Call `parallel_init(on_off='ON', master_chan=1, slave_list=[3])`
3. **Set current on all channels**: Use `set_CC_current()` or `set_CC_current_multiple()`
4. **Turn on channels**: Call `output_on(channel_list=[1, 3])`
5. **Monitor channels**: Use `measure_voltage()`, `measure_current()`, `measure_power()`
6. **Turn off channels**: Call `output_off(channel_list=[1, 3])`

### Recommended Extensions

To enhance the driver for production use:
- Add automatic current range selection (`mode_sel_auto()`)
- Implement error checking (`check_errors()`, `verify_parallel_config()`)
- Add batch measurement capability (`measure_all()`)
- Include context manager support (`__enter__`, `__exit__`)

### Alternative Architectures

For **multi-mainframe parallel operation** (separate instruments):
- Requires multiple VISA connections
- Master/slave configured on separate physical instruments
- External synchronization (GPIB trigger or sequential commands)
- See GitHub examples (Korrigan36/External-Ports) for reference implementation

The current driver provides a solid foundation for single-mainframe multi-channel parallel operation, with clear extension points for enhanced functionality.

---

**Report Generated**: 2025-11-15
**Chroma 63600 Series**: DC Electronic Load
**Purpose**: Parallel Operation with PyVISA Synchronization
