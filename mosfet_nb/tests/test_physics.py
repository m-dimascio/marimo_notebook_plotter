"""Tests for MOSFET physics calculations."""

import pytest
import numpy as np
import sys
sys.path.insert(0, "../src")

from mosfet_sim import (
    MOSFETParams,
    threshold_voltage,
    drain_current,
    oxide_capacitance,
)


@pytest.fixture
def default_device():
    return MOSFETParams.default_device()


def test_threshold_voltage_positive(default_device):
    """Threshold voltage should be positive for NMOS with p-substrate."""
    vth = threshold_voltage(default_device)
    assert vth > 0


def test_drain_current_cutoff(default_device):
    """Drain current should be ~0 below threshold."""
    vth = threshold_voltage(default_device)
    id_off = drain_current(default_device, vgs=vth - 0.5, vds=1.0)
    assert id_off < 1e-12


def test_drain_current_increases_with_vgs(default_device):
    """Drain current should increase with Vgs."""
    vth = threshold_voltage(default_device)
    id_low = drain_current(default_device, vgs=vth + 0.5, vds=2.0)
    id_high = drain_current(default_device, vgs=vth + 1.5, vds=2.0)
    assert id_high > id_low


def test_saturation_current_constant(default_device):
    """In saturation, Id should be relatively constant with Vds."""
    vth = threshold_voltage(default_device)
    vgs = vth + 1.0
    id_sat1 = drain_current(default_device, vgs=vgs, vds=3.0)
    id_sat2 = drain_current(default_device, vgs=vgs, vds=4.0)
    # Should be equal in ideal square-law model
    assert np.isclose(id_sat1, id_sat2, rtol=0.01)


def test_oxide_capacitance_positive(default_device):
    """Oxide capacitance should be positive."""
    cox = oxide_capacitance(default_device)
    assert cox > 0


def test_oxide_capacitance_increases_with_thinner_oxide():
    """Oxide capacitance should increase as oxide gets thinner."""
    device_thick = MOSFETParams(
        channel_length=1e-6,
        channel_width=10e-6,
        oxide_thickness=20e-9,
        substrate_doping=1e17,
        source_drain_doping=1e20,
    )
    device_thin = MOSFETParams(
        channel_length=1e-6,
        channel_width=10e-6,
        oxide_thickness=10e-9,
        substrate_doping=1e17,
        source_drain_doping=1e20,
    )
    cox_thick = oxide_capacitance(device_thick)
    cox_thin = oxide_capacitance(device_thin)
    assert cox_thin > cox_thick
