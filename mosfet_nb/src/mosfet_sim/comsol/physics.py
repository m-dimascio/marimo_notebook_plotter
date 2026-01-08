"""Semiconductor physics setup for COMSOL MOSFET model.

This module configures the drift-diffusion semiconductor physics
in COMSOL, including:
- Semiconductor material properties (Si)
- Doping profiles (p-type body, n+ S/D)
- Metal contacts (Source, Drain, Gate, Body)
- Insulator domain (SiO2 oxide)

The physics interface solves:
- Poisson's equation for electrostatics
- Drift-diffusion equations for electron/hole transport
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mph

from ..device import MOSFETParams

logger = logging.getLogger(__name__)


def setup_semiconductor_physics(
    model: "mph.Model",
    params: MOSFETParams,
    use_finite_volume: bool = True
) -> None:
    """
    Configure drift-diffusion semiconductor physics.

    This sets up COMSOL's Semiconductor interface which solves:
    - Poisson's equation: ∇·(ε∇V) = -ρ
    - Electron continuity: ∂n/∂t = (1/q)∇·Jn - R
    - Hole continuity: ∂p/∂t = -(1/q)∇·Jp - R

    Where Jn, Jp are drift-diffusion currents.

    Args:
        model: COMSOL model instance.
        params: MOSFET device parameters.
        use_finite_volume: If True, use finite volume discretization
                          (more stable for semiconductors).
    """
    java = model.java
    logger.info("Setting up semiconductor physics...")

    # =========================================================================
    # CREATE SEMICONDUCTOR PHYSICS INTERFACE
    # =========================================================================
    phys = java.physics().create("semi", "Semiconductor", "geom1")
    phys.label("Semiconductor Physics")

    # Set formulation options
    phys.prop("EquationFormulation").set("formulation", "DriftDiffusion")

    if use_finite_volume:
        phys.prop("SolverOptions").set("discretization", "FiniteVolume")
    else:
        phys.prop("SolverOptions").set("discretization", "FiniteElement")

    # Use log formulation for better convergence with large concentration ratios
    phys.prop("SolverOptions").set("nlogscale", True)
    phys.prop("SolverOptions").set("plogscale", True)

    # =========================================================================
    # MATERIAL PROPERTIES - SILICON
    # =========================================================================
    # Apply to semiconductor domains (substrate, source, drain)
    _setup_silicon_material(phys, params)

    # =========================================================================
    # DOPING PROFILES
    # =========================================================================
    _setup_doping_profiles(phys, params)

    # =========================================================================
    # METAL CONTACTS
    # =========================================================================
    _setup_contacts(phys, params)

    # =========================================================================
    # INSULATOR DOMAIN (OXIDE)
    # =========================================================================
    _setup_oxide_domain(phys, params)

    logger.info("Semiconductor physics configured successfully")


def _setup_silicon_material(phys, params: MOSFETParams) -> None:
    """Configure silicon semiconductor material properties."""

    # Default semiconductor material model (applies to all semiconductor domains)
    # This sets up the basic material properties

    # Access the default domain feature and modify it
    mat = phys.feature("scsm1")  # Default semiconductor material
    mat.label("Silicon Material")

    # Material properties from params
    eps_r = params.substrate.epsilon_r  # 11.7 for Si
    ni = params.substrate.ni  # 1.5e10 cm^-3 at 300K
    mu_n = params.substrate.mu_n  # 1400 cm^2/V·s
    mu_p = params.substrate.mu_p  # 450 cm^2/V·s
    Eg = params.substrate.bandgap  # 1.12 eV
    T = params.temperature  # 300 K

    # Set material properties
    mat.set("epsilonr", eps_r)
    mat.set("ni", f"{ni}[1/cm^3]")
    mat.set("mun", f"{mu_n}[cm^2/(V*s)]")
    mat.set("mup", f"{mu_p}[cm^2/(V*s)]")
    mat.set("Eg", f"{Eg}[eV]")
    mat.set("T", f"{T}[K]")

    # Use Maxwell-Boltzmann statistics (simpler than Fermi-Dirac)
    mat.set("CarrierStatistics", "MaxwellBoltzmann")

    # Band structure
    mat.set("chi", "4.05[V]")  # Electron affinity for Si

    logger.debug(f"Silicon material: εr={eps_r}, ni={ni:.2e} cm⁻³, μn={mu_n} cm²/V·s")


def _setup_doping_profiles(phys, params: MOSFETParams) -> None:
    """Configure doping profiles for different regions."""

    Na = params.substrate_doping      # Acceptor concentration (body)
    Nd = params.source_drain_doping   # Donor concentration (S/D)

    # -------------------------------------------------------------------------
    # P-TYPE SUBSTRATE DOPING (Acceptors)
    # -------------------------------------------------------------------------
    # This applies uniformly to the substrate body
    dop_body = phys.feature().create("dop_body", "AnalyticDopingModel")
    dop_body.label("Body Doping (p-type)")
    dop_body.selection().all()  # Apply to all, will be overridden by S/D

    dop_body.set("DopingType", "Acceptor")
    dop_body.set("NA0", f"{Na}[1/cm^3]")
    dop_body.set("DopantDistribution", "Uniform")

    # -------------------------------------------------------------------------
    # N+ SOURCE DOPING (Donors)
    # -------------------------------------------------------------------------
    dop_source = phys.feature().create("dop_source", "AnalyticDopingModel")
    dop_source.label("Source Doping (n+)")
    dop_source.selection().named("geom1_source_dom")

    dop_source.set("DopingType", "Donor")
    dop_source.set("ND0", f"{Nd}[1/cm^3]")
    dop_source.set("DopantDistribution", "Uniform")

    # -------------------------------------------------------------------------
    # N+ DRAIN DOPING (Donors)
    # -------------------------------------------------------------------------
    dop_drain = phys.feature().create("dop_drain", "AnalyticDopingModel")
    dop_drain.label("Drain Doping (n+)")
    dop_drain.selection().named("geom1_drain_dom")

    dop_drain.set("DopingType", "Donor")
    dop_drain.set("ND0", f"{Nd}[1/cm^3]")
    dop_drain.set("DopantDistribution", "Uniform")

    logger.debug(f"Doping: Na={Na:.2e} cm⁻³ (body), Nd={Nd:.2e} cm⁻³ (S/D)")


def _setup_contacts(phys, params: MOSFETParams) -> None:
    """Configure metal contacts for device terminals."""

    # -------------------------------------------------------------------------
    # SOURCE CONTACT (Grounded reference)
    # -------------------------------------------------------------------------
    src_contact = phys.feature().create("src_contact", "MetalContact")
    src_contact.label("Source Contact")
    src_contact.selection().named("sel_source_contact")

    src_contact.set("V0", "0[V]")  # Reference ground
    src_contact.set("ContactType", "OhmicContact")

    # -------------------------------------------------------------------------
    # DRAIN CONTACT (Vds applied)
    # -------------------------------------------------------------------------
    drn_contact = phys.feature().create("drn_contact", "MetalContact")
    drn_contact.label("Drain Contact")
    drn_contact.selection().named("sel_drain_contact")

    drn_contact.set("V0", "Vds")  # Parameter-controlled
    drn_contact.set("ContactType", "OhmicContact")

    # -------------------------------------------------------------------------
    # BODY CONTACT (Vbs bias, typically 0 or negative)
    # -------------------------------------------------------------------------
    body_contact = phys.feature().create("body_contact", "MetalContact")
    body_contact.label("Body Contact")
    body_contact.selection().named("sel_body_contact")

    body_contact.set("V0", "Vbs")  # Parameter-controlled
    body_contact.set("ContactType", "OhmicContact")

    # -------------------------------------------------------------------------
    # GATE CONTACT
    # -------------------------------------------------------------------------
    # Gate is special - it's a MOS gate, not a simple metal contact
    gate = phys.feature().create("gate_contact", "ThinInsulatorGate")
    gate.label("Gate Contact")
    gate.selection().named("sel_channel_interface")

    gate.set("V0", "Vgs")  # Parameter-controlled
    gate.set("d_ins", f"{params.oxide_thickness}[m]")
    gate.set("epsilonr_ins", params.oxide.epsilon_r)

    logger.debug("Contacts configured: Source (0V), Drain (Vds), Body (Vbs), Gate (Vgs)")


def _setup_oxide_domain(phys, params: MOSFETParams) -> None:
    """Configure the gate oxide as an insulator domain."""

    # For simple models, the oxide is handled by the ThinInsulatorGate
    # For detailed oxide physics, we would add an InsulatorDomain

    # The ThinInsulatorGate feature already accounts for oxide capacitance
    # via the thickness and permittivity settings

    # If we need explicit oxide domain physics:
    # oxide_dom = phys.feature().create("oxide_dom", "InsulatorDomain")
    # oxide_dom.selection().named("geom1_oxide_dom")
    # oxide_dom.set("epsilonr", params.oxide.epsilon_r)

    logger.debug(f"Oxide: tox={params.oxide_thickness*1e9:.1f}nm, εr={params.oxide.epsilon_r}")


def setup_global_parameters(
    model: "mph.Model",
    vgs: float = 0.0,
    vds: float = 0.0,
    vbs: float = 0.0
) -> None:
    """
    Set up global voltage parameters.

    These parameters can be swept in parametric studies.

    Args:
        model: COMSOL model instance.
        vgs: Initial gate-source voltage [V].
        vds: Initial drain-source voltage [V].
        vbs: Initial body-source voltage [V].
    """
    java = model.java

    java.param().set("Vgs", f"{vgs}[V]", "Gate-source voltage")
    java.param().set("Vds", f"{vds}[V]", "Drain-source voltage")
    java.param().set("Vbs", f"{vbs}[V]", "Body-source voltage")

    logger.debug(f"Parameters set: Vgs={vgs}V, Vds={vds}V, Vbs={vbs}V")


def update_bias_point(
    model: "mph.Model",
    vgs: float,
    vds: float,
    vbs: float = 0.0
) -> None:
    """
    Update bias voltages for a new operating point.

    Args:
        model: COMSOL model instance.
        vgs: Gate-source voltage [V].
        vds: Drain-source voltage [V].
        vbs: Body-source voltage [V].
    """
    java = model.java

    java.param().set("Vgs", f"{vgs}[V]")
    java.param().set("Vds", f"{vds}[V]")
    java.param().set("Vbs", f"{vbs}[V]")


def add_mobility_model(
    model: "mph.Model",
    model_type: str = "constant"
) -> None:
    """
    Add carrier mobility model.

    Args:
        model: COMSOL model instance.
        model_type: Type of mobility model:
            - "constant": Fixed mobility (default)
            - "arora": Doping-dependent (Arora model)
            - "lombardi": Field-dependent (surface mobility)
    """
    java = model.java
    phys = java.physics("semi")

    if model_type == "constant":
        # Already set in material properties
        logger.debug("Using constant mobility model")

    elif model_type == "arora":
        # Doping-dependent mobility
        mob = phys.feature().create("mob_arora", "MobilityModel")
        mob.set("MobilityModel", "Arora")
        logger.debug("Added Arora doping-dependent mobility model")

    elif model_type == "lombardi":
        # Surface/field-dependent mobility (good for MOSFETs)
        mob = phys.feature().create("mob_lombardi", "MobilityModel")
        mob.set("MobilityModel", "Lombardi")
        logger.debug("Added Lombardi surface mobility model")

    else:
        logger.warning(f"Unknown mobility model: {model_type}")
