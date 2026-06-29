"""aerosol.py — aerosol optical property interface for RRTM_LW.

Longwave aerosol absorption optical depths per RRTM LW band (16 bands, 0-indexed).
Scattering is neglected in the longwave; only absorption optical depth is tracked.
"""

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class AerosolLayer:
    """Per-layer aerosol optical properties for RRTM LW.

    Attributes
    ----------
    tau_abs : np.ndarray, shape (16,)
        Absorption optical depth for each of the 16 RRTM LW bands (0-indexed).
        Scattering is neglected in the longwave.
    """
    tau_abs: np.ndarray = field(default_factory=lambda: np.zeros(16))


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_aerosol(nlayers: int) -> list:
    """Return a list of *nlayers* AerosolLayer objects, all tau_abs=0.

    Parameters
    ----------
    nlayers : int
        Number of atmospheric layers.

    Returns
    -------
    list[AerosolLayer]
        Length-nlayers list with every tau_abs initialised to zero.
    """
    return [AerosolLayer() for _ in range(nlayers)]


# ---------------------------------------------------------------------------
# Spectral scaling coefficients
# ---------------------------------------------------------------------------

# Mineral dust LW absorption relative to 550 nm AOD.
# Literature-based estimates following Sokolik & Toon (1999).
# Shape (16,), 0-indexed bands covering roughly 10–3250 cm^-1.
DUST_LW_SCALE = np.array([
    # bands 0–3  (~10–700 cm^-1)
    0.08, 0.12, 0.18, 0.22,
    # bands 4–7  (~700–1180 cm^-1)
    0.30, 0.45, 0.38, 0.25,
    # bands 8–11 (~1180–2080 cm^-1)
    0.18, 0.10, 0.07, 0.04,
    # bands 12–15 (~2080–3250 cm^-1)
    0.03, 0.02, 0.02, 0.01,
], dtype=np.float64)

# Sea salt LW absorption relative to 550 nm AOD.
# Similar spectral shape to dust but stronger at low wavenumbers.
SEA_SALT_LW_SCALE = np.array([
    # bands 0–3
    0.10, 0.15, 0.22, 0.28,
    # bands 4–7
    0.35, 0.50, 0.42, 0.28,
    # bands 8–11
    0.20, 0.11, 0.08, 0.05,
    # bands 12–15
    0.04, 0.03, 0.02, 0.01,
], dtype=np.float64)

# Sulfate/urban aerosol LW absorption relative to 550 nm AOD.
# Mostly transparent in the longwave; small absorption values.
URBAN_LW_SCALE = np.array([
    # bands 0–3
    0.02, 0.03, 0.04, 0.05,
    # bands 4–7
    0.06, 0.08, 0.06, 0.04,
    # bands 8–11
    0.03, 0.02, 0.01, 0.01,
    # bands 12–15
    0.005, 0.003, 0.002, 0.001,
], dtype=np.float64)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _layer_mask(pz: np.ndarray, z_base_km: float, z_top_km: float) -> np.ndarray:
    """Return boolean array (nlayers,) True where layer midpoint is in [z_base, z_top].

    Parameters
    ----------
    pz : np.ndarray, shape (nlayers+1,)
        Pressure level edges in millibars (mb), top-of-atmosphere to surface
        or surface to TOA ordering both work — midpoints are computed from
        adjacent edges.
    z_base_km : float
        Lower altitude bound of the aerosol layer in kilometres.
    z_top_km : float
        Upper altitude bound of the aerosol layer in kilometres.

    Returns
    -------
    np.ndarray of bool, shape (nlayers,)
        True for each layer whose pressure-derived midpoint altitude falls
        within [z_base_km, z_top_km].

    Notes
    -----
    Altitude is estimated via the hypsometric approximation:
        z_km ≈ 44.3 * log(1013.25 / p_mb)
    """
    p_mid = (pz[:-1] + pz[1:]) / 2.0
    z_mid = 44.3 * (1.0 - (p_mid / 1013.25) ** 0.190)
    return (z_mid >= z_base_km) & (z_mid <= z_top_km)


def _distribute_aod(
    aod_550: float,
    z_base_km: float,
    z_top_km: float,
    pz: np.ndarray,
    scale: np.ndarray,
) -> list:
    """Distribute column AOD uniformly across layers in an altitude range.

    Parameters
    ----------
    aod_550 : float
        Column aerosol optical depth at 550 nm.
    z_base_km : float
        Base of the aerosol layer in km.
    z_top_km : float
        Top of the aerosol layer in km.
    pz : np.ndarray, shape (nlayers+1,)
        Pressure level edges in mb.
    scale : np.ndarray, shape (16,)
        Per-band scaling factors relative to 550 nm AOD.

    Returns
    -------
    list[AerosolLayer]
        Length len(pz)-1 list; layers outside the altitude range have tau_abs=0.
    """
    nlayers = len(pz) - 1
    mask = _layer_mask(pz, z_base_km, z_top_km)
    n_in_range = int(mask.sum())

    layers = [AerosolLayer() for _ in range(nlayers)]

    if n_in_range == 0:
        return layers

    tau_per_layer = aod_550 / n_in_range  # uniform distribution
    for i in range(nlayers):
        if mask[i]:
            layers[i].tau_abs = tau_per_layer * scale

    return layers


# ---------------------------------------------------------------------------
# Convenience builders
# ---------------------------------------------------------------------------

def dust_optics(
    aod_550: float,
    z_base_km: float,
    z_top_km: float,
    pz: np.ndarray,
) -> list:
    """Build mineral-dust aerosol optical properties.

    Distributes *aod_550* uniformly across layers whose midpoint altitude
    falls within [z_base_km, z_top_km], scaled to each RRTM LW band using
    DUST_LW_SCALE.

    Parameters
    ----------
    aod_550 : float
        Column aerosol optical depth at 550 nm.
    z_base_km : float
        Base of the dust layer in km.
    z_top_km : float
        Top of the dust layer in km.
    pz : np.ndarray, shape (nlayers+1,)
        Pressure level edges in mb.

    Returns
    -------
    list[AerosolLayer]
        Length len(pz)-1; layers outside the range have tau_abs=0.
    """
    return _distribute_aod(aod_550, z_base_km, z_top_km, pz, DUST_LW_SCALE)


def sea_salt_optics(
    aod_550: float,
    z_base_km: float,
    z_top_km: float,
    pz: np.ndarray,
) -> list:
    """Build sea-salt aerosol optical properties.

    Distributes *aod_550* uniformly across layers whose midpoint altitude
    falls within [z_base_km, z_top_km], scaled to each RRTM LW band using
    SEA_SALT_LW_SCALE.

    Parameters
    ----------
    aod_550 : float
        Column aerosol optical depth at 550 nm.
    z_base_km : float
        Base of the sea-salt layer in km.
    z_top_km : float
        Top of the sea-salt layer in km.
    pz : np.ndarray, shape (nlayers+1,)
        Pressure level edges in mb.

    Returns
    -------
    list[AerosolLayer]
        Length len(pz)-1; layers outside the range have tau_abs=0.
    """
    return _distribute_aod(aod_550, z_base_km, z_top_km, pz, SEA_SALT_LW_SCALE)


def urban_optics(
    aod_550: float,
    z_base_km: float,
    z_top_km: float,
    pz: np.ndarray,
) -> list:
    """Build sulfate/urban aerosol optical properties.

    Distributes *aod_550* uniformly across layers whose midpoint altitude
    falls within [z_base_km, z_top_km], scaled to each RRTM LW band using
    URBAN_LW_SCALE.

    Parameters
    ----------
    aod_550 : float
        Column aerosol optical depth at 550 nm.
    z_base_km : float
        Base of the urban aerosol layer in km.
    z_top_km : float
        Top of the urban aerosol layer in km.
    pz : np.ndarray, shape (nlayers+1,)
        Pressure level edges in mb.

    Returns
    -------
    list[AerosolLayer]
        Length len(pz)-1; layers outside the range have tau_abs=0.
    """
    return _distribute_aod(aod_550, z_base_km, z_top_km, pz, URBAN_LW_SCALE)
