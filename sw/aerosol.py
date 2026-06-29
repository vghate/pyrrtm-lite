"""SW aerosol optical properties.

Each atmospheric layer carries three spectrally resolved quantities
for the 14 RRTM-SW bands:
    tau_ext : total extinction optical depth
    ssa     : single-scattering albedo
    g       : asymmetry parameter
"""

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class AerosolLayerSW:
    tau_ext: np.ndarray = field(default_factory=lambda: np.zeros(14))
    ssa:     np.ndarray = field(default_factory=lambda: np.zeros(14))
    g:       np.ndarray = field(default_factory=lambda: np.zeros(14))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_aerosol(nlayers: int) -> list[AerosolLayerSW]:
    """Return a list of *nlayers* zero-filled AerosolLayerSW objects."""
    return [AerosolLayerSW() for _ in range(nlayers)]


# ---------------------------------------------------------------------------
# Spectral scaling tables — all 14 SW bands, Python 0-indexed
# ---------------------------------------------------------------------------

DUST_SW = dict(
    tau_scale=np.array([0.10, 0.12, 0.15, 0.18, 0.22, 0.28, 0.35, 0.45,
                        0.55, 0.70, 0.85, 0.95, 0.98, 0.08]),
    ssa=np.array([0.80, 0.82, 0.85, 0.87, 0.88, 0.90, 0.91, 0.92,
                  0.93, 0.94, 0.95, 0.95, 0.95, 0.75]),
    g=np.array([0.65, 0.67, 0.68, 0.69, 0.70, 0.71, 0.72, 0.73,
                0.74, 0.74, 0.74, 0.73, 0.72, 0.60]),
)

SEA_SALT_SW = dict(
    tau_scale=np.array([0.08, 0.10, 0.12, 0.14, 0.18, 0.22, 0.28, 0.40,
                        0.55, 0.72, 0.88, 0.96, 0.99, 0.06]),
    ssa=np.array([0.97, 0.97, 0.98, 0.98, 0.98, 0.99, 0.99, 0.99,
                  0.99, 0.99, 0.99, 0.99, 0.99, 0.96]),
    g=np.array([0.70, 0.71, 0.72, 0.72, 0.73, 0.73, 0.74, 0.74,
                0.74, 0.74, 0.73, 0.72, 0.71, 0.65]),
)

URBAN_SW = dict(
    tau_scale=np.array([0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.25,
                        0.38, 0.55, 0.75, 0.90, 0.97, 0.03]),
    ssa=np.array([0.80, 0.82, 0.84, 0.86, 0.88, 0.89, 0.90, 0.91,
                  0.92, 0.93, 0.94, 0.95, 0.95, 0.75]),
    g=np.array([0.62, 0.63, 0.64, 0.65, 0.66, 0.66, 0.67, 0.67,
                0.67, 0.66, 0.65, 0.64, 0.63, 0.55]),
)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _layer_mask(pz: np.ndarray, z_base_km: float, z_top_km: float) -> np.ndarray:
    """Boolean mask: True for layers whose midpoint falls in [z_base, z_top].

    Parameters
    ----------
    pz:
        Pressure at layer *interfaces* in hPa, length nlayers+1.
    z_base_km, z_top_km:
        Altitude bounds in km.

    Returns
    -------
    Boolean array of length nlayers.
    """
    p_mid = (pz[:-1] + pz[1:]) / 2.0
    z_mid = 44.3 * (1.0 - (p_mid / 1013.25) ** 0.190)
    return (z_mid >= z_base_km) & (z_mid <= z_top_km)


def _build_optics(
    aod_550: float,
    z_base_km: float,
    z_top_km: float,
    pz: np.ndarray,
    table: dict,
) -> list[AerosolLayerSW]:
    """Generic builder used by all three aerosol-type helpers.

    The total AOD at 550 nm is distributed uniformly among layers whose
    pressure-level midpoint falls within [z_base_km, z_top_km].  For each
    such layer the per-band optical depth is scaled by *tau_scale*, while
    *ssa* and *g* are taken directly from the spectral table.  Layers
    outside the range are left at zero.
    """
    nlayers = len(pz) - 1
    aerosol = build_aerosol(nlayers)

    mask = _layer_mask(pz, z_base_km, z_top_km)
    n_in_range = int(mask.sum())
    if n_in_range == 0:
        return aerosol

    tau_layer = aod_550 / n_in_range  # AOD per layer at 550 nm

    for i in range(nlayers):
        if mask[i]:
            aerosol[i].tau_ext = tau_layer * table["tau_scale"]
            aerosol[i].ssa     = table["ssa"].copy()
            aerosol[i].g       = table["g"].copy()

    return aerosol


# ---------------------------------------------------------------------------
# Convenience builders
# ---------------------------------------------------------------------------

def dust_optics(
    aod_550: float,
    z_base_km: float,
    z_top_km: float,
    pz: np.ndarray,
) -> list[AerosolLayerSW]:
    """Mineral-dust aerosol optical properties for all SW bands.

    Parameters
    ----------
    aod_550:
        Aerosol optical depth at 550 nm (column-integrated).
    z_base_km, z_top_km:
        Bottom and top of the dust layer in km.
    pz:
        Interface pressure array (hPa), length nlayers+1.

    Returns
    -------
    List of AerosolLayerSW, one per atmospheric layer.
    """
    return _build_optics(aod_550, z_base_km, z_top_km, pz, DUST_SW)


def sea_salt_optics(
    aod_550: float,
    z_base_km: float,
    z_top_km: float,
    pz: np.ndarray,
) -> list[AerosolLayerSW]:
    """Sea-salt aerosol optical properties for all SW bands.

    Parameters
    ----------
    aod_550:
        Aerosol optical depth at 550 nm (column-integrated).
    z_base_km, z_top_km:
        Bottom and top of the sea-salt layer in km.
    pz:
        Interface pressure array (hPa), length nlayers+1.

    Returns
    -------
    List of AerosolLayerSW, one per atmospheric layer.
    """
    return _build_optics(aod_550, z_base_km, z_top_km, pz, SEA_SALT_SW)


def urban_optics(
    aod_550: float,
    z_base_km: float,
    z_top_km: float,
    pz: np.ndarray,
) -> list[AerosolLayerSW]:
    """Urban / pollution aerosol optical properties for all SW bands.

    Parameters
    ----------
    aod_550:
        Aerosol optical depth at 550 nm (column-integrated).
    z_base_km, z_top_km:
        Bottom and top of the urban aerosol layer in km.
    pz:
        Interface pressure array (hPa), length nlayers+1.

    Returns
    -------
    List of AerosolLayerSW, one per atmospheric layer.
    """
    return _build_optics(aod_550, z_base_km, z_top_km, pz, URBAN_SW)
