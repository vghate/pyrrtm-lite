"""
Load SW spectroscopic coefficients from the coeffs.nc NetCDF file.

The file contains 14 groups (band_00 .. band_13) corresponding to Fortran
bands 16-29.  Each group may contain:

    ka        – lower-atmosphere absorption coefficients
    kb        – upper-atmosphere absorption coefficients (if present)
    selfref   – self-continuum reference (if present)
    forref    – foreign-continuum reference (if present)
    sfluxref  – solar flux reference spectrum
    rayl      – Rayleigh scattering coefficient (scalar; if present)

No chi_mls table is needed for SW — species mixing ratios used in taumol
are hardcoded there.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

_DEFAULT_PATH = str(Path(__file__).parent.parent / 'data' / 'sw_coeffs.nc')

_SCALAR_VARS = set()   # rayl handled dynamically (scalar or array per band)
_OPTIONAL_VARS = {"kb", "selfref", "forref", "rayl"}

# Variables stored in the NetCDF with (mg, ...) leading dimension
# that need to be transposed so mg is the LAST dimension.
# 'ka'      stored (mg, jp_a, jt, nspa)  → (nspa, jt, jp_a, mg)
# 'kb'      stored (mg, jp_b, jt, nspb)  → (nspb, jt, jp_b, mg)
# 'selfref' stored (mg, self_temp)        → (self_temp, mg)
# 'forref'  stored (mg, for_temp)         → (for_temp, mg)
# 'rayla'   stored (sflux_nspa, mg)       → (mg, sflux_nspa)  [special for band 8]
_TRANSPOSE_VARS = {"ka", "kb", "selfref"}
# Note: "forref" is NOT transposed here because extract_coeffs.py stores it as
# (for_temp, mg) which is already the shape taumol.py expects for forref[indf, :].

# sfluxref with shape (1, mg) should be squeezed to (mg,)
# sfluxref with shape (n, mg) where n>1 stays as (n, mg)
# rayla is stored (sflux_nspa, mg) → needed (mg, sflux_nspa): transpose


def load(path: str | None = None) -> list[dict]:
    """Load SW spectroscopic coefficients from coeffs.nc.

    Parameters
    ----------
    path : str or None
        Path to the NetCDF coefficient file.  Defaults to
        /home/vghate/Desktop/rrtm_sw_proj/data/coeffs.nc.

    Returns
    -------
    list of dict
        Length-14 list.  Index 0 corresponds to Fortran band 16, index 13
        to Fortran band 29.  Each dict contains the keys present in that
        band's NetCDF group:

            'ka'       – always present
            'kb'       – upper-atmosphere coefficients (if present)
            'selfref'  – self-continuum (if present)
            'forref'   – foreign-continuum (if present)
            'sfluxref' – solar flux reference; always present
            'rayl'     – Rayleigh coefficient as a Python float (if present)

        All array values are float64.
    """
    try:
        import netCDF4 as nc
    except ImportError as exc:
        raise ImportError(
            "netCDF4 is required to load SW coefficients. "
            "Install it with: pip install netCDF4"
        ) from exc

    if path is None:
        path = _DEFAULT_PATH

    bands: list[dict] = []

    with nc.Dataset(path, "r") as ds:
        for i in range(14):
            group_name = f"band_{i:02d}"
            grp = ds.groups[group_name]

            band_dict: dict = {}
            for var_name, var in grp.variables.items():
                data = var[:]  # masked array or ndarray
                # Convert masked arrays to plain ndarray
                if hasattr(data, "filled"):
                    data = data.filled(fill_value=np.nan)

                data = np.asarray(data, dtype=np.float64)

                if var_name in _SCALAR_VARS:
                    # (Currently empty set — kept for future use)
                    band_dict[var_name] = float(data.ravel()[0])

                elif var_name == "rayl":
                    # rayl is a scalar float (0-D) for most bands,
                    # but a per-g-point array (shape (mg,)) for bands 7,9,10,11.
                    if data.ndim == 0:
                        band_dict[var_name] = float(data)
                    else:
                        band_dict[var_name] = data  # shape (mg,)

                elif var_name in _TRANSPOSE_VARS:
                    # Stored as (mg, ...) in NetCDF; transpose to (..., mg) so
                    # that the last axis is always the g-point axis.
                    # ka:      (mg, jp_a, jt, nspa) → (nspa, jt, jp_a, mg)  [nspa>=1]
                    #       or (mg, jp_a, jt, 1)    → (jt, jp_a, mg)         [nspa=1, squeezed]
                    # kb:      (mg, jp_b, jt, nspb) → (nspb, jt, jp_b, mg)  [nspb>=1]
                    #       or (mg, jp_b, jt, 1)    → (jt, jp_b, mg)         [nspb=1, squeezed]
                    # selfref: (mg, self_temp)       → (self_temp, mg)
                    # forref:  (mg, for_temp)        → (for_temp, mg)
                    # Reverse all axes so leading mg goes to the end.
                    transposed = data.T
                    # For ka/kb: if the leading (species) dimension is 1, squeeze it.
                    if var_name in ("ka", "kb") and transposed.shape[0] == 1:
                        transposed = transposed[0]
                    band_dict[var_name] = transposed

                elif var_name == "sfluxref":
                    # Stored as (sflux_nspa, mg).
                    # If sflux_nspa == 1, squeeze to (mg,) for taumol functions
                    # that expect a 1-D sfluxref.
                    if data.shape[0] == 1:
                        band_dict[var_name] = data[0, :]   # (mg,)
                    else:
                        band_dict[var_name] = data          # (sflux_nspa, mg)

                elif var_name == "rayla":
                    # Stored as (sflux_nspa, mg) → needed as (mg, sflux_nspa)
                    band_dict[var_name] = data.T

                else:
                    band_dict[var_name] = data

            bands.append(band_dict)

    return bands
