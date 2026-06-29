"""
cf_reader.py
------------
Read a CF-convention NetCDF sounding file and return an SWSounding object.

Identical to the LW cf_reader except it ALSO detects SW-specific quantities:

  | Quantity          | CF standard_name      | Fallbacks             | Units   |
  |-------------------|-----------------------|-----------------------|---------|
  | Solar zenith angle| solar_zenith_angle    | sza, zenith_angle     | degrees |
  | Surface albedo    | surface_albedo        | albedo, alb           | [0-1]   |
  | Julian day        | global attr julian_day| attr julday           | int 1-365|

Rules
-----
- solar_zenith_angle : REQUIRED. Raises ValueError if absent or all NaN.
- surface_albedo     : optional. Default 0.1 with UserWarning if absent.
- julian_day         : optional. Default 1 (no Earth-Sun correction) with UserWarning.

Surface albedo may be:
  - scalar              : applied to all 14 SW bands
  - per-band (14 values): used directly
  - absent              : 0.1 for all bands

IEMIS=0, IREFLECT=0 are fixed; no semiss parameter.

Public API
----------
    from .sounding import SWSounding
    def read_cf_sounding(path: str, iceflag: int = None) -> SWSounding:
        Opens NetCDF, detects variables, converts units, returns SWSounding.
"""

import warnings
import numpy as np
import netCDF4 as nc


# ---------------------------------------------------------------------------
# Look-up tables: CF standard_name / fallback short names
# (identical to LW plus SW-specific entries)
# ---------------------------------------------------------------------------

_VAR_TABLE = {
    "height": {
        "standard_names": {"height", "altitude"},
        "fallbacks": ["z", "alt", "height_m"],
    },
    "pressure": {
        "standard_names": {"air_pressure"},
        "fallbacks": ["pres", "pressure", "p", "lev"],
    },
    "temperature": {
        "standard_names": {"air_temperature"},
        "fallbacks": ["temp", "T", "ta"],
    },
    "wv_mixr": {
        "standard_names": {"humidity_mixing_ratio"},
        "fallbacks": ["wv", "q_wv", "mixr"],
    },
    "co2": {
        "standard_names": {"mole_fraction_of_carbon_dioxide_in_air"},
        "fallbacks": ["co2"],
    },
    "o3": {
        "standard_names": {"mole_fraction_of_ozone_in_air"},
        "fallbacks": ["o3"],
    },
    "n2o": {
        "standard_names": {"mole_fraction_of_nitrous_oxide_in_air"},
        "fallbacks": ["n2o"],
    },
    "co": {
        "standard_names": {"mole_fraction_of_carbon_monoxide_in_air"},
        "fallbacks": ["co"],
    },
    "ch4": {
        "standard_names": {"mole_fraction_of_methane_in_air"},
        "fallbacks": ["ch4"],
    },
    "surface_temperature": {
        "standard_names": {"surface_temperature"},
        "fallbacks": ["tsk", "skt", "t_skin", "tsfc"],
    },
    "cloud_fraction": {
        "standard_names": {"cloud_area_fraction_in_atmosphere_layer"},
        "fallbacks": ["cf", "cldfrac", "cloud_fraction"],
    },
    "lwp": {
        "standard_names": {"atmosphere_mass_content_of_cloud_liquid_water"},
        "fallbacks": ["lwp"],
    },
    "iwp": {
        "standard_names": {"atmosphere_mass_content_of_cloud_ice"},
        "fallbacks": ["iwp"],
    },
    "re_liq": {
        "standard_names": {"effective_radius_of_cloud_liquid_water_particle"},
        "fallbacks": ["re_liq", "rel"],
    },
    "re_ice": {
        "standard_names": {"effective_radius_of_cloud_ice_particle"},
        "fallbacks": ["re_ice", "rei"],
    },
    # SW-specific -------------------------------------------------------
    "solar_zenith_angle": {
        "standard_names": {"solar_zenith_angle"},
        "fallbacks": ["sza", "zenith_angle"],
    },
    "surface_albedo": {
        "standard_names": {"surface_albedo"},
        "fallbacks": ["albedo", "alb"],
    },
}


# ---------------------------------------------------------------------------
# Helper: find a variable in the dataset
# ---------------------------------------------------------------------------

def _find_var(ds: nc.Dataset, key: str):
    """
    Search *ds* for the physical quantity identified by *key* (a key in
    _VAR_TABLE).  Returns (variable_name, nc.Variable) or (None, None).

    Search order:
      1. Variable whose ``standard_name`` attribute matches any CF name in the
         table for *key*.
      2. Variable whose name matches one of the short-name fallbacks (case-
         sensitive first, then case-insensitive).
    """
    entry = _VAR_TABLE[key]
    cf_names = entry["standard_names"]
    fallbacks = entry["fallbacks"]

    # Pass 1 — match by standard_name attribute
    for vname, var in ds.variables.items():
        sn = getattr(var, "standard_name", None)
        if sn is not None and sn.strip() in cf_names:
            return vname, var

    # Pass 2 — match by variable name (exact)
    for fb in fallbacks:
        if fb in ds.variables:
            return fb, ds.variables[fb]

    # Pass 3 — match by variable name (case-insensitive)
    lower_map = {vname.lower(): vname for vname in ds.variables}
    for fb in fallbacks:
        found = lower_map.get(fb.lower())
        if found is not None:
            return found, ds.variables[found]

    return None, None


# ---------------------------------------------------------------------------
# Helper: read variable as 1-D numpy array, masked -> NaN
# ---------------------------------------------------------------------------

def _read_1d(var) -> np.ndarray:
    """
    Read *var* into a 1-D float64 array, collapsing leading size-1 dimensions
    and converting masked values to NaN.
    """
    data = var[:]
    data = np.ma.filled(data.astype(np.float64), np.nan)
    data = np.squeeze(data)
    if data.ndim != 1:
        while data.ndim > 1:
            data = data[0]
    return data


def _read_scalar(var) -> float:
    """Read a variable that may be scalar or 1-D; return the first finite value."""
    data = np.ma.filled(var[:].astype(np.float64), np.nan)
    data = np.ravel(data)
    finite = data[np.isfinite(data)]
    return float(finite[0]) if finite.size > 0 else np.nan


# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------

def _units_str(var) -> str:
    return getattr(var, "units", "").strip().lower()


def _convert_height(data: np.ndarray, units: str) -> np.ndarray:
    """Return height in km."""
    if units in ("m", "meters", "metre", "metres"):
        return data / 1000.0
    return data


def _convert_pressure(data: np.ndarray, units: str) -> np.ndarray:
    """Return pressure in hPa."""
    if units in ("pa", "pascal", "pascals"):
        return data / 100.0
    return data


def _convert_temperature(data: np.ndarray, units: str) -> np.ndarray:
    """Return temperature in degC."""
    if units in ("k", "kelvin"):
        return data - 273.15
    return data


def _convert_wv_mixr(data: np.ndarray, units: str) -> np.ndarray:
    """Return water-vapour mixing ratio in g/kg."""
    if units in ("kg/kg", "kg kg-1", "kg kg**-1", "1"):
        return data * 1000.0
    return data


def _convert_mole_fraction(data: np.ndarray, units: str) -> np.ndarray:
    """Return mole fraction in ppm."""
    if units in ("mol/mol", "mol mol-1", "mol mol**-1", "1", ""):
        return data * 1.0e6
    return data


def _convert_cloud_fraction(data: np.ndarray, units: str) -> np.ndarray:
    """Return cloud fraction as a fraction in [0, 1]."""
    if units in ("%", "percent"):
        return data / 100.0
    return data


def _convert_path(data: np.ndarray, units: str) -> np.ndarray:
    """Return LWP/IWP in g/m2."""
    if units in ("kg/m2", "kg m-2", "kg m**-2"):
        return data * 1000.0
    return data


def _convert_re(data: np.ndarray, units: str) -> np.ndarray:
    """Return effective radius in microm."""
    if units in ("m", "meters", "metre", "metres"):
        return data * 1.0e6
    return data


def _convert_albedo(data: np.ndarray, units: str) -> np.ndarray:
    """Return surface albedo as a fraction in [0, 1]."""
    if units in ("%", "percent"):
        return data / 100.0
    return data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_cf_sounding(path: str, iceflag: int = None):
    """
    Open a CF-convention NetCDF sounding file at *path*, detect and convert
    all physical quantities (atmospheric + SW-specific), and return an
    :class:`~rrtm_sw.sounding.SWSounding`.

    Parameters
    ----------
    path : str
        Path to the NetCDF file.
    iceflag : int, optional
        Ice-cloud parameterisation flag.  If provided, overrides the value
        stored in the file's global ``iceflag`` attribute.  Defaults to 2
        when neither *iceflag* parameter nor file attribute is present.

    Returns
    -------
    SWSounding

    Raises
    ------
    ValueError
        If solar_zenith_angle is absent from the file or all values are NaN.
    """
    # Lazy import to avoid circular imports
    from .sounding import SWSounding

    ds = nc.Dataset(path, "r")
    try:
        _data = _read_sounding_data(ds, iceflag)
    finally:
        ds.close()

    return SWSounding(**_data)


# ---------------------------------------------------------------------------
# Internal: read everything from an open Dataset
# ---------------------------------------------------------------------------

def _read_sounding_data(ds: nc.Dataset, iceflag_override) -> dict:
    """
    Extract and unit-convert all sounding fields from *ds*.
    Returns a dict suitable for passing to SWSounding(**...).
    """

    # ------------------------------------------------------------------ #
    # Height                                                               #
    # ------------------------------------------------------------------ #
    vname, var = _find_var(ds, "height")
    if var is not None:
        z_raw = _read_1d(var)
        z_km = _convert_height(z_raw, _units_str(var))
    else:
        z_km = None

    # ------------------------------------------------------------------ #
    # Pressure                                                             #
    # ------------------------------------------------------------------ #
    vname, var = _find_var(ds, "pressure")
    if var is not None:
        p_raw = _read_1d(var)
        p_hPa = _convert_pressure(p_raw, _units_str(var))
    else:
        p_hPa = None

    # ------------------------------------------------------------------ #
    # Determine vertical ordering from height or pressure                  #
    # ------------------------------------------------------------------ #
    flip = False
    ref = z_km if z_km is not None else (None if p_hPa is None else -p_hPa)
    if ref is not None and len(ref) > 1:
        valid = ref[~np.isnan(ref)]
        if len(valid) > 1 and valid[-1] < valid[0]:
            flip = True

    def maybe_flip(arr):
        if arr is None:
            return None
        if flip:
            return arr[::-1].copy()
        return arr

    if flip:
        z_km  = maybe_flip(z_km)
        p_hPa = maybe_flip(p_hPa)

    # ------------------------------------------------------------------ #
    # Temperature                                                           #
    # ------------------------------------------------------------------ #
    vname, var = _find_var(ds, "temperature")
    if var is not None:
        T_raw = _read_1d(var)
        T_C = maybe_flip(_convert_temperature(T_raw, _units_str(var)))
    else:
        T_C = None

    # ------------------------------------------------------------------ #
    # Water-vapour mixing ratio                                             #
    # ------------------------------------------------------------------ #
    vname, var = _find_var(ds, "wv_mixr")
    if var is not None:
        wv_raw = _read_1d(var)
        wv_gkg = maybe_flip(_convert_wv_mixr(wv_raw, _units_str(var)))
    else:
        wv_gkg = None

    # ------------------------------------------------------------------ #
    # Trace gases (mole fractions -> ppm)                                  #
    # ------------------------------------------------------------------ #
    def read_gas(key):
        vn, v = _find_var(ds, key)
        if v is None:
            return None
        return maybe_flip(_convert_mole_fraction(_read_1d(v), _units_str(v)))

    co2_ppm = read_gas("co2")
    o3_ppm  = read_gas("o3")
    n2o_ppm = read_gas("n2o")
    co_ppm  = read_gas("co")
    ch4_ppm = read_gas("ch4")

    # ------------------------------------------------------------------ #
    # Cloud fields                                                          #
    # ------------------------------------------------------------------ #
    vname, var = _find_var(ds, "cloud_fraction")
    if var is not None:
        cf = maybe_flip(_convert_cloud_fraction(_read_1d(var), _units_str(var)))
    else:
        cf = None

    vname, var = _find_var(ds, "lwp")
    if var is not None:
        lwp = maybe_flip(_convert_path(_read_1d(var), _units_str(var)))
    else:
        lwp = None

    vname, var = _find_var(ds, "iwp")
    if var is not None:
        iwp = maybe_flip(_convert_path(_read_1d(var), _units_str(var)))
    else:
        iwp = None

    vname, var = _find_var(ds, "re_liq")
    if var is not None:
        re_liq = maybe_flip(_convert_re(_read_1d(var), _units_str(var)))
    else:
        re_liq = None

    vname, var = _find_var(ds, "re_ice")
    if var is not None:
        re_ice = maybe_flip(_convert_re(_read_1d(var), _units_str(var)))
    else:
        re_ice = None

    # ------------------------------------------------------------------ #
    # iceflag                                                               #
    # ------------------------------------------------------------------ #
    if iceflag_override is not None:
        iceflag = int(iceflag_override)
    else:
        iceflag = int(getattr(ds, "iceflag", 2))

    # ------------------------------------------------------------------ #
    # SW-specific: Solar zenith angle (REQUIRED)                           #
    # ------------------------------------------------------------------ #
    vname, var = _find_var(ds, "solar_zenith_angle")
    if var is not None:
        sza_raw = _read_1d(var)
        # Take the first finite scalar value (SZA is typically a scalar or
        # a one-element array; multi-element arrays use the first entry)
        finite_sza = sza_raw[np.isfinite(sza_raw)]
        if finite_sza.size == 0:
            raise ValueError(
                "solar_zenith_angle variable found in '{}' but all values "
                "are NaN.  A valid solar zenith angle is required for SW "
                "calculations.".format(ds.filepath() if hasattr(ds, 'filepath') else path)
            )
        sza_deg = float(finite_sza[0])
    else:
        raise ValueError(
            "solar_zenith_angle not found in the NetCDF file.  "
            "A solar zenith angle (standard_name='solar_zenith_angle' or "
            "variable named 'sza'/'zenith_angle') is required for SW "
            "calculations."
        )

    # ------------------------------------------------------------------ #
    # SW-specific: Surface albedo (optional, default 0.1)                  #
    # ------------------------------------------------------------------ #
    vname, var = _find_var(ds, "surface_albedo")
    if var is not None:
        alb_raw = _read_1d(var)
        alb_raw = _convert_albedo(alb_raw, _units_str(var))
        finite_alb = alb_raw[np.isfinite(alb_raw)]
        if finite_alb.size == 0:
            warnings.warn(
                "surface_albedo variable found but all values are NaN; "
                "defaulting to 0.1 for all SW bands.",
                UserWarning,
                stacklevel=3,
            )
            albedo = 0.1
        elif finite_alb.size == 1:
            # Scalar-like: broadcast to all bands inside SWSounding
            albedo = float(finite_alb[0])
        elif finite_alb.size == 14:
            # Per-band (14 SW bands): pass as array
            albedo = finite_alb.copy()
        else:
            # Unexpected size: use mean and warn
            warnings.warn(
                f"surface_albedo has {finite_alb.size} finite values (expected 1 or 14); "
                "using the mean value.",
                UserWarning,
                stacklevel=3,
            )
            albedo = float(np.mean(finite_alb))
    else:
        warnings.warn(
            "surface_albedo not found in the NetCDF file; "
            "defaulting to 0.1 for all SW bands.",
            UserWarning,
            stacklevel=3,
        )
        albedo = 0.1

    # ------------------------------------------------------------------ #
    # SW-specific: Julian day (optional, default 1)                        #
    # ------------------------------------------------------------------ #
    # Read from global attribute "julian_day" or fallback "julday"
    julian_day_raw = getattr(ds, "julian_day", None)
    if julian_day_raw is None:
        julian_day_raw = getattr(ds, "julday", None)
    if julian_day_raw is not None:
        try:
            julday = int(julian_day_raw)
        except (TypeError, ValueError):
            warnings.warn(
                f"Global attribute 'julian_day' could not be converted to int "
                f"(got {julian_day_raw!r}); defaulting to 1.",
                UserWarning,
                stacklevel=3,
            )
            julday = 1
    else:
        warnings.warn(
            "Global attribute 'julian_day' (or 'julday') not found; "
            "defaulting to Julian day 1 (no Earth-Sun distance correction).",
            UserWarning,
            stacklevel=3,
        )
        julday = 1

    # ------------------------------------------------------------------ #
    # Assemble gas dict                                                     #
    # ------------------------------------------------------------------ #
    gas_ppm = {}
    if co2_ppm is not None:
        gas_ppm['co2'] = co2_ppm
    if o3_ppm is not None:
        gas_ppm['o3'] = o3_ppm
    if n2o_ppm is not None:
        gas_ppm['n2o'] = n2o_ppm
    if co_ppm is not None:
        gas_ppm['co'] = co_ppm
    if ch4_ppm is not None:
        gas_ppm['ch4'] = ch4_ppm

    return dict(
        z_km=z_km,
        P_hPa=p_hPa,
        T_C=T_C,
        wv_gkg=wv_gkg,
        gas_ppm=gas_ppm if gas_ppm else None,
        sza_deg=sza_deg,
        albedo=albedo,
        julday=julday,
        frac=cf,
        lwp_gm2=lwp,
        iwp_gm2=iwp,
        re_liq_um=re_liq,
        re_ice_um=re_ice,
        iceflag=iceflag,
    )
