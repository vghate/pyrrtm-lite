"""
cf_reader.py
------------
Read a CF-convention NetCDF sounding file and return a Sounding object.

Public API
----------
    read_cf_sounding(path: str, iceflag: int = None) -> Sounding
"""

import warnings
import numpy as np
import netCDF4 as nc


# ---------------------------------------------------------------------------
# Look-up tables: CF standard_name → fallback short names
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
# Helper: read variable as 1-D numpy array, masked → NaN
# ---------------------------------------------------------------------------

def _read_1d(var) -> np.ndarray:
    """
    Read *var* into a 1-D float64 array, collapsing leading size-1 dimensions
    and converting masked values to NaN.
    """
    data = var[:]
    data = np.ma.filled(data.astype(np.float64), np.nan)
    # Squeeze leading singleton dimensions (e.g. time=1, ensemble=1)
    data = np.squeeze(data)
    if data.ndim != 1:
        # Take the first element along every extra leading dimension
        while data.ndim > 1:
            data = data[0]
    return data


# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------

def _units_str(var) -> str:
    return getattr(var, "units", "").strip().lower()


def _convert_height(data: np.ndarray, units: str) -> np.ndarray:
    """Return height in km."""
    if units in ("m", "meters", "metre", "metres"):
        return data / 1000.0
    # km, kilometer, kilometres, or unknown → assume km
    return data


def _convert_pressure(data: np.ndarray, units: str) -> np.ndarray:
    """Return pressure in hPa."""
    if units in ("pa", "pascal", "pascals"):
        return data / 100.0
    # hpa, mb, mbar, millibar, millibars → keep
    return data


def _convert_temperature(data: np.ndarray, units: str) -> np.ndarray:
    """
    Return temperature in °C.  Converts from K when units indicate Kelvin.
    (Sounding.__init__ expects T_C in degrees Celsius.)
    """
    if units in ("k", "kelvin"):
        return data - 273.15
    # degC, celsius, °C → keep
    return data


def _convert_wv_mixr(data: np.ndarray, units: str) -> np.ndarray:
    """Return water-vapour mixing ratio in g/kg."""
    if units in ("kg/kg", "kg kg-1", "kg kg**-1", "1"):
        return data * 1000.0
    # g/kg → keep
    return data


def _convert_mole_fraction(data: np.ndarray, units: str) -> np.ndarray:
    """Return mole fraction in ppm."""
    if units in ("mol/mol", "mol mol-1", "mol mol**-1", "1", ""):
        return data * 1.0e6
    # ppm → keep
    return data


def _convert_cloud_fraction(data: np.ndarray, units: str) -> np.ndarray:
    """Return cloud fraction as a fraction in [0, 1]."""
    if units in ("%", "percent"):
        return data / 100.0
    # dimensionless [0,1] → keep
    return data


def _convert_path(data: np.ndarray, units: str) -> np.ndarray:
    """Return LWP/IWP in g/m²."""
    if units in ("kg/m2", "kg m-2", "kg m**-2"):
        return data * 1000.0
    # g/m² → keep
    return data


def _convert_re(data: np.ndarray, units: str) -> np.ndarray:
    """Return effective radius in µm."""
    if units in ("m", "meters", "metre", "metres"):
        return data * 1.0e6
    # µm → keep
    return data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_cf_sounding(path: str, iceflag: int = None):
    """
    Open a CF-convention NetCDF sounding file at *path*, detect and convert
    all physical quantities, and return a :class:`~rrtm_lw.sounding.Sounding`.

    Parameters
    ----------
    path : str
        Path to the NetCDF file.
    iceflag : int, optional
        Ice-cloud parameterisation flag.  If provided, overrides the value
        stored in the file's global ``iceflag`` attribute.  Defaults to 2 when
        neither *iceflag* parameter nor file attribute is present.

    Returns
    -------
    Sounding
    """
    # Lazy import to avoid circular imports
    from .sounding import Sounding

    ds = nc.Dataset(path, "r")
    try:
        _data = _read_sounding_data(ds, iceflag)
    finally:
        ds.close()

    return Sounding(**_data)


# ---------------------------------------------------------------------------
# Internal: read everything from an open Dataset
# ---------------------------------------------------------------------------

def _read_sounding_data(ds: nc.Dataset, iceflag_override) -> dict:
    """
    Extract and unit-convert all sounding fields from *ds*.
    Returns a dict suitable for passing to Sounding(**...).
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
    # We want z increasing upward.  If z is available and decreasing, flip.
    # If only pressure is available and increasing (surface at index 0), flip.
    flip = False
    ref = z_km if z_km is not None else (None if p_hPa is None else -p_hPa)
    if ref is not None and len(ref) > 1:
        # Ignore NaN when checking direction
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
        z_km = maybe_flip(z_km)
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
    # Trace gases (mole fractions → ppm)                                   #
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
    # Surface temperature                                                   #
    # ------------------------------------------------------------------ #
    vname, var = _find_var(ds, "surface_temperature")
    if var is not None:
        tsk_raw = np.atleast_1d(_read_1d(var)).ravel()
        # Surface temperature is a scalar or a single-level field
        tsk_val = tsk_raw[0] if tsk_raw.size >= 1 else np.nan
        tbound_C = _convert_temperature(
            np.array([tsk_val]), _units_str(var)
        )[0]
        if np.isnan(tbound_C):
            tbound_C = None
            warnings.warn(
                "surface_temperature not found; defaulting to lowest-level "
                "air temperature."
            )
    else:
        tbound_C = None
        warnings.warn(
            "surface_temperature not found; defaulting to lowest-level "
            "air temperature."
        )

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
    # Assemble result dict                                                  #
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
        tbound_C=tbound_C,
        frac=cf,
        lwp_gm2=lwp,
        iwp_gm2=iwp,
        re_liq_um=re_liq,
        re_ice_um=re_ice,
        iceflag=iceflag,
    )
