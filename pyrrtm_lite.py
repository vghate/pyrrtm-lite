#!/usr/bin/env python3
"""
pyrrtm-lite (bug-fixed) — Streamlined broadband radiative flux profiles.

This version uses locally fixed LW and SW modules (lw/, sw/) that correct
12 bugs identified in the original pyrrtm Python port.  See pyrrtmlite_bugs.txt.

Usage
-----
    python pyrrtm_lite.py --config config.json --input sounding.nc
    python pyrrtm_lite.py --config config.json --input sounding.nc --output fluxes.nc

Output variables (W/m²)
------------------------
    lw_dn, lw_up, sw_dn, sw_up                       all-sky broadband fluxes
    lw_dn_clr, lw_up_clr, sw_dn_clr, sw_up_clr       clear-sky (if save_clearsky=true)

Dependencies
------------
    numpy, scipy, netCDF4  (standard scientific Python)
"""

import argparse
import json
import math
import multiprocessing
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import netCDF4 as nc
import numpy as np

warnings.filterwarnings('ignore')

# ── Use local bug-fixed modules ────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))          # makes "lw" and "sw" importable
# Also add rrtm_proj so the LW taumol/setcoef sub-imports still resolve
for _fallback in [Path.home() / 'Desktop' / 'rrtm_proj',
                  Path.home() / 'Desktop' / 'rrtm_sw_proj']:
    if _fallback.is_dir() and str(_fallback) not in sys.path:
        sys.path.append(str(_fallback))

from lw.run      import run as _lw_run
from lw.sounding import Sounding as LWSounding
from lw.std_atm  import gas_vmr_std, T_std, P_std
from sw.run      import run as _sw_run
from sw.sounding import SWSounding

_LW_COEFFS = str(_HERE / 'data' / 'lw_coeffs.nc')
_SW_COEFFS = str(_HERE / 'data' / 'sw_coeffs.nc')


# ════════════════════════════════════════════════════════════════════════════
# Config loader (JSON only)
# ════════════════════════════════════════════════════════════════════════════

_DEFAULTS = {
    'surface':       {'sw_albedo': 0.15, 'skin_temperature_C': None},
    'gases':         {'CO2_ppm': 422.0,  'CH4_ppm': 1.9},
    'solar':         {'latitude_deg': None, 'longitude_deg': None, 'fixed_sza_deg': None},
    'vertical_grid': {'resolution_below_15km_km': 0.5},
    'clouds':        {'iceflag': 3, 're_liq_um_default': 10.0, 're_ice_um_default': 40.0},
    'output':        {'save_clearsky': True},
    'processing':    {'n_cpu': 4},
}

def load_config(path: str) -> dict:
    """Load JSON config; strip _c_* hint keys; fill missing keys from defaults."""
    with open(path) as fh:
        data = json.loads(fh.read())

    def _strip(obj):
        if isinstance(obj, dict):
            return {k: _strip(v) for k, v in obj.items() if not k.startswith('_c')}
        return obj

    cfg = _strip(data)
    for section, defaults in _DEFAULTS.items():
        cfg.setdefault(section, {})
        for key, val in defaults.items():
            cfg[section].setdefault(key, val)
    return cfg


# ════════════════════════════════════════════════════════════════════════════
# Vertical grid builder
# ════════════════════════════════════════════════════════════════════════════

def build_output_grid(dz_low_km: float) -> np.ndarray:
    """
    Build output height grid [km]:
      0 – 15 km  at dz_low_km resolution
      15 – 60 km at 1 km resolution
    Returns strictly increasing array of altitudes in km.
    """
    z_low  = np.arange(0.0, 15.0, dz_low_km)
    z_high = np.arange(15.0, 51.0, 1.0)   # 50 km ceiling (RRTM practical limit)
    return np.unique(np.concatenate([z_low, z_high]))


# ════════════════════════════════════════════════════════════════════════════
# Sounding reader
# ════════════════════════════════════════════════════════════════════════════

def read_sounding(nc_path: str):
    """
    Read a CF-NetCDF or ARM interpolated sonde file.

    Returns dict with keys:
        z_km   : (nlevels,)  altitude [km AGL]  — strictly increasing
        time   : (ntimes,)   seconds since 1970-01-01 (float), or None
        p_hPa  : (ntimes, nlevels) or (nlevels,)  pressure [hPa]
        T_C    : same shape  temperature [°C]
        sh_gg  : same shape  specific humidity [g/g]
        lat    : scalar or None
        lon    : scalar or None
        cloud  : dict or None  (frac, lwp_gm2, re_liq_um, iwp_gm2, re_ice_um)
    """
    ds   = nc.Dataset(nc_path, 'r')
    dims = {k: v.size for k, v in ds.dimensions.items()}

    # ── Altitude ──────────────────────────────────────────────────────────
    z_km = None
    for vn in ['height', 'altitude', 'alt', 'z', 'lev', 'level']:
        if vn in ds.variables:
            arr = np.array(ds.variables[vn][:])
            if arr.ndim == 1 and len(arr) > 4:
                units = getattr(ds.variables[vn], 'units', '').lower()
                z_km  = arr / 1000.0 if 'm' in units and 'km' not in units else arr
                if z_km[-1] < z_km[0]:
                    z_km = z_km[::-1]
                    _flip = True
                else:
                    _flip = False
                break
    if z_km is None:
        raise ValueError("Cannot find altitude variable in sounding.")

    # ── Time ──────────────────────────────────────────────────────────────
    time_sec = None
    base_time = 0.0
    if 'base_time' in ds.variables:
        base_time = float(ds.variables['base_time'][:])
    for vn in ['time_offset', 'time']:
        if vn in ds.variables:
            t_arr = np.array(ds.variables[vn][:])
            if t_arr.ndim == 1 and len(t_arr) > 0:
                time_sec = base_time + t_arr
                break

    # ── Lat/lon ───────────────────────────────────────────────────────────
    lat = lon = None
    for vn in ['lat', 'latitude']:
        if vn in ds.variables:
            lat = float(np.array(ds.variables[vn][:]).flat[0])
            break
    for vn in ['lon', 'longitude']:
        if vn in ds.variables:
            lon = float(np.array(ds.variables[vn][:]).flat[0])
            break

    # ── Meteorological fields ─────────────────────────────────────────────
    def _get(names, scale=1.0):
        for vn in names:
            if vn in ds.variables:
                return np.array(ds.variables[vn][:]) * scale
        return None

    # Pressure
    p_raw = _get(['bar_pres', 'pres', 'pressure', 'p'])
    if p_raw is None:
        raise ValueError("Cannot find pressure variable.")
    # Detect unit: kPa → hPa if values look like kPa
    if np.nanmean(p_raw[p_raw > 0]) < 200:
        p_raw = p_raw * 10.0    # kPa → hPa

    T_raw  = _get(['temp', 'temperature', 'T', 'air_temperature'])
    if T_raw is None:
        raise ValueError("Cannot find temperature variable.")
    # K → °C if values look like Kelvin
    if np.nanmean(T_raw[np.isfinite(T_raw)]) > 100:
        T_raw = T_raw - 273.15

    sh_raw = _get(['sh', 'specific_humidity', 'q'])
    if sh_raw is None:
        # try relative humidity and convert
        rh = _get(['rh', 'relative_humidity'])
        if rh is not None:
            # rough: sh ≈ 0.622 * e_s*RH/P  (simplified)
            e_s = 6.112 * np.exp(17.67 * T_raw / (T_raw + 243.5))   # hPa
            sh_raw = 0.622 * e_s * (rh / 100.0) / (p_raw - e_s)
        else:
            sh_raw = np.full_like(p_raw, 0.01 / 1000.0)   # 10 g/kg placeholder

    # ── Cloud fields (optional) ────────────────────────────────────────────
    # Accepts path [g/m²] or content [g/m³] arrays; content is integrated
    # over layer thickness using hydrostatic dz = dP/(rho*g).
    cloud = None
    frac = _get(['cloud_fraction', 'cf', 'cldFrac', 'CLDFRA'])
    lwp  = _get(['lwp', 'liquid_water_path', 'LWP'])
    iwp  = _get(['iwp', 'ice_water_path',    'IWP'])
    re_l = _get(['re_liq', 'effective_radius_liquid', 're_liquid', 're_liq_um'])
    re_i = _get(['re_ice', 'effective_radius_ice', 're_ice_um'])

    # Water content [g/m³] → path [g/m²]: dz from hydrostatic equation
    def _content_to_path(wc, p2d, T2d):
        """wc [g/m³] × dz [m] = path [g/m²].  dz from hydrostatic: dP/(rho*g)."""
        if wc is None: return None
        T_K  = T2d + 273.15 if np.nanmean(T2d[np.isfinite(T2d)]) < 100 else T2d
        P_Pa = p2d * 100.0  if np.nanmean(p2d[p2d > 0]) < 2000    else p2d
        rho  = P_Pa / (287.0 * np.where(T_K > 50, T_K, 250.0))    # kg/m³ dry air
        dP   = np.abs(np.diff(P_Pa, axis=-1, append=P_Pa[..., -1:]))
        dz   = dP / (rho * 9.80665)                                 # m
        return wc * dz                                               # g/m³ × m = g/m²

    if lwp is None:
        wc_liq = _get(['wc_liq', 'liquid_water_content', 'clwc', 'ql'])
        if wc_liq is not None:
            lwp = _content_to_path(wc_liq, p_raw, T_raw)

    if iwp is None:
        wc_ice = _get(['wc_ice', 'ice_water_content', 'ciwc', 'qi'])
        if wc_ice is not None:
            iwp = _content_to_path(wc_ice, p_raw, T_raw)

    # Auto-detect frac from LWP+IWP if not provided (icld logic)
    if frac is None and (lwp is not None or iwp is not None):
        liq = lwp if lwp is not None else 0.0
        ice = iwp if iwp is not None else 0.0
        total = np.where(np.isfinite(liq), np.maximum(liq, 0), 0) + \
                np.where(np.isfinite(ice), np.maximum(ice, 0), 0)
        frac = np.where(total > 0, 1.0, 0.0)

    if frac is not None and (lwp is not None or iwp is not None):
        cloud = dict(frac=frac,
                     lwp_gm2=np.where(np.isfinite(lwp), np.maximum(lwp, 0), 0) if lwp is not None else np.zeros_like(frac),
                     re_liq_um=re_l if re_l is not None else np.full_like(frac, 10.0),
                     iwp_gm2=np.where(np.isfinite(iwp), np.maximum(iwp, 0), 0) if iwp is not None else np.zeros_like(frac),
                     re_ice_um=re_i if re_i is not None else np.full_like(frac, 40.0))

    ds.close()

    # Ensure shape is (ntimes, nlevels)
    def _ensure_2d(arr):
        if arr is None:
            return None
        if arr.ndim == 1:
            return arr[np.newaxis, :]   # (1, nlevels)
        return arr

    p_2d  = _ensure_2d(p_raw)
    T_2d  = _ensure_2d(T_raw)
    sh_2d = _ensure_2d(sh_raw)

    ntimes, nlevels = p_2d.shape
    # Flip level axis to ascending z if needed
    if z_km[0] > z_km[-1]:
        z_km  = z_km[::-1]
        p_2d  = p_2d[:,  ::-1]
        T_2d  = T_2d[:,  ::-1]
        sh_2d = sh_2d[:, ::-1]

    if time_sec is None:
        time_sec = np.zeros(ntimes)

    return dict(z_km=z_km, time=time_sec,
                p_hPa=p_2d, T_C=T_2d, sh_gg=sh_2d,
                lat=lat, lon=lon, cloud=cloud)


# ════════════════════════════════════════════════════════════════════════════
# Solar zenith angle
# ════════════════════════════════════════════════════════════════════════════

def solar_zenith(unix_sec: float, lat_deg: float, lon_deg: float) -> float:
    """
    Compute solar zenith angle [degrees] for a given UTC Unix timestamp
    and geographic location. Uses the Spencer (1971) / Michalsky (1988)
    algorithm; accurate to ~0.01° for most purposes.
    """
    dt  = datetime.fromtimestamp(unix_sec, tz=timezone.utc)
    doy = dt.timetuple().tm_yday
    hour_utc = dt.hour + dt.minute / 60.0 + dt.second / 3600.0

    B  = 2 * math.pi * (doy - 1) / 365.0
    # Equation of time [minutes]
    eot = 229.18 * (0.000075 + 0.001868 * math.cos(B)
                    - 0.032077 * math.sin(B)
                    - 0.014615 * math.cos(2*B)
                    - 0.04089  * math.sin(2*B))
    # Solar declination [rad]
    dec = (0.006918 - 0.399912 * math.cos(B)
           + 0.070257 * math.sin(B)
           - 0.006758 * math.cos(2*B)
           + 0.000907 * math.sin(2*B)
           - 0.002697 * math.cos(3*B)
           + 0.00148  * math.sin(3*B))
    # True solar time [hours]
    tst  = hour_utc + lon_deg / 15.0 + eot / 60.0
    ha   = math.radians((tst - 12.0) * 15.0)   # hour angle [rad]
    lat  = math.radians(lat_deg)
    cos_z = (math.sin(lat) * math.sin(dec)
             + math.cos(lat) * math.cos(dec) * math.cos(ha))
    cos_z = max(-1.0, min(1.0, cos_z))
    return math.degrees(math.acos(cos_z))


# ════════════════════════════════════════════════════════════════════════════
# Profile interpolator
# ════════════════════════════════════════════════════════════════════════════

def interp_profile(z_src, p_src, T_src, sh_src, z_out):
    """
    Interpolate a single sounding column onto z_out [km].
    Fills NaN and out-of-range values with USSA76.
    Returns (z_km, p_hPa, T_C, sh_gg) on z_out grid.
    """
    # USSA76 fallbacks
    p_ussa  = P_std(z_out) / 100.0   # Pa → hPa
    T_ussa  = T_std(z_out) - 273.15  # K → °C

    # Valid source data only
    good = np.isfinite(p_src) & np.isfinite(T_src) & (p_src > 0)
    if good.sum() < 2:
        return z_out, p_ussa, T_ussa, np.full(len(z_out), 5e-3)

    z_g = z_src[good]; p_g = p_src[good]
    T_g = T_src[good]; sh_g = sh_src[good]

    # Sonde levels: use interpolated values; above sonde top: USSA76
    p_out  = np.where(z_out <= z_g[-1],
                      np.interp(z_out, z_g, p_g,  left=p_g[0],  right=p_ussa[-1]),
                      p_ussa)
    T_out  = np.where(z_out <= z_g[-1],
                      np.interp(z_out, z_g, T_g,  left=T_g[0],  right=T_ussa[-1]),
                      T_ussa)
    sh_out = np.where(z_out <= z_g[-1],
                      np.interp(z_out, z_g, sh_g, left=sh_g[0], right=1e-6),
                      1e-6)

    # Enforce strictly decreasing pressure (required by RRTM)
    # Work from top down: cap each level so it is always < the level below it
    for i in range(1, len(p_out)):
        if p_out[i] >= p_out[i - 1]:
            p_out[i] = p_out[i - 1] * 0.85

    # After monotonic fix, apply absolute floor — but keep it strictly decreasing
    # by scaling rather than clamping to a flat value
    p_floor = 0.01   # 0.01 hPa ≈ 80 km — well within RRTM range
    for i in range(len(p_out) - 1, -1, -1):
        if p_out[i] < p_floor:
            p_out[i] = p_floor
            p_floor  = p_floor * 0.5   # next lower level gets a smaller floor

    return z_out, p_out, np.maximum(T_out, -90.0), np.maximum(sh_out, 1e-7)


# ════════════════════════════════════════════════════════════════════════════
# Single-profile radiative transfer
# ════════════════════════════════════════════════════════════════════════════

def _run_one_profile(args):
    """
    Worker function (called by ProcessPoolExecutor).
    Returns dict of broadband fluxes on z_out grid.
    """
    (z_km, p_hPa, T_C, sh_gg, cloud_row,
     cfg, sza_deg, lw_coeffs, sw_coeffs) = args

    co2      = cfg['gases']['CO2_ppm']
    ch4      = cfg['gases']['CH4_ppm']
    alb      = cfg['surface']['sw_albedo']
    skin     = cfg['surface'].get('skin_temperature_C', None)
    save_clr = cfg['output']['save_clearsky']
    iceflag  = int(cfg.get('clouds', {}).get('iceflag', 3))
    re_liq_def = float(cfg.get('clouds', {}).get('re_liq_um_default', 10.0))
    re_ice_def = float(cfg.get('clouds', {}).get('re_ice_um_default', 40.0))

    nlev    = len(z_km)
    wv_gkg  = sh_gg * 1000.0
    o3_vmr  = gas_vmr_std(z_km, 'o3')
    gas_ppm = {
        'co2': np.full(nlev, co2),
        'o3':  o3_vmr * 1e6,
        'n2o': np.full(nlev, 0.314),
        'ch4': np.full(nlev, ch4),
        'co':  np.full(nlev, 0.15),
    }
    T_sfc = skin if skin is not None else float(T_C[0])

    # ── Cloud fields (level grid) ──────────────────────────────────────────
    frac_lev   = np.zeros(nlev)
    lwp_lev    = np.zeros(nlev)
    reliq_lev  = np.zeros(nlev)
    iwp_lev    = np.zeros(nlev)
    reice_lev  = np.zeros(nlev)
    has_cloud  = False

    if cloud_row is not None:
        frac_lev  = np.clip(np.interp(z_km, cloud_row['z_km'], cloud_row['frac']), 0, 1)
        lwp_lev   = np.maximum(np.interp(z_km, cloud_row['z_km'], cloud_row['lwp']),  0)
        iwp_lev   = np.maximum(np.interp(z_km, cloud_row['z_km'], cloud_row['iwp']),  0)
        # Effective radius: use provided values where cloud exists, default elsewhere
        reliq_raw = np.interp(z_km, cloud_row['z_km'], cloud_row['re_liq'])
        reice_raw = np.interp(z_km, cloud_row['z_km'], cloud_row['re_ice'])
        reliq_lev = np.where(lwp_lev > 0, np.maximum(reliq_raw, 2.5), re_liq_def)
        reice_lev = np.where(iwp_lev > 0, np.maximum(reice_raw, 5.0), re_ice_def)
        has_cloud = (frac_lev.max() > 0.01) and ((lwp_lev + iwp_lev).max() > 0)

    # ── LW all-sky ────────────────────────────────────────────────────────
    snd = LWSounding(
        z_km=z_km, T_C=T_C, P_hPa=p_hPa, wv_gkg=wv_gkg,
        gas_ppm=gas_ppm, tbound_C=T_sfc,
        frac=frac_lev, lwp_gm2=lwp_lev, re_liq_um=reliq_lev,
        iwp_gm2=iwp_lev, re_ice_um=reice_lev,
        iceflag=iceflag,
    )
    r_lw = _lw_run(sounding=snd, coeffs_path=lw_coeffs)

    # totdflux/totuflux have nlev points; index 0 = surface, -1 = TOA
    lw_dn = r_lw['totdflux']   # (nlev,)
    lw_up = r_lw['totuflux']

    # ── LW clear-sky ──────────────────────────────────────────────────────
    lw_dn_clr = lw_dn_up_clr = None
    if save_clr and has_cloud:
        snd_clr = LWSounding(
            z_km=z_km, T_C=T_C, P_hPa=p_hPa, wv_gkg=wv_gkg,
            gas_ppm=gas_ppm, tbound_C=T_sfc,
        )
        r_lw_clr    = _lw_run(sounding=snd_clr, coeffs_path=lw_coeffs)
        lw_dn_clr   = r_lw_clr['totdflux']
        lw_dn_up_clr = r_lw_clr['totuflux']
    elif save_clr:
        lw_dn_clr    = lw_dn.copy()
        lw_dn_up_clr = lw_up.copy()

    # ── SW ────────────────────────────────────────────────────────────────
    sw_dn = sw_up = sw_dn_clr = sw_up_clr = None
    daytime = sza_deg < 89.9

    if daytime:
        alb_band = np.full(14, alb)
        ssnd = SWSounding(
            z_km=z_km, T_C=T_C, P_hPa=p_hPa, wv_gkg=wv_gkg,
            gas_ppm=gas_ppm,
            sza_deg=sza_deg, albedo=alb_band,
            frac=frac_lev, lwp_gm2=lwp_lev, re_liq_um=reliq_lev,
            iwp_gm2=iwp_lev, re_ice_um=reice_lev,
            iceflag=2,
        )
        r_sw  = _sw_run(sounding=ssnd, coeffs_path=sw_coeffs)
        sw_dn = r_sw['totdflux']
        sw_up = r_sw['totuflux']

        if save_clr and has_cloud:
            ssnd_clr = SWSounding(
                z_km=z_km, T_C=T_C, P_hPa=p_hPa, wv_gkg=wv_gkg,
                gas_ppm=gas_ppm,
                sza_deg=sza_deg, albedo=alb_band,
            )
            r_sw_clr  = _sw_run(sounding=ssnd_clr, coeffs_path=sw_coeffs)
            sw_dn_clr = r_sw_clr['totdflux']
            sw_up_clr = r_sw_clr['totuflux']
        elif save_clr:
            sw_dn_clr = sw_dn.copy() if sw_dn is not None else None
            sw_up_clr = sw_up.copy() if sw_up is not None else None
    else:
        sw_dn = sw_up = np.zeros(nlev)
        if save_clr:
            sw_dn_clr = sw_up_clr = np.zeros(nlev)

    return dict(
        lw_dn=lw_dn, lw_up=lw_up,
        sw_dn=sw_dn, sw_up=sw_up,
        lw_dn_clr=lw_dn_clr, lw_up_clr=lw_dn_up_clr,
        sw_dn_clr=sw_dn_clr, sw_up_clr=sw_up_clr,
        sza_deg=sza_deg,
    )


# ════════════════════════════════════════════════════════════════════════════
# Main driver
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='pyrrtm-lite: broadband flux profiles from CF-NetCDF soundings.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Example:\n'
            '  python pyrrtm_lite.py --config config.json --input sounding.nc\n'
            '  python pyrrtm_lite.py --config config.json --input sounding.nc --output fluxes.nc'
        ))
    parser.add_argument('--config', required=True, help='JSON config file')
    parser.add_argument('--input',  required=True, help='Input sounding NetCDF')
    parser.add_argument('--output',               help='Output NetCDF (default: <input_stem>_fluxes.nc)')
    args = parser.parse_args()

    # Default output path: same directory as input, stem + _fluxes.nc
    if not args.output:
        inp = Path(args.input)
        args.output = str(inp.parent / (inp.stem + '_fluxes.nc'))
        print(f"Output: {args.output}  (default)")

    # ── Config ────────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    dz_low = float(cfg['vertical_grid']['resolution_below_15km_km'])
    n_cpu  = int(cfg['processing']['n_cpu'])
    if n_cpu < 1:
        n_cpu = multiprocessing.cpu_count()
    save_clr = bool(int(cfg['output']['save_clearsky']))
    print(f"pyrrtm-lite  |  dz<15km={dz_low}km  cores={n_cpu}  "
          f"clear-sky={'yes' if save_clr else 'no'}")

    # ── Read sounding ─────────────────────────────────────────────────────
    print(f"Reading sounding: {args.input}")
    snd = read_sounding(args.input)
    ntimes = snd['p_hPa'].shape[0]
    print(f"  {ntimes} profiles × {len(snd['z_km'])} source levels")

    # ── Build output grid ─────────────────────────────────────────────────
    z_out = build_output_grid(dz_low)
    nlevs = len(z_out)
    print(f"  Output grid: {nlevs} levels  "
          f"({dz_low} km below 15 km, 1 km above)")

    # ── Lat/lon for SZA ───────────────────────────────────────────────────
    lat = cfg['solar'].get('latitude_deg') or snd.get('lat')
    lon = cfg['solar'].get('longitude_deg') or snd.get('lon')
    fixed_sza = cfg['solar'].get('fixed_sza_deg')
    if lat is None or lon is None:
        lat, lon = 36.6, -97.5   # default SGP
        print(f"  WARN: lat/lon not found — defaulting to SGP ({lat}, {lon})")

    # ── Build worker args list ────────────────────────────────────────────
    worker_args = []
    for ti in range(ntimes):
        # Interpolate to output grid
        z_km, p_hPa, T_C, sh_gg = interp_profile(
            snd['z_km'],
            snd['p_hPa'][ti], snd['T_C'][ti], snd['sh_gg'][ti],
            z_out,
        )

        # Cloud row for this time step
        cloud_row = None
        if snd['cloud'] is not None:
            c = snd['cloud']
            idx = min(ti, c['frac'].shape[0] - 1)
            cloud_row = dict(
                z_km=snd['z_km'],
                frac=c['frac'][idx] if c['frac'].ndim > 1 else c['frac'],
                lwp=c['lwp_gm2'][idx] if c['lwp_gm2'].ndim > 1 else c['lwp_gm2'],
                re_liq=c['re_liq_um'][idx] if c['re_liq_um'].ndim > 1 else c['re_liq_um'],
                iwp=c['iwp_gm2'][idx] if c['iwp_gm2'].ndim > 1 else c['iwp_gm2'],
                re_ice=c['re_ice_um'][idx] if c['re_ice_um'].ndim > 1 else c['re_ice_um'],
            )

        # SZA
        if fixed_sza is not None:
            sza = float(fixed_sza)
        else:
            try:
                sza = solar_zenith(float(snd['time'][ti]), lat, lon)
            except Exception:
                sza = 90.0

        worker_args.append((z_km, p_hPa, T_C, sh_gg, cloud_row,
                             cfg, sza, _LW_COEFFS, _SW_COEFFS))

    # ── Run (parallel or serial) ──────────────────────────────────────────
    results = [None] * ntimes
    if n_cpu == 1 or ntimes == 1:
        print(f"Running {ntimes} profile(s) serially...")
        for ti, wargs in enumerate(worker_args):
            results[ti] = _run_one_profile(wargs)
            if (ti + 1) % 50 == 0 or ti == ntimes - 1:
                print(f"  {ti+1}/{ntimes}", end='\r', flush=True)
    else:
        workers = min(n_cpu, ntimes)
        print(f"Running {ntimes} profiles on {workers} cores...")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_one_profile, wa): ti
                       for ti, wa in enumerate(worker_args)}
            done = 0
            for fut in as_completed(futures):
                ti = futures[fut]
                results[ti] = fut.result()
                done += 1
                if done % 50 == 0 or done == ntimes:
                    print(f"  {done}/{ntimes}", end='\r', flush=True)
    print()

    # ── Collect arrays ────────────────────────────────────────────────────
    def _stack(key, fill=0.0):
        return np.array([
            r[key] if r[key] is not None else np.full(nlevs, fill)
            for r in results
        ])   # (ntimes, nlevs)

    lw_dn = _stack('lw_dn')
    lw_up = _stack('lw_up')
    sw_dn = _stack('sw_dn')
    sw_up = _stack('sw_up')
    szas  = np.array([r['sza_deg'] for r in results])

    lw_dn_clr = _stack('lw_dn_clr') if save_clr else None
    lw_up_clr = _stack('lw_up_clr') if save_clr else None
    sw_dn_clr = _stack('sw_dn_clr') if save_clr else None
    sw_up_clr = _stack('sw_up_clr') if save_clr else None

    # ── Write output NetCDF ───────────────────────────────────────────────
    print(f"Writing: {args.output}")
    out = nc.Dataset(args.output, 'w', format='NETCDF4')

    out.createDimension('time',  ntimes)
    out.createDimension('level', nlevs)

    # Coordinates
    vt = out.createVariable('time', 'f8', ('time',))
    vt.units    = 'seconds since 1970-01-01 00:00:00 UTC'
    vt.long_name = 'UTC time'
    vt[:] = snd['time']

    vh = out.createVariable('height', 'f4', ('level',))
    vh.units     = 'km'
    vh.long_name = 'Height above ground level'
    vh[:]  = z_out.astype('f4')

    vsza = out.createVariable('sza', 'f4', ('time',))
    vsza.units    = 'degrees'
    vsza.long_name = 'Solar zenith angle'
    vsza[:] = szas.astype('f4')

    def _write(name, data, long_name, squeeze=True):
        v = out.createVariable(name, 'f4', ('time', 'level'),
                               zlib=True, complevel=4, fill_value=np.nan)
        v.units     = 'W m-2'
        v.long_name = long_name
        v[:] = data.astype('f4')
        if ntimes == 1 and squeeze:
            v.note = 'single profile; time dimension has length 1'

    _write('lw_dn',  lw_dn,  'Downwelling LW flux (all-sky)')
    _write('lw_up',  lw_up,  'Upwelling LW flux (all-sky)')
    _write('sw_dn',  sw_dn,  'Downwelling SW flux (all-sky)')
    _write('sw_up',  sw_up,  'Upwelling SW flux (all-sky)')

    if save_clr:
        _write('lw_dn_clr', lw_dn_clr, 'Downwelling LW flux (clear-sky)')
        _write('lw_up_clr', lw_up_clr, 'Upwelling LW flux (clear-sky)')
        _write('sw_dn_clr', sw_dn_clr, 'Downwelling SW flux (clear-sky)')
        _write('sw_up_clr', sw_up_clr, 'Upwelling SW flux (clear-sky)')

    # ── Global attributes: provenance + full config ───────────────────────
    out.description    = 'pyrrtm-lite broadband radiative flux profiles'
    out.pyrrtm_version = '0.2.0-fixed'
    out.created        = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    out.config_file    = str(args.config)
    out.input_file     = str(args.input)

    # Surface
    out.sw_albedo           = float(cfg['surface']['sw_albedo'])
    skin = cfg['surface']['skin_temperature_C']
    out.skin_temperature_C  = float(skin) if skin is not None else 'from_sounding'

    # Gases
    out.CO2_ppm = float(cfg['gases']['CO2_ppm'])
    out.CH4_ppm = float(cfg['gases']['CH4_ppm'])

    # Cloud parameterization settings
    out.cloud_inflag   = 2                 # always: compute tau from LWP/IWP + re
    out.cloud_liqflag  = 1                 # always: Hu & Stamnes liquid
    out.cloud_iceflag  = int(cfg.get('clouds', {}).get('iceflag', 3))
    out.cloud_re_liq_um_default = float(cfg.get('clouds', {}).get('re_liq_um_default', 10.0))
    out.cloud_re_ice_um_default = float(cfg.get('clouds', {}).get('re_ice_um_default', 40.0))
    out.cloud_fields_in_sounding = 'yes' if snd.get('cloud') is not None else 'no'

    # Solar
    out.latitude_deg  = str(cfg['solar']['latitude_deg']  or 'from_sounding')
    out.longitude_deg = str(cfg['solar']['longitude_deg'] or 'from_sounding')
    out.fixed_sza_deg = str(cfg['solar']['fixed_sza_deg'] or 'computed')

    # Vertical grid
    out.resolution_below_15km_km = dz_low
    out.grid_levels              = nlevs
    out.grid_top_km              = float(z_out[-1])

    # Output options
    out.save_clearsky = 'yes' if save_clr else 'no'

    # Processing
    out.n_cpu_requested = int(cfg['processing']['n_cpu'])
    out.n_cpu_used      = min(int(cfg['processing']['n_cpu'])
                              if cfg['processing']['n_cpu'] > 0
                              else multiprocessing.cpu_count(), ntimes)

    out.close()

    # ── Summary ───────────────────────────────────────────────────────────
    day = szas < 90
    print(f"\nDone.")
    print(f"  Profiles run      : {ntimes}")
    print(f"  Daytime (SZA<90°) : {day.sum()} ({day.mean()*100:.0f}%)")
    print(f"  Output levels     : {nlevs}  "
          f"(0–{z_out[-1]:.0f} km, {dz_low} km / 1 km grid)")
    print(f"  Surface LW↓ mean  : {lw_dn[:, 0].mean():.1f} W/m²")
    if day.sum() > 0:
        print(f"  Surface SW↓ mean (daytime): "
              f"{sw_dn[day, 0].mean():.1f} W/m²")
    print(f"  Output file       : {args.output}")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
