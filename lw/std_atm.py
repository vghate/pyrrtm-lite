"""
US Standard Atmosphere 1976 — temperature, pressure, and gas VMR profiles.

Public API
----------
T_std(z_km)              -> temperature in K
P_std(z_km)              -> pressure in hPa
gas_vmr_std(z_km, gas)   -> volume mixing ratio (dimensionless)
fill_nans(z_km, arrays)  -> NaN-filled copies of the input arrays
"""

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
T0 = 288.15      # K,    sea-level temperature
P0 = 101325.0    # Pa,   sea-level pressure
G  = 9.80665     # m/s2, standard gravity
RD = 287.058     # J/(kg·K), dry-air gas constant

# ---------------------------------------------------------------------------
# Layer definitions  (z in km, lapse in K/km)
# ---------------------------------------------------------------------------
_LAYERS = [
    # (z_base_km, z_top_km, lapse_K_per_km)
    (0.0,  11.0, -6.5),
    (11.0, 20.0,  0.0),
    (20.0, 32.0,  1.0),
    (32.0, 47.0,  2.8),
    (47.0, 51.0,  0.0),
    (51.0, 71.0, -2.8),
    (71.0, 86.0, -2.0),
]

# Pre-compute base T and P for each layer boundary
def _build_layer_anchors():
    anchors = []  # list of (z_base, z_top, lapse_K_km, T_base, P_base)
    T_base = T0
    P_base = P0
    for z_base, z_top, lapse in _LAYERS:
        anchors.append((z_base, z_top, lapse, T_base, P_base))
        dz = z_top - z_base          # km
        if lapse != 0.0:
            T_top = T_base + lapse * dz
            lapse_SI = lapse / 1000.0   # K/m
            P_top = P_base * (T_top / T_base) ** (-G / (lapse_SI * RD))
        else:
            T_top = T_base
            P_top = P_base * np.exp(-G * dz * 1000.0 / (RD * T_base))
        T_base = T_top
        P_base = P_top
    return anchors

_ANCHORS = _build_layer_anchors()

# ---------------------------------------------------------------------------
# Scalar helpers (operate on a single float z in km)
# ---------------------------------------------------------------------------

def _T_scalar(z):
    """Temperature (K) at geometric altitude z (km)."""
    last_z_base, last_z_top, last_lapse, last_T_base, last_P_base = _ANCHORS[-1]
    for z_base, z_top, lapse, T_base, P_base in _ANCHORS:
        if z <= z_top:
            if lapse != 0.0:
                return T_base + lapse * (z - z_base)
            else:
                return T_base
    # Above the top defined layer (86 km) — extrapolate with last layer lapse
    if last_lapse != 0.0:
        return last_T_base + last_lapse * (z - last_z_base)
    else:
        return last_T_base


def _P_scalar(z):
    """Pressure (Pa) at geometric altitude z (km)."""
    last_z_base, last_z_top, last_lapse, last_T_base, last_P_base = _ANCHORS[-1]
    for z_base, z_top, lapse, T_base, P_base in _ANCHORS:
        if z <= z_top:
            dz = z - z_base
            if lapse != 0.0:
                T_z = T_base + lapse * dz
                lapse_SI = lapse / 1000.0  # K/m
                return P_base * (T_z / T_base) ** (-G / (lapse_SI * RD))
            else:
                return P_base * np.exp(-G * dz * 1000.0 / (RD * T_base))
    # Above the top defined layer (86 km) — isothermal extrapolation from 86 km
    # Compute T and P at 86 km (top of last layer) then extrapolate isothermal
    dz_top = last_z_top - last_z_base
    if last_lapse != 0.0:
        T_top = last_T_base + last_lapse * dz_top
        lapse_SI = last_lapse / 1000.0
        P_top = last_P_base * (T_top / last_T_base) ** (-G / (lapse_SI * RD))
    else:
        T_top = last_T_base
        P_top = last_P_base * np.exp(-G * dz_top * 1000.0 / (RD * last_T_base))
    dz = z - last_z_top
    return P_top * np.exp(-G * dz * 1000.0 / (RD * T_top))


# ---------------------------------------------------------------------------
# Vectorised public functions
# ---------------------------------------------------------------------------

def T_std(z_km):
    """
    US Standard Atmosphere 1976 temperature.

    Parameters
    ----------
    z_km : float or array-like
        Geometric altitude in kilometres.

    Returns
    -------
    float or ndarray
        Temperature in K.
    """
    scalar_input = np.ndim(z_km) == 0
    z = np.atleast_1d(np.asarray(z_km, dtype=float))
    out = np.empty_like(z)
    for i, zi in enumerate(z.flat):
        out.flat[i] = _T_scalar(zi)
    return float(out[0]) if scalar_input else out


def P_std(z_km):
    """
    US Standard Atmosphere 1976 pressure.

    Parameters
    ----------
    z_km : float or array-like
        Geometric altitude in kilometres.

    Returns
    -------
    float or ndarray
        Pressure in hPa.
    """
    scalar_input = np.ndim(z_km) == 0
    z = np.atleast_1d(np.asarray(z_km, dtype=float))
    out = np.empty_like(z)
    for i, zi in enumerate(z.flat):
        out.flat[i] = _P_scalar(zi)   # Pa
    out /= 100.0                       # Pa -> hPa
    return float(out[0]) if scalar_input else out


# ---------------------------------------------------------------------------
# Gas VMR profiles
# ---------------------------------------------------------------------------

def _vmr_co2(z):
    return np.full_like(z, 415e-6, dtype=float)


def _vmr_o2(z):
    return np.full_like(z, 0.20946, dtype=float)


def _vmr_ch4(z):
    out = np.where(z <= 30.0, 1.9e-6, 1.9e-6 * np.exp(-(z - 30.0) / 15.0))
    return out.astype(float)


def _vmr_n2o(z):
    out = np.where(z <= 30.0, 0.32e-6, 0.32e-6 * np.exp(-(z - 30.0) / 20.0))
    return out.astype(float)


def _vmr_co(z):
    # 0-10 km: 0.15e-6; 10-30 km: linear decay to 0.05e-6; above 30 km: 0.05e-6
    slope = (0.05e-6 - 0.15e-6) / (30.0 - 10.0)  # K/km
    mid   = 0.15e-6 + slope * (z - 10.0)
    out   = np.where(z <= 10.0, 0.15e-6,
            np.where(z <= 30.0, mid, 0.05e-6))
    return out.astype(float)


def _vmr_o3(z):
    # 0-15 km
    seg0 = 0.06e-6 + z * (0.10e-6 / 15.0)
    # 15-35 km
    seg1 = 0.16e-6 + (z - 15.0) * (3.0e-6 / 20.0)
    # 35-50 km
    seg2 = 3.16e-6 - (z - 35.0) * (3.0e-6 / 15.0)
    # above 50 km
    seg3 = 0.16e-6 * np.exp(-(z - 50.0) / 15.0)
    out  = np.where(z <= 15.0, seg0,
           np.where(z <= 35.0, seg1,
           np.where(z <= 50.0, seg2, seg3)))
    return out.astype(float)


def _vmr_h2o(z):
    h2o_trop = 8e-3 * np.exp(-z / 2.0)
    h2o_strat = 3.0e-6
    out = np.maximum(h2o_trop, h2o_strat)
    return out.astype(float)


_VMR_FUNCS = {
    'h2o':  _vmr_h2o,
    'co2':  _vmr_co2,
    'o3':   _vmr_o3,
    'n2o':  _vmr_n2o,
    'co':   _vmr_co,
    'ch4':  _vmr_ch4,
    'o2':   _vmr_o2,
}


def gas_vmr_std(z_km, gas: str):
    """
    Standard-atmosphere volume mixing ratio for a given gas.

    Parameters
    ----------
    z_km : float or array-like
        Geometric altitude in kilometres.
    gas : str
        One of {'h2o', 'co2', 'o3', 'n2o', 'co', 'ch4', 'o2'}.

    Returns
    -------
    float or ndarray
        VMR (dimensionless mol/mol).
    """
    gas = gas.lower()
    if gas not in _VMR_FUNCS:
        raise ValueError(
            f"Unknown gas '{gas}'. Choose from {set(_VMR_FUNCS.keys())}."
        )
    scalar_input = np.ndim(z_km) == 0
    z = np.atleast_1d(np.asarray(z_km, dtype=float))
    out = _VMR_FUNCS[gas](z)
    return float(out[0]) if scalar_input else out


# ---------------------------------------------------------------------------
# NaN filling
# ---------------------------------------------------------------------------

# Map array keys to the appropriate fill source
_FILL_KEY_TO_GAS = {
    'vmr_h2o': 'h2o',
    'co2':     'co2',
    'o3':      'o3',
    'n2o':     'n2o',
    'co':      'co',
    'ch4':     'ch4',
}


def fill_nans(z_km, arrays: dict) -> dict:
    """
    Replace NaN elements in each array with the standard-atmosphere value.

    Parameters
    ----------
    z_km : array-like, shape (N,)
        Geometric altitudes in kilometres corresponding to each element.
    arrays : dict
        Keys: any subset of {'T_K', 'vmr_h2o', 'co2', 'o3', 'n2o', 'co', 'ch4'}.
        Values: array-like of length N (may contain NaNs).

    Returns
    -------
    dict
        New dict with NaN-filled copies; originals are not modified.
    """
    z = np.asarray(z_km, dtype=float)
    result = {}
    for key, arr in arrays.items():
        arr_copy = np.array(arr, dtype=float)
        mask = np.isnan(arr_copy)
        if not mask.any():
            result[key] = arr_copy
            continue
        if key == 'T_K':
            fill_vals = T_std(z[mask])
        elif key in _FILL_KEY_TO_GAS:
            fill_vals = gas_vmr_std(z[mask], _FILL_KEY_TO_GAS[key])
        else:
            raise ValueError(
                f"Unrecognised array key '{key}'. "
                f"Expected one of {{'T_K'}} | {set(_FILL_KEY_TO_GAS.keys())}."
            )
        arr_copy[mask] = fill_vals
        result[key] = arr_copy
    return result


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    print("=== US Standard Atmosphere 1976 — self-test ===\n")

    # --- Temperature spot checks ---
    cases_T = [
        (0.0,  288.15),
        (11.0, 216.65),
        (20.0, 216.65),
        (32.0, 228.65),
        (47.0, 270.65),
        (51.0, 270.65),
        (71.0, 214.65),
        (86.0, 184.65),   # geometric-altitude formulation
    ]
    print("Temperature checks (T_std):")
    failures = 0
    for z, T_ref in cases_T:
        T_calc = T_std(z)
        ok = abs(T_calc - T_ref) < 0.1
        status = "OK" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"  z={z:5.1f} km  T_ref={T_ref:.2f} K  T_calc={T_calc:.2f} K  [{status}]")

    # --- Pressure spot checks (from standard tables) ---
    cases_P = [
        (0.0,  1013.25),
        (11.0,  226.32),
        (20.0,   54.75),
        (32.0,    8.68),
        (47.0,    1.109),
        (51.0,    0.6694),
        (71.0,    0.03956),
        (86.0,    0.003024),   # geometric-altitude formulation
    ]
    print("\nPressure checks (P_std, hPa):")
    for z, P_ref in cases_P:
        P_calc = P_std(z)
        rel_err = abs(P_calc - P_ref) / P_ref
        ok = rel_err < 0.01   # within 1 %
        status = "OK" if ok else "FAIL"
        if not ok:
            failures += 1
        print(
            f"  z={z:5.1f} km  P_ref={P_ref:10.4f} hPa"
            f"  P_calc={P_calc:10.4f} hPa  rel_err={rel_err:.4f}  [{status}]"
        )

    # --- Gas VMR checks ---
    print("\nGas VMR spot checks (gas_vmr_std):")
    vmr_cases = [
        ('co2',  0.0,  415e-6),
        ('o2',   5.0,  0.20946),
        ('ch4',  0.0,  1.9e-6),
        ('ch4', 45.0,  1.9e-6 * np.exp(-15.0/15.0)),
        ('n2o',  0.0,  0.32e-6),
        ('n2o', 50.0,  0.32e-6 * np.exp(-20.0/20.0)),
        ('co',   5.0,  0.15e-6),
        ('co',  20.0,  0.15e-6 + (0.05e-6-0.15e-6)/20.0*10.0),
        ('co',  35.0,  0.05e-6),
        ('o3',   0.0,  0.06e-6),
        ('o3',  35.0,  3.16e-6),
        ('h2o',  0.0,  8e-3),
        ('h2o', 20.0,  3e-6),
    ]
    for gas, z, ref in vmr_cases:
        val = gas_vmr_std(z, gas)
        rel = abs(val - ref) / max(abs(ref), 1e-30)
        ok = rel < 1e-6
        status = "OK" if ok else "FAIL"
        if not ok:
            failures += 1
        print(
            f"  {gas:4s}  z={z:5.1f} km  ref={ref:.4e}  calc={val:.4e}  [{status}]"
        )

    # --- Array / vectorised usage ---
    print("\nVectorised call check:")
    z_arr = np.array([0.0, 5.0, 10.0, 20.0, 50.0])
    T_arr = T_std(z_arr)
    P_arr = P_std(z_arr)
    print(f"  z    = {z_arr}")
    print(f"  T(K) = {T_arr}")
    print(f"  P(hPa)= {P_arr}")
    assert T_arr.shape == z_arr.shape, "Shape mismatch for T array"
    assert P_arr.shape == z_arr.shape, "Shape mismatch for P array"
    print("  Shape checks OK")

    # --- fill_nans ---
    print("\nfill_nans check:")
    z_test = np.array([0.0, 5.0, 15.0])
    T_with_nan = np.array([300.0, np.nan, np.nan])
    co2_with_nan = np.array([np.nan, 400e-6, np.nan])
    filled = fill_nans(z_test, {'T_K': T_with_nan, 'co2': co2_with_nan})
    # Original arrays unmodified
    assert np.isnan(T_with_nan[1]), "Original T array was modified"
    # Filled values make sense
    T_filled = filled['T_K']
    co2_filled = filled['co2']
    assert T_filled[0] == 300.0,           "Non-NaN value changed"
    assert abs(T_filled[1] - T_std(5.0)) < 1e-6,  "NaN fill for T wrong at 5 km"
    assert abs(T_filled[2] - T_std(15.0)) < 1e-6, "NaN fill for T wrong at 15 km"
    assert abs(co2_filled[0] - 415e-6) < 1e-12,   "NaN fill for co2 wrong at 0 km"
    assert co2_filled[1] == 400e-6,                "Non-NaN co2 value changed"
    print("  fill_nans OK")

    print(f"\n{'All checks passed.' if failures == 0 else f'{failures} check(s) FAILED.'}")
    sys.exit(0 if failures == 0 else 1)
