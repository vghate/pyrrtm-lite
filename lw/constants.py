"""
Physical constants, reference grids, and lookup tables for RRTM_LW.

All values taken verbatim from the Fortran source (rrtm.f, setcoef.f).
Units are CGS throughout (matching the Fortran compilation flags -r8 -i8).
"""
import numpy as np

# ── Physical constants (from rrtm.f DATA statements) ─────────────────────────
PI      = 3.1415926535
PLANCK  = 6.62606876e-27    # erg·s
BOLTZ   = 1.3806503e-16     # erg/K
CLIGHT  = 2.99792458e+10    # cm/s
AVOGAD  = 6.02214199e+23    # mol⁻¹
ALOSMT  = 2.6867775e+19     # cm⁻³ (Loschmidt)
GASCON  = 8.314472e+07      # erg/(mol·K)
RADCN1  = 1.191042722e-12   # 2hν³/c² prefactor
RADCN2  = 1.4387752         # hc/k  [cm·K]
GRAV    = 9.80665e+02       # cm/s²
CPDAIR  = 1.00464           # J/(g·K)
AIRMWT  = 28.964            # g/mol
SECDY   = 8.64e+04          # s/day

# Derived constants
ONEMINUS = 1.0 - 1.0e-6
FLUXFAC  = PI * 2.0e4                         # radiance → flux conversion
HEATFAC  = 1.0e-7 * (GRAV * SECDY) / CPDAIR  # ≈ 8.434  K·day⁻¹ / (W·m⁻²·mb⁻¹)

# Pade approximation constant
PADE  = 0.278
BPADE = 1.0 / PADE

# ── Reference pressure / temperature grids (from setcoef.f) ──────────────────
# 59 levels; ln(PREF[0]) = 6.96, each subsequent differs by -0.2
PREF = np.array([
    1.05363e+03, 8.62642e+02, 7.06272e+02, 5.78246e+02, 4.73428e+02,
    3.87610e+02, 3.17348e+02, 2.59823e+02, 2.12725e+02, 1.74164e+02,
    1.42594e+02, 1.16746e+02, 9.55835e+01, 7.82571e+01, 6.40715e+01,
    5.24573e+01, 4.29484e+01, 3.51632e+01, 2.87892e+01, 2.35706e+01,
    1.92980e+01, 1.57998e+01, 1.29358e+01, 1.05910e+01, 8.67114e+00,
    7.09933e+00, 5.81244e+00, 4.75882e+00, 3.89619e+00, 3.18993e+00,
    2.61170e+00, 2.13828e+00, 1.75067e+00, 1.43333e+00, 1.17351e+00,
    9.60789e-01, 7.86628e-01, 6.44036e-01, 5.27292e-01, 4.31710e-01,
    3.53455e-01, 2.89384e-01, 2.36928e-01, 1.93980e-01, 1.58817e-01,
    1.30029e-01, 1.06458e-01, 8.71608e-02, 7.13612e-02, 5.84256e-02,
    4.78349e-02, 3.91639e-02, 3.20647e-02, 2.62523e-02, 2.14936e-02,
    1.75975e-02, 1.44076e-02, 1.17959e-02, 9.65769e-03,
], dtype=np.float64)

PREFLOG = np.array([
     6.9600e+00,  6.7600e+00,  6.5600e+00,  6.3600e+00,  6.1600e+00,
     5.9600e+00,  5.7600e+00,  5.5600e+00,  5.3600e+00,  5.1600e+00,
     4.9600e+00,  4.7600e+00,  4.5600e+00,  4.3600e+00,  4.1600e+00,
     3.9600e+00,  3.7600e+00,  3.5600e+00,  3.3600e+00,  3.1600e+00,
     2.9600e+00,  2.7600e+00,  2.5600e+00,  2.3600e+00,  2.1600e+00,
     1.9600e+00,  1.7600e+00,  1.5600e+00,  1.3600e+00,  1.1600e+00,
     9.6000e-01,  7.6000e-01,  5.6000e-01,  3.6000e-01,  1.6000e-01,
    -4.0000e-02, -2.4000e-01, -4.4000e-01, -6.4000e-01, -8.4000e-01,
    -1.0400e+00, -1.2400e+00, -1.4400e+00, -1.6400e+00, -1.8400e+00,
    -2.0400e+00, -2.2400e+00, -2.4400e+00, -2.6400e+00, -2.8400e+00,
    -3.0400e+00, -3.2400e+00, -3.4400e+00, -3.6400e+00, -3.8400e+00,
    -4.0400e+00, -4.2400e+00, -4.4400e+00, -4.6400e+00,
], dtype=np.float64)

# Reference temperatures for the 59 MLS pressure levels
TREF = np.array([
    2.9420e+02, 2.8799e+02, 2.7894e+02, 2.6925e+02, 2.5983e+02,
    2.5017e+02, 2.4077e+02, 2.3179e+02, 2.2306e+02, 2.1578e+02,
    2.1570e+02, 2.1570e+02, 2.1570e+02, 2.1706e+02, 2.1858e+02,
    2.2018e+02, 2.2174e+02, 2.2328e+02, 2.2479e+02, 2.2655e+02,
    2.2834e+02, 2.3113e+02, 2.3401e+02, 2.3703e+02, 2.4022e+02,
    2.4371e+02, 2.4726e+02, 2.5085e+02, 2.5457e+02, 2.5832e+02,
    2.6216e+02, 2.6606e+02, 2.6999e+02, 2.7340e+02, 2.7536e+02,
    2.7568e+02, 2.7372e+02, 2.7163e+02, 2.6955e+02, 2.6593e+02,
    2.6211e+02, 2.5828e+02, 2.5360e+02, 2.4854e+02, 2.4348e+02,
    2.3809e+02, 2.3206e+02, 2.2603e+02, 2.2000e+02, 2.1435e+02,
    2.0887e+02, 2.0340e+02, 1.9792e+02, 1.9290e+02, 1.8809e+02,
    1.8329e+02, 1.7849e+02, 1.7394e+02, 1.7212e+02,
], dtype=np.float64)

# ── Pade transmittance lookup tables ─────────────────────────────────────────
# NTBL = 10000; tables indexed 0..NTBL
NTBL   = 10000
TBLINT = 10000.0

def _build_pade_tables():
    tautbl = np.empty(NTBL + 1, dtype=np.float64)
    trans  = np.empty(NTBL + 1, dtype=np.float64)
    tf     = np.empty(NTBL + 1, dtype=np.float64)

    tautbl[0]    = 0.0
    tautbl[NTBL] = 1.0e10
    trans[0]     = 1.0
    trans[NTBL]  = 0.0
    tf[0]        = 0.0
    tf[NTBL]     = 1.0

    itr = np.arange(1, NTBL, dtype=np.float64)
    tfn = itr / NTBL
    tautbl[1:NTBL] = BPADE * tfn / (1.0 - tfn)
    trans[1:NTBL]  = np.exp(-tautbl[1:NTBL])

    thin = tautbl[1:NTBL] < 0.06
    tf[1:NTBL][thin]  = tautbl[1:NTBL][thin] / 6.0
    tf[1:NTBL][~thin] = (1.0 - 2.0 * (
        1.0 / tautbl[1:NTBL][~thin]
        - trans[1:NTBL][~thin] / (1.0 - trans[1:NTBL][~thin])
    ))

    return tautbl, trans, tf


TAUTBL, TRANS, TF = _build_pade_tables()

# ── Tropopause pressure threshold ─────────────────────────────────────────────
# Layers with ln(p) > 4.56 (p > ~96 mb) are in the lower atmosphere (use KA).
# Layers at or above this threshold use KB.
LAYTROP_PLOG_THRESHOLD = 4.56   # ln(pressure in mb)
