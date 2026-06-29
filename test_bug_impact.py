"""
Before/After radiation test: pyrrtm (buggy) vs pyrrtm_fixed (bug-fixed).

Tests four standard AFGL atmospheres (MLS, MLW, SAW, TROP) to quantify
the impact of 12 bug fixes on surface and TOA broadband fluxes.

Bugs tested:
  #1  Aerosol altitude formula wrong constant
  #2  coldry includes H2O (LW+SW gas column ~3% off in humid air)
  #3  SW heating rate sign inverted
  #4  laysolfr guard skips first upper-atm layer
  #6  SW cloud missing delta-M scaling
  #7  Cloud arrays nlayers+1 instead of nlayers
  #12 cos(SZA=90°) bypasses zenith guard
  #17 H2O USSA76 6.6× discontinuity at 12 km
  #19 Aerosol Planck fraction = 1.0 (grey body)
  #21 SW layswtch incorrectly incremented
  #25 COLSO2 not computed in LW setcoef
  #26 adjflux dead parameter in rtrdis
"""

import sys, warnings
import numpy as np

warnings.filterwarnings('ignore')

# ── BUGGY: original pyrrtm ──────────────────────────────────────────────────
PYRRTM_DIR  = '/home/vghate/Desktop/pyrrtm'
RRTM_LW_DIR = '/home/vghate/Desktop/rrtm_proj'
RRTM_SW_DIR = '/home/vghate/Desktop/rrtm_sw_proj'

for p in [RRTM_LW_DIR, RRTM_SW_DIR, PYRRTM_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from pyrrtm.lw.run      import run as lw_buggy
from pyrrtm.lw.sounding import Sounding as LW_Buggy
from pyrrtm.sw.run      import run as sw_buggy
from pyrrtm.sw.sounding import SWSounding as SW_Buggy
from pyrrtm.lw.std_atm  import gas_vmr_std as gvmr_buggy, P_std

LW_COEFFS = PYRRTM_DIR + '/data/lw_coeffs.nc'
SW_COEFFS = PYRRTM_DIR + '/data/sw_coeffs.nc'

# ── FIXED: pyrrtm_fixed local modules ───────────────────────────────────────
FIXED_DIR = '/home/vghate/Desktop/pyrrtm_fixed'
# Insert BEFORE pyrrtm so local lw/sw take precedence
sys.path.insert(0, FIXED_DIR)

# Clear any cached lw/sw modules from pyrrtm import
for key in list(sys.modules.keys()):
    if key in ('lw', 'sw') or key.startswith('lw.') or key.startswith('sw.'):
        del sys.modules[key]

from lw.run      import run as lw_fixed
from lw.sounding import Sounding as LW_Fixed
from sw.run      import run as sw_fixed
from sw.sounding import SWSounding as SW_Fixed
from lw.std_atm  import gas_vmr_std as gvmr_fixed

LW_COEFFS_F = FIXED_DIR + '/data/lw_coeffs.nc'
SW_COEFFS_F = FIXED_DIR + '/data/sw_coeffs.nc'


# ── AFGL standard atmosphere profiles (z in km, T in K, wv in g/kg) ─────────
# Source: Anderson et al. (1986) AFGL Atmospheric Constituent Profiles
ATMOS = {
    'MLS': {   # Mid-Latitude Summer
        'z': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,17,20,25,30,40,50],
        'T_K': [294.0,290.0,285.0,279.0,273.0,267.0,261.0,254.0,247.0,
                240.0,234.0,225.0,217.0,217.0,217.0,217.5,221.5,234.0,
                247.0,262.0,270.0],
        'wv_gkg': [14.95,11.08,7.44,5.11,3.18,1.91,1.02,0.546,0.266,0.139,
                   0.067,0.036,0.022,0.018,0.015,0.013,0.012,0.0050,
                   0.0030,0.0024,0.0020],
    },
    'MLW': {   # Mid-Latitude Winter
        'z': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,17,20,25,30,40,50],
        'T_K': [272.2,265.8,259.0,252.0,245.1,238.4,231.9,225.5,219.1,212.8,
                207.0,201.0,196.0,193.0,193.0,194.0,199.0,216.0,234.0,252.0,263.0],
        'wv_gkg': [3.50,2.49,1.79,1.29,0.872,0.612,0.435,0.266,0.168,0.097,
                   0.050,0.026,0.015,0.010,0.0060,0.0034,0.0024,0.0017,
                   0.0016,0.0014,0.0012],
    },
    'SAW': {   # Sub-Arctic Winter
        'z': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,17,20,25,30,40,50],
        'T_K': [257.1,259.1,255.9,252.7,247.7,240.9,234.1,227.5,220.9,217.3,
                217.3,217.3,217.3,217.3,217.3,216.3,216.3,221.3,235.3,255.3,264.3],
        'wv_gkg': [1.19,1.004,0.832,0.679,0.497,0.327,0.189,0.117,0.069,0.043,
                   0.027,0.018,0.014,0.010,0.0060,0.0040,0.0028,0.0016,
                   0.0014,0.0012,0.0009],
    },
    'TROP': {  # Tropical
        'z': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,17,20,25,30,40,50],
        'T_K': [300.0,294.0,288.0,282.0,275.8,269.5,262.9,256.2,249.4,242.4,
                235.5,228.5,221.6,215.0,202.5,194.5,188.5,202.5,220.5,253.5,266.0],
        'wv_gkg': [19.0,13.0,9.3,6.4,4.2,2.7,1.7,1.0,0.60,0.35,0.20,0.12,
                   0.075,0.052,0.033,0.022,0.016,0.0076,0.0045,0.0028,0.0020],
    },
}

CO2_PPM = 360.0   # ~year-2000 value matching original RRTM test cases

def build_gas(z_km, gvmr_fn, co2=CO2_PPM):
    nlev = len(z_km)
    o3 = gvmr_fn(z_km, 'o3')
    return {
        'co2': np.full(nlev, co2),
        'o3':  o3 * 1e6,
        'n2o': np.full(nlev, 0.314),
        'ch4': np.full(nlev, 1.77),
        'co':  np.full(nlev, 0.15),
    }


# ── Run comparisons ──────────────────────────────────────────────────────────
print('=' * 76)
print('pyrrtm (buggy) vs pyrrtm_fixed — surface and TOA flux comparison')
print('Standard AFGL atmospheres: MLS, MLW, SAW, TROP | SZA=45° for SW')
print('=' * 76)

results = []

for name, atm in ATMOS.items():
    z   = np.array(atm['z'],    dtype=float)
    T_C = np.array(atm['T_K'],  dtype=float) - 273.15
    wv  = np.array(atm['wv_gkg'], dtype=float)
    P   = P_std(z)   # hPa, already correct units

    gas_b = build_gas(z, gvmr_buggy)
    gas_f = build_gas(z, gvmr_fixed)

    # ── LW ──────────────────────────────────────────────────────────────────
    snd_lw_b = LW_Buggy(z_km=z, T_C=T_C, P_hPa=P, wv_gkg=wv, gas_ppm=gas_b)
    snd_lw_f = LW_Fixed(z_km=z, T_C=T_C, P_hPa=P, wv_gkg=wv, gas_ppm=gas_f)

    r_lw_b = lw_buggy(sounding=snd_lw_b, coeffs_path=LW_COEFFS)
    r_lw_f = lw_fixed(sounding=snd_lw_f, coeffs_path=LW_COEFFS_F)

    lw_dn_sfc_b = float(r_lw_b['totdflux'][0])
    lw_up_toa_b = float(r_lw_b['totuflux'][-1])
    lw_dn_sfc_f = float(r_lw_f['totdflux'][0])
    lw_up_toa_f = float(r_lw_f['totuflux'][-1])
    lw_htr_3km_b = float(r_lw_b['htr'][3])   # 3 km layer
    lw_htr_3km_f = float(r_lw_f['htr'][3])

    # ── SW (SZA=45°) ────────────────────────────────────────────────────────
    alb  = np.full(14, 0.10)
    snd_sw_b = SW_Buggy(z_km=z, T_C=T_C, P_hPa=P, wv_gkg=wv,
                        gas_ppm=gas_b, sza_deg=45.0, albedo=alb)
    snd_sw_f = SW_Fixed(z_km=z, T_C=T_C, P_hPa=P, wv_gkg=wv,
                        gas_ppm=gas_f, sza_deg=45.0, albedo=alb)

    r_sw_b = sw_buggy(sounding=snd_sw_b, coeffs_path=SW_COEFFS)
    r_sw_f = sw_fixed(sounding=snd_sw_f, coeffs_path=SW_COEFFS_F)

    sw_dn_sfc_b = float(r_sw_b['totdflux'][0])
    sw_up_toa_b = float(r_sw_b['totuflux'][-1])
    sw_dn_sfc_f = float(r_sw_f['totdflux'][0])
    sw_up_toa_f = float(r_sw_f['totuflux'][-1])

    # SW heating rate at ~3 km (should be POSITIVE = warming for solar absorption)
    sw_htr_3km_b = float(r_sw_b['htr'][3])
    sw_htr_3km_f = float(r_sw_f['htr'][3])

    results.append({
        'atm':        name,
        'T_sfc_C':    float(T_C[0]),
        'lw_dn_sfc_b': lw_dn_sfc_b, 'lw_dn_sfc_f': lw_dn_sfc_f,
        'lw_up_toa_b': lw_up_toa_b, 'lw_up_toa_f': lw_up_toa_f,
        'lw_htr_3km_b': lw_htr_3km_b, 'lw_htr_3km_f': lw_htr_3km_f,
        'sw_dn_sfc_b': sw_dn_sfc_b, 'sw_dn_sfc_f': sw_dn_sfc_f,
        'sw_up_toa_b': sw_up_toa_b, 'sw_up_toa_f': sw_up_toa_f,
        'sw_htr_3km_b': sw_htr_3km_b, 'sw_htr_3km_f': sw_htr_3km_f,
    })

    dln = lw_dn_sfc_f - lw_dn_sfc_b
    dlt = lw_up_toa_f - lw_up_toa_b
    dsn = sw_dn_sfc_f - sw_dn_sfc_b
    dst = sw_up_toa_f - sw_up_toa_b

    print(f'\n{name}  (T_sfc={T_C[0]:.1f}°C)')
    print(f'  {"":40s}  {"Buggy":>8}  {"Fixed":>8}  {"Δ":>8}')
    print(f'  {"LW↓ surface (W/m²)":40s}  {lw_dn_sfc_b:8.2f}  {lw_dn_sfc_f:8.2f}  {dln:+8.2f}')
    print(f'  {"LW↑ TOA / OLR (W/m²)":40s}  {lw_up_toa_b:8.2f}  {lw_up_toa_f:8.2f}  {dlt:+8.2f}')
    print(f'  {"LW heating rate @ 3km (K/day)":40s}  {lw_htr_3km_b:8.3f}  {lw_htr_3km_f:8.3f}  {lw_htr_3km_f-lw_htr_3km_b:+8.3f}')
    print(f'  {"SW↓ surface, SZA=45° (W/m²)":40s}  {sw_dn_sfc_b:8.2f}  {sw_dn_sfc_f:8.2f}  {dsn:+8.2f}')
    print(f'  {"SW↑ TOA, SZA=45° (W/m²)":40s}  {sw_up_toa_b:8.2f}  {sw_up_toa_f:8.2f}  {dst:+8.2f}')
    print(f'  {"SW heating rate @ 3km (K/day)":40s}  {sw_htr_3km_b:8.3f}  {sw_htr_3km_f:8.3f}  '
          f'  sign_ok={sw_htr_3km_f>0}')


# ── Summary ──────────────────────────────────────────────────────────────────
print('\n\n' + '=' * 76)
print('SUMMARY — Mean change across 4 AFGL atmospheres')
print('=' * 76)
keys = [('LW↓ surface',        'lw_dn_sfc'),
        ('LW↑ TOA (OLR)',      'lw_up_toa'),
        ('LW htr @ 3km K/day', 'lw_htr_3km'),
        ('SW↓ surface SZA=45', 'sw_dn_sfc'),
        ('SW↑ TOA SZA=45',     'sw_up_toa'),
        ('SW htr @ 3km K/day', 'sw_htr_3km')]
for label, key in keys:
    vals_b = [r[key+'_b'] for r in results]
    vals_f = [r[key+'_f'] for r in results]
    d = np.array(vals_f) - np.array(vals_b)
    print(f'  {label:<30}  Δ_mean={d.mean():+7.3f}  '
          f'Δ_max={d.max():+7.3f}  Δ_min={d.min():+7.3f}  W/m²')

sw_sign_b = sum(r['sw_htr_3km_b'] > 0 for r in results)
sw_sign_f = sum(r['sw_htr_3km_f'] > 0 for r in results)
print(f'\n  Bug 3 (SW htr sign): buggy {sw_sign_b}/4 positive → fixed {sw_sign_f}/4 positive')


# ── Per-bug impact narrative ──────────────────────────────────────────────────
print('\n' + '=' * 76)
print('PER-BUG IMPACT ANALYSIS')
print('=' * 76)

for r in results:
    nm = r['atm']
    lw_delta = r['lw_dn_sfc_f'] - r['lw_dn_sfc_b']
    sw_delta = r['sw_dn_sfc_f'] - r['sw_dn_sfc_b']
    olr_delta = r['lw_up_toa_f'] - r['lw_up_toa_b']
    htr_fix = r['sw_htr_3km_f'] > 0
    print(f'\n  {nm}:')
    print(f'    Bug 2 (coldry+H2O) → LW↓sfc Δ = {lw_delta:+.3f} W/m²')
    print(f'    Bug 2 (coldry+H2O) → OLR Δ    = {olr_delta:+.3f} W/m²')
    print(f'    Bug 3 (htr sign)   → SW htr sign now correct: {htr_fix}')
    print(f'    All bugs combined  → SW↓sfc Δ  = {sw_delta:+.3f} W/m²')


# ── Save CSV ──────────────────────────────────────────────────────────────────
import csv
csv_path = FIXED_DIR + '/bug_fix_comparison.csv'
with open(csv_path, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=results[0].keys())
    w.writeheader(); w.writerows(results)
print(f'\nCSV: {csv_path}')


# ── Plots ─────────────────────────────────────────────────────────────────────
try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    atm_names = [r['atm'] for r in results]
    x = np.arange(len(atm_names)); w = 0.35

    # Panel 1: absolute fluxes, buggy vs fixed
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('Bug-Fix Impact — pyrrtm vs pyrrtm_fixed\n'
                 '4 AFGL Standard Atmospheres, SZA=45° for SW', fontsize=11)

    def bars(ax, b, f, title, unit='W/m²'):
        ax.bar(x-w/2, b, w, label='Buggy',  color='#d7191c', alpha=0.8, edgecolor='k', lw=0.5)
        ax.bar(x+w/2, f, w, label='Fixed',  color='#1a9641', alpha=0.8, edgecolor='k', lw=0.5)
        ax.set_xticks(x); ax.set_xticklabels(atm_names, fontsize=9)
        ax.set_title(title, fontsize=9); ax.set_ylabel(unit); ax.legend(fontsize=8)

    bars(axes[0,0], [r['lw_dn_sfc_b'] for r in results], [r['lw_dn_sfc_f'] for r in results],
         'LW↓ Surface')
    bars(axes[0,1], [r['lw_up_toa_b'] for r in results], [r['lw_up_toa_f'] for r in results],
         'LW↑ TOA (OLR)')
    bars(axes[0,2], [r['lw_htr_3km_b'] for r in results], [r['lw_htr_3km_f'] for r in results],
         'LW Heating Rate @ 3 km', 'K/day')
    bars(axes[1,0], [r['sw_dn_sfc_b'] for r in results], [r['sw_dn_sfc_f'] for r in results],
         'SW↓ Surface (SZA=45°)')
    bars(axes[1,1], [r['sw_up_toa_b'] for r in results], [r['sw_up_toa_f'] for r in results],
         'SW↑ TOA (SZA=45°)')
    bars(axes[1,2], [r['sw_htr_3km_b'] for r in results], [r['sw_htr_3km_f'] for r in results],
         'SW Heating Rate @ 3 km\n(positive = warming = correct after fix)', 'K/day')

    fig.tight_layout()
    fig.savefig(FIXED_DIR + '/bug_fix_comparison.png', dpi=150, bbox_inches='tight')
    print(f'Plot: {FIXED_DIR}/bug_fix_comparison.png')

    # Panel 2: delta (fixed - buggy)
    fig2, axes2 = plt.subplots(1, 4, figsize=(14, 4))
    fig2.suptitle('Flux Change: Fixed − Buggy (W/m²)', fontsize=11)
    metrics = [
        ([r['lw_dn_sfc_f']-r['lw_dn_sfc_b'] for r in results], 'LW↓ Surface'),
        ([r['lw_up_toa_f']-r['lw_up_toa_b'] for r in results], 'LW↑ TOA (OLR)'),
        ([r['sw_dn_sfc_f']-r['sw_dn_sfc_b'] for r in results], 'SW↓ Surface\nSZA=45°'),
        ([r['sw_up_toa_f']-r['sw_up_toa_b'] for r in results], 'SW↑ TOA\nSZA=45°'),
    ]
    for ax, (vals, title) in zip(axes2, metrics):
        cols = ['#1a9641' if v >= 0 else '#d7191c' for v in vals]
        ax.bar(atm_names, vals, color=cols, alpha=0.85, edgecolor='k', lw=0.5)
        ax.axhline(0, color='k', lw=0.8)
        ax.set_title(title, fontsize=9); ax.set_ylabel('Δ W/m²')
        for i, v in enumerate(vals):
            ax.text(i, v + (max(abs(max(vals,key=abs))*0.02, 0.01)) * (1 if v >= 0 else -2),
                    f'{v:+.2f}', ha='center', fontsize=8)
    fig2.tight_layout()
    fig2.savefig(FIXED_DIR + '/bug_fix_delta.png', dpi=150, bbox_inches='tight')
    print(f'Delta plot: {FIXED_DIR}/bug_fix_delta.png')

except Exception as e:
    print(f'Plot failed: {e}')
