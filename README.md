# pyrrtm-lite (bug-fixed)

A streamlined, self-contained driver for RRTM_LW and RRTM_SW that reads
CF-NetCDF soundings and produces broadband LW + SW flux profiles on a
user-specified vertical grid.

This version corrects **12 bugs** that were introduced during the Python
translation of the original Fortran RRTM code.  All bugs were in the Python
port — the Fortran RRTM itself is correct.  See [`pyrrtmlite_bugs.txt`](pyrrtmlite_bugs.txt)
for the full audit.

---

## What it does

- Reads a time-varying (or single-profile) CF-NetCDF atmospheric sounding
- Interpolates to a user-specified vertical grid: uniform below 15 km, 1-km
  above 15 km up to 50 km
- Runs bug-fixed LW and SW radiative transfer for every profile in parallel
- Writes broadband flux profiles to an output NetCDF; all config settings are
  embedded as global attributes for full reproducibility

---

## Bug fixes included

All 12 bugs were introduced during Python translation of the Fortran RRTM.
None exist in the original Fortran code.

| # | Severity | What was wrong | Impact |
|---|----------|---------------|--------|
| 1 | CRITICAL | Aerosol altitude formula: `44.3×ln(p₀/p)` mixed constants from two incompatible formulae | Aerosol forcing silently zeroed for any altitude-bounded aerosol layer |
| 2 | CRITICAL | `coldry = wbroad.copy()` included H₂O in the "dry air" column | ~3% systematic error in all gas column amounts; largest in humid/tropical air |
| 3 | CRITICAL | SW heating rate sign inverted (`totuflux−totdflux` instead of `totdflux−totuflux`) | All SW heating rates reported with wrong sign — troposphere showed cooling instead of warming |
| 4 | CRITICAL | `laysolfr` guard `lay > nl` skipped the first upper-atmosphere layer | Solar reference flux wrong in 5 SW bands; affects every SW calculation |
| 6 | MAJOR | SW cloud optical depth missing delta-M scaling | Cloud SW optical depths ~7–10% too large |
| 7 | MAJOR | Cloud arrays stored at `nlayers+1` (level grid) instead of `nlayers` (layer grid) | Shape mismatch; extra element in cloud optical depth arrays |
| 12 | MAJOR | `cos(90°) = 6×10⁻¹⁷` in IEEE 754 bypassed the `≤ 0` zenith guard | SZA=90° proceeded with near-zero zenith, causing numerical overflow in direct beam |
| 17 | MAJOR | USSA76 H₂O fill had 6.6× step discontinuity at 12 km | Spurious heating/cooling signal at tropopause in profiles with NaN water vapour above 12 km |
| 19 | MAJOR | Aerosol Planck fraction = 1.0 (grey body) instead of local gas Planck fraction | LW aerosol emission counted in spectrally wrong g-points; error grows with aerosol optical depth |
| 21 | MINOR | SW `layswtch` incorrectly incremented (Fortran never does this) | Wrong transition-layer index in SW taumol |
| 25 | MINOR | LW `colso2` not computed in setcoef | SO₂ column absent from optical depth calculation |
| 26 | MINOR | `adjflux` was a dead parameter in rtrdis (never applied) | Misleading API; callers expecting rtrdis to apply Earth-Sun correction would get wrong SW |

---

## Quantified impact of fixes (4 AFGL standard atmospheres, SZA=45°)

Tested on MLS (Mid-Lat Summer), MLW (Mid-Lat Winter), SAW (Sub-Arctic Winter),
TROP (Tropical). See `bug_fix_comparison.png` and `bug_fix_delta.png`.

| Quantity | Mean Δ (fixed − buggy) | Max Δ | Notes |
|----------|------------------------|-------|-------|
| LW↓ surface | **+0.28 W/m²** | +0.63 (TROP) | Bug 2: coldry fix adds H₂O gas column |
| LW↑ TOA (OLR) | −0.03 W/m² | −0.08 (TROP) | Small, sub-0.1 W/m² |
| SW↓ surface | −9.3 W/m² | −37.8 (TROP) | Bug 4: laysolfr correction changes solar ref flux |
| SW↑ TOA | −4.8 W/m² | −19.1 (TROP) | Same driver as SW↓ |
| SW htr @ 3 km | **+3.5 K/day** | +4.4 (MLS) | Bug 3: sign now correct (positive = solar warming) |

Key result: **SW heating rates are now physically correct** — all four
atmospheres show positive solar warming in the troposphere (buggy code gave
negative values for all). The tropical atmosphere shows the largest SW flux
change (−38 W/m² surface) due to the laysolfr correction in the high-water-vapour
SW bands.

---

## Output variables (W m⁻²)

| Variable | Description |
|----------|-------------|
| `lw_dn` | Downwelling LW, all-sky |
| `lw_up` | Upwelling LW, all-sky |
| `sw_dn` | Downwelling SW, all-sky |
| `sw_up` | Upwelling SW, all-sky |
| `lw_dn_clr` | Downwelling LW, clear-sky *(if `save_clearsky=true`)* |
| `lw_up_clr` | Upwelling LW, clear-sky |
| `sw_dn_clr` | Downwelling SW, clear-sky |
| `sw_up_clr` | Upwelling SW, clear-sky |
| `height` | Height AGL [km] |
| `sza` | Solar zenith angle [degrees] |

All config settings (albedo, CO₂, vertical grid, etc.) are stored as global
attributes in the output file for full reproducibility.

---

## Requirements

```
numpy >= 1.20
scipy >= 1.7
netCDF4 >= 1.5
```

No external pyrrtm package required — LW and SW solvers are self-contained
in the `lw/` and `sw/` subdirectories.

---

## Usage

```bash
# Edit the template config
cp config_template.json my_config.json

# Run (output defaults to <input_stem>_fluxes.nc)
python pyrrtm_lite.py --config my_config.json --input sounding.nc

# Or specify output explicitly
python pyrrtm_lite.py --config my_config.json --input sounding.nc --output fluxes.nc
```

---

## Config fields

| Field | Default | Description |
|-------|---------|-------------|
| `surface.sw_albedo` | `0.15` | Broadband SW surface albedo (0–1) |
| `surface.skin_temperature_C` | `null` | Surface skin T [°C]; `null` = use lowest-level air T |
| `gases.CO2_ppm` | `422.0` | CO₂ mixing ratio [ppm] |
| `gases.CH4_ppm` | `1.9` | CH₄ mixing ratio [ppm] |
| `solar.latitude_deg` | `null` | Latitude [°N]; `null` = read from sounding file |
| `solar.longitude_deg` | `null` | Longitude [°E]; `null` = read from sounding file |
| `solar.fixed_sza_deg` | `null` | Fix SZA for all profiles (overrides computed) |
| `vertical_grid.resolution_below_15km_km` | `0.5` | Output grid spacing below 15 km [km] |
| `output.save_clearsky` | `true` | Also write `_clr` clear-sky flux variables |
| `processing.n_cpu` | `4` | CPU cores; `-1` = all available |

---

## Vertical grid

```
0 – 15 km  : user-specified resolution (e.g. 0.5 km → 30 levels)
15 – 50 km : fixed 1-km resolution (35 levels)
```

Profiles above the sonde top are filled with USSA76 values and monotonic
pressure is enforced to prevent numerical failures at layer boundaries.

---

## Repository structure

```
pyrrtm_fixed/
├── pyrrtm_lite.py        Main driver script
├── config_template.json  Documented config template
├── lw/                   Bug-fixed LW (RRTM_LW v3.3.1) modules
├── sw/                   Bug-fixed SW (RRTM_SW v2.7.2) modules
├── data/
│   ├── lw_coeffs.nc      LW k-coefficient database
│   └── sw_coeffs.nc      SW k-coefficient database
├── pyrrtmlite_bugs.txt   Full bug audit (28 bugs, 12 present, 1 fixed, 5 N/A)
├── test_bug_impact.py    Before/after radiation comparison test
├── bug_fix_comparison.png  Flux comparison plots
└── bug_fix_delta.png       Flux change (fixed − buggy) plots
```

---

## Notes

- Cloud fields (`frac`, `lwp_gm2`, `iwp_gm2`, `re_liq_um`, `re_ice_um`) are
  read from the sounding NetCDF if present; if absent all profiles are clear-sky.
- O₃, N₂O, CO are filled from the US Standard Atmosphere 1976 at each level.
- Nighttime profiles (SZA ≥ 90°) automatically receive SW = 0.

---

## Reference

Mlawer, E. J., Taubman, S. J., Brown, P. D., Iacono, M. J., and Clough, S. A.
(1997): Radiative transfer for inhomogeneous atmospheres: RRTM, a validated
correlated-k model for the longwave. *J. Geophys. Res.*, 102, 16663–16682.
