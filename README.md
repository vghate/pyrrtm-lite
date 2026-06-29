# pyrrtm-lite

A streamlined driver for the pyrrtm radiative transfer model that reads
CF-NetCDF soundings and produces broadband LW + SW flux profiles.

## What it does

* Reads a time-varying (or single-profile) CF-NetCDF atmospheric sounding
* Interpolates to a user-specified vertical grid: uniform below 15 km, 1-km
  above 15 km up to 50 km
* Runs pyrrtm LW and SW for every profile in parallel
* Writes broadband flux profiles to an output NetCDF

## Output variables (W m⁻²)

| Variable | Description |
|----------|-------------|
| `lw_dn` | Downwelling LW all-sky |
| `lw_up` | Upwelling LW all-sky |
| `sw_dn` | Downwelling SW all-sky |
| `sw_up` | Upwelling SW all-sky |
| `lw_dn_clr` | Downwelling LW clear-sky *(if save_clearsky=true)* |
| `lw_up_clr` | Upwelling LW clear-sky *(if save_clearsky=true)* |
| `sw_dn_clr` | Downwelling SW clear-sky *(if save_clearsky=true)* |
| `sw_up_clr` | Upwelling SW clear-sky *(if save_clearsky=true)* |
| `height` | Height AGL [km] |
| `sza` | Solar zenith angle [degrees] |

## Requirements

```
numpy  scipy  netCDF4  pyrrtm (parent package)
```

pyrrtm must be on PYTHONPATH or placed in the parent directory of pyrrtm-lite.

## Usage

The config file can be **JSON** or **NetCDF** — both carry the same settings.

### Option A — JSON config

```bash
cp config_template.json my_config.json
# edit values; _c_* hint lines are ignored automatically
python pyrrtm_lite.py --config my_config.json --input sounding.nc --output fluxes.nc
```

### Option B — NetCDF config

```bash
# generate an editable template
python pyrrtm_lite.py --make-nc-config my_config.nc

# edit with ncview, xarray, or any NetCDF tool, then run
python pyrrtm_lite.py --config my_config.nc --input sounding.nc --output fluxes.nc
```

In the NetCDF config each setting is a scalar variable with a `description`
attribute.  Set any variable to **-9999** to use the built-in default.

## Config fields

| Field | Default | Description |
|-------|---------|-------------|
| `surface.sw_albedo` | `0.15` | Broadband SW surface albedo (0–1) |
| `surface.skin_temperature_C` | `null` | Surface skin T [°C]; null = use lowest-level air T |
| `gases.CO2_ppm` | `422.0` | CO2 mixing ratio [ppm] |
| `gases.CH4_ppm` | `1.9` | CH4 mixing ratio [ppm] |
| `solar.latitude_deg` | `null` | Latitude [°N]; null = read from sounding |
| `solar.longitude_deg` | `null` | Longitude [°E]; null = read from sounding |
| `solar.fixed_sza_deg` | `null` | Override all SZA with this value |
| `vertical_grid.resolution_below_15km_km` | `0.5` | Output grid spacing below 15 km [km] |
| `output.save_clearsky` | `true` | Also write clear-sky fluxes |
| `processing.n_cpu` | `4` | CPU cores; -1 = all available |

## Vertical grid

```
0–15 km  : user-specified resolution (e.g. 0.5 km → 30 levels)
15–50 km : fixed 1-km resolution (35 levels)
```

Total levels: 30 + 35 = 65 (at 0.5 km resolution below 15 km).

## Example: ARM TRACER sonde (1440 profiles, 4 cores)

```
pyrrtm-lite  |  dz<15km=0.5km  cores=4  clear-sky=yes
Reading sounding: houinterpolatedsondeM1.c1.20211008.000030.nc
  1440 profiles × 332 source levels
  Output grid: 66 levels  (0.5 km / 1 km grid)
Running 1440 profiles on 4 cores...
Done.
  Surface LW↓ mean  : 358.2 W/m²  (obs 365.4 W/m²)
  Surface SW↓ (day) : 560.3 W/m²  (obs 509.7 W/m²)
```

## Notes

* Cloud fields (frac, lwp_gm2, iwp_gm2, re_liq_um, re_ice_um) are read from
  the sounding NetCDF if present. If absent, all profiles are clear-sky.
* O3, N2O, CO are filled from the US Standard Atmosphere 1976 at each level.
* Profiles above the sonde top are filled with USSA76 values.
* Nighttime profiles (SZA ≥ 90°) automatically get SW = 0.
