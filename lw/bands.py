"""Band definitions for RRTM_LW — wavenumber ranges, g-point counts, weights."""
import numpy as np

NBANDS = 16
NG_PER_BAND = 16   # all bands use 16 g-points

# Band wavenumber limits (cm⁻¹), 0-indexed (band 0 = Fortran band 1)
WAVENUM1 = np.array([
     10.,  350.,  500.,  630.,  700.,  820.,  980., 1080.,
   1180., 1390., 1480., 1800., 2080., 2250., 2380., 2600.,
], dtype=np.float64)

WAVENUM2 = np.array([
    350.,  500.,  630.,  700.,  820.,  980., 1080., 1180.,
   1390., 1480., 1800., 2080., 2250., 2380., 2600., 3250.,
], dtype=np.float64)

DELWAVE = WAVENUM2 - WAVENUM1  # band widths (cm⁻¹)

# All bands have 16 g-points with equal weight 1/16
NG     = np.full(NBANDS, NG_PER_BAND, dtype=np.int32)
G_WEIGHTS = np.full(NG_PER_BAND, 1.0 / NG_PER_BAND, dtype=np.float64)

BAND_NAMES = [f"Band {i+1} ({int(WAVENUM1[i])}–{int(WAVENUM2[i])} cm⁻¹)"
              for i in range(NBANDS)]
