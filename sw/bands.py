"""
SW band definitions for RRTM_SW.

14 bands covering the shortwave spectrum (Fortran bands 16-29).
Each band uses 16 g-points (NG_PER_BAND).
"""
import numpy as np

NBANDS      = 14
NG_PER_BAND = 16

# Band wavenumber limits (cm⁻¹)
WAVENUM1 = np.array(
    [2600, 3250, 4000, 4650, 5150, 6150, 7700, 8050, 12850, 16000, 22650, 29000, 38000, 820],
    dtype=np.float64,
)
WAVENUM2 = np.array(
    [3250, 4000, 4650, 5150, 6150, 7700, 8050, 12850, 16000, 22650, 29000, 38000, 50000, 2600],
    dtype=np.float64,
)
DELWAVE = WAVENUM2 - WAVENUM1

# Number of reference species (lower / upper atmosphere) per band
NSPA = np.array([9, 9, 9, 9, 1, 9, 9, 1, 9, 1, 0, 1, 9, 1], dtype=np.int32)
NSPB = np.array([1, 5, 1, 1, 1, 5, 1, 0, 1, 0, 0, 1, 5, 1], dtype=np.int32)

# Mapping from Python band index (0-13) to Fortran band number (16-29)
FORTRAN_BAND = np.arange(16, 30, dtype=np.int32)

# Number of g-points per band (all 16 for SW)
NG = np.full(NBANDS, NG_PER_BAND, dtype=np.int32)

# Human-readable band names
BAND_NAMES = [
    f"Band {FORTRAN_BAND[i]} ({int(WAVENUM1[i])}–{int(WAVENUM2[i])} cm⁻¹)"
    for i in range(NBANDS)
]


def EARTH_SUN(juldat: int) -> float:
    """Earth-Sun distance correction factor (squared ratio of mean/actual distance).

    Matches the Fortran EARTH_SUN function in src/rrtm.f.

    Parameters
    ----------
    juldat : int
        Julian day, 1-365.

    Returns
    -------
    float
        Correction factor to be applied to the solar constant.
    """
    return 1.0 + 0.033412 * np.cos(2.0 * np.pi * (juldat - 3) / 365.0)
