"""
Load k-coefficients and Planck tables from coeffs.nc into Python dicts.
"""
from pathlib import Path
import numpy as np
import netCDF4 as nc

_DEFAULT_PATH = Path(__file__).parent.parent / 'data' / 'lw_coeffs.nc'


def load(path=None):
    """
    Load coeffs.nc and return:
        bands   : list of 16 dicts (index 0 = Fortran band 1), each with keys:
                    'ka', 'kb' (if present), 'selfref', 'forref',
                    and any minor-gas arrays ('ka_mn2', 'kb_mco2', etc.)
        totplnk : np.ndarray shape (181, 16), integrated Planck function per band
        totplk16: np.ndarray shape (181,),    integrated Planck for band 16 (full range)
        chi_mls : np.ndarray shape (9, 59),   MLS reference gas VMRs

    All arrays are float64.
    """
    if path is None:
        path = _DEFAULT_PATH
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"coeffs.nc not found at {path}. "
            "Run scripts/extract_coeffs.py first."
        )

    with nc.Dataset(path, 'r') as ds:
        # Root-level Planck + chi tables
        totplnk  = ds.variables['totplnk'][:].astype(np.float64)
        totplk16 = ds.variables['totplk16'][:].astype(np.float64)
        chi_mls  = ds.variables['chi_mls'][:].astype(np.float64)

        # Per-band k-coefficient dicts
        bands = []
        for b in range(1, 17):
            grp = ds.groups[f"band_{b:02d}"]
            d = {}
            for name in grp.variables:
                d[name] = grp.variables[name][:].astype(np.float64)
            bands.append(d)

        # Planck fraction tables (fracrefa shape: (nsp_a, 16); fracrefb or None)
        planck = []
        for b in range(1, 17):
            grp = ds.groups[f"planck_{b:02d}"]
            fa = grp.variables['fracrefa'][:].astype(np.float64)  # (nsp_a, 16)
            # Squeeze 1D case: shape (1,16) → (16,)
            if fa.shape[0] == 1:
                fa = fa[0]
            fb = None
            if 'fracrefb' in grp.variables:
                fb = grp.variables['fracrefb'][:].astype(np.float64)
                if fb.shape[0] == 1:
                    fb = fb[0]
            planck.append({'fracrefa': fa, 'fracrefb': fb})

    return bands, totplnk, totplk16, chi_mls, planck
