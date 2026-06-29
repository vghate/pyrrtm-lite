"""
run.py -- Top-level driver for the RRTM_LW Python port.

Public API
----------
run(input_path=None, output_path=None, sounding=None,
    aerosol=None, coeffs_path=None,
    iscat=1, numangs=2, iout=99, iceflag=2) -> dict
"""

from pathlib import Path

import numpy as np

from . import load_coeffs, io_nc, setcoef, taumol, rtreg
from . import cf_reader
from .sounding import Sounding
from .cloud_optics import cloud_optical_depth

_DEFAULT_COEFFS = Path(__file__).parent.parent / 'data' / 'lw_coeffs.nc'


def run(input_path=None, output_path=None, sounding=None,
        aerosol=None, coeffs_path=None,
        iscat=1, numangs=2, iout=99, iceflag=2):
    """Execute the full RRTM_LW pipeline.

    Parameters
    ----------
    input_path : str or Path or None
        Path to the input file.  Files ending in ``.nc`` are read with
        :func:`cf_reader.read_cf_sounding` and returned as a
        :class:`~rrtm_lw.sounding.Sounding`; all other paths are assumed
        to be Fortran-formatted and parsed with
        :func:`io_nc.parse_fortran_input`.  Ignored when *sounding* is
        provided.
    output_path : str or Path or None
        Destination NetCDF file for the radiative-transfer results.  When
        *None* the output step is skipped.
    sounding : Sounding or None
        A pre-built :class:`~rrtm_lw.sounding.Sounding` instance.  When
        supplied, *input_path* is ignored.
    aerosol : array-like or None
        Aerosol optical depth array forwarded to :func:`rtreg.rtreg`.
    coeffs_path : str or Path or None
        Path to the coefficients NetCDF file.  Defaults to
        ``<repo_root>/data/coeffs.nc`` when *None*.
    iscat : int
        Scattering flag (currently unused internally; reserved for future
        use).  Default ``1``.
    numangs : int
        Number of quadrature angles passed to :func:`rtreg.rtreg`.
        Default ``2``.
    iout : int
        Output level flag forwarded to :func:`rtreg.rtreg`.  Default
        ``99`` (all levels).
    iceflag : int
        Ice-optics parameterisation flag (``2`` or ``3``).  Used when
        reading a CF-NetCDF sounding.  Default ``2``.

    Returns
    -------
    dict
        Result dictionary as produced by :func:`rtreg.rtreg`.
    """
    # ------------------------------------------------------------------
    # 1. Load spectral coefficients
    # ------------------------------------------------------------------
    if coeffs_path is None:
        coeffs_path = _DEFAULT_COEFFS

    bands, totplnk, totplk16, chi_mls, planck = load_coeffs.load(coeffs_path)

    # ------------------------------------------------------------------
    # 2. Obtain a sounding
    # ------------------------------------------------------------------
    _fortran_path = False  # tracks whether we used the legacy dict path

    if sounding is not None:
        # Caller supplied a Sounding object directly.
        pass

    elif input_path is not None:
        input_path = str(input_path)
        if input_path.endswith(".nc"):
            sounding = cf_reader.read_cf_sounding(input_path, iceflag=iceflag)
        else:
            # Fortran-format: returns a plain dict; no cloud fields.
            inp = io_nc.parse_fortran_input(input_path)
            _fortran_path = True
    else:
        raise ValueError("Either 'input_path' or 'sounding' must be provided.")

    # ------------------------------------------------------------------
    # 3. Build inp dict and compute cloud optical depth
    # ------------------------------------------------------------------
    if _fortran_path:
        # inp already set above; no cloud information available.
        tau_cloud = None
        frac_clean = None
    else:
        # sounding is a Sounding instance.
        inp = sounding.to_dict()
        # has: pz, tz, pavel, tavel, wkl, wbroad, coldry, tbound, semiss

        def _lev_to_lay(arr):
            """Average adjacent level values to produce layer values."""
            return (arr[:-1] + arr[1:]) / 2

        tau_cloud, frac_clean = cloud_optical_depth(
            _lev_to_lay(sounding.frac),
            _lev_to_lay(sounding.lwp_gm2),
            _lev_to_lay(sounding.re_liq_um),
            _lev_to_lay(sounding.iwp_gm2),
            _lev_to_lay(sounding.re_ice_um),
            iceflag=sounding.iceflag,
        )

        # If no clouds present, pass None to rtreg for efficiency.
        if frac_clean.max() == 0:
            tau_cloud = None
            frac_clean = None

    # ------------------------------------------------------------------
    # 4. Compute optical-path coefficients
    # ------------------------------------------------------------------
    sc = setcoef.setcoef(
        inp["pavel"],
        inp["tavel"],
        inp["tz"],
        inp["tbound"],
        inp["wkl"],
        inp["coldry"],
        inp["wbroad"],
        inp["semiss"],
        totplnk,
        totplk16,
        chi_mls,
    )

    # ------------------------------------------------------------------
    # 5. Compute optical depths and Planck fractions
    #    tau shape: (16, nlayers, 16)
    # ------------------------------------------------------------------
    tau, fracs = taumol.taumol_all(sc, bands, planck, chi_mls)

    # ------------------------------------------------------------------
    # 6. Radiative transfer
    # ------------------------------------------------------------------
    result = rtreg.rtreg(
        tau, fracs, sc, inp["pz"], inp["semiss"],
        numangs=numangs,
        aerosol=aerosol,
        tau_cloud=tau_cloud,
        frac_cloud=frac_clean,
    )

    # ------------------------------------------------------------------
    # 7. Write output (optional)
    # ------------------------------------------------------------------
    if output_path is not None:
        io_nc.write_output(output_path, result, inp["pz"])

    return result
