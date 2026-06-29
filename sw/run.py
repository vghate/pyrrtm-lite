"""Top-level driver for RRTM_SW."""

from pathlib import Path

from . import load_coeffs, io_nc, setcoef, taumol, rtrdis
from . import cf_reader
from .sounding import SWSounding
from .cloud_optics import cloud_optical_depth

_DEFAULT_COEFFS = Path(__file__).parent.parent / 'data' / 'sw_coeffs.nc'


def _lev_to_lay(arr):
    """Average adjacent levels to produce layer-centred values."""
    return (arr[:-1] + arr[1:]) / 2.0


def run(input_path=None, output_path=None, sounding=None,
        aerosol=None, coeffs_path=None, iout=99, iceflag=2):
    """Run the RRTM_SW pipeline.

    Parameters
    ----------
    input_path : str or Path, optional
        Path to a Fortran INPUT_RRTM file or a CF-convention NetCDF sounding
        (detected by a ``.nc`` extension).
    output_path : str or Path, optional
        Destination NetCDF file for radiative-transfer results.
    sounding : SWSounding, optional
        Pre-built :class:`~rrtm_sw.sounding.SWSounding` instance.  When
        provided, *input_path* is ignored for sounding data.
    aerosol : object, optional
        Aerosol optical property object passed through to ``rtrdis``.
    coeffs_path : str or Path, optional
        Path to the band-coefficient NetCDF file.  Defaults to
        ``<package_root>/data/coeffs.nc``.
    iout : int, optional
        Output level selector (passed through for future use).
    iceflag : int, optional
        Ice-cloud parameterisation flag (2 or 3).  Used when reading a CF
        sounding from *input_path*; ignored if *sounding* already carries its
        own ``iceflag`` attribute.

    Returns
    -------
    result : dict
        Dictionary of computed flux arrays as returned by ``rtrdis``.
    """
    # 1. Load band coefficients (no chi_mls in SW)
    if coeffs_path is None:
        coeffs_path = _DEFAULT_COEFFS
    bands = load_coeffs.load(coeffs_path)

    # 2. Parse sounding
    if isinstance(sounding, SWSounding):
        # Caller supplied a ready-made SWSounding — use it directly.
        pass
    elif input_path is not None and str(input_path).endswith('.nc'):
        # CF-NetCDF sounding file
        sounding = cf_reader.read_cf_sounding(input_path, iceflag=iceflag)
    elif input_path is not None:
        # Fortran-format INPUT_RRTM_SW_* file
        inp = io_nc.parse_fortran_input(input_path)
        sounding = inp   # keep as plain dict; cloud fields not present
    else:
        raise ValueError(
            "Provide either 'sounding' (an SWSounding instance), "
            "'input_path' pointing to a CF-NetCDF (.nc) file, or "
            "'input_path' pointing to a Fortran INPUT_RRTM file."
        )

    # 3. Extract standard profile arrays
    if isinstance(sounding, SWSounding):
        pavel       = sounding.pavel
        tavel       = sounding.tavel
        tz          = sounding.tz
        pz          = sounding.pz
        wkl         = sounding.wkl
        coldry      = sounding.coldry
        wbroad      = sounding.wbroad
        zenith      = sounding.zenith
        albedo_band = sounding.albedo_band
        adjflux     = sounding.adjflux
        tbound      = sounding.tbound
    else:
        # Plain dict from parse_fortran_input
        pavel       = sounding['pavel']
        tavel       = sounding['tavel']
        tz          = sounding['tz']
        pz          = sounding['pz']
        wkl         = sounding['wkl']
        coldry      = sounding['coldry']
        wbroad      = sounding['wbroad']
        zenith      = sounding['zenith']
        albedo_band = sounding['albedo_band']
        adjflux     = sounding['adjflux']
        tbound      = sounding.get('tbound', tz[0])

    # 4. Compute optical-path coefficients
    sc = setcoef.setcoef(pavel, tavel, tz, wkl, coldry, wbroad)

    # 5. Compute optical depths, single-scatter albedos, and solar flux
    tau, ssa, sfluxzen = taumol.taumol_all(sc, bands, adjflux)

    # 6. Compute cloud optical depth (level arrays -> layer arrays)
    tau_cloud = ssa_cloud = g_cloud = frac_c = None
    if isinstance(sounding, SWSounding) and sounding.frac is not None:
        tau_cloud, ssa_cloud, g_cloud, frac_c = cloud_optical_depth(
            _lev_to_lay(sounding.frac),
            _lev_to_lay(sounding.lwp_gm2),
            _lev_to_lay(sounding.re_liq_um),
            _lev_to_lay(sounding.iwp_gm2),
            _lev_to_lay(sounding.re_ice_um),
            iceflag=sounding.iceflag,
        )
        if frac_c.max() == 0:
            tau_cloud = ssa_cloud = g_cloud = frac_c = None

    # 7. Radiative transfer (discrete ordinate)
    result = rtrdis.rtrdis(
        tau, ssa, sfluxzen, zenith, albedo_band, pz,
        aerosol=aerosol,
        tau_cloud=tau_cloud,
        ssa_cloud=ssa_cloud,
        g_cloud=g_cloud,
        frac_cloud=frac_c,
    )

    # 8. Write output if requested
    if output_path is not None:
        io_nc.write_output(output_path, result, pz)

    return result
