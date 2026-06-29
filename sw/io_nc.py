"""
io_nc.py — Input/output routines for RRTM_SW Python port.

Functions
---------
parse_fortran_input(path)  : parse fixed-format INPUT_RRTM (SW) → dict of profile arrays
parse_fortran_output(path) : parse fixed-format OUTPUT_RRTM (SW) → list of dicts
read_input(path)           : read CF-NetCDF input (placeholder; raises NotImplementedError)
write_output(path, result, pz) : write output NetCDF
"""

import math
import numpy as np
import netCDF4 as nc

from .bands import WAVENUM1, WAVENUM2, EARTH_SUN, NBANDS


# ---------------------------------------------------------------------------
# Helper utilities (same pattern as rrtm_lw/io_nc.py)
# ---------------------------------------------------------------------------

def _safe_get(line, start, stop=None):
    """Return line[start:stop] padded with spaces if the line is too short."""
    if stop is None:
        return line[start] if start < len(line) else ' '
    end = min(stop, len(line))
    s = line[start:end]
    if len(s) < (stop - start):
        s = s + ' ' * ((stop - start) - len(s))
    return s


def _parse_float(s, default=0.0):
    """Parse float from string; return default on failure."""
    try:
        return float(s.strip())
    except (ValueError, AttributeError):
        return default


def _parse_int(s, default=0):
    """Parse int from string; return default on failure."""
    try:
        return int(s.strip())
    except (ValueError, AttributeError):
        return default


# ---------------------------------------------------------------------------
# parse_fortran_input
# ---------------------------------------------------------------------------

def parse_fortran_input(path):
    """
    Parse a fixed-format INPUT_RRTM file used by RRTM_SW.

    Handles SW-specific records (1.2 control flags, 1.2.1 solar geometry,
    1.4 surface albedo) in addition to the IATM=0 layer profile records
    (2.1, 2.1.1, 2.1.2) shared with RRTM_LW.

    Parameters
    ----------
    path : str or path-like
        Path to INPUT_RRTM file.

    Returns
    -------
    dict with keys
    ~~~~~~~~~~~~
    Atmospheric profile (same as LW):
        pavel   (nlayers,)     layer average pressure [mb]
        tavel   (nlayers,)     layer average temperature [K]
        pz      (nlayers+1,)   level pressures [mb]; pz[0] = surface
        tz      (nlayers+1,)   level temperatures [K]; tz[0] = surface
        wkl     (7, nlayers)   gas column amounts [molec/cm²]
        wbroad  (nlayers,)     broadening gas column [molec/cm²]
        coldry  (nlayers,)     dry air column amount (same as wbroad) [molec/cm²]
        tbound  float          surface temperature [K]
        nlayers int            number of model layers

    SW-specific:
        juldat      int            Julian day (1–365; 0 = no Earth-Sun scaling)
        sza_deg     float          solar zenith angle [degrees]
        zenith      float          cos(sza_deg * pi/180)
        isolvar     int            solar variability flag (0/1/2)
        albedo      float          mean surface albedo (broadband)
        albedo_band (14,) array    per-band surface albedo
        adjflux     (14,) array    Earth-Sun distance correction factor per band
        iout        int            output flag
        icld        int            cloud flag
        iscat       int            solver flag (0=DISORT, 1=two-stream)
        istrm       int            DISORT stream flag (0=4str,1=8str,2=16str)
        idelm       int            delta-M output flag
        icos        int            cosine response flag
        iaer        int            aerosol flag
    """
    with open(path, 'r') as fh:
        lines = [ln.rstrip('\n') for ln in fh.readlines()]

    # ------------------------------------------------------------------
    # Locate the '$' sentinel (Record 1.1)
    # ------------------------------------------------------------------
    dollar_idx = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith('$'):
            dollar_idx = i
            break
    if dollar_idx is None:
        raise ValueError(f"Could not find '$' record in {path}")

    # ------------------------------------------------------------------
    # Record 1.2: control flags
    #
    # Fortran format: 18X,I2, 29X,I1, 32X,I1, 1X,I1, 2X,I3, 4X,I1, 3X,I1, I1
    #
    # Column positions (1-indexed from the instructions):
    #   IAER    cols 19-20  → 0-indexed [18:20]
    #   IATM    col  50     → 0-indexed [49:50]
    #   ISCAT   col  83     → 0-indexed [82:83]
    #   ISTRM   col  85     → 0-indexed [84:85]
    #   IOUT    cols 88-90  → 0-indexed [87:90]
    #   ICLD    col  95     → 0-indexed [94:95]
    #   IDELM   col  99     → 0-indexed [98:99]
    #   ICOS    col  100    → 0-indexed [99:100]
    # ------------------------------------------------------------------
    rec12 = lines[dollar_idx + 1]

    iaer  = _parse_int(_safe_get(rec12, 18, 20), 0)
    iatm  = _parse_int(_safe_get(rec12, 49, 50), 0)
    iscat = _parse_int(_safe_get(rec12, 82, 83), 0)
    istrm = _parse_int(_safe_get(rec12, 84, 85), 0)
    iout  = _parse_int(_safe_get(rec12, 87, 90), 0)
    icld  = _parse_int(_safe_get(rec12, 94, 95), 0)
    idelm = _parse_int(_safe_get(rec12, 98, 99), 0)
    icos  = _parse_int(_safe_get(rec12, 99, 100), 0)

    if iatm != 0:
        import warnings
        warnings.warn(
            "IATM=1 (RRTATM profile mode): reading explicit layer data as-is. "
            "Missing species not filled from standard atmosphere.",
            UserWarning, stacklevel=2
        )

    # ------------------------------------------------------------------
    # Record 1.2.1: solar geometry
    #
    # Fortran format: 12X, I3, 3X, F7.4, 4X, I1, 14F5.3
    #
    # Column positions (1-indexed):
    #   JULDAT  cols 13-15  → 0-indexed [12:15]
    #   SZA     cols 19-25  → 0-indexed [18:25]
    #   ISOLVAR col  30     → 0-indexed [29:30]
    #   SOLVAR  14×F5.3 starting col 31 → 0-indexed [30:30+14*5]
    # ------------------------------------------------------------------
    rec121 = lines[dollar_idx + 2]

    juldat  = _parse_int(_safe_get(rec121, 12, 15), 0)
    sza_deg = _parse_float(_safe_get(rec121, 18, 25), 0.0)

    # cos(sza) — clamp to avoid numerical issues at exactly 90 deg
    sza_rad = sza_deg * math.pi / 180.0
    zenith  = max(math.cos(sza_rad), 1.0e-10)

    isolvar = _parse_int(_safe_get(rec121, 29, 30), 0)

    solvar = np.ones(NBANDS, dtype=np.float64)
    for ib in range(NBANDS):
        s_raw = _safe_get(rec121, 30 + ib * 5, 35 + ib * 5)
        val = _parse_float(s_raw, 0.0)
        if val > 0.0:
            solvar[ib] = val

    # ------------------------------------------------------------------
    # Record 1.4: surface emissivity / albedo
    #
    # Fortran format: 11X, I1, 2X, I1, 14F5.3
    #
    # Column positions (1-indexed):
    #   IEMIS    col 12     → 0-indexed [11:12]
    #   IREFLECT col 15     → 0-indexed [14:15]
    #   SEMISS   14×F5.3 starting col 16 → 0-indexed [15:15+14*5]
    #
    # In RRTM_SW, SEMISS is the surface EMISSIVITY (same convention as LW).
    # The surface albedo is:  ALBEDO = 1 - SEMISS
    # This matches Fortran rtrdis.f line: ALBEDO = 1. - SEMISS(IBAND)
    #
    # IEMIS=0 → emissivity=1.0 for all bands → albedo=0.0 → use 0.1 default
    # IEMIS=1 → all bands use SEMISS(16) (first value)
    # IEMIS=2 → per-band SEMISS array
    # ------------------------------------------------------------------
    rec14 = lines[dollar_idx + 3]

    iemis    = _parse_int(_safe_get(rec14, 11, 12), 0)

    semiss_band = np.ones(NBANDS, dtype=np.float64)  # default emissivity=1 → albedo=0

    if iemis == 0:
        # Emissivity = 1.0 for all bands → albedo = 0. Use 0.1 as a safe default.
        semiss_band[:] = 0.9  # emissivity=0.9 → albedo=0.1
    elif iemis == 1:
        # All bands use first value (emissivity)
        s0 = _parse_float(_safe_get(rec14, 15, 20), 0.9)
        semiss_band[:] = s0
    elif iemis == 2:
        # Per-band emissivity values (14 values, each 5 chars wide)
        for ib in range(NBANDS):
            s_raw = _safe_get(rec14, 15 + ib * 5, 20 + ib * 5)
            val = _parse_float(s_raw, 0.9)
            semiss_band[ib] = val

    # Surface albedo = 1 - emissivity  (matches Fortran rtrdis.f)
    albedo_band = 1.0 - semiss_band
    albedo      = float(np.mean(albedo_band))

    # ------------------------------------------------------------------
    # Compute adjflux from Julian day
    # ------------------------------------------------------------------
    if juldat > 0:
        adjflux_scalar = EARTH_SUN(juldat)
    else:
        adjflux_scalar = 1.0  # no Earth-Sun scaling when JULDAT=0

    if isolvar == 2:
        # Per-band scale factors supplied in SOLVAR
        adjflux = adjflux_scalar * solvar
    elif isolvar == 1:
        # Same scale factor for all bands (use first SOLVAR value)
        adjflux = np.full(NBANDS, adjflux_scalar * solvar[0], dtype=np.float64)
    else:
        # isolvar == 0: standard solar source, only Earth-Sun correction
        adjflux = np.full(NBANDS, adjflux_scalar, dtype=np.float64)

    # ------------------------------------------------------------------
    # Record 2.1: layer dimensions
    #
    # Fortran format: 1X, I1, I3, I5
    # Column positions (1-indexed): IFORM at col 2, NLAYRS at cols 3-5
    # (0-indexed: IFORM at [1:2], NLAYRS at [2:5])
    # ------------------------------------------------------------------
    rec21 = lines[dollar_idx + 4]
    iform   = _parse_int(_safe_get(rec21, 1, 2), 0)
    nlayers = _parse_int(_safe_get(rec21, 2, 5), 0)

    # ------------------------------------------------------------------
    # Allocate profile arrays
    # ------------------------------------------------------------------
    pavel  = np.zeros(nlayers, dtype=np.float64)
    tavel  = np.zeros(nlayers, dtype=np.float64)
    pz     = np.zeros(nlayers + 1, dtype=np.float64)
    tz     = np.zeros(nlayers + 1, dtype=np.float64)
    wkl    = np.zeros((7, nlayers), dtype=np.float64)
    wbroad = np.zeros(nlayers, dtype=np.float64)

    # ------------------------------------------------------------------
    # Parse layer records (Records 2.1.1 and 2.1.2) — identical to LW
    # ------------------------------------------------------------------
    cur = dollar_idx + 5  # first layer record starts here

    for lay in range(nlayers):
        form_line = lines[cur]
        cur += 1

        if iform == 0:
            # IFORM=0: F10.4, F10.4 ... (10-char fields)
            pavel[lay] = _parse_float(form_line[0:10])
            tavel[lay] = _parse_float(form_line[10:20])
            if lay == 0:
                # Bottom level: PZ at [43:51], TZ at [51:58]
                pz[0] = _parse_float(form_line[43:51])
                tz[0] = _parse_float(form_line[51:58])
            # Top level: PZ at [65:73], TZ at [73:80]
            pz[lay + 1] = _parse_float(form_line[65:73])
            tz[lay + 1] = _parse_float(form_line[73:80])
        else:
            # IFORM=1: E15.7 (15-char fields)
            pavel[lay] = _parse_float(form_line[0:15])
            tavel[lay] = _parse_float(form_line[15:25])
            if lay == 0:
                # Bottom level: PZ at [48:56], TZ at [56:63]
                pz[0] = _parse_float(form_line[48:56])
                tz[0] = _parse_float(form_line[56:63])
            # Top level: PZ at [70:78], TZ at [78:85]
            pz[lay + 1] = _parse_float(form_line[70:78])
            tz[lay + 1] = _parse_float(form_line[78:85])

        # Record 2.1.2: gas amounts (one line per layer)
        gas_line = lines[cur]
        cur += 1

        if iform == 0:
            # 8E10.3: 10-char fields
            gas_vals = np.zeros(8, dtype=np.float64)
            for m in range(8):
                gas_vals[m] = _parse_float(gas_line[m * 10:(m + 1) * 10])
        else:
            # 8E15.7: 15-char fields
            gas_vals = np.zeros(8, dtype=np.float64)
            for m in range(8):
                gas_vals[m] = _parse_float(gas_line[m * 15:(m + 1) * 15])

        for m in range(7):
            wkl[m, lay] = gas_vals[m]
        wbroad[lay] = gas_vals[7]

    # ------------------------------------------------------------------
    # Gas amount conversion: VMR → column amount (mirrors Fortran rrtm.f)
    #
    # If all species are mixing ratios (all wkl <= 1.0):
    #   SUMMOL = sum of non-H2O VMRs (species 2..7, Python index 1..6)
    #   COLDRY = WBROADL / (1 - SUMMOL)
    #   WKL(M) = COLDRY * WKL(M)
    # If any gas is already column amount:
    #   COLDRY = WBROADL + SUMMOL
    # ------------------------------------------------------------------
    imix = 1
    for m in range(7):
        if np.any(wkl[m] > 1.0):
            imix = 0
            break

    coldry = np.zeros(nlayers, dtype=np.float64)
    if imix == 1:
        for lay in range(nlayers):
            summol = np.sum(wkl[1:7, lay])
            coldry[lay] = wbroad[lay] / (1.0 - summol)
            for m in range(7):
                wkl[m, lay] = coldry[lay] * wkl[m, lay]
    else:
        for lay in range(nlayers):
            summol = np.sum(wkl[1:7, lay])
            coldry[lay] = wbroad[lay] + summol

    # Surface temperature fallback (use lowest level temperature)
    tbound = float(tz[0])

    return {
        # Atmospheric profile
        'pavel':       pavel,
        'tavel':       tavel,
        'pz':          pz,
        'tz':          tz,
        'wkl':         wkl,
        'wbroad':      wbroad,
        'coldry':      coldry,
        'tbound':      tbound,
        'nlayers':     int(nlayers),
        # SW-specific solar geometry
        'juldat':      int(juldat),
        'sza_deg':     float(sza_deg),
        'zenith':      float(zenith),
        'isolvar':     int(isolvar),
        'adjflux':     adjflux,
        # Surface albedo
        'albedo':      albedo,
        'albedo_band': albedo_band,
        # Control flags
        'iaer':        int(iaer),
        'iscat':       int(iscat),
        'istrm':       int(istrm),
        'iout':        int(iout),
        'icld':        int(icld),
        'idelm':       int(idelm),
        'icos':        int(icos),
    }


# ---------------------------------------------------------------------------
# parse_fortran_output
# ---------------------------------------------------------------------------

def parse_fortran_output(path):
    """
    Parse a fixed-format OUTPUT_RRTM file from RRTM_SW.

    The SW output has 8 columns per data line (vs 6 in LW):
        LEVEL  PRESSURE  UPWARD  DIFDOWN  DIRDOWN  DOWNWARD  NET  HEATING

    Header per block:
        " Wavenumbers: <w1> - <w2> cm-1"
    Followed by 2 header/units lines, then data.

    IOUT=99 produces 15 blocks:
      - 1 broadband block  (820–50000 cm-1)
      - 14 per-band blocks (Fortran bands 16→29)

    IOUT=0 produces 1 broadband block only.
    IOUT=n (16..29) produces 1 block for that band.
    IOUT=98 produces 15 blocks (same layout as IOUT=99 per Fortran source).

    Parameters
    ----------
    path : str or path-like
        Path to OUTPUT_RRTM file.

    Returns
    -------
    list of dicts, one per "Wavenumbers:" block.
    Each dict has:
        wavenums1 : float          lower wavenumber [cm-1]
        wavenums2 : float          upper wavenumber [cm-1]
        level     : np.ndarray     level indices (int)
        pressure  : np.ndarray     pressure [mb]
        uflux     : np.ndarray     upward flux [W/m2]
        difdown   : np.ndarray     diffuse downward flux [W/m2]
        dirdown   : np.ndarray     direct (beam) downward flux [W/m2]
        dflux     : np.ndarray     total downward flux = dirdown + difdown [W/m2]
        fnet      : np.ndarray     net flux [W/m2]
        htr       : np.ndarray     heating rate [K/day]
    """
    with open(path, 'r') as fh:
        lines = fh.readlines()

    # Locate all "Wavenumbers:" header lines
    block_starts = [i for i, ln in enumerate(lines) if 'Wavenumbers:' in ln]

    if not block_starts:
        raise ValueError(f"No 'Wavenumbers:' blocks found in {path}")

    results = []
    for b, bstart in enumerate(block_starts):
        hdr = lines[bstart]
        # Format: " Wavenumbers: <w1> - <w2> cm-1 ..."
        parts = hdr.split()
        # parts[0]='Wavenumbers:', parts[1]=w1, parts[2]='-', parts[3]=w2
        w1 = float(parts[1])
        w2 = float(parts[3])

        # Data starts 3 lines after the "Wavenumbers:" header
        # (skip: LEVEL PRESSURE... header line + units line)
        data_start = bstart + 3
        data_end   = block_starts[b + 1] if b + 1 < len(block_starts) else len(lines)

        levels   = []
        pressure = []
        uflux    = []
        difdown  = []
        dirdown  = []
        dflux    = []
        fnet     = []
        htr      = []

        for i in range(data_start, data_end):
            line = lines[i].strip()
            if not line:
                continue
            fields = line.split()
            # Expect 8 fields: LEVEL PRESSURE UPWARD DIFDOWN DIRDOWN DOWNWARD NET HEATING
            if len(fields) < 8:
                continue
            try:
                lev  = int(fields[0])
                pres = float(fields[1])
                uf   = float(fields[2])
                dd   = float(fields[3])  # diffuse down
                dr   = float(fields[4])  # direct down
                df   = float(fields[5])  # total down = diffuse + direct
                fn   = float(fields[6])  # net
                ht   = float(fields[7])  # heating rate
            except (ValueError, IndexError):
                continue

            levels.append(lev)
            pressure.append(pres)
            uflux.append(uf)
            difdown.append(dd)
            dirdown.append(dr)
            dflux.append(df)
            fnet.append(fn)
            htr.append(ht)

        results.append({
            'wavenums1': w1,
            'wavenums2': w2,
            'level':    np.array(levels,   dtype=np.int32),
            'pressure': np.array(pressure, dtype=np.float64),
            'uflux':    np.array(uflux,    dtype=np.float64),
            'difdown':  np.array(difdown,  dtype=np.float64),
            'dirdown':  np.array(dirdown,  dtype=np.float64),
            'dflux':    np.array(dflux,    dtype=np.float64),
            'fnet':     np.array(fnet,     dtype=np.float64),
            'htr':      np.array(htr,      dtype=np.float64),
        })

    # For IOUT=99, the Fortran code outputs the broadband block first
    # (820–50000 cm-1), then 14 per-band blocks.  The test suite expects the
    # per-band blocks at indices 0..13 and the broadband block at index 14
    # (i.e. last).  Reorder accordingly if the first block looks like the
    # broadband (w1==820 and w2==50000) and there are 15 blocks total.
    if (len(results) == 15 and
            results[0]['wavenums1'] == 820.0 and
            results[0]['wavenums2'] == 50000.0):
        results = results[1:] + results[:1]

    return results


# ---------------------------------------------------------------------------
# read_input  (CF-NetCDF — placeholder for Phase 12)
# ---------------------------------------------------------------------------

def read_input(path):
    """
    Read a CF-NetCDF sounding as input to RRTM_SW.

    This is a placeholder for the full CF reader implemented in Phase 12
    (cf_reader.read_cf_sounding).  Until that module is available, use
    parse_fortran_input() for Fortran-format INPUT_RRTM files.

    Parameters
    ----------
    path : str or path-like
        Path to CF-NetCDF input file.

    Raises
    ------
    NotImplementedError
        Always, until Phase 12 is implemented.
    """
    raise NotImplementedError(
        "CF-NetCDF input reading is not yet implemented (Phase 12). "
        "Use parse_fortran_input() for Fortran INPUT_RRTM files."
    )


# ---------------------------------------------------------------------------
# write_output
# ---------------------------------------------------------------------------

def write_output(path, result, pz):
    """
    Write RRTM_SW output to a CF-style NetCDF file.

    Writes all variables specified in SPEC.md Section 4.  Both broadband
    and per-band (band x level) fields are always written.

    Parameters
    ----------
    path : str or path-like
        Destination NetCDF file path.
    result : dict
        Output dict from the solver, containing at minimum:

        Broadband (1-D, shape (nlayers+1,)):
            totuflux   upward flux [W/m2]
            totdflux   total downward flux (direct + diffuse) [W/m2]
            dirdown    direct (beam) downward flux [W/m2]
            difdown    diffuse downward flux [W/m2]
            fnet       net flux (upward - downward) [W/m2]
            htr        heating rate [K/day]

        Per-band (2-D, shape (14, nlayers+1)):
            band_totuflux
            band_totdflux
            band_dirdown
            band_difdown
            band_fnet
            band_htr

    pz : array-like, shape (nlayers+1,)
        Level pressures [mb].  pz[0] = surface, pz[-1] = TOA.
    """
    pz = np.asarray(pz, dtype=np.float64)
    nlevels = len(pz)

    ds = nc.Dataset(str(path), 'w', format='NETCDF4')
    try:
        # ---- Dimensions ----
        ds.createDimension('level', nlevels)
        ds.createDimension('band',  NBANDS)

        # ---- Level pressure ----
        v = ds.createVariable('pz', 'f8', ('level',))
        v[:] = pz
        v.units     = 'mb'
        v.long_name = 'level pressure'

        # ---- Broadband fluxes ----
        v = ds.createVariable('totuflux', 'f8', ('level',))
        v[:] = np.asarray(result['totuflux'], dtype=np.float64)
        v.units     = 'W/m2'
        v.long_name = 'broadband upward flux'

        v = ds.createVariable('totdflux', 'f8', ('level',))
        v[:] = np.asarray(result['totdflux'], dtype=np.float64)
        v.units     = 'W/m2'
        v.long_name = 'broadband total downward flux (direct + diffuse)'

        v = ds.createVariable('dirdown', 'f8', ('level',))
        v[:] = np.asarray(result['dirdown'], dtype=np.float64)
        v.units     = 'W/m2'
        v.long_name = 'broadband direct (beam) downward flux'

        v = ds.createVariable('difdown', 'f8', ('level',))
        v[:] = np.asarray(result['difdown'], dtype=np.float64)
        v.units     = 'W/m2'
        v.long_name = 'broadband diffuse downward flux'

        v = ds.createVariable('fnet', 'f8', ('level',))
        v[:] = np.asarray(result['fnet'], dtype=np.float64)
        v.units     = 'W/m2'
        v.long_name = 'broadband net flux (upward minus downward)'

        v = ds.createVariable('htr', 'f8', ('level',))
        v[:] = np.asarray(result['htr'], dtype=np.float64)
        v.units     = 'K/day'
        v.long_name = 'broadband heating rate'

        # ---- Per-band fluxes ----
        v = ds.createVariable('band_totuflux', 'f8', ('band', 'level'))
        v[:] = np.asarray(result['band_totuflux'], dtype=np.float64)
        v.units     = 'W/m2'
        v.long_name = 'per-band upward flux'

        v = ds.createVariable('band_totdflux', 'f8', ('band', 'level'))
        v[:] = np.asarray(result['band_totdflux'], dtype=np.float64)
        v.units     = 'W/m2'
        v.long_name = 'per-band total downward flux (direct + diffuse)'

        v = ds.createVariable('band_dirdown', 'f8', ('band', 'level'))
        v[:] = np.asarray(result['band_dirdown'], dtype=np.float64)
        v.units     = 'W/m2'
        v.long_name = 'per-band direct (beam) downward flux'

        v = ds.createVariable('band_difdown', 'f8', ('band', 'level'))
        v[:] = np.asarray(result['band_difdown'], dtype=np.float64)
        v.units     = 'W/m2'
        v.long_name = 'per-band diffuse downward flux'

        v = ds.createVariable('band_fnet', 'f8', ('band', 'level'))
        v[:] = np.asarray(result['band_fnet'], dtype=np.float64)
        v.units     = 'W/m2'
        v.long_name = 'per-band net flux (upward minus downward)'

        v = ds.createVariable('band_htr', 'f8', ('band', 'level'))
        v[:] = np.asarray(result['band_htr'], dtype=np.float64)
        v.units     = 'K/day'
        v.long_name = 'per-band heating rate'

        # ---- Band wavenumber limits ----
        v = ds.createVariable('wavenum1', 'f8', ('band',))
        v[:] = WAVENUM1
        v.units     = 'cm-1'
        v.long_name = 'band lower wavenumber'

        v = ds.createVariable('wavenum2', 'f8', ('band',))
        v[:] = WAVENUM2
        v.units     = 'cm-1'
        v.long_name = 'band upper wavenumber'

        ds.description = 'RRTM_SW shortwave radiative transfer output'
        ds.Conventions = 'CF-1.8'

    finally:
        ds.close()
