"""
io_nc.py — Input/output routines for RRTM_LW Python port.

Functions
---------
parse_fortran_input(path)  : parse fixed-format INPUT_RRTM → dict of profile arrays
parse_fortran_output(path) : parse fixed-format OUTPUT_RRTM → list of dicts
read_input(path)           : read NetCDF input → dict of profile arrays
write_output(path, result, pz) : write output NetCDF
"""

import numpy as np
import netCDF4 as nc


# ---------------------------------------------------------------------------
# Band wavenumber limits (1-based band index → 0-based array)
# ---------------------------------------------------------------------------
_WAVENUM1 = np.array([
    10., 350., 500., 630., 700., 820., 980., 1080.,
    1180., 1390., 1480., 1800., 2080., 2250., 2380., 2600.
], dtype=np.float64)

_WAVENUM2 = np.array([
    350., 500., 630., 700., 820., 980., 1080., 1180.,
    1390., 1480., 1800., 2080., 2250., 2380., 2600., 3250.
], dtype=np.float64)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _safe_get(line, start, stop=None):
    """Return line[start:stop] (or line[start]) padded with spaces if too short."""
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
    Parse a fixed-format INPUT_RRTM file used by RRTM_LW.

    Parameters
    ----------
    path : str or path-like
        Path to INPUT_RRTM file.

    Returns
    -------
    dict with keys:
        pavel   (nlayers,)     layer average pressure [mb]
        tavel   (nlayers,)     layer average temperature [K]
        pz      (nlayers+1,)   level pressures [mb]; pz[0]=surface
        tz      (nlayers+1,)   level temperatures [K]; tz[0]=surface
        wkl     (7, nlayers)   gas column amounts [molec/cm²]
        wbroad  (nlayers,)     broadening gas (dry air) column [molec/cm²]
        coldry  (nlayers,)     dry air column (same as wbroad) [molec/cm²]
        tbound  float          surface temperature [K]
        semiss  (16,)          surface emissivity per band
        iout    int            output flag
        iscat   int            solver flag
        numangs int            quadrature angles
        icld    int            cloud flag
        nlayers int            number of layers
    """
    with open(path, 'r') as fh:
        lines = [l.rstrip('\n') for l in fh.readlines()]

    # ------------------------------------------------------------------
    # Find the first "$" line (Record 1.1)
    # ------------------------------------------------------------------
    dollar_idx = None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith('$'):
            dollar_idx = i
            break
    if dollar_idx is None:
        raise ValueError(f"Could not find '$' record in {path}")

    # Record 1.2: control flags (line immediately after "$")
    rec12 = lines[dollar_idx + 1]
    iatm    = _parse_int(_safe_get(rec12, 49), 0)
    # ixsect = _parse_int(_safe_get(rec12, 69), 0)  # not used
    iscat   = _parse_int(_safe_get(rec12, 82), 0)
    numangs = _parse_int(_safe_get(rec12, 83, 85), 0)
    iout    = _parse_int(_safe_get(rec12, 87, 90), 0)
    icld    = _parse_int(_safe_get(rec12, 94), 0)

    if iatm != 0:
        raise NotImplementedError(
            "IATM=1 (RRTATM profile mode) is not implemented. "
            "Only IATM=0 (explicit profile) is supported."
        )

    # Record 1.4: surface temperature and emissivity (next line)
    rec14 = lines[dollar_idx + 2]
    tbound_raw = _safe_get(rec14, 0, 10)
    tbound = _parse_float(tbound_raw, 0.0)

    iemiss_s = _safe_get(rec14, 11, 12).strip()
    iemiss = _parse_int(iemiss_s, 0)

    semiss = np.ones(16, dtype=np.float64)
    if iemiss == 1:
        # All bands use first value
        s0_raw = _safe_get(rec14, 15, 20)
        s0 = _parse_float(s0_raw, 1.0)
        semiss[:] = s0
    elif iemiss == 2:
        # Per-band emissivity (16 values, each 5 chars)
        for i in range(16):
            s_raw = _safe_get(rec14, 15 + i * 5, 20 + i * 5)
            semiss[i] = _parse_float(s_raw, 1.0)
    # else iemiss == 0: all semiss = 1.0 (already set)

    # Record 2.1: profile dimensions (next line after rec14)
    rec21 = lines[dollar_idx + 3]
    iform   = _parse_int(_safe_get(rec21, 1, 2), 0)
    nlayers = _parse_int(_safe_get(rec21, 2, 5), 0)
    # nmol  = _parse_int(_safe_get(rec21, 5, 10), 7)  # not used beyond 7

    # ------------------------------------------------------------------
    # Allocate output arrays
    # ------------------------------------------------------------------
    pavel  = np.zeros(nlayers, dtype=np.float64)
    tavel  = np.zeros(nlayers, dtype=np.float64)
    pz     = np.zeros(nlayers + 1, dtype=np.float64)
    tz     = np.zeros(nlayers + 1, dtype=np.float64)
    wkl    = np.zeros((7, nlayers), dtype=np.float64)
    wbroad = np.zeros(nlayers, dtype=np.float64)

    # ------------------------------------------------------------------
    # Parse layer data starting at line dollar_idx + 4
    # ------------------------------------------------------------------
    cur = dollar_idx + 4

    for lay in range(nlayers):
        form_line = lines[cur]
        cur += 1

        if iform == 0:
            # IFORM=0: 3F10.4 ...
            pavel[lay] = _parse_float(form_line[0:10])
            tavel[lay] = _parse_float(form_line[10:20])
            # chars 20:35 ignored (altitude, molecule count)
            if lay == 0:
                # Bottom level
                pz[0] = _parse_float(form_line[43:51])
                tz[0] = _parse_float(form_line[51:58])
            # Top level
            pz[lay + 1] = _parse_float(form_line[65:73])
            tz[lay + 1] = _parse_float(form_line[73:80])
        else:
            # IFORM=1: G15.7, G10.4 ...
            pavel[lay] = _parse_float(form_line[0:15])
            tavel[lay] = _parse_float(form_line[15:25])
            # chars 25:41 ignored
            if lay == 0:
                # Bottom level (1X at 41, then ALTBOT=41:48, PZ=48:56, TZ=56:63)
                pz[0] = _parse_float(form_line[48:56])
                tz[0] = _parse_float(form_line[56:63])
            # Top level: ALTTOP=63:70, PZ=70:78, TZ=78:85
            pz[lay + 1] = _parse_float(form_line[70:78])
            tz[lay + 1] = _parse_float(form_line[78:85])

        # FORM3: gas amounts (one line)
        gas_line = lines[cur]
        cur += 1

        if iform == 0:
            # 8E10.3: each field is 10 chars
            gas_vals = np.zeros(8, dtype=np.float64)
            for m in range(8):
                gas_vals[m] = _parse_float(gas_line[m * 10:(m + 1) * 10])
        else:
            # 8G15.7: each field is 15 chars
            gas_vals = np.zeros(8, dtype=np.float64)
            for m in range(8):
                gas_vals[m] = _parse_float(gas_line[m * 15:(m + 1) * 15])

        for m in range(7):
            wkl[m, lay] = gas_vals[m]
        wbroad[lay] = gas_vals[7]

    # ------------------------------------------------------------------
    # Gas amount conversion: VMR → column amount (Fortran rrtm.f logic)
    #
    # Fortran checks WKL(M,1) > 1.0 for any species to detect mixing ratios.
    # If all species are mixing ratios (IMIX=1):
    #   SUMMOL = sum of non-H2O VMRs (species 2..NMOL = Python index 1..6)
    #   COLDRY = WBRODL / (1 - SUMMOL)
    #   WKL(M) = COLDRY * WKL(M)  for each layer
    # If any gas is already column amount (IMIX=0):
    #   COLDRY = WBRODL + SUMMOL
    # ------------------------------------------------------------------
    # Determine IMIX: check if all gas amounts are mixing ratios (<=1.0)
    imix = 1
    for m in range(7):
        if np.any(wkl[m] > 1.0):
            imix = 0
            break

    coldry = np.zeros(nlayers, dtype=np.float64)
    if imix == 1:
        # All gases are mixing ratios → convert using Fortran COLDRY formula
        for lay in range(nlayers):
            # SUMMOL = sum of species 2..7 (Python indices 1..6)
            summol = np.sum(wkl[1:7, lay])
            coldry[lay] = wbroad[lay] / (1.0 - summol)
            for m in range(7):
                wkl[m, lay] = coldry[lay] * wkl[m, lay]
    else:
        # Column amounts already given
        for lay in range(nlayers):
            summol = np.sum(wkl[1:7, lay])
            coldry[lay] = wbroad[lay] + summol

    # ------------------------------------------------------------------
    # Surface temperature fallback
    # ------------------------------------------------------------------
    if tbound < 0.0:
        tbound = tz[0]

    return {
        'pavel':   pavel,
        'tavel':   tavel,
        'pz':      pz,
        'tz':      tz,
        'wkl':     wkl,
        'wbroad':  wbroad,
        'coldry':  coldry,
        'tbound':  float(tbound),
        'semiss':  semiss,
        'iout':    int(iout),
        'iscat':   int(iscat),
        'numangs': int(numangs),
        'icld':    int(icld),
        'nlayers': int(nlayers),
    }


# ---------------------------------------------------------------------------
# parse_fortran_output
# ---------------------------------------------------------------------------
def parse_fortran_output(path):
    """
    Parse a fixed-format OUTPUT_RRTM file.

    Parameters
    ----------
    path : str or path-like
        Path to OUTPUT_RRTM file.

    Returns
    -------
    list of dicts, one per "Wavenumbers:" block.
    Each dict has:
        wavenums1 : float          lower wavenumber [cm⁻¹]
        wavenums2 : float          upper wavenumber [cm⁻¹]
        level     : np.ndarray     level indices (int)
        pressure  : np.ndarray     pressure [mb]
        uflux     : np.ndarray     upward flux [W/m²]
        dflux     : np.ndarray     downward flux [W/m²]
        fnet      : np.ndarray     net flux [W/m²]
        htr       : np.ndarray     heating rate [K/day]
    """
    with open(path, 'r') as fh:
        lines = fh.readlines()

    # Find all "Wavenumbers:" line indices
    block_starts = []
    for i, line in enumerate(lines):
        if 'Wavenumbers:' in line:
            block_starts.append(i)

    if not block_starts:
        raise ValueError(f"No 'Wavenumbers:' blocks found in {path}")

    results = []
    for b, bstart in enumerate(block_starts):
        hdr = lines[bstart]
        # Parse: " Wavenumbers: <w1> - <w2> cm-1, ATM      <n>"
        parts = hdr.split()
        # parts[0]='Wavenumbers:', parts[1]=w1, parts[2]='-', parts[3]=w2
        w1 = float(parts[1])
        w2 = float(parts[3])

        # Data lines start after the header (1 line) + units lines (2 lines)
        data_start = bstart + 3  # skip header + LEVEL header + units
        data_end = block_starts[b + 1] if b + 1 < len(block_starts) else len(lines)

        levels   = []
        pressure = []
        uflux    = []
        dflux    = []
        fnet     = []
        htr      = []

        for i in range(data_start, data_end):
            line = lines[i].strip()
            if not line:
                continue
            # Skip any non-data lines (e.g. revision info at end of file)
            fields = line.split()
            if len(fields) < 6:
                continue
            try:
                lev = int(fields[0])
                pres = float(fields[1])
                uf   = float(fields[2])
                df   = float(fields[3])
                fn   = float(fields[4])
                ht   = float(fields[5])
            except (ValueError, IndexError):
                continue

            levels.append(lev)
            pressure.append(pres)
            uflux.append(uf)
            dflux.append(df)
            fnet.append(fn)
            htr.append(ht)

        results.append({
            'wavenums1': w1,
            'wavenums2': w2,
            'level':    np.array(levels,   dtype=np.int32),
            'pressure': np.array(pressure, dtype=np.float64),
            'uflux':    np.array(uflux,    dtype=np.float64),
            'dflux':    np.array(dflux,    dtype=np.float64),
            'fnet':     np.array(fnet,     dtype=np.float64),
            'htr':      np.array(htr,      dtype=np.float64),
        })

    return results


# ---------------------------------------------------------------------------
# read_input
# ---------------------------------------------------------------------------
def read_input(path):
    """
    Read a NetCDF input file produced from INPUT_RRTM.

    Parameters
    ----------
    path : str or path-like
        Path to input NetCDF file.

    Returns
    -------
    dict with the same keys as parse_fortran_input:
        pavel, tavel, pz, tz, wkl, wbroad, coldry,
        tbound, semiss, iout, iscat, numangs, icld, nlayers
    """
    ds = nc.Dataset(str(path), 'r')
    try:
        pavel   = np.array(ds.variables['pavel'][:],  dtype=np.float64)
        tavel   = np.array(ds.variables['tavel'][:],  dtype=np.float64)
        pz      = np.array(ds.variables['pz'][:],     dtype=np.float64)
        tz      = np.array(ds.variables['tz'][:],     dtype=np.float64)
        wkl     = np.array(ds.variables['wkl'][:],    dtype=np.float64)
        wbroad  = np.array(ds.variables['wbroad'][:], dtype=np.float64)

        tbound  = float(ds.variables['tbound'][:])
        semiss  = np.array(ds.variables['semiss'][:], dtype=np.float64)

        iout    = int(ds.variables['iout'][:])
        icld    = int(ds.variables['icld'][:])
        iscat   = int(ds.variables['iscat'][:])
        numangs = int(ds.variables['numangs'][:])

        nlayers = len(pavel)
        coldry  = wbroad.copy()
    finally:
        ds.close()

    return {
        'pavel':   pavel,
        'tavel':   tavel,
        'pz':      pz,
        'tz':      tz,
        'wkl':     wkl,
        'wbroad':  wbroad,
        'coldry':  coldry,
        'tbound':  tbound,
        'semiss':  semiss,
        'iout':    iout,
        'iscat':   iscat,
        'numangs': numangs,
        'icld':    icld,
        'nlayers': nlayers,
    }


# ---------------------------------------------------------------------------
# write_output
# ---------------------------------------------------------------------------
def write_output(path, result, pz):
    """
    Write RRTM_LW output to a NetCDF file.

    Parameters
    ----------
    path : str or path-like
        Destination NetCDF file path.
    result : dict
        Output dict from rtreg(), containing at minimum:
            totuflux  (nlayers+1,)   broadband upward flux [W/m²]
            totdflux  (nlayers+1,)   broadband downward flux [W/m²]
            fnet      (nlayers+1,)   broadband net flux [W/m²]
            htr       (nlayers+1,)   broadband heating rate [K/day]
        Optionally (when iout=99):
            band_totuflux  (16, nlayers+1)
            band_totdflux  (16, nlayers+1)
            band_fnet      (16, nlayers+1)
            band_htr       (16, nlayers+1)
    pz : np.ndarray, shape (nlayers+1,)
        Level pressures [mb]. pz[0] = surface.
    """
    pz = np.asarray(pz, dtype=np.float64)
    nlevels = len(pz)

    ds = nc.Dataset(str(path), 'w', format='NETCDF4')
    try:
        # Dimensions
        ds.createDimension('level', nlevels)

        has_bands = 'band_totuflux' in result
        if has_bands:
            ds.createDimension('band', 16)

        # ---- level pressure ----
        v = ds.createVariable('pz', 'f8', ('level',))
        v[:] = pz
        v.units = 'mb'
        v.long_name = 'level pressure'

        # ---- broadband fluxes ----
        v = ds.createVariable('totuflux', 'f8', ('level',))
        v[:] = np.asarray(result['totuflux'], dtype=np.float64)
        v.units = 'W/m2'
        v.long_name = 'total broadband upward flux'

        v = ds.createVariable('totdflux', 'f8', ('level',))
        v[:] = np.asarray(result['totdflux'], dtype=np.float64)
        v.units = 'W/m2'
        v.long_name = 'total broadband downward flux'

        v = ds.createVariable('fnet', 'f8', ('level',))
        v[:] = np.asarray(result['fnet'], dtype=np.float64)
        v.units = 'W/m2'
        v.long_name = 'total net flux (upward minus downward)'

        v = ds.createVariable('htr', 'f8', ('level',))
        v[:] = np.asarray(result['htr'], dtype=np.float64)
        v.units = 'K/day'
        v.long_name = 'total heating rate'

        # ---- per-band variables (only when result contains band_totuflux) ----
        if has_bands:
            v = ds.createVariable('band_totuflux', 'f8', ('band', 'level'))
            v[:] = np.asarray(result['band_totuflux'], dtype=np.float64)
            v.units = 'W/m2'
            v.long_name = 'per-band upward flux'

            v = ds.createVariable('band_totdflux', 'f8', ('band', 'level'))
            v[:] = np.asarray(result['band_totdflux'], dtype=np.float64)
            v.units = 'W/m2'
            v.long_name = 'per-band downward flux'

            v = ds.createVariable('band_fnet', 'f8', ('band', 'level'))
            v[:] = np.asarray(result['band_fnet'], dtype=np.float64)
            v.units = 'W/m2'
            v.long_name = 'per-band net flux (upward minus downward)'

            v = ds.createVariable('band_htr', 'f8', ('band', 'level'))
            v[:] = np.asarray(result['band_htr'], dtype=np.float64)
            v.units = 'K/day'
            v.long_name = 'per-band heating rate'

            v = ds.createVariable('wavenum1', 'f8', ('band',))
            v[:] = _WAVENUM1
            v.units = 'cm-1'
            v.long_name = 'band lower wavenumber'

            v = ds.createVariable('wavenum2', 'f8', ('band',))
            v[:] = _WAVENUM2
            v.units = 'cm-1'
            v.long_name = 'band upper wavenumber'

        ds.description = 'RRTM_LW longwave radiative transfer output'

    finally:
        ds.close()
