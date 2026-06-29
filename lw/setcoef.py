"""
Coefficient interpolation — Python translation of RRTM_LW setcoef.f.

setcoef() computes:
  - Planck function arrays (planklay, planklev, plankbnd)
  - Pressure/temperature interpolation indices and weights
  - Column amounts and continuum scaling factors
  - Binary species ratio arrays used by taumol

All index arrays (jp, jt, jt1, indself, indfor, indminor) are 0-based.
"""
import numpy as np
from .constants import PREFLOG, TREF, LAYTROP_PLOG_THRESHOLD


def setcoef(pavel, tavel, tz, tbound, wkl, coldry, wbroad, semiss,
            totplnk, totplk16, chi_mls, istart=0):
    """
    Compute interpolation coefficients for one atmospheric profile.

    Parameters
    ----------
    pavel   : (nlayers,)      layer average pressure [mb]
    tavel   : (nlayers,)      layer average temperature [K]
    tz      : (nlayers+1,)    level temperatures [K]; tz[0] = surface level
    tbound  : scalar          surface (boundary) temperature [K]
    wkl     : (nmol, nlayers) gas column amounts [mol/cm²]; col 0=H2O,1=CO2,…
    coldry  : (nlayers,)      dry air column amount [mol/cm²]
    wbroad  : (nlayers,)      broadening gas column [mol/cm²]
    semiss  : (16,)           surface emissivity per band (0-indexed)
    totplnk : (181, 16)       integrated Planck function table
    totplk16: (181,)          integrated Planck for band 16 (full range)
    chi_mls : (9, 59)         MLS reference gas VMR ratios (0-indexed)
    istart  : int             0-based starting band (0 = broadband, 15 = band 16 only)

    Returns
    -------
    sc : dict containing all arrays needed by taumol and rtreg
    """
    nlayers = len(pavel)
    nbands  = 16

    STPFAC = 296.0 / 1013.0

    # ── Planck function index for surface and level 0 ────────────────────────
    # TOTPLNK table covers temperatures 160 K..340 K in 1 K steps (indices 0..180)
    # Fortran: INDBOUND = INT(TBOUND - 159), clamped 1..180
    # Python 0-based: indbound = int(TBOUND-160) clamped 0..179
    indbound = int(tbound - 160.0)
    indbound = max(0, min(179, indbound))
    tbndfrac = tbound - 160.0 - float(indbound)

    indlev0  = int(tz[0] - 160.0)
    indlev0  = max(0, min(179, indlev0))
    t0frac   = tz[0] - 160.0 - float(indlev0)

    # ── Allocate output arrays ───────────────────────────────────────────────
    plankbnd = np.zeros(nbands, dtype=np.float64)
    planklay = np.zeros((nlayers, nbands), dtype=np.float64)
    planklev = np.zeros((nlayers + 1, nbands), dtype=np.float64)  # index 0 = surface

    jp       = np.zeros(nlayers, dtype=np.int32)
    jt       = np.zeros(nlayers, dtype=np.int32)
    jt1      = np.zeros(nlayers, dtype=np.int32)
    fac00    = np.zeros(nlayers, dtype=np.float64)
    fac01    = np.zeros(nlayers, dtype=np.float64)
    fac10    = np.zeros(nlayers, dtype=np.float64)
    fac11    = np.zeros(nlayers, dtype=np.float64)

    colh2o   = np.zeros(nlayers, dtype=np.float64)
    colco2   = np.zeros(nlayers, dtype=np.float64)
    colo3    = np.zeros(nlayers, dtype=np.float64)
    coln2o   = np.zeros(nlayers, dtype=np.float64)
    colco    = np.zeros(nlayers, dtype=np.float64)
    colch4   = np.zeros(nlayers, dtype=np.float64)
    colo2    = np.zeros(nlayers, dtype=np.float64)
    colbrd   = np.zeros(nlayers, dtype=np.float64)

    selffac  = np.zeros(nlayers, dtype=np.float64)
    selffrac = np.zeros(nlayers, dtype=np.float64)
    indself  = np.zeros(nlayers, dtype=np.int32)

    forfac   = np.zeros(nlayers, dtype=np.float64)
    forfrac  = np.zeros(nlayers, dtype=np.float64)
    indfor   = np.zeros(nlayers, dtype=np.int32)

    minorfrac    = np.zeros(nlayers, dtype=np.float64)
    indminor     = np.zeros(nlayers, dtype=np.int32)
    scaleminor   = np.zeros(nlayers, dtype=np.float64)
    scaleminorn2 = np.zeros(nlayers, dtype=np.float64)

    rat_h2oco2   = np.zeros(nlayers, dtype=np.float64)
    rat_h2oco2_1 = np.zeros(nlayers, dtype=np.float64)
    rat_h2oo3    = np.zeros(nlayers, dtype=np.float64)
    rat_h2oo3_1  = np.zeros(nlayers, dtype=np.float64)
    rat_h2on2o   = np.zeros(nlayers, dtype=np.float64)
    rat_h2on2o_1 = np.zeros(nlayers, dtype=np.float64)
    rat_h2och4   = np.zeros(nlayers, dtype=np.float64)
    rat_h2och4_1 = np.zeros(nlayers, dtype=np.float64)
    rat_n2oco2   = np.zeros(nlayers, dtype=np.float64)
    rat_n2oco2_1 = np.zeros(nlayers, dtype=np.float64)
    rat_o3co2    = np.zeros(nlayers, dtype=np.float64)
    rat_o3co2_1  = np.zeros(nlayers, dtype=np.float64)
    rat_ch4so2   = np.zeros(nlayers, dtype=np.float64)
    rat_ch4so2_1 = np.zeros(nlayers, dtype=np.float64)

    laytrop = 0

    # ── Per-layer loop ────────────────────────────────────────────────────────
    for lay in range(nlayers):   # lay: 0-based (Fortran LAY = lay+1)

        # --- Planck function indices (Fortran uses 1-based; Python uses 0-based) ---
        # Fortran: INDLAY = INT(TAVEL(LAY) - 159.)  clamped 1..180
        # Python 0-based: int(T - 160) clamped 0..179
        indlay  = max(0, min(179, int(tavel[lay] - 160.0)))
        tlayfrac = tavel[lay] - 160.0 - float(indlay)

        indlev  = max(0, min(179, int(tz[lay + 1] - 160.0)))
        tlevfrac = tz[lay + 1] - 160.0 - float(indlev)

        # --- Planck function for bands 1..15 (Python 0..14) ---
        for ib in range(15):
            if lay == 0:
                dbdt = totplnk[indbound + 1, ib] - totplnk[indbound, ib]
                plankbnd[ib] = semiss[ib] * (totplnk[indbound, ib] + tbndfrac * dbdt)
                dbdt = totplnk[indlev0 + 1, ib] - totplnk[indlev0, ib]
                planklev[0, ib] = totplnk[indlev0, ib] + t0frac * dbdt

            dbdt = totplnk[indlev + 1, ib] - totplnk[indlev, ib]
            dbdt_lay = totplnk[indlay + 1, ib] - totplnk[indlay, ib]
            planklay[lay, ib] = totplnk[indlay, ib] + tlayfrac * dbdt_lay
            planklev[lay + 1, ib] = totplnk[indlev, ib] + tlevfrac * dbdt

        # --- Band 16 Planck (Python index 15) ---
        ib = 15
        # For broadband (istart==0): use TOTPLNK for band 16 (same as bands 1-15)
        # For single-band 16 (istart==15): use TOTPLK16 (full range to infinity)
        if istart == 15:
            tp = totplk16
        else:
            tp = totplnk[:, ib]

        if lay == 0:
            dbdt = tp[indbound + 1] - tp[indbound]
            plankbnd[ib] = semiss[ib] * (tp[indbound] + tbndfrac * dbdt)
            dbdt = tp[indlev0 + 1] - tp[indlev0]
            planklev[0, ib] = tp[indlev0] + t0frac * dbdt

        dbdt = tp[indlev + 1] - tp[indlev]
        dbdt_lay = tp[indlay + 1] - tp[indlay]
        planklay[lay, ib] = tp[indlay] + tlayfrac * dbdt_lay
        planklev[lay + 1, ib] = tp[indlev] + tlevfrac * dbdt

        # --- Pressure interpolation ---
        plog = np.log(pavel[lay])

        # JP: 0-based (Fortran JP clamped 1..58 → Python 0..57)
        jp_f = int(36.0 - 5.0 * (plog + 0.04))
        jp_f = max(1, min(58, jp_f))
        jp[lay] = jp_f - 1   # 0-based
        jp1 = jp[lay] + 1    # Python 0-based index for the next pressure level

        fp = 5.0 * (PREFLOG[jp[lay]] - plog)

        # --- Temperature interpolation ---
        jt_f = int(3.0 + (tavel[lay] - TREF[jp[lay]]) / 15.0)
        jt_f = max(1, min(4, jt_f))
        jt[lay] = jt_f - 1   # 0-based (0..3)
        ft = (tavel[lay] - TREF[jp[lay]]) / 15.0 - float(jt_f - 3)

        jt1_f = int(3.0 + (tavel[lay] - TREF[jp1]) / 15.0)
        jt1_f = max(1, min(4, jt1_f))
        jt1[lay] = jt1_f - 1  # 0-based
        ft1 = (tavel[lay] - TREF[jp1]) / 15.0 - float(jt1_f - 3)

        water     = wkl[0, lay] / coldry[lay]
        scalefac  = pavel[lay] * STPFAC / tavel[lay]

        # --- Bilinear interpolation weights ---
        compfp      = 1.0 - fp
        fac10[lay]  = compfp * ft
        fac00[lay]  = compfp * (1.0 - ft)
        fac11[lay]  = fp * ft1
        fac01[lay]  = fp * (1.0 - ft1)

        # --- Column amounts (mol/cm² × 1e-20) ---
        colh2o[lay] = 1.0e-20 * wkl[0, lay]
        colco2[lay] = 1.0e-20 * wkl[1, lay]
        colo3[lay]  = 1.0e-20 * wkl[2, lay]
        coln2o[lay] = 1.0e-20 * wkl[3, lay]
        colco[lay]  = 1.0e-20 * wkl[4, lay]
        colch4[lay] = 1.0e-20 * wkl[5, lay]
        colo2[lay]  = 1.0e-20 * wkl[6, lay]
        colbrd[lay] = 1.0e-20 * wbroad[lay]

        # Zero-fill guard
        if colco2[lay] == 0.0: colco2[lay] = 1.0e-32 * coldry[lay]
        if colo3[lay]  == 0.0: colo3[lay]  = 1.0e-32 * coldry[lay]
        if coln2o[lay] == 0.0: coln2o[lay] = 1.0e-32 * coldry[lay]
        if colco[lay]  == 0.0: colco[lay]  = 1.0e-32 * coldry[lay]
        if colch4[lay] == 0.0: colch4[lay] = 1.0e-32 * coldry[lay]

        # --- Layer type: lower (below tropopause) or upper ---
        if plog > LAYTROP_PLOG_THRESHOLD:
            # Lower atmosphere (Fortran LAY ≤ LAYTROP)
            laytrop += 1

            forfac_val = scalefac / (1.0 + water)
            factor = (332.0 - tavel[lay]) / 36.0
            indfor_val = min(2, max(1, int(factor)))          # Fortran 1..2
            forfac[lay]  = forfac_val
            forfrac[lay] = factor - float(indfor_val)
            indfor[lay]  = indfor_val - 1                     # Python 0-based

            selffac_val = water * forfac_val
            factor = (tavel[lay] - 188.0) / 7.2
            # Fortran: INDSELF = MIN(9, MAX(1, INT(FACTOR)-7))
            indself_val = min(9, max(1, int(factor) - 7))     # Fortran 1..9
            selffac[lay]  = colh2o[lay] * selffac_val
            selffrac[lay] = factor - float(indself_val + 7)
            indself[lay]  = indself_val - 1                   # Python 0-based

            scaleminor[lay]   = pavel[lay] / tavel[lay]
            scaleminorn2[lay] = (pavel[lay] / tavel[lay]) * (
                wbroad[lay] / (coldry[lay] + wkl[0, lay]))

            factor = (tavel[lay] - 180.8) / 7.2
            indminor_val = min(18, max(1, int(factor)))       # Fortran 1..18
            indminor[lay]  = indminor_val - 1                 # Python 0-based
            minorfrac[lay] = factor - float(indminor_val)

            # chi_mls: 0-indexed (sp: 0..8, ip: 0..58)
            jp0, jp0p1 = jp[lay], jp[lay] + 1
            rat_h2oco2[lay]   = chi_mls[0, jp0]   / chi_mls[1, jp0]
            rat_h2oco2_1[lay] = chi_mls[0, jp0p1] / chi_mls[1, jp0p1]
            rat_h2oo3[lay]    = chi_mls[0, jp0]   / chi_mls[2, jp0]
            rat_h2oo3_1[lay]  = chi_mls[0, jp0p1] / chi_mls[2, jp0p1]
            rat_h2on2o[lay]   = chi_mls[0, jp0]   / chi_mls[3, jp0]
            rat_h2on2o_1[lay] = chi_mls[0, jp0p1] / chi_mls[3, jp0p1]
            rat_h2och4[lay]   = chi_mls[0, jp0]   / chi_mls[5, jp0]
            rat_h2och4_1[lay] = chi_mls[0, jp0p1] / chi_mls[5, jp0p1]
            rat_n2oco2[lay]   = chi_mls[3, jp0]   / chi_mls[1, jp0]
            rat_n2oco2_1[lay] = chi_mls[3, jp0p1] / chi_mls[1, jp0p1]
            rat_ch4so2[lay]   = chi_mls[5, jp0]   / chi_mls[8, jp0]
            rat_ch4so2_1[lay] = chi_mls[5, jp0p1] / chi_mls[8, jp0p1]

        else:
            # Upper atmosphere (Fortran LAY > LAYTROP)
            forfac_val = scalefac / (1.0 + water)
            factor = (tavel[lay] - 188.0) / 36.0
            # Fortran: INDFOR = 3, FORFRAC = factor - 1.0
            forfac[lay]  = forfac_val
            forfrac[lay] = factor - 1.0
            indfor[lay]  = 2                                  # Python 0-based (Fortran 3)

            scaleminor[lay]   = pavel[lay] / tavel[lay]
            scaleminorn2[lay] = (pavel[lay] / tavel[lay]) * (
                wbroad[lay] / (coldry[lay] + wkl[0, lay]))

            factor = (tavel[lay] - 180.8) / 7.2
            indminor_val = min(18, max(1, int(factor)))
            indminor[lay]  = indminor_val - 1
            minorfrac[lay] = factor - float(indminor_val)

            jp0, jp0p1 = jp[lay], jp[lay] + 1
            rat_h2oco2[lay]   = chi_mls[0, jp0]   / chi_mls[1, jp0]
            rat_h2oco2_1[lay] = chi_mls[0, jp0p1] / chi_mls[1, jp0p1]
            rat_o3co2[lay]    = chi_mls[2, jp0]   / chi_mls[1, jp0]
            rat_o3co2_1[lay]  = chi_mls[2, jp0p1] / chi_mls[1, jp0p1]
            rat_ch4so2[lay]   = chi_mls[5, jp0]   / chi_mls[8, jp0]
            rat_ch4so2_1[lay] = chi_mls[5, jp0p1] / chi_mls[8, jp0p1]

            # Rescale selffac for upper atmosphere (SELFFAC=0 above laytrop in Fortran)
            selffac[lay]  = 0.0
            selffrac[lay] = 0.0
            indself[lay]  = 0

        # Rescale forfac by H2O column (done at end of each layer in Fortran)
        forfac[lay] = colh2o[lay] * forfac[lay]

    colso2 = np.zeros(nlayers, dtype=np.float64)
    if wkl.shape[0] > 8:
        colso2 = 1.0e-20 * wkl[8, :]
    colso2[colso2 == 0.0] = 1.0e-32 * coldry[colso2 == 0.0]

    return dict(
        laytrop=laytrop,
        pavel=pavel, coldry=coldry,
        plankbnd=plankbnd, planklay=planklay, planklev=planklev,
        jp=jp, jt=jt, jt1=jt1,
        fac00=fac00, fac01=fac01, fac10=fac10, fac11=fac11,
        colh2o=colh2o, colco2=colco2, colo3=colo3, coln2o=coln2o,
        colco=colco, colch4=colch4, colo2=colo2, colbrd=colbrd,
        colso2=colso2,
        selffac=selffac, selffrac=selffrac, indself=indself,
        forfac=forfac, forfrac=forfrac, indfor=indfor,
        minorfrac=minorfrac, indminor=indminor,
        scaleminor=scaleminor, scaleminorn2=scaleminorn2,
        rat_h2oco2=rat_h2oco2, rat_h2oco2_1=rat_h2oco2_1,
        rat_h2oo3=rat_h2oo3,   rat_h2oo3_1=rat_h2oo3_1,
        rat_h2on2o=rat_h2on2o, rat_h2on2o_1=rat_h2on2o_1,
        rat_h2och4=rat_h2och4, rat_h2och4_1=rat_h2och4_1,
        rat_n2oco2=rat_n2oco2, rat_n2oco2_1=rat_n2oco2_1,
        rat_o3co2=rat_o3co2,   rat_o3co2_1=rat_o3co2_1,
        rat_ch4so2=rat_ch4so2, rat_ch4so2_1=rat_ch4so2_1,
    )
