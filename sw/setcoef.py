"""
Coefficient interpolation — Python translation of RRTM_SW setcoef.f.

setcoef() computes:
  - Pressure/temperature interpolation indices and weights (jp, jt, jt1, fac*)
  - Column amounts for all radiatively active gases
  - Water vapour self- and foreign-continuum scaling factors
  - CO2MULT: trace-gas CO2 scaling for SW bands
  - COLMOL: total air column (dry + H2O)
  - Three tropopause/troposphere split indices: laytrop, layswtch, laylow

No Planck function is computed (SW has no thermal emission).

All index arrays (jp, jt, jt1, indself, indfor) are 0-based.
Fortran COMMON /SELF/ and /FOREIGN/ arrays are 1-based; subtract 1 here.
"""
import math
import numpy as np
from .constants import PREFLOG, TREF, LAYTROP_PLOG_THRESHOLD, LAYSWTCH_PLOG_THRESHOLD


def setcoef(pavel, tavel, tz, wkl, coldry, wbroad):
    """
    Compute interpolation coefficients for one SW atmospheric profile.

    Parameters
    ----------
    pavel   : (nlayers,)    layer average pressure [mb]
    tavel   : (nlayers,)    layer average temperature [K]
    tz      : (nlayers+1,)  level temperatures [K]; tz[0]=surface
    wkl     : (7, nlayers)  gas column amounts [molec/cm²]:
                            index 0=H2O, 1=CO2, 2=O3, 3=N2O, 4=CO, 5=CH4, 6=O2
    coldry  : (nlayers,)    dry air column [molec/cm²]
    wbroad  : (nlayers,)    broadening gas column [molec/cm²]

    Returns
    -------
    sc : dict with all interpolation arrays needed by taumol SW routines.
         Keys: laytrop, layswtch, laylow,
               jp, jt, jt1,
               fac00, fac01, fac10, fac11,
               colh2o, colco2, colo3, coln2o, colch4, colo2,
               co2mult, colmol, coldry,
               selffac, selffrac, indself,
               forfac, forfrac, indfor,
               pavel
    """
    nlayers = len(pavel)

    # Fortran STPFAC = 296./1013.
    STPFAC = 296.0 / 1013.0

    # ── Allocate output arrays ────────────────────────────────────────────────
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
    colch4   = np.zeros(nlayers, dtype=np.float64)
    colo2    = np.zeros(nlayers, dtype=np.float64)
    colmol   = np.zeros(nlayers, dtype=np.float64)
    co2mult  = np.zeros(nlayers, dtype=np.float64)

    selffac  = np.zeros(nlayers, dtype=np.float64)
    selffrac = np.zeros(nlayers, dtype=np.float64)
    indself  = np.zeros(nlayers, dtype=np.int32)

    forfac   = np.zeros(nlayers, dtype=np.float64)
    forfrac  = np.zeros(nlayers, dtype=np.float64)
    indfor   = np.zeros(nlayers, dtype=np.int32)

    # ── Transition-level counters (initialised to 0, Fortran lines 109-111) ──
    laytrop  = 0   # last layer with plog > 4.56   (~96 mb, tropopause)
    layswtch = 0   # last layer with plog >= 6.62  (~750 mb)
    laylow   = 0   # same as layswtch in Fortran

    # ── Per-layer loop (Fortran DO 7000 LAY=1,NLAYERS) ───────────────────────
    for lay in range(nlayers):

        # -------------------------------------------------------------------
        # Pressure interpolation index JP (Fortran 1-based 1..58)
        # Fortran: JP(LAY) = INT(36. - 5*(PLOG+0.04)), clamped 1..58
        # Python 0-based: jp[lay] = jp_f - 1
        # -------------------------------------------------------------------
        plog   = math.log(pavel[lay])
        jp_f   = int(36.0 - 5.0 * (plog + 0.04))
        jp_f   = max(1, min(58, jp_f))
        jp[lay] = jp_f - 1          # 0-based index into PREFLOG / TREF

        jp1 = jp[lay] + 1           # 0-based index for the next pressure level
        # Fortran JP1 = JP(LAY)+1 (Fortran 1-based), so TREF(JP1) → TREF[jp1]

        fp = 5.0 * (PREFLOG[jp[lay]] - plog)

        # -------------------------------------------------------------------
        # Temperature interpolation index JT at JP (Fortran 1-based 1..4)
        # Fortran: JT(LAY) = INT(3. + (TAVEL(LAY)-TREF(JP(LAY)))/15.), clamped 1..4
        # -------------------------------------------------------------------
        jt_f   = int(3.0 + (tavel[lay] - TREF[jp[lay]]) / 15.0)
        jt_f   = max(1, min(4, jt_f))
        jt[lay] = jt_f - 1          # 0-based (0..3)
        ft = (tavel[lay] - TREF[jp[lay]]) / 15.0 - float(jt_f - 3)

        # -------------------------------------------------------------------
        # Temperature interpolation index JT1 at JP1 (Fortran 1-based 1..4)
        # Fortran: JT1(LAY) = INT(3. + (TAVEL(LAY)-TREF(JP1))/15.), clamped 1..4
        # -------------------------------------------------------------------
        jt1_f   = int(3.0 + (tavel[lay] - TREF[jp1]) / 15.0)
        jt1_f   = max(1, min(4, jt1_f))
        jt1[lay] = jt1_f - 1        # 0-based (0..3)
        ft1 = (tavel[lay] - TREF[jp1]) / 15.0 - float(jt1_f - 3)

        # -------------------------------------------------------------------
        # Water vapour mixing ratio and pressure-scaled factor
        # Fortran lines 149-150
        # -------------------------------------------------------------------
        water    = wkl[0, lay] / coldry[lay]
        scalefac = pavel[lay] * STPFAC / tavel[lay]

        # -------------------------------------------------------------------
        # Branch: lower atmosphere (plog > 4.56) vs upper (plog <= 4.56)
        # Fortran: IF (PLOG .LE. 4.56) GO TO 5300
        # -------------------------------------------------------------------
        if plog > LAYTROP_PLOG_THRESHOLD:
            # ── Lower atmosphere ───────────────────────────────────────────
            # Fortran lines 154-196
            laytrop += 1

            # layswtch and laylow both track layers with plog >= 6.62
            # Fortran line 156: IF (PLOG .GE. 6.62) LAYLOW = LAYLOW + 1
            if plog >= LAYSWTCH_PLOG_THRESHOLD:
                laylow   += 1

            # Foreign continuum (Fortran lines 160-163)
            # FORFAC(LAY) = SCALEFAC / (1.+WATER)
            # FACTOR = (332.0-TAVEL(LAY))/36.0
            # INDFOR(LAY) = MIN(2, MAX(1, INT(FACTOR)))   [Fortran 1-based]
            # FORFRAC(LAY) = FACTOR - FLOAT(INDFOR(LAY))
            forfac_val   = scalefac / (1.0 + water)
            factor       = (332.0 - tavel[lay]) / 36.0
            indfor_f     = min(2, max(1, int(factor)))    # Fortran 1..2
            forfac[lay]  = forfac_val
            forfrac[lay] = factor - float(indfor_f)
            indfor[lay]  = indfor_f - 1                   # Python 0-based (0..1)

            # Self continuum (Fortran lines 167-170)
            # SELFFAC(LAY) = WATER * FORFAC(LAY)
            # FACTOR = (TAVEL(LAY)-188.0)/7.2
            # INDSELF(LAY) = MIN(9, MAX(1, INT(FACTOR)-7))  [Fortran 1-based]
            # SELFFRAC(LAY) = FACTOR - FLOAT(INDSELF(LAY)+7)
            selffac_val   = water * forfac_val
            factor        = (tavel[lay] - 188.0) / 7.2
            indself_f     = min(9, max(1, int(factor) - 7))   # Fortran 1..9
            selffac[lay]  = selffac_val
            selffrac[lay] = factor - float(indself_f + 7)
            indself[lay]  = indself_f - 1                 # Python 0-based (0..8)

            # Column amounts (Fortran lines 173-191)
            # Note Fortran WKL(1)=H2O, (2)=CO2, (3)=O3, (4)=N2O, (6)=CH4, (7)=O2
            # Python wkl is 0-indexed: 0=H2O, 1=CO2, 2=O3, 3=N2O, 5=CH4, 6=O2
            colh2o[lay] = 1.0e-20 * wkl[0, lay]
            colco2[lay] = 1.0e-20 * wkl[1, lay]
            colo3[lay]  = 1.0e-20 * wkl[2, lay]
            coln2o[lay] = 1.0e-20 * wkl[3, lay]
            colch4[lay] = 1.0e-20 * wkl[5, lay]
            colo2[lay]  = 1.0e-20 * wkl[6, lay]
            # Fortran: COLMOL(LAY) = 1.E-20 * COLDRY(LAY) + COLH2O(LAY)
            # COLH2O is already scaled; only COLDRY gets the 1e-20 here.
            colmol[lay] = 1.0e-20 * coldry[lay] + colh2o[lay]

            # Zero-fill guards (Fortran lines 188-191)
            if colco2[lay] == 0.0: colco2[lay] = 1.0e-32 * coldry[lay]
            if coln2o[lay] == 0.0: coln2o[lay] = 1.0e-32 * coldry[lay]
            if colch4[lay] == 0.0: colch4[lay] = 1.0e-32 * coldry[lay]
            if colo2[lay]  == 0.0: colo2[lay]  = 1.0e-32 * coldry[lay]

            # CO2MULT (Fortran lines 193-195)
            # Using E = 1334.2 cm-1
            # CO2REG = 3.55E-24 * COLDRY(LAY)
            # CO2MULT(LAY) = (COLCO2(LAY) - CO2REG) *
            #      272.63*EXP(-1919.4/TAVEL(LAY))/(8.7604E-4*TAVEL(LAY))
            co2reg       = 3.55e-24 * coldry[lay]
            co2mult[lay] = ((colco2[lay] - co2reg) *
                            272.63 * math.exp(-1919.4 / tavel[lay]) /
                            (8.7604e-4 * tavel[lay]))

        else:
            # ── Upper atmosphere (label 5300, Fortran lines 199-222) ───────

            # Foreign continuum (Fortran lines 203-206)
            # FORFAC(LAY) = SCALEFAC / (1.+WATER)
            # FACTOR = (TAVEL(LAY)-188.0)/36.0
            # INDFOR(LAY) = 3                   [Fortran hardcoded]
            # FORFRAC(LAY) = FACTOR - 1.0
            forfac_val   = scalefac / (1.0 + water)
            factor       = (tavel[lay] - 188.0) / 36.0
            forfac[lay]  = forfac_val
            forfrac[lay] = factor - 1.0
            indfor[lay]  = 2            # Python 0-based (Fortran INDFOR=3)

            # No SELFFAC/SELFFRAC/INDSELF in upper atmosphere; arrays stay 0.

            # Column amounts (Fortran lines 209-219)
            colh2o[lay] = 1.0e-20 * wkl[0, lay]
            colco2[lay] = 1.0e-20 * wkl[1, lay]
            colo3[lay]  = 1.0e-20 * wkl[2, lay]
            coln2o[lay] = 1.0e-20 * wkl[3, lay]
            colch4[lay] = 1.0e-20 * wkl[5, lay]
            colo2[lay]  = 1.0e-20 * wkl[6, lay]
            colmol[lay] = 1.0e-20 * coldry[lay] + colh2o[lay]

            # Zero-fill guards (Fortran lines 216-219)
            if colco2[lay] == 0.0: colco2[lay] = 1.0e-32 * coldry[lay]
            if coln2o[lay] == 0.0: coln2o[lay] = 1.0e-32 * coldry[lay]
            if colch4[lay] == 0.0: colch4[lay] = 1.0e-32 * coldry[lay]
            if colo2[lay]  == 0.0: colo2[lay]  = 1.0e-32 * coldry[lay]

            # CO2MULT (Fortran lines 220-222, identical formula to lower atm)
            co2reg       = 3.55e-24 * coldry[lay]
            co2mult[lay] = ((colco2[lay] - co2reg) *
                            272.63 * math.exp(-1919.4 / tavel[lay]) /
                            (8.7604e-4 * tavel[lay]))

        # -------------------------------------------------------------------
        # Bilinear interpolation weights — applied to ALL layers
        # (label 5400, Fortran lines 233-237)
        # COMPFP = 1. - FP
        # FAC10(LAY) = COMPFP * FT
        # FAC00(LAY) = COMPFP * (1. - FT)
        # FAC11(LAY) = FP * FT1
        # FAC01(LAY) = FP * (1. - FT1)
        # -------------------------------------------------------------------
        compfp       = 1.0 - fp
        fac10[lay]   = compfp * ft
        fac00[lay]   = compfp * (1.0 - ft)
        fac11[lay]   = fp * ft1
        fac01[lay]   = fp * (1.0 - ft1)

    # ── Return coefficient dict ───────────────────────────────────────────────
    return dict(
        laytrop=laytrop,
        layswtch=layswtch,
        laylow=laylow,
        jp=jp,
        jt=jt,
        jt1=jt1,
        fac00=fac00,
        fac01=fac01,
        fac10=fac10,
        fac11=fac11,
        colh2o=colh2o,
        colco2=colco2,
        colo3=colo3,
        coln2o=coln2o,
        colch4=colch4,
        colo2=colo2,
        co2mult=co2mult,
        colmol=colmol,
        coldry=coldry,
        selffac=selffac,
        selffrac=selffrac,
        indself=indself,
        forfac=forfac,
        forfrac=forfrac,
        indfor=indfor,
        pavel=pavel,
    )
