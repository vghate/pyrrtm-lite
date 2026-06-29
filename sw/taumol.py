"""
Optical depth computation — Python translation of RRTM_SW taumoldis.f.

taumol_all(sc, bands, adjflux) → (tau, ssa, sfluxzen)
  tau     : np.ndarray shape (14, nlayers, 16)  — total optical depth
  ssa     : np.ndarray shape (14, nlayers, 16)  — single-scattering albedo
  sfluxzen: np.ndarray shape (14, 16)           — solar flux at reference level

Band mapping (Python index → Fortran band):
  0 → 16  (2600-3250 cm-1)
  1 → 17  (3250-4000 cm-1)
  2 → 18  (4000-4650 cm-1)
  3 → 19  (4650-5150 cm-1)
  4 → 20  (5150-6150 cm-1)
  5 → 21  (6150-7700 cm-1)
  6 → 22  (7700-8050 cm-1)
  7 → 23  (8050-12850 cm-1)
  8 → 24  (12850-16000 cm-1)
  9 → 25  (16000-22650 cm-1)
  10→ 26  (22650-29000 cm-1) — pure Rayleigh
  11→ 27  (29000-38000 cm-1)
  12→ 28  (38000-50000 cm-1)
  13→ 29  (820-2600 cm-1)

Index conventions (0-based throughout):
  band b=0..13 corresponds to Fortran IBAND=16..29
  g-point ig=0..15
  layer lay=0..nlayers-1 (lay=0 is bottom; Fortran LAY=1)

Fortran JP, JT, JT1, INDSELF, INDFOR are stored 0-based in sc.

Fortran ABSA layout (column-major EQUIVALENCE, 1-based in Fortran):
  ka stored as (NSPA, 5, 13, 16) in coeffs.nc.
  ABSA(IND, IG) where:
    IND = ((JP-1)*5 + (JT-1))*NSPA + JS   (Fortran 1-based)
  In Python (0-based):
    ka[js, jt, jp, ig]  where js=JS-1, jt=JT-1, jp=JP-1

  For NSPA=9 strides in Fortran:
    ABSA(IND0+1,  IG) = ka[js+1, jt,   jp, ig]   (+1 species)
    ABSA(IND0+9,  IG) = ka[js,   jt+1, jp, ig]   (+9 = NSPA, next temp)
    ABSA(IND0+10, IG) = ka[js+1, jt+1, jp, ig]

  For NSPB=5 strides in upper atm:
    ABSB(IND0+1, IG) = kb[js+1, jt,   jp_u, ig]
    ABSB(IND0+5, IG) = kb[js,   jt+1, jp_u, ig]   (+5 = NSPB)
    ABSB(IND0+6, IG) = kb[js+1, jt+1, jp_u, ig]

  For NSPA=1 (simple): ka stored as (5, 13, 16).
    ABSA(IND0,   IG) = ka[jt,   jp, ig]
    ABSA(IND0+1, IG) = ka[jt+1, jp, ig]
    ABSA(IND1,   IG) = ka[jt1,  jp+1, ig]
    ABSA(IND1+1, IG) = ka[jt1+1,jp+1, ig]

  For NSPB=1 (simple upper): kb stored as (5, 47, 16).
    jp_u = jp - 12  (0-based; Fortran JP-13 → Python jp-12)
    ABSB(IND0,   IG) = kb[jt,   jp_u, ig]
    ABSB(IND0+1, IG) = kb[jt+1, jp_u, ig]
    ABSB(IND1,   IG) = kb[jt1,  jp_u+1, ig]
    ABSB(IND1+1, IG) = kb[jt1+1,jp_u+1, ig]
"""
import numpy as np
from .constants import ONEMINUS


# ── Low-level interpolation helpers ──────────────────────────────────────────

def _lerp16(arr, i, frac):
    """Linear interpolation: arr[i] + frac*(arr[i+1]-arr[i]), shape (16,)."""
    return arr[i] + frac * (arr[i + 1] - arr[i])


def _self_for(selfref, forref, inds, selffac, selffrac, indf, forfac, forfrac):
    """Self + foreign continuum (returned as separate arrays, shape (16,))."""
    tau_self = selffac * (selfref[inds, :] + selffrac * (selfref[inds+1, :] - selfref[inds, :]))
    tau_for  = forfac  * (forref[indf, :]  + forfrac  * (forref[indf+1,  :] - forref[indf,  :]))
    return tau_self, tau_for


def _simple_ka(col, ka, jp, jt, jt1, f00, f10, f01, f11):
    """
    NSPA=1 lower-atmosphere bilinear interpolation.
    ka shape: (5, 13, 16).
    Fortran: ABSA(IND0)=ka(JT,JP), ABSA(IND0+1)=ka(JT+1,JP),
             ABSA(IND1)=ka(JT1,JP+1), ABSA(IND1+1)=ka(JT1+1,JP+1)
    """
    return col * (f00 * ka[jt,     jp,   :] +
                  f10 * ka[jt + 1, jp,   :] +
                  f01 * ka[jt1,    jp+1, :] +
                  f11 * ka[jt1+1,  jp+1, :])


def _simple_kb(col, kb, jp, jt, jt1, f00, f10, f01, f11):
    """
    NSPB=1 upper-atmosphere bilinear interpolation.
    kb shape: (5, 47, 16).
    jp_u = jp - 12 (0-based upper index; Fortran ((JP-13)*5+(JT-1))*1+1).
    """
    jp_u = jp - 12
    return col * (f00 * kb[jt,     jp_u,   :] +
                  f10 * kb[jt + 1, jp_u,   :] +
                  f01 * kb[jt1,    jp_u+1, :] +
                  f11 * kb[jt1+1,  jp_u+1, :])


def _nspa9(speccomb, ka, js, jp, jt, jt1, f00, f10, f01, f11, fs):
    """
    NSPA=9 lower-atmosphere bilinear interpolation (linear in species).
    ka shape: (9, 5, 13, 16).

    Fortran IND0 = ((JP-1)*5+(JT-1))*9+JS, IND1 = (JP*5+(JT1-1))*9+JS
    With Python 0-based jp, jt, jt1, js:

    IND0: ka[js, jt,   jp,   :],  ka[js+1, jt,   jp,   :],
          ka[js, jt+1, jp,   :],  ka[js+1, jt+1, jp,   :]
    IND1: ka[js, jt1,  jp+1, :],  ka[js+1, jt1,  jp+1, :],
          ka[js, jt1+1,jp+1, :],  ka[js+1, jt1+1,jp+1, :]
    """
    fac000 = (1.0 - fs) * f00;  fac100 = fs * f00
    fac010 = (1.0 - fs) * f10;  fac110 = fs * f10
    fac001 = (1.0 - fs) * f01;  fac101 = fs * f01
    fac011 = (1.0 - fs) * f11;  fac111 = fs * f11
    return speccomb * (
        fac000 * ka[js,   jt,    jp,   :] +
        fac100 * ka[js+1, jt,    jp,   :] +
        fac010 * ka[js,   jt+1,  jp,   :] +
        fac110 * ka[js+1, jt+1,  jp,   :] +
        fac001 * ka[js,   jt1,   jp+1, :] +
        fac101 * ka[js+1, jt1,   jp+1, :] +
        fac011 * ka[js,   jt1+1, jp+1, :] +
        fac111 * ka[js+1, jt1+1, jp+1, :])


def _nspb5(speccomb, kb, js, jp_u, jt, jt1, f00, f10, f01, f11, fs):
    """
    NSPB=5 upper-atmosphere bilinear interpolation (linear in species).
    kb shape: (5, 5, 47, 16).

    Fortran IND0=((JP-13)*5+(JT-1))*5+JS, IND1=((JP-12)*5+(JT1-1))*5+JS
    Strides: +1=species, +5=temp.
    jp_u = jp - 12.

    IND0: kb[js, jt,   jp_u, :], kb[js+1, jt,   jp_u, :],
          kb[js, jt+1, jp_u, :], kb[js+1, jt+1, jp_u, :]
    IND1: kb[js, jt1,  jp_u+1,:], kb[js+1, jt1,  jp_u+1,:],
          kb[js, jt1+1,jp_u+1,:], kb[js+1, jt1+1,jp_u+1,:]
    """
    fac000 = (1.0 - fs) * f00;  fac100 = fs * f00
    fac010 = (1.0 - fs) * f10;  fac110 = fs * f10
    fac001 = (1.0 - fs) * f01;  fac101 = fs * f01
    fac011 = (1.0 - fs) * f11;  fac111 = fs * f11
    return speccomb * (
        fac000 * kb[js,   jt,    jp_u,   :] +
        fac100 * kb[js+1, jt,    jp_u,   :] +
        fac010 * kb[js,   jt+1,  jp_u,   :] +
        fac110 * kb[js+1, jt+1,  jp_u,   :] +
        fac001 * kb[js,   jt1,   jp_u+1, :] +
        fac101 * kb[js+1, jt1,   jp_u+1, :] +
        fac011 * kb[js,   jt1+1, jp_u+1, :] +
        fac111 * kb[js+1, jt1+1, jp_u+1, :])


def _specparm_jsfs(col_a, col_b, strrat, specmult_factor=8.0):
    """
    Compute speccomb, js (0-based), fs for a binary species mixture.
    Returns (speccomb, js, fs).
    """
    speccomb = col_a + strrat * col_b
    specparm = col_a / speccomb
    if specparm >= ONEMINUS:
        specparm = ONEMINUS
    specmult = specmult_factor * specparm
    js = int(specmult)     # 0-based Python index (Fortran JS-1)
    fs = specmult - js
    return speccomb, js, fs


def _ssa(tauray, tau_total):
    """SSA = tauray / tau_total (0 where tau_total <= 0)."""
    return np.where(tau_total > 0.0, tauray / tau_total, 0.0)


# ── Band 0 (Fortran 16): 2600-3250 cm-1 ──────────────────────────────────────
# Low:  H2O + CH4  NSPA=9  SPECMULT=8  STRRAT1=252.131
# High: CH4        NSPB=1
# LAYREFFR=18 (upper loop check)
# sfluxref shape (16,) — scalar, set at LAYSOLFR

def taumol_b00(sc, c):
    """Band 0 (Fortran 16): 2600-3250 cm-1. Low H2O+CH4, high CH4."""
    ka       = c['ka']          # (9,5,13,16)
    kb       = c['kb']          # (5,47,16)
    selfref  = c['selfref']     # (10,16)
    forref   = c['forref']      # (3,16)
    sfluxref = c['sfluxref']    # (16,)
    rayl     = float(c['rayl'])

    STRRAT1  = 252.131
    LAYREFFR = 18               # Fortran 1-based threshold

    nl      = sc['laytrop']
    nlayers = len(sc['jp'])
    tau     = np.zeros((nlayers, 16))
    ssa     = np.zeros((nlayers, 16))
    sfluxzen = np.zeros(16)

    # ── Lower atmosphere ──────────────────────────────────────────────────────
    for lay in range(nl):
        jp  = sc['jp'][lay];  jt = sc['jt'][lay];  jt1 = sc['jt1'][lay]
        f00 = sc['fac00'][lay];  f10 = sc['fac10'][lay]
        f01 = sc['fac01'][lay];  f11 = sc['fac11'][lay]
        inds = sc['indself'][lay];  indf = sc['indfor'][lay]

        speccomb, js, fs = _specparm_jsfs(
            sc['colh2o'][lay], sc['colch4'][lay], STRRAT1, 8.0)

        taug_k = _nspa9(speccomb, ka, js, jp, jt, jt1, f00, f10, f01, f11, fs)

        tau_self, tau_for = _self_for(
            selfref, forref, inds, sc['selffac'][lay], sc['selffrac'][lay],
            indf, sc['forfac'][lay], sc['forfrac'][lay])

        tauray = sc['colmol'][lay] * rayl

        # Fortran: TAUG = SPECCOMB*(FAC*ABSA) + COLH2O*(SELF+FOR) + TAURAY
        tau[lay, :] = taug_k + sc['colh2o'][lay] * (tau_self + tau_for) + tauray
        ssa[lay, :] = _ssa(tauray, tau[lay, :])

    # ── Upper atmosphere ──────────────────────────────────────────────────────
    laysolfr = nlayers - 1
    for lay in range(nl, nlayers):
        jp  = sc['jp'][lay];  jt = sc['jt'][lay];  jt1 = sc['jt1'][lay]
        f00 = sc['fac00'][lay];  f10 = sc['fac10'][lay]
        f01 = sc['fac01'][lay];  f11 = sc['fac11'][lay]

        # Fortran: IF (JP(LAY-1)<LAYREFFR .AND. JP(LAY)>=LAYREFFR) LAYSOLFR=LAY
        # Python 0-based: jp[lay-1] < LAYREFFR-1 and jp[lay] >= LAYREFFR-1
        if lay >= nl and sc['jp'][lay-1] < LAYREFFR - 1 and jp >= LAYREFFR - 1:
            laysolfr = lay

        taug_k  = _simple_kb(sc['colch4'][lay], kb, jp, jt, jt1, f00, f10, f01, f11)
        tauray  = sc['colmol'][lay] * rayl

        tau[lay, :] = taug_k + tauray
        ssa[lay, :] = _ssa(tauray, tau[lay, :])
        if lay == laysolfr:
            sfluxzen[:] = sfluxref[:]

    return tau, ssa, sfluxzen


# ── Band 1 (Fortran 17): 3250-4000 cm-1 ──────────────────────────────────────
# Low:  H2O + CO2  NSPA=9  SPECMULT=8  STRRAT=0.364641
# High: H2O + CO2  NSPB=5  SPECMULT=4
# LAYREFFR=30 (upper loop check)
# sfluxref shape (5,16) — interpolated in species at LAYSOLFR

def taumol_b01(sc, c):
    """Band 1 (Fortran 17): 3250-4000 cm-1. Low/High H2O+CO2."""
    ka       = c['ka']          # (9,5,13,16)
    kb       = c['kb']          # (5,5,47,16)
    selfref  = c['selfref']     # (10,16)
    forref   = c['forref']      # (4,16)
    sfluxref = c['sfluxref']    # (5,16)
    rayl     = float(c['rayl'])

    STRRAT   = 0.364641
    LAYREFFR = 30

    nl      = sc['laytrop']
    nlayers = len(sc['jp'])
    tau     = np.zeros((nlayers, 16))
    ssa     = np.zeros((nlayers, 16))
    sfluxzen = np.zeros(16)

    # ── Lower atmosphere ──────────────────────────────────────────────────────
    for lay in range(nl):
        jp  = sc['jp'][lay];  jt = sc['jt'][lay];  jt1 = sc['jt1'][lay]
        f00 = sc['fac00'][lay];  f10 = sc['fac10'][lay]
        f01 = sc['fac01'][lay];  f11 = sc['fac11'][lay]
        inds = sc['indself'][lay];  indf = sc['indfor'][lay]

        speccomb, js, fs = _specparm_jsfs(
            sc['colh2o'][lay], sc['colco2'][lay], STRRAT, 8.0)

        taug_k = _nspa9(speccomb, ka, js, jp, jt, jt1, f00, f10, f01, f11, fs)
        tau_self, tau_for = _self_for(
            selfref, forref, inds, sc['selffac'][lay], sc['selffrac'][lay],
            indf, sc['forfac'][lay], sc['forfrac'][lay])
        tauray = sc['colmol'][lay] * rayl

        # Fortran: SPECCOMB*(FAC*ABSA) + COLH2O*(SELF+FOR) + TAURAY
        tau[lay, :] = taug_k + sc['colh2o'][lay] * (tau_self + tau_for) + tauray
        ssa[lay, :] = _ssa(tauray, tau[lay, :])

    # ── Upper atmosphere ──────────────────────────────────────────────────────
    laysolfr = nlayers - 1
    for lay in range(nl, nlayers):
        jp  = sc['jp'][lay];  jt = sc['jt'][lay];  jt1 = sc['jt1'][lay]
        f00 = sc['fac00'][lay];  f10 = sc['fac10'][lay]
        f01 = sc['fac01'][lay];  f11 = sc['fac11'][lay]
        indf = sc['indfor'][lay]
        jp_u = jp - 12

        if lay >= nl and sc['jp'][lay-1] < LAYREFFR - 1 and jp >= LAYREFFR - 1:
            laysolfr = lay

        speccomb, js, fs = _specparm_jsfs(
            sc['colh2o'][lay], sc['colco2'][lay], STRRAT, 4.0)

        taug_k  = _nspb5(speccomb, kb, js, jp_u, jt, jt1, f00, f10, f01, f11, fs)
        # Fortran: + COLH2O * FORFAC * (FORREF(INDF) + FORFRAC*(FORREF(INDF+1)-FORREF(INDF)))
        tau_for = (sc['forfac'][lay] *
                   (forref[indf, :] + sc['forfrac'][lay] *
                    (forref[indf+1, :] - forref[indf, :])))
        tauray  = sc['colmol'][lay] * rayl

        tau[lay, :] = taug_k + sc['colh2o'][lay] * tau_for + tauray
        ssa[lay, :] = _ssa(tauray, tau[lay, :])
        if lay == laysolfr:
            sfluxzen[:] = (sfluxref[js, :] +
                           fs * (sfluxref[js+1, :] - sfluxref[js, :]))

    return tau, ssa, sfluxzen


# ── Band 2 (Fortran 18): 4000-4650 cm-1 ──────────────────────────────────────
# Low:  H2O + CH4  NSPA=9  SPECMULT=8  STRRAT=38.9589
# High: CH4        NSPB=1
# LAYREFFR=6 (lower loop check)
# sfluxref shape (9,16) — interpolated in species at LAYSOLFR (set in lower loop)

def taumol_b02(sc, c):
    """Band 2 (Fortran 18): 4000-4650 cm-1. Low H2O+CH4, high CH4."""
    ka       = c['ka']          # (9,5,13,16)
    kb       = c['kb']          # (5,47,16)
    selfref  = c['selfref']     # (10,16)
    forref   = c['forref']      # (3,16)
    sfluxref = c['sfluxref']    # (9,16)
    rayl     = float(c['rayl'])

    STRRAT   = 38.9589
    LAYREFFR = 6

    nl      = sc['laytrop']
    nlayers = len(sc['jp'])
    tau     = np.zeros((nlayers, 16))
    ssa     = np.zeros((nlayers, 16))
    sfluxzen = np.zeros(16)

    # LAYSOLFR starts at LAYTROP (Fortran: LAYSOLFR=LAYTROP)
    laysolfr = nl - 1 if nl > 0 else 0

    # ── Lower atmosphere ──────────────────────────────────────────────────────
    for lay in range(nl):
        jp  = sc['jp'][lay];  jt = sc['jt'][lay];  jt1 = sc['jt1'][lay]
        f00 = sc['fac00'][lay];  f10 = sc['fac10'][lay]
        f01 = sc['fac01'][lay];  f11 = sc['fac11'][lay]
        inds = sc['indself'][lay];  indf = sc['indfor'][lay]

        # Fortran: IF (JP(LAY)<LAYREFFR .AND. JP(LAY+1)>=LAYREFFR) LAYSOLFR=MIN(LAY+1,LAYTROP)
        # Python 0-based: jp[lay] < LAYREFFR-1 and jp[lay+1] >= LAYREFFR-1
        if (lay < nlayers - 1 and
                sc['jp'][lay] < LAYREFFR - 1 and
                sc['jp'][lay + 1] >= LAYREFFR - 1):
            laysolfr = min(lay + 1, nl - 1)

        speccomb, js, fs = _specparm_jsfs(
            sc['colh2o'][lay], sc['colch4'][lay], STRRAT, 8.0)

        taug_k = _nspa9(speccomb, ka, js, jp, jt, jt1, f00, f10, f01, f11, fs)
        tau_self, tau_for = _self_for(
            selfref, forref, inds, sc['selffac'][lay], sc['selffrac'][lay],
            indf, sc['forfac'][lay], sc['forfrac'][lay])
        tauray = sc['colmol'][lay] * rayl

        tau[lay, :] = taug_k + sc['colh2o'][lay] * (tau_self + tau_for) + tauray
        ssa[lay, :] = _ssa(tauray, tau[lay, :])
        if lay == laysolfr:
            sfluxzen[:] = (sfluxref[js, :] +
                           fs * (sfluxref[js+1, :] - sfluxref[js, :]))

    # ── Upper atmosphere ──────────────────────────────────────────────────────
    for lay in range(nl, nlayers):
        jp  = sc['jp'][lay];  jt = sc['jt'][lay];  jt1 = sc['jt1'][lay]
        f00 = sc['fac00'][lay];  f10 = sc['fac10'][lay]
        f01 = sc['fac01'][lay];  f11 = sc['fac11'][lay]

        taug_k = _simple_kb(sc['colch4'][lay], kb, jp, jt, jt1, f00, f10, f01, f11)
        tauray = sc['colmol'][lay] * rayl

        tau[lay, :] = taug_k + tauray
        ssa[lay, :] = _ssa(tauray, tau[lay, :])

    return tau, ssa, sfluxzen


# ── Band 3 (Fortran 19): 4650-5150 cm-1 ──────────────────────────────────────
# Low:  H2O + CO2  NSPA=9  SPECMULT=8  STRRAT=5.49281
# High: CO2        NSPB=1
# LAYREFFR=3 (lower loop check)
# sfluxref shape (9,16)

def taumol_b03(sc, c):
    """Band 3 (Fortran 19): 4650-5150 cm-1. Low H2O+CO2, high CO2."""
    ka       = c['ka']          # (9,5,13,16)
    kb       = c['kb']          # (5,47,16)
    selfref  = c['selfref']     # (10,16)
    forref   = c['forref']      # (3,16)
    sfluxref = c['sfluxref']    # (9,16)
    rayl     = float(c['rayl'])

    STRRAT   = 5.49281
    LAYREFFR = 3

    nl      = sc['laytrop']
    nlayers = len(sc['jp'])
    tau     = np.zeros((nlayers, 16))
    ssa     = np.zeros((nlayers, 16))
    sfluxzen = np.zeros(16)

    laysolfr = nl - 1 if nl > 0 else 0

    # ── Lower atmosphere ──────────────────────────────────────────────────────
    for lay in range(nl):
        jp  = sc['jp'][lay];  jt = sc['jt'][lay];  jt1 = sc['jt1'][lay]
        f00 = sc['fac00'][lay];  f10 = sc['fac10'][lay]
        f01 = sc['fac01'][lay];  f11 = sc['fac11'][lay]
        inds = sc['indself'][lay];  indf = sc['indfor'][lay]

        if (lay < nlayers - 1 and
                sc['jp'][lay] < LAYREFFR - 1 and
                sc['jp'][lay + 1] >= LAYREFFR - 1):
            laysolfr = min(lay + 1, nl - 1)

        speccomb, js, fs = _specparm_jsfs(
            sc['colh2o'][lay], sc['colco2'][lay], STRRAT, 8.0)

        taug_k = _nspa9(speccomb, ka, js, jp, jt, jt1, f00, f10, f01, f11, fs)
        tau_self, tau_for = _self_for(
            selfref, forref, inds, sc['selffac'][lay], sc['selffrac'][lay],
            indf, sc['forfac'][lay], sc['forfrac'][lay])
        tauray = sc['colmol'][lay] * rayl

        tau[lay, :] = taug_k + sc['colh2o'][lay] * (tau_self + tau_for) + tauray
        ssa[lay, :] = _ssa(tauray, tau[lay, :])
        if lay == laysolfr:
            sfluxzen[:] = (sfluxref[js, :] +
                           fs * (sfluxref[js+1, :] - sfluxref[js, :]))

    # ── Upper atmosphere ──────────────────────────────────────────────────────
    for lay in range(nl, nlayers):
        jp  = sc['jp'][lay];  jt = sc['jt'][lay];  jt1 = sc['jt1'][lay]
        f00 = sc['fac00'][lay];  f10 = sc['fac10'][lay]
        f01 = sc['fac01'][lay];  f11 = sc['fac11'][lay]

        taug_k = _simple_kb(sc['colco2'][lay], kb, jp, jt, jt1, f00, f10, f01, f11)
        tauray = sc['colmol'][lay] * rayl

        tau[lay, :] = taug_k + tauray
        ssa[lay, :] = _ssa(tauray, tau[lay, :])

    return tau, ssa, sfluxzen


# ── Band 4 (Fortran 20): 5150-6150 cm-1 ──────────────────────────────────────
# Low:  H2O  NSPA=1  self+for
# High: H2O  NSPB=1  for only
# LAYREFFR=3 (lower loop check)
# sfluxref shape (16,) — scalar
# Extra: ABSCH4 per g-point CH4 absorption (added outside COLH2O multiply)
#
# Fortran lower:
#   TAUG = COLH2O * ((FAC*ABSA) + SELFFAC*SELF + FORFAC*FOR) + TAURAY + COLCH4*ABSCH4
# Fortran upper:
#   TAUG = COLH2O * ((FAC*ABSB) + FORFAC*FOR) + TAURAY + COLCH4*ABSCH4

def taumol_b04(sc, c):
    """Band 4 (Fortran 20): 5150-6150 cm-1. Low/High H2O. Extra CH4 absorption."""
    ka       = c['ka']          # (5,13,16)
    kb       = c['kb']          # (5,47,16)
    selfref  = c['selfref']     # (10,16)
    forref   = c['forref']      # (4,16)
    sfluxref = c['sfluxref']    # (16,)
    absch4   = c['absch4']      # (16,) per g-point
    rayl     = float(c['rayl'])

    LAYREFFR = 3

    nl      = sc['laytrop']
    nlayers = len(sc['jp'])
    tau     = np.zeros((nlayers, 16))
    ssa     = np.zeros((nlayers, 16))
    sfluxzen = np.zeros(16)

    laysolfr = nl - 1 if nl > 0 else 0

    # ── Lower atmosphere ──────────────────────────────────────────────────────
    for lay in range(nl):
        jp  = sc['jp'][lay];  jt = sc['jt'][lay];  jt1 = sc['jt1'][lay]
        f00 = sc['fac00'][lay];  f10 = sc['fac10'][lay]
        f01 = sc['fac01'][lay];  f11 = sc['fac11'][lay]
        inds = sc['indself'][lay];  indf = sc['indfor'][lay]

        if (lay < nlayers - 1 and
                sc['jp'][lay] < LAYREFFR - 1 and
                sc['jp'][lay + 1] >= LAYREFFR - 1):
            laysolfr = min(lay + 1, nl - 1)

        # NSPA=1: ka = absa_base (no species mixing)
        absa_base = (f00 * ka[jt,     jp,   :] +
                     f10 * ka[jt + 1, jp,   :] +
                     f01 * ka[jt1,    jp+1, :] +
                     f11 * ka[jt1+1,  jp+1, :])

        tau_self = (sc['selffac'][lay] *
                    (selfref[inds, :] + sc['selffrac'][lay] *
                     (selfref[inds+1, :] - selfref[inds, :])))
        tau_for  = (sc['forfac'][lay] *
                    (forref[indf, :] + sc['forfrac'][lay] *
                     (forref[indf+1, :] - forref[indf, :])))
        tauray = sc['colmol'][lay] * rayl

        # Fortran: COLH2O*(ABSA + SELFFAC*SELF + FORFAC*FOR) + TAURAY + COLCH4*ABSCH4
        tau[lay, :] = (sc['colh2o'][lay] * (absa_base + tau_self + tau_for) +
                       tauray + sc['colch4'][lay] * absch4[:])
        ssa[lay, :] = _ssa(tauray, tau[lay, :])
        if lay == laysolfr:
            sfluxzen[:] = sfluxref[:]

    # ── Upper atmosphere ──────────────────────────────────────────────────────
    for lay in range(nl, nlayers):
        jp  = sc['jp'][lay];  jt = sc['jt'][lay];  jt1 = sc['jt1'][lay]
        f00 = sc['fac00'][lay];  f10 = sc['fac10'][lay]
        f01 = sc['fac01'][lay];  f11 = sc['fac11'][lay]
        indf = sc['indfor'][lay]
        jp_u = jp - 12

        absb_base = (f00 * kb[jt,     jp_u,   :] +
                     f10 * kb[jt + 1, jp_u,   :] +
                     f01 * kb[jt1,    jp_u+1, :] +
                     f11 * kb[jt1+1,  jp_u+1, :])
        tau_for = (sc['forfac'][lay] *
                   (forref[indf, :] + sc['forfrac'][lay] *
                    (forref[indf+1, :] - forref[indf, :])))
        tauray = sc['colmol'][lay] * rayl

        # Fortran: COLH2O*(ABSB + FORFAC*FOR) + TAURAY + COLCH4*ABSCH4
        tau[lay, :] = (sc['colh2o'][lay] * (absb_base + tau_for) +
                       tauray + sc['colch4'][lay] * absch4[:])
        ssa[lay, :] = _ssa(tauray, tau[lay, :])

    return tau, ssa, sfluxzen


# ── Band 5 (Fortran 21): 6150-7700 cm-1 ──────────────────────────────────────
# Low:  H2O + CO2  NSPA=9  SPECMULT=8  STRRAT=0.0045321
# High: H2O + CO2  NSPB=5  SPECMULT=4
# LAYREFFR=8 (lower loop check)
# sfluxref shape (9,16)
#
# Fortran lower:
#   TAUG = SPECCOMB*(FAC*ABSA) + TAURAY + COLH2O*(SELFFAC*SELF + FORFAC*FOR)
# Fortran upper:
#   TAUG = SPECCOMB*(FAC*ABSB) + TAURAY + COLH2O*FORFAC*FOR

def taumol_b05(sc, c):
    """Band 5 (Fortran 21): 6150-7700 cm-1. Low/High H2O+CO2."""
    ka       = c['ka']          # (9,5,13,16)
    kb       = c['kb']          # (5,5,47,16)
    selfref  = c['selfref']     # (10,16)
    forref   = c['forref']      # (4,16)
    sfluxref = c['sfluxref']    # (9,16)
    rayl     = float(c['rayl'])

    STRRAT   = 0.0045321
    LAYREFFR = 8

    nl      = sc['laytrop']
    nlayers = len(sc['jp'])
    tau     = np.zeros((nlayers, 16))
    ssa     = np.zeros((nlayers, 16))
    sfluxzen = np.zeros(16)

    laysolfr = nl - 1 if nl > 0 else 0

    # ── Lower atmosphere ──────────────────────────────────────────────────────
    for lay in range(nl):
        jp  = sc['jp'][lay];  jt = sc['jt'][lay];  jt1 = sc['jt1'][lay]
        f00 = sc['fac00'][lay];  f10 = sc['fac10'][lay]
        f01 = sc['fac01'][lay];  f11 = sc['fac11'][lay]
        inds = sc['indself'][lay];  indf = sc['indfor'][lay]

        if (lay < nlayers - 1 and
                sc['jp'][lay] < LAYREFFR - 1 and
                sc['jp'][lay + 1] >= LAYREFFR - 1):
            laysolfr = min(lay + 1, nl - 1)

        speccomb, js, fs = _specparm_jsfs(
            sc['colh2o'][lay], sc['colco2'][lay], STRRAT, 8.0)

        taug_k = _nspa9(speccomb, ka, js, jp, jt, jt1, f00, f10, f01, f11, fs)
        tau_self, tau_for = _self_for(
            selfref, forref, inds, sc['selffac'][lay], sc['selffrac'][lay],
            indf, sc['forfac'][lay], sc['forfrac'][lay])
        tauray = sc['colmol'][lay] * rayl

        # Fortran: SPECCOMB*(FAC*ABSA) + TAURAY + COLH2O*(SELF+FOR)
        tau[lay, :] = taug_k + tauray + sc['colh2o'][lay] * (tau_self + tau_for)
        ssa[lay, :] = _ssa(tauray, tau[lay, :])
        if lay == laysolfr:
            sfluxzen[:] = (sfluxref[js, :] +
                           fs * (sfluxref[js+1, :] - sfluxref[js, :]))

    # ── Upper atmosphere ──────────────────────────────────────────────────────
    for lay in range(nl, nlayers):
        jp  = sc['jp'][lay];  jt = sc['jt'][lay];  jt1 = sc['jt1'][lay]
        f00 = sc['fac00'][lay];  f10 = sc['fac10'][lay]
        f01 = sc['fac01'][lay];  f11 = sc['fac11'][lay]
        indf = sc['indfor'][lay]
        jp_u = jp - 12

        speccomb, js, fs = _specparm_jsfs(
            sc['colh2o'][lay], sc['colco2'][lay], STRRAT, 4.0)

        taug_k = _nspb5(speccomb, kb, js, jp_u, jt, jt1, f00, f10, f01, f11, fs)
        tau_for = (sc['forfac'][lay] *
                   (forref[indf, :] + sc['forfrac'][lay] *
                    (forref[indf+1, :] - forref[indf, :])))
        tauray = sc['colmol'][lay] * rayl

        # Fortran: SPECCOMB*(FAC*ABSB) + TAURAY + COLH2O*FORFAC*FOR
        tau[lay, :] = taug_k + tauray + sc['colh2o'][lay] * tau_for
        ssa[lay, :] = _ssa(tauray, tau[lay, :])

    return tau, ssa, sfluxzen


# ── Band 6 (Fortran 22): 7700-8050 cm-1 ──────────────────────────────────────
# Low:  H2O + O2   NSPA=9  SPECMULT=8  STRRAT_eff = O2ADJ*STRRAT = 1.6*0.022708
# High: O2         NSPB=1
# LAYREFFR=2 (lower loop check)
# O2ADJ=1.6, O2CONT=4.35e-4*COLO2/(350*2)
# sfluxref shape (9,16)
#
# Fortran lower:
#   SPECCOMB = COLH2O + O2ADJ*STRRAT*COLO2
#   TAUG = SPECCOMB*(FAC*ABSA) + TAURAY + COLH2O*(SELFFAC*SELF+FORFAC*FOR) + O2CONT
# Fortran upper:
#   TAUG = COLO2*O2ADJ*(FAC*ABSB) + TAURAY + O2CONT

def taumol_b06(sc, c):
    """Band 6 (Fortran 22): 7700-8050 cm-1. Low H2O+O2, high O2."""
    ka       = c['ka']          # (9,5,13,16)
    kb       = c['kb']          # (5,47,16)
    selfref  = c['selfref']     # (10,16)
    forref   = c['forref']      # (3,16)
    sfluxref = c['sfluxref']    # (9,16)
    rayl     = float(c['rayl'])

    STRRAT   = 0.022708
    LAYREFFR = 2
    O2ADJ    = 1.6

    nl      = sc['laytrop']
    nlayers = len(sc['jp'])
    tau     = np.zeros((nlayers, 16))
    ssa     = np.zeros((nlayers, 16))
    sfluxzen = np.zeros(16)

    laysolfr = nl - 1 if nl > 0 else 0

    # ── Lower atmosphere ──────────────────────────────────────────────────────
    for lay in range(nl):
        jp  = sc['jp'][lay];  jt = sc['jt'][lay];  jt1 = sc['jt1'][lay]
        f00 = sc['fac00'][lay];  f10 = sc['fac10'][lay]
        f01 = sc['fac01'][lay];  f11 = sc['fac11'][lay]
        inds = sc['indself'][lay];  indf = sc['indfor'][lay]

        if (lay < nlayers - 1 and
                sc['jp'][lay] < LAYREFFR - 1 and
                sc['jp'][lay + 1] >= LAYREFFR - 1):
            laysolfr = min(lay + 1, nl - 1)

        o2cont = 4.35e-4 * sc['colo2'][lay] / (350.0 * 2.0)

        # Fortran: SPECCOMB = COLH2O + O2ADJ*STRRAT*COLO2
        speccomb, js, fs = _specparm_jsfs(
            sc['colh2o'][lay], sc['colo2'][lay], O2ADJ * STRRAT, 8.0)

        taug_k = _nspa9(speccomb, ka, js, jp, jt, jt1, f00, f10, f01, f11, fs)
        tau_self, tau_for = _self_for(
            selfref, forref, inds, sc['selffac'][lay], sc['selffrac'][lay],
            indf, sc['forfac'][lay], sc['forfrac'][lay])
        tauray = sc['colmol'][lay] * rayl

        # Fortran: SPECCOMB*(FAC*ABSA) + TAURAY + COLH2O*(SELF+FOR) + O2CONT
        tau[lay, :] = (taug_k + tauray +
                       sc['colh2o'][lay] * (tau_self + tau_for) + o2cont)
        ssa[lay, :] = _ssa(tauray, tau[lay, :])
        if lay == laysolfr:
            sfluxzen[:] = (sfluxref[js, :] +
                           fs * (sfluxref[js+1, :] - sfluxref[js, :]))

    # ── Upper atmosphere ──────────────────────────────────────────────────────
    for lay in range(nl, nlayers):
        jp  = sc['jp'][lay];  jt = sc['jt'][lay];  jt1 = sc['jt1'][lay]
        f00 = sc['fac00'][lay];  f10 = sc['fac10'][lay]
        f01 = sc['fac01'][lay];  f11 = sc['fac11'][lay]

        o2cont = 4.35e-4 * sc['colo2'][lay] / (350.0 * 2.0)

        # Fortran: COLO2*O2ADJ * ABSB
        taug_k = _simple_kb(sc['colo2'][lay] * O2ADJ, kb, jp, jt, jt1,
                             f00, f10, f01, f11)
        tauray = sc['colmol'][lay] * rayl

        tau[lay, :] = taug_k + tauray + o2cont
        ssa[lay, :] = _ssa(tauray, tau[lay, :])

    return tau, ssa, sfluxzen


# ── Band 7 (Fortran 23): 8050-12850 cm-1 ─────────────────────────────────────
# Low:  H2O  NSPA=1  GIVFAC=1.029
# High: pure Rayleigh only (SSA=1.0)
# LAYREFFR=6 (lower loop check)
# sfluxref shape (16,)
# RAYL is per-g-point array in this band (not scalar)
#
# Fortran lower:
#   TAUG = COLH2O*(GIVFAC*(FAC*ABSA) + SELFFAC*SELF + FORFAC*FOR) + TAURAY
# Fortran upper:
#   TAUG = COLMOL*RAYL(IG);  SSA=1.0

def taumol_b07(sc, c):
    """Band 7 (Fortran 23): 8050-12850 cm-1. Low H2O+Giver, high pure Rayleigh."""
    ka       = c['ka']          # (5,13,16)
    selfref  = c['selfref']     # (10,16)
    forref   = c['forref']      # (3,16)
    sfluxref = c['sfluxref']    # (16,)
    rayl     = c['rayl']        # (16,) per g-point array

    LAYREFFR = 6
    GIVFAC   = 1.029

    nl      = sc['laytrop']
    nlayers = len(sc['jp'])
    tau     = np.zeros((nlayers, 16))
    ssa     = np.zeros((nlayers, 16))
    sfluxzen = np.zeros(16)

    laysolfr = nl - 1 if nl > 0 else 0

    # ── Lower atmosphere ──────────────────────────────────────────────────────
    for lay in range(nl):
        jp  = sc['jp'][lay];  jt = sc['jt'][lay];  jt1 = sc['jt1'][lay]
        f00 = sc['fac00'][lay];  f10 = sc['fac10'][lay]
        f01 = sc['fac01'][lay];  f11 = sc['fac11'][lay]
        inds = sc['indself'][lay];  indf = sc['indfor'][lay]

        if (lay < nlayers - 1 and
                sc['jp'][lay] < LAYREFFR - 1 and
                sc['jp'][lay + 1] >= LAYREFFR - 1):
            laysolfr = min(lay + 1, nl - 1)

        # NSPA=1 base without col multiply
        absa_base = (f00 * ka[jt,     jp,   :] +
                     f10 * ka[jt + 1, jp,   :] +
                     f01 * ka[jt1,    jp+1, :] +
                     f11 * ka[jt1+1,  jp+1, :])

        tau_self = (sc['selffac'][lay] *
                    (selfref[inds, :] + sc['selffrac'][lay] *
                     (selfref[inds+1, :] - selfref[inds, :])))
        tau_for  = (sc['forfac'][lay] *
                    (forref[indf, :] + sc['forfrac'][lay] *
                     (forref[indf+1, :] - forref[indf, :])))
        tauray = sc['colmol'][lay] * rayl[:]

        # Fortran: TAUG = COLH2O*(GIVFAC*ABSA + SELF + FOR) + TAURAY
        tau[lay, :] = (sc['colh2o'][lay] * (GIVFAC * absa_base + tau_self + tau_for) +
                       tauray)
        ssa[lay, :] = _ssa(tauray, tau[lay, :])
        if lay == laysolfr:
            sfluxzen[:] = sfluxref[:]

    # ── Upper atmosphere: pure Rayleigh ───────────────────────────────────────
    for lay in range(nl, nlayers):
        tauray = sc['colmol'][lay] * rayl[:]
        tau[lay, :] = tauray
        ssa[lay, :] = 1.0

    return tau, ssa, sfluxzen


# ── Band 8 (Fortran 24): 12850-16000 cm-1 ────────────────────────────────────
# Low:  H2O + O2  NSPA=9  SPECMULT=8  STRRAT=0.124692
# High: O2        NSPB=1
# LAYREFFR=1 (lower loop check)
# Extra: ABSO3A(16) lower, ABSO3B(16) upper
# Rayleigh: RAYLA(16,9) in lower (per gpt, per species mix), RAYLB(16) in upper
# sfluxref shape (9,16)
#
# Fortran lower:
#   TAURAY = COLMOL*(RAYLA(IG,JS) + FS*(RAYLA(IG,JS+1)-RAYLA(IG,JS)))
#   TAUG = SPECCOMB*(FAC*ABSA) + COLO3*ABSO3A + TAURAY + COLH2O*(SELF+FOR)
# Fortran upper:
#   TAURAY = COLMOL*RAYLB(IG)
#   TAUG = COLO2*(FAC*ABSB) + COLO3*ABSO3B + TAURAY

def taumol_b08(sc, c):
    """Band 8 (Fortran 24): 12850-16000 cm-1. Low H2O+O2, high O2. O3 extra."""
    ka       = c['ka']          # (9,5,13,16)
    kb       = c['kb']          # (5,47,16)
    selfref  = c['selfref']     # (10,16)
    forref   = c['forref']      # (3,16)
    sfluxref = c['sfluxref']    # (9,16)
    abso3a   = c['abso3a']      # (16,)
    abso3b   = c['abso3b']      # (16,)
    rayla    = c['rayla']        # (16,9) — RAYLA(IG, JS) in Fortran 1-based
    raylb    = c['raylb']        # (16,)

    STRRAT   = 0.124692
    LAYREFFR = 1

    nl      = sc['laytrop']
    nlayers = len(sc['jp'])
    tau     = np.zeros((nlayers, 16))
    ssa     = np.zeros((nlayers, 16))
    sfluxzen = np.zeros(16)

    laysolfr = nl - 1 if nl > 0 else 0

    # ── Lower atmosphere ──────────────────────────────────────────────────────
    for lay in range(nl):
        jp  = sc['jp'][lay];  jt = sc['jt'][lay];  jt1 = sc['jt1'][lay]
        f00 = sc['fac00'][lay];  f10 = sc['fac10'][lay]
        f01 = sc['fac01'][lay];  f11 = sc['fac11'][lay]
        inds = sc['indself'][lay];  indf = sc['indfor'][lay]

        if (lay < nlayers - 1 and
                sc['jp'][lay] < LAYREFFR - 1 and
                sc['jp'][lay + 1] >= LAYREFFR - 1):
            laysolfr = min(lay + 1, nl - 1)

        speccomb, js, fs = _specparm_jsfs(
            sc['colh2o'][lay], sc['colo2'][lay], STRRAT, 8.0)

        # Per-gpt Rayleigh interpolated in species: RAYLA(IG,JS)+FS*(RAYLA(IG,JS+1)-...)
        # rayla shape (16,9): dim0=gpt, dim1=species (0-based js)
        tauray = sc['colmol'][lay] * (rayla[:, js] + fs * (rayla[:, js+1] - rayla[:, js]))

        taug_k = _nspa9(speccomb, ka, js, jp, jt, jt1, f00, f10, f01, f11, fs)
        tau_self, tau_for = _self_for(
            selfref, forref, inds, sc['selffac'][lay], sc['selffrac'][lay],
            indf, sc['forfac'][lay], sc['forfrac'][lay])

        # Fortran: SPECCOMB*(FAC*ABSA) + COLO3*ABSO3A + TAURAY + COLH2O*(SELF+FOR)
        tau[lay, :] = (taug_k + sc['colo3'][lay] * abso3a[:] + tauray +
                       sc['colh2o'][lay] * (tau_self + tau_for))
        ssa[lay, :] = _ssa(tauray, tau[lay, :])
        if lay == laysolfr:
            sfluxzen[:] = (sfluxref[js, :] +
                           fs * (sfluxref[js+1, :] - sfluxref[js, :]))

    # ── Upper atmosphere ──────────────────────────────────────────────────────
    for lay in range(nl, nlayers):
        jp  = sc['jp'][lay];  jt = sc['jt'][lay];  jt1 = sc['jt1'][lay]
        f00 = sc['fac00'][lay];  f10 = sc['fac10'][lay]
        f01 = sc['fac01'][lay];  f11 = sc['fac11'][lay]

        taug_k = _simple_kb(sc['colo2'][lay], kb, jp, jt, jt1, f00, f10, f01, f11)
        tauray = sc['colmol'][lay] * raylb[:]

        tau[lay, :] = taug_k + sc['colo3'][lay] * abso3b[:] + tauray
        ssa[lay, :] = _ssa(tauray, tau[lay, :])

    return tau, ssa, sfluxzen


# ── Band 9 (Fortran 25): 16000-22650 cm-1 ────────────────────────────────────
# Low:  H2O  NSPA=1
# High: pure O3 + Rayleigh (no KB)
# LAYREFFR=2 (lower loop check)
# sfluxref shape (16,)
# RAYL per g-point array; ABSO3A(16) lower, ABSO3B(16) upper
#
# Fortran lower:
#   TAUG = COLH2O*(FAC*ABSA) + COLO3*ABSO3A(IG) + TAURAY
# Fortran upper:
#   TAUG = COLO3*ABSO3B(IG) + TAURAY

def taumol_b09(sc, c):
    """Band 9 (Fortran 25): 16000-22650 cm-1. Low H2O+O3. High O3+Rayleigh."""
    ka       = c['ka']          # (5,13,16)
    sfluxref = c['sfluxref']    # (16,)
    abso3a   = c['abso3a']      # (16,)
    abso3b   = c['abso3b']      # (16,)
    rayl     = c['rayl']        # (16,) per g-point

    LAYREFFR = 2

    nl      = sc['laytrop']
    nlayers = len(sc['jp'])
    tau     = np.zeros((nlayers, 16))
    ssa     = np.zeros((nlayers, 16))
    sfluxzen = np.zeros(16)

    laysolfr = nl - 1 if nl > 0 else 0

    # ── Lower atmosphere ──────────────────────────────────────────────────────
    for lay in range(nl):
        jp  = sc['jp'][lay];  jt = sc['jt'][lay];  jt1 = sc['jt1'][lay]
        f00 = sc['fac00'][lay];  f10 = sc['fac10'][lay]
        f01 = sc['fac01'][lay];  f11 = sc['fac11'][lay]

        if (lay < nlayers - 1 and
                sc['jp'][lay] < LAYREFFR - 1 and
                sc['jp'][lay + 1] >= LAYREFFR - 1):
            laysolfr = min(lay + 1, nl - 1)

        taug_k = _simple_ka(sc['colh2o'][lay], ka, jp, jt, jt1, f00, f10, f01, f11)
        tauray = sc['colmol'][lay] * rayl[:]

        tau[lay, :] = taug_k + sc['colo3'][lay] * abso3a[:] + tauray
        ssa[lay, :] = _ssa(tauray, tau[lay, :])
        if lay == laysolfr:
            sfluxzen[:] = sfluxref[:]

    # ── Upper atmosphere ──────────────────────────────────────────────────────
    for lay in range(nl, nlayers):
        tauray = sc['colmol'][lay] * rayl[:]
        tau[lay, :] = sc['colo3'][lay] * abso3b[:] + tauray
        ssa[lay, :] = _ssa(tauray, tau[lay, :])

    return tau, ssa, sfluxzen


# ── Band 10 (Fortran 26): 22650-29000 cm-1 ───────────────────────────────────
# Pure Rayleigh everywhere; no gas absorption
# LAYSOLFR = LAYTROP (i.e., last lower-atm layer sets sfluxzen)
# SSA = 1.0 everywhere
# sfluxref shape (16,)
# RAYL per g-point array

def taumol_b10(sc, c):
    """Band 10 (Fortran 26): 22650-29000 cm-1. Pure Rayleigh, SSA=1.0."""
    sfluxref = c['sfluxref']    # (16,)
    rayl     = c['rayl']        # (16,) per g-point

    nl      = sc['laytrop']
    nlayers = len(sc['jp'])
    tau     = np.zeros((nlayers, 16))
    ssa     = np.zeros((nlayers, 16))
    sfluxzen = np.zeros(16)

    # LAYSOLFR = LAYTROP → last lower-atm layer (0-based: nl-1)
    laysolfr = nl - 1 if nl > 0 else 0

    # ── Lower atmosphere ──────────────────────────────────────────────────────
    for lay in range(nl):
        tauray = sc['colmol'][lay] * rayl[:]
        tau[lay, :] = tauray
        ssa[lay, :] = 1.0
        if lay == laysolfr:
            sfluxzen[:] = sfluxref[:]

    # ── Upper atmosphere ──────────────────────────────────────────────────────
    for lay in range(nl, nlayers):
        tauray = sc['colmol'][lay] * rayl[:]
        tau[lay, :] = tauray
        ssa[lay, :] = 1.0

    return tau, ssa, sfluxzen


# ── Band 11 (Fortran 27): 29000-38000 cm-1 ───────────────────────────────────
# Low:  O3  NSPA=1
# High: O3  NSPB=1
# LAYREFFR=32 (upper loop check)
# SCALEKUR=50.15/48.37 (applied to sfluxref at LAYSOLFR)
# RAYL per g-point array
# sfluxref shape (16,)
#
# Note: LAYSOLFR set in upper-atm loop (unlike most bands that set it in lower loop)
# Fortran lower: no LAYSOLFR update, no sfluxzen set
# Fortran upper: IF (JP(LAY-1)<LAYREFFR .AND. JP(LAY)>=LAYREFFR) LAYSOLFR=LAY

def taumol_b11(sc, c):
    """Band 11 (Fortran 27): 29000-38000 cm-1. Low/High O3. SCALEKUR applied."""
    ka       = c['ka']          # (5,13,16)
    kb       = c['kb']          # (5,47,16)
    sfluxref = c['sfluxref']    # (16,)
    rayl     = c['rayl']        # (16,) per g-point

    LAYREFFR = 32
    SCALEKUR  = 50.15 / 48.37

    nl      = sc['laytrop']
    nlayers = len(sc['jp'])
    tau     = np.zeros((nlayers, 16))
    ssa     = np.zeros((nlayers, 16))
    sfluxzen = np.zeros(16)

    # ── Lower atmosphere: no LAYSOLFR update here ──────────────────────────
    for lay in range(nl):
        jp  = sc['jp'][lay];  jt = sc['jt'][lay];  jt1 = sc['jt1'][lay]
        f00 = sc['fac00'][lay];  f10 = sc['fac10'][lay]
        f01 = sc['fac01'][lay];  f11 = sc['fac11'][lay]

        taug_k = _simple_ka(sc['colo3'][lay], ka, jp, jt, jt1, f00, f10, f01, f11)
        tauray = sc['colmol'][lay] * rayl[:]

        tau[lay, :] = taug_k + tauray
        ssa[lay, :] = _ssa(tauray, tau[lay, :])

    # ── Upper atmosphere ──────────────────────────────────────────────────────
    laysolfr = nlayers - 1      # Fortran: LAYSOLFR=NLAYERS
    for lay in range(nl, nlayers):
        jp  = sc['jp'][lay];  jt = sc['jt'][lay];  jt1 = sc['jt1'][lay]
        f00 = sc['fac00'][lay];  f10 = sc['fac10'][lay]
        f01 = sc['fac01'][lay];  f11 = sc['fac11'][lay]

        # Fortran: IF (JP(LAY-1)<LAYREFFR .AND. JP(LAY)>=LAYREFFR) LAYSOLFR=LAY
        if lay >= nl and sc['jp'][lay-1] < LAYREFFR - 1 and jp >= LAYREFFR - 1:
            laysolfr = lay

        taug_k = _simple_kb(sc['colo3'][lay], kb, jp, jt, jt1, f00, f10, f01, f11)
        tauray = sc['colmol'][lay] * rayl[:]

        tau[lay, :] = taug_k + tauray
        ssa[lay, :] = _ssa(tauray, tau[lay, :])
        if lay == laysolfr:
            sfluxzen[:] = SCALEKUR * sfluxref[:]

    return tau, ssa, sfluxzen


# ── Band 12 (Fortran 28): 38000-50000 cm-1 ───────────────────────────────────
# Low:  O3 + O2  NSPA=9  SPECMULT=8  STRRAT=6.67029e-7
# High: O3 + O2  NSPB=5  SPECMULT=4
# LAYREFFR=58 (upper loop check)
# RAYL scalar=2.02e-5 (very large UV Rayleigh)
# sfluxref shape (5,16)
# Lower: clamp tau >= tauray (Fortran: IF TAUG-TAURAY<0 TAUG=TAURAY)
#
# Fortran lower:
#   TAUG = SPECCOMB*(FAC*ABSA) + TAURAY; clamp to >= TAURAY
# Fortran upper:
#   TAUG = SPECCOMB*(FAC*ABSB) + TAURAY

def taumol_b12(sc, c):
    """Band 12 (Fortran 28): 38000-50000 cm-1. Low/High O3+O2. Large Rayleigh."""
    ka       = c['ka']          # (9,5,13,16)
    kb       = c['kb']          # (5,5,47,16)
    sfluxref = c['sfluxref']    # (5,16)
    rayl     = float(c['rayl'])

    STRRAT   = 6.67029e-7
    LAYREFFR = 58

    nl      = sc['laytrop']
    nlayers = len(sc['jp'])
    tau     = np.zeros((nlayers, 16))
    ssa     = np.zeros((nlayers, 16))
    sfluxzen = np.zeros(16)

    # ── Lower atmosphere ──────────────────────────────────────────────────────
    for lay in range(nl):
        jp  = sc['jp'][lay];  jt = sc['jt'][lay];  jt1 = sc['jt1'][lay]
        f00 = sc['fac00'][lay];  f10 = sc['fac10'][lay]
        f01 = sc['fac01'][lay];  f11 = sc['fac11'][lay]

        speccomb, js, fs = _specparm_jsfs(
            sc['colo3'][lay], sc['colo2'][lay], STRRAT, 8.0)

        taug_k = _nspa9(speccomb, ka, js, jp, jt, jt1, f00, f10, f01, f11, fs)
        tauray = sc['colmol'][lay] * rayl

        # Fortran: TAUG=SPECCOMB*(FAC*ABSA)+TAURAY; then clamp TAUG>=TAURAY
        tau_total = taug_k + tauray
        tau_total = np.maximum(tau_total, tauray)   # enforce TAUG>=TAURAY

        tau[lay, :] = tau_total
        ssa[lay, :] = _ssa(tauray, tau[lay, :])

    # ── Upper atmosphere ──────────────────────────────────────────────────────
    laysolfr = nlayers - 1
    for lay in range(nl, nlayers):
        jp  = sc['jp'][lay];  jt = sc['jt'][lay];  jt1 = sc['jt1'][lay]
        f00 = sc['fac00'][lay];  f10 = sc['fac10'][lay]
        f01 = sc['fac01'][lay];  f11 = sc['fac11'][lay]
        jp_u = jp - 12

        if lay >= nl and sc['jp'][lay-1] < LAYREFFR - 1 and jp >= LAYREFFR - 1:
            laysolfr = lay

        speccomb, js, fs = _specparm_jsfs(
            sc['colo3'][lay], sc['colo2'][lay], STRRAT, 4.0)

        taug_k = _nspb5(speccomb, kb, js, jp_u, jt, jt1, f00, f10, f01, f11, fs)
        tauray = sc['colmol'][lay] * rayl

        tau[lay, :] = taug_k + tauray
        ssa[lay, :] = _ssa(tauray, tau[lay, :])
        if lay == laysolfr:
            sfluxzen[:] = (sfluxref[js, :] +
                           fs * (sfluxref[js+1, :] - sfluxref[js, :]))

    return tau, ssa, sfluxzen


# ── Band 13 (Fortran 29): 820-2600 cm-1 ──────────────────────────────────────
# Low:  H2O  NSPA=1  self+for + CO2 extra
# High: CO2  NSPB=1 + H2O extra
# LAYREFFR=49 (upper loop check)
# sfluxref shape (16,)
# RAYL scalar=9.30e-11
# Extra: ABSCO2(16) per g-point, ABSH2O(16) per g-point
#
# Fortran lower:
#   TAUG = COLH2O*((FAC*ABSA) + SELFFAC*SELF + FORFAC*FOR) + TAURAY + COLCO2*ABSCO2
# Fortran upper:
#   TAUG = COLCO2*(FAC*ABSB) + COLH2O*ABSH2O + TAURAY

def taumol_b13(sc, c):
    """Band 13 (Fortran 29): 820-2600 cm-1. Low H2O+CO2extra, High CO2+H2Oextra."""
    ka       = c['ka']          # (5,13,16)
    kb       = c['kb']          # (5,47,16)
    selfref  = c['selfref']     # (10,16)
    forref   = c['forref']      # (4,16)
    sfluxref = c['sfluxref']    # (16,)
    absco2   = c['absco2']      # (16,) per g-point
    absh2o   = c['absh2o']      # (16,) per g-point
    rayl     = float(c['rayl'])

    LAYREFFR = 49

    nl      = sc['laytrop']
    nlayers = len(sc['jp'])
    tau     = np.zeros((nlayers, 16))
    ssa     = np.zeros((nlayers, 16))
    sfluxzen = np.zeros(16)

    # ── Lower atmosphere ──────────────────────────────────────────────────────
    for lay in range(nl):
        jp  = sc['jp'][lay];  jt = sc['jt'][lay];  jt1 = sc['jt1'][lay]
        f00 = sc['fac00'][lay];  f10 = sc['fac10'][lay]
        f01 = sc['fac01'][lay];  f11 = sc['fac11'][lay]
        inds = sc['indself'][lay];  indf = sc['indfor'][lay]

        # NSPA=1 base without col multiply
        absa_base = (f00 * ka[jt,     jp,   :] +
                     f10 * ka[jt + 1, jp,   :] +
                     f01 * ka[jt1,    jp+1, :] +
                     f11 * ka[jt1+1,  jp+1, :])
        tau_self = (sc['selffac'][lay] *
                    (selfref[inds, :] + sc['selffrac'][lay] *
                     (selfref[inds+1, :] - selfref[inds, :])))
        tau_for  = (sc['forfac'][lay] *
                    (forref[indf, :] + sc['forfrac'][lay] *
                     (forref[indf+1, :] - forref[indf, :])))
        tauray = sc['colmol'][lay] * rayl

        # Fortran: COLH2O*((FAC*ABSA)+SELF+FOR) + TAURAY + COLCO2*ABSCO2
        tau[lay, :] = (sc['colh2o'][lay] * (absa_base + tau_self + tau_for) +
                       tauray + sc['colco2'][lay] * absco2[:])
        ssa[lay, :] = _ssa(tauray, tau[lay, :])

    # ── Upper atmosphere ──────────────────────────────────────────────────────
    laysolfr = nlayers - 1
    for lay in range(nl, nlayers):
        jp  = sc['jp'][lay];  jt = sc['jt'][lay];  jt1 = sc['jt1'][lay]
        f00 = sc['fac00'][lay];  f10 = sc['fac10'][lay]
        f01 = sc['fac01'][lay];  f11 = sc['fac11'][lay]

        if lay >= nl and sc['jp'][lay-1] < LAYREFFR - 1 and jp >= LAYREFFR - 1:
            laysolfr = lay

        taug_k = _simple_kb(sc['colco2'][lay], kb, jp, jt, jt1, f00, f10, f01, f11)
        tauray = sc['colmol'][lay] * rayl

        # Fortran: COLCO2*(FAC*ABSB) + COLH2O*ABSH2O + TAURAY
        tau[lay, :] = taug_k + sc['colh2o'][lay] * absh2o[:] + tauray
        ssa[lay, :] = _ssa(tauray, tau[lay, :])
        if lay == laysolfr:
            sfluxzen[:] = sfluxref[:]

    return tau, ssa, sfluxzen


# ── Dispatcher ────────────────────────────────────────────────────────────────

def taumol_all(sc, bands, adjflux):
    """
    Compute optical depths, single-scattering albedo, and solar fluxes
    for all 14 SW bands.

    Parameters
    ----------
    sc      : dict from setcoef()
    bands   : list of 14 per-band k-coeff dicts (from load_coeffs.load())
    adjflux : (14,) Earth-Sun distance correction factor per band

    Returns
    -------
    tau     : (14, nlayers, 16) total optical depth (gas + Rayleigh)
    ssa     : (14, nlayers, 16) single-scattering albedo = Rayleigh / tau
    sfluxzen: (14, 16) solar flux per g-point at reference level, adjusted
              by adjflux (Earth-Sun distance correction)
    """
    nlayers  = len(sc['jp'])
    tau      = np.zeros((14, nlayers, 16))
    ssa      = np.zeros((14, nlayers, 16))
    sfluxzen = np.zeros((14, 16))

    tau[0],  ssa[0],  sfluxzen[0]  = taumol_b00(sc, bands[0])   # Fortran 16
    tau[1],  ssa[1],  sfluxzen[1]  = taumol_b01(sc, bands[1])   # Fortran 17
    tau[2],  ssa[2],  sfluxzen[2]  = taumol_b02(sc, bands[2])   # Fortran 18
    tau[3],  ssa[3],  sfluxzen[3]  = taumol_b03(sc, bands[3])   # Fortran 19
    tau[4],  ssa[4],  sfluxzen[4]  = taumol_b04(sc, bands[4])   # Fortran 20
    tau[5],  ssa[5],  sfluxzen[5]  = taumol_b05(sc, bands[5])   # Fortran 21
    tau[6],  ssa[6],  sfluxzen[6]  = taumol_b06(sc, bands[6])   # Fortran 22
    tau[7],  ssa[7],  sfluxzen[7]  = taumol_b07(sc, bands[7])   # Fortran 23
    tau[8],  ssa[8],  sfluxzen[8]  = taumol_b08(sc, bands[8])   # Fortran 24
    tau[9],  ssa[9],  sfluxzen[9]  = taumol_b09(sc, bands[9])   # Fortran 25
    tau[10], ssa[10], sfluxzen[10] = taumol_b10(sc, bands[10])  # Fortran 26
    tau[11], ssa[11], sfluxzen[11] = taumol_b11(sc, bands[11])  # Fortran 27
    tau[12], ssa[12], sfluxzen[12] = taumol_b12(sc, bands[12])  # Fortran 28
    tau[13], ssa[13], sfluxzen[13] = taumol_b13(sc, bands[13])  # Fortran 29

    # Apply Earth-Sun distance correction to solar flux
    for b in range(14):
        sfluxzen[b] *= adjflux[b]

    return tau, ssa, sfluxzen
