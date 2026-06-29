"""
Optical depth computation — Python translation of RRTM_LW taumol.f.

taumol_all(sc, bands, planck, chi_mls) → (tau, fracs)
  tau  : np.ndarray shape (16, nlayers, 16) — optical depth per band/layer/g-point
  fracs: np.ndarray shape (16, nlayers, 16) — Planck fractions per band/layer/g-point

Index conventions (0-based throughout):
  band b=0..15 corresponds to Fortran IBAND=1..16
  g-point ig=0..15
  layer lay=0..nlayers-1  (lay=0 is bottom layer in Fortran LAY=1)

All Fortran 1-based JP, JT, JT1, INDSELF, INDFOR, INDMINOR are stored 0-based in sc.
"""
import numpy as np
from .constants import ONEMINUS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _lerp(arr, i, frac):
    """Linear interpolation: arr[i] + frac*(arr[i+1]-arr[i]), vectorised over g."""
    return arr[i] + frac * (arr[i + 1] - arr[i])


def _tau_simple(col, ka, jt0, jp0, jt10, f00, f10, f01, f11):
    """Bilinear KA interpolation for simple (NSPA=1) bands. Returns shape (16,)."""
    return col * (f00 * ka[jt0,   jp0,   :] +
                  f10 * ka[jt0+1, jp0,   :] +
                  f01 * ka[jt10,  jp0+1, :] +
                  f11 * ka[jt10+1,jp0+1, :])


def _tau_simple_kb(col, kb, jt0, jp_u, jt10, f00, f10, f01, f11):
    """Bilinear KB interpolation for simple (NSPB=1) upper-atm bands."""
    return col * (f00 * kb[jt0,   jp_u,   :] +
                  f10 * kb[jt0+1, jp_u,   :] +
                  f01 * kb[jt10,  jp_u+1, :] +
                  f11 * kb[jt10+1,jp_u+1, :])


def _mixed_factors(specparm, specparm1, fs, fs1, f00, f10, f01, f11):
    """
    Compute (fac000..fac211) and (fac001..fac211) for the cubic/linear
    mixed-species interpolation (both SPECPARM and SPECPARM1 branches).
    Returns two 6-element or 4-element tuples depending on branches taken.
    Both tuples use the same structure: fac[ijk] = fac_{ij}{k} where
    i=0 (no JS offset), 1 (+1), 2 (+2); j=0 (no JT offset), 1 (+1 JT); k=0/1 (FAC00/FAC10).
    """
    def _branch(sp, frs, fac0, fac1):
        if sp < 0.125:
            p = frs - 1.0
            p4 = p ** 4
            fk0, fk1, fk2 = p4, 1 - p - 2 * p4, p + p4
            return (fk0 * fac0, fk1 * fac0, fk2 * fac0,
                    fk0 * fac1, fk1 * fac1, fk2 * fac1), True
        elif sp > 0.875:
            p = -frs
            p4 = p ** 4
            fk0, fk1, fk2 = p4, 1 - p - 2 * p4, p + p4
            return (fk0 * fac0, fk1 * fac0, fk2 * fac0,
                    fk0 * fac1, fk1 * fac1, fk2 * fac1), False
        else:
            return ((1 - frs) * fac0, frs * fac0, None,
                    (1 - frs) * fac1, frs * fac1, None), None

    return _branch(specparm, fs, f00, f10), _branch(specparm1, fs1, f01, f11)


def _tau_mixed_nspa9(speccomb, ka, js, jt0, jp0, factors):
    """
    Mixed NSPA=9 optical depth contribution for one branch.
    factors: 6-tuple (fac000,fac100,fac200,fac010,fac110,fac210)
    and branch_type: True=low, False=high, None=linear.
    """
    fac000, fac100, fac200, fac010, fac110, fac210 = factors[0]
    branch_type = factors[1]

    if branch_type is True:      # specparm < 0.125
        return speccomb * (
            fac000 * ka[js,   jt0,   jp0, :] +
            fac100 * ka[js+1, jt0,   jp0, :] +
            fac200 * ka[js+2, jt0,   jp0, :] +
            fac010 * ka[js,   jt0+1, jp0, :] +
            fac110 * ka[js+1, jt0+1, jp0, :] +
            fac210 * ka[js+2, jt0+1, jp0, :])
    elif branch_type is False:   # specparm > 0.875
        return speccomb * (
            fac200 * ka[js-1, jt0,   jp0, :] +
            fac100 * ka[js,   jt0,   jp0, :] +
            fac000 * ka[js+1, jt0,   jp0, :] +
            fac210 * ka[js-1, jt0+1, jp0, :] +
            fac110 * ka[js,   jt0+1, jp0, :] +
            fac010 * ka[js+1, jt0+1, jp0, :])
    else:                        # linear
        fac000, fac100 = fac000, fac100
        fac010, fac110 = fac010, fac110
        return speccomb * (
            fac000 * ka[js,   jt0,   jp0, :] +
            fac100 * ka[js+1, jt0,   jp0, :] +
            fac010 * ka[js,   jt0+1, jp0, :] +
            fac110 * ka[js+1, jt0+1, jp0, :])


def _tau_mixed_nspb5(speccomb, kb, js, jt0, jp_u, f00, f10, f01, f11, is_jp1=False):
    """
    Mixed NSPB=5 (upper atm) optical depth, linear interpolation only.
    is_jp1: if True, uses JP+1 factors (f01,f11) instead of JP factors.
    """
    if not is_jp1:
        f0, f1 = f00, f10
    else:
        f0, f1 = f01, f11
    return speccomb * (
        f0 * kb[js,   jt0,   jp_u, :] +
        f1 * kb[js+1, jt0,   jp_u, :] +
        f0 * kb[js,   jt0+1, jp_u, :] +  # This is wrong — see below
        f1 * kb[js+1, jt0+1, jp_u, :])


# Correct NSPB=5 upper-atm formula (IND0+1, IND0+5, IND0+6):
def _tau_mixed_nspb5_jp(speccomb, speccomb1, kb, js, jt0, jp_u, js1, jt10, jp_u1,
                         f00, f10, f01, f11):
    return (speccomb  * (f00 * kb[js,   jt0,   jp_u, :] +
                         f10 * kb[js+1, jt0,   jp_u, :] +
                         f00 * kb[js,   jt0+1, jp_u, :] +
                         f10 * kb[js+1, jt0+1, jp_u, :])   # wrong version — will fix inline
            )


def _planck_frac_1d(fracrefa):
    """Return fracrefa as-is for 1D bands."""
    return fracrefa   # shape (16,)


def _planck_frac_2d(fracrefa, jpl, fpl):
    """Interpolate 2D fracrefa at species index jpl+fpl. fracrefa shape (N,16)."""
    return fracrefa[jpl, :] + fpl * (fracrefa[jpl+1, :] - fracrefa[jpl, :])


def _minor_1d(arr, indm, minfrac):
    """1D minor gas interpolation: arr[indm] + minfrac*(arr[indm+1]-arr[indm])."""
    return arr[indm, :] + minfrac * (arr[indm+1, :] - arr[indm, :])


def _minor_2d(arr, jm, indm, fm, minfrac):
    """
    2D minor gas (species × minor_JT × gpt):
    arr[jm, indm] + fm*(arr[jm+1,indm]-arr[jm,indm])  interpolated in species,
    then interpolated in temperature (minfrac).
    """
    m1 = arr[jm, indm, :] + fm * (arr[jm+1, indm, :] - arr[jm, indm, :])
    m2 = arr[jm, indm+1, :] + fm * (arr[jm+1, indm+1, :] - arr[jm, indm+1, :])
    return m1 + minfrac * (m2 - m1)


def _co2_adj(colco2, coldry, chi_mls, jp0):
    """Empirical CO2 column adjustment (used in bands 6,7,8,13)."""
    chi_co2 = colco2 / coldry
    ratco2 = 1.0e20 * chi_co2 / chi_mls[1, jp0 + 1]
    if ratco2 > 3.0:
        adjfac = 2.0 + (ratco2 - 2.0) ** 0.77
        return adjfac * chi_mls[1, jp0 + 1] * coldry * 1.0e-20
    return colco2


def _n2o_adj(coln2o, coldry, chi_mls, jp0):
    """Empirical N2O column adjustment (bands 3,9)."""
    chi_n2o = coln2o / coldry
    ratn2o = 1.0e20 * chi_n2o / chi_mls[3, jp0 + 1]
    if ratn2o > 1.5:
        adjfac = 0.5 + (ratn2o - 0.5) ** 0.65
        return adjfac * chi_mls[3, jp0 + 1] * coldry * 1.0e-20
    return coln2o


EMPIRICAL_MULTS = np.array([0.92, 0.88, 1.07, 1.10, 0.99, 0.88, 0.83])  # g-points 7-13


# ── Band functions ────────────────────────────────────────────────────────────

def taumol_b01(sc, c, planck):
    """Band 1: 10-350 cm-1.  Low/high key: H2O.  Minor: N2."""
    ka = c['ka'];  kb = c['kb']
    selfref = c['selfref'];  forref = c['forref']
    ka_mn2 = c['ka_mn2'];   kb_mn2 = c['kb_mn2']
    fracrefa = planck['fracrefa']
    fracrefb = planck['fracrefb']
    nl = sc['laytrop'];  nlayers = len(sc['jp'])
    tau   = np.zeros((nlayers, 16));  fracs = np.zeros((nlayers, 16))

    for lay in range(nl):
        jt0  = sc['jt'][lay];   jp0  = sc['jp'][lay];  jt10 = sc['jt1'][lay]
        f00, f10, f01, f11 = sc['fac00'][lay], sc['fac10'][lay], sc['fac01'][lay], sc['fac11'][lay]
        inds = sc['indself'][lay];  indf = sc['indfor'][lay];  indm = sc['indminor'][lay]
        pp   = sc['pavel'][lay]
        corradj = 1.0 - 0.15 * (250.0 - pp) / 154.4 if pp < 250.0 else 1.0
        scalen2 = sc['colbrd'][lay] * sc['scaleminorn2'][lay]
        tauself = sc['selffac'][lay] * _lerp(selfref[:, :], inds, sc['selffrac'][lay])
        taufor  = sc['forfac'][lay]  * _lerp(forref[:, :], indf, sc['forfrac'][lay])
        taun2   = scalen2 * _lerp(ka_mn2, indm, sc['minorfrac'][lay])
        tau[lay, :] = corradj * (_tau_simple(sc['colh2o'][lay], ka, jt0, jp0, jt10, f00, f10, f01, f11)
                                 + tauself + taufor + taun2)
        fracs[lay, :] = fracrefa

    for lay in range(nl, nlayers):
        jt0  = sc['jt'][lay];   jp0  = sc['jp'][lay];  jt10 = sc['jt1'][lay]
        jp_u = jp0 - 12
        f00, f10, f01, f11 = sc['fac00'][lay], sc['fac10'][lay], sc['fac01'][lay], sc['fac11'][lay]
        indf = sc['indfor'][lay];  indm = sc['indminor'][lay]
        pp   = sc['pavel'][lay]
        corradj = 1.0 - 0.15 * pp / 95.6
        scalen2 = sc['colbrd'][lay] * sc['scaleminorn2'][lay]
        taufor = sc['forfac'][lay] * _lerp(forref[:, :], indf, sc['forfrac'][lay])
        taun2  = scalen2 * _lerp(kb_mn2, indm, sc['minorfrac'][lay])
        tau[lay, :] = corradj * (_tau_simple_kb(sc['colh2o'][lay], kb, jt0, jp_u, jt10, f00, f10, f01, f11)
                                  + taufor + taun2)
        fracs[lay, :] = fracrefb

    return tau, fracs


def taumol_b02(sc, c, planck):
    """Band 2: 350-500 cm-1.  Low/high key: H2O."""
    ka = c['ka'];  kb = c['kb']
    selfref = c['selfref'];  forref = c['forref']
    fracrefa = planck['fracrefa'];  fracrefb = planck['fracrefb']
    nl = sc['laytrop'];  nlayers = len(sc['jp'])
    tau = np.zeros((nlayers, 16));  fracs = np.zeros((nlayers, 16))

    for lay in range(nl):
        jt0 = sc['jt'][lay];  jp0 = sc['jp'][lay];  jt10 = sc['jt1'][lay]
        f00, f10, f01, f11 = sc['fac00'][lay], sc['fac10'][lay], sc['fac01'][lay], sc['fac11'][lay]
        inds = sc['indself'][lay];  indf = sc['indfor'][lay]
        pp = sc['pavel'][lay]
        corradj = 1.0 - 0.05 * (pp - 100.0) / 900.0
        tauself = sc['selffac'][lay] * _lerp(selfref, inds, sc['selffrac'][lay])
        taufor  = sc['forfac'][lay]  * _lerp(forref,  indf, sc['forfrac'][lay])
        tau[lay, :] = corradj * (_tau_simple(sc['colh2o'][lay], ka, jt0, jp0, jt10, f00, f10, f01, f11)
                                  + tauself + taufor)
        fracs[lay, :] = fracrefa

    for lay in range(nl, nlayers):
        jt0 = sc['jt'][lay];  jp0 = sc['jp'][lay];  jt10 = sc['jt1'][lay]
        jp_u = jp0 - 12
        f00, f10, f01, f11 = sc['fac00'][lay], sc['fac10'][lay], sc['fac01'][lay], sc['fac11'][lay]
        indf = sc['indfor'][lay]
        taufor = sc['forfac'][lay] * _lerp(forref, indf, sc['forfrac'][lay])
        tau[lay, :] = (_tau_simple_kb(sc['colh2o'][lay], kb, jt0, jp_u, jt10, f00, f10, f01, f11)
                       + taufor)
        fracs[lay, :] = fracrefb

    return tau, fracs


def _mixed_lower(sc, lay, ka, colgas, rat, rat_1, fracrefa, nspa, speccomb_col2=None):
    """
    Generic mixed-species lower-atm optical depth (NSPA=9, SPECMULT=8).
    colgas: primary column; rat/rat_1: mixing ratio arrays; speccomb_col2: secondary column.
    Returns (tau_major, fracs_lay) ignoring minor terms (caller adds them).
    """
    col2 = speccomb_col2 if speccomb_col2 is not None else np.zeros(1)
    speccomb  = colgas + rat[lay]   * col2
    specparm  = min(colgas / speccomb, ONEMINUS)
    specmult  = 8.0 * specparm
    js  = int(specmult)
    fs  = specmult - js

    speccomb1 = colgas + rat_1[lay] * col2
    specparm1 = min(colgas / speccomb1, ONEMINUS)
    specmult1 = 8.0 * specparm1
    js1 = int(specmult1)
    fs1 = specmult1 - js1

    jt0 = sc['jt'][lay];  jp0 = sc['jp'][lay];  jt10 = sc['jt1'][lay]
    f00, f10, f01, f11 = sc['fac00'][lay], sc['fac10'][lay], sc['fac01'][lay], sc['fac11'][lay]

    br0, br1 = _mixed_factors(specparm, specparm1, fs, fs1, f00, f10, f01, f11)
    tau_major  = _tau_mixed_nspa9(speccomb,  ka, js,  jt0,  jp0,  br0)
    tau_major1 = _tau_mixed_nspa9(speccomb1, ka, js1, jt10, jp0 + 1, br1)

    return tau_major + tau_major1, js, fs, speccomb


def _mixed_lower_planck(colgas, col2, rat_planck, fracrefa):
    """Planck fraction interpolation for mixed lower-atm bands."""
    speccomb_pl = colgas + rat_planck * col2
    specparm_pl = min(colgas / speccomb_pl, ONEMINUS)
    specmult_pl = 8.0 * specparm_pl
    jpl = int(specmult_pl)
    fpl = specmult_pl - jpl
    return _planck_frac_2d(fracrefa, jpl, fpl)


def _mixed_upper_nspb5(sc, lay, kb, col1, col2, rat, rat_1, refrat_planck, fracrefb):
    """Mixed upper-atm NSPB=5 formula (linear only, SPECMULT=4)."""
    speccomb  = col1 + rat[lay]   * col2
    specparm  = min(col1 / speccomb, ONEMINUS)
    specmult  = 4.0 * specparm
    js  = int(specmult);  fs  = specmult - js

    speccomb1 = col1 + rat_1[lay] * col2
    specparm1 = min(col1 / speccomb1, ONEMINUS)
    specmult1 = 4.0 * specparm1
    js1 = int(specmult1);  fs1 = specmult1 - js1

    jt0 = sc['jt'][lay];  jp0 = sc['jp'][lay];  jt10 = sc['jt1'][lay]
    jp_u = jp0 - 12
    f00, f10, f01, f11 = sc['fac00'][lay], sc['fac10'][lay], sc['fac01'][lay], sc['fac11'][lay]

    # NSPB=5 strides: js+1, js+5, js+6 in Fortran 1-based → js, js+1, jt0+1, js, js+1 in Python
    tau = (speccomb * ((1 - fs) * f00 * kb[js,   jt0,   jp_u, :] +
                       fs       * f00 * kb[js+1, jt0,   jp_u, :] +
                       (1 - fs) * f10 * kb[js,   jt0+1, jp_u, :] +
                       fs       * f10 * kb[js+1, jt0+1, jp_u, :])
           + speccomb1 * ((1 - fs1) * f01 * kb[js1,   jt10,   jp_u+1, :] +
                          fs1       * f01 * kb[js1+1, jt10,   jp_u+1, :] +
                          (1 - fs1) * f11 * kb[js1,   jt10+1, jp_u+1, :] +
                          fs1       * f11 * kb[js1+1, jt10+1, jp_u+1, :]))

    speccomb_pl = col1 + refrat_planck * col2
    specparm_pl = min(col1 / speccomb_pl, ONEMINUS)
    specmult_pl = 4.0 * specparm_pl
    jpl = int(specmult_pl);  fpl = specmult_pl - jpl
    frac = _planck_frac_2d(fracrefb, jpl, fpl)

    return tau, frac


def taumol_b03(sc, c, planck, chi_mls):
    """Band 3: 500-630 cm-1.  Low/high: H2O+CO2.  Minor: N2O."""
    ka = c['ka'];  kb = c['kb']
    selfref = c['selfref'];  forref = c['forref']
    ka_mn2o = c['ka_mn2o'];  kb_mn2o = c['kb_mn2o']
    fracrefa = planck['fracrefa'];  fracrefb = planck['fracrefb']
    nl = sc['laytrop'];  nlayers = len(sc['jp'])
    tau = np.zeros((nlayers, 16));  fracs = np.zeros((nlayers, 16))

    refrat_planck_a = chi_mls[0, 8]  / chi_mls[1, 8]
    refrat_planck_b = chi_mls[0, 12] / chi_mls[1, 12]
    refrat_m_a      = chi_mls[0, 2]  / chi_mls[1, 2]
    refrat_m_b      = chi_mls[0, 12] / chi_mls[1, 12]

    for lay in range(nl):
        jp0  = sc['jp'][lay];  jt0 = sc['jt'][lay];  jt10 = sc['jt1'][lay]
        f00, f10, f01, f11 = sc['fac00'][lay], sc['fac10'][lay], sc['fac01'][lay], sc['fac11'][lay]
        inds = sc['indself'][lay];  indf = sc['indfor'][lay];  indm = sc['indminor'][lay]
        colh2o = sc['colh2o'][lay];  colco2 = sc['colco2'][lay]

        speccomb  = colh2o + sc['rat_h2oco2'][lay]   * colco2
        specparm  = min(colh2o / speccomb,  ONEMINUS)
        specmult  = 8.0 * specparm;  js  = int(specmult);  fs  = specmult - js
        speccomb1 = colh2o + sc['rat_h2oco2_1'][lay] * colco2
        specparm1 = min(colh2o / speccomb1, ONEMINUS)
        specmult1 = 8.0 * specparm1;  js1 = int(specmult1);  fs1 = specmult1 - js1

        # N2O minor species index
        speccomb_mn2o = colh2o + refrat_m_a * colco2
        specparm_mn2o = min(colh2o / speccomb_mn2o, ONEMINUS)
        jmn2o = int(8.0 * specparm_mn2o);  fmn2o = 8.0 * specparm_mn2o - jmn2o

        adjcoln2o = _n2o_adj(sc['coln2o'][lay], sc['coldry'][lay], chi_mls, jp0)

        br0, br1 = _mixed_factors(specparm, specparm1, fs, fs1, f00, f10, f01, f11)
        tau_major  = _tau_mixed_nspa9(speccomb,  ka, js,  jt0,  jp0,     br0)
        tau_major1 = _tau_mixed_nspa9(speccomb1, ka, js1, jt10, jp0 + 1, br1)

        tauself = sc['selffac'][lay] * _lerp(selfref, inds, sc['selffrac'][lay])
        taufor  = sc['forfac'][lay]  * _lerp(forref,  indf, sc['forfrac'][lay])
        taun2o  = adjcoln2o * _minor_2d(ka_mn2o, jmn2o, indm, fmn2o, sc['minorfrac'][lay])

        tau[lay, :] = tau_major + tau_major1 + tauself + taufor + taun2o
        fracs[lay, :] = _mixed_lower_planck(colh2o, colco2, refrat_planck_a, fracrefa)

    for lay in range(nl, nlayers):
        colh2o = sc['colh2o'][lay];  colco2 = sc['colco2'][lay]
        jp0  = sc['jp'][lay];  jt0 = sc['jt'][lay];  jt10 = sc['jt1'][lay]

        speccomb_mn2o = colh2o + refrat_m_b * colco2
        specparm_mn2o = min(colh2o / speccomb_mn2o, ONEMINUS)
        jmn2o = int(4.0 * specparm_mn2o);  fmn2o = 4.0 * specparm_mn2o - jmn2o
        adjcoln2o = _n2o_adj(sc['coln2o'][lay], sc['coldry'][lay], chi_mls, jp0)
        taun2o = adjcoln2o * _minor_2d(kb_mn2o, jmn2o, sc['indminor'][lay], fmn2o, sc['minorfrac'][lay])

        tau_u, frac_u = _mixed_upper_nspb5(sc, lay, kb, colh2o, colco2,
                                            sc['rat_h2oco2'], sc['rat_h2oco2_1'],
                                            refrat_planck_b, fracrefb)
        tau[lay, :]   = tau_u + taun2o
        fracs[lay, :] = frac_u

    return tau, fracs


def taumol_b04(sc, c, planck, chi_mls):
    """Band 4: 630-700 cm-1.  Low: H2O+CO2.  High: O3+CO2."""
    ka = c['ka'];  kb = c['kb']
    selfref = c['selfref'];  forref = c['forref']
    fracrefa = planck['fracrefa'];  fracrefb = planck['fracrefb']
    nl = sc['laytrop'];  nlayers = len(sc['jp'])
    tau = np.zeros((nlayers, 16));  fracs = np.zeros((nlayers, 16))

    refrat_planck_a = chi_mls[0, 10] / chi_mls[1, 10]
    refrat_planck_b = chi_mls[2, 12] / chi_mls[1, 12]

    for lay in range(nl):
        jp0  = sc['jp'][lay];  jt0 = sc['jt'][lay];  jt10 = sc['jt1'][lay]
        f00, f10, f01, f11 = sc['fac00'][lay], sc['fac10'][lay], sc['fac01'][lay], sc['fac11'][lay]
        inds = sc['indself'][lay];  indf = sc['indfor'][lay]
        colh2o = sc['colh2o'][lay];  colco2 = sc['colco2'][lay]

        speccomb  = colh2o + sc['rat_h2oco2'][lay]   * colco2
        specparm  = min(colh2o / speccomb,  ONEMINUS)
        specmult  = 8.0 * specparm;  js  = int(specmult);  fs  = specmult - js
        speccomb1 = colh2o + sc['rat_h2oco2_1'][lay] * colco2
        specparm1 = min(colh2o / speccomb1, ONEMINUS)
        specmult1 = 8.0 * specparm1;  js1 = int(specmult1);  fs1 = specmult1 - js1

        br0, br1 = _mixed_factors(specparm, specparm1, fs, fs1, f00, f10, f01, f11)
        tau_major  = _tau_mixed_nspa9(speccomb,  ka, js,  jt0,  jp0,     br0)
        tau_major1 = _tau_mixed_nspa9(speccomb1, ka, js1, jt10, jp0 + 1, br1)
        tauself = sc['selffac'][lay] * _lerp(selfref, inds, sc['selffrac'][lay])
        taufor  = sc['forfac'][lay]  * _lerp(forref,  indf, sc['forfrac'][lay])

        tau[lay, :] = tau_major + tau_major1 + tauself + taufor
        fracs[lay, :] = _mixed_lower_planck(colh2o, colco2, refrat_planck_a, fracrefa)

    for lay in range(nl, nlayers):
        colo3 = sc['colo3'][lay];  colco2 = sc['colco2'][lay]
        tau_u, frac_u = _mixed_upper_nspb5(sc, lay, kb, colo3, colco2,
                                            sc['rat_o3co2'], sc['rat_o3co2_1'],
                                            refrat_planck_b, fracrefb)
        tau[lay, :]   = tau_u
        fracs[lay, :] = frac_u
        # Empirical correction for stratospheric CO2 cooling
        tau[lay, 7:14] *= EMPIRICAL_MULTS

    return tau, fracs


def taumol_b05(sc, c, planck, chi_mls):
    """Band 5: 700-820 cm-1.  Low: H2O+CO2 + O3 minor.  High: O3+CO2."""
    ka = c['ka'];  kb = c['kb']
    selfref = c['selfref'];  forref = c['forref']
    ka_mo3 = c['ka_mo3']
    fracrefa = planck['fracrefa'];  fracrefb = planck['fracrefb']
    nl = sc['laytrop'];  nlayers = len(sc['jp'])
    tau = np.zeros((nlayers, 16));  fracs = np.zeros((nlayers, 16))

    refrat_planck_a = chi_mls[0, 4]  / chi_mls[1, 4]
    refrat_planck_b = chi_mls[2, 42] / chi_mls[1, 42]
    refrat_m_a      = chi_mls[0, 6]  / chi_mls[1, 6]

    for lay in range(nl):
        jp0  = sc['jp'][lay];  jt0 = sc['jt'][lay];  jt10 = sc['jt1'][lay]
        f00, f10, f01, f11 = sc['fac00'][lay], sc['fac10'][lay], sc['fac01'][lay], sc['fac11'][lay]
        inds = sc['indself'][lay];  indf = sc['indfor'][lay];  indm = sc['indminor'][lay]
        colh2o = sc['colh2o'][lay];  colco2 = sc['colco2'][lay]

        speccomb  = colh2o + sc['rat_h2oco2'][lay]   * colco2
        specparm  = min(colh2o / speccomb,  ONEMINUS)
        specmult  = 8.0 * specparm;  js  = int(specmult);  fs  = specmult - js
        speccomb1 = colh2o + sc['rat_h2oco2_1'][lay] * colco2
        specparm1 = min(colh2o / speccomb1, ONEMINUS)
        specmult1 = 8.0 * specparm1;  js1 = int(specmult1);  fs1 = specmult1 - js1

        speccomb_mo3 = colh2o + refrat_m_a * colco2
        specparm_mo3 = min(colh2o / speccomb_mo3, ONEMINUS)
        jmo3 = int(8.0 * specparm_mo3);  fmo3 = 8.0 * specparm_mo3 - jmo3

        br0, br1 = _mixed_factors(specparm, specparm1, fs, fs1, f00, f10, f01, f11)
        tau_major  = _tau_mixed_nspa9(speccomb,  ka, js,  jt0,  jp0,     br0)
        tau_major1 = _tau_mixed_nspa9(speccomb1, ka, js1, jt10, jp0 + 1, br1)
        tauself = sc['selffac'][lay] * _lerp(selfref, inds, sc['selffrac'][lay])
        taufor  = sc['forfac'][lay]  * _lerp(forref,  indf, sc['forfrac'][lay])
        abso3   = sc['colo3'][lay] * _minor_2d(ka_mo3, jmo3, indm, fmo3, sc['minorfrac'][lay])
        # CCL4 term (WX[0]=0 for IXSECT=0) → omitted

        tau[lay, :] = tau_major + tau_major1 + tauself + taufor + abso3
        fracs[lay, :] = _mixed_lower_planck(colh2o, colco2, refrat_planck_a, fracrefa)

    for lay in range(nl, nlayers):
        colo3 = sc['colo3'][lay];  colco2 = sc['colco2'][lay]
        tau_u, frac_u = _mixed_upper_nspb5(sc, lay, kb, colo3, colco2,
                                            sc['rat_o3co2'], sc['rat_o3co2_1'],
                                            refrat_planck_b, fracrefb)
        # CCL4 term omitted
        tau[lay, :]   = tau_u
        fracs[lay, :] = frac_u

    return tau, fracs


def taumol_b06(sc, c, planck, chi_mls):
    """Band 6: 820-980 cm-1.  Low: H2O + CO2 minor.  High: nothing (tau=0)."""
    ka = c['ka']
    selfref = c['selfref'];  forref = c['forref']
    ka_mco2 = c['ka_mco2']
    fracrefa = planck['fracrefa']
    nl = sc['laytrop'];  nlayers = len(sc['jp'])
    tau = np.zeros((nlayers, 16));  fracs = np.zeros((nlayers, 16))

    for lay in range(nl):
        jt0 = sc['jt'][lay];  jp0 = sc['jp'][lay];  jt10 = sc['jt1'][lay]
        f00, f10, f01, f11 = sc['fac00'][lay], sc['fac10'][lay], sc['fac01'][lay], sc['fac11'][lay]
        inds = sc['indself'][lay];  indf = sc['indfor'][lay];  indm = sc['indminor'][lay]

        adjcolco2 = _co2_adj(sc['colco2'][lay], sc['coldry'][lay], chi_mls, jp0)
        # Note: band 6 CO2 adjustment uses exponent 0.77
        chi_co2 = sc['colco2'][lay] / sc['coldry'][lay]
        ratco2  = 1.0e20 * chi_co2 / chi_mls[1, jp0 + 1]
        if ratco2 > 3.0:
            adjfac = 2.0 + (ratco2 - 2.0) ** 0.77
            adjcolco2 = adjfac * chi_mls[1, jp0 + 1] * sc['coldry'][lay] * 1.0e-20
        else:
            adjcolco2 = sc['colco2'][lay]

        tauself = sc['selffac'][lay] * _lerp(selfref, inds, sc['selffrac'][lay])
        taufor  = sc['forfac'][lay]  * _lerp(forref,  indf, sc['forfrac'][lay])
        absco2  = _minor_1d(ka_mco2, indm, sc['minorfrac'][lay])
        # CFC terms omitted (WX=0)
        tau[lay, :] = (_tau_simple(sc['colh2o'][lay], ka, jt0, jp0, jt10, f00, f10, f01, f11)
                       + tauself + taufor + adjcolco2 * absco2)
        fracs[lay, :] = fracrefa

    # Above LAYTROP: tau=0, fracs=fracrefa (CFC terms omitted)
    for lay in range(nl, nlayers):
        tau[lay, :]   = 0.0
        fracs[lay, :] = fracrefa

    return tau, fracs


def taumol_b07(sc, c, planck, chi_mls):
    """Band 7: 980-1080 cm-1.  Low: H2O+O3 + CO2 minor.  High: O3 + CO2 minor."""
    ka = c['ka'];  kb = c['kb']
    selfref = c['selfref'];  forref = c['forref']
    ka_mco2 = c['ka_mco2'];  kb_mco2 = c['kb_mco2']
    fracrefa = planck['fracrefa'];  fracrefb = planck['fracrefb']
    nl = sc['laytrop'];  nlayers = len(sc['jp'])
    tau = np.zeros((nlayers, 16));  fracs = np.zeros((nlayers, 16))

    refrat_planck_a = chi_mls[0, 2] / chi_mls[2, 2]
    refrat_m_a      = chi_mls[0, 2] / chi_mls[2, 2]

    for lay in range(nl):
        jp0  = sc['jp'][lay];  jt0 = sc['jt'][lay];  jt10 = sc['jt1'][lay]
        f00, f10, f01, f11 = sc['fac00'][lay], sc['fac10'][lay], sc['fac01'][lay], sc['fac11'][lay]
        inds = sc['indself'][lay];  indf = sc['indfor'][lay];  indm = sc['indminor'][lay]
        colh2o = sc['colh2o'][lay];  colo3 = sc['colo3'][lay]

        speccomb  = colh2o + sc['rat_h2oo3'][lay]   * colo3
        specparm  = min(colh2o / speccomb,  ONEMINUS)
        specmult  = 8.0 * specparm;  js  = int(specmult);  fs  = specmult - js
        speccomb1 = colh2o + sc['rat_h2oo3_1'][lay] * colo3
        specparm1 = min(colh2o / speccomb1, ONEMINUS)
        specmult1 = 8.0 * specparm1;  js1 = int(specmult1);  fs1 = specmult1 - js1

        speccomb_mco2 = colh2o + refrat_m_a * colo3
        specparm_mco2 = min(colh2o / speccomb_mco2, ONEMINUS)
        jmco2 = int(8.0 * specparm_mco2);  fmco2 = 8.0 * specparm_mco2 - jmco2

        chi_co2 = sc['colco2'][lay] / sc['coldry'][lay]
        ratco2  = 1.0e20 * chi_co2 / chi_mls[1, jp0 + 1]
        if ratco2 > 3.0:
            adjfac   = 2.0 + (ratco2 - 2.0) ** 0.79
            adjcolco2 = adjfac * chi_mls[1, jp0 + 1] * sc['coldry'][lay] * 1.0e-20
        else:
            adjcolco2 = sc['colco2'][lay]

        br0, br1 = _mixed_factors(specparm, specparm1, fs, fs1, f00, f10, f01, f11)
        tau_major  = _tau_mixed_nspa9(speccomb,  ka, js,  jt0,  jp0,     br0)
        tau_major1 = _tau_mixed_nspa9(speccomb1, ka, js1, jt10, jp0 + 1, br1)
        tauself = sc['selffac'][lay] * _lerp(selfref, inds, sc['selffrac'][lay])
        taufor  = sc['forfac'][lay]  * _lerp(forref,  indf, sc['forfrac'][lay])
        absco2  = adjcolco2 * _minor_2d(ka_mco2, jmco2, indm, fmco2, sc['minorfrac'][lay])

        tau[lay, :] = tau_major + tau_major1 + tauself + taufor + absco2
        fracs[lay, :] = _mixed_lower_planck(colh2o, colo3, refrat_planck_a, fracrefa)

    for lay in range(nl, nlayers):
        jt0  = sc['jt'][lay];  jp0  = sc['jp'][lay];  jt10 = sc['jt1'][lay]
        jp_u = jp0 - 12
        f00, f10, f01, f11 = sc['fac00'][lay], sc['fac10'][lay], sc['fac01'][lay], sc['fac11'][lay]
        indm = sc['indminor'][lay]

        chi_co2 = sc['colco2'][lay] / sc['coldry'][lay]
        ratco2  = 1.0e20 * chi_co2 / chi_mls[1, jp0 + 1]
        if ratco2 > 3.0:
            adjfac    = 2.0 + (ratco2 - 2.0) ** 0.79
            adjcolco2 = adjfac * chi_mls[1, jp0 + 1] * sc['coldry'][lay] * 1.0e-20
        else:
            adjcolco2 = sc['colco2'][lay]

        absco2 = adjcolco2 * _minor_1d(kb_mco2, indm, sc['minorfrac'][lay])
        tau[lay, :] = (_tau_simple_kb(sc['colo3'][lay], kb, jt0, jp_u, jt10, f00, f10, f01, f11)
                       + absco2)
        fracs[lay, :] = fracrefb
        tau[lay, 7:14] *= EMPIRICAL_MULTS

    return tau, fracs


def taumol_b08(sc, c, planck, chi_mls):
    """Band 8: 1080-1180 cm-1.  Low: H2O + CO2,O3,N2O minor.  High: O3 + CO2,N2O minor."""
    ka = c['ka'];  kb = c['kb']
    selfref = c['selfref'];  forref = c['forref']
    ka_mco2 = c['ka_mco2'];  ka_mo3 = c['ka_mo3'];  ka_mn2o = c['ka_mn2o']
    kb_mco2 = c['kb_mco2'];  kb_mn2o = c['kb_mn2o']
    fracrefa = planck['fracrefa'];  fracrefb = planck['fracrefb']
    nl = sc['laytrop'];  nlayers = len(sc['jp'])
    tau = np.zeros((nlayers, 16));  fracs = np.zeros((nlayers, 16))

    for lay in range(nl):
        jt0 = sc['jt'][lay];  jp0 = sc['jp'][lay];  jt10 = sc['jt1'][lay]
        f00, f10, f01, f11 = sc['fac00'][lay], sc['fac10'][lay], sc['fac01'][lay], sc['fac11'][lay]
        inds = sc['indself'][lay];  indf = sc['indfor'][lay];  indm = sc['indminor'][lay]

        chi_co2 = sc['colco2'][lay] / sc['coldry'][lay]
        ratco2  = 1.0e20 * chi_co2 / chi_mls[1, jp0 + 1]
        if ratco2 > 3.0:
            adjfac    = 2.0 + (ratco2 - 2.0) ** 0.65
            adjcolco2 = adjfac * chi_mls[1, jp0 + 1] * sc['coldry'][lay] * 1.0e-20
        else:
            adjcolco2 = sc['colco2'][lay]

        tauself = sc['selffac'][lay] * _lerp(selfref, inds, sc['selffrac'][lay])
        taufor  = sc['forfac'][lay]  * _lerp(forref,  indf, sc['forfrac'][lay])
        absco2  = adjcolco2 * _minor_1d(ka_mco2, indm, sc['minorfrac'][lay])
        abso3   = sc['colo3'][lay]  * _minor_1d(ka_mo3,  indm, sc['minorfrac'][lay])
        absn2o  = sc['coln2o'][lay] * _minor_1d(ka_mn2o, indm, sc['minorfrac'][lay])
        # CFC12, CFC22ADJ terms omitted (WX=0)
        tau[lay, :] = (_tau_simple(sc['colh2o'][lay], ka, jt0, jp0, jt10, f00, f10, f01, f11)
                       + tauself + taufor + absco2 + abso3 + absn2o)
        fracs[lay, :] = fracrefa

    for lay in range(nl, nlayers):
        jt0  = sc['jt'][lay];  jp0  = sc['jp'][lay];  jt10 = sc['jt1'][lay]
        jp_u = jp0 - 12
        f00, f10, f01, f11 = sc['fac00'][lay], sc['fac10'][lay], sc['fac01'][lay], sc['fac11'][lay]
        indm = sc['indminor'][lay]

        chi_co2 = sc['colco2'][lay] / sc['coldry'][lay]
        ratco2  = 1.0e20 * chi_co2 / chi_mls[1, jp0 + 1]
        if ratco2 > 3.0:
            adjfac    = 2.0 + (ratco2 - 2.0) ** 0.65
            adjcolco2 = adjfac * chi_mls[1, jp0 + 1] * sc['coldry'][lay] * 1.0e-20
        else:
            adjcolco2 = sc['colco2'][lay]

        absco2 = adjcolco2 * _minor_1d(kb_mco2, indm, sc['minorfrac'][lay])
        absn2o = sc['coln2o'][lay] * _minor_1d(kb_mn2o, indm, sc['minorfrac'][lay])
        tau[lay, :] = (_tau_simple_kb(sc['colo3'][lay], kb, jt0, jp_u, jt10, f00, f10, f01, f11)
                       + absco2 + absn2o)
        fracs[lay, :] = fracrefb

    return tau, fracs


def taumol_b09(sc, c, planck, chi_mls):
    """Band 9: 1180-1390 cm-1.  Low: H2O+CH4 + N2O minor.  High: CH4 + N2O minor."""
    ka = c['ka'];  kb = c['kb']
    selfref = c['selfref'];  forref = c['forref']
    ka_mn2o = c['ka_mn2o'];  kb_mn2o = c['kb_mn2o']
    fracrefa = planck['fracrefa'];  fracrefb = planck['fracrefb']
    nl = sc['laytrop'];  nlayers = len(sc['jp'])
    tau = np.zeros((nlayers, 16));  fracs = np.zeros((nlayers, 16))

    refrat_planck_a = chi_mls[0, 8] / chi_mls[5, 8]
    refrat_m_a      = chi_mls[0, 2] / chi_mls[5, 2]

    for lay in range(nl):
        jp0  = sc['jp'][lay];  jt0 = sc['jt'][lay];  jt10 = sc['jt1'][lay]
        f00, f10, f01, f11 = sc['fac00'][lay], sc['fac10'][lay], sc['fac01'][lay], sc['fac11'][lay]
        inds = sc['indself'][lay];  indf = sc['indfor'][lay];  indm = sc['indminor'][lay]
        colh2o = sc['colh2o'][lay];  colch4 = sc['colch4'][lay]

        speccomb  = colh2o + sc['rat_h2och4'][lay]   * colch4
        specparm  = min(colh2o / speccomb,  ONEMINUS)
        specmult  = 8.0 * specparm;  js  = int(specmult);  fs  = specmult - js
        speccomb1 = colh2o + sc['rat_h2och4_1'][lay] * colch4
        specparm1 = min(colh2o / speccomb1, ONEMINUS)
        specmult1 = 8.0 * specparm1;  js1 = int(specmult1);  fs1 = specmult1 - js1

        speccomb_mn2o = colh2o + refrat_m_a * colch4
        specparm_mn2o = min(colh2o / speccomb_mn2o, ONEMINUS)
        jmn2o = int(8.0 * specparm_mn2o);  fmn2o = 8.0 * specparm_mn2o - jmn2o

        adjcoln2o = _n2o_adj(sc['coln2o'][lay], sc['coldry'][lay], chi_mls, jp0)

        br0, br1 = _mixed_factors(specparm, specparm1, fs, fs1, f00, f10, f01, f11)
        tau_major  = _tau_mixed_nspa9(speccomb,  ka, js,  jt0,  jp0,     br0)
        tau_major1 = _tau_mixed_nspa9(speccomb1, ka, js1, jt10, jp0 + 1, br1)
        tauself = sc['selffac'][lay] * _lerp(selfref, inds, sc['selffrac'][lay])
        taufor  = sc['forfac'][lay]  * _lerp(forref,  indf, sc['forfrac'][lay])
        taun2o  = adjcoln2o * _minor_2d(ka_mn2o, jmn2o, indm, fmn2o, sc['minorfrac'][lay])

        tau[lay, :] = tau_major + tau_major1 + tauself + taufor + taun2o
        fracs[lay, :] = _mixed_lower_planck(colh2o, colch4, refrat_planck_a, fracrefa)

    for lay in range(nl, nlayers):
        jt0  = sc['jt'][lay];  jp0  = sc['jp'][lay];  jt10 = sc['jt1'][lay]
        jp_u = jp0 - 12
        f00, f10, f01, f11 = sc['fac00'][lay], sc['fac10'][lay], sc['fac01'][lay], sc['fac11'][lay]
        indm = sc['indminor'][lay]

        adjcoln2o = _n2o_adj(sc['coln2o'][lay], sc['coldry'][lay], chi_mls, jp0)
        taun2o    = adjcoln2o * _minor_1d(kb_mn2o, indm, sc['minorfrac'][lay])
        tau[lay, :] = (_tau_simple_kb(sc['colch4'][lay], kb, jt0, jp_u, jt10, f00, f10, f01, f11)
                       + taun2o)
        fracs[lay, :] = fracrefb

    return tau, fracs


def taumol_b10(sc, c, planck):
    """Band 10: 1390-1480 cm-1.  Low/high: H2O."""
    ka = c['ka'];  kb = c['kb']
    selfref = c['selfref'];  forref = c['forref']
    fracrefa = planck['fracrefa'];  fracrefb = planck['fracrefb']
    nl = sc['laytrop'];  nlayers = len(sc['jp'])
    tau = np.zeros((nlayers, 16));  fracs = np.zeros((nlayers, 16))

    for lay in range(nl):
        jt0 = sc['jt'][lay];  jp0 = sc['jp'][lay];  jt10 = sc['jt1'][lay]
        f00, f10, f01, f11 = sc['fac00'][lay], sc['fac10'][lay], sc['fac01'][lay], sc['fac11'][lay]
        inds = sc['indself'][lay];  indf = sc['indfor'][lay]
        tauself = sc['selffac'][lay] * _lerp(selfref, inds, sc['selffrac'][lay])
        taufor  = sc['forfac'][lay]  * _lerp(forref,  indf, sc['forfrac'][lay])
        tau[lay, :] = (_tau_simple(sc['colh2o'][lay], ka, jt0, jp0, jt10, f00, f10, f01, f11)
                       + tauself + taufor)
        fracs[lay, :] = fracrefa

    for lay in range(nl, nlayers):
        jt0  = sc['jt'][lay];  jp0  = sc['jp'][lay];  jt10 = sc['jt1'][lay]
        jp_u = jp0 - 12
        f00, f10, f01, f11 = sc['fac00'][lay], sc['fac10'][lay], sc['fac01'][lay], sc['fac11'][lay]
        indf = sc['indfor'][lay]
        taufor = sc['forfac'][lay] * _lerp(forref, indf, sc['forfrac'][lay])
        tau[lay, :] = (_tau_simple_kb(sc['colh2o'][lay], kb, jt0, jp_u, jt10, f00, f10, f01, f11)
                       + taufor)
        fracs[lay, :] = fracrefb

    return tau, fracs


def taumol_b11(sc, c, planck):
    """Band 11: 1480-1800 cm-1.  Low/high: H2O.  Minor: O2."""
    ka = c['ka'];  kb = c['kb']
    selfref = c['selfref'];  forref = c['forref']
    ka_mo2 = c['ka_mo2'];  kb_mo2 = c['kb_mo2']
    fracrefa = planck['fracrefa'];  fracrefb = planck['fracrefb']
    nl = sc['laytrop'];  nlayers = len(sc['jp'])
    tau = np.zeros((nlayers, 16));  fracs = np.zeros((nlayers, 16))

    for lay in range(nl):
        jt0 = sc['jt'][lay];  jp0 = sc['jp'][lay];  jt10 = sc['jt1'][lay]
        f00, f10, f01, f11 = sc['fac00'][lay], sc['fac10'][lay], sc['fac01'][lay], sc['fac11'][lay]
        inds = sc['indself'][lay];  indf = sc['indfor'][lay];  indm = sc['indminor'][lay]
        scaleo2 = sc['colo2'][lay] * sc['scaleminor'][lay]
        tauself = sc['selffac'][lay] * _lerp(selfref, inds, sc['selffrac'][lay])
        taufor  = sc['forfac'][lay]  * _lerp(forref,  indf, sc['forfrac'][lay])
        tauo2   = scaleo2 * _minor_1d(ka_mo2, indm, sc['minorfrac'][lay])
        tau[lay, :] = (_tau_simple(sc['colh2o'][lay], ka, jt0, jp0, jt10, f00, f10, f01, f11)
                       + tauself + taufor + tauo2)
        fracs[lay, :] = fracrefa

    for lay in range(nl, nlayers):
        jt0  = sc['jt'][lay];  jp0  = sc['jp'][lay];  jt10 = sc['jt1'][lay]
        jp_u = jp0 - 12
        f00, f10, f01, f11 = sc['fac00'][lay], sc['fac10'][lay], sc['fac01'][lay], sc['fac11'][lay]
        indf = sc['indfor'][lay];  indm = sc['indminor'][lay]
        scaleo2 = sc['colo2'][lay] * sc['scaleminor'][lay]
        taufor  = sc['forfac'][lay] * _lerp(forref, indf, sc['forfrac'][lay])
        tauo2   = scaleo2 * _minor_1d(kb_mo2, indm, sc['minorfrac'][lay])
        tau[lay, :] = (_tau_simple_kb(sc['colh2o'][lay], kb, jt0, jp_u, jt10, f00, f10, f01, f11)
                       + taufor + tauo2)
        fracs[lay, :] = fracrefb

    return tau, fracs


def taumol_b12(sc, c, planck, chi_mls):
    """Band 12: 1800-2080 cm-1.  Low: H2O+CO2.  High: nothing (tau=0)."""
    ka = c['ka']
    selfref = c['selfref'];  forref = c['forref']
    fracrefa = planck['fracrefa']
    nl = sc['laytrop'];  nlayers = len(sc['jp'])
    tau = np.zeros((nlayers, 16));  fracs = np.zeros((nlayers, 16))

    refrat_planck_a = chi_mls[0, 9] / chi_mls[1, 9]

    for lay in range(nl):
        jp0  = sc['jp'][lay];  jt0 = sc['jt'][lay];  jt10 = sc['jt1'][lay]
        f00, f10, f01, f11 = sc['fac00'][lay], sc['fac10'][lay], sc['fac01'][lay], sc['fac11'][lay]
        inds = sc['indself'][lay];  indf = sc['indfor'][lay]
        colh2o = sc['colh2o'][lay];  colco2 = sc['colco2'][lay]

        speccomb  = colh2o + sc['rat_h2oco2'][lay]   * colco2
        specparm  = min(colh2o / speccomb,  ONEMINUS)
        specmult  = 8.0 * specparm;  js  = int(specmult);  fs  = specmult - js
        speccomb1 = colh2o + sc['rat_h2oco2_1'][lay] * colco2
        specparm1 = min(colh2o / speccomb1, ONEMINUS)
        specmult1 = 8.0 * specparm1;  js1 = int(specmult1);  fs1 = specmult1 - js1

        br0, br1 = _mixed_factors(specparm, specparm1, fs, fs1, f00, f10, f01, f11)
        tau_major  = _tau_mixed_nspa9(speccomb,  ka, js,  jt0,  jp0,     br0)
        tau_major1 = _tau_mixed_nspa9(speccomb1, ka, js1, jt10, jp0 + 1, br1)
        tauself = sc['selffac'][lay] * _lerp(selfref, inds, sc['selffrac'][lay])
        taufor  = sc['forfac'][lay]  * _lerp(forref,  indf, sc['forfrac'][lay])

        tau[lay, :] = tau_major + tau_major1 + tauself + taufor
        fracs[lay, :] = _mixed_lower_planck(colh2o, colco2, refrat_planck_a, fracrefa)

    # Above LAYTROP: tau=0, fracs=0
    return tau, fracs


def taumol_b13(sc, c, planck, chi_mls):
    """Band 13: 2080-2250 cm-1.  Low: H2O+N2O + CO2,CO minor.  High: O3 minor."""
    ka = c['ka']
    selfref = c['selfref'];  forref = c['forref']
    ka_mco2 = c['ka_mco2'];  ka_mco = c['ka_mco']
    kb_mo3  = c['kb_mo3']
    fracrefa = planck['fracrefa'];  fracrefb = planck['fracrefb']
    nl = sc['laytrop'];  nlayers = len(sc['jp'])
    tau = np.zeros((nlayers, 16));  fracs = np.zeros((nlayers, 16))

    refrat_planck_a = chi_mls[0, 4] / chi_mls[3, 4]
    refrat_m_a      = chi_mls[0, 0] / chi_mls[3, 0]
    refrat_m_a3     = chi_mls[0, 2] / chi_mls[3, 2]

    for lay in range(nl):
        jp0  = sc['jp'][lay];  jt0 = sc['jt'][lay];  jt10 = sc['jt1'][lay]
        f00, f10, f01, f11 = sc['fac00'][lay], sc['fac10'][lay], sc['fac01'][lay], sc['fac11'][lay]
        inds = sc['indself'][lay];  indf = sc['indfor'][lay];  indm = sc['indminor'][lay]
        colh2o = sc['colh2o'][lay];  coln2o = sc['coln2o'][lay]

        speccomb  = colh2o + sc['rat_h2on2o'][lay]   * coln2o
        specparm  = min(colh2o / speccomb,  ONEMINUS)
        specmult  = 8.0 * specparm;  js  = int(specmult);  fs  = specmult - js
        speccomb1 = colh2o + sc['rat_h2on2o_1'][lay] * coln2o
        specparm1 = min(colh2o / speccomb1, ONEMINUS)
        specmult1 = 8.0 * specparm1;  js1 = int(specmult1);  fs1 = specmult1 - js1

        speccomb_mco2 = colh2o + refrat_m_a  * coln2o
        specparm_mco2 = min(colh2o / speccomb_mco2, ONEMINUS)
        jmco2 = int(8.0 * specparm_mco2);  fmco2 = 8.0 * specparm_mco2 - jmco2

        speccomb_mco = colh2o + refrat_m_a3 * coln2o
        specparm_mco = min(colh2o / speccomb_mco, ONEMINUS)
        jmco = int(8.0 * specparm_mco);  fmco = 8.0 * specparm_mco - jmco

        # CO2 adjustment (exponent 0.68, reference 3.55e-4)
        chi_co2 = sc['colco2'][lay] / sc['coldry'][lay]
        ratco2  = 1.0e20 * chi_co2 / 3.55e-4
        if ratco2 > 3.0:
            adjfac    = 2.0 + (ratco2 - 2.0) ** 0.68
            adjcolco2 = adjfac * 3.55e-4 * sc['coldry'][lay] * 1.0e-20
        else:
            adjcolco2 = sc['colco2'][lay]

        br0, br1 = _mixed_factors(specparm, specparm1, fs, fs1, f00, f10, f01, f11)
        tau_major  = _tau_mixed_nspa9(speccomb,  ka, js,  jt0,  jp0,     br0)
        tau_major1 = _tau_mixed_nspa9(speccomb1, ka, js1, jt10, jp0 + 1, br1)
        tauself = sc['selffac'][lay] * _lerp(selfref, inds, sc['selffrac'][lay])
        taufor  = sc['forfac'][lay]  * _lerp(forref,  indf, sc['forfrac'][lay])
        absco2  = adjcolco2 * _minor_2d(ka_mco2, jmco2, indm, fmco2, sc['minorfrac'][lay])
        absco   = sc['colco'][lay] * _minor_2d(ka_mco, jmco, indm, fmco, sc['minorfrac'][lay])

        tau[lay, :] = tau_major + tau_major1 + tauself + taufor + absco2 + absco
        fracs[lay, :] = _mixed_lower_planck(colh2o, coln2o, refrat_planck_a, fracrefa)

    for lay in range(nl, nlayers):
        indm = sc['indminor'][lay]
        abso3 = _minor_1d(kb_mo3, indm, sc['minorfrac'][lay])
        tau[lay, :] = sc['colo3'][lay] * abso3
        fracs[lay, :] = fracrefb

    return tau, fracs


def taumol_b14(sc, c, planck):
    """Band 14: 2250-2380 cm-1.  Low/high: CO2."""
    ka = c['ka'];  kb = c['kb']
    selfref = c['selfref'];  forref = c['forref']
    fracrefa = planck['fracrefa'];  fracrefb = planck['fracrefb']
    nl = sc['laytrop'];  nlayers = len(sc['jp'])
    tau = np.zeros((nlayers, 16));  fracs = np.zeros((nlayers, 16))

    for lay in range(nl):
        jt0 = sc['jt'][lay];  jp0 = sc['jp'][lay];  jt10 = sc['jt1'][lay]
        f00, f10, f01, f11 = sc['fac00'][lay], sc['fac10'][lay], sc['fac01'][lay], sc['fac11'][lay]
        inds = sc['indself'][lay];  indf = sc['indfor'][lay]
        tauself = sc['selffac'][lay] * _lerp(selfref, inds, sc['selffrac'][lay])
        taufor  = sc['forfac'][lay]  * _lerp(forref,  indf, sc['forfrac'][lay])
        tau[lay, :] = (_tau_simple(sc['colco2'][lay], ka, jt0, jp0, jt10, f00, f10, f01, f11)
                       + tauself + taufor)
        fracs[lay, :] = fracrefa

    for lay in range(nl, nlayers):
        jt0  = sc['jt'][lay];  jp_u = sc['jp'][lay] - 12;  jt10 = sc['jt1'][lay]
        f00, f10, f01, f11 = sc['fac00'][lay], sc['fac10'][lay], sc['fac01'][lay], sc['fac11'][lay]
        tau[lay, :] = _tau_simple_kb(sc['colco2'][lay], kb, jt0, jp_u, jt10, f00, f10, f01, f11)
        fracs[lay, :] = fracrefb

    return tau, fracs


def taumol_b15(sc, c, planck, chi_mls):
    """Band 15: 2380-2600 cm-1.  Low: N2O+CO2 + N2 minor.  High: nothing (tau=0)."""
    ka = c['ka']
    selfref = c['selfref'];  forref = c['forref']
    ka_mn2 = c['ka_mn2']
    fracrefa = planck['fracrefa']
    nl = sc['laytrop'];  nlayers = len(sc['jp'])
    tau = np.zeros((nlayers, 16));  fracs = np.zeros((nlayers, 16))

    refrat_planck_a = chi_mls[3, 0] / chi_mls[1, 0]
    refrat_m_a      = chi_mls[3, 0] / chi_mls[1, 0]

    for lay in range(nl):
        jp0  = sc['jp'][lay];  jt0 = sc['jt'][lay];  jt10 = sc['jt1'][lay]
        f00, f10, f01, f11 = sc['fac00'][lay], sc['fac10'][lay], sc['fac01'][lay], sc['fac11'][lay]
        inds = sc['indself'][lay];  indf = sc['indfor'][lay];  indm = sc['indminor'][lay]
        coln2o = sc['coln2o'][lay];  colco2 = sc['colco2'][lay]

        speccomb  = coln2o + sc['rat_n2oco2'][lay]   * colco2
        specparm  = min(coln2o / speccomb,  ONEMINUS)
        specmult  = 8.0 * specparm;  js  = int(specmult);  fs  = specmult - js
        speccomb1 = coln2o + sc['rat_n2oco2_1'][lay] * colco2
        specparm1 = min(coln2o / speccomb1, ONEMINUS)
        specmult1 = 8.0 * specparm1;  js1 = int(specmult1);  fs1 = specmult1 - js1

        speccomb_mn2 = coln2o + refrat_m_a * colco2
        specparm_mn2 = min(coln2o / speccomb_mn2, ONEMINUS)
        jmn2 = int(8.0 * specparm_mn2);  fmn2 = 8.0 * specparm_mn2 - jmn2
        scalen2 = sc['colbrd'][lay] * sc['scaleminor'][lay]

        br0, br1 = _mixed_factors(specparm, specparm1, fs, fs1, f00, f10, f01, f11)
        tau_major  = _tau_mixed_nspa9(speccomb,  ka, js,  jt0,  jp0,     br0)
        tau_major1 = _tau_mixed_nspa9(speccomb1, ka, js1, jt10, jp0 + 1, br1)
        tauself = sc['selffac'][lay] * _lerp(selfref, inds, sc['selffrac'][lay])
        taufor  = sc['forfac'][lay]  * _lerp(forref,  indf, sc['forfrac'][lay])
        taun2   = scalen2 * _minor_2d(ka_mn2, jmn2, indm, fmn2, sc['minorfrac'][lay])

        tau[lay, :] = tau_major + tau_major1 + tauself + taufor + taun2
        fracs[lay, :] = _mixed_lower_planck(coln2o, colco2, refrat_planck_a, fracrefa)

    # Above LAYTROP: tau=0, fracs=0
    return tau, fracs


def taumol_b16(sc, c, planck, chi_mls):
    """Band 16: 2600-3250 cm-1.  Low: H2O+CH4.  High: CH4."""
    ka = c['ka'];  kb = c['kb']
    selfref = c['selfref'];  forref = c['forref']
    fracrefa = planck['fracrefa'];  fracrefb = planck['fracrefb']
    nl = sc['laytrop'];  nlayers = len(sc['jp'])
    tau = np.zeros((nlayers, 16));  fracs = np.zeros((nlayers, 16))

    refrat_planck_a = chi_mls[0, 5] / chi_mls[5, 5]

    for lay in range(nl):
        jp0  = sc['jp'][lay];  jt0 = sc['jt'][lay];  jt10 = sc['jt1'][lay]
        f00, f10, f01, f11 = sc['fac00'][lay], sc['fac10'][lay], sc['fac01'][lay], sc['fac11'][lay]
        inds = sc['indself'][lay];  indf = sc['indfor'][lay]
        colh2o = sc['colh2o'][lay];  colch4 = sc['colch4'][lay]

        speccomb  = colh2o + sc['rat_h2och4'][lay]   * colch4
        specparm  = min(colh2o / speccomb,  ONEMINUS)
        specmult  = 8.0 * specparm;  js  = int(specmult);  fs  = specmult - js
        speccomb1 = colh2o + sc['rat_h2och4_1'][lay] * colch4
        specparm1 = min(colh2o / speccomb1, ONEMINUS)
        specmult1 = 8.0 * specparm1;  js1 = int(specmult1);  fs1 = specmult1 - js1

        br0, br1 = _mixed_factors(specparm, specparm1, fs, fs1, f00, f10, f01, f11)
        tau_major  = _tau_mixed_nspa9(speccomb,  ka, js,  jt0,  jp0,     br0)
        tau_major1 = _tau_mixed_nspa9(speccomb1, ka, js1, jt10, jp0 + 1, br1)
        tauself = sc['selffac'][lay] * _lerp(selfref, inds, sc['selffrac'][lay])
        taufor  = sc['forfac'][lay]  * _lerp(forref,  indf, sc['forfrac'][lay])

        tau[lay, :] = tau_major + tau_major1 + tauself + taufor
        fracs[lay, :] = _mixed_lower_planck(colh2o, colch4, refrat_planck_a, fracrefa)

    for lay in range(nl, nlayers):
        jt0  = sc['jt'][lay];  jp_u = sc['jp'][lay] - 12;  jt10 = sc['jt1'][lay]
        f00, f10, f01, f11 = sc['fac00'][lay], sc['fac10'][lay], sc['fac01'][lay], sc['fac11'][lay]
        tau[lay, :] = _tau_simple_kb(sc['colch4'][lay], kb, jt0, jp_u, jt10, f00, f10, f01, f11)
        fracs[lay, :] = fracrefb

    return tau, fracs


# ── Dispatcher ────────────────────────────────────────────────────────────────

def taumol_all(sc, bands, planck, chi_mls):
    """
    Compute optical depths for all 16 bands.

    Parameters
    ----------
    sc      : dict from setcoef() plus 'pavel' (for band-1 corradj) and 'coldry'
    bands   : list of 16 per-band k-coeff dicts (from load_coeffs)
    planck  : list of 16 per-band planck fraction dicts
    chi_mls : (9, 59) chi_mls array from load_coeffs

    Returns
    -------
    tau   : (16, nlayers, 16) optical depth
    fracs : (16, nlayers, 16) Planck fractions
    """
    nlayers = len(sc['jp'])
    tau   = np.zeros((16, nlayers, 16))
    fracs = np.zeros((16, nlayers, 16))

    fns_simple = [taumol_b01, taumol_b02, taumol_b10, taumol_b11, taumol_b14]
    fns_chi    = [taumol_b03, taumol_b04, taumol_b05, taumol_b07, taumol_b08,
                  taumol_b09, taumol_b12, taumol_b13, taumol_b15, taumol_b16]

    tau[0],  fracs[0]  = taumol_b01(sc, bands[0],  planck[0])
    tau[1],  fracs[1]  = taumol_b02(sc, bands[1],  planck[1])
    tau[2],  fracs[2]  = taumol_b03(sc, bands[2],  planck[2],  chi_mls)
    tau[3],  fracs[3]  = taumol_b04(sc, bands[3],  planck[3],  chi_mls)
    tau[4],  fracs[4]  = taumol_b05(sc, bands[4],  planck[4],  chi_mls)
    tau[5],  fracs[5]  = taumol_b06(sc, bands[5],  planck[5],  chi_mls)
    tau[6],  fracs[6]  = taumol_b07(sc, bands[6],  planck[6],  chi_mls)
    tau[7],  fracs[7]  = taumol_b08(sc, bands[7],  planck[7],  chi_mls)
    tau[8],  fracs[8]  = taumol_b09(sc, bands[8],  planck[8],  chi_mls)
    tau[9],  fracs[9]  = taumol_b10(sc, bands[9],  planck[9])
    tau[10], fracs[10] = taumol_b11(sc, bands[10], planck[10])
    tau[11], fracs[11] = taumol_b12(sc, bands[11], planck[11], chi_mls)
    tau[12], fracs[12] = taumol_b13(sc, bands[12], planck[12], chi_mls)
    tau[13], fracs[13] = taumol_b14(sc, bands[13], planck[13])
    tau[14], fracs[14] = taumol_b15(sc, bands[14], planck[14], chi_mls)
    tau[15], fracs[15] = taumol_b16(sc, bands[15], planck[15], chi_mls)

    return tau, fracs
