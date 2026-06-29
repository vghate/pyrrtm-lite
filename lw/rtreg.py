"""
Longwave radiative transfer solver for RRTM_LW.

Implements both:
  - NUMANGS=0 (RTR, diffusivity approximation, SECDIFF=1.66)
  - NUMANGS=1..4 (RTREG, Gaussian quadrature)

Matches the Fortran rtr.f / rtreg.f subroutines.

Performance: the inner g-point loop (16 points) is eliminated by processing
all g-points simultaneously using NumPy broadcasting.  For RTREG, the angle
loop is also vectorized so all (16 g-points x numang angles) are handled in
a single array pass per layer.
"""
import numpy as np

from .constants import BPADE, TBLINT, TRANS, TF, NTBL, FLUXFAC, HEATFAC
from .bands import DELWAVE

# Diffusivity approximation constant (1.66) used for NUMANGS=0
SECDIFF = 1.66
# Solid-angle weight for diffuse flux (0.5) used for NUMANGS=0
WTDIFF = 0.5
# 1/6 for thin-limit Taylor expansion
REC_6 = 1.0 / 6.0

# Gaussian quadrature angles (secants) and weights for NUMANGS=1..4
# SECREG[i, j] = secant of angle i (0-based) for j+1 total angles (0-based)
# WTREG[i, j]  = weight of angle i for j+1 total angles
# Data from Fortran DATA statements in rtreg.f (1-based indices converted to 0-based)
SECREG = np.array([
    # numang=1       numang=2              numang=3                  numang=4
    [1.5,          1.18350343,          1.09719858,              1.06056257],  # angle 1
    [0.0,          2.81649655,          1.69338507,              1.38282560],  # angle 2
    [0.0,          0.0,                 4.70941630,              2.40148179],  # angle 3
    [0.0,          0.0,                 0.0,                     7.15513024],  # angle 4
], dtype=np.float64)

WTREG = np.array([
    # numang=1       numang=2              numang=3                  numang=4
    [0.50000000,   0.3180413817,        0.2009319137,            0.1355069134],  # angle 1
    [0.0,          0.1819586183,        0.2292411064,            0.2034645680],  # angle 2
    [0.0,          0.0,                 0.0698269799,            0.1298475476],  # angle 3
    [0.0,          0.0,                 0.0,                     0.0311809710],  # angle 4
], dtype=np.float64)


# ---------------------------------------------------------------------------
# Vectorized Pade helpers — operate on arrays of arbitrary shape
# ---------------------------------------------------------------------------

def _vec_pade(odepth):
    """
    Compute transmittance factor (atrans) and source-function weighting factor
    (tausfac) for an array of optical depths using the Pade approximation table.

    For odepth <= 0.06 the thin-limit Taylor expansion is used:
        atrans = odepth - 0.5*odepth^2
        tausfac = odepth/6   (stored as od6; actual tausfac irrelevant in thin path)

    For odepth > 0.06 the lookup table TRANS/TF is used.

    Parameters
    ----------
    odepth : np.ndarray, any shape, float64
        Optical depths (already scaled by secant angle).

    Returns
    -------
    atrans  : np.ndarray, same shape — effective absorptance
    tausfac : np.ndarray, same shape — source-function factor (od6 for thin,
              TF[itr] for thick)
    """
    odepth = np.maximum(odepth, 0.0)
    thin = odepth <= 0.06

    # --- Thin limit ---
    od6_thin    = odepth * REC_6
    atrans_thin = odepth - 0.5 * odepth * odepth  # = od*(1 - 0.5*od)

    # --- Thick limit (Pade table) ---
    # Compute table index, guarding against the thin-path entries
    tblind    = np.where(thin, 0.0, odepth / (BPADE + odepth))
    itr       = np.clip((TBLINT * tblind + 0.5).astype(np.int32), 0, NTBL)
    atrans_th = 1.0 - TRANS[itr]
    tausfac_th = TF[itr]

    atrans  = np.where(thin, atrans_thin, atrans_th)
    tausfac = np.where(thin, od6_thin,    tausfac_th)
    return atrans, tausfac


def _vec_layer_source(plfrac, blay, dplankup, dplankdn, tausfac):
    """
    Compute BBU and BBD source terms for an array of (layer, g-point) cells.

    The tausfac argument already encodes od6 (thin path) or TF[itr] (thick path)
    via _vec_pade, so a single formula covers both branches:

        bbd = plfrac * (blay + tausfac * dplankdn)
        bbu = plfrac * (blay + tausfac * dplankup)

    Parameters
    ----------
    plfrac   : array (...,)
    blay     : scalar or array broadcastable to plfrac
    dplankup : scalar or array broadcastable to plfrac
    dplankdn : scalar or array broadcastable to plfrac
    tausfac  : array, same shape as plfrac

    Returns
    -------
    bbu, bbd : arrays, same shape as plfrac
    """
    bbd = plfrac * (blay + tausfac * dplankdn)
    bbu = plfrac * (blay + tausfac * dplankup)
    return bbu, bbd


# ---------------------------------------------------------------------------
# Public solver
# ---------------------------------------------------------------------------

def rtreg(tau, fracs, sc, pz, semiss, numangs=0, aerosol=None,
          tau_cloud=None, frac_cloud=None, ireflect=0):
    """
    Longwave radiative transfer solver.

    Parameters
    ----------
    tau    : (16, nlayers, 16) float64
        Optical depth per band, layer, g-point (from taumol_all).
    fracs  : (16, nlayers, 16) float64
        Planck fractions per band, layer, g-point (from taumol_all).
    sc     : dict
        Output of setcoef().  Required keys:
          'planklay'  (nlayers, 16)   layer Planck function
          'planklev'  (nlayers+1, 16) level Planck function (index 0 = surface)
          'plankbnd'  (16,)           surface Planck emission per band
    pz     : (nlayers+1,) float64
        Level pressures [mb]; pz[0] = surface (highest pressure),
        pz[nlayers] = TOA (lowest pressure).
    semiss : (16,) float64
        Surface emissivity per band.
    numangs : int, optional
        Number of quadrature angles (0 = diffusivity approximation / RTR,
        1..4 = Gaussian quadrature / RTREG).  Default: 0.
    aerosol : list of AerosolLayer or None, optional
        Per-layer aerosol objects.  Each must have a .tau_abs attribute of
        shape (16,) giving band-integrated absorption optical depth.
        When None (default), aerosol is ignored.
    tau_cloud : np.ndarray, shape (nlayers, 16), or None, optional
        Cloud absorption optical depth per layer and band.
        When None (default), cloud is ignored.
    frac_cloud : np.ndarray, shape (nlayers,), or None, optional
        Cloud fraction per layer [0, 1].  Required when tau_cloud is not None.
    ireflect : int, optional
        0 = Lambertian reflection (default), 1 = specular reflection.

    Returns
    -------
    dict with keys:
        totuflux      : (nlayers+1,)    total upward flux   [W/m2]
        totdflux      : (nlayers+1,)    total downward flux [W/m2]
        fnet          : (nlayers+1,)    net flux            [W/m2]
        htr           : (nlayers+1,)    heating rate        [K/day]; htr[nlayers]=0
        band_totuflux : (16, nlayers+1) per-band upward flux
        band_totdflux : (16, nlayers+1) per-band downward flux
        band_fnet     : (16, nlayers+1) per-band net flux
        band_htr      : (16, nlayers+1) per-band heating rate
    """
    if numangs is None:
        numangs = 0

    planklay = sc['planklay']   # (nlayers, 16)
    planklev = sc['planklev']   # (nlayers+1, 16)
    plankbnd = sc['plankbnd']   # (16,)

    nlayers = planklay.shape[0]
    nlevels = nlayers + 1

    # Accumulated broadband flux arrays (pre-FLUXFAC)
    totuflux = np.zeros(nlevels, dtype=np.float64)
    totdflux = np.zeros(nlevels, dtype=np.float64)

    # Per-band flux arrays (pre-FLUXFAC)
    band_totuflux = np.zeros((16, nlevels), dtype=np.float64)
    band_totdflux = np.zeros((16, nlevels), dtype=np.float64)

    # Pre-build aerosol tau_abs array for vectorized access: (nlayers, 16)
    # aer_tau[lay, b] = aerosol optical depth for that layer and band.
    if aerosol is not None:
        aer_tau = np.stack([aerosol[lay].tau_abs for lay in range(nlayers)], axis=0)
        # shape: (nlayers, 16)
    else:
        aer_tau = None

    if numangs == 0:
        # ================================================================
        # RTR: diffusivity approximation — single effective angle SECDIFF
        # Vectorize over all 16 g-points simultaneously.
        # ================================================================

        # Radiance accumulators reset per band: (nlevels, 16)
        urad = np.zeros((nlevels, 16), dtype=np.float64)
        drad = np.zeros((nlevels, 16), dtype=np.float64)

        for b in range(16):
            # Planck quantities for this band — scalars per level/layer
            blay_band = planklay[:, b]   # (nlayers,)
            blev_band = planklev[:, b]   # (nlevels,)

            # Gas optical depth and Planck fracs for all layers and g-points
            tau_b   = tau[b, :, :]    # (nlayers, 16)
            fracs_b = fracs[b, :, :]  # (nlayers, 16)

            # Build effective clear-sky optical depth and Planck fraction
            # for all layers and g-points at once.
            if aer_tau is not None:
                # aer_tau[:, b] shape: (nlayers,) — broadcast over g-points
                tau_aer_b = aer_tau[:, b][:, np.newaxis]          # (nlayers, 1)
                tau_clr   = tau_b + tau_aer_b                     # (nlayers, 16)
                denom     = np.maximum(tau_clr, 1e-30)
                plfrac_clr = (tau_b * fracs_b + tau_aer_b * fracs_b) / denom  # (nlayers, 16)
            else:
                tau_clr    = tau_b                                  # (nlayers, 16)
                plfrac_clr = fracs_b                               # (nlayers, 16)

            # Determine which layers have cloud
            has_cloud_layer = np.zeros(nlayers, dtype=bool)
            f_cld_arr       = np.zeros(nlayers, dtype=np.float64)
            tau_cld_b       = None  # cloud OD for this band: (nlayers, 16)

            if tau_cloud is not None and frac_cloud is not None:
                f_cld_arr = np.asarray(frac_cloud, dtype=np.float64)
                has_cloud_layer = f_cld_arr > 0.0
                if has_cloud_layer.any():
                    # tau_cloud shape: (nlayers, 16) — already per-band
                    tau_cld_b = tau_clr + tau_cloud[:, b][:, np.newaxis]
                    # shape: (nlayers, 16)

            # ---- Downward sweep (TOA -> surface), all 16 g-points at once ----
            radld = np.zeros(16, dtype=np.float64)        # (16,)
            bbu_all    = np.empty((nlayers, 16), dtype=np.float64)
            atrans_all = np.empty((nlayers, 16), dtype=np.float64)

            for lay in range(nlayers - 1, -1, -1):
                blay     = blay_band[lay]                  # scalar
                dplankup = blev_band[lay + 1] - blay       # scalar
                dplankdn = blev_band[lay]     - blay       # scalar
                plfrac   = plfrac_clr[lay, :]              # (16,)

                if not has_cloud_layer[lay]:
                    # Pure clear-sky — fully vectorized over 16 g-points
                    odepth = SECDIFF * tau_clr[lay, :]     # (16,)
                    at, tausfac = _vec_pade(odepth)        # (16,), (16,)
                    bbu_v, bbd = _vec_layer_source(
                        plfrac, blay, dplankup, dplankdn, tausfac)
                    atrans_all[lay] = at
                    bbu_all[lay]    = bbu_v
                    radld = radld + (bbd - radld) * at

                else:
                    # Split into clear and cloudy sub-columns, both vectorized
                    f_c = f_cld_arr[lay]  # scalar cloud fraction

                    # Clear sub-column
                    od_clr = SECDIFF * tau_clr[lay, :]          # (16,)
                    at_clr, tsf_clr = _vec_pade(od_clr)
                    bbu_clr, bbd_clr = _vec_layer_source(
                        plfrac, blay, dplankup, dplankdn, tsf_clr)

                    # Cloudy sub-column
                    od_cld = SECDIFF * tau_cld_b[lay, :]        # (16,)
                    at_cld, tsf_cld = _vec_pade(od_cld)
                    bbu_cld, bbd_cld = _vec_layer_source(
                        plfrac, blay, dplankup, dplankdn, tsf_cld)

                    # Combine sub-columns
                    atrans_all[lay] = (1.0 - f_c) * at_clr + f_c * at_cld
                    bbu_all[lay]    = (1.0 - f_c) * bbu_clr + f_c * bbu_cld
                    bbd_eff         = (1.0 - f_c) * bbd_clr + f_c * bbd_cld
                    radld = radld + (bbd_eff - radld) * atrans_all[lay]

                # Accumulate downwelling radiance at level lay (bottom of layer lay)
                drad[lay] += radld

            # ---- Surface boundary condition ----
            rad0    = fracs_b[0, :] * plankbnd[b]    # (16,) — bottom-layer fracs
            reflect = 1.0 - semiss[b]
            if ireflect == 1:
                # Specular reflection
                radlu = rad0 + reflect * radld
            else:
                # Lambertian reflection
                radlu = rad0 + reflect * (2.0 * radld * WTDIFF)
            urad[0] += radlu

            # ---- Upward sweep (surface -> TOA), all 16 g-points at once ----
            for lay in range(nlayers):
                radlu = radlu + (bbu_all[lay] - radlu) * atrans_all[lay]
                urad[lay + 1] += radlu

            # ---- Convert radiance -> flux and accumulate band totals ----
            uflux = urad * WTDIFF                             # (nlevels, 16)
            dflux = drad * WTDIFF                             # (nlevels, 16)
            band_totuflux[b] = uflux.sum(axis=1) * DELWAVE[b]
            band_totdflux[b] = dflux.sum(axis=1) * DELWAVE[b]
            totuflux += band_totuflux[b]
            totdflux += band_totdflux[b]

            # Reset accumulators for the next band
            urad[:] = 0.0
            drad[:] = 0.0

    else:
        # ================================================================
        # RTREG: multi-angle Gaussian quadrature
        # Vectorize over all (16 g-points x numang angles) simultaneously.
        # Shape convention inside the band loop:
        #   (nlayers, 16, numang)  — layer x g-point x angle
        # ================================================================
        numang   = min(numangs, 4)
        secang   = SECREG[:numang, numang - 1]   # (numang,)
        angweigh = WTREG[:numang, numang - 1]    # (numang,)

        # Radiance accumulators per band: (nlevels, numang)
        # We accumulate weighted sums over g-points into these.
        urad = np.zeros((nlevels, numang), dtype=np.float64)
        drad = np.zeros((nlevels, numang), dtype=np.float64)

        for b in range(16):
            blay_band = planklay[:, b]   # (nlayers,)
            blev_band = planklev[:, b]   # (nlevels,)

            tau_b   = tau[b, :, :]    # (nlayers, 16)
            fracs_b = fracs[b, :, :]  # (nlayers, 16)

            # Effective clear-sky tau and plfrac: (nlayers, 16)
            if aer_tau is not None:
                tau_aer_b  = aer_tau[:, b][:, np.newaxis]
                tau_clr    = tau_b + tau_aer_b
                denom      = np.maximum(tau_clr, 1e-30)
                plfrac_clr = (tau_b * fracs_b + tau_aer_b * fracs_b) / denom
            else:
                tau_clr    = tau_b
                plfrac_clr = fracs_b

            # Cloud bookkeeping
            has_cloud_layer = np.zeros(nlayers, dtype=bool)
            f_cld_arr       = np.zeros(nlayers, dtype=np.float64)
            tau_cld_b       = None

            if tau_cloud is not None and frac_cloud is not None:
                f_cld_arr = np.asarray(frac_cloud, dtype=np.float64)
                has_cloud_layer = f_cld_arr > 0.0
                if has_cloud_layer.any():
                    tau_cld_b = tau_clr + tau_cloud[:, b][:, np.newaxis]
                    # (nlayers, 16)

            # ---- Downward sweep, all 16 g-points x numang angles at once ----
            # Working shape: (16, numang) — g-point x angle
            # secang broadcast: (1, numang)
            secang_row = secang[np.newaxis, :]   # (1, numang)

            radld = np.zeros((16, numang), dtype=np.float64)   # (16, numang)
            bbu_all    = np.empty((nlayers, 16, numang), dtype=np.float64)
            atrans_all = np.empty((nlayers, 16, numang), dtype=np.float64)

            for lay in range(nlayers - 1, -1, -1):
                blay     = blay_band[lay]                 # scalar
                dplankup = blev_band[lay + 1] - blay      # scalar
                dplankdn = blev_band[lay]     - blay      # scalar
                # plfrac: (16,) -> expand to (16, numang) by broadcasting
                plfrac   = plfrac_clr[lay, :, np.newaxis]  # (16, 1) -> broadcasts

                if not has_cloud_layer[lay]:
                    # Clear-sky: tau_clr[lay,:] shape (16,)
                    # secang_row shape (1, numang) -> odepth shape (16, numang)
                    odepth = tau_clr[lay, :, np.newaxis] * secang_row  # (16, numang)
                    at, tausfac = _vec_pade(odepth)        # (16, numang)
                    bbu_v, bbd = _vec_layer_source(
                        plfrac, blay, dplankup, dplankdn, tausfac)
                    atrans_all[lay] = at
                    bbu_all[lay]    = bbu_v
                    radld = radld + (bbd - radld) * at

                else:
                    f_c = f_cld_arr[lay]  # scalar

                    # Clear sub-column
                    od_clr = tau_clr[lay, :, np.newaxis] * secang_row   # (16, numang)
                    at_clr, tsf_clr = _vec_pade(od_clr)
                    bbu_clr, bbd_clr = _vec_layer_source(
                        plfrac, blay, dplankup, dplankdn, tsf_clr)

                    # Cloudy sub-column
                    od_cld = tau_cld_b[lay, :, np.newaxis] * secang_row  # (16, numang)
                    at_cld, tsf_cld = _vec_pade(od_cld)
                    bbu_cld, bbd_cld = _vec_layer_source(
                        plfrac, blay, dplankup, dplankdn, tsf_cld)

                    # Combine
                    atrans_all[lay] = (1.0 - f_c) * at_clr + f_c * at_cld
                    bbu_all[lay]    = (1.0 - f_c) * bbu_clr + f_c * bbu_cld
                    bbd_eff         = (1.0 - f_c) * bbd_clr + f_c * bbd_cld
                    radld = radld + (bbd_eff - radld) * atrans_all[lay]

                # Accumulate downwelling over 16 g-points (sum, not mean)
                # drad shape: (nlevels, numang); radld shape: (16, numang)
                drad[lay] += radld.sum(axis=0)

            # ---- Surface boundary condition ----
            # radld shape: (16, numang) — downwelling at surface per g-point/angle
            rad0    = fracs_b[0, :, np.newaxis] * plankbnd[b]  # (16, 1) -> broadcasts
            reflect = 1.0 - semiss[b]

            if ireflect == 1:
                # Specular: each angle sees its own radld
                radlu = rad0 + reflect * radld              # (16, numang)
            else:
                # Lambertian: all angles see the angularly averaged downwelling.
                # radsum per g-point = sum_iang (angweigh[iang] * radld[:, iang])
                # shape: (16,) -> expand to (16, numang) by broadcasting
                radsum = (radld * angweigh[np.newaxis, :]).sum(axis=1)  # (16,)
                radlu  = rad0 + 2.0 * reflect * radsum[:, np.newaxis]   # (16, numang)

            urad[0] += radlu.sum(axis=0)   # sum over 16 g-points

            # ---- Upward sweep, all g-points x angles at once ----
            for lay in range(nlayers):
                radlu = radlu + (bbu_all[lay] - radlu) * atrans_all[lay]
                urad[lay + 1] += radlu.sum(axis=0)

            # ---- Convert radiance -> flux and accumulate band totals ----
            # urad/drad: (nlevels, numang) — weighted sums over g-points
            # angweigh: (numang,)
            uflux = np.dot(urad, angweigh)   # (nlevels,)
            dflux = np.dot(drad, angweigh)   # (nlevels,)
            band_totuflux[b] = uflux * DELWAVE[b]
            band_totdflux[b] = dflux * DELWAVE[b]
            totuflux += band_totuflux[b]
            totdflux += band_totdflux[b]

            # Reset accumulators for the next band
            urad[:] = 0.0
            drad[:] = 0.0

    # Apply flux factor (pi * 2e4) to convert to W/m2
    totuflux      *= FLUXFAC
    totdflux      *= FLUXFAC
    band_totuflux *= FLUXFAC
    band_totdflux *= FLUXFAC

    # Net flux
    fnet      = totuflux - totdflux
    band_fnet = band_totuflux - band_totdflux

    # Heating rate [K/day]
    # HTR(lev) = HEATFAC * (FNET(lev) - FNET(lev+1)) / (PZ(lev) - PZ(lev+1))
    # pz[0] = surface (highest pressure), pz[nlayers] = TOA
    dp = pz[:nlayers] - pz[1:nlevels]          # (nlayers,) — positive (decreasing P)
    htr = np.zeros(nlevels, dtype=np.float64)
    htr[:nlayers] = HEATFAC * (fnet[:nlayers] - fnet[1:nlevels]) / dp

    band_htr = np.zeros((16, nlevels), dtype=np.float64)
    band_htr[:, :nlayers] = HEATFAC * (
        band_fnet[:, :nlayers] - band_fnet[:, 1:nlevels]
    ) / dp[np.newaxis, :]

    return {
        'totuflux':      totuflux,
        'totdflux':      totdflux,
        'fnet':          fnet,
        'htr':           htr,
        'band_totuflux': band_totuflux,
        'band_totdflux': band_totdflux,
        'band_fnet':     band_fnet,
        'band_htr':      band_htr,
    }
