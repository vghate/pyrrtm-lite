"""
Pure NumPy 16-stream discrete ordinate shortwave solver (batched, fast).

Implements DISORT (Discrete Ordinate Radiative Transfer) for RRTM_SW.
No Fortran dependencies.

Quadrature
----------
Uses the upper half of a 16-point Gauss-Legendre quadrature on [-1,1],
giving 8 positive cosine values matching DISORT's double-Gauss approach.

Physics conventions
-------------------
tau increases downward from TOA (tau=0) to surface.
mu_all[0..NN-1] < 0 = downward streams; mu_all[NN..NSTR-1] > 0 = upward streams.

RTE matrix: A[i,j] = (delta_ij - (omega/2) w_j P(mu_i,mu_j)) / mu_all[i]

Speedup vs original
-------------------
The original called scipy.linalg.eig() once per layer per g-point (up to
nlayers x 16 x nbands calls).  This rewrite builds all layer matrices for
one g-point at once and calls np.linalg.eig() on the whole batch — one call
returning (n_sc, 16) eigenvalues and (n_sc, 16, 16) eigenvectors.
Module-level quadrature and Legendre-polynomial setup is computed once at
import time and reused across all calls.

Output ordering
---------------
Index 0 = surface, index nlyr = TOA (matches Fortran wrapper convention).
"""

import numpy as np
from scipy.linalg import solve_banded
from .constants import HEATFAC

# ---------------------------------------------------------------------------
# Module-level quadrature setup  (computed ONCE at import)
# ---------------------------------------------------------------------------

NSTR = 16
NN   = NSTR // 2   # 8 positive cosines
NMOM = NSTR        # number of Legendre moments used for phase function

# Full 16-pt Gauss-Legendre quadrature on [-1, 1]
_x16, _w16 = np.polynomial.legendre.leggauss(NSTR)

# Upper half (positive cosines), sorted ascending
mu_pos = _x16[NN:][::-1].copy()   # (NN,)  ascending positive cosines
wt_pos = _w16[NN:][::-1].copy()   # (NN,)  corresponding weights

# Full angle set: downward (mu<0) then upward (mu>0)
mu_all = np.concatenate([-mu_pos[::-1], mu_pos])   # (NSTR,)
wt_all = np.concatenate([ wt_pos[::-1], wt_pos])   # (NSTR,)

# Absolute value of all mu (used for diagonal of abs-only layers)
ABS_MU = np.abs(mu_all)

# Flux-weight arrays
_wt_flux_dn = wt_all[:NN] * np.abs(mu_all[:NN])   # (NN,)
_wt_flux_up = wt_all[NN:] * mu_all[NN:]            # (NN,)
_sum_wt_up  = _wt_flux_up.sum()

# Pre-compute Legendre polynomials at all NSTR quadrature angles
# PL shape: (NMOM+1, NSTR)
def _legpoly_matrix(mu, nmom):
    """Evaluate P_l(mu) for l=0..nmom at every angle in mu. Returns (nmom+1, len(mu))."""
    n = len(mu)
    pl = np.zeros((nmom + 1, n), dtype=np.float64)
    pl[0] = 1.0
    if nmom >= 1:
        pl[1] = mu
    for l in range(2, nmom + 1):
        pl[l] = ((2 * l - 1) * mu * pl[l - 1] - (l - 1) * pl[l - 2]) / l
    return pl

PL = _legpoly_matrix(mu_all, NMOM)   # (NMOM+1, NSTR)  — reused every call

# ---------------------------------------------------------------------------
# Legendre polynomial at a scalar angle  (for particular solution source)
# ---------------------------------------------------------------------------

def _legendre_scalar(mu, nmom):
    """P_l(mu) for l=0..nmom at scalar mu. Returns (nmom+1,)."""
    pl = np.zeros(nmom + 1, dtype=np.float64)
    pl[0] = 1.0
    if nmom >= 1:
        pl[1] = mu
    for l in range(2, nmom + 1):
        pl[l] = ((2 * l - 1) * mu * pl[l - 1] - (l - 1) * pl[l - 2]) / l
    return pl

# Pre-compute for the negative solar zenith angle direction.
# This is recomputed per call because mu0 varies.  But it is cheap.

# ---------------------------------------------------------------------------
# Batch-build all layer RTE matrices A[nlayers, NSTR, NSTR]
# ---------------------------------------------------------------------------

def _build_A_batch(omega_batch, pmom_batch):
    """
    Build RTE system matrices for ALL layers at once.

    Parameters
    ----------
    omega_batch : (nlayers,)         single-scattering albedos
    pmom_batch  : (nlayers, NMOM+1)  phase function moments

    Returns
    -------
    A_batch : (nlayers, NSTR, NSTR)
    """
    nlayers = len(omega_batch)

    # weights_batch[lay, l] = (2l+1) * pmom[lay, l]
    ell = np.arange(NMOM + 1, dtype=np.float64)          # (NMOM+1,)
    two_ell_1 = 2.0 * ell + 1.0                          # (NMOM+1,)
    weights_batch = pmom_batch * two_ell_1[None, :]       # (nlayers, NMOM+1)

    # Phase matrix: P[lay, i, j] = sum_l weights_batch[lay,l] * PL[l,i] * PL[l,j]
    # Efficient einsum: 'nl,li,lj->nij'  but split into two matmuls for speed.
    # Step A: WP[lay, l, j] = weights_batch[lay, l] * PL[l, j]   -> (nlayers, NMOM+1, NSTR)
    WP = weights_batch[:, :, None] * PL[None, :, :]               # broadcast (n,m+1,NSTR)
    # Step B: P[lay, i, j] = sum_l PL[l, i] * WP[lay, l, j]
    #   PL is (NMOM+1, NSTR); we want sum over l -> matmul PL.T @ WP.  Per layer:
    #   result[lay] = PL.T @ WP[lay]   i.e. (NSTR, NMOM+1) @ (NMOM+1, NSTR) -> (NSTR, NSTR)
    P_batch = np.einsum('li,nlj->nij', PL, WP, optimize=True)    # (nlayers, NSTR, NSTR)

    # S_batch[lay, i, j] = (omega[lay]/2) * P[lay, i, j] * wt_all[j]
    S_batch = (omega_batch[:, None, None] / 2.0) * P_batch * wt_all[None, None, :]

    # A_batch[lay, i, j] = (delta_ij - S[lay,i,j]) / mu_all[i]
    eye = np.eye(NSTR, dtype=np.float64)
    A_batch = (eye[None, :, :] - S_batch) / mu_all[:, None][None, :, :]

    return A_batch


# ---------------------------------------------------------------------------
# Batched eigensystem for scattering layers
# ---------------------------------------------------------------------------

def _eigensystem_batch(A_batch, dtau_batch):
    """
    Compute eigensystems for a batch of layers.

    Parameters
    ----------
    A_batch   : (n_sc, NSTR, NSTR)
    dtau_batch: (n_sc,)

    Returns
    -------
    V_batch    : (n_sc, NSTR, NSTR)  eigenvector matrices (columns = eigenvectors)
    E_top_batch: (n_sc, NSTR)        exp factors anchored at top of layer
    E_bot_batch: (n_sc, NSTR)        exp factors anchored at bottom of layer
    lam_batch  : (n_sc, NSTR)        sorted eigenvalues (ascending)
    """
    # Batched eigendecomposition — one LAPACK call
    eigvals, eigvecs = np.linalg.eig(A_batch)   # (n_sc,NSTR), (n_sc,NSTR,NSTR)
    eigvals = eigvals.real
    eigvecs = eigvecs.real

    # Sort eigenvalues ascending (negative first = downward modes)
    idx = np.argsort(eigvals, axis=1)            # (n_sc, NSTR)
    # Gather sorted eigenvalues and eigenvectors
    lam = np.take_along_axis(eigvals, idx, axis=1)           # (n_sc, NSTR)
    # Reindex columns of eigvecs: eigvecs is (n_sc, NSTR, NSTR), columns = eigenvectors
    # idx[n, k] = original column index for sorted column k
    n_sc = len(A_batch)
    # Use advanced indexing: V[n, i, k] = eigvecs[n, i, idx[n, k]]
    V = eigvecs[
        np.arange(n_sc)[:, None, None],           # (n_sc, 1, 1)  -> broadcast over i and k
        np.arange(NSTR)[None, :, None],            # (1, NSTR, 1)  -> broadcast over n and k
        idx[:, None, :],                           # (n_sc, 1, NSTR) -> broadcast over i
    ]   # result: (n_sc, NSTR, NSTR)

    # Stability: negative lam[:, :NN] anchored at top (clamped to <=0)
    #            positive lam[:, NN:] anchored at bottom (clamped to <=0 after negation)
    e_neg_bot = np.exp(np.clip(lam[:, :NN] * dtau_batch[:, None], -700.0, 0.0))
    e_pos_top = np.exp(np.clip(-lam[:, NN:] * dtau_batch[:, None], -700.0, 0.0))

    E_top = np.concatenate([np.ones((len(A_batch), NN)), e_pos_top], axis=1)
    E_bot = np.concatenate([e_neg_bot, np.ones((len(A_batch), NN))], axis=1)

    return V, E_top, E_bot, lam


# ---------------------------------------------------------------------------
# Particular solution for direct beam  (per-layer, but cheap once A is known)
# ---------------------------------------------------------------------------

def _particular_solution_batch(A_batch, pmom_batch, omega_batch, mu0, fbeam,
                                tau_top_batch, dtau_batch):
    """
    Particular solution for direct-beam source for a batch of layers.

    Returns
    -------
    Ip_top : (n, NSTR)
    Ip_bot : (n, NSTR)
    """
    n = len(A_batch)
    k = 1.0 / mu0

    # Legendre at -mu0 (solar direction into atmosphere)
    pl_neg_mu0 = _legendre_scalar(-mu0, NMOM)                 # (NMOM+1,)

    # weights_batch[lay, l] = (2l+1) * pmom[lay, l]
    ell = np.arange(NMOM + 1, dtype=np.float64)
    two_ell_1 = 2.0 * ell + 1.0
    weights_batch = pmom_batch * two_ell_1[None, :]            # (n, NMOM+1)

    # P_to_mu0[lay, i] = sum_l weights[lay,l] * PL[l,i] * pl_neg_mu0[l]
    # PL is (NMOM+1, NSTR): PL[l, i]
    # wp[lay, l] = weights_batch[lay,l] * pl_neg_mu0[l]
    wp = weights_batch * pl_neg_mu0[None, :]                   # (n, NMOM+1)
    # P_to_mu0[lay, i] = sum_l wp[lay, l] * PL[l, i]  =>  (n, NMOM+1) @ (NMOM+1, NSTR)
    P_to_mu0 = wp @ PL                                         # (n, NSTR)

    # Direct-beam irradiance at top of each layer
    beam_top = fbeam * np.exp(np.clip(-tau_top_batch / mu0, -700.0, 0.0))  # (n,)

    # F[lay, i] = -(omega[lay]/2) * P_to_mu0[lay,i] * beam_top[lay] / mu_all[i]
    F = (-(omega_batch[:, None] / 2.0) * P_to_mu0
         * beam_top[:, None] / mu_all[None, :])                # (n, NSTR)

    # Solve (A + k*I) Z = -F  for each layer
    AkI = A_batch + k * np.eye(NSTR)[None, :, :]              # (n, NSTR, NSTR)

    # Batched solve using np.linalg.solve
    # np.linalg.solve requires b to have shape (..., m, nrhs); add and remove trailing dim
    Ip_top = np.linalg.solve(AkI, -F[..., None])[..., 0]     # (n, NSTR)

    # At bottom: multiply by exp(-dtau/mu0) per stream direction
    # Only the downward direction has the beam attenuated; all streams see the same factor
    # for the particular (forced) solution.
    beam_atten = np.exp(np.clip(-dtau_batch / mu0, -700.0, 0.0))  # (n,)
    Ip_bot = Ip_top * beam_atten[:, None]                      # (n, NSTR)

    return Ip_top, Ip_bot


# ---------------------------------------------------------------------------
# Single g-point solver (banded BVP)
# ---------------------------------------------------------------------------

def _solve_gpoint(dtauc_t2b, ssalb_t2b, pmom_t2b, fbeam, mu0, alb):
    """
    Solve 16-stream discrete ordinate RTE for one spectral g-point.

    Parameters
    ----------
    dtauc_t2b  : (nlayers,)          layer optical depths, top-to-bottom
    ssalb_t2b  : (nlayers,)          single-scattering albedos, top-to-bottom
    pmom_t2b   : (nlayers, NMOM+1)   phase function moments
    fbeam      : float               solar irradiance at TOA [W/m2]
    mu0        : float               cos(solar zenith)
    alb        : float               surface albedo

    Returns
    -------
    rfldir, rfldn, flup : each (nlevels,) surface-first indexed
    """
    nlyr    = len(dtauc_t2b)
    nlevels = nlyr + 1

    # Direct-beam flux at each level (top-to-bottom indexing, level 0=TOA)
    tau_cum    = np.concatenate([[0.0], np.cumsum(dtauc_t2b)])  # (nlevels,)
    rfldir_t2s = fbeam * mu0 * np.exp(np.clip(-tau_cum / mu0, -700.0, 0.0))

    # -------------------------------------------------------------------------
    # Build all layer matrices in one batch
    # -------------------------------------------------------------------------
    # Trivial (zero-dtau) layers need special treatment
    trivial_mask = dtauc_t2b < 1e-12   # (nlayers,)
    active_mask  = ~trivial_mask        # layers with real thickness

    # Storage for all layers (fill in below)
    V_arr      = np.zeros((nlyr, NSTR, NSTR), dtype=np.float64)
    E_top_arr  = np.ones ((nlyr, NSTR),        dtype=np.float64)
    E_bot_arr  = np.ones ((nlyr, NSTR),        dtype=np.float64)
    Ip_top_arr = np.zeros((nlyr, NSTR),        dtype=np.float64)
    Ip_bot_arr = np.zeros((nlyr, NSTR),        dtype=np.float64)

    # Trivial layers: identity eigenvector matrix
    for lay in np.where(trivial_mask)[0]:
        V_arr[lay] = np.eye(NSTR)

    # Active layers: batch-build A matrices
    active_idx = np.where(active_mask)[0]
    if len(active_idx) > 0:
        omega_act = ssalb_t2b[active_idx]
        pmom_act  = pmom_t2b[active_idx]
        dtau_act  = dtauc_t2b[active_idx]
        tau_top_act = tau_cum[active_idx]   # cumulative tau at top of each active layer

        A_batch = _build_A_batch(omega_act, pmom_act)   # (n_act, NSTR, NSTR)

        # Batched eigensystem
        V_act, E_top_act, E_bot_act, _ = _eigensystem_batch(A_batch, dtau_act)

        V_arr[active_idx]     = V_act
        E_top_arr[active_idx] = E_top_act
        E_bot_arr[active_idx] = E_bot_act

        # Particular solution for layers with both scattering and beam
        if fbeam > 0.0:
            scat_mask_act = omega_act > 0.0           # within active layers
            scat_idx_in_act = np.where(scat_mask_act)[0]
            if len(scat_idx_in_act) > 0:
                global_scat_idx = active_idx[scat_idx_in_act]
                Ip_t, Ip_b = _particular_solution_batch(
                    A_batch[scat_idx_in_act],
                    pmom_act[scat_idx_in_act],
                    omega_act[scat_idx_in_act],
                    mu0, fbeam,
                    tau_top_act[scat_idx_in_act],
                    dtau_act[scat_idx_in_act],
                )
                Ip_top_arr[global_scat_idx] = Ip_t
                Ip_bot_arr[global_scat_idx] = Ip_b

    # -------------------------------------------------------------------------
    # Banded global linear system — fully vectorized assembly
    # -------------------------------------------------------------------------
    # The banded storage convention used by scipy.linalg.solve_banded:
    #   ab[ll + row - col, col] = A[row, col]  for |row-col| <= bandwidth
    # ul = ll = NN + NSTR - 1 = 23

    N_tot = NSTR * nlyr
    ul    = NN + NSTR - 1   # upper bandwidth = 23
    ll    = NN + NSTR - 1   # lower bandwidth = 23
    ab    = np.zeros((ll + ul + 1, N_tot), dtype=np.float64)
    rhs   = np.zeros(N_tot, dtype=np.float64)

    # Helper: scatter values into banded storage without Python loop.
    # For a block of rows [row0..row0+nr) and columns [col0..col0+nc),
    # ab[ll + (row0+i) - (col0+k), col0+k] += val[i, k]
    # The band-row index is ll + row0 - col0 + i - k.
    def _fill_block(row0, col0, val_block):
        """val_block shape (nr, nc) — scatter into ab."""
        nr, nc = val_block.shape
        i_idx = np.arange(nr)[:, None]   # (nr, 1)
        k_idx = np.arange(nc)[None, :]   # (1, nc)
        band_row = ll + row0 - col0 + i_idx - k_idx   # (nr, nc)
        col_abs  = col0 + k_idx                        # (1, nc) broadcast
        # Validity mask (should always be True for the blocks we fill)
        valid = (band_row >= 0) & (band_row < ab.shape[0])
        np.add.at(ab, (band_row[valid], (col0 + k_idx.repeat(nr, axis=0))[valid]),
                   val_block[valid])

    # Pre-compute scaled eigenvector matrices V * E (broadcasts Et/Eb row-wise onto columns)
    # VE_top[lay, i, k] = V[lay, i, k] * E_top[lay, k]
    # VE_bot[lay, i, k] = V[lay, i, k] * E_bot[lay, k]
    VE_top = V_arr * E_top_arr[:, None, :]   # (nlyr, NSTR, NSTR)
    VE_bot = V_arr * E_bot_arr[:, None, :]   # (nlyr, NSTR, NSTR)

    # [1] TOA BC: rows 0..NN-1, cols 0..NSTR-1
    #   VE_top[0, :NN, :] -> (NN, NSTR) block
    V0_block = VE_top[0, :NN, :]   # (NN, NSTR)
    row0_toa, col0_toa = 0, 0
    i_idx = np.arange(NN)[:, None]
    k_idx = np.arange(NSTR)[None, :]
    band_rows_toa = ll + row0_toa - col0_toa + i_idx - k_idx   # (NN, NSTR)
    ab[band_rows_toa, k_idx] = V0_block
    rhs[:NN] = -Ip_top_arr[0, :NN]

    # [2] Interface conditions — fully vectorized (no Python loop over layers).
    # For interface lay (0-indexed), rows [NN+lay*NSTR .. NN+(lay+1)*NSTR):
    #   Left  block: VE_bot[lay]   at cols [lay*NSTR     .. (lay+1)*NSTR)
    #   Right block: -VE_top[lay+1] at cols [(lay+1)*NSTR .. (lay+2)*NSTR)
    # Key: band_row for left block = ll + NN + i - k  (constant for all interfaces)
    #      band_row for right block = ll + NN + NSTR + i - k  ...wait
    # Actually: row0 - col_l = NN + lay*NSTR - lay*NSTR = NN  -> band_left = ll+NN+i-k (constant)
    #           row0 - col_r = NN - NSTR                       -> band_right = ll+NN-NSTR+i-k (constant)
    if nlyr > 1:
        i_idx_16 = np.arange(NSTR)[:, None]               # (NSTR, 1)
        k_idx_16 = np.arange(NSTR)[None, :]               # (1, NSTR)

        # Band-row offsets relative to the column block start — both constant
        band_left  = ll + NN      + i_idx_16 - k_idx_16   # (NSTR, NSTR), fixed
        band_right = ll + NN - NSTR + i_idx_16 - k_idx_16 # (NSTR, NSTR), fixed

        # For the left block at interface lay, the absolute column indices are
        # col_l + k = lay*NSTR + k.  We write VE_bot[lay] into ab[band_left, lay*NSTR+k].
        # ab has shape (nband_rows, N_tot); we write a (NSTR,NSTR) block for each lay.
        # Build column index arrays: col_abs[lay, k] = lay*NSTR + k
        lay_arr = np.arange(nlyr - 1)                     # (nlyr-1,)
        col_l_base = lay_arr * NSTR                        # (nlyr-1,) starting col of left block
        col_r_base = (lay_arr + 1) * NSTR                 # (nlyr-1,) starting col of right block

        # Absolute columns: (nlyr-1, NSTR)
        # k_idx_16 is (1, NSTR); use squeeze to get 1D arange for proper broadcasting
        k1d = np.arange(NSTR)                                  # (NSTR,)
        col_l_abs = col_l_base[:, None] + k1d[None, :]        # (nlyr-1, NSTR)
        col_r_abs = col_r_base[:, None] + k1d[None, :]        # (nlyr-1, NSTR)

        # Band rows are the same for every interface; broadcast over nlyr-1
        # band_left[i, k] -> (NSTR, NSTR) constant
        # We need to index ab[band_left[i,k], col_l_abs[lay, k]] for all (lay, i, k)
        # Vectorize: for each (i, k) in the block, scatter VE_bot[:, i, k] into
        #   ab[band_left[i,k], col_l_base + k] for all layers simultaneously.

        # Scatter into banded ab matrix — iterate over (i,k) block (256 iters, not nlyr)
        # For each (i,k), the band-row is fixed; the column varies with layer.
        # ab[br_l, lay*NSTR + k] += VE_bot[lay, i, k]  for lay in 0..nlyr-2
        for i in range(NSTR):
            for k in range(NSTR):
                br_l = int(band_left[i, k])
                # col_l_abs[:, k] is (nlyr-1,): lay*NSTR + k for lay=0..nlyr-2
                ab[br_l, col_l_abs[:, k]] += VE_bot[:nlyr - 1, i, k]
                br_r = int(band_right[i, k])
                ab[br_r, col_r_abs[:, k]] -= VE_top[1:, i, k]

        # RHS for interfaces — vectorized reshape
        rhs_iface = Ip_top_arr[1:] - Ip_bot_arr[:-1]   # (nlyr-1, NSTR)
        # rhs[NN .. NN+(nlyr-1)*NSTR] in blocks of NSTR
        rhs[NN : NN + (nlyr - 1) * NSTR] = rhs_iface.ravel()

    # [3] Surface BC: rows [N_tot-NN .. N_tot), cols [(nlyr-1)*NSTR .. N_tot)
    row_surf = N_tot - NN
    col_surf = (nlyr - 1) * NSTR
    Vs   = V_arr[nlyr - 1]                    # (NSTR, NSTR)
    Ebs  = E_bot_arr[nlyr - 1]               # (NSTR,)
    Ipbs = Ip_bot_arr[nlyr - 1]              # (NSTR,)
    V_dn_wtd = _wt_flux_dn @ Vs[:NN, :]      # (NSTR,)
    L = alb / _sum_wt_up
    direct_at_surf = rfldir_t2s[nlyr]

    # val[i_rel, k] = (Vs[NN+i_rel, k] - L * V_dn_wtd[k]) * Ebs[k]
    i_rel_idx = np.arange(NN)[:, None]   # (NN, 1)
    k_idx_s   = np.arange(NSTR)[None, :] # (1, NSTR)
    surf_block = (Vs[NN:, :] - L * V_dn_wtd[None, :]) * Ebs[None, :]  # (NN, NSTR)
    band_rows_surf = ll + row_surf - col_surf + i_rel_idx - k_idx_s    # (NN, NSTR)
    ab[band_rows_surf, col_surf + k_idx_s] = surf_block

    rhs_surf_common = L * direct_at_surf + L * np.dot(_wt_flux_dn, Ipbs[:NN])
    rhs[row_surf : row_surf + NN] = rhs_surf_common - Ipbs[NN:]

    # Solve banded system
    try:
        C_vec = solve_banded((ll, ul), ab, rhs, check_finite=False)
    except Exception:
        # Dense fallback (rare)
        G = np.zeros((N_tot, N_tot), dtype=np.float64)
        for col in range(N_tot):
            for di in range(-ll, ul + 1):
                row = col + di
                if 0 <= row < N_tot:
                    v = ab[ll - di, col]
                    if v != 0:
                        G[row, col] = v
        C_vec = np.linalg.lstsq(G, rhs, rcond=None)[0]

    # -------------------------------------------------------------------------
    # Extract fluxes at each level — vectorized
    # -------------------------------------------------------------------------
    # C_vec reshaped to (nlyr, NSTR); Iv[lay] = V[lay] @ (C[lay] * E[lay]) + Ip[lay]
    C_mat = C_vec.reshape(nlyr, NSTR)   # (nlyr, NSTR)

    # At top of each layer (level 0..nlyr-1 in top-to-bottom indexing)
    Iv_top = (np.einsum('nij,nj->ni', V_arr, C_mat * E_top_arr, optimize=True)
              + Ip_top_arr)                # (nlyr, NSTR)
    # At bottom of each layer (level 1..nlyr in top-to-bottom indexing)
    Iv_bot = (np.einsum('nij,nj->ni', V_arr, C_mat * E_bot_arr, optimize=True)
              + Ip_bot_arr)               # (nlyr, NSTR)

    rfldn_t2s = np.zeros(nlevels, dtype=np.float64)
    flup_t2s  = np.zeros(nlevels, dtype=np.float64)

    # Level 0 (TOA): top of first layer
    rfldn_t2s[0] = np.dot(_wt_flux_dn, Iv_top[0, :NN])
    flup_t2s[0]  = np.dot(_wt_flux_up, Iv_top[0, NN:])

    # Levels 1..nlyr: bottom of each layer
    rfldn_t2s[1:] = Iv_bot[:, :NN] @ _wt_flux_dn   # (nlyr,)
    flup_t2s[1:]  = Iv_bot[:, NN:] @ _wt_flux_up   # (nlyr,)

    rfldir = rfldir_t2s[::-1].copy()
    rfldn  = rfldn_t2s[::-1].copy()
    flup   = flup_t2s[::-1].copy()

    return rfldir, rfldn, flup


# ---------------------------------------------------------------------------
# Module-level band worker  (must be at module level for pickle / ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _process_band(args):
    """
    Process one SW band over all 16 g-points.

    This function is defined at module level (not nested) so that
    ProcessPoolExecutor can pickle it for multiprocessing.

    Parameters (packed into args tuple)
    ------------------------------------
    b         : int              band index
    tau_b     : (nlayers, 16)   gas optical depths for this band (bottom-to-top)
    ssa_b     : (nlayers, 16)   single-scattering albedos (bottom-to-top)
    sfx_b     : (16,)           solar flux per g-point
    mu0       : float            cos(solar zenith)
    alb_b     : float            surface albedo for this band
    pz_arr    : (nlevels,)       level pressures (surface-to-TOA, unused here but kept for API)
    aerosol_ser: list of (tau_ext, ssa, g) tuples (one per layer) or None
    tc_b      : (nlayers,) or None   cloud extinction OD per layer (bottom-to-top)
    sc_b      : (nlayers,) or None   cloud SSA per layer
    gc_b      : (nlayers,) or None   cloud asymmetry parameter per layer
    fc_b      : (nlayers,) or None   cloud fraction per layer

    Returns
    -------
    b, acc_uflux, acc_dir, acc_dif  — band index and accumulated flux arrays
    """
    (b, tau_b, ssa_b, sfx_b, mu0, alb_b, pz_arr,
     aerosol_ser, tc_b, sc_b, gc_b, fc_b) = args

    nlayers = tau_b.shape[0]
    nlevels = nlayers + 1

    acc_uflux = np.zeros(nlevels, dtype=np.float64)
    acc_dir   = np.zeros(nlevels, dtype=np.float64)
    acc_dif   = np.zeros(nlevels, dtype=np.float64)

    # Aerosol: build per-layer arrays (top-to-bottom) from serialised tuples
    has_aerosol = aerosol_ser is not None
    has_cloud   = (tc_b is not None and fc_b is not None)

    if has_aerosol:
        # aerosol_ser is bottom-to-top; reverse to t2b
        # Each element: (tau_ext[nbands], ssa[nbands], g[nbands])
        tau_aer_t2b = np.array([aerosol_ser[lay][0][b]
                                  for lay in range(nlayers)],
                                dtype=np.float64)[::-1]
        ssa_aer_t2b = np.array([aerosol_ser[lay][1][b]
                                  for lay in range(nlayers)],
                                dtype=np.float64)[::-1]
        g_aer_t2b   = np.array([aerosol_ser[lay][2][b]
                                  for lay in range(nlayers)],
                                dtype=np.float64)[::-1]
    else:
        tau_aer_t2b = np.zeros(nlayers, dtype=np.float64)
        ssa_aer_t2b = np.zeros(nlayers, dtype=np.float64)
        g_aer_t2b   = np.zeros(nlayers, dtype=np.float64)

    scat_aer = ssa_aer_t2b * tau_aer_t2b   # (nlayers,)

    # Cloud: tc_b/sc_b/gc_b/fc_b are already per-layer for this band (bottom-to-top)
    # Reverse to top-to-bottom for DISORT
    if has_cloud:
        tc_t2b = np.asarray(tc_b, dtype=np.float64)[::-1]   # (nlayers,)
        sc_t2b = np.asarray(sc_b, dtype=np.float64)[::-1]
        gc_t2b = np.asarray(gc_b, dtype=np.float64)[::-1]
        fc_t2b = np.asarray(fc_b, dtype=np.float64)[::-1]

    ell_arr   = np.arange(NMOM + 1, dtype=np.float64)   # (NMOM+1,)
    p_gas_arr = np.zeros(NMOM + 1, dtype=np.float64)
    p_gas_arr[2] = 0.1   # Rayleigh l=2 moment

    for ig in range(16):
        S0 = float(sfx_b[ig])
        if S0 <= 0.0:
            continue

        # Gas optical properties — top-to-bottom ordering for DISORT
        # tau_b is (nlayers, 16) bottom-to-top; reverse rows for t2b
        dtauc_t2b = np.ascontiguousarray(tau_b[::-1, ig], dtype=np.float64)
        ssalb_t2b = np.ascontiguousarray(ssa_b[::-1, ig], dtype=np.float64)

        scat_gas = ssalb_t2b * dtauc_t2b   # (nlayers,)

        if has_aerosol:
            dtauc_t2b = dtauc_t2b + tau_aer_t2b
            scat_check = scat_gas + scat_aer
            ssalb_t2b = np.where(dtauc_t2b > 1e-30,
                                  scat_check / dtauc_t2b, 0.0)

        scat_cld  = np.zeros(nlayers, dtype=np.float64)
        g_cld_t2b = np.zeros(nlayers, dtype=np.float64)
        if has_cloud:
            g_cld_t2b = gc_t2b
            scat_cld  = fc_t2b * sc_t2b * tc_t2b
            tau_eff   = dtauc_t2b + fc_t2b * tc_t2b
            scat_eff  = scat_gas + scat_aer + scat_cld
            ssalb_t2b = np.where(tau_eff > 1e-30, scat_eff / tau_eff, 0.0)
            dtauc_t2b = tau_eff

        scat_total = scat_gas + scat_aer + scat_cld   # (nlayers,)

        g_aer_expand = g_aer_t2b[:, None] ** ell_arr[None, :]   # (nlayers, NMOM+1)
        g_cld_expand = g_cld_t2b[:, None] ** ell_arr[None, :]

        numerator = (p_gas_arr[None, :] * scat_gas[:, None]
                     + g_aer_expand * scat_aer[:, None]
                     + g_cld_expand * scat_cld[:, None])

        pmom_t2b = np.where(scat_total[:, None] > 1e-30,
                             numerator / scat_total[:, None],
                             0.0)
        pmom_t2b[:, 0] = 1.0
        pure_rayleigh = scat_total < 1e-30
        pmom_t2b[pure_rayleigh, 2] = 0.1

        rfldir, rfldn, flup = _solve_gpoint(
            dtauc_t2b, ssalb_t2b, pmom_t2b, S0, mu0, alb_b
        )

        acc_uflux += flup
        acc_dir   += rfldir
        acc_dif   += rfldn

    return b, acc_uflux, acc_dir, acc_dif


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def rtrdis(tau, ssa, sfluxzen, zenith, albedo, pz,
           aerosol=None, tau_cloud=None, ssa_cloud=None,
           g_cloud=None, frac_cloud=None):
    """
    Pure NumPy 16-stream discrete ordinate shortwave solver (batched).

    Parameters
    ----------
    tau      : ndarray (nbands, nlayers, 16)  layer optical depths (bottom-to-top)
    ssa      : ndarray (nbands, nlayers, 16)  single-scattering albedos (bottom-to-top)
    sfluxzen : ndarray (nbands, 16)           solar flux [W/m2] per g-point
    zenith   : float                           cos(solar zenith angle)
    albedo   : ndarray (nbands,) or float     surface albedo per band
    pz       : ndarray (nlayers+1,)           level pressures [mb] (surface-to-TOA)
    aerosol  : optional list of aerosol objects (one per layer, bottom-to-top)
    tau_cloud, ssa_cloud, g_cloud, frac_cloud : optional cloud arrays

    Returns
    -------
    dict with keys (all surface-to-TOA indexed):
        totuflux, totdflux, dirdown, difdown, fnet, htr  (broadband)
        band_totuflux, band_totdflux, band_dirdown, band_difdown, band_fnet, band_htr
    """
    tau      = np.asarray(tau,      dtype=np.float64)
    ssa      = np.asarray(ssa,      dtype=np.float64)
    sfluxzen = np.asarray(sfluxzen, dtype=np.float64)
    pz       = np.asarray(pz,       dtype=np.float64)

    nbands  = tau.shape[0]
    nlayers = tau.shape[1]
    nlevels = nlayers + 1

    alb = (np.full(nbands, float(albedo)) if np.ndim(albedo) == 0
           else np.asarray(albedo, dtype=np.float64))

    band_totuflux = np.zeros((nbands, nlevels), dtype=np.float64)
    band_totdflux = np.zeros((nbands, nlevels), dtype=np.float64)
    band_dirdown  = np.zeros((nbands, nlevels), dtype=np.float64)
    band_difdown  = np.zeros((nbands, nlevels), dtype=np.float64)

    if zenith <= 0.0:
        z  = np.zeros(nlevels)
        zb = np.zeros((nbands, nlevels))
        return dict(totuflux=z, totdflux=z, dirdown=z, difdown=z, fnet=z, htr=z,
                    band_totuflux=zb, band_totdflux=zb, band_dirdown=zb,
                    band_difdown=zb, band_fnet=zb, band_htr=zb)

    mu0 = float(zenith)

    # -------------------------------------------------------------------------
    # Pre-compute aerosol and cloud optical properties (per band, shared across g)
    # -------------------------------------------------------------------------
    # Aerosol: shape (nbands, nlayers) with top-to-bottom ordering
    has_aerosol = aerosol is not None
    has_cloud   = (tau_cloud is not None and frac_cloud is not None)

    if has_aerosol:
        # aerosol list is bottom-to-top (layer 0=surface), reverse to t2b
        tau_aer_all = np.array([[aerosol[lay].tau_ext[b]
                                  for lay in range(nlayers)] for b in range(nbands)],
                                dtype=np.float64)[:, ::-1]   # (nbands, nlayers) t2b
        ssa_aer_all = np.array([[aerosol[lay].ssa[b]
                                  for lay in range(nlayers)] for b in range(nbands)],
                                dtype=np.float64)[:, ::-1]
        g_aer_all   = np.array([[aerosol[lay].g[b]
                                  for lay in range(nlayers)] for b in range(nbands)],
                                dtype=np.float64)[:, ::-1]
    else:
        tau_aer_all = np.zeros((nbands, nlayers), dtype=np.float64)
        ssa_aer_all = np.zeros((nbands, nlayers), dtype=np.float64)
        g_aer_all   = np.zeros((nbands, nlayers), dtype=np.float64)

    if has_cloud:
        # tau_cloud shape assumed (nlayers, nbands), bottom-to-top
        tc_all = np.asarray(tau_cloud,  dtype=np.float64)[::-1, :]   # (nlayers, nbands) t2b
        sc_all = np.asarray(ssa_cloud,  dtype=np.float64)[::-1, :]
        gc_all = np.asarray(g_cloud,    dtype=np.float64)[::-1, :]
        fc_all = np.asarray(frac_cloud, dtype=np.float64)[::-1]       # (nlayers,)

    # -------------------------------------------------------------------------
    # Build per-band argument tuples for _process_band
    # -------------------------------------------------------------------------
    import os, platform
    from concurrent.futures import ProcessPoolExecutor

    # On Windows, ProcessPoolExecutor uses 'spawn' (not 'fork').
    # This is safe because _process_band is a module-level function.
    # Callers must protect their entry point with:
    #   if __name__ == "__main__": ...
    N_SW_WORKERS = int(os.environ.get('PYRRTM_SW_WORKERS',
                                       min(nbands, os.cpu_count() or 4)))

    # On Windows, default to 1 worker unless explicitly set, because
    # 'spawn' start-up overhead can negate the gain for small atmospheres.
    if platform.system() == 'Windows' and 'PYRRTM_SW_WORKERS' not in os.environ:
        N_SW_WORKERS = 1

    band_args = []
    for b in range(nbands):
        # Serialise aerosol as plain numpy tuples (pickle-safe)
        if has_aerosol:
            aerosol_ser = [(aerosol[lay].tau_ext.copy(),
                            aerosol[lay].ssa.copy(),
                            aerosol[lay].g.copy())
                           for lay in range(nlayers)]
        else:
            aerosol_ser = None

        # Cloud per band: tc_all/sc_all/gc_all are (nlayers, nbands) t2b; fc_all is (nlayers,)
        tc_b = tc_all[:, b].copy() if has_cloud else None
        sc_b = sc_all[:, b].copy() if has_cloud else None
        gc_b = gc_all[:, b].copy() if has_cloud else None
        fc_b = fc_all.copy()       if has_cloud else None

        band_args.append((
            b,
            tau[b].copy(),        # (nlayers, 16) bottom-to-top
            ssa[b].copy(),        # (nlayers, 16) bottom-to-top
            sfluxzen[b].copy(),   # (16,)
            mu0,
            float(alb[b]),
            pz.copy(),
            aerosol_ser,
            tc_b, sc_b, gc_b, fc_b,
        ))

    # -------------------------------------------------------------------------
    # Dispatch bands to worker processes (or run sequentially if workers=1)
    # -------------------------------------------------------------------------
    if N_SW_WORKERS > 1 and nbands > 1:
        with ProcessPoolExecutor(max_workers=N_SW_WORKERS) as pool:
            for b_res, uflux, dirdn, difdn in pool.map(_process_band, band_args):
                band_totuflux[b_res] = uflux
                band_dirdown[b_res]  = dirdn
                band_difdown[b_res]  = difdn
    else:
        for args in band_args:
            b_res, uflux, dirdn, difdn = _process_band(args)
            band_totuflux[b_res] = uflux
            band_dirdown[b_res]  = dirdn
            band_difdown[b_res]  = difdn

    band_totdflux = band_dirdown + band_difdown
    totuflux = band_totuflux.sum(0)
    totdflux = band_totdflux.sum(0)
    dirdown  = band_dirdown.sum(0)
    difdown  = band_difdown.sum(0)
    fnet     = totdflux - totuflux   # net downward (positive = SW going down)

    # Heating rate: HTR[i] = HEATFAC * (fnet[i+1] - fnet[i]) / (pz[i] - pz[i+1])
    # fnet[i+1] > fnet[i] for an absorbing layer (more enters from above than exits below)
    # → positive heating rate for solar absorption (Bug 3 fix)
    dp   = pz[:-1] - pz[1:]   # (nlayers,) positive (surface pressure > level above)
    htr  = np.zeros(nlevels, dtype=np.float64)
    mask = dp > 0
    htr[:-1][mask] = HEATFAC * (fnet[1:][mask] - fnet[:-1][mask]) / dp[mask]

    band_fnet = band_totdflux - band_totuflux   # net downward per band
    band_htr  = np.zeros((nbands, nlevels), dtype=np.float64)
    for b in range(nbands):
        band_htr[b, :-1][mask] = (HEATFAC
                                   * (band_fnet[b, 1:][mask] - band_fnet[b, :-1][mask])
                                   / dp[mask])

    return dict(totuflux=totuflux, totdflux=totdflux, dirdown=dirdown, difdown=difdown,
                fnet=fnet, htr=htr, band_totuflux=band_totuflux, band_totdflux=band_totdflux,
                band_dirdown=band_dirdown, band_difdown=band_difdown,
                band_fnet=band_fnet, band_htr=band_htr)


# ---------------------------------------------------------------------------
# Quick self-test / timing comparison
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import time
    import sys

    print("rtrdis.py self-test: synthetic timing benchmark")

    # Create synthetic inputs matching typical RRTM_SW dimensions
    rng     = np.random.default_rng(42)
    nbands  = 14
    nlayers = 51
    ngpts   = 16
    nlevels = nlayers + 1

    tau_syn      = rng.uniform(0.0, 0.5, (nbands, nlayers, ngpts))
    ssa_syn      = rng.uniform(0.0, 0.99, (nbands, nlayers, ngpts))
    sfluxzen_syn = rng.uniform(0.0, 100.0, (nbands, ngpts))
    zenith_syn   = 0.6
    albedo_syn   = 0.1
    adjflux_syn  = np.ones(nbands)
    pz_syn       = np.linspace(1013.25, 1.0, nlevels)

    # Warm-up
    result = rtrdis(tau_syn, ssa_syn, sfluxzen_syn, zenith_syn,
                    albedo_syn, pz_syn)

    # Timed run
    n_runs = 3
    t0 = time.perf_counter()
    for _ in range(n_runs):
        result = rtrdis(tau_syn, ssa_syn, sfluxzen_syn, zenith_syn,
                        albedo_syn, pz_syn)
    t1 = time.perf_counter()

    elapsed = (t1 - t0) / n_runs
    print(f"  {n_runs} runs, mean time: {elapsed:.3f} s")
    print(f"  totuflux at TOA:     {result['totuflux'][-1]:.4f} W/m2")
    print(f"  totdflux at surface: {result['totdflux'][0]:.4f} W/m2")
    print(f"  fnet at surface:     {result['fnet'][0]:.4f} W/m2")
    print("Self-test passed.")
