"""
Internal sounding representation for RRTM-SW.

Converts user-supplied units (°C, hPa, g/kg, ppm) to the internal
RRTM-ready quantities.  All NaN-filling from USSA76 is performed here.
SW-specific additions: solar zenith angle, surface albedo, and Earth-Sun
distance correction factor (adjflux).
"""

import numpy as np

from .std_atm import fill_nans, T_std, P_std
from .bands import EARTH_SUN, NBANDS


class SWSounding:
    """
    Atmospheric sounding in RRTM-SW-ready internal units.
    Constructed from user units (°C, hPa, g/kg, ppm).
    All NaN-filling from USSA76 happens here.

    SW extensions over LW:
      - sza_deg  : solar zenith angle [degrees] — required
      - albedo   : surface albedo, scalar or (14,) per-band
      - julday   : Julian day 1–365 for Earth-Sun correction
      - adjflux  : (14,) Earth-Sun distance correction factor per band
    """

    def __init__(
        self,
        z_km,           # (nlevels,) geometric height [km], increasing upward
        T_C,            # (nlevels,) temperature [°C]; NaN -> USSA76
        P_hPa=None,     # (nlevels,) pressure [hPa]; None -> hydrostatic from z+T
        wv_gkg=None,    # (nlevels,) water vapor mixing ratio [g/kg]; NaN -> USSA76
        gas_ppm=None,   # dict: 'co2','o3','n2o','co','ch4' in ppm; NaN -> USSA76
        # SW-specific (required for solar calc):
        sza_deg=None,   # scalar solar zenith angle [degrees]; raise ValueError if None
        albedo=0.1,     # scalar or (14,) surface albedo per SW band
        julday=1,       # Julian day 1-365 for Earth-Sun correction
        # Cloud fields: NaN -> clear-sky
        frac=None,      # (nlevels,) cloud fraction [0-1]; NaN -> 0
        lwp_gm2=None,   # (nlevels,) liquid water path [g/m²]; NaN -> 0
        re_liq_um=None, # (nlevels,) liquid re [µm]; NaN -> 0
        iwp_gm2=None,   # (nlevels,) ice water path [g/m²]; NaN -> 0
        re_ice_um=None, # (nlevels,) ice re/dge [µm]; NaN -> 0
        iceflag=2,
    ):
        # ------------------------------------------------------------------
        # Step 1: Validate sza_deg and compute zenith
        # ------------------------------------------------------------------
        if sza_deg is None:
            raise ValueError(
                "sza_deg (solar zenith angle) is required for SW calculations."
            )
        zenith = np.cos(float(sza_deg) * np.pi / 180.0)
        if zenith < 1.0e-6:
            raise ValueError(
                f"sza_deg={sza_deg} yields cos(sza)={zenith:.6f} <= 0; "
                "sun must be above the horizon (sza_deg < 90)."
            )

        # ------------------------------------------------------------------
        # Step 2: Compute adjflux — Earth-Sun distance correction (14 bands)
        # ------------------------------------------------------------------
        es_factor = EARTH_SUN(int(julday))
        adjflux = np.full(NBANDS, es_factor, dtype=np.float64)

        # ------------------------------------------------------------------
        # Step 3: Store albedo as (14,) array
        # ------------------------------------------------------------------
        albedo_arr = np.asarray(albedo, dtype=np.float64)
        if albedo_arr.ndim == 0:
            albedo_band = np.full(NBANDS, float(albedo_arr), dtype=np.float64)
        else:
            albedo_band = np.broadcast_to(albedo_arr, (NBANDS,)).copy()

        # ------------------------------------------------------------------
        # Step 4: Validate z_km
        # ------------------------------------------------------------------
        z_km = np.array(z_km, dtype=float)
        if len(z_km) < 2:
            raise ValueError("z_km must have at least 2 levels.")
        nlayers = len(z_km) - 1
        if nlayers > 999:
            raise ValueError(
                f"nlayers={nlayers} exceeds maximum of 999."
            )
        if not np.all(np.diff(z_km) > 0):
            raise ValueError("z_km must be strictly increasing.")

        # ------------------------------------------------------------------
        # Step 5: Unit conversion (before NaN fill)
        # ------------------------------------------------------------------
        T_K = np.array(T_C, dtype=float) + 273.15

        # Surface skin temperature: use lowest-level air temperature
        tbound_K = T_K[0]

        # Water vapor: g/kg -> VMR
        if wv_gkg is not None:
            wv_gkg = np.asarray(wv_gkg, dtype=float)
            vmr_h2o = np.where(
                np.isfinite(wv_gkg),
                (wv_gkg / 1000.0) / (0.622 + wv_gkg / 1000.0),
                np.nan,
            )
        else:
            vmr_h2o = np.full(len(z_km), np.nan)

        # Trace gases: ppm -> VMR
        gas_vmr = {}
        for g in ('co2', 'o3', 'n2o', 'co', 'ch4'):
            arr = gas_ppm.get(g) if gas_ppm else None
            if arr is not None:
                arr = np.asarray(arr, dtype=float)
                gas_vmr[g] = np.where(np.isfinite(arr), arr * 1e-6, np.nan)
            else:
                gas_vmr[g] = np.full(len(z_km), np.nan)

        # ------------------------------------------------------------------
        # Step 6: NaN fill from USSA76
        # ------------------------------------------------------------------
        fill_input = {'T_K': T_K, 'vmr_h2o': vmr_h2o}
        fill_input.update(gas_vmr)   # keys: 'co2','o3','n2o','co','ch4'

        filled = fill_nans(z_km, fill_input)

        T_K     = filled['T_K']
        vmr_h2o = filled['vmr_h2o']
        for g in ('co2', 'o3', 'n2o', 'co', 'ch4'):
            gas_vmr[g] = filled[g]
        gas_vmr['o2'] = np.full(len(z_km), 0.20946)

        # ------------------------------------------------------------------
        # Step 7: Pressure
        # ------------------------------------------------------------------
        if P_hPa is None:
            P = np.empty(len(z_km))
            P[-1] = P_std(z_km[-1])   # anchor at top level [hPa]
            for i in range(len(z_km) - 2, -1, -1):
                dz_m  = (z_km[i + 1] - z_km[i]) * 1000.0
                T_avg = (T_K[i] + T_K[i + 1]) / 2.0
                P[i]  = P[i + 1] * np.exp(9.80665 * dz_m / (287.058 * T_avg))
            pz = P   # hPa (= mb)
        else:
            pz = np.array(P_hPa, dtype=float)

        if not (pz[0] > pz[-1]):
            raise ValueError(
                "Pressure profile must be decreasing upward (pz[0] > pz[-1]); "
                "surface must have the highest pressure."
            )

        # ------------------------------------------------------------------
        # Step 8: Layer quantities
        # ------------------------------------------------------------------
        pavel = (pz[:-1] + pz[1:]) / 2.0       # mb, layer mean pressure
        tavel = (T_K[:-1] + T_K[1:]) / 2.0     # K,  layer mean temperature
        tz    = T_K                              # K,  level temperatures

        AVOGAD = 6.02214199e23
        M_DRY  = 28.964     # g/mol
        G      = 980.665    # cm/s²

        # Column air density per layer [molec/cm²]
        # dp [mb] -> dp [dyne/cm²] = dp * 1000
        wbroad = (pz[:-1] - pz[1:]) * 1e3 / G * AVOGAD / M_DRY

        # Layer VMR = average of adjacent levels
        vmr_layer = {}
        for g in ('co2', 'o3', 'n2o', 'co', 'ch4', 'o2'):
            arr = gas_vmr[g]
            vmr_layer[g] = (arr[:-1] + arr[1:]) / 2.0
        vmr_layer['h2o'] = (vmr_h2o[:-1] + vmr_h2o[1:]) / 2.0

        # Gas column amounts [molec/cm²], shape (7, nlayers)
        wkl = np.zeros((7, nlayers))
        wkl[0] = vmr_layer['h2o'] * wbroad
        coldry = wbroad - wkl[0]          # dry-air column excludes H2O
        wkl[1] = vmr_layer['co2'] * coldry
        wkl[2] = vmr_layer['o3']  * coldry
        wkl[3] = vmr_layer['n2o'] * coldry
        wkl[4] = vmr_layer['co']  * coldry
        wkl[5] = vmr_layer['ch4'] * coldry
        wkl[6] = vmr_layer['o2']  * coldry

        # ------------------------------------------------------------------
        # Step 9: Cloud fields (NaN -> 0, clear-sky)
        # ------------------------------------------------------------------
        def _clean(arr, n):
            if arr is not None:
                arr = np.asarray(arr, dtype=float)
                return np.where(np.isfinite(arr), arr, 0.0)
            return np.zeros(n)

        self.frac      = _clean(frac,      nlayers)
        self.lwp_gm2   = _clean(lwp_gm2,   nlayers)
        self.re_liq_um = _clean(re_liq_um, nlayers)
        self.iwp_gm2   = _clean(iwp_gm2,   nlayers)
        self.re_ice_um = _clean(re_ice_um, nlayers)
        self.iceflag   = iceflag

        # ------------------------------------------------------------------
        # Step 10: Store attributes
        # ------------------------------------------------------------------
        self.pz         = pz
        self.tz         = tz
        self.pavel      = pavel
        self.tavel      = tavel
        self.wkl        = wkl
        self.wbroad     = wbroad
        self.coldry     = coldry
        self.tbound     = tbound_K
        self.zenith     = zenith
        self.albedo_band = albedo_band
        self.adjflux    = adjflux
        self.nlayers    = nlayers
        self.z_km       = z_km

    # ----------------------------------------------------------------------
    def to_dict(self):
        """Return the RRTM-SW-ready fields as a plain dictionary."""
        return dict(
            pz         = self.pz,
            tz         = self.tz,
            pavel      = self.pavel,
            tavel      = self.tavel,
            wkl        = self.wkl,
            wbroad     = self.wbroad,
            coldry     = self.coldry,
            tbound     = self.tbound,
            zenith     = self.zenith,
            albedo_band = self.albedo_band,
            adjflux    = self.adjflux,
            nlayers    = self.nlayers,
        )
