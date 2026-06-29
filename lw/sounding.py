"""
Internal sounding representation for RRTM-LW.

Converts user-supplied units (°C, hPa, g/kg, ppm) to the internal
RRTM-ready quantities.  All NaN-filling from USSA76 is performed here.
"""

import warnings
import numpy as np

from . import std_atm


class Sounding:
    """
    Atmospheric sounding in RRTM-ready internal units.
    Constructed from user units (°C, hPa, g/kg, ppm).
    All NaN-filling from USSA76 happens here.
    """

    def __init__(
        self,
        z_km,           # (nlevels,) float64, increasing upward
        T_C,            # (nlevels,) temperature [°C]; NaN -> USSA76
        P_hPa=None,     # (nlevels,) pressure [hPa]; None -> hydrostatic from z+T
        wv_gkg=None,    # (nlevels,) water vapor mixing ratio [g/kg]; NaN -> USSA76
        gas_ppm=None,   # dict: keys 'co2','o3','n2o','co','ch4' -> (nlevels,) [ppm]; NaN -> USSA76
        tbound_C=None,  # scalar surface skin temperature [°C]; None -> T_C[0] + warn
        # Cloud fields: NaN -> 0 (clear-sky), NOT USSA76
        frac=None,      # (nlevels,) cloud fraction [0-1]
        lwp_gm2=None,   # (nlevels,) liquid water path [g/m²]
        re_liq_um=None, # (nlevels,) liquid effective radius [µm]
        iwp_gm2=None,   # (nlevels,) ice water path [g/m²]
        re_ice_um=None, # (nlevels,) ice effective radius/dge [µm]
        iceflag=2,      # 2 or 3
    ):
        # ------------------------------------------------------------------
        # Step 1: Validate inputs
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
        # Step 2: Unit conversion (before NaN fill)
        # ------------------------------------------------------------------
        T_K = np.array(T_C, dtype=float) + 273.15

        if tbound_C is None:
            warnings.warn(
                "surface_temperature not supplied; using lowest-level air "
                "temperature as tbound."
            )
            tbound_K = T_K[0]
        else:
            tbound_K = float(tbound_C) + 273.15

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
        # Step 3: NaN fill from USSA76
        # ------------------------------------------------------------------
        fill_input = {'T_K': T_K, 'vmr_h2o': vmr_h2o}
        fill_input.update(gas_vmr)   # keys: 'co2','o3','n2o','co','ch4'

        filled = std_atm.fill_nans(z_km, fill_input)

        T_K     = filled['T_K']
        vmr_h2o = filled['vmr_h2o']
        for g in ('co2', 'o3', 'n2o', 'co', 'ch4'):
            gas_vmr[g] = filled[g]
        gas_vmr['o2'] = np.full(len(z_km), 0.20946)

        # ------------------------------------------------------------------
        # Step 4: Pressure
        # ------------------------------------------------------------------
        if P_hPa is None:
            P = np.empty(len(z_km))
            P[-1] = std_atm.P_std(z_km[-1])   # anchor at top level [hPa]
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
        # Step 5: Layer quantities
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
        coldry = wbroad - wkl[0]
        wkl[1] = vmr_layer['co2'] * coldry
        wkl[2] = vmr_layer['o3']  * coldry
        wkl[3] = vmr_layer['n2o'] * coldry
        wkl[4] = vmr_layer['co']  * coldry
        wkl[5] = vmr_layer['ch4'] * coldry
        wkl[6] = vmr_layer['o2']  * coldry

        # ------------------------------------------------------------------
        # Step 6: Cloud fields (NaN -> 0, clear-sky)
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
        # Step 7: Store attributes
        # ------------------------------------------------------------------
        self.pz      = pz
        self.tz      = tz
        self.pavel   = pavel
        self.tavel   = tavel
        self.wkl     = wkl
        self.wbroad  = wbroad
        self.coldry  = coldry
        self.tbound  = tbound_K
        self.semiss  = np.ones(16)   # IEMIS=0 fixed
        self.nlayers = nlayers
        self.z_km    = z_km

    # ----------------------------------------------------------------------
    def to_dict(self):
        """Return the RRTM-ready fields as a plain dictionary."""
        return dict(
            pz      = self.pz,
            tz      = self.tz,
            pavel   = self.pavel,
            tavel   = self.tavel,
            wkl     = self.wkl,
            wbroad  = self.wbroad,
            coldry  = self.coldry,
            tbound  = self.tbound,
            semiss  = self.semiss,
            nlayers = self.nlayers,
        )
