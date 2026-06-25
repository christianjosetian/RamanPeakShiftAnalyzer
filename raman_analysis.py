"""raman_analysis.py -- Raman/SERS spot-data helpers.

Slimmed to the two functions used by run.py:

    read_tseries_renishaw(file_path) -> (wn, signal)
        Read a Renishaw ASCII time-series export into a wavenumber axis and
        a (n_frames, n_wavenumbers) float32 array.

    describe_peak(wn, sig, search_min, search_max, model='lorentzian',
                  noise_threshold=None) -> dict
        Fit a single Raman band with lmfit and report
        center / center_err / FWHM / height / area / R-squared / AIC / BIC,
        the fitted curve (x_fit, y_fit), plus an above_noise flag when a
        noise threshold is supplied.

Other helpers (Lumerical/FDTD fitting, FFT correlation, violin plots, map and
spot readers, plotting helpers) live in raman_legacy.py -- import from there if
a script needs them.
"""

import numpy as np


def read_tseries_renishaw(file_path):
    data = np.loadtxt(file_path)
    wn = np.unique(data[:, 1])
    n_wn = len(wn)

    # Validate the file is a whole number of equal-length frames.  A truncated or
    # malformed export would otherwise be silently reshaped, misaligning every
    # downstream spectrum with no error raised.
    if n_wn == 0 or len(data) % n_wn != 0:
        raise ValueError(
            f"{file_path}: {len(data)} rows is not an integer multiple of "
            f"{n_wn} unique wavenumbers -- file may be truncated or malformed."
        )
    n_time = len(data) // n_wn

    # The first frame's wavenumber grid must match the global axis; if it
    # doesn't, the frames were not sampled on a consistent grid.
    if not np.allclose(np.sort(data[:n_wn, 1]), wn, rtol=0.0, atol=1e-6):
        raise ValueError(
            f"{file_path}: per-frame wavenumber grid does not match the "
            f"global axis -- inconsistent sampling."
        )

    signal = np.empty((n_time, n_wn), dtype=np.float32)
    for i in range(n_time):
        block = data[i * n_wn:(i + 1) * n_wn]
        # Sort each frame by its own wavenumber column so the intensities line
        # up with `wn`, whatever order the file stored them in.
        order = np.argsort(block[:, 1])
        signal[i] = block[order, 2]
    return wn, signal


def describe_peak(wn, sig, search_min, search_max, model='lorentzian',
                  noise_threshold=None):
    """Fit a single (possibly shifting) Raman band and describe it.

    Built for experiments where the band MOVES with a control variable (e.g. a
    proton-donor concentration that shifts the peak frequency).  The fit locates
    the peak wherever it sits inside [search_min, search_max] and reports its
    position, width, height, and area for that one spectrum.

    Fit baseline-corrected data.  Prefer the *unsmoothed* (despiked) spectrum:
    Savitzky-Golay smoothing before fitting correlates the noise and biases
    height/FWHM and the R-squared / noise gate.  run.py fits the despiked,
    baseline-corrected spectrum by default (see its --smooth-before-fit flag).

    Parameters
    ----------
    wn : 1-D array
        Wavenumber axis (cm^-1).
    sig : 1-D array
        Intensities for ONE spectrum, same length as wn.  Use baseline-corrected
        data.  For a concentration recorded as a time series, average the frames
        first (higher signal-to-noise) or call this per frame.
    search_min, search_max : float
        Wavenumber bounds of the fit window.  Make it wide enough to contain the
        band across every concentration, but narrow enough to isolate this band
        from its neighbours.
    model : {'lorentzian', 'gaussian', 'voigt'}
        Line shape.  'lorentzian' is the usual Raman default; 'voigt' adds
        Gaussian (instrument) broadening; 'gaussian' is also available.
    noise_threshold : float, optional
        If given, the fitted peak `height` is compared against this value (the
        spot's noise floor) and reported via `above_noise`.  When None, no
        judgement is made and `above_noise` defaults to True.

    Returns
    -------
    dict
        center      : fitted peak position in cm^-1  (the SHIFT you track)
        center_err  : 1-sigma uncertainty on center (cm^-1)
        height      : peak amplitude (intensity at the fitted center)
        fwhm        : full width at half maximum (cm^-1)
        area        : integrated area of the fitted curve over the window
        r_squared   : goodness of fit (1.0 = perfect)
        aic, bic    : lmfit information criteria, for line-shape model selection
        x_fit, y_fit: the fitted curve sampled over the window (for QA overlays;
                      None on failure).  Lets callers plot the TRUE fit instead
                      of reconstructing it from center/height/FWHM.
        above_noise : True if height >= noise_threshold (or no threshold given);
                      False if the peak sits at or below the noise floor, so a
                      noise-level "peak" can be excluded from the shift result
        model       : the line shape used
        success     : True if the fit converged; otherwise False and the numeric
                      fields are NaN (so a bad spectrum never crashes a batch)
    """
    from lmfit.models import GaussianModel, LorentzianModel, VoigtModel

    wn = np.asarray(wn, dtype=float)
    sig = np.asarray(sig, dtype=float)

    # Isolate the fit window.
    mask = (wn >= search_min) & (wn <= search_max)
    x, y = wn[mask], sig[mask]

    fail = dict(center=np.nan, center_err=np.nan, height=np.nan, fwhm=np.nan,
                area=np.nan, r_squared=np.nan, aic=np.nan, bic=np.nan,
                x_fit=None, y_fit=None, above_noise=False, model=model,
                success=False)
    if x.size < 5:
        return fail  # too few points in the window to fit reliably

    # lmfit's peak models expose center, height, fwhm and amplitude (the
    # analytic area) directly, each with a propagated 1-sigma error.
    if model == 'gaussian':
        peak = GaussianModel()
    elif model == 'voigt':
        peak = VoigtModel()
    else:  # lorentzian (default)
        peak = LorentzianModel()

    try:
        # Seed parameters from the data, then constrain them physically and keep
        # the centre inside the search window.  Guesses are clamped to the bounds
        # so lmfit never rejects an out-of-range starting value.
        params = peak.guess(y, x=x)
        params['amplitude'].set(value=max(float(params['amplitude'].value), 1e-12), min=0)
        c0 = min(max(float(params['center'].value), search_min), search_max)
        params['center'].set(value=c0, min=search_min, max=search_max)
        params['sigma'].set(value=max(float(params['sigma'].value), 1e-12), min=0)
        if model == 'voigt':
            # Free the Lorentzian width (gamma defaults to tracking sigma).
            params['gamma'].set(value=float(params['sigma'].value), vary=True, min=0, expr='')
        out = peak.fit(y, params, x=x)
    except Exception:
        return fail  # non-convergence -> report failure instead of crashing

    if not out.success:
        return fail

    def _val(name):
        p = out.params.get(name)
        return float(p.value) if (p is not None and p.value is not None) else np.nan

    def _err(name):
        p = out.params.get(name)
        return float(p.stderr) if (p is not None and p.stderr is not None) else np.nan

    center = _val('center')
    center_err = _err('center')   # 1-sigma, propagated by lmfit
    height = _val('height')       # derived param: intensity at the peak
    fwhm = _val('fwhm')           # derived param: full width at half maximum

    # Area = numerical integral of the fitted curve over the search window.
    # (This is deliberately the windowed area, not lmfit's `amplitude`, which is
    # the analytic integral over an infinite range and so is larger.)
    # np.trapezoid (NumPy >= 2.0) falls back to np.trapz on older NumPy.
    y_fit = out.best_fit
    _trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)
    area = float(_trapz(y_fit, x))

    # Prefer lmfit's own R-squared when available; fall back to the manual form.
    r_squared = getattr(out, 'rsquared', None)
    if r_squared is None:
        ss_res = float(np.sum((y - y_fit) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    else:
        r_squared = float(r_squared)

    aic = float(getattr(out, 'aic', np.nan))
    bic = float(getattr(out, 'bic', np.nan))

    # Signal-detection gate: is the fitted band above the noise floor?
    above_noise = True if noise_threshold is None else bool(
        np.isfinite(height) and height >= noise_threshold)

    return dict(center=center, center_err=center_err, height=height, fwhm=fwhm,
                area=area, r_squared=r_squared, aic=aic, bic=bic,
                x_fit=np.asarray(x, dtype=float), y_fit=np.asarray(y_fit, dtype=float),
                above_noise=above_noise, model=model, success=True)
