"""
raman_map_analysis.py

Helper functions for Renishaw Raman/SERS mapping data.

This module is intentionally small and focused on the map workflow. It keeps the
Brolo Group plotting style from the original raman_analysis.py file and adds a
robust map reader, baseline correction, cosmic-ray despiking, and single-band
peak fitting for the 4-MBN nitrile band.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve


# =============================================================================
# Brolo Group plotting style retained from the original raman_analysis.py
# =============================================================================

brolo_style = {
    'figure.figsize': (3.33, 3.33),  # Width will be set and height will adjust
    'font.family': 'Arial',
    'figure.dpi': 600,
    'font.size': 10,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'axes.linewidth': 2,
    'axes.labelweight': 'bold',
    'axes.spines.top': True,
    'axes.spines.right': True,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.grid': False,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'xtick.major.size': 5,
    'xtick.major.width': 2,
    'ytick.major.size': 5,
    'ytick.major.width': 2,
    'xtick.minor.size': 2.5,
    'xtick.minor.width': 1.5,
    'ytick.minor.size': 2.5,
    'ytick.minor.width': 1.5,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'lines.linewidth': 2,
    'errorbar.capsize': 4,
    'lines.markersize': 10,
    'lines.markeredgecolor': 'none',
    'legend.fontsize': 10,
    'legend.frameon': False,
    'legend.loc': 'best',
    'legend.handlelength': 2,
    'legend.borderpad': 0.2,
    'legend.labelspacing': 0.2,
    'legend.handletextpad': 0.2,
    'legend.columnspacing': 0.5,
    'patch.linewidth': 1,
    'patch.edgecolor': 'black',
    'text.usetex': False,
    'axes.labelcolor': 'black',
    'axes.edgecolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'figure.subplot.bottom': 0.15,
    'figure.subplot.left': 0.15,
    'figure.autolayout': True,  # Adjust layout if needed
    'grid.linewidth': 0,
    'axes.unicode_minus': False,  # to allow for minus ticks
}
plt.rcParams.update(brolo_style)


@dataclass
class MapData:
    """Container for a Renishaw map."""

    wn: np.ndarray                  # shape: (n_wavenumbers,)
    spectra: np.ndarray             # shape: (n_points, n_wavenumbers)
    coords: np.ndarray              # shape: (n_points, 2), columns x and y
    x_unique: np.ndarray            # sorted unique x values
    y_unique: np.ndarray            # sorted unique y values

    @property
    def n_points(self) -> int:
        return int(self.spectra.shape[0])

    @property
    def n_wavenumbers(self) -> int:
        return int(self.spectra.shape[1])

    @property
    def grid_shape(self) -> Tuple[int, int]:
        """Return grid shape as (n_y, n_x), matching imshow indexing."""
        return (len(self.y_unique), len(self.x_unique))


def _load_numeric_table(file_path: str) -> np.ndarray:
    """Load a whitespace- or comma-delimited Raman text/CSV table."""
    attempts = [
        dict(comments="#"),
        dict(comments="#", delimiter=","),
        dict(skiprows=1),
        dict(skiprows=1, delimiter=","),
    ]
    last_error: Optional[Exception] = None
    for kwargs in attempts:
        try:
            arr = np.loadtxt(file_path, **kwargs)
            if arr.ndim == 2 and arr.shape[1] >= 4:
                return arr[:, :4].astype(float)
        except Exception as exc:  # keep trying alternate delimiters/header handling
            last_error = exc
    raise ValueError(
        f"Could not read a numeric Raman map table from {file_path!r}. "
        "Expected at least four columns: X, Y, Wave, Intensity."
    ) from last_error


def read_renishaw_map(file_path: str, ensure_ascending: bool = True) -> MapData:
    """Read a Renishaw map export with columns X, Y, Wave, Intensity.

    The attached AgNP/4-MBN map has one contiguous block per spatial point. This
    function is robust to acquisition order: it groups rows by coordinate while
    preserving the first-seen map-point order.

    Parameters
    ----------
    file_path:
        Path to the Renishaw map text/CSV export.
    ensure_ascending:
        If True, spectra are returned with the wavenumber axis ascending.

    Returns
    -------
    MapData
        Wavenumber axis, spectra, coordinates, and map-grid metadata.
    """
    data = _load_numeric_table(file_path)
    x_all = data[:, 0]
    y_all = data[:, 1]
    wave_all = data[:, 2]
    int_all = data[:, 3]

    coord_all = np.column_stack([x_all, y_all])
    unique_coords, first_idx, inverse = np.unique(
        coord_all, axis=0, return_index=True, return_inverse=True
    )
    order = np.argsort(first_idx)
    coords = unique_coords[order]

    spectra = []
    wn_ref = None
    for old_unique_index in order:
        idx = np.where(inverse == old_unique_index)[0]
        # Preserve acquisition order inside the block. If the file has been
        # shuffled, sorting by original row index reconstructs the spectrum.
        idx = np.sort(idx)
        wn_i = wave_all[idx].astype(float)
        y_i = int_all[idx].astype(float)

        if ensure_ascending and wn_i[0] > wn_i[-1]:
            wn_i = wn_i[::-1]
            y_i = y_i[::-1]

        if wn_ref is None:
            wn_ref = wn_i
        else:
            if len(wn_i) != len(wn_ref) or not np.allclose(wn_i, wn_ref, rtol=0, atol=1e-6):
                raise ValueError(
                    "Not all map points share the same wavenumber axis. "
                    "Interpolate to a common axis before averaging/fitting."
                )
        spectra.append(y_i)

    x_unique = np.unique(coords[:, 0])
    y_unique = np.unique(coords[:, 1])

    return MapData(
        wn=np.asarray(wn_ref, dtype=float),
        spectra=np.asarray(spectra, dtype=np.float32),
        coords=np.asarray(coords, dtype=float),
        x_unique=x_unique,
        y_unique=y_unique,
    )


def values_to_grid(coords: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Place one value per coordinate into a 2-D grid for heatmaps.

    Returns
    -------
    grid, x_unique, y_unique
        grid has shape (n_y, n_x). Missing values are NaN.
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    x_unique = np.unique(coords[:, 0])
    y_unique = np.unique(coords[:, 1])

    grid = np.full((len(y_unique), len(x_unique)), np.nan, dtype=float)
    x_index = {x: i for i, x in enumerate(x_unique)}
    y_index = {y: i for i, y in enumerate(y_unique)}

    for (x, y), value in zip(coords, values):
        grid[y_index[y], x_index[x]] = value
    return grid, x_unique, y_unique


def whitaker_hayes_despike(spectrum: np.ndarray, threshold: float = 6.0, window: int = 3) -> np.ndarray:
    """Remove single-pixel cosmic ray spikes using a modified-z score test.

    This mirrors the logic used in run.py: real Raman/SERS bands are broad, while
    CCD spikes are usually very sharp.
    """
    y = np.asarray(spectrum, dtype=float)
    delta = np.diff(y, prepend=y[0])
    med = np.median(delta)
    mad = np.median(np.abs(delta - med))
    z = np.zeros_like(delta) if mad == 0 else 0.6745 * (delta - med) / mad
    spikes = np.abs(z) > threshold

    y_out = y.copy()
    for k in np.flatnonzero(spikes):
        lo, hi = max(k - window, 0), min(k + window + 1, y.size)
        nb = np.arange(lo, hi)
        good = nb[~spikes[nb]]
        if good.size:
            y_out[k] = y[good].mean()
    return y_out


def safe_savgol(y: np.ndarray, window_length: int = 21, polyorder: int = 5) -> np.ndarray:
    """Savitzky-Golay smoothing with automatic adjustment for short spectra."""
    y = np.asarray(y, dtype=float)
    if y.size <= polyorder + 2:
        return y.copy()
    win = min(window_length, y.size if y.size % 2 == 1 else y.size - 1)
    win = max(win, polyorder + 2 + ((polyorder + 2) % 2 == 0))
    if win % 2 == 0:
        win += 1
    if win > y.size:
        win = y.size if y.size % 2 == 1 else y.size - 1
    return savgol_filter(y, window_length=win, polyorder=polyorder, mode="nearest")


def baseline_asls(y: np.ndarray, lam: float = 1e6, p: float = 0.01, niter: int = 10) -> np.ndarray:
    """Asymmetric least-squares baseline correction.

    This is a built-in fallback so the map workflow does not require pybaselines.
    It follows the common Eilers/Boelens ALS baseline approach.
    """
    y = np.asarray(y, dtype=float)
    length = y.size
    if length < 3:
        return np.zeros_like(y)

    d = sparse.eye(length, format="csc")
    d = d[1:] - d[:-1]
    d = d[1:] - d[:-1]
    d = d.T

    w = np.ones(length)
    for _ in range(niter):
        w_matrix = sparse.diags(w, 0, shape=(length, length))
        z_matrix = w_matrix + lam * d.dot(d.T)
        baseline = spsolve(z_matrix, w * y)
        w = p * (y > baseline) + (1.0 - p) * (y < baseline)
    return baseline


def estimate_baseline(y: np.ndarray, lam: float = 1e6, p: float = 0.01) -> np.ndarray:
    """Estimate a spectral baseline.

    If pybaselines is installed, use its arPLS implementation. Otherwise, fall
    back to the local ALS baseline implementation above.
    """
    try:
        import pybaselines as bs  # type: ignore
        return bs.whittaker.arpls(np.asarray(y, dtype=float), lam=lam)[0]
    except Exception:
        return baseline_asls(y, lam=lam, p=p, niter=10)


def preprocess_spectra(
    spectra: np.ndarray,
    despike: bool = True,
    smooth: bool = True,
    savgol_window: int = 21,
    savgol_polyorder: int = 5,
    baseline_lam: float = 1e6,
    baseline_p: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Despike, smooth, and baseline-correct a stack of spectra.

    Returns
    -------
    spectra_processed:
        Despiked/smoothed spectra before baseline correction.
    baselines:
        Estimated baseline for each spectrum.
    spectra_corrected:
        Baseline-corrected spectra.
    """
    spectra = np.asarray(spectra, dtype=float)
    processed = []
    baselines = []
    corrected = []

    for y in spectra:
        y_work = whitaker_hayes_despike(y) if despike else y.astype(float)
        y_work = safe_savgol(y_work, savgol_window, savgol_polyorder) if smooth else y_work
        baseline = estimate_baseline(y_work, lam=baseline_lam, p=baseline_p)
        processed.append(y_work)
        baselines.append(baseline)
        corrected.append(y_work - baseline)

    return (
        np.asarray(processed, dtype=np.float32),
        np.asarray(baselines, dtype=np.float32),
        np.asarray(corrected, dtype=np.float32),
    )


def robust_noise(y: np.ndarray, wn: np.ndarray, peak_min: float, peak_max: float, flank_width: float = 80.0) -> float:
    """Estimate local noise near, but outside, the peak fitting window."""
    y = np.asarray(y, dtype=float)
    wn = np.asarray(wn, dtype=float)
    left = (wn >= peak_min - flank_width) & (wn <= peak_min - 10.0)
    right = (wn >= peak_max + 10.0) & (wn <= peak_max + flank_width)
    mask = left | right
    vals = y[mask]
    if vals.size < 8:
        vals = y
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.std(vals))
    return float(sigma if sigma > 0 else np.nan)


def describe_peak(
    wn: np.ndarray,
    sig: np.ndarray,
    search_min: float,
    search_max: float,
    model: str = "lorentzian",
    fit_linear_baseline: bool = True,
) -> Dict[str, float | str | bool]:
    """Fit one Raman band and return peak properties.

    The default model fits a Lorentzian peak plus an optional local linear
    baseline inside the selected search window. The returned height and area are
    for the peak component only, not for the local baseline.
    """
    wn = np.asarray(wn, dtype=float)
    sig = np.asarray(sig, dtype=float)
    model = model.lower()

    fail = dict(
        center=np.nan,
        center_err=np.nan,
        height=np.nan,
        fwhm=np.nan,
        area=np.nan,
        r_squared=np.nan,
        model=model,
        success=False,
    )

    mask = (wn >= search_min) & (wn <= search_max)
    x = wn[mask]
    y = sig[mask]
    if x.size < 8 or not np.all(np.isfinite(y)):
        return fail

    x0_scale = float(np.mean(x))

    def gaussian_peak(xv, amp, center, sigma):
        return amp * np.exp(-((xv - center) ** 2) / (2.0 * sigma ** 2))

    def lorentzian_peak(xv, amp, center, gamma):
        return amp * gamma ** 2 / ((xv - center) ** 2 + gamma ** 2)

    def voigt_peak(xv, amp, center, sigma, gamma):
        from scipy.special import wofz
        z = ((xv - center) + 1j * gamma) / (sigma * np.sqrt(2.0))
        # normalized Voigt, rescaled so amp is approximately peak height
        v = np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))
        vmax = np.nanmax(v)
        return amp * v / vmax if vmax > 0 else amp * v

    if model == "gaussian":
        peak = gaussian_peak
        peak_param_count = 3
    elif model == "voigt":
        peak = voigt_peak
        peak_param_count = 4
    else:
        model = "lorentzian"
        peak = lorentzian_peak
        peak_param_count = 3

    edge_count = max(2, min(5, x.size // 5))
    y_edges = np.r_[y[:edge_count], y[-edge_count:]]
    offset0 = float(np.median(y_edges))
    slope0 = 0.0
    amp0 = float(max(np.max(y) - offset0, np.max(y), 1.0))
    center0 = float(x[np.argmax(y)])
    width0 = max((search_max - search_min) / 8.0, abs(x[1] - x[0]))

    if fit_linear_baseline:
        if model == "voigt":
            def fit_func(xv, offset, slope, amp, center, sigma, gamma):
                return offset + slope * (xv - x0_scale) + peak(xv, amp, center, sigma, gamma)
            p0 = [offset0, slope0, amp0, center0, width0, width0]
            lower = [-np.inf, -np.inf, 0.0, search_min, 0.0, 0.0]
            upper = [np.inf, np.inf, np.inf, search_max, np.inf, np.inf]
        else:
            def fit_func(xv, offset, slope, amp, center, width):
                return offset + slope * (xv - x0_scale) + peak(xv, amp, center, width)
            p0 = [offset0, slope0, amp0, center0, width0]
            lower = [-np.inf, -np.inf, 0.0, search_min, 0.0]
            upper = [np.inf, np.inf, np.inf, search_max, np.inf]
    else:
        if model == "voigt":
            def fit_func(xv, amp, center, sigma, gamma):
                return peak(xv, amp, center, sigma, gamma)
            p0 = [amp0, center0, width0, width0]
            lower = [0.0, search_min, 0.0, 0.0]
            upper = [np.inf, search_max, np.inf, np.inf]
        else:
            def fit_func(xv, amp, center, width):
                return peak(xv, amp, center, width)
            p0 = [amp0, center0, width0]
            lower = [0.0, search_min, 0.0]
            upper = [np.inf, search_max, np.inf]

    try:
        params, pcov = curve_fit(
            fit_func, x, y, p0=p0, bounds=(lower, upper), maxfev=50000
        )
    except Exception:
        return fail

    y_fit_total = fit_func(x, *params)

    if fit_linear_baseline:
        peak_params = params[2:]
        peak_cov_start = 2
    else:
        peak_params = params
        peak_cov_start = 0

    amp = float(peak_params[0])
    center = float(peak_params[1])

    try:
        perr = np.sqrt(np.diag(pcov))
        center_err = float(perr[peak_cov_start + 1])
    except Exception:
        center_err = np.nan

    if model == "gaussian":
        fwhm = float(2.0 * np.sqrt(2.0 * np.log(2.0)) * peak_params[2])
        peak_curve = gaussian_peak(x, *peak_params)
    elif model == "voigt":
        f_g = 2.0 * np.sqrt(2.0 * np.log(2.0)) * peak_params[2]
        f_l = 2.0 * peak_params[3]
        fwhm = float(0.5346 * f_l + np.sqrt(0.2166 * f_l ** 2 + f_g ** 2))
        peak_curve = voigt_peak(x, *peak_params)
    else:
        fwhm = float(2.0 * peak_params[2])
        peak_curve = lorentzian_peak(x, *peak_params)

    trapz = getattr(np, "trapezoid", None) or np.trapz
    area = float(trapz(peak_curve, x))

    ss_res = float(np.sum((y - y_fit_total) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    return dict(
        center=center,
        center_err=center_err,
        height=amp,
        fwhm=fwhm,
        area=area,
        r_squared=r_squared,
        model=model,
        success=True,
    )


def reconstruct_peak_curve(
    x: np.ndarray,
    center: float,
    height: float,
    fwhm: float,
    model: str = "lorentzian",
) -> np.ndarray:
    """Reconstruct an approximate peak-only curve from stored fit outputs."""
    x = np.asarray(x, dtype=float)
    model = model.lower()
    if model == "gaussian":
        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        return height * np.exp(-((x - center) ** 2) / (2.0 * sigma ** 2))
    # For Voigt, use a Lorentzian approximation for visual QA unless full Voigt
    # parameters were explicitly stored.
    gamma = fwhm / 2.0
    return height * gamma ** 2 / ((x - center) ** 2 + gamma ** 2)


def bootstrap_mean_ci(values: Iterable[float], n_boot: int = 5000, ci: float = 95.0, seed: int = 1) -> Tuple[float, float]:
    """Bootstrap confidence interval for the mean."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(n_boot, arr.size), replace=True)
    means = np.mean(draws, axis=1)
    alpha = (100.0 - ci) / 2.0
    return tuple(np.percentile(means, [alpha, 100.0 - alpha]).astype(float))
