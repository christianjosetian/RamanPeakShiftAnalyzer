# =============================================================================
#  run.py  --  Raman / SERS spot-data processing pipeline
# =============================================================================
#
#  AUTHOR
#  ------
#  Christian Tian, Department of Chemistry, University of Victoria, Victoria, B.C., Canada
#
#  PURPOSE
#  -------
#  Reads a folder of Renishaw spot files (time-series or map exports),
#  processes each spot through a full spectral pipeline, and writes peak
#  measurements to disk.
#  Designed for SERS substrate evaluation: signal intensity fluctuates over
#  time at each spot, and peak frequency may shift with a control variable
#  (e.g. proton-donor concentration).
#
#  INPUT
#  -----
#  A directory of Renishaw ASCII exports.  Each file is one measurement spot,
#  either a time-series (3-column) or a spatial map (4-column) export.
#  read_renishaw_spots() (raman_analysis.py) dispatches on the column layout
#  and returns the wavenumber axis and a 2-D array shaped
#  (n_frames, n_wavenumbers) -- map positions are treated as frames.
#  The first 3 edge points of each spectrum are trimmed on load.  Files whose
#  wavenumber axis differs from the first spot are resampled onto the common
#  grid (use --strict-axis to reject them instead).
#
#  PROCESSING PIPELINE  (applied per spot)
#  ----------------------------------------
#  1. Cosmic-ray removal  Whitaker-Hayes modified z-score despiking (local
#                         implementation; the variant in RamanSPy 0.2.10 has
#                         bugs -- see its issue #19).
#  2. Smoothing           Savitzky-Golay, window 21, order 5 (scipy) -- used
#                         for QA display by default; only applied BEFORE fitting
#                         when --smooth-before-fit is set.
#  3. Baseline removal    arPLS by default (fast); spline mixture model
#                         available via --useSplineMixtureBaseline.  Lambda is
#                         estimated per spectrum unless --fixedArplsLambda.
#  4. Noise threshold     2 * std of the quietest wavenumber across frames.
#                         The averaged-spectrum gate is scaled by 1/sqrt(N) to
#                         match the sqrt(N) noise reduction of averaging.
#  5. Peak fitting        Lorentzian (default) / Gaussian / Voigt via
#                         describe_peak() (lmfit) over a search window; returns
#                         center, FWHM, height, area, R2, AIC/BIC and the
#                         above_noise flag.  --fit-mode perframe (default) fits
#                         every frame and combines the per-frame centers with a
#                         robust estimator (--center-agg trimmed by default),
#                         taking shape/area/R2 from the per-spot averaged
#                         spectrum; --fit-mode average fits only the averaged
#                         spectrum (faster, less accurate center).
#
#  OUTPUT FILES  (written to the working directory)
#  ------------------------------------------------
#  spectra_treatment_<label>.png   QA figure: baseline, corrected, threshold
#  fit_qa_<label>_spot<N>.png      per-spot fit-quality figure (one per fitted spot)
#  peak_fits_<label>.csv           per-spot fit results
#  peak_centers_<label>.png        fitted center per spot with error bars
#
#  COMMAND-LINE USAGE
#  ------------------
#  python run.py [DATA_DIR]
#         [--label STR] [--model {lorentzian,gaussian,voigt}]
#         [--useSplineMixtureBaseline] [--fixedArplsLambda]
#         [--peak-min FLOAT] [--peak-max FLOAT]
#         [--fit-mode {average,perframe}]
#         [--center-agg {mean,median,trimmed}] [--trim-fraction FLOAT]
#         [--smooth-before-fit]
#         [--strict-axis]
#         [--qa-spot INT] [--qa-frame INT] [--no-qa-plots]
#         [--parallel] [--jobs INT]
#
#  The despike / baseline / fit stages are serial by default; pass --parallel
#  to run them across worker processes (--jobs sets the worker count).  Each
#  run prints start / end timestamps and per-stage elapsed times so the
#  speed-up from --parallel can be measured.
#
#  DEPENDENCIES
#  ------------
#  numpy, scipy, matplotlib, pybaselines, lmfit
#    (pip install numpy scipy matplotlib pybaselines lmfit)
#  raman_analysis.py  --  must sit in the same folder as this script
# =============================================================================

import os
import csv
import time
import argparse
import functools
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.signal import savgol_filter
import pybaselines as bs

from raman_analysis import read_renishaw_spots, describe_peak


# -----------------------------------------------------------------------------
#  Baseline helpers
# -----------------------------------------------------------------------------
def estimate_arpls_lambda(spectrum, base_lam=1e5, reference_noise_ratio=0.008,
                          min_lam=1e4, max_lam=1e6):
    """Per-spectrum arPLS lambda from a robust (MAD) noise estimate."""
    y = np.asarray(spectrum, dtype=float)
    delta = np.diff(y)
    med = np.median(delta)
    mad = np.median(np.abs(delta - med))
    noise = 0.6745 * mad
    scale = np.median(np.abs(y)) + 1e-12
    noise_ratio = noise / scale
    lam = base_lam * (noise_ratio / reference_noise_ratio) ** 2
    return float(np.clip(lam, min_lam, max_lam))


def estimate_baseline(spectrum, *, use_spline, auto_lambda, lam=1e5):
    """Baseline for ONE spectrum.

    Configuration is passed in explicitly rather than read from globals.
    """
    if use_spline:
        return bs.spline.mixture_model(
            spectrum, lam=5000.0, p=0.05, num_knots=100, spline_degree=3,
            diff_order=3, max_iter=50, tol=0.001, weights=None,
            symmetric=False, num_bins=None,
        )[0]
    if auto_lambda:
        lam = estimate_arpls_lambda(spectrum, base_lam=lam)
    return bs.whittaker.arpls(spectrum, lam=lam)[0]


# -----------------------------------------------------------------------------
#  Parallel helpers
# -----------------------------------------------------------------------------
#  The per-spot stages -- despiking, baseline correction and peak fitting --
#  are independent across spots and dominate the runtime, so they are mapped
#  across worker processes.  The work is CPU-bound NumPy/SciPy/lmfit, so
#  processes (not threads) are used to sidestep the GIL.  Workers are
#  module-level functions taking picklable arguments so they survive the
#  spawn start method used on Windows/macOS.
def resolve_jobs(requested, n_items):
    """Clamp the requested worker count to [1, min(n_items, cpu_count)]."""
    cpu = os.cpu_count() or 1
    want = cpu if not requested or requested <= 0 else requested
    return max(1, min(want, n_items, cpu))


def run_parallel(worker, items, executor):
    """map(worker, items), order-preserving, with a serial fast path.

    Runs serially when `executor` is None or there is a single item, which
    avoids paying process-startup overhead on small workloads.
    """
    items = list(items)
    if executor is None or len(items) <= 1:
        return [worker(it) for it in items]
    return list(executor.map(worker, items))


def _noop(_):
    """Trivial task used to pre-spawn the worker pool before timing begins."""
    return None


# -----------------------------------------------------------------------------
#  Whitaker-Hayes despiking -- local, bug-fixed reimplementation
# -----------------------------------------------------------------------------
#  Self-contained Whitaker-Hayes despiker (no external Raman library needed).
#  RamanSPy 0.2.10's version has three bugs (its open issue #19): it divides by
#  zero on flat regions (NaN/inf z-scores), misaligns the spike mask by one
#  because np.diff shortens the array, and never inspects or repairs the final
#  channel.  This implementation fixes all three.
def _wh_modified_z_score(values):
    """Modified z-scores; zeros (not NaN/inf) when the MAD is zero."""
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        return np.zeros_like(values, dtype=float)
    return 0.6745 * (values - median) / mad


def _wh_spike_mask(spectrum, threshold):
    """Boolean spike mask aligned to `spectrum` (length n, endpoints inspected)."""
    scores = np.abs(_wh_modified_z_score(np.diff(spectrum)))
    # np.diff yields n-1 scores; scores[j] measures the jump spectrum[j+1] -
    # spectrum[j], so after prepending, score i maps to the left-hand jump into
    # spectrum[i].  Repeat the first score for the boundary (NOT the global max):
    # forcing the boundary to scores.max() made channel 0 exceed the threshold
    # whenever ANY spike existed in the frame, so the low-wavenumber edge was
    # overwritten in every despiked spectrum.  Repeating scores[0] still inspects
    # channel 0 on its own gradient -- a genuine edge spike makes diff[0] large,
    # so it is still caught -- without flagging it because of a distant spike.
    scores = np.insert(scores, 0, scores[0] if scores.size else 0.0)
    return scores > threshold


def _wh_despike_spectrum(spectrum, kernel_size, threshold):
    """Despike ONE spectrum, replacing spikes by the mean of clean neighbours."""
    out = np.array(spectrum, dtype=float, copy=True)
    spikes = _wh_spike_mask(out, threshold)
    while spikes.any():
        changed = False
        for i in np.flatnonzero(spikes):
            # +1 in the upper bound so the final channel is a valid neighbour.
            neighbours = np.arange(max(0, i - kernel_size),
                                   min(len(out), i + 1 + kernel_size))
            clean = neighbours[spikes[neighbours] == 0]
            if clean.size == 0:
                continue  # no clean neighbour yet; revisit on a later pass
            value = np.mean(out[clean])
            if np.isnan(value):
                continue
            out[i] = value
            spikes[i] = False
            changed = True
        if not changed:
            break
    return out


def _whitaker_hayes_despike_2d(spot2d, kernel_size=3, threshold=6.0):
    """Apply the bug-fixed despiker to each frame of a (n_frames, n_wn) array."""
    return np.apply_along_axis(
        _wh_despike_spectrum, -1, spot2d,
        kernel_size=kernel_size, threshold=threshold).astype(np.float32)


def _despike_spot(spot2d, *, kernel_size=3, threshold=6.0):
    return _whitaker_hayes_despike_2d(spot2d, kernel_size=kernel_size,
                                      threshold=threshold)


def _savgol_spot(spot2d, *, window_length=21, polyorder=5):
    return savgol_filter(spot2d, window_length=window_length,
                         polyorder=polyorder, axis=-1).astype(np.float32)


def _baseline_spot(spot2d, *, use_spline, auto_lambda):
    bl = np.array(
        [estimate_baseline(row, use_spline=use_spline, auto_lambda=auto_lambda)
         for row in spot2d],
        dtype=np.float32)
    return bl, (spot2d - bl).astype(np.float32)


def aggregate_centers(centers, method='trimmed', trim_fraction=0.2):
    """Combine per-frame peak centers into one estimate + uncertainty.

    Returns (center, center_err, n) where `n` is the number of finite centers
    used.  `center_err` is NaN when it cannot be estimated (fewer than two
    centers); callers fall back to the averaged fit's covariance error then.

    method:
      'mean'     arithmetic mean, SEM = std/sqrt(n).  Minimum-variance / ML
                 estimator, but optimal only for clean, outlier-free Gaussian
                 scatter -- a single bad frame drags it off.
      'median'   robust to outliers and SERS blinking; err is the asymptotic
                 standard error of the median (1.2533 * MAD_std / sqrt(n)).
      'trimmed'  symmetric trimmed mean dropping `trim_fraction` of each tail.
                 Rejects outliers while keeping most of the mean's efficiency;
                 err uses the Winsorized (Tukey-McLaughlin) trimmed SEM.
    """
    c = np.asarray(centers, dtype=float)
    c = c[np.isfinite(c)]
    n = c.size
    if n == 0:
        return float('nan'), float('nan'), 0
    if n == 1:
        return float(c[0]), float('nan'), 1

    if method == 'mean':
        return float(np.mean(c)), float(np.std(c, ddof=1) / np.sqrt(n)), n

    if method == 'median':
        center = float(np.median(c))
        mad = np.median(np.abs(c - center))
        robust_sd = 1.4826 * mad                 # MAD -> Gaussian-consistent std
        return center, float(1.2533 * robust_sd / np.sqrt(n)), n

    # 'trimmed'
    g = int(np.floor(trim_fraction * n))         # points cut from each tail
    s = np.sort(c)
    if g == 0:                                   # too few frames to trim
        return float(np.mean(c)), float(np.std(c, ddof=1) / np.sqrt(n)), n
    if 2 * g >= n:                               # would trim everything
        center = float(np.median(c))
        mad = np.median(np.abs(c - center))
        return center, float(1.2533 * 1.4826 * mad / np.sqrt(n)), n
    center = float(np.mean(s[g:n - g]))
    # Winsorize the trimmed tails, then the Tukey-McLaughlin trimmed SEM.
    w = s.copy()
    w[:g] = s[g]
    w[n - g:] = s[n - g - 1]
    sw = np.std(w, ddof=1)
    gamma = g / n
    return center, float(sw / ((1.0 - 2.0 * gamma) * np.sqrt(n))), n


def _fit_spot(packed, *, wn, search_min, search_max, model, fit_mode,
              center_agg='trimmed', trim_fraction=0.2):
    """Fit one spot; returns
    (result_dict, averaged_spectrum, lo_env, hi_env, lo_keep, hi_keep).

    `packed` is (spot_index, spot_2d, noise_2sigma) to keep map() single-arg.
    `lo_env`/`hi_env` are the per-wavenumber min/max across the spot's frames
    (or map positions); `lo_keep`/`hi_keep` are the central percentile band
    a trimmed aggregator keeps (the [trim_fraction, 1-trim_fraction] quantiles
    at each wavenumber), so a plot can shade the ignored tails separately from
    the retained core.
    """
    i, spot2d, noise_i = packed
    avg_spec = np.mean(spot2d, axis=0)
    lo_env = np.min(spot2d, axis=0)   # quietest position at each wavenumber
    hi_env = np.max(spot2d, axis=0)   # loudest position at each wavenumber
    # Per-wavenumber band a trimmed aggregator keeps vs. ignores: the central
    # (1 - 2*trim_fraction) of positions at each wavenumber.
    q = 100.0 * trim_fraction
    lo_keep = np.percentile(spot2d, q, axis=0)
    hi_keep = np.percentile(spot2d, 100.0 - q, axis=0)
    n = spot2d.shape[0]
    gate_avg = noise_i / np.sqrt(n)   # averaging cuts noise by sqrt(N)

    # Averaged fit: representative shape, area, R2 and the QA curve.
    favg = describe_peak(wn, avg_spec, search_min, search_max,
                         model=model, noise_threshold=gate_avg)

    if fit_mode == 'perframe':
        per = [describe_peak(wn, frame, search_min, search_max,
                             model=model, noise_threshold=noise_i)
               for frame in spot2d]
        centers = [f['center'] for f in per if f['success'] and f['above_noise']]
        res = dict(favg)  # shape/area/R2/QA curve from the averaged fit
        if centers:
            center, cerr, k = aggregate_centers(
                centers, method=center_agg, trim_fraction=trim_fraction)
            res['center'] = center
            res['center_err'] = cerr if np.isfinite(cerr) else favg['center_err']
            res['above_noise'] = True
            res['n_frames_fit'] = k
            res['center_agg'] = center_agg
            # Per-frame centers are valid even if the averaged fit failed, so
            # mark the spot successful when any frame fit converged.
            res['success'] = True
        else:
            res['n_frames_fit'] = 0
            res['center_agg'] = center_agg
            res['success'] = favg['success']
    else:
        res = dict(favg)
        res['n_frames_fit'] = n if favg['success'] else 0
        res['center_agg'] = 'average'

    res.update(spot=i)
    return res, avg_spec, lo_env, hi_env, lo_keep, hi_keep


# -----------------------------------------------------------------------------
#  X-axis calibration
# -----------------------------------------------------------------------------
def calibrate_axis(wn):
    """Return the wavenumber axis unchanged -- calibration is intentionally a
    no-op, and no software calibration is needed currently.

    The spectrometer is calibrated at the instrument before each run, and a run
    is only kept if the calibration is verified accurate both BEFORE and AFTER
    acquisition.  The exported axis is therefore already on the true Raman-shift
    scale, so re-calibrating it in software would add nothing and could only
    introduce error.

    The step is kept in the pipeline as a deliberate, documented placeholder:
    it is the single place to add in-software calibration if a future setup ever
    needs it (e.g. an uncalibrated instrument, or anchoring to an internal
    reference band) without disturbing the rest of the pipeline.  As written it
    does nothing to the data.
    """
    return wn  # pass the axis straight through, untouched


# -----------------------------------------------------------------------------
#  Plotting
# -----------------------------------------------------------------------------
def setup_rcparams():
    plt.rcParams.update({
        'font.size': 10,
        'font.weight': 'normal',
        'axes.labelsize': 10,
        'axes.labelweight': 'normal',
        'axes.titlesize': 10,
        'axes.titleweight': 'normal',
        'axes.linewidth': 1.0,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'xtick.minor.width': 0.75,
        'ytick.minor.width': 0.75,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'legend.fontsize': 9,
        'legend.framealpha': 0.8,
        'lines.markersize': 4,
        'lines.linewidth': 1.5,
        'axes.grid': False,
        'figure.autolayout': True,
    })


def qa_plot(wn, traces, title, window=None, *, ylabel='Counts / arb. units',
            save=None, show=True, dpi=800):
    """One full-axis QA figure with an optional shaded search window.

    `traces` is a list of (y, label).  Returns immediately when there is
    nothing to display or save.
    """
    if not show and save is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for y, label in traces:
        ax.plot(wn, y, label=label)
    if window is not None:
        ax.axvspan(window[0], window[1], alpha=0.15, color='steelblue',
                   label='Search window')
    ax.set_title(title)
    ax.set_xlabel('Raman shift / cm$^{-1}$')
    ax.set_ylabel(ylabel)
    ax.locator_params(axis='x', nbins=8)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(3))
    ax.legend()
    if save is not None:
        plt.savefig(save, dpi=dpi)
        print(f"\n  Saved: {save}")
    if show:
        plt.show()
    plt.close(fig)


# -----------------------------------------------------------------------------
#  IO
# -----------------------------------------------------------------------------
def load_spots(data_dir, *, trim=3, resample=True):
    """Load every readable Renishaw .txt spot in `data_dir`.

    Returns (wn, spectra_raw, files) kept parallel: a spot is appended only
    when its read succeeds.  Files whose axis differs from the first spot are
    resampled onto the common grid (or skipped when resample=False).
    """
    all_entries = sorted(os.listdir(data_dir))
    txts = [e for e in all_entries
            if os.path.isfile(os.path.join(data_dir, e))
            and e.lower().endswith('.txt')]
    skipped = [e for e in all_entries if e not in txts]
    if skipped:
        print(f"  Ignoring {len(skipped)} non-Renishaw-txt item(s): "
              f"{', '.join(skipped[:5])}{'...' if len(skipped) > 5 else ''}")
    if not txts:
        raise SystemExit(f"No .txt files found in: {data_dir}")

    files, spectra, wn = [], [], None
    for fname in txts:
        path = os.path.join(data_dir, fname)
        try:
            wn_raw, signal = read_renishaw_spots(path)
        except Exception as exc:
            print(f"  [skip] {fname}: {exc}")
            continue

        wn_c = wn_raw[trim:]
        sig_c = signal[:, trim:]
        if wn is None:
            wn = wn_c
        elif wn_c.shape != wn.shape or not np.allclose(wn_c, wn, rtol=0.0, atol=1e-8):
            if resample:
                sig_c = np.vstack(
                    [np.interp(wn, wn_c, frame) for frame in sig_c]
                ).astype(np.float32)
                print(f"  [resampled] {fname}: axis differed; mapped onto common grid")
            else:
                print(f"  [skip] {fname}: wavenumber axis mismatch")
                continue

        spectra.append(sig_c)
        files.append(fname)

    if not spectra:
        raise SystemExit("No files could be read. "
                         "Check that the folder contains valid Renishaw .txt exports.")
    print(f"  Loaded {len(spectra)} spot(s) from {data_dir}")
    return wn, spectra, files


# -----------------------------------------------------------------------------
#  Args
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Renishaw/SERS spot-data peak-shift pipeline.")
    p.add_argument('path', nargs='?', default=r"C:\ram\Data",
                   help='Directory containing Renishaw time-series files.')
    p.add_argument('--label', default='OI',
                   help='Short label stamped into output filenames (default: OI).')
    p.add_argument('--model', default='lorentzian',
                   choices=['lorentzian', 'gaussian', 'voigt'],
                   help='Peak line shape (default: lorentzian).')
    p.add_argument('--useSplineMixtureBaseline', action='store_true',
                   help='Use the slower spline mixture baseline instead of arPLS.')
    p.add_argument('--fixedArplsLambda', action='store_true',
                   help='Use the default arPLS lambda instead of the per-spectrum estimate.')
    p.add_argument('--peak-min', type=float, default=None,
                   help='Lower wavenumber bound for peak search (cm^-1).')
    p.add_argument('--peak-max', type=float, default=None,
                   help='Upper wavenumber bound for peak search (cm^-1).')
    p.add_argument('--fit-mode', default='perframe',
                   choices=['average', 'perframe'],
                   help='perframe (default, most accurate): fit every frame and '
                        'combine the per-frame centers with --center-agg, taking '
                        'shape/area/R2 from the averaged spectrum. average: fit '
                        'only the per-spot averaged spectrum (faster, center from '
                        'one fit).')
    p.add_argument('--center-agg', default='trimmed',
                   choices=['mean', 'median', 'trimmed'],
                   help='How to combine per-frame centers in --fit-mode perframe '
                        '(default: trimmed). trimmed: symmetric trimmed mean -- '
                        'rejects outliers while keeping most of the efficiency; '
                        'median: most robust; mean: plain mean +/- SEM (optimal '
                        'only when there are no outlier frames).')
    p.add_argument('--trim-fraction', type=float, default=0.2,
                   help='Fraction trimmed from each tail when --center-agg trimmed '
                        '(default: 0.2 = 20%% per tail). Ignored otherwise.')
    p.add_argument('--smooth-before-fit', action='store_true',
                   help='Apply Savitzky-Golay before fitting (legacy behavior). '
                        'Off by default: fitting smoothed data biases width/R2.')
    p.add_argument('--strict-axis', action='store_true',
                   help='Reject files whose axis differs from the first spot '
                        'instead of resampling onto the common grid.')
    p.add_argument('--qa-spot', type=int, default=None,
                   help='Spot index used in QA plots (default: midpoint).')
    p.add_argument('--qa-frame', type=int, default=None,
                   help='Frame index used in QA plots (default: midpoint).')
    p.add_argument('--no-qa-plots', action='store_true',
                   help='Suppress interactive QA windows (saved figures still written). '
                        'Use for headless / batch runs.')
    p.add_argument('--parallel', action='store_true',
                   help='Run the despike / baseline / fit stages across worker '
                        'processes. Off by default (serial).')
    p.add_argument('--jobs', type=int, default=0,
                   help='Worker process count when --parallel is set. '
                        '0 (default) uses all CPU cores. Ignored without --parallel.')
    return p.parse_args()


# -----------------------------------------------------------------------------
#  Pipeline
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    setup_rcparams()
    show = not args.no_qa_plots
    label = args.label

    if not 0.0 <= args.trim_fraction < 0.5:
        raise SystemExit(f"--trim-fraction must be in [0, 0.5); got "
                         f"{args.trim_fraction}.")

    # --- Read ----------------------------------------------------------------
    wn, spectra_raw, files = load_spots(args.path, resample=not args.strict_axis)
    wn = calibrate_axis(wn)   # no-op: instrument is hardware-calibrated per run

    # QA indices default to the data midpoints; override with --qa-spot/--qa-frame.
    qa_i = min(args.qa_spot if args.qa_spot is not None else len(spectra_raw) // 2,
               len(spectra_raw) - 1)
    n_frames_qa = spectra_raw[qa_i].shape[0]
    qa_j = min(args.qa_frame if args.qa_frame is not None else n_frames_qa // 2,
               n_frames_qa - 1)

    # Search window -- defined here so every plot can shade it.
    search_min = args.peak_min if args.peak_min is not None else float(wn.min())
    search_max = args.peak_max if args.peak_max is not None else float(wn.max())
    if search_min >= search_max:
        raise SystemExit(f"Invalid peak window: --peak-min ({search_min}) "
                         f"must be < --peak-max ({search_max}).")
    window = (search_min, search_max)

    qa_plot(wn, [(spectra_raw[qa_i][qa_j], 'Raw')],
            f'Plot 1: Raw spectrum - data loading check (spot {qa_i}, frame {qa_j})',
            window, save=None, show=show)

    # Spin up one shared worker pool for the three CPU-bound per-spot stages
    # (despiking, baseline correction, peak fitting).  These are independent
    # across spots and dominate the runtime, so mapping them over cores is the
    # main speed-up.  Parallelism is opt-in (--parallel); serial by default.
    if args.parallel:
        n_jobs = resolve_jobs(args.jobs, len(spectra_raw))
    else:
        n_jobs = 1
    executor = ProcessPoolExecutor(max_workers=n_jobs) if n_jobs > 1 else None
    if executor is not None:
        print(f"  Parallel stages using {n_jobs} worker process(es).")
        # Spawn the workers now so their one-time start-up cost (process spawn
        # plus re-importing this module) is not charged to the despike timing.
        list(executor.map(_noop, range(n_jobs)))
    elif args.parallel:
        print("  Serial processing (only one worker available for this run).")
    else:
        print("  Serial processing (pass --parallel to use multiple cores).")

    # Time only the per-spot compute stages, so the reported figure reflects the
    # work parallelism actually affects (interactive QA plots are excluded).
    stage_times = []
    proc_start_wall = datetime.now()
    proc_start = time.perf_counter()
    print(f"  Processing start: {proc_start_wall:%Y-%m-%d %H:%M:%S}")

    # --- Despike (always) + Savitzky-Golay (QA / optional pre-fit) -----------
    # threshold=6.0 is more sensitive than the Whitaker-Hayes default of 8: the
    # lower cutoff flags fainter cosmic-ray spikes.  Genuine SERS bands survive
    # because they are broad -- their point-to-point differences stay well below
    # the z-score cutoff, so only narrow (1-2 channel) spikes are removed.

    try:
        _t = time.perf_counter()
        spectra_despiked = run_parallel(_despike_spot, spectra_raw, executor)
        stage_times.append(('despike', time.perf_counter() - _t))

        # Smoothed view of the QA spot for the raw-vs-smoothed comparison.
        qa_smoothed = _savgol_spot(spectra_despiked[qa_i])
        qa_plot(wn,
                [(spectra_raw[qa_i][qa_j], 'Raw'), (qa_smoothed[qa_j], 'Smoothed')],
                f'Plot 2: Raw vs smoothed spectrum (spot {qa_i}, frame {qa_j})',
                window, save=None, show=show)

        # Fit input: despiked-but-unsmoothed by default; SavGol only if requested.
        if args.smooth_before_fit:
            _t = time.perf_counter()
            fit_input = run_parallel(_savgol_spot, spectra_despiked, executor)
            stage_times.append(('smooth', time.perf_counter() - _t))
        else:
            fit_input = spectra_despiked

        # --- Baseline correction ---------------------------------------------
        _t = time.perf_counter()
        baseline_pairs = run_parallel(
            functools.partial(_baseline_spot,
                              use_spline=args.useSplineMixtureBaseline,
                              auto_lambda=not args.fixedArplsLambda),
            fit_input, executor)
        stage_times.append(('baseline', time.perf_counter() - _t))
        baselines = [bl for bl, _ in baseline_pairs]
        spectra_b = [corr for _, corr in baseline_pairs]
        del baseline_pairs

        if fit_input is not spectra_despiked:
            del fit_input
        del spectra_despiked

        qa_plot(wn,
                [(spectra_raw[qa_i][qa_j], 'Raw'),
                 (spectra_b[qa_i][qa_j], 'Corrected'),
                 (baselines[qa_i][qa_j], 'Baseline')],
                f'Plot 3: Baseline correction check (spot {qa_i}, frame {qa_j})',
                window, save=None, show=show)

        # Baseline parameter check, run on the same fit input the pipeline uses.
        base_temp = estimate_baseline(spectra_b[qa_i][qa_j] + baselines[qa_i][qa_j],
                                      use_spline=args.useSplineMixtureBaseline,
                                      auto_lambda=not args.fixedArplsLambda)
        qa_plot(wn,
                [(spectra_b[qa_i][qa_j] + baselines[qa_i][qa_j], 'Fit input'),
                 (base_temp, 'Baseline')],
                f'Plot 4: Baseline parameter check (spot {qa_i}, frame {qa_j})',
                window, save=None, show=show)

        del spectra_raw  # last used above

        # --- Noise threshold -------------------------------------------------
        # 2-sigma of the quietest channel (smallest mean |signal|) across frames.
        noise_2sigma = []
        for s in spectra_b:
            av = np.mean(s, axis=0)
            q = int(np.argmin(np.abs(av)))
            col = s[:, q]
            nz = col[col != 0]
            noise_2sigma.append(2.0 * float(np.std(nz if nz.size else col)))

        # Treatment figure (saved): baseline, corrected, per-frame noise floor.
        tr = noise_2sigma[qa_i] * np.ones(len(wn))
        qa_plot(wn,
                [(baselines[qa_i][qa_j], 'Baseline'),
                 (spectra_b[qa_i][qa_j], 'Corrected spectra'),
                 (tr, 'Zero signal threshold')],
                'Plot 5: Spectral treatment: baseline, corrected spectrum and noise threshold',
                window, save=f'spectra_treatment_{label}.png', show=show)

        del baselines

        # --- Peak fitting ----------------------------------------------------
        # Average all frames per spot before fitting (higher SNR); each averaged
        # spectrum is one measurement condition (e.g. one proton-donor concentration)
        # and its fitted center is the shifted Raman frequency.  In perframe mode we
        # additionally fit every frame and report center as mean +/- SEM, which
        # propagates the real temporal scatter (SERS blinking) into the uncertainty.
        print(f"\nFitting peaks in [{search_min:.1f}, {search_max:.1f}] cm^-1 "
              f"({args.model}, --fit-mode {args.fit_mode}) ...")
        _t = time.perf_counter()
        fit_out = run_parallel(
            functools.partial(_fit_spot, wn=wn, search_min=search_min,
                              search_max=search_max, model=args.model,
                              fit_mode=args.fit_mode, center_agg=args.center_agg,
                              trim_fraction=args.trim_fraction),
            list(zip(range(len(spectra_b)), spectra_b, noise_2sigma)),
            executor)
        stage_times.append(('fit', time.perf_counter() - _t))
    finally:
        if executor is not None:
            executor.shutdown()   # release workers even if a stage raised

    # --- Timing report (compute stages) --------------------------------------
    proc_end_wall = datetime.now()
    compute_elapsed = sum(t for _, t in stage_times)
    mode = f"parallel ({n_jobs} workers)" if executor is not None else "serial"
    print(f"\n  Compute stages done: {proc_end_wall:%Y-%m-%d %H:%M:%S}   ({mode})")
    print("  Stage compute times (s): "
          + "  ".join(f"{name}={t:.2f}" for name, t in stage_times))
    print(f"  Elapsed (compute stages): {compute_elapsed:.2f} s")

    # Worker output arrives in spot order, so the CSV row order and the per-spot
    # QA figures match the original serial behavior.
    results = []
    qa_fits = []  # (spot, avg, lo_env, hi_env, lo_keep, hi_keep, result) per spot
    for res, avg_spec, lo_env, hi_env, lo_keep, hi_keep in fit_out:
        res['file'] = files[res['spot']]
        results.append(res)
        if res['success'] and res.get('x_fit') is not None:
            qa_fits.append((res['spot'], avg_spec, lo_env, hi_env,
                            lo_keep, hi_keep, res))

    del fit_out, spectra_b  # largest arrays, no longer needed after fitting

    # --- Terminal table ------------------------------------------------------
    hdr = (f"{'spot':>5}  {'center(cm-1)':>12}  {'+-err':>7}  "
           f"{'fwhm':>7}  {'height':>9}  {'area':>10}  {'R2':>6}  {'>noise':>6}  file")
    print(hdr)
    print('-' * len(hdr))
    for r in results:
        if r['success']:
            print(f"{r['spot']:>5}  {r['center']:>12.3f}  {r['center_err']:>7.4f}  "
                  f"{r['fwhm']:>7.3f}  {r['height']:>9.1f}  {r['area']:>10.1f}  "
                  f"{r['r_squared']:>6.4f}  {('yes' if r['above_noise'] else 'no'):>6}  "
                  f"{r['file']}")
        else:
            print(f"{r['spot']:>5}  {'FAILED':>12}  {'':>7}  "
                  f"{'':>7}  {'':>9}  {'':>10}  {'':>6}  {'':>6}  {r['file']}")

    # --- CSV (opens directly in Excel / Origin / MATLAB) ---------------------
    csv_path = f'peak_fits_{label}.csv'
    fieldnames = ['spot', 'file', 'center', 'center_err', 'fwhm', 'height',
                  'area', 'r_squared', 'aic', 'bic', 'n_frames_fit',
                  'center_agg', 'above_noise', 'model', 'success']
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Saved: {csv_path}")

    # --- Plot 1: fit-quality check (true fitted curve, no reconstruction) ----
    # One figure per spot: averaged spectrum, the true fitted curve, and the
    # min-max band across that spot's positions.  Saved so every spot is kept.
    # Figures are built first and shown together at the end, so all spot windows
    # appear at once instead of blocking one-at-a-time on each plt.show().
    mask = (wn >= search_min) & (wn <= search_max)
    qa_figs = []
    for qa_spot, qa_spec, qa_lo, qa_hi, qa_klo, qa_khi, r_qa in qa_fits:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        # Spread of the positions around the average.  When the center was
        # aggregated with a trimmed mean, split the band: a lighter outer band
        # for the tails the trim ignores and a darker inner band for the
        # central positions it keeps.
        if r_qa.get('center_agg') == 'trimmed':
            pct = int(round(100.0 * (1.0 - 2.0 * args.trim_fraction)))
            ax.fill_between(wn[mask], qa_lo[mask], qa_hi[mask],
                            color='steelblue', alpha=0.12,
                            label='Positions ignored (trimmed tails)')
            ax.fill_between(wn[mask], qa_klo[mask], qa_khi[mask],
                            color='steelblue', alpha=0.30,
                            label=f'Central {pct}% retained')
        else:
            ax.fill_between(wn[mask], qa_lo[mask], qa_hi[mask],
                            color='steelblue', alpha=0.20,
                            label='Min-max across positions')
        ax.plot(wn[mask], qa_spec[mask],
                label='Averaged spectrum (all positions)')
        ax.plot(r_qa['x_fit'], r_qa['y_fit'], '--',
                label=f"Fitted {r_qa['model']}  R$^2$={r_qa['r_squared']:.4f}")
        ax.axvline(r_qa['center'], color='gray', linestyle=':',
                   label=f"Center {r_qa['center']:.2f} cm$^{{-1}}$")
        ax.set_xlabel('Raman shift / cm$^{-1}$')
        ax.set_ylabel('Counts / arb. units')
        ax.set_title(f"Plot 6: Fit QA - spot {qa_spot}  ({r_qa['file']})")
        ax.locator_params(axis='x', nbins=8)
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(3))
        ax.legend()
        qa_path = f'fit_qa_{label}_spot{qa_spot}.png'
        plt.savefig(qa_path, dpi=800)
        print(f"\n  Saved: {qa_path}")
        qa_figs.append(fig)

    # Show every spot's figure together (one blocking call), then close them all.
    if show and qa_figs:
        plt.show()
    for fig in qa_figs:
        plt.close(fig)

    # --- Plot 2: fitted center per spot (the shift result) -------------------
    good = [(r['spot'], r['center'], r['center_err'])
            for r in results if r['success'] and r['above_noise']]
    n_below = sum(1 for r in results if r['success'] and not r['above_noise'])
    if n_below:
        print(f"  Excluded {n_below} sub-threshold peak(s) from the shift result "
              f"(still listed in the CSV).")
    if good:
        s_vals, c_vals, e_vals = zip(*good)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.errorbar(s_vals, c_vals, yerr=e_vals, fmt='o-', capsize=4)
        ax.set_xlabel('Spot')
        ax.set_ylabel('Peak center / cm$^{-1}$')
        ax.set_title('Plot 7: Fitted peak frequency per spot')
        ax.locator_params(axis='x', nbins=min(len(s_vals), 10))
        ax.yaxis.set_minor_locator(AutoMinorLocator(3))
        plt.savefig(f'peak_centers_{label}.png', dpi=800)
        print(f"\n  Saved: peak_centers_{label}.png")
        if show:
            plt.show()
        plt.close(fig)

    # --- Total wall-clock time (compute + QA plots + output) -----------------
    print(f"\n  Processing end:   {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  Elapsed (incl. QA plots & output): "
          f"{time.perf_counter() - proc_start:.2f} s")


if __name__ == '__main__':
    main()
