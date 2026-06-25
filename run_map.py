"""
run_map.py -- Raman/SERS map processing pipeline for 4-MBN on AgNP substrates

This script processes one Renishaw mapping file at a time. It is intended for
single-condition measurements, such as one AgNP/4-MBN substrate immersed in one
DMF/protic-solvent mixture. Later, the one-row map_summary CSV files from many
runs can be combined into a master solvatochromic trend table.

USER EDITS FOR NORMAL USE
-------------------------
Change only these two global variables first:

    MAP_FILE_PATH : path to the Renishaw map .txt/.csv file
    OUTPUT_DIR    : folder where output figures and CSV files will be saved

Other analysis parameters are grouped below and can be tuned after you inspect
initial outputs.
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, MaxNLocator, FuncFormatter

from raman_map_analysis import (
    brolo_style,
    bootstrap_mean_ci,
    describe_peak,
    preprocess_spectra,
    read_renishaw_map,
    reconstruct_peak_curve,
    robust_noise,
    values_to_grid,
)

plt.rcParams.update(brolo_style)


# =============================================================================
# Global input/output variables -- edit these for each measurement
# =============================================================================

MAP_FILE_PATH = r"C:\path\to\your\input\txt\file"
OUTPUT_DIR = r"C:\path\to\your\output\directory"


# =============================================================================
# Analysis configuration
# =============================================================================

# If SAMPLE_NAME is None, a safe name is generated from the input filename.
SAMPLE_NAME = None

# 4-MBN nitrile band. Widen/narrow after inspecting the QA plots.
PEAK_MIN = 2180.0
PEAK_MAX = 2260.0
PEAK_MODEL = "lorentzian"          # "lorentzian", "gaussian", or "voigt"
FIT_LOCAL_LINEAR_BASELINE = True    # extra protection against small residual baseline

# Preprocessing settings. These preserve the spirit of the original run.py.
DESPIKE = True
SMOOTH = True
SAVGOL_WINDOW = 21
SAVGOL_POLYORDER = 5
BASELINE_LAMBDA = 1e6
BASELINE_P = 0.01

# Fit acceptance criteria. These can be relaxed/tightened later.
MIN_R2 = 0.70
MIN_SNR = 3.0
MIN_FWHM = 2.0
MAX_FWHM = 80.0

# Uncertainty band on the averaged corrected spectrum: "sd" or "sem".
UNCERTAINTY_BAND = "sd"


# =============================================================================
# Small utility functions
# =============================================================================

def safe_sample_name(path: str, sample_name: str | None = None) -> str:
    """Return a filesystem-safe sample name."""
    if sample_name:
        base = sample_name
    else:
        base = Path(path).stem
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_")
    return base[:120] if len(base) > 120 else base


def save_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    """Write rows to CSV, preserving field order."""
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def accepted_mask_from_results(results: list[dict]) -> np.ndarray:
    """Boolean mask for peak fits that pass quality criteria."""
    mask = []
    for r in results:
        ok = bool(r["success"])
        ok = ok and np.isfinite(r["center"])
        ok = ok and np.isfinite(r["fwhm"])
        ok = ok and np.isfinite(r["r_squared"])
        ok = ok and np.isfinite(r["snr"])
        ok = ok and r["r_squared"] >= MIN_R2
        ok = ok and r["snr"] >= MIN_SNR
        ok = ok and MIN_FWHM <= r["fwhm"] <= MAX_FWHM
        mask.append(ok)
    return np.asarray(mask, dtype=bool)


def plot_mean_raw_vs_corrected(wn, spectra_raw, spectra_corrected, out_path: Path) -> None:
    """Plot mean raw/original spectrum against mean corrected spectrum."""
    mean_raw = np.mean(spectra_raw, axis=0)
    mean_corr = np.mean(spectra_corrected, axis=0)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(wn, mean_raw, label="Mean original spectrum")
    ax.plot(wn, mean_corr, label="Mean baseline-corrected spectrum")
    ax.axvspan(PEAK_MIN, PEAK_MAX, alpha=0.15, label="Nitrile fit window")
    ax.set_xlabel("Raman shift / cm$^{-1}$")
    ax.set_ylabel("Intensity / arb. units")
    ax.set_title("Mean spectrum before and after baseline correction")
    ax.locator_params(axis="x", nbins=8)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(3))
    ax.legend()
    fig.savefig(out_path, dpi=800)
    plt.close(fig)


def plot_average_corrected_spectrum(wn, spectra_corrected, accepted, out_path: Path) -> None:
    """Plot mean corrected spectrum with uncertainty shading."""
    spectra_for_mean = spectra_corrected[accepted] if np.any(accepted) else spectra_corrected
    mean_corr = np.mean(spectra_for_mean, axis=0)
    sd_corr = np.std(spectra_for_mean, axis=0, ddof=1) if spectra_for_mean.shape[0] > 1 else np.zeros_like(mean_corr)
    if UNCERTAINTY_BAND.lower() == "sem" and spectra_for_mean.shape[0] > 1:
        band = sd_corr / np.sqrt(spectra_for_mean.shape[0])
        band_label = "± SEM across accepted map points"
    else:
        band = sd_corr
        band_label = "± SD across accepted map points"

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(wn, mean_corr, label="Mean corrected spectrum")
    ax.fill_between(wn, mean_corr - band, mean_corr + band, alpha=0.25, label=band_label)
    ax.axvspan(PEAK_MIN, PEAK_MAX, alpha=0.15, label="Nitrile fit window")
    ax.set_xlabel("Raman shift / cm$^{-1}$")
    ax.set_ylabel("Corrected intensity / arb. units")
    ax.set_title("Mean baseline-corrected spectrum")
    ax.locator_params(axis="x", nbins=8)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(3))
    ax.legend()
    fig.savefig(out_path, dpi=800)
    plt.close(fig)


def plot_average_nitrile_fit(wn, spectra_corrected, accepted, avg_fit, out_path: Path) -> None:
    """Plot averaged corrected spectrum near the nitrile band with fitted center."""
    spectra_for_mean = spectra_corrected[accepted] if np.any(accepted) else spectra_corrected
    mean_corr = np.mean(spectra_for_mean, axis=0)
    sd_corr = np.std(spectra_for_mean, axis=0, ddof=1) if spectra_for_mean.shape[0] > 1 else np.zeros_like(mean_corr)
    if UNCERTAINTY_BAND.lower() == "sem" and spectra_for_mean.shape[0] > 1:
        band = sd_corr / np.sqrt(spectra_for_mean.shape[0])
    else:
        band = sd_corr

    margin = 25.0
    mask = (wn >= PEAK_MIN - margin) & (wn <= PEAK_MAX + margin)
    x = wn[mask]
    y = mean_corr[mask]
    b = band[mask]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(x, y, label="Mean corrected spectrum")
    ax.fill_between(x, y - b, y + b, alpha=0.25, label=f"± {UNCERTAINTY_BAND.upper()}")

    if avg_fit["success"]:
        fit_mask = (x >= PEAK_MIN) & (x <= PEAK_MAX)
        x_fit = x[fit_mask]
        y_fit = reconstruct_peak_curve(
            x_fit,
            center=avg_fit["center"],
            height=avg_fit["height"],
            fwhm=avg_fit["fwhm"],
            model=avg_fit["model"],
        )
        ax.plot(x_fit, y_fit, "--", label=f"{avg_fit['model']} fit, R²={avg_fit['r_squared']:.3f}")
        ax.axvline(avg_fit["center"], linestyle=":", label=f"Center = {avg_fit['center']:.2f} cm$^{{-1}}$")

    ax.axvspan(PEAK_MIN, PEAK_MAX, alpha=0.12, label="Fit window")
    ax.set_xlabel("Raman shift / cm$^{-1}$")
    ax.set_ylabel("Corrected intensity / arb. units")
    ax.set_title("Averaged nitrile-band fit")
    ax.locator_params(axis="x", nbins=8)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(3))
    ax.legend()
    fig.savefig(out_path, dpi=800)
    plt.close(fig)


def plot_heatmap(coords, values, out_path: Path, title: str, colorbar_label: str) -> None:
    """Plot a spatial heatmap for one fitted quantity."""
    grid, x_unique, y_unique = values_to_grid(coords, values)
    extent = [x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()]

    # Slightly wider figure to prevent colourbar label/tick cramping.
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    im = ax.imshow(
        grid,
        origin="lower",
        extent=extent,
        aspect="auto",
        interpolation="nearest",
    )

    # Add extra spacing between the heatmap, colourbar ticks, and colourbar label.
    cbar = fig.colorbar(im, ax=ax, pad=0.045)
    cbar.set_label(colorbar_label, labelpad=16)

    # Case-specific colourbar formatting.
    lower_label = colorbar_label.lower()

    if "centre" in lower_label or "center" in lower_label:
        # Nitrile-centre heatmap: whole-number colourbar labels reduce cramping.
        cbar.locator = MaxNLocator(integer=True, nbins=5)
        cbar.update_ticks()

    elif "height" in lower_label or "intensity" in lower_label:
        # SERS-intensity heatmap: display thousands as 1k, 2k, 3k, etc.
        def thousands_formatter(x, pos):
            if abs(x) >= 1000:
                return f"{x / 1000:.0f}k"
            return f"{x:.0f}"

        cbar.formatter = FuncFormatter(thousands_formatter)
        cbar.locator = MaxNLocator(nbins=5)
        cbar.update_ticks()

    ax.set_xlabel("X position / µm")
    ax.set_ylabel("Y position / µm")
    ax.set_title(title)

    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    fig.savefig(out_path, dpi=800, bbox_inches="tight")
    plt.close(fig)


def plot_center_distribution(centres, out_path: Path) -> None:
    """Plot histogram of accepted nitrile centres."""
    centres = np.asarray(centres, dtype=float)
    centres = centres[np.isfinite(centres)]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    if centres.size:
        bins = min(15, max(5, int(np.sqrt(centres.size))))
        counts, _, _ = ax.hist(
            centres,
            bins=bins,
            color="tab:blue",
            edgecolor="black",
            label="Map points",
        )

        mean_centre = np.mean(centres)
        median_centre = np.median(centres)
        line_colour = "tab:orange"

        ax.axvline(
            mean_centre,
            color=line_colour,
            linestyle="--",
            linewidth=2.5,
            label=f"Mean = {mean_centre:.2f} cm$^{{-1}}$",
        )
        ax.axvline(
            median_centre,
            color=line_colour,
            linestyle=":",
            linewidth=2.8,
            label=f"Median = {median_centre:.2f} cm$^{{-1}}$",
        )

        # Give the tallest bar and mean/median markers some headroom, but avoid
        # making the histogram feel detached from the legend.
        if counts.size and np.nanmax(counts) > 0:
            ax.set_ylim(top=np.nanmax(counts) * 1.18)

        # Keep only modest horizontal padding. This avoids excessive dead space
        # while still giving the legend room to sit clearly inside the axes.
        x_min, x_max = np.nanmin(centres), np.nanmax(centres)
        x_range = max(x_max - x_min, 1.0)
        ax.set_xlim(x_min - 0.10 * x_range, x_max + 0.15 * x_range)

        # Use larger, clearer divisions so the wavenumber labels do not merge.
        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))

        # Use the normal legend text size from the plotting style.
        ax.legend(loc="upper right", framealpha=0.9)

    else:
        ax.text(
            0.5,
            0.5,
            "No accepted fits",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_xlabel("Fitted nitrile centre / cm$^{-1}$")
    ax.set_ylabel("Number of map points")
    ax.set_title("Distribution of fitted nitrile centres")

    fig.savefig(out_path, dpi=800, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main analysis
# =============================================================================

def main() -> None:
    sample = safe_sample_name(MAP_FILE_PATH, SAMPLE_NAME)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading map: {MAP_FILE_PATH}")
    map_data = read_renishaw_map(MAP_FILE_PATH)
    wn = map_data.wn
    spectra_raw = map_data.spectra
    coords = map_data.coords

    print(f"Loaded {map_data.n_points} map point(s) with {map_data.n_wavenumbers} wavenumber values each.")
    print(f"Map grid shape: {map_data.grid_shape[1]} x {map_data.grid_shape[0]} (X x Y)")

    print("Preprocessing spectra: despike, smooth, baseline-correct ...")
    spectra_processed, baselines, spectra_corrected = preprocess_spectra(
        spectra_raw,
        despike=DESPIKE,
        smooth=SMOOTH,
        savgol_window=SAVGOL_WINDOW,
        savgol_polyorder=SAVGOL_POLYORDER,
        baseline_lam=BASELINE_LAMBDA,
        baseline_p=BASELINE_P,
    )
    # baselines and spectra_processed are retained until QA figures are created.

    print(f"Fitting nitrile peak in [{PEAK_MIN:.1f}, {PEAK_MAX:.1f}] cm^-1 ...")
    results = []
    for i, y in enumerate(spectra_corrected):
        fit = describe_peak(
            wn,
            y,
            PEAK_MIN,
            PEAK_MAX,
            model=PEAK_MODEL,
            fit_linear_baseline=FIT_LOCAL_LINEAR_BASELINE,
        )
        noise = robust_noise(y, wn, PEAK_MIN, PEAK_MAX)
        snr = fit["height"] / noise if fit["success"] and np.isfinite(noise) and noise > 0 else np.nan
        row = {
            "point": i,
            "x": coords[i, 0],
            "y": coords[i, 1],
            "center": fit["center"],
            "center_err": fit["center_err"],
            "fwhm": fit["fwhm"],
            "height": fit["height"],
            "area": fit["area"],
            "r_squared": fit["r_squared"],
            "model": fit["model"],
            "success": fit["success"],
            "noise": noise,
            "snr": snr,
        }
        results.append(row)

    accepted = accepted_mask_from_results(results)
    for row, ok in zip(results, accepted):
        row["accepted"] = bool(ok)

    centers_all = np.asarray([r["center"] for r in results], dtype=float)
    heights_all = np.asarray([r["height"] for r in results], dtype=float)
    areas_all = np.asarray([r["area"] for r in results], dtype=float)
    centers_valid = centers_all[accepted]
    heights_valid = heights_all[accepted]
    areas_valid = areas_all[accepted]

    spectra_for_average = spectra_corrected[accepted] if np.any(accepted) else spectra_corrected
    mean_corrected = np.mean(spectra_for_average, axis=0)
    avg_fit = describe_peak(
        wn,
        mean_corrected,
        PEAK_MIN,
        PEAK_MAX,
        model=PEAK_MODEL,
        fit_linear_baseline=FIT_LOCAL_LINEAR_BASELINE,
    )

    ci_low, ci_high = bootstrap_mean_ci(centers_valid, n_boot=5000, ci=95.0, seed=1)
    center_mean = float(np.mean(centers_valid)) if centers_valid.size else np.nan
    center_sd = float(np.std(centers_valid, ddof=1)) if centers_valid.size > 1 else np.nan
    center_sem = float(center_sd / np.sqrt(centers_valid.size)) if centers_valid.size > 1 else np.nan

    summary = [{
        "sample": sample,
        "source_file": os.path.basename(MAP_FILE_PATH),
        "n_total": int(map_data.n_points),
        "n_success": int(np.sum([bool(r["success"]) for r in results])),
        "n_accepted": int(np.sum(accepted)),
        "peak_min_cm-1": PEAK_MIN,
        "peak_max_cm-1": PEAK_MAX,
        "center_mean_cm-1": center_mean,
        "center_sd_cm-1": center_sd,
        "center_sem_cm-1": center_sem,
        "center_median_cm-1": float(np.median(centers_valid)) if centers_valid.size else np.nan,
        "center_iqr_cm-1": float(np.percentile(centers_valid, 75) - np.percentile(centers_valid, 25)) if centers_valid.size else np.nan,
        "center_bootstrap95_low_cm-1": ci_low,
        "center_bootstrap95_high_cm-1": ci_high,
        "mean_height": float(np.mean(heights_valid)) if heights_valid.size else np.nan,
        "mean_area": float(np.mean(areas_valid)) if areas_valid.size else np.nan,
        "average_spectrum_center_cm-1": avg_fit["center"],
        "average_spectrum_center_err_cm-1": avg_fit["center_err"],
        "average_spectrum_fwhm_cm-1": avg_fit["fwhm"],
        "average_spectrum_r_squared": avg_fit["r_squared"],
    }]

    # Save tabular outputs.
    fits_csv = output_dir / f"map_peak_fits_{sample}.csv"
    save_csv(
        fits_csv,
        [
            "point", "x", "y", "center", "center_err", "fwhm", "height",
            "area", "r_squared", "model", "success", "noise", "snr", "accepted",
        ],
        results,
    )

    summary_csv = output_dir / f"map_summary_{sample}.csv"
    save_csv(summary_csv, list(summary[0].keys()), summary)

    # Save figures.
    plot_mean_raw_vs_corrected(
        wn,
        spectra_raw,
        spectra_corrected,
        output_dir / f"mean_raw_vs_corrected_{sample}.png",
    )
    plot_average_corrected_spectrum(
        wn,
        spectra_corrected,
        accepted,
        output_dir / f"average_corrected_spectrum_{sample}.png",
    )
    plot_average_nitrile_fit(
        wn,
        spectra_corrected,
        accepted,
        avg_fit,
        output_dir / f"nitrile_fit_average_{sample}.png",
    )

    center_for_map = centers_all.copy()
    center_for_map[~accepted] = np.nan
    height_for_map = heights_all.copy()
    height_for_map[~accepted] = np.nan
    area_for_map = areas_all.copy()
    area_for_map[~accepted] = np.nan

    plot_heatmap(
        coords,
        center_for_map,
        output_dir / f"nitrile_center_map_{sample}.png",
        "Fitted nitrile center across map",
        "Peak center / cm$^{-1}$",
    )
    plot_heatmap(
        coords,
        height_for_map,
        output_dir / f"nitrile_intensity_map_{sample}.png",
        "Nitrile SERS intensity across map",
        "Fitted peak height / arb. units",
    )
    plot_center_distribution(
        centers_valid,
        output_dir / f"nitrile_center_distribution_{sample}.png",
    )

    print("\nSaved outputs:")
    for path in sorted(output_dir.glob(f"*_{sample}.*")):
        print(f"  {path}")

    print("\nSummary:")
    print(f"  Accepted fits: {int(np.sum(accepted))} / {map_data.n_points}")
    print(f"  Mean nitrile center: {center_mean:.3f} ± {center_sem:.3f} cm^-1 (SEM across accepted map points)")
    print(f"  Bootstrap 95% CI: [{ci_low:.3f}, {ci_high:.3f}] cm^-1")
    print(f"  Average-spectrum fit center: {avg_fit['center']:.3f} cm^-1")


if __name__ == "__main__":
    main()
