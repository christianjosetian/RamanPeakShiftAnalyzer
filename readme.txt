RamanPeakShiftAnalyzer

Automated Raman/SERS spectroscopy analysis pipeline for processing Renishaw time-series and map measurements, extracting Raman peak characteristics, and tracking peak-frequency shifts across experimental conditions.

Overview

RamanPeakShiftAnalyzer is a Python-based analysis framework developed to process large collections of Raman and Surface-Enhanced Raman Spectroscopy (SERS) measurements. The software automates the complete workflow from raw instrument exports to quantitative peak measurements suitable for scientific analysis and publication.

The project was developed to support experiments where Raman peak positions, intensities, and line shapes change as a function of an external variable such as concentration, chemical environment, or surface interaction.

The pipeline emphasizes:

Reproducible data processing
Robust noise handling
Automated peak fitting
Parallel execution for large datasets
Publication-quality quality-assurance visualizations
Features
Data Import
Reads Renishaw ASCII time-series and map exports (dispatched by column layout; map positions are treated as frames)
Validates file structure and spectral consistency
Automatically aligns spectra to a common wavenumber axis
Optional strict axis validation mode
Spectral Processing
Cosmic-ray removal using the Whitaker-Hayes modified z-score method
Savitzky-Golay smoothing (optional)
Adaptive baseline correction using:
arPLS (default)
Spline Mixture Model (optional)
Automatic baseline parameter estimation
Noise Characterization
Estimates per-spot noise levels from the quietest spectral channels
Implements noise-aware peak detection thresholds
Accounts for signal-to-noise improvement from frame averaging
Peak Analysis

Supports:

Lorentzian fitting
Gaussian fitting
Voigt fitting

For each detected peak the software reports:

Peak center
Peak position uncertainty
Peak height
Full Width at Half Maximum (FWHM)
Peak area
R² goodness of fit
Akaike Information Criterion (AIC)
Bayesian Information Criterion (BIC)
Time-Series Support

Two fitting modes are available:

Average Mode

Fits the averaged spectrum for each measurement spot
Maximizes signal-to-noise ratio

Per-Frame Mode

Fits every frame individually
Combines the per-frame centers with a robust estimator (trimmed mean by default; mean or median also available), reporting the aggregated center and its standard error
Preserves temporal variability such as SERS blinking behavior
Performance
Optional multiprocessing support
Parallel despiking
Parallel baseline correction
Parallel peak fitting
Automatic worker allocation based on available CPU cores
Output

The software produces:

CSV Results

peak_fits_<label>.csv

Containing:

Peak center
Uncertainty
FWHM
Height
Area
Fit quality metrics
Noise classification
Processing metadata
Quality Assurance Figures
Raw spectrum inspection
Smoothing comparison
Baseline correction validation
Spectral treatment summary
Peak fitting verification
Scientific Results

peak_centers_<label>.png

Displaying fitted Raman peak positions and uncertainties across all measurement spots.

Scientific Workflow

Raw Spectra

↓ Cosmic-Ray Removal

↓ Optional Savitzky-Golay Smoothing

↓ Baseline Correction

↓ Noise Estimation

↓ Peak Detection & Fitting

↓ Statistical Analysis

↓ CSV Export & Visualization

Installation

Required packages:

pip install numpy scipy matplotlib pybaselines lmfit

Clone the repository:

git clone https://github.com/yourusername/RamanPeakShiftAnalyzer.git
cd RamanPeakShiftAnalyzer
Usage

Process a directory of Renishaw exports (time-series or map):

python run.py path_to_data

Example:

python run.py C:\Raman\Data

Specify a fitting model:

python run.py Data --model voigt

Run parallel processing:

python run.py Data --parallel

Fit a specific spectral region:

python run.py Data --peak-min 1150 --peak-max 1250

Use per-frame fitting:

python run.py Data --fit-mode perframe
Project Structure
run.py
    Main analysis pipeline

raman_analysis.py
    Spectral file readers and peak fitting routines

peak_fits_*.csv
    Quantitative fitting results

peak_centers_*.png
    Peak shift visualization

spectra_treatment_*.png
    Quality assurance figures

fit_qa_*.png
    Per-spot fit-quality figures
Software Engineering Highlights

This project demonstrates:

Scientific computing with Python
Signal processing
Spectral analysis
Statistical fitting
Multiprocessing and performance optimization
Automated quality assurance workflows
Robust error handling
Modular software design
Reproducible research practices
Author

Christian Tian

Department of Chemistry

University of Victoria

Victoria, British Columbia, Canada

License

This project is provided for educational and research purposes. Please contact the author regarding reuse or commercial applications.
