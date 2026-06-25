"""raman_legacy.py -- additional Raman/SERS helpers.

These functions are not used by run.py, whose pipeline relies on
read_tseries_renishaw() and describe_peak() from raman_analysis.py. Import from
this module when you need the extras: map and spot readers, plotting and
animation helpers, violin plots, Lumerical near-field fitting, or FFT
correlation.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib
import os
from natsort import natsorted
import numpy as np
import h5py
import scipy
from scipy.signal import savgol_filter
from scipy.stats import norm
from matplotlib import animation
import matplotlib
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
     'axes.unicode_minus': False, #to allow for minus ticks
}
plt.rcParams.update(brolo_style)


# =============================================================================
# Reading data
# =============================================================================
# path=r"Z:\Brolo-backup\Arash\Hannah\Map data\2023_05_26_Map2_633nm_50x_05p_1s_1200lmm_Copy.txt"
# path=r"C:\Users\aphys\OneDrive - University of Victoria\Experiments\Apr 4\633\0.5mM-5nM"
# acqu='temporal'
def read_renishaw(path, acqu='single', cut_spectra=None):
    if acqu == 'map':
        txt_file = np.loadtxt(path)
        if txt_file.ndim != 2 or txt_file.shape[0] < 3:
            raise ValueError(f"Invalid map file shape: {txt_file.shape}")

        wn_counts = 1
        stop = 0
        while stop == 0 and (wn_counts + 1) < len(txt_file):
            if (txt_file[wn_counts + 1, 0] == txt_file[wn_counts, 0]) and \
                    (txt_file[wn_counts + 1, 1] == txt_file[wn_counts, 1]):
                wn_counts = wn_counts + 1
            else:
                stop = 1
        wn_counts = wn_counts + 1
        xtemp = np.sort(txt_file[:, 0])
        ytemp = np.sort(txt_file[:, 1])
        tempindx = 0
        indx = [0]
        tempindy = 0
        indy = [0]
        for i in range(len(xtemp) - 1):
            if xtemp[i + 1] == xtemp[i]:
                indx.append(tempindx)
            else:
                tempindx = tempindx + 1
                indx.append(tempindx)
        for i in range(len(ytemp) - 1):
            if ytemp[i + 1] == ytemp[i]:
                indy.append(tempindy)
            else:
                tempindy = tempindy + 1
                indy.append(tempindy)
        spectra = np.zeros([tempindx + 1, tempindy + 1, wn_counts])
        wn = txt_file[:wn_counts, 2]
        i = 0
        for j in range(len(txt_file)):
            indxx = indx[np.where(xtemp == txt_file[j, 0])[0][0]]
            indyy = indy[np.where(ytemp == txt_file[j, 1])[0][0]]
            spectra[indxx, indyy, i] = txt_file[j, 3]
            if i < (wn_counts - 1):
                i = i + 1
            else:
                i = 0
    if acqu == 'temporal':
        txt_file = np.loadtxt(path)
        if txt_file.ndim != 2 or txt_file.shape[0] < 3:
            raise ValueError(f"Invalid temporal file shape: {txt_file.shape}")

        wn_counts = 1
        stop = 0
        while stop == 0 and (wn_counts + 1) < len(txt_file):
            if (txt_file[wn_counts + 1, 0] == txt_file[wn_counts, 0]) and \
                    (np.abs((txt_file[wn_counts + 1, 1] - txt_file[wn_counts, 1]) - \
                            (txt_file[wn_counts, 1] - txt_file[wn_counts - 1, 1])) < 10):
                wn_counts = wn_counts + 1
            else:
                stop = 1
        block = wn_counts + 1            # rows per spectrum
        rpt = len(txt_file) // block
        spectra = []
        wn_full = txt_file[0:block, 1]
        wn = wn_full if cut_spectra is None else wn_full[:cut_spectra]
        for i in range(rpt):
            seg = txt_file[i * block:(i + 1) * block, 2]
            spectra.append(seg if cut_spectra is None else seg[:cut_spectra])
    if acqu == 'single':
        txt_file = np.loadtxt(path)
        if txt_file.shape[1] == 2:
            wn = txt_file[:, 0]
            spectra = txt_file[:, 1]
        else:

            if cut_spectra == None:
                wn = txt_file[:, 1]
                spectra = txt_file[:, 2]
            else:
                wn = txt_file[:, 1][:cut_spectra]
                spectra = txt_file[:, 2][:cut_spectra]
    return (spectra, wn)


# =============================================================================
# Making animation from plots
# =============================================================================
# def plot_anim(spectra, wn, interval_t=100, frame_ps=10, save_anim=0, path="c:", name='a', user_ylim=None):
#     num_frames = len(spectra)
#
#     def animate_func(t):
#         # plt.pause(0.2)
#         ax.clear()
#         ax.plot(wn, spectra[t])
#         # ax.plot(wn[0],np.squeeze(spectra[1][0][t]),color="red",linewidth=1)
#         # ax.legend(["1mM scaled","100uM"],fontsize=18)
#         ax.set_xlabel("Wavenumber (cm-1)")
#         ax.set_ylabel("Counts")
#         ax.legend(f"{t}")
#         if user_ylim != None:
#             ax.set_ylim(0, user_ylim)
#         # ax.plot(bw_wn,500*((bw_s))/np.amax(bw_s))
#
#     fig = plt.figure()
#     ax = plt.axes()
#     line_ani = animation.FuncAnimation(fig, animate_func, interval=interval_t, frames=num_frames)
#     f = path + "\\" + name + '.gif'
#     # writervideo = animation.FFMpegWriter(fps=20)
#     writergif = animation.PillowWriter(fps=10)
#     if save_anim == 1:
#         matplotlib.rcParams['animation.ffmpeg_path'] = "C:\\ffmpeg\\ffmpeg-master-latest-win64-gpl\\bin\\ffmpeg.exe"
#         # line_ani.save(path+"\\"+'a.gif', writer=writervideo)
#         line_ani.save(f, writer=writergif)
#     return (line_ani)


# =============================================================================
# Peak selection
# =============================================================================
# def peak_find(wn, spectra, peaks_wn, x_min_set, x_max_set, interval=10, fit_model=[1], fit_peak=0, show_image=0):
#     # Fitting models
#
#     # Model 1
#     def gaussian(x, a, x0, sigma):
#         return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
#
#     # Model 2
#     def lorentzian(x, ampL1, cenL1, widL1):
#         return (ampL1 * widL1 ** 2 / ((x - cenL1) ** 2 + widL1 ** 2))
#
#     # Model 3
#     def pearson(x, amplitude, mean, sigma, power):
#
#         return amplitude * (1 + ((x - mean) / sigma) ** 2) ** (-power)
#
#     # Model 4
#     def combined_gaussian(x, amp1, cen1, wid1, amp2, cen2, wid2):  # Model 4
#         return gaussian(x, amp1, cen1, wid1) + gaussian(x, amp2, cen2, wid2)
#
#     # Model 5
#     def voigt(x, ampG1, cenGL, sigmaG1, ampL1, widL1):
#         return (ampG1 * (1 / (sigmaG1 * (np.sqrt(2 * np.pi)))) * (np.exp(-((x - cenGL) ** 2) / ((2 * sigmaG1) ** 2)))) + \
#             ((ampL1 * widL1 ** 2 / ((x - cenGL) ** 2 + widL1 ** 2)))
#
#         # Model 6
#
#     def double_gaussian(x, amp1, cen1, wid1, amp2, cen2, wid2):
#         f1 = gaussian
#         f2 = gaussian
#         return f1(x, amp1, cen1, wid1) + f2(x, amp2, cen2, wid2)
#
#     # Model 7
#     def double_lorentzian(x, amp1, cen1, wid1, amp2, cen2, wid2):
#         f1 = lorentzian
#         f2 = lorentzian
#         return f1(x, amp1, cen1, wid1) + f2(x, amp2, cen2, wid2)
#
#     # Model 8
#     def lorentzian_gaussian(x, amp1, cen1, wid1, amp2, cen2, wid2):
#         f1 = gaussian
#         f2 = lorentzian
#         return f1(x, amp1, cen1, wid1) + f2(x, amp2, cen2, wid2)
#
#     # Model 9
#     def triple_fitting(x, amp1, cen1, wid1, amp2, cen2, wid2, amp3, cen3, wid3):
#         f1 = gaussian
#         f2 = gaussian
#         f3 = gaussian
#         return f1(x, amp1, cen1, wid1) + f2(x, amp2, cen2, wid2) + f3(x, amp3, cen3, wid3)
#
#     # Cutting the appropriate region
#     cr_s = []
#     cr_ws = []
#     # peaks_wn=[1100]
#     # x_min_set=[1035]
#     # x_max_set=[1125]
#     for ii in range(len(peaks_wn)):
#         x_min = x_min_set[ii]
#         x_max = x_max_set[ii]
#         s_indice = np.where((wn >= x_min) & (wn <= x_max))[0]
#         cr_s.append(spectra[s_indice])
#         cr_ws.append(wn[s_indice])
#     num_peaks = len(peaks_wn)
#
#     # Method1: Maximum counts
#     peak_counts_raw = []
#     for i in range(num_peaks):
#         ind_wn = int(np.where(np.abs(wn - peaks_wn[i]) == np.min(np.abs(wn - peaks_wn[i])))[0])
#         peak_counts_raw.append(np.amax(spectra[ind_wn - interval:ind_wn + interval]))
#
#     # Method2: Maximum of fitting and area
#     # fitting_model= combined_gaussian
#     # test_image=1
#     area = []
#     fit_params = []
#     residu = []
#     cov = []
#     y_pred = []
#     n_c = []
#     if fit_peak == 1:
#         for j in range(num_peaks):
#             model = fit_model[j]
#             if model == 1:
#                 fitting_model = gaussian
#                 pp0 = [peak_counts_raw[j], 0.5 * (x_min_set[j] + x_max_set[j]), 6]
#                 n_curve = 1
#             if model == 2:
#                 fitting_model = lorentzian
#                 pp0 = [peak_counts_raw[j], 0.5 * (x_min_set[j] + x_max_set[j]), 6]
#                 n_curve = 1
#             if model == 3:
#                 fitting_model = pearson
#                 pp0 = [peak_counts_raw[j], 0.5 * (x_min_set[j] + x_max_set[j]), 6, 1]
#                 n_curve = 1
#             if model == 4:
#                 fitting_model = combined_gaussian
#                 pp0 = [peak_counts_raw[j], 0.5 * (x_min_set[j] + x_max_set[j]), 6, peak_counts_raw[j],
#                        0.5 * (x_min_set[j] + x_max_set[j]), 6]
#                 n_curve = 2
#             if model == 5:
#                 fitting_model = voigt
#                 pp0 = [peak_counts_raw[j], 0.5 * (x_min_set[j] + x_max_set[j]), 6, peak_counts_raw[j], 6]
#                 n_curve = 1
#             if model == 6:
#                 f1 = gaussian
#                 f2 = gaussian
#                 fitting_model = double_gaussian
#                 pp0 = [peak_counts_raw[j], 0.5 * (x_min_set[j] + x_max_set[j]), 6, peak_counts_raw[j],
#                        0.5 * (x_min_set[j] + x_max_set[j]), 1]
#                 n_curve = 2
#             if model == 7:
#                 f1 = lorentzian
#                 f2 = lorentzian
#                 fitting_model = double_lorentzian
#                 pp0 = [peak_counts_raw[j], 0.5 * (x_min_set[j] + x_max_set[j]), 6, peak_counts_raw[j],
#                        0.5 * (x_min_set[j] + x_max_set[j]), 1]
#                 n_curve = 2
#             if model == 8:
#                 f1 = gaussian
#                 f2 = lorentzian
#                 fitting_model = lorentzian_gaussian
#                 pp0 = [peak_counts_raw[j], 0.5 * (x_min_set[j] + x_max_set[j]), 6, peak_counts_raw[j],
#                        0.5 * (x_min_set[j] + x_max_set[j]), 1]
#                 n_curve = 2
#             if model == 9:
#                 f1 = gaussian
#                 f2 = gaussian
#                 f3 = gaussian
#                 fitting_model = triple_fitting
#                 pp0 = [peak_counts_raw[j], 0.5 * (x_min_set[j] + x_max_set[j]), 6, peak_counts_raw[j],
#                        0.5 * (x_min_set[j] + x_max_set[j]), 6 \
#                     , peak_counts_raw[j], 0.5 * (x_min_set[j] + x_max_set[j]), 10]
#                 n_curve = 3
#             # if peak_counts_raw[0]>10:
#             # params, pcov1 = scipy.optimize.curve_fit(fitting_model, cr_ws[j], cr_s[j],maxfev=1800000,p0=pp0,bounds=((0,0,0,0,0,0),(np.inf,np.inf,np.inf,np.inf,np.inf,8))) #Add limitations to make sure that it fits well (based on physical facts)
#             params, pcov1 = scipy.optimize.curve_fit(fitting_model, cr_ws[j], cr_s[j], maxfev=1800000, p0=pp0,
#                                                      bounds=((0, 0, 0), (np.inf, np.inf, np.inf)))
#             #
#             fit_params.append(params)
#             cov.append(pcov1)
#             area.append(np.trapz(wn, fitting_model(wn, *params)))
#             y_pred.append(fitting_model(wn, *params))
#             res = scipy.stats.linregress(spectra, y_pred[j])
#             residu.append(res)
#             n_c.append(n_curve)
#         if show_image == 1:
#             fig, ax = plt.subplots()
#             ax.plot(wn, spectra)
#             for j in range(num_peaks):
#                 ax.plot(wn, y_pred[j], linewidth=1, color='black')
#                 if n_c[j] == 2:
#                     ax.plot(wn, f1(wn, *fit_params[j][:3]), '--')
#                     ax.plot(wn, f2(wn, *fit_params[j][3:6]), '--')
#                 if n_c[j] == 3:
#                     ax.plot(wn, f1(wn, *fit_params[j][:3]), '--')
#                     ax.plot(wn, f2(wn, *fit_params[j][3:6]), '--')
#                     ax.plot(wn, f3(wn, *fit_params[j][6:9]), '--')
#     return (fit_params, cov, area, y_pred, residu, peak_counts_raw)


# =============================================================================
# This code reads data collected from differenct spots,
#  - Spot=0: all spots or specify a particular spot
#  - time=0: all times or specify a particular time
#  - The wavenumber for all the spectra between different spots should be the
#    same
#  - Include the string "spot" in the of different spots, so the spots name
#    should be like, spot(1), spot(2) and ...
# =============================================================================
def read_spots(path, spot=0, time=0):
    folders_temp = natsorted(os.listdir(path), key=lambda y: y.lower())
    if spot > 0:
        folders_temp = [folders_temp[spot - 1]]
    folders = []
    spectra = []
    wn = []
    i = 0
    for a in (folders_temp):
        if (('spot' in a) or ('Spot' in a)):
            folders.append(a)
    for folder in (folders):
        folder_path = os.path.join(path, folder)
        f1 = os.listdir(folder_path)
        f1 = natsorted(f1, key=lambda y: y.lower())
        f1 = [file for file in f1 if file.lower().endswith('.txt')]
        if time > 0:
            f1 = [f1[time - 1]]
        spectra.append([])
        for list_files1 in f1:
            file_path = os.path.join(folder_path, list_files1)
            arr = np.loadtxt(file_path)
            spectra[i].append(np.flip(arr[:, 1]))
            if (list_files1 == f1[0] and folder == folders[0]):
                wn.append(np.flip(arr[:, 0]))
        i = i + 1
    return spectra, wn


# % =============================================================================
# Multi gaussian functions
# %  =============================================================================
def g_fit(x, *g_params):
    y = np.zeros(len(x))
    for i in range(int(len(g_params) / 3)):
        c = g_params[3 * i]
        a = g_params[(3 * i) + 1]
        sig = g_params[(3 * i) + 2]
        y = y + a * np.exp(-((x - c) / sig) ** 2)
    return (y)


# =============================================================================
# Voigt function
# =============================================================================
def voigt(x, ampG1, cenG1, sigmaG1, ampL1, cenL1, widL1):
    return (ampG1 * (1 / (sigmaG1 * (np.sqrt(2 * np.pi)))) * (np.exp(-((x - cenG1) ** 2) / ((2 * sigmaG1) ** 2)))) + \
        ((ampL1 * widL1 ** 2 / ((x - cenL1) ** 2 + widL1 ** 2)))


# a1, pcov = scipy.optimize.curve_fit(voigt, cr_ws, cr_s[i],p0=[ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  1.12191586e+04,
#     1.07793023e+03, -5.17346931e+00],maxfev=18000)
# area_s=np.trapz(wn_s,voigt(wn_s,*a1))
# =============================================================================
# Smoothing
# =============================================================================
def smooth_spectrum(y):
    y_smooth = savgol_filter(y, window_length=15, polyorder=2, mode='nearest')
    return y_smooth

# %% =============================================================================
# Violinplot
# =============================================================================
# def vplot(data, ax1, xlabel='x', ylabel='y'):
#     import matplotlib.pyplot as plt
#     import numpy as np
#
#     def adjacent_values(vals, q1, q3):
#         upper_adjacent_value = q3 + (q3 - q1) * 1.5
#         upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])
#
#         lower_adjacent_value = q1 - (q3 - q1) * 1.5
#         lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
#         return lower_adjacent_value, upper_adjacent_value
#
#     # def set_axis_style(ax, labels):
#     #     ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
#     #     ax.set_xlim(0.25, len(labels) + 0.75)
#     #     ax.set_xlabel('Sample name')
#
#     # fig, ax1 = plt.subplots(figsize=(9,4))
#
#     ax1.set_xlabel(xlabel)
#     ax1.set_ylabel(ylabel)
#     ax1.violinplot(data, showmeans=False, widths=0.9)
#     ax1.set_xticks([y + 1 for y in range(len(data))])
#     ax1.grid(visible=None)
#     # set style for the axes
#     # labels = str(np.arange(len(data)))
#     # set_axis_style(ax1, labels)
#
#     plt.subplots_adjust(bottom=0.15, wspace=0.05)
#     return (ax1)

# % =============================================================================
# Binning function based on lower and higher (data independent)
# =============================================================================
def binning(data, lower_b=0, higher_b=10, num_bins=5, pop_type='log'):
    # Create constant bin edges
    bins_edges = np.linspace(lower_b, higher_b, num_bins + 1)

    # Calculate histogram using numpy with fixed bins
    counts, _ = np.histogram(data, bins=bins_edges)

    if pop_type == 'log':
        counts = np.log10(counts, where=counts > 0)
    elif pop_type == 'density':
        bin_width = bins_edges[1] - bins_edges[0]
        counts = counts / (len(data) * bin_width)

    return counts, bins_edges

# =============================================================================
# Reading Lumerical saved data
# =============================================================================
def load_lumerical(path, partial=0):
    with h5py.File(path, 'r') as file:
        a = file['E'][:]
        if partial == 1:
            E = file['E'][:, :, :, 23]
            wl = file['lambda'][0]
        else:
            E = file['E'][:]
            wl = file['lambda'][:]
        x = file['x'][:]
        y = file['y'][:]
        z = file['z'][:]
        wl = file['lambda'][:]
    return E, wl, x, y, z

#% =============================================================================
# Fitting the HS by generalized gaussian function
# =============================================================================
def gg_fit(path, vis=1):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    import h5py

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 20,
        'axes.titlesize': 18,
        'axes.labelsize': 18,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 15,
        'figure.titlesize': 20,
    })

    # Load the data
    def load_data(file_path):
            data=h5py.File(file_path, 'r')
            E = np.array(data['E'])
            E=E**4
            wl = 3e8/np.array(data['lambda'])
            z = np.array(data['x'])
            y = np.array(data['y'])
            x = np.array(data['z'])
            gap= np.array(data['gap'])
            radius=np.array(data['radi'])
            return E, wl, x, y, z,radius,gap

    E, wl, xo, yo, zo,radius, gap = load_data(path)
    x = np.arange(xo.shape[1])
    y = np.arange(yo.shape[1])
    z = np.arange(zo.shape[1])

    # Extract the core region where E > 0
    core_indices = np.where(E > 0)
    if not core_indices[0].size:
        raise ValueError("No core region found where E > 0")

    x_min, x_max = core_indices[0].min(), core_indices[0].max()
    y_min, y_max = core_indices[1].min(), core_indices[1].max()
    z_min, z_max = core_indices[2].min(), core_indices[2].max()

    E_core = E[x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1]
    x_core = x[x_min:x_max + 1]
    y_core = y[y_min:y_max + 1]
    z_core = z[z_min:z_max + 1]

    # Prepare the data for fitting
    x_mesh, y_mesh, z_mesh = np.meshgrid(x_core, y_core, z_core, indexing='ij')
    x_flat = x_mesh.flatten()
    y_flat = y_mesh.flatten()
    z_flat = z_mesh.flatten()
    E_flat = E_core.flatten()
    # Define the generalized Gaussian function
    def generalized_gaussian_model(X, A, mu_x, sigma_x, beta_x, mu_y, sigma_y, beta_y, mu_z, sigma_z, beta_z, B):
        x, y, z = X
        generalized_gauss_x = np.exp(-((np.abs(x - mu_x) / sigma_x) ** beta_x))
        generalized_gauss_y = np.exp(-((np.abs(y - mu_y) / sigma_y) ** beta_y))
        generalized_gauss_z = np.exp(-((np.abs(z - mu_z) / sigma_z) ** beta_z))
        return A * generalized_gauss_x * generalized_gauss_y * generalized_gauss_z + B

    # Initial parameters for fitting
    initial_params = [
        np.max(E_flat),
        np.mean(x_core), np.std(x_core), 2,
        np.mean(y_core), np.std(y_core), 2,
        np.mean(z_core), np.std(z_core), 2,
        0.5
    ]

    # Fit the generalized Gaussian model
    bounds = (
        [0, x_core.min(), 0, 0, y_core.min(), 0, 0, z_core.min(), 0, 0, 0],
        [np.inf, x_core.max(), np.inf, np.inf, y_core.max(), np.inf, np.inf, z_core.max(), np.inf, np.inf, 1]
    )
    popt, _ = curve_fit(
        generalized_gaussian_model, (x_flat, y_flat, z_flat), E_flat,
        p0=initial_params, bounds=bounds, maxfev=20000
    )

    # Predictions from the fitted model
    E_pred_flat = generalized_gaussian_model((x_flat, y_flat, z_flat), *popt)
    E_pred = E_pred_flat.reshape(E_core.shape)

    # Calculate R-squared value
    ss_res = np.sum((E_flat - E_pred_flat) ** 2)
    ss_tot = np.sum((E_flat - np.mean(E_flat)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    print(f'R-squared (Generalized Gaussian): {r_squared:.4f}')

    # Visualization
    if vis == 1:
        fig, axes = plt.subplots(figsize=(20, 10))
        z_mid = E_core.shape[2] // 2
        y_mid = E_core.shape[1] // 2
        axes.plot((E_core[:, y_mid, z_mid]))
        axes.plot((E_pred[:, y_mid, z_mid]))
        axes.set_title('Original XY Plane')
        plt.show()
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        z_mid = E_core.shape[2] // 2
        im0 = axes[0].imshow(E_core[:, :, z_mid], cmap='plasma')
        axes[0].set_title('Original XY Plane')
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(E_pred[:, :, z_mid], cmap='plasma')
        axes[1].set_title(f'Predicted XY Plane (R^2: {r_squared:.4f})')
        fig.colorbar(im1, ax=axes[1])

        plt.show()
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        y_mid = E_core.shape[1] // 2
        im0 = axes[0].imshow(E_core[:, y_mid, :], cmap='plasma')
        axes[0].set_title('Original XZ Plane')
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(E_pred[:, y_mid, :], cmap='plasma')
        axes[1].set_title(f'Predicted XZ Plane (R^2: {r_squared:.4f})')
        fig.colorbar(im1, ax=axes[1])

        plt.show()
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        x_mid = E_core.shape[0] // 2
        im0 = axes[0].imshow(E_core[x_mid, :, :], cmap='plasma')
        axes[0].set_title('Original YZ Plane')
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(E_pred[x_mid, :, :], cmap='plasma')
        axes[1].set_title(f'Predicted YZ Plane (R^2: {r_squared:.4f})')
        fig.colorbar(im1, ax=axes[1])

        plt.show()
        plt.close(fig)

    fitting_params = {
        'A': popt[0],
        'mu_x': popt[1],
        'sigma_x': popt[2],
        'beta_x': popt[3],
        'mu_y': popt[4],
        'sigma_y': popt[5],
        'beta_y': popt[6],
        'mu_z': popt[7],
        'sigma_z': popt[8],
        'beta_z': popt[9],
        'B': popt[10]
    }

    return r_squared, fitting_params, E_pred,radius, gap

#% Read the raw data form txt file (Renishaw raman microscope)
#% Finding the peak counts, integral
def peak_find(wn,sig,wn_int,interval):
    import numpy as np
    wn_ind=np.where(np.abs(wn-wn_int)==np.min(np.abs(wn-wn_int)))[0][0]
    interval=int(interval//(wn[1]-wn[0]))
    lo = max(0, wn_ind - interval)
    hi = min(len(sig), wn_ind + interval + 1)
    count=np.max(sig[lo:hi])
    return count


#%============================================================================
# Making animation from plots
# =============================================================================

def plot_anim(spectra, wn, frame_ps=10, save_anim=0, path="C:\\Users\\aphys\\Documents\\", name='a'):
    num_frames = len(spectra)
    fig, ax = plt.subplots()

    def animate_func(t):
        ax.clear()
        ax.plot(wn, spectra[t])
        ax.set_xlabel("Wavenumber (cm-1)")
        ax.set_ylim([0, 2500])  # Correct usage of set_ylim
        ax.set_ylabel("Counts")

    line_ani = animation.FuncAnimation(fig, animate_func, frames=num_frames, interval=100)

    if save_anim == 1:
        f = path + name + '.gif'
        writergif = animation.PillowWriter(fps=frame_ps)
        matplotlib.rcParams['animation.ffmpeg_path'] = "C:\\ffmpeg\\ffmpeg-master-latest-win64-gpl\\bin\\ffmpeg.exe"
        line_ani.save(f, writer=writergif)
        plt.close(fig)

    plt.show()
    return line_ani


#%% ============================================================================
# Brolo standard plot
# =============================================================================
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, FuncFormatter, ScalarFormatter


class ScalarFormatterWithMath(ScalarFormatter):
    def _set_format(self):
        self.format = '$%1.1f$'


def format_plot(ax):
    """
    Apply Brolo Group formatting criteria to an existing plot.

    Parameters:
    ax (matplotlib.axes.Axes): The axes to format.
    """
    # Set axis labels and title font sizes if they exist
    if ax.get_xlabel():
        ax.set_xlabel(ax.get_xlabel(), fontsize=24, fontweight='bold', labelpad=10)
    if ax.get_ylabel():
        ax.set_ylabel(ax.get_ylabel(), fontsize=24, fontweight='bold', labelpad=10)
    if ax.get_title():
        ax.set_title(ax.get_title(), fontsize=24, fontweight='bold', pad=20)

    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=20, width=2, length=6)
    ax.tick_params(axis='both', which='minor', labelsize=16, width=1, length=3)

    # Set axis line width
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    # Remove gridlines
    ax.grid(False)

    # Set data point size and error bars
    for line in ax.get_lines():
        line.set_markersize(10)
        line.set_markeredgewidth(1.75)
        line.set_linewidth(2)
        line.set_color('red')

    # Set limits and ticks
    ax.set_xlim(auto=True)
    ax.set_ylim(auto=True)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # Round the values on the ticks
    ax.xaxis.set_major_formatter(ScalarFormatterWithMath())
    ax.yaxis.set_major_formatter(ScalarFormatterWithMath())
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(0, 0))

    # Adjust the plot to fit within the figure area
    plt.tight_layout()


# Example usage
# x = np.linspace(0, 10, 100)
# y = np.sin(x)
# yerr = 0.1 * np.ones_like(y)
#
# fig, ax = plt.subplots()
# ax.errorbar(x, y, yerr=yerr, fmt='o', color='blue', ecolor='black', capsize=5)
# ax.set_xlabel('Wavelength / nm')
# ax.set_ylabel('Intensity / a.u.')
# ax.set_title('Figure 1: Example of a graph depicting a single data set')
#
# format_plot(ax)
# plt.show()

#%% ============================================================================
# Violin plots
# =============================================================================
def vplot(data):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    """
    Generates a gradient bar plot with an additional bar plot on top and adds mean value lines.
    A color bar is added to the right side of the bottom axis to indicate the color gradient.

    Parameters:
        data (list of lists): A list containing 20 lists, each with 2000 numerical values.
    """
    # Normalize data
    normalized_data = [np.array(spot) / np.mean(spot) for spot in data]
    max_values = [np.max(spot) for spot in data]
    mean_values = [np.mean(spot) for spot in data]

    # Create figure and gridspec layout
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(2, 2, width_ratios=[1, 0.05], height_ratios=[1, 3], hspace=0.1, wspace=0.05)

    # Top bar plot (ax2)
    ax2 = fig.add_subplot(gs[0, 0])
    ax2.bar(range(len(data)), mean_values, color='red', alpha=1)
    ax2.set_ylabel("Mean counts", fontsize=24)
    ax2.set_xticks(range(len(data)))
    ax2.set_ylim(0,np.max(mean_values)+150)
    ax2.set_xticklabels([])  # Hide x-axis labels on the top subplot
    ax2.set_yticks([])

    for i, mean_val in enumerate(mean_values):
        ax2.text(i, mean_val, str(int(mean_val)), ha='center', va='bottom', rotation=90, color='black', fontsize=16)

    # Gradient bar plot (ax1)
    ax1 = fig.add_subplot(gs[1, 0])
    for i, spot in enumerate(normalized_data):
        counts, bin_edges = np.histogram(spot, bins=500, range=(0, np.max(spot)))
        counts = counts / counts.max()  # Normalize counts to [0, 1] for color mapping
        for count, edge in zip(counts, bin_edges[:-1]):
            color = plt.cm.viridis(count)  # Map count to color
            ax1.bar(i, height=(edge - bin_edges[0]), width=0.5, bottom=edge, color=color, alpha=0.9)

        # Add mean value line
        ax1.plot([i - 0.4, i + 0.4], [1] * 2, color='black', linestyle='-', linewidth=2)

    # Set y-axis to logarithmic scale
    ax1.set_yscale('log')
    ax1.set_xlabel("Spot #", fontsize=24)
    ax1.set_ylabel("Logarithmic Normalized counts/ arb. unit", fontsize=24)
    # ax1.locator_params(axis='x', nbins=4)
    # ax1.locator_params(axis='y', nbins=4)
    ax1.set_xticks([0,4,9,14,19])#(range(len(data)))
    ax1.set_xticklabels(["1","5","10","15","20"])#(range(1, len(data) + 1))
    # ax1.set_yticks([1e-2, 5, 10, 15, 20])  # (range(len(data)))
    # ax1.set_yticklabels(["1", "5", "10", "15", "20"])  # (range(1, len(data) + 1))

    # Add color bar outside the main plotting area, aligned with bottom ax
    cbar_ax = fig.add_subplot(gs[:, 1])
    norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Density', fontsize=24)

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make room for color bar
    # plt.show()


# # Generate sample data for testing
# np.random.seed(0)
# sample_data = [np.random.normal(loc=i, scale=1.0, size=2000) for i in range(20)]
#
# # Run the function with the sample data
# gradient_barplot_with_top_bar(sample_data)


#%% ============================================================================
# Number of bins for histogram bins
# =============================================================================
def calculate_bins(data, method="doane"):
    n = len(data)

    if method == "rice":
        k = 2 * np.cbrt(n)
    elif method == "scott":
        h = 3.5 * np.std(data) / np.cbrt(n)
        k = (np.max(data) - np.min(data)) / h
    elif method == "freedman-diaconis":
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        h = 2 * iqr / np.cbrt(n)
        k = (np.max(data) - np.min(data)) / h
    elif method == "doane":
        g = np.mean(((data - np.mean(data)) / np.std(data)) ** 3)
        sigma_g = np.sqrt(6 * (n - 2) / ((n + 1) * (n + 3)))
        k = 1 + np.log2(n) + np.log2(1 + np.abs(g) / sigma_g)
    else:
        raise ValueError("Unknown method")

    return int(np.ceil(k))

# %% ============================================================================
# Number of bins for histogram bins
# =============================================================================
def gg_model(X,A, sigma_xy, beta_xy, sigma_z, beta_z, B,Gap):
        x, y, z = X

        generalized_gauss_x = np.exp(-((np.abs(x) / sigma_xy) ** beta_xy))
        generalized_gauss_y = np.exp(-((np.abs(y) / sigma_xy) ** beta_xy))
        generalized_gauss_z = np.exp(-((np.abs(z) / sigma_z) ** beta_z))

        if np.isscalar(z):
            if (z > Gap / 2) or (z < -Gap / 2):
                generalized_gauss_z = 0
        else:
            generalized_gauss_z[(z > Gap / 2) | (z < -Gap / 2)] = 0

        EF=A*np.log10(generalized_gauss_x * generalized_gauss_y * generalized_gauss_z+(B/A))
        vis=0
        if vis==1:
            try:
                import plotly.graph_objects as go
            except Exception:
                return EF

            custom_colorscale = [
                [0, 'rgba(0, 0, 0, 0)'],
                [0.1, 'rgba(0, 0, 255, 0.1)'],
                [0.2, 'rgba(0, 0, 255, 0.2)'],
                [0.3, 'rgba(0, 0, 255, 0.3)'],
                [0.4, 'rgba(0, 0, 255, 0.4)'],
                [0.5, 'rgba(0, 255, 0, 0.5)'],
                [0.6, 'rgba(0, 255, 0, 0.6)'],
                [0.7, 'rgba(255, 255, 0, 0.7)'],
                [0.8, 'rgba(255, 255, 0, 0.8)'],
                [0.9, 'rgba(255, 0, 0, 0.9)'],
                [1, 'rgba(255, 0, 0, 1)']
            ]

            x_flat = np.ravel(x)
            y_flat = np.ravel(y)
            z_flat = np.ravel(z)
            values_flat = np.ravel(EF)

            threshold = 0.01
            mask = values_flat > threshold
            x_flat = x_flat[mask]
            y_flat = y_flat[mask]
            z_flat = z_flat[mask]
            values_flat = values_flat[mask]

            fig = go.Figure(data=[go.Scatter3d(
                x=x_flat, y=y_flat, z=z_flat, mode='markers',
                marker=dict(size=2, color=values_flat, colorscale=custom_colorscale, opacity=0.3)
            )])
            fig.update_layout(
                title='3D Scatter Plot of gg_model Output',
                scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
            )
            fig.show()
        return EF
# %% ============================================================================
# Fast correlation calculation
# =============================================================================
import numpy as np

def corr_fft(signal, time_resolution, maximum_delay=0.1, mode="g2"):
    """
    mode:
      - "g2"     -> <I(t) I(t+tau)> / <I>^2       (baseline ~ 1)
      - "g2cov"  -> 1 + Cov[I,I_tau]/<I>^2        (baseline exactly 1)
    """
    x = np.asarray(signal, dtype=float)
    n = x.size
    mu = x.mean()

    # choose series for FFT
    y = x if mode == "g2" else (x - mu)

    # FFT length for linear autocorr (not circular)
    L = 1 << (2*n - 1).bit_length()   # next pow2 ≥ 2n-1
    Y = np.fft.rfft(y, L)
    ac = np.fft.irfft(Y * np.conj(Y), L)[:n]      # lags 0..n-1

    # unbiased per-lag normalization by overlaps
    overlaps = n - np.arange(n)
    ac /= overlaps

    # pick number of lags
    if maximum_delay is None:
        steps = n - 1
    else:
        steps = min(int(np.floor(maximum_delay / time_resolution)), n - 1)

    if mode == "g2":
        corr = ac[:steps+1] / (mu * mu)
    else:  # "g2cov"
        corr = 1.0 + ac[:steps+1] / (mu * mu)

    tau = time_resolution * np.arange(steps + 1)
    return corr, tau


