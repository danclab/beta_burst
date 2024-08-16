"""Run time-frequency analysis using superlet algorithm and extract bursts.

We will also study the bursts waveforms and some features.
"""

from matplotlib import pyplot as plt
from mne.io import read_epochs_fieldtrip
from mne import create_info
import numpy as np
from betaburst.detection.burst_detection import TfBursts
from fooof import FOOOF

# %% [markdown]
# ### Loading the data
# Here we load the EEG data in FieldTrip format.
#
# Regardless of the data format, the important features necessary for further analyses are:
#  * Raw trial signal (here from the single sensor)
#  * The array with time points of the trial (**trial x time**)
#  * sampling frequency (*sfreq* variable)

# %%
ch_names = [
    "L1",
    "L2",
    "L3",
    "L4",
    "R1",
    "R2",
    "R3",
    "R4",
    "F3",
    "Fz",
    "F4",
    "C3",
    "C4",
    "Cz",
    "AA",
]
sfreq = 512
info = create_info(ch_names, sfreq, ch_types="eeg", verbose=None)
epochs = read_epochs_fieldtrip(
    "resting_ft_EpochedConc_s06_MED_ON.mat",
    info,
    data_name="resting_ft_EpochedConc",
    trialinfo_column=6,
)
times = epochs.times
trials = epochs.get_data()[:, 11, :]
fs = epochs.info["sfreq"]

# %% [markdown]
# ### Superlet Time-Frequency decomposition
# 1. Set the max frequency (*max_freq* variable)
# 2. Create an array of frequencies of interest (*foi* variable) and create scales used by the superlet algorithm
# 3. Apply the superlet TF decomposition algorithm
#     * from our testing, the max order of 40, and 4 cycles work well with burst activity in beta range
#     * adaptive version of algorithm is mandatory
# 4. Extract the absolute value (power) from the results
# 5. Converting to a single precision float may help with memory and storage issues for big datasets
#
# The TF decomposed dataset has, **trials x frequency points x time** dimensions

# %%
# Burst detection parameters
max_freq = 120
foi = np.linspace(1, max_freq, 400)
freqs = foi
upto_gamma_band = np.array([8, 40])
upto_gamma_range = np.where(
    np.logical_and(freqs >= upto_gamma_band[0], freqs <= upto_gamma_band[1])
)[0]

bm = TfBursts(
    fs,
    freqs=freqs,
    fr_band=upto_gamma_band,
    band_search_range=upto_gamma_range,
    band_limits=[8, 10, 35],
    remove_fooof=False,
)

# %%
tf_trials = bm._apply_tf(epochs, order_max=40, c_1=4)

# %% [markdown]
# ### Defining the frequency range for the burst detection
# Variable *search_range* defines a space in frequency dimension in which bursts are going to be detected and corresponds to the indices in the *foi* variable. The range is 3 Hz wider than frequency range of interest (variable *beta_lims*) to avoid the edge effect, where bursts on the edge of the frequency of interest may be rejected or the TF might be incomplete. Only the bursts with peak frequency within *beta_lims* are going to be kept.

# %%
search_range = np.where((foi >= 10) & (foi <= 33))[0]
beta_lims = [13, 30]

# %% [markdown]
# ### Computing the aperiodic threshold
# In our method we use the aperiodic threshold (1/f fit) to subtract the single trial time-frequency spectrum. The PSD has to be in the same scale as the TF decomposition. To achieve that, the trials are averaged in time and across the trials (subset).

# %%
average_psd = np.average(tf_trials, axis=(2, 0))

# %%
f, ax = plt.subplots(1, 1, figsize=(10, 5))
ff = FOOOF()
ff.fit(foi, average_psd, [1, 120])
ap = 10**ff._ap_fit
ax.plot(foi, average_psd, c="red", label="averaged PSD - 10 trials")
ax.plot(foi, ap, c="blue", label="FOOOF AP FIT")
ax.legend()

# %%
f, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(foi[search_range], ap[search_range], c="blue", label="FOOOF AP FIT")
ax.plot(
    foi[search_range],
    average_psd[search_range],
    c="red",
    label="averaged PSD - 10 trials",
)
ax.legend()

residual_search_power = average_psd[search_range] - ap[search_range]

# %% [markdown]
# ### BURST EXTRACTION
# see: https://github.com/danclab/burst_detection#burst-detection-algorithm---usage-notes
#
# The *search_range* selected aperiodic spectrum has to be reshaped to match the frequency dimension of the TF trials.
#

# %%
bursts = bm.burst_extraction(trials, band="beta", std_noise=2, regress_ERF=False)[0]

# %%
bursts.keys()

# %%
f, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.hist(bursts["fwhm_time"], bins=20, color="#DFFF00", edgecolor="black", linewidth=0.2)
ax.set_title("Burst Duration")

# %%
f, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.hist(bursts["fwhm_freq"], bins=10, color="#DFFF00", edgecolor="black", linewidth=0.2)
ax.set_title("Frequency Span")

# %%
f, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.hist(bursts["peak_freq"], bins=18, color="#DFFF00", edgecolor="black", linewidth=0.2)
ax.plot(
    foi[search_range],
    residual_search_power * 0.75e2,
    lw=0.5,
    c="black",
    label="scaled periodic power",
)
ax.legend()
ax.set_title("Peak Frequency")

# %%
f, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.hist(
    bursts["peak_amp_base"], bins=20, color="#DFFF00", edgecolor="black", linewidth=0.2
)
ax.set_title("Peak Amplitude")

# %%
f, ax = plt.subplots(1, 1, figsize=(5, 5))
mean_waveform = np.mean(bursts["waveform"], axis=0)
ax.plot(bursts["waveform_times"], bursts["waveform"].T, lw=0.2)
ax.plot(bursts["waveform_times"], mean_waveform, lw=1, c="black")
ax.set_xlim(bursts["waveform_times"][0], bursts["waveform_times"][-1])
ax.set_ylim(-20, 20)

# %%
trial = 10
tr_map = bursts["trial"] == trial
f, ax = plt.subplots(1, 1, figsize=(15, 7))
spectrum = ax.imshow(
    tf_trials[trial][search_range],
    cmap="Wistia",
    origin="lower",
    extent=(times[0], times[-1], foi[search_range][0], foi[search_range][-1]),
    aspect="auto",
)
ax.scatter(
    bursts["peak_time"][tr_map],
    bursts["peak_freq"][tr_map],
    s=10,
    marker="+",
    c="black",
)
ax.axhline(beta_lims[0], lw=0.5, c="black")
ax.axhline(beta_lims[1], lw=0.5, c="black")
plt.colorbar(spectrum, ax=ax)
plt.tight_layout()

# %%
