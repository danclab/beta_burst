# %%
import requests
import zipfile
import os
from matplotlib import pyplot as plt
import numpy as np

import mne
from mne.channels import make_standard_montage
from mne.io import read_raw_cnt

from betaburst.detection.burst_detection import TfBursts

# %% [markdown]
# ## Download data from the Zhou dataset
# [1] Zhou B, Wu X, Lv Z, Zhang L, Guo X (2016) A Fully Automated Trial Selection Method for Optimization of Motor Imagery Based Brain-Computer Interface. PLoS ONE 11(9). https://doi.org/10.1371/journal.pone.0162657

# %%
data_path = "https://ndownloader.figshare.com/files/3662952"

zip_file_path = "zhou_data.zip"
extract_folder = "zhou_data"

# Download the compressed archive
response = requests.get(data_path)
with open(zip_file_path, "wb") as file:
    file.write(response.content)

# Decompress the archive in extract folder
os.makedirs(extract_folder, exist_ok=True)
with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
    zip_ref.extractall(extract_folder)
os.remove(zip_file_path)

print(f"File downloaded and extracted to {extract_folder}")

# %% [markdown]
# ### Read data and extract epochs using [MNE](https://mne.tools/dev/) toolbox
# We will load data only from a single subject and a single session.

# %%
subject = 2
session = "1A"
fname = f".\\zhou_data\\data\\S{subject}_{session}.cnt"
raw = read_raw_cnt(fname, preload=True, eog=["VEOU", "VEOL"])
stim = raw.annotations.description.astype(np.dtype("<10U"))
stim[stim == "1"] = "left_hand"
stim[stim == "2"] = "right_hand"
stim[stim == "3"] = "feet"
raw.annotations.description = stim
raw.set_montage(make_standard_montage("standard_1005"))
fs = raw.info["sfreq"]

# %% [markdown]
# #### Notch filtering to remove line noise artifacts

# %%
raw.notch_filter(freqs=[50])

# %% [markdown]
# #### Create epochs

# %%
event_id = {"left_hand": 1, "right_hand": 2}

# Keep only left/right hand motor imagery trials
events, _ = mne.events_from_annotations(raw, event_id=event_id)

# Create the epochs object
tmin, tmax = -1, 7.0  # from +500ms to 4.5s after the event
epochs = mne.Epochs(
    raw, events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=(-1, 0), preload=True
)
print(epochs)

# %%
times = epochs.times
# Keep only C3 electrode to speed up computations
c3_pos = np.where(np.array(epochs.ch_names) == "C3")[0][0]
trials = epochs.get_data(copy=True)[:, c3_pos, :]
trials = trials[:, np.newaxis, :-1]
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

# %% [markdown]
# ### Defining the frequency range for the burst detection
# Variable *search_range* defines a space in frequency dimension in which bursts are going to be detected and corresponds to the indices in the *foi* variable. The range is 3 Hz wider than frequency range of interest (variable *beta_lims*) to avoid the edge effect, where bursts on the edge of the frequency of interest may be rejected or the TF might be incomplete. Only the bursts with peak frequency within *beta_lims* are going to be kept.
#
# see: https://github.com/danclab/burst_detection#burst-detection-algorithm---usage-notes
#
# The *search_range* selected aperiodic spectrum has to be reshaped to match the frequency dimension of the TF trials.

# %%
# Burst detection parameters
max_freq = 45
foi = np.linspace(1, max_freq, 90)
freqs = foi
upto_gamma_band = np.array([8, 40])
upto_gamma_range = np.where(
    np.logical_and(freqs >= upto_gamma_band[0], freqs <= upto_gamma_band[1])
)[0]

bm = TfBursts(
    fs,
    tmin=tmin,
    tmax=tmax,
    freqs=freqs,
    fr_band=upto_gamma_band,
    band_search_range=upto_gamma_range,
    band_limits=[8, 10, 35],
    remove_fooof=False,
)

# %%
tf_trials = bm._apply_tf(trials, order_max=40, c_1=4)

# %% [markdown]
# ### BURST EXTRACTION

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
