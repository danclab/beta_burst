p  # %% [markdown]
# ## Example of Principal Component Analysis with the waveforms
#
# This example builds on top of the previous tutorial.

# %%
import requests
import zipfile
import os
from matplotlib import pyplot as plt
import numpy as np

import mne
from mne.channels import make_standard_montage
from mne.io import read_raw_cnt

import pandas as pd
from matplotlib import ticker
import matplotlib.pylab as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from betaburst.detection.burst_detection import TfBursts

# %% [markdown]
# #### This step is similar to what we have performed previously but now on all the channels. We won't download again the data

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
tmin, tmax = 0.5, 4.5  # from +500ms to 4.5s after the event
epochs = mne.Epochs(
    raw,
    events,
    event_id=event_id,
    tmin=tmin,
    tmax=tmax,
    baseline=(None, None),
    preload=True,
)
epochs.pick_channels(["C3", "C4"])
print(epochs)

# %%
times = epochs.times
trials = epochs.get_data(copy=True)[:, :, :-1]
fs = epochs.info["sfreq"]

# %%
max_freq = 45
freqs = np.linspace(1, max_freq, 50)
search_range = np.where((freqs >= 10) & (freqs <= 33))[0]
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
    band_limits=[8, 13, 30],
    remove_fooof=False,
)

bursts_all_results = bm.burst_extraction(
    epochs=trials, band="beta", std_noise=4, regress_ERF=False
)

# %% [markdown]
# Here we will combine burst features into a single data frame, and waveforms into a separate array. The first dimensions of the data frame and array are same, so boolean array based on e.g. "sensor" column can be used on the waveform array.

# %% [markdown]
# #### The PC scores can be analysed in many ways. Simple example here illustrates the principles of selecting the waveforms based on the dataframe. See the paper (https://doi.org/10.1016/j.pneurobio.2023.102490) for different ways the PC score was employed.

# %% [markdown]
# Here each burst PC score will be divided into quartiles and based on this selection, the average waveform from each quintile is going to be plotted.

# %%
from betaburst.analysis.burst_analysis import BurstSpace

bs = BurstSpace(perc=0.5, nb_quartiles=10, tmin=tmin, tmax=tmax, time_step=0.2)
scores_dists = bs.fit_transform(bursts_all_results)
bs.plot_waveforms()

# %% [markdown]
# We can also plot heatmaps corresponding to the burst rates of each quartile and PC during time

# %%
modulation_index, comp_waveforms = bs.waveforms_rate()
bs.plot_burst_rates()
