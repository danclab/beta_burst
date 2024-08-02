import numpy as np

from betaburst.detection.burst_analysis import TfBursts
from betaburst._utils.help_func import load_exp_variables

from neurodsp.sim import sim_combined
from neurodsp.utils import set_random_seed, create_times

set_random_seed(42) # Reproducibility

# Simulation settings
fs = 250
n_seconds = 1*2*5
components = {'sim_synaptic_current' : {'n_neurons':1000, 'firing_rate':2,
                                        't_ker':1.0, 'tau_r':0.002, 'tau_d':0.02},
              'sim_bursty_oscillation' : {'freq' : 10, 'enter_burst' : .2, 'leave_burst' : .2}}

# Simulate a signal with a bursty oscillation with an aperiodic component & a time vector
sig = sim_combined(n_seconds, fs, components)
times = create_times(n_seconds, fs)
epochs = sig.reshape((1, 2, 5*250)) # MNE format: (epoch, channel, trial)

variables_path = "{}variables.json".format("betaburst/")
experimental_vars = load_exp_variables(json_filename=variables_path)

freq_step = 0.5
freqs = np.arange(5.0, 47.0, freq_step)
upto_gamma_band = np.array([5, 40])
upto_gamma_range = np.where(
    np.logical_and(freqs >= upto_gamma_band[0], freqs <= upto_gamma_band[1])
)[0]

bm = TfBursts(
    fs,
    freqs = freqs,
    fr_band = upto_gamma_band,
    band_search_range = upto_gamma_range,
    band_limits=[8, 12, 30],
    remove_fooof=False,
)

tfs = bm._apply_tf(epochs)
print(tfs.shape)

bursts = bm.burst_extraction(epochs, band="beta")
print(bursts)