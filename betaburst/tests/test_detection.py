import numpy as np

from betaburst.detection.burst_analysis import TfBursts

from neurodsp.sim import sim_combined
from neurodsp.utils import set_random_seed, create_times

set_random_seed(42) # Reproducibility

# Simulation settings
fs = 250
n_seconds = 1*2*5
components = {'sim_synaptic_current' : {'n_neurons':1000, 'firing_rate':2,
                                        't_ker':1.0, 'tau_r':0.002, 'tau_d':0.02},
              'sim_bursty_oscillation' : {'freq' : 10, 'enter_burst' : .2, 'leave_burst' : .2}}
sig = sim_combined(n_seconds, fs, components)
times = create_times(n_seconds, fs)
epochs = sig.reshape((1, 2, 5*250)) # MNE format: (epoch, channel, trial)

# Burst detection parameters
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
assert tfs.shape == (1,2,84,1250), 'Test failed, TF decomposition format is not correct.'

bursts = bm.burst_extraction(epochs, band="beta")
assert len(bursts) == 2, 'Test failed, there should be one dictionnary per channel.'
assert bursts[0]['waveform_times'] == [0.5, 1.5], 'Test failed, no bursts detected at the proper time location (0.5s and 1.5s).'