import numpy as np
import pytest

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

@pytest.fixture
def bm():
    return TfBursts(
        fs,
        freqs = freqs,
        fr_band = upto_gamma_band,
        band_search_range = upto_gamma_range,
        band_limits=[8, 12, 30],
        remove_fooof=False,
    )

def test_tf_decomposition(bm):
    tfs = bm._apply_tf(epochs)
    assert tfs.shape == (1,2,84,1250), 'TF decomposition format is not correct.'

def test_burst_extraction(bm):
    bursts = bm.burst_extraction(epochs, band="beta")
    assert len(bursts) == 2, 'There should be one dictionnary per channel.'
    assert bursts[0]['waveform_times'] == [0.5, 1.5], 'Bursts not detected at the proper time locations (0.5s and 1.5s).'

def test_burst_extraction_with_fooof(bm):
    bm.remove_fooof = True
    bursts = bm.burst_extraction(epochs, band="beta")
    assert len(bursts) == 2, 'There should be one dictionnary per channel.'
    # Asserting burst detection with FOOOF is more challenging, 
    # as the simulated signal is not perfectly periodic.
    # We can check if at least some bursts are detected.
    assert len(bursts[0]['waveform_times']) > 0, 'No bursts detected with FOOOF.'

def test_custom_fr_range(bm):
    # Simulate a PSD with a clear peak in the beta band
    ch_av_psd = np.zeros((2, len(freqs)))
    peak_freq = 20
    peak_idx = np.where(freqs == peak_freq)[0][0]
    ch_av_psd[:, peak_idx] = 10
    
    (
        mu_bands,
        beta_bands,
        mu_search_ranges,
        beta_search_ranges,
        aperiodic_params,
    ) = bm._custom_fr_range(ch_av_psd)

    # Assert that the beta band is correctly identified
    assert beta_bands[0, 0] < peak_freq < beta_bands[0, 1]
    assert beta_search_ranges[0].size > 1

    # Assert that the mu band is not identified (as there is no peak in the mu band)
    assert np.isnan(mu_bands[0, 0])
    assert mu_search_ranges[0].size == 1

# Run the tests
if __name__ == "__main__":
    pytest.main(["-v", "-x", __file__])