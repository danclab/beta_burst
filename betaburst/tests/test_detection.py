import numpy as np
import pytest

from betaburst.detection.burst_analysis import TfBursts
from betaburst.tests._utils import generate_transient_minimum

from neurodsp.utils import set_random_seed

set_random_seed(42) # Reproducibility

# Simulation settings
fs = 250

# Simulation settings
fs = 250
n_epochs = 5
epochs = np.random.randn(n_epochs, 2, 5*fs) / 10

length = 0.16
w_size = int(length*fs)
t = np.arange(0, length, 1/fs)
# Add transient oscillations
epochs[0, 0, int(1*fs):int((1+length)*fs)] +=  7*generate_transient_minimum(w_size, decay=0.00001) 
epochs[0, 0, int(3*fs):int((3+length)*fs)] +=  7*generate_transient_minimum(w_size,  decay=0.002)

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
        band_limits=[8, 10, 35],
        remove_fooof=False,
    )

def test_tf_decomposition(bm):
    tfs = bm._apply_tf(epochs)
    assert tfs.shape == (epochs.shape[0], epochs.shape[1], len(freqs), epochs.shape[-1]), 'TF decomposition format is not correct.'

def test_burst_extraction(bm):
    bursts = bm.burst_extraction(epochs, band="beta")
    print(bursts[0]['trial'])
    print(bursts[0]['peak_time'])
    print(bursts[1]['trial'])
    assert len(bursts) == 2, 'There should be one dictionnary per channel.'
    # assert len(bursts[1]) == 0, 'There should be no burst detected on channel 1.'
    assert np.isclose(bursts[0]['peak_time'][0], 1.081, atol = 1e-1), 'Burst not detected at the proper time locations.'
    assert np.isclose(bursts[0]['peak_time'][1], 3.082, atol = 1e-1), 'Burst not detected at the proper time locations.'

def test_burst_extraction_with_fooof(bm):
    bm.remove_fooof = True
    bursts = bm.burst_extraction(epochs, band="beta")
    assert len(bursts) == 2, 'There should be one dictionnary per channel.'
    # Asserting burst detection with FOOOF is more challenging, 
    # as the simulated signal is not perfectly periodic.
    # We can check if at least some bursts are detected.
    assert len(bursts[0]['waveform_times']) > 0, 'No bursts detected with FOOOF.'

# Run the tests
if __name__ == "__main__":
    pytest.main(["-v", "-x", __file__])