# Authors: Ludovic Darmet <ludovic.darmet@isc.cnrs.fr>

import pytest
import numpy as np
from betaburst.analysis.burst_analysis import BurstSpace


@pytest.fixture
def burst_data():
    """Fixture to create sample burst data."""
    nb_bursts_ch1 = 100
    burst_data_ch1 = {
        "waveform": np.random.rand(nb_bursts_ch1, 64),
        "peak_time": np.random.uniform(0, 10, nb_bursts_ch1),
        "waveform_times": np.arange(nb_bursts_ch1),
    }

    nb_bursts_ch2 = 250
    burst_data_ch2 = {
        "waveform": np.random.rand(nb_bursts_ch2, 64),
        "peak_time": np.random.uniform(0, 10, nb_bursts_ch2),
        "waveform_times": np.arange(nb_bursts_ch2),
    }

    nb_bursts_ch3 = 95
    burst_data_ch3 = {
        "waveform": np.random.rand(nb_bursts_ch3, 64),
        "peak_time": np.random.uniform(0, 10, nb_bursts_ch3),
        "waveform_times": np.arange(nb_bursts_ch3),
    }

    return [burst_data_ch1, burst_data_ch2, burst_data_ch3]


def test_burst_space_init():
    """Test initialization of BurstSpace class."""
    bs = BurstSpace(tmin=0, tmax=10, time_step=0.1, perc=0.5, nb_quartiles=5)
    assert bs.perc == 0.5
    assert bs.nb_quartiles == 5
    assert bs.tmin == 0
    assert bs.tmax == 10
    assert bs.time_step == 0.1


def test_concatenate_bursts(burst_data):
    """Test _concatenate_bursts method."""
    bs = BurstSpace(tmin=0, tmax=10, time_step=0.1, perc=0.5, nb_quartiles=5)
    print(burst_data[0].keys())
    burst_dict = bs._concatenate_bursts(burst_data)
    assert len(burst_dict["waveform"]) == 445
    assert len(burst_dict["peak_time"]) == 445


def test_fit_transform(burst_data):
    """Test fit_transform method."""
    bs = BurstSpace(tmin=0, tmax=10, time_step=0.1, perc=0.5, nb_quartiles=5)
    scores_dists = bs.fit_transform(burst_data, n_components=10)
    assert scores_dists.shape == (10, 5, 100)


def test_waveforms_rate(burst_data):
    """Test waveforms_rate method."""
    bs = BurstSpace(tmin=0, tmax=10, time_step=0.1, perc=0.5, nb_quartiles=5)
    scores_dists = bs.fit_transform(burst_data)
    modulation_index, comp_waveforms = bs.dist_scores()
    assert modulation_index.shape == (20, 5, 100)
    assert comp_waveforms.shape == (20, 5, 64)


def test_plot_burst_rates(burst_data):
    """Test plot_burst_rates method."""
    bs = BurstSpace(tmin=0, tmax=10, time_step=0.1, perc=0.5, nb_quartiles=5)
    scores_dists = bs.fit_transform(burst_data)
    modulation_index, comp_waveforms = bs.dist_scores()
    bs.plot_burst_rates()
    # Only test if it runs without error


# Run the tests
if __name__ == "__main__":
    pytest.main(["-v", "-x", __file__])
