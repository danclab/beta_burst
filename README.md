---

# Beta Bursts

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

**BetaBursts** is a Python package designed to detect beta bursts in brain signals, specifically EEG and MEG data. Beta bursts are rapid oscillations in the beta frequency band (13-30 Hz) and are crucial for movement-related cortical dynamics. This package provides a reliable method for detecting these bursts using advanced signal processing techniques.

The method is described in the paper by [Maciek Szul](http://www.isc.cnrs.fr/index.rvt?member=maciek%5F%5Fszul) et al. (2023) "Diverse beta burst waveform motifs characterize movement-related cortical dynamics" Progress in Neurobiology. https://doi.org/10.1016/j.pneurobio.2023.102490

The code have been developed by Maciek Szul, [Sotirios Papadopoulos](http://www.isc.cnrs.fr/index.rvt?member=sotiris%5Fpapadopoulos), [Ludovic DARMET](http://www.isc.cnrs.fr/index.rvt?language=en&member=ludovic%5Fdarmet) and [Jimmy Bonaiuto](http://www.isc.cnrs.fr/index.rvt?member=james%5Fbonaiuto), head of the [DANC lab](https://www.danclab.com/).

![Visual Description of the algorithm](./img/algo_description.png)
## Features

- Detect beta bursts in EEG and MEG signals.
- Analysis of the extracted bursts using PCA, to find specific waveforms modulated by the task.
- Includes a suite of tests to ensure the accuracy and reliability of the detection algorithms.

## Installation

To install the BetaBurst package, clone the repository and run:

```bash
git clone https://github.com/danclab/beta_burst
cd beta_burst
pip install -e .
```

### Requirements

- Python 3.x
- numpy
- scipy
- sklearn
- matplotlib
- fooof

These dependencies will be installed automatically when you run the setup script.

## Usage

Here is a basic example of how to extract burst from M/EEG data:

```python
from betaburst.detection.burst_detection import TfBursts

# Example usage with M/EEG data
meeg_data = ...  # Your M/EEG data here
# Burst detection parameters
freq_step = 0.5
freqs = np.arange(5.0, 47.0, freq_step)
upto_gamma_band = np.array([5, 40])
upto_gamma_range = np.where(
    np.logical_and(freqs >= upto_gamma_band[0], freqs <= upto_gamma_band[1])
)[0]

tfbursts =TfBursts(
        fs,
        freqs = freqs,
        fr_band = upto_gamma_band,
        band_search_range = upto_gamma_range,
        band_limits=[8, 10, 35],
        remove_fooof=False,
    )

bursts = bm.burst_extraction(epochs, band="beta")
print("Detected bursts:", bursts)

# You can access time-frequency decompositions (using superlets)
tfs = bm.tfs
# Or without running burts extraction
tfs = bm._apply_tf(meeg_data)
```

It is then possible to study the bursts waveforms using PCA.

```python
from betaburst.analysis.burst_analysis import BurstSpace

bs = BurstSpace(perc=0.5, nb_quartiles=10, tmin=0, tmax=5, time_step=0.2)
# Feat a PCA model (along the time axis of bursts) and compute a score
# for each burst. The score is the distance to the PCA axis.
scores_dists = bs.fit_transform(bursts)
# Distribution of waveforms along each PC axis
bs.plot_waveforms()

modulation_index, comp_waveforms = bs.waveforms_rate()
# Heatmap of modulation index during time (bursts rates of each quartile)
bs.plot_burst_rates()
```

## Testing

To run the tests with `pytest`, use the following command:

```bash
pytest betaburst/tests
```

This will execute all tests located in the `betaburst/tests` directory, ensuring the correctness of the implementation.

## Contributing

Contributions are welcome! If you find any bugs or have ideas for improvements, feel free to open an [issue](https://github.com/danc_lab/beta_burst/issues) or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
