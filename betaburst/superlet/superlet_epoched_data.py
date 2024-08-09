"""Parallelized superlets time-frequency transform of multiple trials.

Author: James Bonaiuto <james.bonaiuto@isc.cnrs.fr>

Adaptation: Sotirios Papadopoulos <sotirios.papadopoulos@univ-lyon1.fr>
"""

import numpy as np

from joblib import Parallel, delayed

from betaburst.superlet.superlet import scale_from_period, superlet


def superlets_mne_epochs(epochs, freqs, sfreq=250, order_max=50, order_min=4, c_1=3, n_jobs=1):
    """
    Returns a list of transformed superlets.
    """

    def do_superlet(epoch, ix, scales, sfreq):
        signal = epoch[ix, :]
        spec = superlet(
            signal,
            samplerate=sfreq,
            scales=scales,
            order_max=order_max,
            order_min=order_min,
            c_1=c_1,
            adaptive=True,
        )
        chan_list[ix, :] = np.single(np.abs(spec))
        del spec

    foi = freqs
    scales = scale_from_period(1 / foi)
    epochs_list = []

    for ep_idx in range(epochs.shape[0]):
        epoch = epochs[ep_idx, :]
        chan_list = np.zeros((epoch.shape[0], len(foi), epoch.shape[1]))
        Parallel(n_jobs=n_jobs, require="sharedmem")(
            delayed(do_superlet)(epoch, ix, scales, sfreq)
            for ix in range(epoch.shape[0])
        )
        epochs_list.append(chan_list)

    return np.array(epochs_list)