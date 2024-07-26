"""Burst detection based on superlettime-frequency decomposition.

Removal of aperiodic activity based on the FOOOF algorithm,
and estimation of true beta and mu bands based on the
periodic activity.


Authors: Sotirios Papadopoulos <sotirios.papadopoulos@univ-lyon1.fr> 
        James Bonaiuto <james.bonaiuto@isc.cnrs.fr>
        Maciej Szul <maciej.szul@isc.cnrs.fr>

Packaging: Ludovic Darmet <ludovic.darmet@isc.cnrs.fr>
"""

import pickle
import numpy as np
from os.path import join, exists

from fooof import FOOOFGroup

from joblib import Parallel, delayed

from burst_detection import extract_bursts
from superlet_mne import superlets_mne_epochs
from plot_tf_activity import plot_sub_tfs


class TfBursts:
    """Burst detection based on superlets time-frequency analysis.

    Transform input data from time domain to time-frequency domain
    using the wavelet or superlet transform.

    Superlets algorithm is based on Moca et al., 2021.
    Implementation by Gregor Mönke: https://github.com/tensionhead.

    Burst detection per recording channel is based on the DANC Lab
    implementation: https://github.com/danclab/burst_detection.

    Parameters
    ----------
    exp_variables: dict
                   Experimental variables contained in the corresponding
                   'variables.json' file.
    freqs: 1D numpy array
           Frequency axis corresponding to the time-frequency anaysis.
    fr_band: two-element list or 1D array
             'Canonical' frequency band limits for adjusting the bands based
             on periodic peaks fitted while computing the FOOOF model.
    band_search_range: two-element numpy array or list
                       Indices of 'freqs' for fitting the FOOOF model.
    remove_fooof: bool, optional
                  Remove aperiodic FOOOF spectrum fit from time-frequency
                  matrices.
                  Defaults to "True".
    n_cycles: int or None, optional
              Number of cycles used for time-frequency decomposition using
              the wavelets algorithm. Only used when 'tf_method' is set to
              "wavelets".
              Defaults to "None".
    band_limits: list, optional
                 Frequency limits for splitting detected periodic peaks of
                 the FOOOF model in the custom mu and beta bands.
                 Defaults to [8,15,30] Hz.
    produce_plots: bool, optional
                   Plot the time-frequency maps, PSD and aperiodic fits before
                   burst extraction. Mainly useful for debugging purposes.
                   Defaults to "True".
    plot_format: str {"pdf", "png"}, optional
                 File format. Prefer "pdf" for editing with vector graphics
                 applications, or "png" for less space usage and better
                 integration with presentations. Ignored if 'produce_plots'
                 is set to "False".
                 Defaults to "pdf".

    Attributes
    ----------
    channels: list
              Names of channels to include in the analysis.
    tmin, tmax: float
                Start and end time of the epochs in seconds, relative to
                the time-locked event.
    sfreq: int
           Sampling frequency of the recordings in Hz.
    exp_time: 1D numpy array
              Experimental (cropped) time axis.
    exp_time_periods: 4-element list or 1D array
                      Beginning of baseline period, beginning of task period,
                      end of task period and end of rebound period (in seconds
                      relative to the time-locked event).
    savepath: str
              Parent directory that contains all results. Defaults to the
              path provided in the corresponding 'variables.json' file.

    References
    ----------
    Superlets:
    [1] Moca VV, Bârzan H, Nagy-Dăbâcan A, Mureșan RC. Time-frequency super-resolution with superlets.
    Nat Commun. 2021 Jan 12;12(1):337. doi: 10.1038/s41467-020-20539-9. PMID: 33436585; PMCID: PMC7803992.

    Thresholding power based on FOOOF aperiodic fits:
    [2] Brady B, Bardouille T. Periodic/Aperiodic parameterization of transient oscillations (PAPTO)-Implications
    for healthy ageing. Neuroimage. 2022 May 1;251:118974. doi: 10.1016/j.neuroimage.2022.118974. Epub 2022
    Feb 4. PMID: 35131434.
    """

    def __init__(
        self,
        exp_variables: dict,
        freqs: np.ndarray,
        fr_band: np.ndarray,
        band_search_range: list,
        remove_fooof=True,
        n_cycles=None,
        band_limits=[8, 15, 30],
        produce_plots=True,
        plot_format="pdf",
    ):
        self.exp_variables = exp_variables
        self.freqs = freqs
        self.fr_band = fr_band
        self.band_search_range = band_search_range
        self.remove_fooof = remove_fooof
        self.n_cycles = n_cycles
        self.band_limits = band_limits
        self.produce_plots = produce_plots
        self.plot_format = plot_format

        # Direct access to variables, and time axis creation.
        self.channels = exp_variables["channels"]
        self.savepath = exp_variables["dataset_path"]
        self.exp_time_periods = exp_variables["exp_time_periods"]

        self.tmin = exp_variables["tmin"]
        self.tmax = exp_variables["tmax"]
        self.sfreq = exp_variables["sfreq"]
        self.exp_time = np.linspace(
            self.tmin,
            self.tmax,
            int((np.abs(self.tmax - self.tmin)) * self.sfreq),
        )
        self.exp_time = np.around(self.exp_time, decimals=3)

    def _apply_tf(self, sub_dir: str, epochs: np.ndarray, save_results=True) -> np.ndarray:
        """Transform time-domain data to time-frequency domain.

        Save the results if run for the first time, else load them.

        Parameters
        ----------
        sub_dir: str
                 Subdirectory path for a subject.
        epochs: numpy array
                Array containing data in time domain.
        save_results: bool, optional
                      If "True" save the time-frequency matrices.
                      Recommened for speeding up the process.
                      Defaults to "True".

        Returns
        -------
        tfs: numpy array
             Array of time-frequency matrices for a set of channels and
             all trials of a single subject.
        """

        # Number of channels and number of matrices for data saving.
        n_channels = epochs.shape[1]
        total_chunks = n_channels // 10 + 1

        # Time-frequency decomposition.
        print(
            "Checking for existence of {}-based time-frequency matrices for subject {}...".format(
                self.subject
            )
        )
        # Check for existing files.
        filename = join(sub_dir, "tfs_{}.npy".format(self.tf_method))
        if exists(filename):
            print("Loading file {}...".format(filename))
            if n_channels <= 10:
                tfs = np.load(filename)
            else:
                tfs = np.zeros(
                    (epochs.shape[0], epochs.shape[1], len(self.freqs), epochs.shape[2])
                )
                for chunk in range(total_chunks):
                    if chunk == 0:
                        filename = join(sub_dir, "tfs_{}.npy".format(self.tf_method))
                        tfs[:, :10, :, :] = np.load(filename)
                    else:
                        filename = join(
                            sub_dir, "tfs_{}_({}).npy".format(self.tf_method, chunk)
                        )
                        tf = np.load(filename)
                        tfs[:, chunk * 10 : chunk * 10 + tf.shape[1], :, :] = tf

        # If not precomputed, apply corresponding transform.
        else:
            print("No such file found...")
            print(
                "Computing {}-based time-frequency matrices for subject {}...".format(
                    self.tf_method, self.subject
                )
            )

            tfs = superlets_mne_epochs(epochs, self.freqs, n_jobs=-1)

            # Store results.
            if save_results == True:
                print("Saving time-frequency matrices...")
                if n_channels <= 10:
                    np.save(join(sub_dir, "tfs_{}".format(self.tf_method)), tfs)
                else:
                    for chunk in range(total_chunks):
                        if chunk == 0:
                            np.save(
                                join(sub_dir, "tfs_{}".format(self.tf_method)),
                                tfs[:, :10, :],
                            )
                        else:
                            start = chunk * 10
                            end = start + 10
                            np.save(
                                join(
                                    sub_dir, "tfs_{}_({})".format(self.tf_method, chunk)
                                ),
                                tfs[:, start:end, :],
                            )

        return tfs

    def _custom_fr_range(self, ch_av_psd:np.ndarray, channel_ids="all"):
        """
        Identification of individualized frequency band ranges for burst detection, based
        on the peaks of a FOOOF model. This function aims to restrain (if possible)
        the frequency range of interest around frequency peaks in the power spectrum.

        When identifying two bands (roughly corresponding to mu and beta), the higher
        band's lower limit is bounded by the lower's band upper limit.

        Parameters
        ----------
        ch_av_psd: numpy array
                   Across-trials average PSD per channel.
        channel_ids: str or list, optional
                     Indices of channels to take into account while fitting the
                     aperiodic activity. If set to "all" all channels are used.
                     Defaults to "all".

        Returns
        -------
        mu_bands, beta_bands: numpy array
                              Minimum and maximum frequencies for burst extraction
                              per channel.
        mu_search_ranges, beta_search_ranges: numpy array
                                              Extended channel-specific frequency bands
                                              indices used during burst detection, with
                                              respect to 'self.freqs'.
        aperiodic_params: list
                          Aperiodic parameters of custom FOOOF fits per channel.
        """

        freq_step = self.freqs[1] - self.freqs[0] # Frequency resolution

        all_channels_fg = FOOOFGroup(
            peak_width_limits=[2.0, 12.0], peak_threshold=1.5, max_n_peaks=5
        )  # FOOOF Group model for all channels
        if channel_ids == "all":
            all_channels_fg.fit(
                self.freqs[self.band_search_range], ch_av_psd[:, self.band_search_range]
            )
        else:
            all_channels_fg.fit(
                self.freqs[self.band_search_range],
                ch_av_psd[channel_ids, :][:, self.band_search_range],
            )

        all_channels_gauss = all_channels_fg.get_params("gaussian_params")
        aperiodic_params = all_channels_fg.get_params("aperiodic_params")

        # Adjustment of frequency band limits depending on the fitted model
        # (iteratively for each channel)
        mu_bands = []
        mu_search_ranges = []
        beta_bands = []
        beta_search_ranges = []

        for ch_id in range(ch_av_psd.shape[0]):
            this_channel = np.where(
                (all_channels_gauss[:, -1] == ch_id)
                & (all_channels_gauss[:, 0] >= self.band_limits[0])
                & (all_channels_gauss[:, 0] <= self.band_limits[2])
            )[0] # Exclusion of peaks below or above 'self.band_limits'

            # If the model does not iclude any periodic activity peaks for
            # a channel, place empty variables and continue.
            if len(this_channel) == 0:
                mu_band = np.array([np.NAN, np.NAN])
                beta_band = np.array([np.NAN, np.NAN])
                mu_search_range = np.array([np.NAN])
                beta_search_range = np.array([np.NAN])

                mu_bands.append(mu_band)
                beta_bands.append(mu_band)
                mu_search_ranges.append(mu_search_range)
                beta_search_ranges.append(beta_search_range)

                continue
            else:
                channel_band_peaks = all_channels_gauss[this_channel, 0]
                channel_gauss = all_channels_gauss[this_channel, :][:, [0, 2]]

            # If many peaks have been detected, keep any peak in the
            # 'canonical' frequency range
            if len(channel_band_peaks) > 1:
                band_peaks_ids = np.where(
                    (channel_band_peaks >= self.fr_band[0])
                    & (channel_band_peaks <= self.fr_band[1])
                )[0]
                channel_band_peaks = channel_band_peaks[band_peaks_ids]
                channel_gauss = channel_gauss[band_peaks_ids]

            
            channel_bandwidths = []
            for gauss in channel_gauss: # Fit a gaussian to each peak, and compute the full width half maximum
                fwhm = 2 * np.sqrt(2 * np.log(2)) * gauss[1]
                channel_bandwidths.append(np.around(fwhm * freq_step))

            
            low = np.where(channel_band_peaks <= self.band_limits[1])[0] # Split criterion
            high = np.where(channel_band_peaks > self.band_limits[1])[0]

            # Dual band scenario: split into mu and beta.
            if low.size > 0 and high.size > 0:
                # If many peaks have been detected, expand the frequency range
                # from the lowest to highest; else symmetric around single peak
                mu_band = [
                    np.floor(channel_band_peaks[low[0]] - channel_bandwidths[low[0]]),
                    np.ceil(channel_band_peaks[low[-1]] + channel_bandwidths[low[-1]]),
                ]
                beta_band = [
                    np.floor(channel_band_peaks[high[0]] - channel_bandwidths[high[0]]),
                    np.ceil(
                        channel_band_peaks[high[-1]] + channel_bandwidths[high[-1]]
                    ),
                ]

                
                if mu_band[0] < self.band_limits[0] - 2: # Limit the bands
                    mu_band[0] = self.band_limits[0] - 2

                if mu_band[1] > self.band_limits[1]:
                    mu_band[1] = self.band_limits[1]

                if beta_band[0] <= mu_band[1]:
                    beta_band[0] = mu_band[1] + freq_step

                mu_search_range = np.where(
                    (self.freqs >= mu_band[0] - 3) & (self.freqs <= mu_band[1] + 3)
                )[0]
                beta_search_range = np.where(
                    (self.freqs >= beta_band[0] - 3) & (self.freqs <= beta_band[1] + 3)
                )[0]

                mu_band = np.hstack(mu_band)
                beta_band = np.hstack(beta_band)

                mu_bands.append(mu_band)
                beta_bands.append(beta_band)
                mu_search_ranges.append(mu_search_range)
                beta_search_ranges.append(beta_search_range)

            
            elif low.size > 0 and high.size == 0: # Single band scenario
                # If many peaks have been detected, expand the frequency range
                # from the lowest to highest; else symmetric around single peak
                mu_band = [
                    np.floor(channel_band_peaks[0] - channel_bandwidths[0]),
                    np.ceil(channel_band_peaks[-1] + channel_bandwidths[-1]),
                ]

                
                if mu_band[0] < self.band_limits[0] - 2: # Limit the band
                    mu_band[0] = self.band_limits[0] - 2

                if mu_band[1] > self.band_limits[1]:
                    mu_band[1] = self.band_limits[1]

                mu_band = np.hstack(mu_band)

                
                mu_search_range = np.where( 
                    (self.freqs >= mu_band[0] - 3) & (self.freqs <= mu_band[1] + 3)
                )[0] # Use the custom frequency bands instead of the 'canonical'

                mu_bands.append(mu_band)
                mu_search_ranges.append(mu_search_range)

                beta_bands.append(np.array([np.NAN, np.NAN]))
                beta_search_ranges.append(np.array([np.NAN]))

            elif low.size == 0 and high.size > 0:
                # If many peaks have been detected, expand the frequency range
                # from the lowest to highest; else symmetric around single peak.
                beta_band = [
                    np.floor(channel_band_peaks[0] - channel_bandwidths[0]),
                    np.ceil(channel_band_peaks[-1] + channel_bandwidths[-1]),
                ]

                
                if beta_band[0] < self.band_limits[1] - 2: # Limit the band
                    beta_band[0] = self.band_limits[1] - 2

                beta_band = np.hstack(beta_band)

                
                beta_search_range = np.where(
                    (self.freqs >= beta_band[0] - 3) & (self.freqs <= beta_band[1] + 3)
                )[0] # Use the custom frequency bands instead of the 'canonical'

                beta_bands.append(beta_band)
                beta_search_ranges.append(beta_search_range)

                mu_bands.append(np.array([np.NAN, np.NAN]))
                mu_search_ranges.append(np.array([np.NAN]))

        mu_bands = np.array(mu_bands)
        beta_bands = np.array(beta_bands)

        return (
            mu_bands,
            beta_bands,
            mu_search_ranges,
            beta_search_ranges,
            aperiodic_params,
        )

    def burst_extraction(self, subject: int, epochs, labels: np.ndarray, band="beta") -> None:
        """ Time-frequency analysis with optional plotting and burst extraction
        per subject.

        Following data loading, time-frequency decomposition is performed either using the
        wavelets or the superlets algorithm. Then, 1) the PSD for each trial and channel is
        estimated as the time-averaged activity and 2) an aperiodic fit based on the FOOOF
        algorithm is computed from the trial-averaged PSD of each channel.

        After creating the FOOOF model, the fitted periodic component (model peaks) is used
        to guide a "smart" identification of frequency band(s) that correspond its FWHM. If
        more than one peaks are detected then the frequency range is estimated as the range
        spanning the lowest peak minus its FWHM to the highest peak plus its FWHM.

        Intermediate results corresponding to the number of experimental trials, labels,
        meta-info, time-frequency matrices, FOOOF model parameters, individualized bands are
        saved in the corresponding directory. Detected bursts are also saved in the same
        directory.

        Parameters
        ----------
        subject: int
                 Subject whose data are analysed (from a list of subjects).
        epochs: MNE epochs object or Numpy array
                The recordings corresponding to the subject and classes we are interested in.
        labels: numpy array
                Array of strings containing the labels for each trial in 'epochs'.
        band: str {"mu", "beta"}, optional
              Select band for burst detection.
              Defaults to "beta".

        Attributes
        ----------
        subject: int
                 Subject whose data are analysed (from a list of subjects).
        """

        # Related to the time axis.
        try:
            baseline_begin = int(np.where(self.exp_time == self.exp_time_periods[0])[0])
        except:
            baseline_begin = np.where(self.exp_time >= self.exp_time_periods[0])[0][0]
        try:
            task_begin = int(np.where(self.exp_time == self.exp_time_periods[1])[0])
        except:
            task_begin = np.where(self.exp_time >= self.exp_time_periods[1])[0][0]
        try:
            rebound_end = int(np.where(self.exp_time == self.exp_time_periods[3])[0])
        except:
            rebound_end = np.where(self.exp_time <= self.exp_time_periods[3])[0][-1]

        base_time_lims = [baseline_begin, task_begin]
        erds_time_lims = [baseline_begin, rebound_end]

        erds_time = self.exp_time[erds_time_lims[0] : erds_time_lims[1] + 1]

        # Directory for saving intermediate data.
        self.subject = subject
        sub_dir = join(self.savepath, "sub_{}/".format(self.subject))

        # ----- #
        # 1. Time-frequency decomposition and PSD computation.
        tfs = self._apply_tf(sub_dir, epochs)

        # Removal of edge effects introduced by the time-frequency transform.
        print("Trimming time-frequency matrices in order to remove edge-effects....")
        tfs = tfs[:, :, :, erds_time_lims[0] : erds_time_lims[1] + 1]

        # ----- #
        av_psds = np.mean(
            tfs[:, :, :, base_time_lims[0] : base_time_lims[1]], axis=(0, 3)
        )

        # 2. Creation of FOOOF model for removing aperiodic activity and
        # adjustment of mu and beta bands range per channel.

        if self.remove_fooof == True:
            try:
                mu_bands = np.load(
                    join(sub_dir, "mu_bands_{}.npy".format(self.tf_method))
                )
                beta_bands = np.load(
                    join(sub_dir, "beta_bands_{}.npy".format(self.tf_method))
                )

                with open(
                    join(sub_dir, "mu_search_ranges_{}.pkl".format(self.tf_method)),
                    "rb",
                ) as pickle_file:
                    mu_search_ranges = pickle.load(pickle_file)
                with open(
                    join(sub_dir, "beta_search_ranges_{}.pkl".format(self.tf_method)),
                    "rb",
                ) as pickle_file:
                    beta_search_ranges = pickle.load(pickle_file)

                with open(
                    join(sub_dir, "mu_fooof_thresholds_{}.pkl".format(self.tf_method)),
                    "rb",
                ) as pickle_file:
                    mu_thresholds = pickle.load(pickle_file)
                with open(
                    join(
                        sub_dir, "beta_fooof_thresholds_{}.pkl".format(self.tf_method)
                    ),
                    "rb",
                ) as pickle_file:
                    beta_thresholds = pickle.load(pickle_file)

                aperiodic_params = np.load(
                    join(sub_dir, "aperiodic_params_{}.npy".format(self.tf_method))
                )

                print(
                    "Loading custom, subject- and channel-specific adjusted frequency bands and FOOOF thresholds..."
                )

            except:
                print(
                    "Computing custom, subject- and channel-specific adjusted frequency bands and FOOOF thresholds..."
                )

                # Adjust mu and beta band limits depending on the fitted model.
                (
                    mu_bands,
                    beta_bands,
                    mu_search_ranges,
                    beta_search_ranges,
                    aperiodic_params,
                ) = self._custom_fr_range(av_psds)

                # Baseline noise (in linear space)
                mu_thresholds = []
                beta_thresholds = []
                for ch_id, (mu_search_range, beta_search_range) in enumerate(
                    zip(mu_search_ranges, beta_search_ranges)
                ):
                    if mu_search_range.size == 1:
                        mu_threshold = []  # Empty list if no mu band is detected
                    else:
                        mu_threshold = np.power(
                            10, aperiodic_params[ch_id, 0].reshape(-1, 1)
                        ) / np.power(
                            self.freqs[mu_search_range],
                            aperiodic_params[ch_id, 1].reshape(-1, 1),
                        )
                    mu_thresholds.append(mu_threshold)

                    if beta_search_range.size == 1:
                        beta_threshold = [] # Empty list if no beta band is detected
                    else:
                        beta_threshold = np.power(
                            10, aperiodic_params[ch_id, 0].reshape(-1, 1)
                        ) / np.power(
                            self.freqs[beta_search_range],
                            aperiodic_params[ch_id, 1].reshape(-1, 1),
                        )
                    beta_thresholds.append(beta_threshold)

                # Model parameters saving.
                print(
                    "Saving FOOOF model aperiodic parameters and the custom frequency bands for future reference..."
                )
                np.save(
                    join(sub_dir, "aperiodic_params_{}".format(self.tf_method)),
                    aperiodic_params,
                )
                np.save(join(sub_dir, "mu_bands_{}".format(self.tf_method)), mu_bands)
                np.save(
                    join(sub_dir, "beta_bands_{}".format(self.tf_method)), beta_bands
                )

                with open(
                    join(sub_dir, "mu_search_ranges_{}.pkl".format(self.tf_method)),
                    "wb",
                ) as pickle_file:
                    pickle.dump(mu_search_ranges, pickle_file)
                with open(
                    join(sub_dir, "beta_search_ranges_{}.pkl".format(self.tf_method)),
                    "wb",
                ) as pickle_file:
                    pickle.dump(beta_search_ranges, pickle_file)
                with open(
                    join(sub_dir, "mu_fooof_thresholds_{}.pkl".format(self.tf_method)),
                    "wb",
                ) as pickle_file:
                    pickle.dump(mu_thresholds, pickle_file)
                with open(
                    join(
                        sub_dir, "beta_fooof_thresholds_{}.pkl".format(self.tf_method)
                    ),
                    "wb",
                ) as pickle_file:
                    pickle.dump(beta_thresholds, pickle_file)

            print(
                "The custom mu bands span a range of frequencies from {} to {} Hz.".format(
                    np.nanmin(mu_bands), np.nanmax(mu_bands)
                )
            )
            print(
                "The custom beta bands span a range of frequencies from {} to {} Hz.".format(
                    np.nanmin(beta_bands), np.nanmax(beta_bands)
                )
            )

        del av_psds

        # ----- #
        # 3. Selection of proper variables to use, based on the 'band' parameter.
        if self.remove_fooof:
            if band == "mu":
                band_search_ranges = mu_search_ranges
                canon_band = [self.band_limits[0], self.band_limits[1]]
                bd_bands = mu_bands
                thresholds = mu_thresholds
                w_size = 0.6
            elif band == "beta":
                band_search_ranges = beta_search_ranges
                canon_band = [self.band_limits[1], self.band_limits[2]]
                bd_bands = beta_bands
                thresholds = beta_thresholds
                w_size = 0.26

            msg = "with aperiodic activity subtraction"
            fooof_save_str = ""

        else:
            if band == "mu":
                canon_band = [self.band_limits[0], self.band_limits[1]]
                w_size = 0.6
            elif band == "beta":
                canon_band = [self.band_limits[1], self.band_limits[2]]
                w_size = 0.26

            msg = "without aperiodic activity subtraction"
            fooof_save_str = "_nfs"

        canon_band_range = np.where(
            (self.freqs >= canon_band[0] - 3) & (self.freqs <= canon_band[1] + 3)
        )[0]

        # ----- #
        # 4. Optional plot of time-frequency decomposition and fits for
        # channels C3 and C4.
        if self.produce_plots == True and\
            self.remove_fooof == True:
            c3 = int(np.where((np.array(self.channels) == "C3"))[0])
            c4 = int(np.where((np.array(self.channels) == "C4"))[0])

            if len(thresholds[c3]) > 0 and len(thresholds[c4]) > 0:
                psds = np.mean(
                    tfs[:, :, :, base_time_lims[0] : base_time_lims[1]], axis=3
                )
                c3c4 = [c3, c4]
                plot_sub_tfs(
                    self.subject,
                    labels,
                    tfs,
                    psds,
                    thresholds,
                    c3c4,
                    erds_time,
                    self.freqs,
                    band_search_ranges,
                    band,
                    self.tf_method,
                    self.savepath,
                    plot_format=self.plot_format,
                )
                del psds

            else:
                print(
                    "One or both of channels C3 and C4 have no periodic activity in the selected band. Plotting aborted..."
                )

        # ----- #
        # 5. Burst detection.
        if self.remove_fooof:
            msg = "with aperiodic activity subtraction"
        else:
            msg = "without aperiodic activity subtraction"

        print("Initiating {} band burst extraction {}...".format(band, msg))

        # Option to remove aperiodic activity or proceed with lagged coherence
        # instead of power-based TF representations.
        if self.remove_fooof:

            # Use canocical beta band for channels without periodic activity.
            for ch_id in range(epochs.shape[1]):
                if band_search_ranges[ch_id].size == 1:

                    warn = (
                        "\tThis channel has no periodic acivity in the {} band. ".format(
                            band
                        )
                        + "Proceeding with 'canonical' {} band{} without aperiodic activity subtraction."
                    )
                    print(warn)
                
                else:
                    print(
                        "\tBurst extraction in custom {} band from {} to {} Hz.".format(
                            band, bd_bands[ch_id, 0], bd_bands[ch_id, 1]
                        )
                    )

                canon_threshold = np.power(
                    10, aperiodic_params[ch_id, 0].reshape(-1, 1)
                ) / np.power(
                    self.freqs[canon_band_range],
                    aperiodic_params[ch_id, 1].reshape(-1, 1),
                )

                band_search_ranges[ch_id] = canon_band_range
                thresholds[ch_id] = canon_threshold
                bd_bands[ch_id] = canon_band
        
            sub_bursts = Parallel(n_jobs=epochs.shape[1], require="sharedmem")(
                delayed(extract_bursts)(
                    epochs[:, ch_id, erds_time_lims[0] : erds_time_lims[1] + 1],
                    tfs[:, ch_id, band_search_ranges[ch_id]],
                    erds_time,
                    self.freqs[band_search_ranges[ch_id]],
                    bd_bands[ch_id],
                    thresholds[ch_id].reshape(-1,1),
                    self.sfreq,
                    self.subject,
                    ch_id,
                    labels,
                    w_size=w_size,
                    remove_fooof=self.remove_fooof
                )
                for ch_id in range(epochs.shape[1])
            )  # Extraction is performed on parallel along channels
        

        else:

            print(
                "\tBurst extraction for all channels: from {} to {} Hz.".format(
                    canon_band[0], canon_band[1]
                )
            )

            null_threshold = np.zeros((self.freqs[canon_band_range].shape[0], 1))
            sub_bursts = Parallel(n_jobs=epochs.shape[1], require="sharedmem")(
                delayed(extract_bursts)(
                    epochs[:, ch_id, erds_time_lims[0] : erds_time_lims[1] + 1],
                    tfs[:, ch_id, canon_band_range],
                    erds_time,
                    self.freqs[canon_band_range],
                    canon_band,
                    null_threshold,
                    self.sfreq,
                    self.subject,
                    ch_id,
                    labels,
                    w_size=w_size,
                    remove_fooof=False
                )
                for ch_id in range(epochs.shape[1])
            ) # Extraction is performed on parallel along channels

        # Burst results saving.
        print(
            "Saving {} band burst dictionary for subject {}...".format(
                band, self.subject
            )
        )
        np.save(
            join(
                sub_dir, "{}_bursts_{}{}".format(band, self.tf_method, fooof_save_str)
            ),
            sub_bursts,
        )
        print("Results saved.")

        del epochs
        del tfs
        if self.remove_fooof == True and self.tf_method != "lagged_coherence":
            del aperiodic_params
            del thresholds
        del sub_bursts
