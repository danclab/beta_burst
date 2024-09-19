"""Burst dictionary creation and dimensionality reduction of
burst waveforms with PCA.

Authors: Sotirios Papadopoulos <sotirios.papadopoulos@univ-lyon1.fr> 
        James Bonaiuto <james.bonaiuto@isc.cnrs.fr>
        Maciej Szul <maciej.szul@isc.cnrs.fr>

Packaging: Ludovic Darmet <ludovic.darmet@isc.cnrs.fr>
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA


class BurstSpace:
    """Analysis of the bursts waveforms using PCA.

    Search for waveforms with burst rate modulated in correlation with behavior.

    Parameters
    -----------
    tmin: float.
        Left time limit to compute burst rate.
    tmax: float.
        Right time limit to compute burst rate.
    time_step: float.
        Time step to compute burst rate.
    perc: float. Default to 0.5.
        Subset of the total amount of waveforms, in percentage, to be used in the PCA anaysis.
    nb_quartiles: int. Default to 5.
        Number of quartile to divide each PCA axis.
    """

    def __init__(
        self, tmin: float, tmax: float, time_step: float, perc=0.5, nb_quartiles=5
    ):
        assert 0 < perc <= 1, "`Perc` should be between 0 and 1."
        assert nb_quartiles > 1, "`nb_quartiles` should be greater than 1."
        assert tmin < tmax, "`tmin` should be less than `tmax`.`"
        assert time_step > 0, "`time_step` should be greater than 0."
        assert (
            tmax - tmin
        ) / 2 > time_step, "`time_step` should divide total time at least in 2."
        self.perc = perc
        self.nb_quartiles = nb_quartiles
        self.tmin = tmin
        self.tmax = tmax
        self.time_step = time_step

    def _concatenate_bursts(self, burst_data) -> dict:
        """Concatenate bursts along different channel in the same dict.

        Parameters
        -----------
        burst_data: np.array or list.
            List of the bursts and information along the different channels.

        Return
        ------
        burst_dict: dict
            A single dictionnary with bursts along all channels.
        """
        burst_dict = {}
        for key in burst_data[0]:
            burst_dict[key] = []

        for bursts in burst_data:
            for key in bursts:
                if key == "waveform_times":
                    burst_dict[key] = bursts[key]
                else:
                    burst_dict[key].extend(bursts[key])

        return burst_dict

    def _apply_solver(self, burst_waveforms, n_components) -> None:
        """Adjustment of dimensionality reduction step based on the selected algorithm.
        The function instantiates a class model based on the 'fit_transform' function
        of the selected 'solver', and the corresponding components.

        Parameters
        ----------
        burst_waveforms: numpy array
                         Numpy array containing the waveforms of all bursts that
                         will be used to fit the dimensionality reduction model.
                         It corresponds to a 2D matrix of
                         dimensions [#bursts, #waveform_time_points].
        n_components: int or float.
                      Number of components to be returned by the scikit PCA function
                      for any dimensionality reduction method, or percentage of
                      explained variance.

        Attributes
        ----------
        drm: scikit-learn
            PCA model
        components: numpy array
            Array containing the transformed bursts as returned by the
                   'fit_transform' method of PCA.
        """
        self.drm = PCA(n_components=n_components, svd_solver="full")
        self.drm.fit(burst_waveforms)

    def fit_transform(self, burst_dict, n_components=20):
        """Along each PCA axis, compute distance between mean waveform and waveforms identified by extreme quartiles.

        Parameters
        ----------
        burst_dict: dict
            Dictionnary with bursts from all channels

        n_components: int or float. Default to 20.
            Number of PCA components to keep.
            Float code for the percentage of the PCA components to keep.

        Return
        ------
        scores_dists: np.array.
            Distance to the mean waveform for each bursts.
        """
        if len(burst_dict) > 1:
            print("Concatenating bursts along the different channels...")
            burst_dict = self._concatenate_bursts(burst_dict)

        self.burst_dict = burst_dict
        scaler = RobustScaler()
        waveforms = np.array(self.burst_dict["waveform"])
        # Selecting bursts between 0.5 percentile and 99.5 percentile of max amplitude to
        # limit the impact of outliers on model fit. Then taking a random self.perc of bursts of that.
        amp_max = np.max(waveforms, axis=1)
        amp_map = (amp_max >= np.percentile(amp_max, 0.5)) & (
            amp_max <= np.percentile(amp_max, 99.5)
        )
        pca_subset = np.random.choice(
            np.arange(amp_max.shape[0])[amp_map],
            size=int(amp_max.shape[0] * self.perc),
            replace=False,
        )

        standardized_bursts = scaler.fit_transform(waveforms[pca_subset])
        self._apply_solver(standardized_bursts, n_components=n_components)
        self.components = self.drm.transform(waveforms)
        self.modulation_index, self.comp_waveforms = self.dist_scores()

        return self.scores_dists

    def dist_scores(self):
        """Compute distance between mean waveform and waveforms spread on quartiles.
        Also compute waveforms rate for each quartile.

        Return:
        -------
        modulation_index: np.array. Shape (n_components, nb_quartiles, nb_binning).
            Heatmaps for each components and the burst rates corresponding to the waveforms of each quartile.

        comp_waveforms: np.array. Shape (n_components, nb_quartiles, len_waveform).
            Average waveforme corresponding to each component and quartile.
        """

        self.binning = np.arange(self.tmin, self.tmax + self.time_step, self.time_step)
        modulation_index = np.empty(
            (self.components.shape[1], self.nb_quartiles, len(self.binning) - 1)
        )
        comp_waveforms = np.empty(
            (
                self.components.shape[1],
                self.nb_quartiles,
                np.array(self.burst_dict["waveform"]).shape[-1],
            )
        )

        pc_labels = ["PC_{}".format(i + 1) for i in range(self.components.shape[1])]
        features_scores = pd.DataFrame.from_dict(
            {i: self.components[:, ix] for ix, i in enumerate(pc_labels)}
        )
        quartiles = np.linspace(0, 100, num=self.nb_quartiles)
        quartiles = list(zip(quartiles[:-1], quartiles[1:]))
        for pc_ix, pc in enumerate(pc_labels):
            scores = features_scores[
                pc
            ].values  # select the apropriate principal component from the dataframe
            for q_ix, (b, e) in enumerate(quartiles):
                q_map = (scores > np.percentile(scores, b)) & (
                    scores <= np.percentile(scores, e)
                )  # create a boolean map to select the waveforms
                q_mean = np.mean(np.array(self.burst_dict["waveform"])[q_map], axis=0)
                selected_peak_time = np.array(self.burst_dict["peak_time"])[q_map]  # Peaks times for the corresponding quartile
                hist, _ = np.histogram(selected_peak_time, bins=self.binning)  # Distribution of the peaks times
                # Store results
                modulation_index[pc_ix, q_ix, :] = hist
                comp_waveforms[pc_ix, q_ix, :] = q_mean

        return modulation_index, comp_waveforms

    def plot_burst_rates(self) -> None:
        """Plot the corresponding heatmaps."""

        if not hasattr(self, 'modulation_index'):
            self.modulation_index, self.comp_waveforms = self.dist_scores()

        vmin = np.min(self.modulation_index)
        vmax = np.max(self.modulation_index)

        x_axis = self.binning[
            :-1
        ]  # Exclude the last element to match modulation_index shape
        y_axis = np.arange(
            self.modulation_index.shape[1]
        )  # Index range for comps_groups
        quartiles = np.arange(self.nb_quartiles)
        colors = plt.cm.cool(np.linspace(0, 1, num=len(quartiles)))

        # Define an offset to spread the waveforms
        waveform_offset = 5

        for i in range(
            self.modulation_index.shape[0]
        ):  # Iterate over the first dimension of modulation_index
            # Create a figure with a specific grid layout
            fig = plt.figure(figsize=(12, 6))
            gs = gridspec.GridSpec(
                1, 2, width_ratios=[1, 3], wspace=0.4
            )  # Adjust width_ratios and wspace

            # Left subplot for the waveforms
            ax_waveform = fig.add_subplot(gs[0])
            for j in range(self.nb_quartiles):
                # Apply an offset to spread out the waveforms vertically
                ax_waveform.plot(
                    self.comp_waveforms[i, j, :] + j * waveform_offset,
                    color=colors[j],
                    label=f"Group {j+1}",
                )

            ax_waveform.set_title("Average Waveforms")
            ax_waveform.set_xlabel("Time (ms)")
            ax_waveform.set_ylabel("Shapes")

            # Invert y-axis to match the heatmap orientation
            ax_waveform.invert_yaxis()

            # Remove y-ticks as they are not directly comparable to the heatmap
            ax_waveform.set_yticks([])

            # Right subplot for the heatmap
            ax_heatmap = fig.add_subplot(gs[1])
            heatmap = ax_heatmap.imshow(
                self.modulation_index[i, :, :],
                aspect="auto",
                extent=[x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]],
                origin="lower",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )

            ax_heatmap.set_xlabel("Time (s)")
            ax_heatmap.set_ylabel("Group")
            ax_heatmap.set_title(f"Burst rates for PC {i+1}")

            ax_heatmap.set_xticks(
                np.linspace(x_axis[0], x_axis[-1], num=int(len(x_axis) / 4))
            )
            ax_heatmap.set_yticks(np.linspace(y_axis[0], y_axis[-1], num=len(y_axis)))
            ax_heatmap.set_yticklabels(
                range(1, self.nb_quartiles + 1)
            )  # Match group labels

            fig.colorbar(
                heatmap,
                ax=ax_heatmap,
                label="Burst rate",
                ticks=np.linspace(vmin, vmax, num=5),
            )

    def plot_waveforms(self) -> None:
        """Viz of the burst dictionary and deviation from the mean."""
        waveforms = np.array(self.burst_dict["waveform"])
        waveform_times = self.burst_dict["waveform_times"]
        pc_labels = ["PC_{}".format(i + 1) for i in range(self.components.shape[1])]
        quartiles = np.linspace(0, 100, num=self.nb_quartiles)
        quartiles = list(zip(quartiles[:-1], quartiles[1:]))
        col_range = plt.cm.cool(np.linspace(0, 1, num=len(quartiles)))
        mean_waveform = np.mean(waveforms, axis=0)

        _, ax = plt.subplots(5, 4, figsize=(20, 25))
        ax = ax.flatten()
        for pc_ix, pc in enumerate(pc_labels):
            ax[pc_ix].set_title(pc.replace("_", " "))  # set the nice title
            for q_ix, (b, e) in enumerate(quartiles):
                ax[pc_ix].plot(
                    waveform_times,
                    self.scores_dists[q_ix, pc_ix],
                    lw=2,
                    c=col_range[q_ix],
                    label="Q {}".format(q_ix + 1),
                )
            ax[pc_ix].plot(waveform_times, mean_waveform, lw=2, c="black", label="mean")
        ax[0].legend(fontsize=10)
