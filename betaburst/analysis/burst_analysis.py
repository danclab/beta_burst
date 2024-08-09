"""Burst dictionary creation and dimensionality reduction of
burst waveforms with PCA.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

class BurstSpace:
    """TODO

    Parameters
    -----------
    perc: float
    """

    def __init__(self, perc):
       assert 0 < perc <= 1, "`Perc` should be between 0 and 1."
       self.perc = perc

    def concatenate_bursts(self, burst_data):
        """Concatenate bursts along different channel in the same dict.
        
        Parameters
        -----------
        burst_data: np.array or list.
            List of the bursts and information along the different channels. 

        
        """
        burst_dict = {}
        for key in burst_data[0]:
            burst_dict[key] = []

        for bursts in burst_data:
            for key in bursts:
                burst_dict[key].append(bursts[key])

        return burst_dict

    def _apply_solver(self, burst_waveforms, n_components=0.9):
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
        n_components: int or float. Defauls to 0.90.
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
        self.components = self.drm.fit_transform(burst_waveforms)

    def fit_transform(self, burst_dict, n_components=20):
        """Variation in the PCA space.
        """
        scaler = RobustScaler()
        waveforms = burst_dict["waveform"]
        mean_waveform = np.mean(waveforms, axis=0)
        # Selecting bursts between 0.5 percentile and 99.5 percentile of max amplitude to 
        # limit the impact of outliers on model fit. Then taking a random 50% of bursts of that.
        amp_max = np.max(waveforms, axis=1)
        amp_map = (amp_max >= np.percentile(amp_max, 0.5)) & (amp_max <= np.percentile(amp_max, 99.5))
        pca_subset = np.random.choice(np.arange(amp_max.shape[0])[amp_map], size=int(amp_max.shape[0]*self.perc))

        standardized_bursts = scaler.fit_transform(pca_subset)
        self._apply_solver(standardized_bursts, n_components=n_components)
        scores_dists = self._dist_scores(mean_waveform, waveforms)

        return scores_dists
        

    def _dist_scores(self, mean_waveform, waveforms):
        """Compute the distribution of scores in a given subset a trials."""
        pc_labels = ["PC_{}".format(i+1) for i in range(self.components.shape[1])]
        features_scores  = pd.DataFrame.from_dict({i: self.components[:,ix] for ix, i in enumerate(pc_labels)})
        quartiles = np.linspace(0,100, num=5)
        quartiles = list(zip(quartiles[:-1], quartiles[1:]))
        scores_dists = np.zeros((len(quartiles), len(pc_labels)))
        for pc_ix, pc in enumerate(pc_labels):
            scores = features_scores[pc].values # select the apropriate principal component from the dataframe
            for q_ix, (b,e) in enumerate(quartiles):
                q_map = (scores > np.percentile(scores, b)) & (scores <= np.percentile(scores, e)) # create a boolean map to select the waveforms
                q_mean = np.mean(waveforms[q_map], axis=0)
                scores_dists[q_ix, pc_ix] = np.linalg.norm(q_mean - q_map)

        return scores_dists

    def estimate_waveforms(self):
        """Transformation of bursts used in the dimensionality reduction model
        acroos each of its axes, and estimation of the burst waveforms in a
        score-resolved manner.
        """
        return 

    def plot_dict(self):
        """Viz of the burst dictionary and deviation from the mean.
        """
        return