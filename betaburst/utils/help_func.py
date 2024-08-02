"""
Helper functions.
"""

import json, sys
import numpy as np
from os import makedirs
from os.path import join, exists
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from mne.stats import permutation_cluster_test


def load_exp_variables(json_filename):
    """
    Load json file that contains variables relevant to the experiment.
    """

    check_existence(json_filename)

    with open(json_filename) as var_file:
        dataset_vars = json.load(var_file)

    return dataset_vars


def check_existence(filename):
    """
    Check the existence of a file. If it does not exist inform
    and quit the program.
    """

    if exists(filename):
        pass
    else:
        warn = "The data path you have provided does not exist: '{}'".format(filename)
        raise NameError(warn)


def dir_creation(subject, savepath):
    """
    Create subdirectories for each subject of a dataset given a parent directory.

    Parameters
    ----------
    subject : int
              Number of subject for data analysis.
    savepath: str
              Parent directory that contains all results
    """

    data_dirname = join(savepath, "sub_{}".format(subject))

    if not exists(data_dirname):
        makedirs(data_dirname)


def diff_multiple_lists(lists):
    """
    Compare multiple lists and find the elements that are missing from each
    comparison.

    The input should be ordered from the lengthiest to the sortest list.

    Parameters
    ----------
    lists: list
           List of lists with partially overlapping elements.

    Returns
    -------
    missing: list
             List of missing elements across all comparisons.
    """

    # List with all possible trials.
    the_list = list(set.union(*map(set, [c for c in lists])))

    missing = []
    for lst in lists:
        lists_comp = [the_list, lst]
        comp = set.difference(*map(set, [lc for lc in lists_comp]))
        if len(comp) != 0:
            missing.append(list(comp))

    if len(missing) > 0:
        missing = list(set.union(*map(set, [c for c in missing])))

    return missing


def ascertain_trials(burst_data, channel_ids, labels):
    """
    Parse all channels of a subject and keep bursts from
    trials that are available for all channels.

    Parameters
    ----------
    burst_data: numpy array
                Array of dictionaries, each one containing the detected
                bursts corresponding to a signle channel.
    channel_ids: list or str
                 Indices of channels to take into account during burst
                 dictionary creation. If set to 'all' take into account
                 all available channels.
    labels: list
            List containing the labels of all experimental trials.

    Returns
    -------
    common_trials: list
                   List containing all trials with detected bursts across
                   channels per experimental condition.
    missing_trials: list
                    List containing all trials without detected bursts in
                    at least one channel.
    """

    # Indices of available channels and trials.
    unique_channels = []
    for d in range(len(burst_data)):
        try:
            unique_channels.append(np.unique(burst_data[d]["channel"])[0])
        except:
            continue

    # Keep only channels of interest.
    if channel_ids == "all":
        unique_trials = [np.unique(burst_data[ch]["trial"]) for ch in unique_channels]
    elif channel_ids != "all":
        unique_trials = [
            np.unique(burst_data[ch]["trial"])
            for ch in unique_channels
            if ch in channel_ids
        ]

    # Find common trials.
    common_trials = list(set.intersection(*map(set, [l for l in unique_trials])))
    common_trials.sort()

    # Notify and exit if condition is not met.
    if len(common_trials) == 0:
        msg = (
            "The selected channels have no common trials with detected bursts. "
            + "Consider using alternative channels or not using this subject..."
        )
        raise RuntimeError(msg)

    # Get rid of labels of missing trials.
    missing_trials = diff_multiple_lists(unique_trials)
    # In case some trials are not present for any channel, take
    # them into account as well.
    if len(missing_trials) + len(common_trials) < len(labels):
        theor_trials = np.arange(0, len(labels), 1)
        unique_trials.insert(0, theor_trials)
        missing_trials = diff_multiple_lists(unique_trials)
    labels = np.delete(labels, missing_trials)

    # Split trials in conditions.
    unique_labels = np.unique(labels)
    cond1 = np.array(common_trials)[np.where(labels == unique_labels[0])[0]].tolist()
    if len(unique_labels) > 1:
        cond2 = np.array(common_trials)[np.where(labels == unique_labels[1])[0]].tolist()

    # Return the common trials groupped by condition.
    if len(unique_labels) > 1:
        common_trials = [cond1, cond2]
    else:
        common_trials = [cond1[:len(cond1)//2], cond1[len(cond1)//2:]]

    return common_trials, missing_trials


def dict_bycondition(subject_dictionary, trials_condition):
    """
    Parse a subjects' bursts dictionary and keep trials correspdonding to
    a given experimental condition.

    Parameters
    ----------
    subject_dictionary: dict
                        Dictionary containing all detected bursts of 'subject'.
    trials_condition: 1D nupy array
                      Trials corresponding to the given condition.

    Returns
    -------
    condition_dictionary: dict
                          Dictionary containing all detected bursts of 'subject'
                          for the given condition.
    dictionary_condition_trials: list
                                 Indices of the total trials in 'subject dictionary'
                                 that correspond to the 'trials_condition'.
    """

    # Variables initialization.
    condition_dictionary = {}
    dictionary_condition_trials = []

    # Bursts parsing.
    for t, tr in enumerate(trials_condition):
        trials_ids = np.where(subject_dictionary["trial"] == tr)[0]
        dictionary_condition_trials.append(trials_ids)

        for key, value in subject_dictionary.items():
            if t == 0:
                if key != "waveform_times":
                    condition_dictionary[key] = value[trials_ids]
                else:
                    condition_dictionary[key] = value
            else:
                if key == "waveform_times":
                    continue
                elif key == "waveform":
                    condition_dictionary[key] = np.vstack(
                        (condition_dictionary[key], value[trials_ids])
                    )
                else:
                    condition_dictionary[key] = np.hstack(
                        (condition_dictionary[key], value[trials_ids])
                    )

    return condition_dictionary, dictionary_condition_trials


def circumference(matrix):
    """
    Given a matrix with NANs and 1s, create a new matrix of
    the same shape with 1s only on the borders of 1-clusters.

    Parameters
    ----------
    matrix : numpy array
             Input matrix containing NANs, and potentially
             clusters of 1s.

    Returns
    -------
    circ_matrix : numpy array
                  Output matrix of the same dimensionsality
                  that contains NANs. Only the 1s forming
                  the clusters' contours are kept.
    """

    # Initialization of output matrix.
    circ_matrix = np.copy(matrix) * np.nan

    for x in range(circ_matrix.shape[0]):
        for y in range(circ_matrix.shape[1]):
            # Matrix corners.
            if x == 0 and y == 0:
                if matrix[x, y + 1] == 1 or matrix[x + 1, y] == 1:
                    circ_matrix[x, y] = 1
            elif x == 0 and y == circ_matrix.shape[1] - 1:
                if matrix[x, y - 1] == 1 or matrix[x + 1, y] == 1:
                    circ_matrix[x, y] = 1
            elif x == circ_matrix.shape[0] - 1 and y == 0:
                if matrix[x, y + 1] == 1 or matrix[x - 1, y] == 1:
                    circ_matrix[x, y] = 1
            elif x == circ_matrix.shape[0] - 1 and y == circ_matrix.shape[1] - 1:
                if matrix[x, y - 1] == 1 or matrix[x - 1, y] == 1:
                    circ_matrix[x, y] = 1

            # Matrix edges.
            elif x == 0 and (y > 0 and y < circ_matrix.shape[1] - 1):
                if (
                    matrix[x, y] == 1
                    and np.nansum(
                        [matrix[x, y - 1], matrix[x, y + 1], matrix[x + 1, y]]
                    )
                    <= 3
                ):
                    circ_matrix[x, y] = 1
            elif x == circ_matrix.shape[0] - 1 and (
                y > 0 and y < circ_matrix.shape[1] - 1
            ):
                if (
                    matrix[x, y] == 1
                    and np.nansum(
                        [matrix[x, y - 1], matrix[x, y + 1], matrix[x - 1, y]]
                    )
                    <= 3
                ):
                    circ_matrix[x, y] = 1
            elif y == 0 and (x > 0 and x < circ_matrix.shape[0] - 1):
                if (
                    matrix[x, y] == 1
                    and np.nansum(
                        [matrix[x - 1, y], matrix[x + 1, y], matrix[x, y + 1]]
                    )
                    <= 3
                ):
                    circ_matrix[x, y] = 1
            elif y == circ_matrix.shape[1] - 1 and (
                x > 0 and x < circ_matrix.shape[0] - 1
            ):
                if (
                    matrix[x, y] == 1
                    and np.nansum(
                        [matrix[x - 1, y], matrix[x + 1, y], matrix[x, y - 1]]
                    )
                    <= 3
                ):
                    circ_matrix[x, y] = 1

            # Rest.
            else:
                if (
                    matrix[x, y] == 1
                    and np.nansum(
                        [
                            matrix[x - 1, y],
                            matrix[x + 1, y],
                            matrix[x, y + 1],
                            matrix[x, y - 1],
                        ]
                    )
                    < 4
                ):
                    circ_matrix[x, y] = 1

    return circ_matrix


def significant_features(
    matrix,
    scores_lims,
    scores_bins,
    task_bins,
    significant_features_ids,
    comp,
    comps_groups,
    significance_threshold=0.25,
):
    """
    Given a matrix of clusters, find the feature groups that correspond to
    the clusters.

    Parameters
    ----------
    matrix: numpy array
            Input matrix containing NANs, and potentially clusters of 1s.
    scores_lims : numpy array
                  Array containing the scores corresponding to the limits
                  for splitting a component axis to a given number of features.
    scores_bins: int
                Number of bins for score axis discretization.
    task_bins : numpy array
                Indices of bins of trimmed experimental time corresponding
                to the task period.
    significant_features_ids: list
                              List filled iteratively with the indices of the
                              features that overlap with statistically significant
                              clusters along each component axis.
    comp: int
          Index of component analyzed.
    comps_groups: int
                  Number of groups the scores of each component axis should be
                  split into.
    significance_threshold: float, optional
                            Fraction of pixels that indicates the lower bound
                            for marking a feature as significant.
                            Defaults to 0.25.
    """

    # Reorder matrix values to match those of the figures.
    matrix = np.flip(matrix.T)

    # Correspondance between feature groups limits and score space for figures.
    scores_space = np.linspace(scores_lims[0, 0], scores_lims[-1, -1], scores_bins)

    # Identification of overlap of all clusters within each feature group.
    previous_id = 0
    for j, limit in enumerate(scores_lims[:, 1]):
        # Ensure numerical stability.
        score_id = np.where(np.around(scores_space, 5) >= np.around(limit, 5))[0][0]

        # Total number of pixels corresponding to significant clusters.
        total_pixels = (
            matrix[previous_id:score_id, task_bins].shape[0]
            * matrix[previous_id:score_id, task_bins].shape[1]
        )
        cluster_pixels = np.nansum(matrix[previous_id:score_id, task_bins])

        # Apply a threshold and mark significant features.
        if cluster_pixels >= significance_threshold * total_pixels:
            significant_features_ids.append(j + comp * comps_groups)

        previous_id = score_id


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor="none", **kwargs):
    """
    Taken from:
    https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html#sphx-glr-gallery-statistics-confidence-ellipse-py

    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    if x.size == 0 or y.size == 0:
        sys.exit()

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs
    )

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)

    return ax.add_patch(ellipse)


def vis_permutation_cluster_test(
    data, data_labels, colors, sub_id, threshold=0.05, n_perm=1024
):
    """
    Perform two-sided permutation cluster test on 2D arrays, time
    being the second dimension. Then return an array that indicates
    significant clusters, the "direction", as well as the colors to
    be used for visualization.

    Parameters
    ----------
    data: list
          List containing the data of multiple groups of observations
          D = [X1, X2, ..., Xn].
    data_labels: list
                 List of strings, each one indicating what each group
                 in the 'data' represents.
    colors: list
            List of colors, each corresponding to a group in 'data'.
    sub_id: int or None
            Index of subject whose data are currently plotted. If None,
            assume use of across-subjects average data.
    threshold: float or None, optional
               Threshold for evaluating the statistical significance of
               t-tests, implementing Sidak correction for multiple tests.
               If set to 'None' it will be determined as the 95th
               percentile of an empirical distribution of maximal statistics.
               Defaults to 0.05.
    n_perm: int, optional
            Number of permutation to be performed.
            Defaults to 1024.

    Returns
    -------
    perm_cl_test: list
                  List of lists, each containing statistically significant
                  data points, and the corresponding color.
    """

    perm_cl_test = []

    # Iterate through all possible comparisons of "conv" and rest of the data.
    txt_of_int = "conv" if "conv" in data_labels else "conv_sliding"
    conv_id = np.where(np.array(data_labels) == txt_of_int)[0][0]

    for dat, data_label, color in zip(data, data_labels, colors):
        if data_label == "conv":
            pass

        # Calculate the observed difference of the the distributions
        # per time step.
        if sub_id != None:
            obs_diff = [
                np.mean(data[conv_id][sub_id, :], axis=-1).T
                - np.mean(dat[sub_id, :], axis=-1).T,
                np.mean(dat[sub_id, :], axis=-1).T
                - np.mean(data[conv_id][sub_id, :], axis=-1).T,
            ]
        else:
            obs_diff = [
                np.mean(data[conv_id], axis=(0, -1))
                - np.mean(dat, axis=(0, -1)),
                np.mean(dat, axis=(0, -1))
                - np.mean(data[conv_id], axis=(0, -1)),
            ]

        # Data concatenation.
        if sub_id != None:
            all_data = [data[conv_id][sub_id, :].T, dat[sub_id, :].T]
        else:
            all_data = [np.mean(data[conv_id], axis=0).T, np.mean(dat, axis=0).T]

        # Null distribution matrix initialization.
        null_dist = np.empty((n_perm, all_data[0].shape[1]))

        # Initialization of p-values for each time step.
        pvals = np.empty((2, all_data[0].shape[1]))

        # Permutations.
        for p in range(n_perm):
            # Selection of random indices at each time step.
            indices = np.zeros(
                (all_data[0].shape[0] + all_data[1].shape[0], all_data[0].shape[1])
            )
            all_perm = np.zeros(
                (all_data[0].shape[0] + all_data[1].shape[0], all_data[0].shape[1])
            )

            for t in range(all_data[0].shape[1]):
                indices[:, t] = np.random.permutation(indices.shape[0])
                all_perm[:, t] = np.vstack(all_data)[indices[:, t].astype(int), t]

            # Splitting of random distribution.
            dat1_rand = all_perm[: all_data[0].shape[0], :]
            dat2_rand = all_perm[all_data[0].shape[0] :, :]

            # Null distribution mean.
            null_dist[p, :] = np.mean(dat1_rand, axis=0) - np.mean(dat2_rand, axis=0)

        # Null distribution sorting.
        for c in range(null_dist.shape[1]):
            null_dist[:, c] = np.sort(null_dist[:, c])
        
        # Empirical ditribution of maximal statistic of null distribution.
        if threshold == None:
            threshold = np.percentile(np.max(null_dist[-1, :]), 95)
        else:
            threshold = 1 - (1 - threshold) ** (1 / all_data[0].shape[1])

        # P-value for each time step.
        for t in range(all_data[0].shape[1]):
            pvals[0, t] = np.sum(np.abs(null_dist[:, t] >= obs_diff[0][t])) / n_perm
            pvals[1, t] = np.sum(np.abs(null_dist[:, t] >= obs_diff[1][t])) / n_perm

        # Keep only significant time points.
        pvals[np.where(pvals > threshold)] = np.NAN

        # Assign suitable-for-plotting values to retained time points.
        pvals[~np.isnan(pvals)] = 1.0 + 0.03 + (0.02 * len(perm_cl_test))

        perm_cl_test.append([pvals, [colors[conv_id], color]])

    return perm_cl_test
