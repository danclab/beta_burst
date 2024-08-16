import numpy as np


def gaus2d(x=0, y=0, mx=0, my=0, sx=1, sy=1) -> np.ndarray:
    """Two-dimensional Gaussian function.

    Parameters
    ----------
    x : array_like
        x grid.
    y : array_like
        y grid.
    mx : float
        Mean in x dimension.
    my : float
        Mean in y dimension.
    sx : float
        Standard deviation in x dimension.
    sy : float
        Standard deviation in y dimension.

    Returns
    -------
    array_like
        Two-dimensional Gaussian distribution.
    """
    return np.exp(
        -((x - mx) ** 2.0 / (2.0 * sx**2.0) + (y - my) ** 2.0 / (2.0 * sy**2.0))
    )


def overlap(list1: list, list2: list) -> bool:
    """Find if two ranges overlap

    Parameters
    ----------
    list1 : list
        First range [low, high].
    list2 : list
        Second range [low, high].

    Returns
    -------
    bool
        True if ranges overlap, False otherwise.
    """
    return list1[0] <= list2[0] <= list1[1] or list2[0] <= list1[0] <= list2[1]


def fwhm_burst_norm(tf: np.ndarray, peak: list) -> tuple:
    """Find two-dimensional Full Width at Half Maximum (FWHM).

    Parameters
    ----------
    tf : 2D array
        TF spectrum.
    peak : list or array
        Peak of activity [freq, time].

    Returns
    -------
    tuple
        Right, left, up, and down limits for FWHM.
    """

    right_loc = np.nan
    cand = np.where(tf[peak[0], peak[1] :] <= tf[peak] / 2)[0]
    if len(cand):
        right_loc = cand[0]

    up_loc = np.nan
    cand = np.where(tf[peak[0] :, peak[1]] <= tf[peak] / 2)[0]
    if len(cand):
        up_loc = cand[0]

    left_loc = np.nan
    cand = np.where(tf[peak[0], : peak[1]] <= tf[peak] / 2)[0]
    if len(cand):
        left_loc = peak[1] - cand[-1]

    down_loc = np.nan
    cand = np.where(tf[: peak[0], peak[1]] <= tf[peak] / 2)[0]
    if len(cand):
        down_loc = peak[0] - cand[-1]

    if down_loc is np.nan:
        down_loc = up_loc
    if up_loc is np.nan:
        up_loc = down_loc
    if left_loc is np.nan:
        left_loc = right_loc
    if right_loc is np.nan:
        right_loc = left_loc

    horiz = np.nanmin([left_loc, right_loc])
    vert = np.nanmin([up_loc, down_loc])
    right_loc = horiz
    left_loc = horiz
    up_loc = vert
    down_loc = vert

    return right_loc, left_loc, up_loc, down_loc


def fwhm_burst_norm(tf: np.ndarray, peak: list):
    """
    Find two-dimensional FWHM
    :param tf: TF spectrum
    :param peak: peak of activity [freq, time]
    :return: right, left, up, down limits for FWM
    """
    right_loc = np.nan
    cand = np.where(tf[peak[0], peak[1] :] <= tf[peak] / 2)[
        0
    ]  # Find right limit (values to right of peak less than half value at peak)
    if len(cand):  # If any found, take the first one
        right_loc = cand[0]

    up_loc = np.nan
    cand = np.where(tf[peak[0] :, peak[1]] <= tf[peak] / 2)[
        0
    ]  # Find up limit (values above peak less than half value at peak)
    if len(cand):  # If any found, take the first one
        up_loc = cand[0]

    left_loc = np.nan
    cand = np.where(tf[peak[0], : peak[1]] <= tf[peak] / 2)[
        0
    ]  # Find left limit (values below peak less than half value at peak)
    if len(cand):  # If any found, take the last one
        left_loc = peak[1] - cand[-1]

    down_loc = np.nan
    cand = np.where(tf[: peak[0], peak[1]] <= tf[peak] / 2)[
        0
    ]  # Find down limit (values below peak less than half value at peak)
    if len(cand):  # If any found, take the last one
        down_loc = peak[0] - cand[-1]

    if down_loc is np.nan:  # Set arms equal if only one found
        down_loc = up_loc
    if up_loc is np.nan:
        up_loc = down_loc
    if left_loc is np.nan:
        left_loc = right_loc
    if right_loc is np.nan:
        right_loc = left_loc

    horiz = np.nanmin(
        [left_loc, right_loc]
    )  # Use the minimum arm in each direction (forces Gaussian to be symmetric in each dimension)
    vert = np.nanmin([up_loc, down_loc])
    right_loc = horiz
    left_loc = horiz
    up_loc = vert
    down_loc = vert
    return right_loc, left_loc, up_loc, down_loc
