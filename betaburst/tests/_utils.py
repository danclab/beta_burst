import numpy as np


def generate_transient_minimum(w_size, freq=1, amplitude=0.5, decay=0.002):
    """
    Generate a transient oscillation with a clear local minimum at the center.

    Parameters:
        w_size (int): The size of the window (number of data points).
        freq (float): The frequency of the oscillation (controls how many oscillations occur).
        amplitude (float): The amplitude of the oscillation.
        decay (float): Decay factor for the transient effect.

    Returns:
        np.ndarray: Array containing the transient oscillation with a minimum at the center.
    """

    t = np.linspace(-1, 1, w_size)
    oscillation = -amplitude * np.cos(2 * np.pi * freq * t)
    window = np.exp(-decay * (t * w_size) ** 2)  # Transient effect
    transient_oscillation = oscillation * window

    return transient_oscillation
