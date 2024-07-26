import numpy as np


def gen_superlet_testdata(freqs=[20, 40, 60], cycles=11, fs=1000, eps=0):
    """
    Harmonic superposition of multiple
    few-cycle oscillations akin to the
    example of Figure 3 in Moca et al. 2021 NatComm
    """

    signal = []
    for freq in freqs:

        # 10 cycles of f1
        tvec = np.arange(cycles / freq, step=1 / fs)

        harmonic = np.cos(2 * np.pi * freq * tvec)
        f_neighbor = np.cos(2 * np.pi * (freq + 10) * tvec)
        packet = harmonic + f_neighbor

        # 2 cycles time neighbor
        delta_t = np.zeros(int(2 / freq * fs))

        # 5 cycles break
        pad = np.zeros(int(5 / freq * fs))

        signal.extend([pad, packet, delta_t, harmonic])

    # stack the packets together with some padding
    signal.append(pad)
    signal = np.concatenate(signal)

    # additive white noise
    if eps > 0:
        signal = np.random.randn(len(signal)) * eps + signal

    return signal


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    fs = 1000  # sampling frequency
    A = 10  # amplitude
    signal = A * gen_superlet_testdata(fs=fs, eps=0)  # 20Hz, 40Hz and 60Hz

    # frequencies of interest in Hz
    foi = np.linspace(1, 100, 50)
    scales = scale_from_period(1 / foi)

    spec = superlet(
        signal,
        samplerate=fs,
        scales=scales,
        order_max=30,
        order_min=1,
        c_1=5,
        adaptive=True,
    )

    ampls = np.abs(spec) # amplitude scalogram

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [1, 3]}, figsize=(6, 6)
    )

    ax1.plot(np.arange(signal.size) / fs, signal, c="cornflowerblue")
    ax1.set_ylabel("signal (a.u.)")

    extent = [0, len(signal) / fs, foi[0], foi[-1]]
    im = ax2.imshow(ampls, cmap="magma", aspect="auto", extent=extent, origin="lower")

    plt.colorbar(
        im,
        ax=ax2,
        orientation="horizontal",
        shrink=0.7,
        pad=0.2,
        label="amplitude (a.u.)",
    )

    ax2.plot([0, len(signal) / fs], [20, 20], "--", c="0.5")
    ax2.plot([0, len(signal) / fs], [40, 40], "--", c="0.5")
    ax2.plot([0, len(signal) / fs], [60, 60], "--", c="0.5")

    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("frequency (Hz)")

    fig.tight_layout()
