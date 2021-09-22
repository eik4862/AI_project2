from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2, entropy
from scipy.interpolate import CubicSpline
from scipy.signal import welch, coherence
from Util import *


@final
class Simulation:
    """
    Toolbox for simulation.
    """

    def __init__(self) -> None:
        raise NotImplementedError

    @classmethod
    def signal(cls, n_source: int, amp: List[float], freq: List[float], t: float, fs: float, noise: bool = True,
               noise_amp: float = .1, seed: Optional[int] = None) -> np.ndarray:
        """
        Simulates sinusoidal signals.

        :param n_source: # of signal sources.
        :param amp: Amplitude of each source.
        :param freq: Frequency of each source in Hz.
        :param t: Length of observation in sec.
        :param fs: Sampling frequency in Hz.
        :param noise: If true, it adds white noise.
        :param noise_amp: Amplitude of white noise.
        :param seed: Seed for random process.
        :return: Simulated signal.
        """
        t_sample: np.ndarray = np.linspace(0, t, num=int(t * fs), endpoint=True)
        soucrce: List[np.ndarray] = []

        for i in range(n_source):
            soucrce.append(amp[i] * np.sin(2 * np.pi * freq[i] * t_sample))

        soucrce: np.ndarray = np.array(soucrce).transpose()

        if noise:
            np.random.seed(seed)
            noise: np.ndarray = noise_amp * np.random.normal(size=len(t_sample))

            return np.sum(soucrce, axis=1) + noise
        else:
            return np.sum(soucrce, axis=1)

    @classmethod
    def psd(cls, signal: np.ndarray, fs: float, plot: bool = False, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes power spectral density.

        :param signal: Signal whose PSD is to be computed.
        :param fs: Sampling frequency in Hz.
        :param plot: If true, it plots the result.
        :param kwargs: Additional arguments for function welch.
        :return: Frequency of PSD and PSD itself.
        """
        freq, Pxx = welch(signal, fs, **kwargs)

        if plot:
            plt.plot(freq, Pxx)
            plt.show()

        return freq, Pxx

    @classmethod
    def csd(cls, signal_1: np.ndarray, signal_2: np.ndarray, fs: float, plot: bool = False,
            **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes cross power spectral density.

        :param signal_1: Signal whose CSD is to be computed.
        :param signal_2: Signal whose CSD is to be computed.
        :param fs: Sampling frequency in Hz.
        :param plot: If true, it plots the result.
        :param kwargs: Additional arguments for function coherence.
        :return: Frequency of CSD and CSD itself.
        """
        freq, Cxy = coherence(signal_1, signal_2, fs, **kwargs)

        if plot:
            plt.plot(freq, Cxy)
            plt.show()

        return freq, Cxy

    @classmethod
    def entropy(cls, signal: np.ndarray, fs: float, **kwargs: Any) -> np.ndarray:
        """
        Computes entropy.

        :param signal: Signal whose entropy is to be computed.
        :param fs: Sampling frequency in Hz.
        :param kwargs: Additional arguments for function welch.
        :return: Computed entropy.
        """
        _, Pxx = welch(signal, fs, **kwargs)
        return entropy(Pxx).squeeze()


if __name__ == '__main__':
    # The code below is NOT directly related to the main analysis.
    fs, t = 10000, 100
    signal_1: np.ndarray = Simulation.signal(3, [1, .5, 2], [1000, 2000, 500], t, fs)
    signal_2: np.ndarray = Simulation.signal(2, [1, 2], [1000, 500], t, fs)
    freq, Pxx = Simulation.psd(signal_1, fs, nperseg=1024)
    freq, Cxy = Simulation.csd(signal_1, signal_2, fs, nperseg=1024)

    header: List[str] = ['f', 'psd']
    body: List[List[Any]] = np.vstack([freq, Pxx]).transpose().tolist()
    Writer.write('psd.csv', header, body)

    header = ['f', 'csd']
    body = np.vstack([freq, Cxy]).transpose().tolist()
    Writer.write('csd.csv', header, body)

    amp: List[float] = (chi2.pdf(np.linspace(1, 10, num=30, endpoint=True), df=5) * 100).tolist()
    freq: List[float] = (np.linspace(1, 10, num=30, endpoint=True) * 500).tolist()
    complex_signal: np.ndarray = Simulation.signal(len(freq), amp, freq, t, fs, noise_amp=50)
    simple_signal: np.ndarray = Simulation.signal(1, [1], [1000], t, fs)
    freq, Pxx_complex = Simulation.psd(complex_signal, fs, nperseg=1024)
    _, Pxx_simple = Simulation.psd(simple_signal, fs, nperseg=1024)

    header = ['t', 'signal']
    body = np.vstack([np.linspace(0, t, num=int(t * fs), endpoint=True)[:500],
                      complex_signal['signal'][:500]]).transpose().tolist()
    Writer.write('signal.csv', header, body)

    header = ['f', 'psd']
    body = np.vstack([freq, Pxx_complex]).transpose().tolist()
    Writer.write('psd_complex.csv', header, body)

    header = ['f', 'psd']
    body = np.vstack([freq, Pxx_simple]).transpose().tolist()
    Writer.write('psd_simple.csv', header, body)

    print(Simulation.entropy(simple_signal, fs))
    print(Simulation.entropy(complex_signal, fs))

    _, Pxx = Simulation.psd(complex_signal, fs, True, nperseg=64)
    Pxx: np.ndarray = CubicSpline(np.linspace(0, 100, num=Pxx.shape[0], endpoint=True), Pxx)(np.arange(101))

    header = ['f', 'psd']
    body = np.vstack([np.arange(101), Pxx]).transpose().tolist()
    Writer.write('psd_band.csv', header, body)
