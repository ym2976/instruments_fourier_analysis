"""
visualization
=============

Matplotlib-based visualizations for waveforms, spectrograms, MFCCs, and harmonic
partials to help explain instrument characteristics.
"""

from typing import Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


def plot_waveform_and_spectrogram(
    waveform: np.ndarray,
    sample_rate: int,
    title: str | None = None,
    n_fft: int = 2048,
    hop_length: int = 512,
):
    """
    Plots waveform and magnitude spectrogram side by side.
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    times = np.arange(len(waveform)) / sample_rate
    axes[0].plot(times, waveform)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(title or "Waveform")

    spec = np.abs(librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length))
    img = librosa.display.specshow(
        librosa.amplitude_to_db(spec, ref=np.max),
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="log",
        ax=axes[1],
        cmap="magma",
    )
    axes[1].set_title("Spectrogram")
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")
    fig.tight_layout()
    return fig, axes


def plot_mfcc(mfcc: np.ndarray, sample_rate: int, hop_length: int = 512, title: str | None = None):
    """
    Plots MFCCs with a time axis.
    """

    fig, ax = plt.subplots(figsize=(8, 4))
    img = librosa.display.specshow(mfcc, x_axis="time", sr=sample_rate, hop_length=hop_length, ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title or "MFCCs")
    fig.tight_layout()
    return fig, ax


def plot_partials(partials: Sequence[Tuple[float, float]], title: str = "Estimated Partials"):
    """
    Visualizes partial frequencies and amplitudes as a stem plot.
    """

    if not partials:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No partials detected", ha="center", va="center")
        ax.set_axis_off()
        return fig, ax

    freqs, amps = zip(*partials)
    fig, ax = plt.subplots(figsize=(8, 4))
    markerline, stemlines, baseline = ax.stem(freqs, amps, use_line_collection=True)
    plt.setp(markerline, "markerfacecolor", "orange")
    plt.setp(stemlines, "color", "gray")
    plt.setp(baseline, "color", "black", "linewidth", 1)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax
