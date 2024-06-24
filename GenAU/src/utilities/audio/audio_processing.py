import torch
import numpy as np
import librosa.util as librosa_util
from scipy.signal import get_window
from src.tools.torch_utils import random_uniform
from scipy.io.wavfile import write

def window_sumsquare(
    window,
    n_frames,
    hop_length,
    win_length,
    n_fft,
    dtype=np.float32,
    norm=None,
):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.

    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.

    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`

    n_frames : int > 0
        The number of analysis frames

    hop_length : int > 0
        The number of samples to advance between frames

    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.

    n_fft : int > 0
        The length of each analysis frame.

    dtype : np.dtype
        The data type of the output

    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm) ** 2
    win_sq = librosa_util.pad_center(win_sq, n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample : min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]
    return x


def griffin_lim(magnitudes, stft_fn, n_iters=30):
    """
    PARAMS
    ------
    magnitudes: spectrogram magnitudes
    stft_fn: STFT class with transform (STFT) and inverse (ISTFT) methods
    """

    angles = np.angle(np.exp(2j * np.pi * np.random.rand(*magnitudes.size())))
    angles = angles.astype(np.float32)
    angles = torch.autograd.Variable(torch.from_numpy(angles))
    signal = stft_fn.inverse(magnitudes, angles).squeeze(1)

    for i in range(n_iters):
        _, angles = stft_fn.transform(signal)
        signal = stft_fn.inverse(magnitudes, angles).squeeze(1)
    return signal


def dynamic_range_compression(x, normalize_fun=torch.log, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return normalize_fun(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def frequency_masking(self, log_mel_spec, freqm):
    bs, freq, tsteps = log_mel_spec.size()
    mask_len = int(random_uniform(freqm // 8, freqm))
    mask_start = int(random_uniform(start=0, end=freq - mask_len))
    log_mel_spec[:, mask_start : mask_start + mask_len, :] *= 0.0
    return log_mel_spec

def time_masking(self, log_mel_spec, timem):
    bs, freq, tsteps = log_mel_spec.size()
    mask_len = int(random_uniform(timem // 8, timem))
    mask_start = int(random_uniform(start=0, end=tsteps - mask_len))
    log_mel_spec[:, :, mask_start : mask_start + mask_len] *= 0.0
    return log_mel_spec

def get_mel_from_wav(audio, _stft):
    audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
    audio = torch.autograd.Variable(audio, requires_grad=False)
    melspec, magnitudes, phases, energy = _stft.mel_spectrogram(audio)
    melspec = torch.squeeze(melspec, 0).numpy().astype(np.float32)
    magnitudes = torch.squeeze(magnitudes, 0).numpy().astype(np.float32)
    energy = torch.squeeze(energy, 0).numpy().astype(np.float32)
    return melspec, magnitudes, energy


def inv_mel_spec(mel, out_filename, _stft, griffin_iters=60):
    mel = torch.stack([mel])
    mel_decompress = _stft.spectral_de_normalize(mel)
    mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
    spec_from_mel_scaling = 1000
    spec_from_mel = torch.mm(mel_decompress[0], _stft.mel_basis)
    spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
    spec_from_mel = spec_from_mel * spec_from_mel_scaling

    audio = griffin_lim(
        torch.autograd.Variable(spec_from_mel[:, :, :-1]), _stft._stft_fn, griffin_iters
    )

    audio = audio.squeeze()
    audio = audio.cpu().numpy()
    audio_path = out_filename
    write(audio_path, _stft.sampling_rate, audio)
