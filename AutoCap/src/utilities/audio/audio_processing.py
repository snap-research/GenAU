# Author: Moayed Haji Alu
# Email: mh155@rice.edu
import torch
import numpy as np
import librosa.util as librosa_util
from scipy.signal import get_window
from src.tools.torch_utils import random_uniform
from scipy.io.wavfile import write
import torchaudio

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


def random_uniform(start, end):
        val = torch.rand(1).item()
        return start + (end - start) * val

def resample_wav(waveform, sr, resampling_rate):
    waveform = torchaudio.functional.resample(waveform, sr, resampling_rate)
    return waveform

def normalize_wav(waveform):
    waveform = waveform - np.mean(waveform)
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    return waveform * 0.5  # Manually limit the maximum amplitude into 0.5

def random_segment_wav(waveform, target_length, random_start=None):
    waveform_length = waveform.shape[-1]
    assert waveform_length > 100, "Waveform is too short, %s" % waveform_length

    # Too short
    if (waveform_length - target_length) <= 0:
        return waveform, 0

    if random_start is None:
        for i in range(10):
            random_start = int(random_uniform(0, waveform_length - target_length))
            if torch.max(
                torch.abs(waveform[:, random_start : random_start + target_length])
                > 1e-4
            ):
                break

    return waveform[:, random_start : random_start + target_length], random_start

def pad_wav(waveform, target_length):
        waveform_length = waveform.shape[-1]
        assert waveform_length > 100, "Waveform is too short, %s" % waveform_length

        if waveform_length == target_length:
            return waveform

        # Pad
        temp_wav = np.zeros((1, target_length), dtype=np.float32)
        rand_start = 0

        temp_wav[:, rand_start : rand_start + waveform_length] = waveform
        return temp_wav

def process_wavform(waveform, sr, resampling_rate, duration):
        waveform = resample_wav(waveform, sr, resampling_rate)
        waveform = waveform.numpy()[0, ...]

        waveform = normalize_wav(waveform)

        waveform = waveform[None, ...]
        waveform = pad_wav(
            waveform, target_length=int(resampling_rate * duration)
        )
        return waveform

def read_wav_file(filename, resampling_rate, duration, random_start=None,):
        # waveform, sr = librosa.load(filename, sr=None, mono=True) # 4 times slower
        waveform, sr = torchaudio.load(filename)

        waveform, random_start = random_segment_wav(
            waveform, target_length=int(sr * duration), random_start=random_start
        )
        waveform = process_wavform(waveform, sr, 
                                   resampling_rate=resampling_rate,
                                   duration=duration)

        return waveform, random_start
