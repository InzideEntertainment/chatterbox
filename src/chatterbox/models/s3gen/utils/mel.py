"""mel-spectrogram extraction in Matcha-TTS"""
import logging
from librosa.filters import mel as librosa_mel_fn
import torch
import numpy as np

logger = logging.getLogger(__name__)


# NOTE: they decalred these global vars
mel_basis = {}
hann_window = {}


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

"""
feat_extractor: !name:matcha.utils.audio.mel_spectrogram
    n_fft: 1920
    num_mels: 80
    sampling_rate: 24000
    hop_size: 480
    win_size: 1920
    fmin: 0
    fmax: 8000
    center: False

"""

def mel_spectrogram(y, n_fft=1920, num_mels=80, sampling_rate=24000, hop_size=480, win_size=1920,
                    fmin=0, fmax=8000, center=False):
    """Copied from https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/utils/audio.py
    Set default values according to Cosyvoice's config.
    """

    if isinstance(y, np.ndarray):
        y = torch.tensor(y).float()

    if len(y.shape) == 1:
        y = y[None, ]

    # Debug: Check for audio clipping (values outside [-1.0, 1.0] range)
    min_val = torch.min(y)
    max_val = torch.max(y)
    if min_val < -1.0 or max_val > 1.0:
        logger.warning(f"Audio values outside normalized range: min={min_val.item():.4f}, max={max_val.item():.4f}")

    global mel_basis, hann_window  # pylint: disable=global-statement,global-variable-not-assigned
    device = y.device
    device_str = str(device)
    cpu_device = torch.device("cpu")

    mel_key = f"{str(fmax)}_{device_str}"
    mel_cpu_key = f"{str(fmax)}_cpu"

    if mel_key not in mel_basis or mel_cpu_key not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_tensor = torch.from_numpy(mel).float()
        mel_basis[mel_cpu_key] = mel_tensor
        mel_basis[mel_key] = mel_tensor.to(device)

    if "cpu" not in hann_window:
        hann_window["cpu"] = torch.hann_window(win_size).to(cpu_device)
    if device_str not in hann_window:
        hann_window[device_str] = hann_window["cpu"].to(device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
    )
    y = y.squeeze(1)

    # Ensure the padded waveform length aligns with hop size to avoid off-by-one frame issues.
    frame_mod = (y.shape[-1] - win_size) % hop_size
    if frame_mod != 0 and (y.shape[-1] - frame_mod) >= win_size:
        y = y[..., :-frame_mod]
        logger.warning(f"Trimming {frame_mod} samples to align mel frames with hop size.")

    window_cpu = hann_window["cpu"]

    spec = torch.view_as_real(
        torch.stft(
            y.to(cpu_device),
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=window_cpu,
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    ).to(device)

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[mel_key], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def clear_mel_cache():
    """Drop cached mel filters and windows to release memory between jobs."""
    mel_basis.clear()
    hann_window.clear()
