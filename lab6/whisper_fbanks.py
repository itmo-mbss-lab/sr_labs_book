# Import of modules
from typing import Union, Optional

import numpy as np

import torch
import torchaudio
import torch.nn.functional as F
import torch.nn as nn

from whisper.audio import (
    N_SAMPLES,
    N_MELS,
    N_FFT,
    HOP_LENGTH,
    mel_filters,
)


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    # Pad or trim the audio array to N_SAMPLES, as expected by the encoder
    
    if array.shape[axis] > length:
        array = array.index_select(
            dim=axis,
            index=torch.arange(length, device=array.device),
        )

    if array.shape[axis] < length:
        pad_widths = [(0, 0)] * array.ndim
        pad_widths[axis] = (0, length - array.shape[axis])
        array = F.pad(
            array,
            [
                pad
                for sizes in pad_widths[::-1]
                for pad in sizes
            ],
        )
    
    return array


def trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    # Trim the audio array to N_SAMPLES, as expected by the encoder
    
    if array.shape[axis] > length:
        array = array.index_select(
            dim=axis,
            index=torch.arange(length, device=array.device),
        )
    
    return array


def log_mel_spectrogram(audio: torch.Tensor, n_mels: int = N_MELS,):
    # Compute log of filterbank energies
    
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(
        audio,
        N_FFT,
        HOP_LENGTH,
        window=window,
        return_complex=True,
    )
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    
    return log_spec

class ExtractWhisperFbanks80(nn.Module):
    # Extract Whisper's acoustic features
    
    def __init__(self, pad_data) -> None:
        super().__init__()

        self.instance_norm = nn.InstanceNorm1d(num_features=1, eps=1e-05)

        if pad_data:
            self.pipeline = [
                pad_or_trim,
                log_mel_spectrogram,
            ]
        else:
            self.pipeline = [
                trim,
                log_mel_spectrogram,
            ]

    def forward(
            self, samples: torch.Tensor, sample_rate: int = 16_000, **kwargs
    ) -> torch.Tensor:
        if sample_rate != 16_000:
            samples = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16_000
            )(samples)

        samples = torch.unsqueeze(samples, 1)
        samples = self.instance_norm(samples)
        samples = torch.squeeze(samples, 1)
        
        for preprocess in self.pipeline:
            samples = preprocess(samples)

        return samples