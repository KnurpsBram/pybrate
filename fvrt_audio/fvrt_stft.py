import librosa

import numpy as np

import torch
import torch.nn.functional as F

from fvrt_audio.overlapping_frames import pad_for_overlapping_frames, nsamples_to_n_overlapping_frames
from fvrt_audio.loudness           import amplitude_to_intensity, amplitude_to_db
from fvrt_audio.misc               import get_device_obj

def stft(audio, win_length, hop_length, center=1, n_fft=None, **kwargs):
    
    if n_fft is None:
        n_fft = win_length

    if center == 0:
        n_to_pad = 0
    elif center == 1:
        n_to_pad = win_length // 2
    elif center == 2:
        n_to_pad = (win_length - hop_length)/2
    else:
        raise Exception(f"Unexpected value for center {center}")

    audio = F.pad(audio, (int(np.floor(n_to_pad)), int(np.ceil(n_to_pad))), mode='reflect')

    spect = torch.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=False, **kwargs)

    return spect

def istft(spect, hop_length, center=1, win_length=None, **kwargs):
    
    n_fft = (spect.shape[-2] - 1) * 2

    if win_length is None:
        win_length = n_fft

    if center == 0:
        n_to_crop = 0
    elif center == 1:
        n_to_crop = win_length // 2
    elif center == 2:
        n_to_crop = (win_length - hop_length)//2
    else:
        raise Exception(f"Unexpected value for center {center}")

    audio = torch.istft(spect, n_fft, hop_length=hop_length, win_length=win_length, center=False, **kwargs)

    ### THERE'S A BUG IN PYTORCH ISTFT
    # https://github.com/pytorch/pytorch/issues/79778
    # when calling istft with center=False, the audio that comes back has one too few samples. (center=True seems to work fine)
    # this doesn't match the behaviour of librosa.
    # really weird.
    audio = F.pad(audio, (0, 1), mode='reflect')

    if n_to_crop > 0:
        audio = audio[..., n_to_crop:-n_to_crop]

    return audio

class FVRTSTFT():
    def __init__(
        self,
        sr         = 16000,
        win_length = 1024,
        hop_length = 256,
        n_fft      = 1024,
        center     = 2,
        spect_type = "log_norm",
        mel        = True,
        n_mels     = 80,
        device     = "cuda:0",
    ):
        """
        TODO: Documentation
        """

        allowed_spect_types = ["complex", "mag", "log", "log_norm", "pow", "db"] # TODO: db_norm?
        assert spect_type in allowed_spect_types, f"{spect_type} not in {allowed_spect_types}"

        assert not (spect_type=="complex" and mel), "complex spectrogram can't be melscale"

        self.sr         = sr
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft      = n_fft
        self.center     = center
        self.spect_type = spect_type
        self.mel        = mel
        self.n_mels     = n_mels
        self.device     = get_device_obj(device)

        self.n_rows  = self.n_mels if self.mel else (self.n_fft // 2) + 1
        self.nyquist = self.sr // 2

        if mel:
            self.lin_to_mel_mat = torch.FloatTensor(librosa.filters.mel(sr=self.sr, n_fft=self.win_length, n_mels=self.n_mels)).to(self.device)
        else:
            self.bin_width_hz = self.nyquist / self.n_rows

        self.window = torch.hann_window(self.win_length).to(self.device)

    def audio_lengths_to_spect_lengths(self, audio_lengths):
        return nsamples_to_n_overlapping_frames

    def __call__(self, audio): # TODO: make spect_type overwriteable through kwarg?

        audio = pad_for_overlapping_frames(
            audio,
            win_length = self.win_length,
            hop_length = self.hop_length,
            center     = self.center
        )

        spect = torch.stft(
            audio,
            win_length     = self.win_length,
            hop_length     = self.hop_length,
            n_fft          = self.win_length,
            center         = False, # padding is handled by self.pad_audio() and should be omitted by torch.stft()
            window         = self.window,
            return_complex = True
        )

        if self.spect_type == "complex":
            return spect

        spect = torch.abs(spect)
        if self.mel:
            spect = self.lin_to_mel_mat @ spect

        if   self.spect_type == "mag":
            return spect
        elif self.spect_type == "log":
            return torch.log(spect)
        elif self.spect_type == "log_norm":
            return (torch.log(spect) + 5.0) / 5.0
        elif self.spect_type == "pow":
            spect = amplitude_to_intensity(spect)
            return spect
        elif self.spect_type == "db":
            spect = amplitude_to_db(spect)
            return spect

    def istft(self, spect):

        spect = torch.istft(
            spect,
            hop_length     = self.hop_length,
            n_fft          = self.win_length,
            center         = False,
            window         = self.window,
        )

        if self.center == 0:
            n_to_crop = self.win_length // 2
        elif self.center == 1:
            n_to_crop = 0
        elif self.center == 2:
            n_to_crop == (self.win_length - self.hop_length)//2
        else:
            raise Exception(f"Unexpected value for center {self.center}")

        if n_to_crop > 0:
            audio = audio[..., n_to_crop:-n_to_crop]

        return audio
