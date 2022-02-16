import torch
import torch.nn.functional as F

from fvrt_audio.overlapping_frames import pad_for_overlapping_frames, nsamples_to_n_overlapping_frames
from fvrt_audio.loudness           import amplitude_to_intensity, amplitude_to_db
from fvrt_audio.misc               import get_device_obj

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

    def __call__(self, audio):

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
