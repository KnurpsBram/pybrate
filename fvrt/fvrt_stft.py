import torch
import torch.nn.functional as F

from fvrt.loudness import amplitude_to_intensity
from fvrt.misc     import get_device_obj

def pad_audio(audio, win_length, hop_length, center=2):
    """
    Prepare the audio for an operation that requires overlapping frames
    """
    if center == 0:
        n_to_pad = 0
    elif center == 1:
        n_to_pad = int(self.win_length // 2)
    elif center == 2:
        n_to_pad = int((self.win_length - self.hop_length) // 2)
    else:
        raise Exception(f"Unexpected value for center: {self.center}")

    audio = F.pad(audio.unsqueeze(0), (n_to_pad, n_to_pad), mode='reflect').squeeze(0)

    n_too_much = (audio.size(-1)-win_length)%hop_length
    if n_too_much > 0:
        audio = audio[..., :-n_too_much]

    return audio

class FVRTSTFT():
    def __init__(
        self,
        sr         = 16000,
        win_length = 1024,
        hop_length = 256,
        n_fft      = 1024,
        center     = 2,
        type       = "log_norm",
        mel        = True,
        n_mels     = 80,
        device     = "cuda:0",
    ):
        """
        TODO: more docstring

        `center`: int (default: 2)
            Specifies how the frames relate to the waveforms
            0 - the nth frame starts at the index `n*hop_length`
            (equal to librosa's center=False)
            1 - the nth frame is centered over the index `n*hop_length`.
            In order to achieve this, the waveform is padded with `win_length//2` waveform samples on either side
            (equal to librosa's center=True)
            2 - the nth frame is centered over the index `(n+0.5)*hop_length`.
            In order to achieve this, the waveform is padded with `(win_length-hop_length)//2` samples on either side
            With this setting a spectrogram column can be thought of representing `hop_length` waveform samples (with some waveform samples left at the right tail)
        """

        allowed_spect_types = ["complex", "mag", "log", "log_norm", "pow", "db"] # TODO: db_norm?
        assert spect_type in allowed_spect_types, f"{spect_type} not in {allowed_spect_types}"

        assert not (spect_type=="complex" and mel), "complex spectrogram can't be melscale"

        self.sr         = sr
        self.win_length = win_length
        self.hop_lenght = hop_length
        self.n_fft      = n_fft
        self.center     = center
        self.spect_type = spect_type
        self.mel        = mel
        self.n_mels     = n_mels
        self.device     = get_device_obj(device)

        self.n_rows = self.n_mels if self.mel else (self.n_fft // 2) + 1

        self.lin_to_mel_mat = torch.FloatTensor(librosa.filters.mel(sr=self.sr, n_fft=self.win_length, n_mels=self.n_mels)).to(self.device)
        self.window         = torch.hann_window(self.win_length).to(self.device)

    def __call__(self, audio):

        audio = pad_audio(
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
