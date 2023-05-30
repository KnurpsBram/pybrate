import torch

from pybrate.overlapping_frames import pad_for_overlapping_frames, get_padding
from pybrate.misc import default_window

def stft(
    audio, 
    win_length, 
    hop_length, 
    center=2, 
    return_complex=True,
    n_fft=None, 
    drop_incomplete_frame=True, 
    window = None, 
    **kwargs
):
    
    if n_fft is None:
        n_fft = win_length

    if window is None:
        window = default_window(win_length)

    audio = pad_for_overlapping_frames(audio, win_length, hop_length, center, drop_incomplete_frame)

    spect = torch.stft(
        audio, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        win_length=win_length, 
        center=False,
        window=window,
        return_complex=return_complex, 
        **kwargs
    )

    return spect


def istft(
    spect, 
    hop_length, 
    center=2, 
    win_length=None, 
    window=None, 
    **kwargs
):
    
    n_fft = (spect.shape[-2] - 1) * 2

    if win_length is None:
        win_length = n_fft

    if window is None:
        window = default_window(win_length)

    audio = torch.istft(
        spect, 
        n_fft, 
        hop_length=hop_length, 
        win_length=win_length, 
        center=False, 
        window=window, 
        **kwargs
    )

    crop_left, crop_right = get_padding(win_length, hop_length, center)
    if crop_left > 0:
        audio = audio[..., crop_left:]
    if crop_right > 0:
        audio = audio[..., :-crop_right]

    return audio
