import numpy as np

import torch.nn.functional as F


def audio_to_overlapping_frames(audio, win_length, hop_length, center=2, drop_incomplete_frame=True):
    """
    Returns a view of the audio tensor with contains slices of size `win_length`.

    The padding strategy depends on the values of `center`:

    - 0: the nth frame starts at the index `n*hop_length`. Equivalent to `center=False` in the convention of the `librosa` package.
    - 1: the nth frame is centered over the index `n*hop_length`. In order to achieve this, the waveform is padded with `win_length//2` waveform samples on either side. Equivalent to `center=True` in the convention of the `librosa` package.
    - 2: the nth frame is centered over the index `(n+0.5)*hop_length`. In order to achieve this, the waveform is padded with `(win_length-hop_length)//2` samples on either side. With this setting a spectrogram column can be thought of representing `hop_length` waveform samples (excluding any leftover waveform samples that can't form a frame on the right tail)

    Example (psuedocode):
    ```
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    print(audio_to_overlapping_frames(x, win_length=5, hop_length=3, center=0, drop_incomplete_frame=True))
    >>> [[0, 1, 2, 3, 4], [3, 4, 5, 6, 7], [6, 7, 8, 9, 10]]

    print(audio_to_overlapping_frames(x, win_length=5, hop_length=3, center=1, drop_incomplete_frame=True))
    >>> [[1, 0, 0, 1, 2], [1, 2, 3, 4, 5], [4, 5, 6, 7, 8], [7, 8, 9, 10, 11]]

    print(audio_to_overlapping_frames(x, win_length=5, hop_length=3, center=2, drop_incomplete_frame=True))
    >>> [[0, 0, 1, 2, 3], [2, 3, 4, 5, 6], [5, 6, 7, 8, 9], [8, 9, 10, 11, 0]]
    ```

    Parameters
    ----------
    audio : torch.FloatTensor of shape __(B, T)__
        The tensor of audio
    win_length : int
        The length of a frame expressed in waveform samples
    hop_length : int
        The step size between frames, expressed in waveform samples
    center : int
        The parameter that determines how the audio will be padded
    drop_incomplete_frame: bool
        Whether to crop the right tail if there are waveform values that don't exactly fit in a frame

    Returns
    -------
    frames : torch.FloatTensor of shape __(B, W, N)__
        Overlapping frames of audio.  `W` is `win_length`, `N` is the amount of frames.
    """
    audio  = pad_for_overlapping_frames(audio, win_length=win_length, hop_length=hop_length, center=center, drop_incomplete_frame=drop_incomplete_frame)
    frames = audio.unfold(dimension=-1, size=win_length, step=hop_length) # TODO: what if drop_incomplete_frames is False?
    frames = frames.permute(0, 2, 1) # (B, N, W) to (B, W, N)
    return frames


def pad_for_overlapping_frames(audio, win_length, hop_length, center=2, drop_incomplete_frame=True):
    """
    Prepare the audio for an operation that requires overlapping frames

    See `audio_to_overlapping_frames()`

    Parameters
    ----------
    audio : torch.FloatTensor of shape __(B, T)__
        The tensor of audio
    win_length : int
        The length of a frame expressed in waveform samples
    hop_length : int
        The step size between frames, expressed in waveform samples
    center : int
        The parameter that determines how the audio will be padded
    drop_incomplete_frame: bool
        Whether to crop the right tail if there are waveform values that don't exactly fit in a frame

    Returns
    -------
    audio : torch.FloatTensor of shape __(B, T')__
        Padded (and/or cropped) audio
    """

    padding_left, padding_right = get_padding(win_length, hop_length, center=center)

    audio = F.pad(audio.unsqueeze(0), (padding_left, padding_right), mode='reflect').squeeze(0)

    if drop_incomplete_frame:
        n_too_much = (audio.size(-1)-win_length)%hop_length
        if n_too_much > 0:
            audio = audio[..., :-n_too_much]

    return audio


def get_padding(win_length, hop_length, center=2):
    """
    Determine how much padding an audio needs to be prepared for an operation that requires overlapping frames

    See `audio_to_overlapping_frames()`

    Parameters
    ----------
    win_length : int
        The length of a frame expressed in waveform samples
    hop_length : int
        The step size between frames, expressed in waveform samples
    center : int
        The parameter that determines how the audio will be padded

    Returns
    -------
    padding_left : int
        Amount of samples to be padded to the left of the audio
    padding_right : int
        Amount of samples to be padded to the right of the audio
    """

    if center == 0:
        padding_left = padding_right = 0
    elif center == 1:
        padding_left = padding_right = int(win_length // 2)
    elif center == 2:
        n_to_pad = (win_length - hop_length)/2
        padding_left = int(np.floor(n_to_pad))
        padding_right = int(np.ceil(n_to_pad))
    else:
        raise Exception(f"Unexpected value for center: {center}")
    
    return padding_left, padding_right