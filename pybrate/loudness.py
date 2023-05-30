import torch
import torch.nn.functional as F

from pybrate.overlapping_frames import audio_to_overlapping_frames
from pybrate.misc import default_window


def get_magnitude_contour(audio, win_length, hop_length, center=2, drop_incomplete_frame=True, window=None):
    """
    Get the magnitude contour of an audio

    Parameters
    ----------
    audio : torch.FloatTensor of shape __(B, T)__
        The tensor of audio
    win_length : int
        The length of a frame expressed in waveform samples
    hop_length : int
        The step size between frames, expressed in waveform samples

    Returns
    -------
    contour : torch.FloatTensor of shape __(B, T')__
        The magnitude contour
    """

    if window is None:
        window = default_window(win_length)

    window = window.reshape(1, win_length, 1)
        
    audio = audio_to_overlapping_frames(audio, win_length, hop_length, center=center, drop_incomplete_frame=drop_incomplete_frame)
    audio = audio * window 

    magnitude_contour = torch.mean(torch.abs(audio), dim=1) / torch.mean(window, dim=1)

    return magnitude_contour


def get_rms_contour(audio, win_length, hop_length, center=2, drop_incomplete_frame=True, window=None, safe=True, eps=1e-7):
    """
    Get the Root-Mean-Square contour of an audio

    Parameters
    ----------
    audio : torch.FloatTensor of shape __(B, T)__
        The tensor of audio
    win_length : int
        The length of a frame expressed in waveform samples
    hop_length : int
        The step size between frames, expressed in waveform samples

    Returns
    -------
    contour : torch.FloatTensor of shape __(B, T')__
        The magnitude contour
    """

    if window is None:
        window = default_window(win_length)

    window = window.reshape(1, win_length, 1)
        
    audio = audio_to_overlapping_frames(audio, win_length, hop_length, center=center, drop_incomplete_frame=drop_incomplete_frame)
    audio = audio * window 

    if safe:
        rms_contour = (torch.sqrt(torch.mean(audio**2, dim=1) + eps) - torch.sqrt(eps)) / torch.mean(window, dim=1)
    else:
        rms_contour = torch.sqrt(torch.mean(audio**2, dim=1)) / torch.mean(window, dim=1)

    return rms_contour


def magnitude_to_intensity(x):
    return x**2

def intensity_to_magnitude(x, safe=True, eps=1e-7):
    if safe:
        # gradient of sqrt(0) is undefined, so it should be prevented
        return torch.sqrt(x + eps) - torch.sqrt(eps)
    else:
        return torch.sqrt(x)

def intensity_to_db(x, safe=True, min=-100):
    db = 10*torch.log10(x)
    if safe:
        db = torch.clamp(db, min=min)
    return db

def db_to_intensity(x):
    return 10**(x/10)

def magnitude_to_db(x):
    return intensity_to_db(magnitude_to_intensity(x))

def db_to_magnitude(x):
    return intensity_to_magnitude(db_to_intensity(x))

def measure_loudness(x):
    intensity = torch.mean(magnitude_to_intensity(x))
    return intensity_to_db(intensity)

def apply_gain(x, gain=0):
    return x * db_to_magnitude(gain)

def impose_loudness(x, desired_db):
    measured_db = measure_loudness(x)
    gain = desired_db - measured_db
    return apply_gain(x, gain)
