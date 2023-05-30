import math
import torch

from pybrate.time_unit import seconds_to_samples, samples_to_seconds


def get_wave_angles(frequency, sample_rate, samples=None, seconds=None, angle_offset=0, sample_dither=0):
    """
    Angle values of 2*pi*k (where k are real numbers) are considered the start and end of a wave cycle.

    sample_dither = 0.0 is more precise for waveform generation using math (like sine wave)
    sample_dither = 0.5 is more precise for waveform generation using counts (like square wave)
    """

    if samples is None:
        samples = seconds_to_samples(seconds, sample_rate)

    sample_indices = torch.arange(0, samples) + sample_dither
    wave_angles = (sample_indices * 2 * math.pi * frequency / sample_rate) + angle_offset

    return wave_angles


def generate_sine_wave(frequency, sample_rate, samples=None, seconds=None, angle_offset=0):

    angles = get_wave_angles(frequency, sample_rate, samples, seconds, angle_offset)
    sine_wave = torch.sin(angles)

    return sine_wave


def generate_square_wave(frequency, sample_rate, samples=None, seconds=None, angle_offset=0):

    angles = get_wave_angles(frequency, sample_rate, samples, seconds, angle_offset, sample_dither=0.5)
    square_wave = torch.where(angles % (2 * math.pi) >= math.pi, -1., 1.)

    return square_wave


def generate_sawtooth(frequency, sample_rate, samples=None, seconds=None, angle_offset=0):

    angles = get_wave_angles(frequency, sample_rate, samples, seconds, angle_offset)
    sawtooth = (torch.remainder(angles, 2 * math.pi) / math.pi) - 1

    return sawtooth


def generate_pulse_train(frequency, sample_rate, samples=None, seconds=None, angle_offset=0):

    angles = get_wave_angles(frequency, sample_rate, samples, seconds, angle_offset, sample_dither=0.5)
    threshold = 2 * math.pi * frequency / sample_rate
    pulse_train = torch.where(angles % (2 * math.pi) < threshold, 1., 0.)

    return pulse_train