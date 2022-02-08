import torch

# TODO: documentation
def amplitude_to_intensity(x):
    return x**2

def intensity_to_amplitude(x):
    return torch.sqrt(x)

def intensity_to_db(x, safe=True):
    db = 10*torch.log10(x)
    if safe:
        db = torch.clamp(db, min=-100)
    return db

def db_to_intensity(x):
    return 10**(x/10)

def amplitude_to_db(x):
    return intensity_to_db(amplitude_to_intensity(x))

def db_to_amplitude(x):
    return intensity_to_amplitude(db_to_intensity(x))

def measure_loudness(x):
    intensity = torch.mean(amplitude_to_intensity(x))
    return intensity_to_db(intensity)

def apply_gain(x, gain=0):
    return x * db_to_amplitude(gain)

def impose_loudness(x, desired_db):
    measured_db = measure_loudness(x)
    gain = desired_db - measured_db
    return apply_gain(x, gain)
