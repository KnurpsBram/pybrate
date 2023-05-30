class TimeUnit():
    def __init__(
        self,
        sample_rate = None,
        samples = None,
        milliseconds = None,
        seconds = None,
        hertz = None,
    ):
        self._sample_rate = sample_rate

        assert sum([samples is not None, milliseconds is not None, seconds is not None, hertz is not None]) == 1, "Exactly one of samples, milliseconds, seconds, hertz must be specified"

        if samples is not None:
            self._samples = samples
        elif milliseconds is not None:
            self._samples = milliseconds_to_samples(milliseconds, sample_rate)
        elif seconds is not None:
            self._samples = seconds_to_samples(seconds, sample_rate)
        elif hertz is not None:
            self._samples = hertz_to_samples(hertz, sample_rate)
    
    def __add__(self, other):
        if isinstance(other, TimeUnit):
            assert self._sample_rate == other._sample_rate, "Sample rates must match"
            return TimeUnit(sample_rate=self._sample_rate, samples=self._samples + other.samples)
        else:
            raise Exception(f"Unsupported type {type(other)} for addition")

    def __sub__(self, other):
        if isinstance(other, TimeUnit):
            assert self._sample_rate == other._sample_rate, "Sample rates must match"
            return TimeUnit(sample_rate=self._sample_rate, samples=self._samples - other.samples)
        else:
            raise Exception(f"Unsupported type {type(other)} for subtraction")

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return TimeUnit(sample_rate=self._sample_rate, samples=self._samples * other)
        else:
            raise Exception(f"Unsupported type {type(other)} for multiplication")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return TimeUnit(sample_rate=self._sample_rate, samples=self._samples / other)
        else:
            raise Exception(f"Unsupported type {type(other)} for division")

    def __floordiv__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return TimeUnit(sample_rate=self._sample_rate, samples=self._samples // other)
        else:
            raise Exception(f"Unsupported type {type(other)} for division")
        
    @property
    def sample_rate(self):
        return self._sample_rate
    
    @property
    def samples(self):
        return self._samples

    @property
    def milliseconds(self):
        return samples_to_milliseconds(self._samples, sample_rate=self._sample_rate)
    
    @property
    def seconds(self):
        return samples_to_seconds(self._samples, sample_rate=self._sample_rate)
    
    @property
    def hertz(self):
        return samples_to_hertz(self._samples, sample_rate=self._sample_rate)


def seconds_to_samples(seconds, sample_rate):
    return int(seconds * sample_rate)

def samples_to_seconds(samples, sample_rate):
    return samples / sample_rate

def milliseconds_to_samples(milliseconds, sample_rate):
    return seconds_to_samples(milliseconds/1000, sample_rate=sample_rate)

def samples_to_milliseconds(samples, sample_rate):
    return 1000 * samples_to_seconds(samples, sample_rate=sample_rate)

def seconds_to_hertz(seconds):
    return 1/seconds

def hertz_to_seconds(hertz):
    return 1/hertz

def hertz_to_samples(hertz, sample_rate):
    seconds = hertz_to_seconds(hertz)
    return seconds_to_samples(seconds, sample_rate=sample_rate)

def samples_to_hertz(samples, sample_rate):
    seconds = samples_to_seconds(samples, sample_rate=sample_rate)
    return seconds_to_hertz(seconds)
