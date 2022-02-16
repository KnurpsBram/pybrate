def nsec_to_nsamples(nsec, sr):
    """
    Convert number of seconds to number of waveform samples

    Parameters
    ----------
    nsec : float
        Number of seconds
    sr : int
        Samplerate of the audio

    Returns
    -------
    nsamples : int
        Number of waveform samples
    """
    return int(nsec * sr)

def nsamples_to_nsec(nsamples, sr):
    """
    Convert number of waveform samples to number of seconds

    Parameters
    ----------
    nsamples : int
        Number of waveform samples
    sr : int
        Samplerate of the audio

    Returns
    -------
    nsec : float
        Number of seconds
    """
    return nsamples / sr

def nms_to_nsamples(nms, sr):
    """
    Convert number of milliseconds to number of waveform samples

    Parameters
    ----------
    nms : float
        Number of milliseconds
    sr : int
        Samplerate of the audio

    Returns
    -------
    nsamples : int
        Number of waveform samples
    """
    return nsec_to_nsamples(nms/1000, sr=sr)

def nsamples_to_nms(nsamples, sr):
    """
    Convert number of waveform samples to number of milliseconds

    Parameters
    ----------
    nsamples : int
        Number of waveform samples
    sr : int
        Samplerate of the audio

    Returns
    -------
    nms : float
        Number of milliseconds
    """
    return 1000 * nsamples_to_nsec(nsamples, sr=sr)

def nsec_to_nframes(nsec, frame_length_nms):
    """
    Convert number of seconds to number of frames

    Assume frames are non-overlapping

    Parameters
    ----------
    nsec : float
        Number of seconds
    frame_length_nms : float
        Duration of one frame in milliseconds

    Returns
    -------
    nframes : int
        Number of frames
    """
    return nms_to_nframes(nsec*1000, frame_length_nms=frame_length_nms)

def nframes_to_nsec(nframes, frame_length_nms):
    """
    Convert number of frames to number of seconds

    Assume frames are non-overlapping

    Parameters
    ----------
    nframes : int
        Number of frames
    frame_length_nms : float
        Duration of one frame in milliseconds

    Returns
    -------
    nsec : float
        Number of seconds
    """
    return nframes_to_nms(nframes, frame_length_nms=frame_length_nms) / 1000

def nms_to_nframes(nms, frame_length_nms):
    """
    Convert number of milliseconds to number of frames

    Assume frames are non-overlapping

    Parameters
    ----------
    nms : float
        Number of milliseconds
    frame_length_nms : float
        Duration of one frame in milliseconds

    Returns
    -------
    nframes : int
        Number of frames
    """
    return int(nms / frame_length_nms)

def nframes_to_nms(frames, frame_length_nms):
    """
    Convert number of milliseconds to number of frames

    Assume frames are non-overlapping
    
    Parameters
    ----------
    nframes : int
        Number of frames
    frame_length_nms : float
        Duration of one frame in milliseconds

    Returns
    -------
    nms : float
        Number of milliseconds
    """
    return frames * frame_length_nms

def get_nsamples_nms(sr, nsamples=None, nms=None):
    """
    Express a property in both number of samples and number of milliseconds depending on which of those two is known and which is unknown.

    At least one of `nsamples` and `nms` must be submitted. If both are submitted, they must match.

    Parameters
    ----------
    nsamples : int
        Number of frames
    nms : float
        Duration of one frame in milliseconds
    sr: int
        Samplerate of the audio

    Returns
    -------
    nsamples : int
        Number of milliseconds
    nms : float
        Number of milliseconds
    """

    if (nsamples is not None) and (nms is not None):
        nsamples_ = nms_to_nsamples(nms, sr=sr)
        nms_ = nsamples_to_nms(nsamples, sr=sr)
        assert nsamples == nsamples_, f"got conflicting values for nsamples: {nsamples} and {nsamples_}"
        assert nms == nms_,         f"got conflicting values for nms_: {nms} and {nms_}"
    elif (nsamples is None) and (nms is not None):
        nsamples = nms_to_nsamples(nms, sr=sr)
    elif (nsamples is not None) and (nms is None):
        nms = nsamples_to_nms(nsamples, sr=sr)
    else:
        raise Exception("Expected at least one nsamples and nms to not be None, but both are None")

    return nsamples, nms
