def nsec_to_nsamples(sec, sr):
    """number of seconds to number of waveform samples"""
    return int(sec * sr)

def nsamples_to_nsec(samples, sr):
    """number of waveform samples to number of seconds"""
    return samples / sr

def nms_to_nsamples(ms, sr):
    """number of milliseconds to number of waveform samples"""
    return nsec_to_nsamples(ms/1000, sr=sr)

def nsamples_to_nms(samples, sr):
    """number of waveform samples to number of milliseconds"""
    return 1000 * nsamples_to_nsec(samples, sr=sr)

def nsec_to_nframes(sec, frame_length_nms):
    """number of seconds to number of frames"""
    return nms_to_nframes(sec*1000, frame_length_nms=frame_length_nms)

def nframes_to_nsec(frames, frame_length_nms):
    """number of frames to number of seconds"""
    return nframes_to_nms(frames, frame_length_nms=frame_length_nms) / 1000

def nms_to_nframes(ms, frame_length_nms):
    """number of milliseconds to number of frames"""
    return int(ms / frame_length_nms)

def nframes_to_nms(frames, frame_length_nms):
    """number of frames to number of milliseconds"""
    return frames * frame_length_nms
