import torch

import pybrate


def test_stft_istft():
    
    win_length = 1024
    hop_length = 256
    
    my_audio = torch.randn(1, hop_length*100)

    spect = pybrate.stft(my_audio, win_length, hop_length)
    audio_recon = pybrate.istft(spect, hop_length)

    assert torch.allclose(my_audio, audio_recon, atol=1e-6)
