import torch

def default_window(win_length):
    return torch.hann_window(win_length+2)[1:-1]

