import torch

def get_device_obj(device):
    """
    Takes as input either a string or torch.device and always returns a torch.device
    """
    if isinstance(device, torch.device):
        return device
    elif isinstance(device, str):
        return torch.device(device)
    else:
        raise Exception(f"Got unexpected type {type(device)}")

def cut_from_middle(x, n):
    n_to_remove = x.size(-1) - n
    n_to_remove_left, n_to_remove_right = int(np.floor(n_to_remove/2)), int(np.ceil(n_to_remove/2))
    if n_to_remove_right > 0:
        x = x[..., n_to_remove_left : -n_to_remove_right]
    return x
