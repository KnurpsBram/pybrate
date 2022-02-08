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
