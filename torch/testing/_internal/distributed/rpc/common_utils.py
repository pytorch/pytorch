import torch


def _get_device_name(idx):
    if torch.accelerator.current_accelerator() is None:
        return "cpu"
    if idx < 0:
        return f"{torch.accelerator.current_accelerator().type}"
    return f"{torch.accelerator.current_accelerator().type}:{idx}"
