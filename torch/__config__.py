import torch


def show():
    """
    Return a human-readable string with descriptions of the
    configuration of PyTorch.
    """
    return torch._C._show_config()

# TODO: In principle, we could provide more structured version/config
# information here.  We're not for now; considering doing so if someone
# asks for it.
