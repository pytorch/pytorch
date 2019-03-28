import torch


def detailed_version():
    """
    Return a human-readable string with descriptions of versions of all
    dependencies that ATen was built with.
    """
    return torch._C._detailed_version()
