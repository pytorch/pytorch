import torch

__all__ = ["get_cpu_capability", ]


def get_cpu_capability() -> str:
    r"""Returns cpu capability as a string value.

    Possible values:
    - "DEFAULT"
    - "VSX"
    - "Z VECTOR"
    - "NO AVX"
    - "AVX2"
    - "AVX512"
    """
    return torch._C._get_cpu_capability()
