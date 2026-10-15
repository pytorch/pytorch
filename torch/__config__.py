import torch


def show() -> str:
    """
    Return a human-readable string with descriptions of the
    configuration of PyTorch.
    """
    return torch._C._show_config()


# TODO: In principle, we could provide more structured version/config
# information here. For now only CXX_FLAGS, which Timer uses, and the
# compiler are exposed.
def _cxx_flags() -> str:
    """Returns the CXX_FLAGS used when building PyTorch."""
    return torch._C._cxx_flags()


def _cxx_compiler() -> tuple[str, str]:
    """Returns the compiler that built PyTorch, as (gcc, clang or msvc, its
    version as show() prints it)."""
    return torch._C._cxx_compiler()


def parallel_info() -> str:
    r"""Returns detailed string with parallelization settings"""
    return torch._C._parallel_info()
