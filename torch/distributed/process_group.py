_GLOO_AVAILABLE = True
_NCCL_AVAILABLE = True
_MPI_AVAILABLE = True

try:
    from . import ProcessGroupGloo  # noqa: F401
except ImportError:
    _GLOO_AVAILABLE = False

try:
    from . import ProcessGroupNCCL  # noqa: F401
except ImportError:
    _NCCL_AVAILABLE = False

try:
    from . import ProcessGroupMPI  # noqa: F401
except ImportError:
    _MPI_AVAILABLE = False


def is_gloo_available():
    """
    Returns if the Gloo backend is available.

    """
    return _GLOO_AVAILABLE


def is_nccl_available():
    """
    Returns if the NCCL backend is available.

    """
    return _NCCL_AVAILABLE


def is_mpi_available():
    """
    Returns if the MPI backend is available.

    """
    return _MPI_AVAILABLE


__all__ = [
    'is_gloo_available',
    'is_nccl_available',
    'is_mpi_available',
]

if is_gloo_available():
    __all__.append('ProcessGroupGloo')

if is_nccl_available():
    __all__.append('ProcessGroupNCCL')

if is_mpi_available():
    __all__.append('ProcessGroupMPI')
