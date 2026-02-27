import torch


_GreenContext = object
SUPPORTED = False

if hasattr(torch._C, "_CUDAGreenContext"):
    _GreenContext = torch._C._CUDAGreenContext  # type: ignore[misc]
    SUPPORTED = True


# Python shim helps Sphinx process docstrings more reliably.
# pyrefly: ignore [invalid-inheritance]
class GreenContext(_GreenContext):
    r"""Wrapper around a CUDA green context.

    .. warning::
       This API is in beta and may change in future releases.
    """

    @staticmethod
    def create(num_sms: int, device_id: int = 0) -> _GreenContext:
        r"""Create a CUDA green context.

        Arguments:
            num_sms (int): The number of SMs to use in the green context.
            device_id (int, optional): The device index of green context.
        """
        if not SUPPORTED:
            raise RuntimeError("PyTorch was not built with Green Context support!")
        return _GreenContext.create(num_sms, device_id)  # type: ignore[attr-defined]

    # Note that these functions are bypassed by we define them here
    # for Sphinx documentation purposes
    def set_context(self) -> None:  # pylint: disable=useless-parent-delegation
        r"""Make the green context the current context."""
        return super().set_context()  # type: ignore[misc]

    def pop_context(self) -> None:  # pylint: disable=useless-parent-delegation
        r"""Assuming the green context is the current context, pop it from the
        context stack and restore the previous context.
        """
        return super().pop_context()  # type: ignore[misc]

    def Stream(self) -> torch.Stream:
        r"""Return the CUDA Stream used by the green context."""
        return super().Stream()
