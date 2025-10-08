import torch


_GreenContext = object
SUPPORTED = False

if hasattr(torch._C, "GreenContext"):
    _GreenContext = torch._C.GreenContext  # type: ignore[misc]
    SUPPORTED = True


# Python shim helps Sphinx process docstrings more reliably.
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
    def make_current(self) -> None:
        r"""Make the green context the current context."""
        return super().make_current()  # type: ignore[misc]

    def pop_current(self) -> None:
        r"""Assuming the green context is the current context, pop it from the
        context stack and restore the previous context.
        """
        return super().pop_current()  # type: ignore[misc]
