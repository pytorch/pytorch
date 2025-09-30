import torch


GreenContext_ = object
SUPPORTED = False

if hasattr(torch._C, "GreenContext"):
    GreenContext_ = torch._C.GreenContext
    SUPPORTED = True

# Python shim helps Sphinx process docstrings more reliably.
class GreenContext(GreenContext_):
    r"""Wrapper around a CUDA green context.

     .. warning::
        This API is in beta and may change in future releases.
    """

    def create(num_sms: int, device_id: int = 0) -> GreenContext_:
        r"""Create a CUDA green context.

        Arguments:
            num_sms (int): The number of SMs to use in the green context.
            device_id (int, optional): The device index of green context.
        """
        if not SUPPORTED:
            raise RuntimeError("PyTorch was not built with Green Context support!")
        return GreenContext_.create(num_sms, device_id)

    # Note that these functions are bypassed by we define them here
    # for Sphinx documentation purposes
    def make_current(self) -> None:
        r"""Make the green context the current context.
        """
        return super().make_current()

    def pop_current(self) -> None:
        r"""Assuming the green context is the current context, pop it from the
            context stack and restore the previous context.
        """
        return super().pop_current()
