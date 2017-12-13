import torch


class no_grad(object):
    """Context-manager that disabled gradient calculation.

    Disabling gradient calculation is useful for inference, when you are sure
    that you will not call :meth:`Variable.backward()`. It will reduce memory
    consumption for computations that would otherwise have `requires_grad=True`.
    In this mode, the result of every computation will have
    `requires_grad=False`, even when the inputs have `requires_grad=True`.
    """

    def __init__(self):
        self.prev = torch.is_grad_enabled()

    def __enter__(self):
        torch.set_grad_enabled(False)

    def __exit__(self, *args):
        torch.set_grad_enabled(self.prev)
        return False
