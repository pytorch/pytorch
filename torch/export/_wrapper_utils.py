import torch


class _WrapperModule(torch.nn.Module):
    def __init__(self, f):  # type: ignore[no-untyped-def]
        super().__init__()
        self.f = f

    def forward(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return self.f(*args, **kwargs)
