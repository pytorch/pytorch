from typing_extensions import deprecated

import torch


# Preserved only for BC reasons
@deprecated(
    "`torch._streambase._StreamBase` is deprecated. Please use `torch.Stream` instead.",
    category=FutureWarning,
)
class _StreamBase(torch.Stream):
    pass


@deprecated(
    "`torch._streambase._EventBase` is deprecated. Please use `torch.Event` instead.",
    category=FutureWarning,
)
class _EventBase(torch.Event):
    pass
