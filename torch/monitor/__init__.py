from torch._C._monitor import *  # noqa: F403

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


STAT_EVENT = "torch.monitor.Stat"


class TensorboardEventHandler:
    """
    TensorboardEventHandler is an event handler that will write known events to
    the provided SummaryWriter.

    This currently only supports ``torch.monitor.Stat`` events which are logged
    as scalars.

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_MONITOR)
        >>> # xdoctest: +REQUIRES(module:tensorboard)
        >>> from torch.utils.tensorboard import SummaryWriter
        >>> from torch.monitor import TensorboardEventHandler, register_event_handler
        >>> writer = SummaryWriter("log_dir")
        >>> register_event_handler(TensorboardEventHandler(writer))
    """
    def __init__(self, writer: "SummaryWriter") -> None:
        """
        Constructs the ``TensorboardEventHandler``.
        """
        self._writer = writer

    def __call__(self, event: Event) -> None:
        if event.name == STAT_EVENT:
            for k, v in event.data.items():
                self._writer.add_scalar(k, v, walltime=event.timestamp.timestamp())
