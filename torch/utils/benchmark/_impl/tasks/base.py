import abc

from torch.utils.benchmark._impl.workers import base


class TaskBase(abc.ABC):

    @abc.abstractproperty
    def worker(self) -> base.WorkerBase:
        ...
