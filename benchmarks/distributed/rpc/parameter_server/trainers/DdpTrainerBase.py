from abc import abstractmethod

from .TrainerBase import TrainerBase


class DdpTrainerBase(TrainerBase):

    HOOK_FUTURE_METRIC = "hook_future_metric"

    def __init__(self, rank):
        r"""
        Inits DdpTrainerBase class.
        Args:
            rank (int): worker rank
        """
        super().__init__(rank)

    @staticmethod
    @abstractmethod
    def hook(state, bucket):
        r"""
        A method to be implemented by child class that will implement a DDP
        training algorithm.
        Args:
            state (object): maintains state during the training process
            bucket (GradBucket): gradient bucket
        """
        return

    def record_hook_fut_start(self, key, name, cuda=True):
        r"""
        A helper method that records a hook future metric
        for the given key. A user should call this before
        sending async request in the DDP communication hook.
        Args:
            key (str): unique id for metric within a group
            cuda (bool): indicator to determine if this is a CUDA metric
        """
        self.record_start(self.HOOK_FUTURE_METRIC, key, name, cuda)

    def record_hook_fut_end(self, key):
        r"""
        A helper method that records a hook future metric
        for the given key. A user should call this in a callback
        attached to the future returned by an async request.
        Args:
            key (str): unique id for metric within a group
        """
        self.record_end(self.HOOK_FUTURE_METRIC, key)
