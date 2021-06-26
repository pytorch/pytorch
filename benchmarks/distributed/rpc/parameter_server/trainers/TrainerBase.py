import functools
import time
from abc import ABC, abstractmethod

from metrics.MetricsLogger import MetricsLogger


class TrainerBase(ABC):

    BATCH_LEVEL_METRIC = "batch_level_metric"
    BATCH_ALL = "batch_all"
    FORWARD_METRIC = "forward_metric"
    FORWARD_PASS = "forward_pass"
    BACKWARD_METRIC = "backward_metric"
    BACKWARD = "backward"

    def __init__(self, rank):
        r"""
        Inits TrainerBase class.
        Args:
            rank (int): worker rank
        """
        self.__metrics_logger = MetricsLogger(rank)

    @abstractmethod
    def train(self):
        r"""
        A method to be implemented by child class that will train a neural network.
        """
        return

    def record_start(self, type, key, name, cuda=True):
        r"""
        A method that records the start event for a metric.
        Args:
            type (str): group id for metric
            key (str): unique id for metric within a group
            name (str): description of the metric
            cuda (bool): indicator to determine if this is a CUDA metric
        """
        self.__metrics_logger.record_start(
            type,
            key,
            name,
            cuda
        )

    def record_end(self, type, key):
        r"""
        A method that records the end event for a metric.
        Args:
            type (str): group id for metric
            key (str): unique id for metric within a group
        """
        self.__metrics_logger.record_end(
            type,
            key
        )

    def record_batch_start(self, key, cuda=True):
        r"""
        A helper method that records a batch metric for the
        given key. A user should call this at the start of an
        iteration step during training.
        Args:
            key (str): unique id for metric within a group
            cuda (bool): indicator to determine if this is a CUDA metric
        """
        self.__metrics_logger.record_start(
            self.BATCH_LEVEL_METRIC,
            key,
            self.BATCH_ALL,
            cuda
        )

    def record_batch_end(self, key):
        r"""
        A helper method that records a batch metric for the
        given key. A user should call this at the end of an
        iteration step during training.
        Args:
            key (str): unique id for metric within a group
        """
        self.__metrics_logger.record_end(
            self.BATCH_LEVEL_METRIC,
            key
        )

    def record_forward_start(self, key, cuda=True):
        r"""
        A helper method that records a forward metric
        for the given key. A user should call this before
        their neural network forward.
        Args:
            key (str): unique id for metric within a group
            cuda (bool): indicator to determine if this is a CUDA metric
        """
        self.__metrics_logger.record_start(
            self.FORWARD_METRIC,
            key,
            self.FORWARD_PASS,
            cuda
        )

    def record_forward_end(self, key):
        r"""
        A helper method that records a forward metric
        for the given key. A user should call this after their
        neural network forward.
        Args:
            key (str): unique id for metric within a group
        """
        self.__metrics_logger.record_end(
            self.FORWARD_METRIC,
            key
        )

    def record_backward_start(self, key, cuda=True):
        r"""
        A helper method that records a backward metric
        for the given key. A user should call this before
        their .backward() call.
        Args:
            key (str): unique id for metric within a group
            cuda (bool): indicator to determine if this is a CUDA metric
        """
        self.__metrics_logger.record_start(
            self.BACKWARD_METRIC,
            key,
            self.BACKWARD,
            cuda
        )

    def record_backward_end(self, key):
        r"""
        A helper method that records a backward metric
        for the given key. A user should call this after
        .backward().
        Args:
            key (str): unique id for metric within a group
        """
        self.__metrics_logger.record_end(
            self.BACKWARD_METRIC,
            key
        )

    @staticmethod
    def methodmetric(name, type="method_metric", cuda=True):
        r"""
        A decorator that records a metric for the decorated method.
        Args:
            name (str): description of the metric
            type (str): group id for metric
            cuda (bool): indicator to determine if this is a CUDA metric
        """
        def decorator(function):
            @functools.wraps(function)
            def wrapper(self, *args):
                key = time.time()
                self.__metrics_logger.record_start(type, key, name, cuda)
                result = function(self, *args)
                self.__metrics_logger.record_end(type, key)
                return result
            return wrapper
        return decorator

    def get_metrics(self):
        r"""
        A method that returns metrics captured by the __metrics_logger.
        """
        return self.__metrics_logger.get_processed_metrics()

    def clear_metrics(self):
        r"""
        A method that clears __metrics_logger recorded metrics.
        """
        return self.__metrics_logger.clear_metrics()
