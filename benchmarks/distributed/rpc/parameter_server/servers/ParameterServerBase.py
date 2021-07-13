import functools
import time
from abc import ABC, abstractmethod

from metrics.MetricsLogger import MetricsLogger


class ParameterServerBase(ABC):

    PARAMETER_SERVER_BATCH_METRIC = "parameter_server_batch_metric"
    PARAMETER_SERVER_STRAGGLER_METRIC = "parameter_server_straggler_metric"
    PARAM_INDEX_STRAGGLER = "param_index_straggler"
    PARAM_INDEX_BATCH = "param_index_batch"

    def __init__(self, rank):
        r"""
        Inits ParameterServerBase class.
        Args:
            rank (int): worker rank
        """
        self.__metrics_logger = MetricsLogger(rank)

    @abstractmethod
    def process_gradient(self):
        r"""
        A method to be implemented by child class that will process a
        gradient received by a server.
        """
        return

    @staticmethod
    @abstractmethod
    def average_gradient():
        r"""
        A method to be implemented by child class that will average
        gradients.
        """
        return

    @staticmethod
    @abstractmethod
    def reset_state():
        r"""
        A method to be implemented by child class that will reset
        the server state.
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
        A method that records the end event for a metric
        Args:
            type (str): group id for metric
            key (str): unique id for metric within a group
        """
        self.__metrics_logger.record_end(
            type,
            key
        )

    def record_straggler_start(self, key, cuda=True):
        r"""
        A helper method that records a straggler metric
        for the given key. A user should call this when
        the first gradient for the param location is received.
        Args:
            key (str): unique id for metric within a group
            cuda (bool): indicator to determine if this is a CUDA metric
        """
        self.__metrics_logger.record_start(
            self.PARAMETER_SERVER_STRAGGLER_METRIC,
            key,
            self.PARAM_INDEX_STRAGGLER,
            cuda
        )

    def record_straggler_end(self, key):
        r"""
        A helper method that records a straggler metric
        for the given key. A user should call this when
        the last gradient for the param location is received.
        Args:
            key (str): unique id for metric within a group
        """
        self.__metrics_logger.record_end(
            self.PARAMETER_SERVER_STRAGGLER_METRIC,
            key
        )

    def record_batch_start(self, key, cuda=True):
        r"""
        A helper method that records a batch metric
        for the given key. A user should call this when
        the first gradient for the param location is received.
        Args:
            key (str): unique id for metric within a group
            cuda (bool): indicator to determine if this is a CUDA metric
        """
        self.__metrics_logger.record_start(
            self.PARAMETER_SERVER_BATCH_METRIC,
            key,
            self.PARAM_INDEX_BATCH,
            cuda
        )

    def record_batch_end(self, key):
        r"""
        A helper method that records a batch metric
        for the given key. A user should call this when
        all futures for a param location have had their
        result set.
        Args:
            key (str): unique id for metric within a group
        """
        self.__metrics_logger.record_end(
            self.PARAMETER_SERVER_BATCH_METRIC,
            key
        )

    @staticmethod
    def record_method(name, type="method_metric", cuda=True):
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

    @staticmethod
    def get_metrics(server_rref):
        r"""
        A staticmethod that returns metrics captured by the __metrics_logger.
        Args:
            server_rref (RRef): remote reference to the server
        """
        self = server_rref.local_value()
        return self.__metrics_logger.get_processed_metrics()

    def clear_metrics(self):
        r"""
        A method that clears __metrics_logger recorded metrics.
        """
        return self.__metrics_logger.clear_metrics()
