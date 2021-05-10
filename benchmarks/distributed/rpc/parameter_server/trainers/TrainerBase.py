import functools
import time
from abc import ABC, abstractmethod

from metrics.MetricsLogger import MetricsLogger


class TrainerBase(ABC):

    BATCH_LEVEL_METRIC = "batch_level_metric"
    BATCH_ALL = "batch_all"
    FORWARD_METRIC = "foward_metric"
    FORWARD_PASS = "forward_pass"
    BACKWARD_METRIC = "backward_metric"
    BACKWARD = "backward"

    def __init__(self, rank):
        self.__metrics_logger = MetricsLogger(rank)

    @abstractmethod
    def train(self):
        return

    def record_start(self, type, key, name, cuda=True):
        self.__metrics_logger.record_start(
            type,
            key,
            name,
            cuda
        )

    def record_end(self, type, key):
        self.__metrics_logger.record_end(
            type,
            key
        )

    def record_batch_start(self, key, cuda=True):
        self.__metrics_logger.record_start(
            self.BATCH_LEVEL_METRIC,
            key,
            self.BATCH_ALL,
            cuda
        )

    def record_batch_end(self, key):
        self.__metrics_logger.record_end(
            self.BATCH_LEVEL_METRIC,
            key
        )

    def record_forward_start(self, key, cuda=True):
        self.__metrics_logger.record_start(
            self.FORWARD_METRIC,
            key,
            self.FORWARD_PASS,
            cuda
        )

    def record_forward_end(self, key):
        self.__metrics_logger.record_end(
            self.FORWARD_METRIC,
            key
        )

    def record_backward_start(self, key, cuda=True):
        self.__metrics_logger.record_start(
            self.BACKWARD_METRIC,
            key,
            self.BACKWARD,
            cuda
        )

    def record_backward_end(self, key):
        self.__metrics_logger.record_end(
            self.BACKWARD_METRIC,
            key
        )

    @staticmethod
    def methodmetric(name, type="method_metric", cuda=True):
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
        return self.__metrics_logger.get_processed_metrics()

    def clear_metrics(self):
        return self.__metrics_logger.clear_metrics()
