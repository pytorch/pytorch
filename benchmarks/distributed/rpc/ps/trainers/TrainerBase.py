import functools
from abc import ABC, abstractmethod

from metrics.MetricsLogger import MetricsLogger


class TrainerBase(ABC):

    BATCH_LEVEL_METRIC = "batch_level_metric"
    BATCH_ALL = "batch_all"
    FORWARD_METRIC = "foward_metric"
    FORWARD_PASS = "forward_pass"
    BACKWARD_METRIC = "backward_metric"
    BACKWARD = "backward"

    def __init__(self, rank, metric_class="cuda", overwrite_metrics=False):
        self.__metrics_logger = MetricsLogger(rank, metric_class, overwrite_metrics)

    @abstractmethod
    def train(self):
        return

    def record_start(self, metric_type, key, metric_name):
        self.__metrics_logger.add_metric(
            metric_type,
            key,
            metric_name
        )

    def record_end(self, metric_type, key):
        self.__metrics_logger.add_metric_end(
            metric_type,
            key
        )

    def record_batch_start(self, key):
        self.__metrics_logger.add_metric(
            self.BATCH_LEVEL_METRIC,
            key,
            self.BATCH_ALL
        )

    def record_batch_end(self, key):
        self.__metrics_logger.add_metric_end(
            self.BATCH_LEVEL_METRIC,
            key
        )

    def record_forward_start(self, key):
        self.__metrics_logger.add_metric(
            self.FORWARD_METRIC,
            key,
            self.FORWARD_PASS
        )

    def record_forward_end(self, key):
        self.__metrics_logger.add_metric_end(
            self.FORWARD_METRIC,
            key
        )

    def record_backward_start(self, key):
        self.__metrics_logger.add_metric(
            self.BACKWARD_METRIC,
            key,
            self.BACKWARD
        )

    def record_backward_end(self, key):
        self.__metrics_logger.add_metric_end(
            self.BACKWARD_METRIC,
            key
        )

    @staticmethod
    def methodmetric(metric_name, metric_type="method_metric"):
        def decorator(function):
            @functools.wraps(function)
            def wrapper(self, *args):
                key = time.time()
                self.__metrics_logger.add_metric(metric_type, key, metric_name)
                result = function(self, *args)
                self.__metrics_logger.add_metric_end(metric_type, key)
                return result
            return wrapper
        return decorator

    def get_metrics(self):
        return self.__metrics_logger.get_processed_metrics()

    def clear_metrics(self):
        return self.__metrics_logger.clear_metrics()
