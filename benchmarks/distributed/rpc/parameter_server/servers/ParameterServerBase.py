import functools
import time
from abc import ABC, abstractmethod

import torch
from metrics.MetricsLogger import MetricsLogger


class ParameterServerBase(ABC):

    PARAMETER_SERVER_BATCH_METRIC = "parameter_server_batch_metric"
    PARAMETER_SERVER_STRAGGLER_METRIC = "parameter_server_straggler_metric"
    BP_LOC_STRAGGLER = "bp_location_staggler"
    BP_LOC_BATCH = "bp_location_batch"

    def __init__(self, rank):
        self.__metrics_logger = MetricsLogger(rank)

    @abstractmethod
    def process_gradient(self):
        return

    @staticmethod
    @abstractmethod
    def reset_state():
        return

    @staticmethod
    @abstractmethod
    def get_metrics_rpc():
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

    def record_straggler_start(self, key, cuda=True):
        self.__metrics_logger.record_start(
            self.PARAMETER_SERVER_STRAGGLER_METRIC,
            key,
            self.BP_LOC_STRAGGLER,
            cuda
        )

    def record_straggler_end(self, key):
        self.__metrics_logger.record_end(
            self.PARAMETER_SERVER_STRAGGLER_METRIC,
            key
        )

    def record_batch_start(self, key, cuda=True):
        self.__metrics_logger.record_start(
            self.PARAMETER_SERVER_BATCH_METRIC,
            key,
            self.BP_LOC_BATCH,
            cuda
        )

    def record_batch_end(self, key):
        self.__metrics_logger.record_end(
            self.PARAMETER_SERVER_BATCH_METRIC,
            key
        )

    @staticmethod
    def record_method(name, type="method_metric", cuda=True):
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

    def sparse_tensor_to_rpc_format(self, sparse_tensor):
        sparse_tensor = sparse_tensor.coalesce()
        return [sparse_tensor.indices(), sparse_tensor.values(), torch.tensor(sparse_tensor.size())]

    def sparse_rpc_format_to_tensor(self, sparse_rpc_format):
        return torch.sparse_coo_tensor(
            sparse_rpc_format[0], sparse_rpc_format[1], torch.Size(sparse_rpc_format[2])
        ).coalesce()
