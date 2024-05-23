import functools
import threading
import time
from abc import ABC, abstractmethod

import torch
import torch.distributed.rpc as rpc

from metrics.MetricsLogger import MetricsLogger
from utils import sparse_rpc_format_to_tensor, sparse_tensor_to_rpc_format


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
        self.__metrics_logger.record_start(type, key, name, cuda)

    def record_end(self, type, key):
        r"""
        A method that records the end event for a metric
        Args:
            type (str): group id for metric
            key (str): unique id for metric within a group
        """
        self.__metrics_logger.record_end(type, key)

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
            cuda,
        )

    def record_straggler_end(self, key):
        r"""
        A helper method that records a straggler metric
        for the given key. A user should call this when
        the last gradient for the param location is received.
        Args:
            key (str): unique id for metric within a group
        """
        self.__metrics_logger.record_end(self.PARAMETER_SERVER_STRAGGLER_METRIC, key)

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
            self.PARAMETER_SERVER_BATCH_METRIC, key, self.PARAM_INDEX_BATCH, cuda
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
        self.__metrics_logger.record_end(self.PARAMETER_SERVER_BATCH_METRIC, key)

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


class AverageParameterServer(ParameterServerBase):
    def __init__(self, rank, trainer_count, use_cuda_rpc):
        r"""
        A parameter server that averages the gradients
        from trainers for each training iteration step.
        Gradients are added as they are received from trainers.
        When all gradients have been received, the sum is
        divided by the number of trainers.
        Args:
            rank (int): worker rank
            trainer_count (int): count of trainers sending
                gradients to the server
            use_cuda_rpc (bool): indicator for CUDA RPC
        """
        super().__init__(rank)

        self.lock = threading.Lock()
        self.rank = rank
        self.trainer_count = trainer_count
        self.use_cuda_rpc = use_cuda_rpc

        self.batch_number = 0
        self.futures = {}
        self.gradient_dict = {}

    @staticmethod
    def reset_state(server_rref):
        r"""
        A method that clears the state of the server.
        Args:
            server_rref (RRef): remote reference to the server
        """
        self = server_rref.local_value()
        self.batch_number = 0
        self.futures.clear()
        self.gradient_dict.clear()
        self.clear_metrics()

    def param_key(self, param_loc):
        r"""
        A method that returns an encoded key that represents
        the current batch and param location.
        Args:
            param_loc (int): bucket location sent by the trainer
                containing the gradient
        """
        return f"{self.batch_number},{param_loc}"

    def clear_batch_state(self):
        r"""
        Clears the current server batch state.
        """
        self.futures.clear()
        self.gradient_dict.clear()

    def process_gradient(self, gradient, param_loc):
        r"""
        Stores the gradient if param_loc is not in gradient_dict.
        Adds the gradient to param_loc if it is in gradient_dict.
        Args:
            gradient (torch.Tensor): tensor sent from trainer
            param_loc (int): bucket location sent by the trainer
                containing the gradient
        """
        if param_loc not in self.gradient_dict:
            self.record_straggler_start(self.param_key(param_loc))
            self.record_batch_start(self.param_key(param_loc))
            self.gradient_dict[param_loc] = gradient
        else:
            self.gradient_dict[param_loc] += gradient

    @ParameterServerBase.record_method(name="average computation")
    def average(self, param_loc):
        r"""
        Obtains the tensor at the param_loc in the gradient_dict
        and then divides by number of trainers.
        Args:
            param_loc (int): bucket location sent by the trainer
                containing the gradient
        """
        param_loc_avg = self.gradient_dict[param_loc]
        param_loc_avg / (1.0 * self.trainer_count)
        return param_loc_avg

    @staticmethod
    @rpc.functions.async_execution
    def average_gradient(server_rref, received_batch_number, param_loc, gradient):
        r"""
        An asynchronous function that will average gradients
        sent from trainers.
        Args:
            server_rref (RRef): remote reference to the server
            received_batch_number (int): batch number sent by
                the trainer
            param_loc (int): bucket location sent by the trainer
                containing the gradient
            gradient (torch.Tensor or list): tensor sent by the trainer
        """
        self = server_rref.local_value()
        if type(gradient) is list:
            gradient = sparse_rpc_format_to_tensor(gradient)
        gradient = gradient.cuda(self.rank)
        fut = torch.futures.Future()
        with self.lock:
            if self.batch_number < received_batch_number:
                self.batch_number = received_batch_number
                self.clear_batch_state()
            self.process_gradient(gradient, param_loc)
            if param_loc not in self.futures:
                self.futures[param_loc] = []
            self.futures[param_loc].append(fut)
            if len(self.futures[param_loc]) == self.trainer_count:
                self.record_straggler_end(self.param_key(param_loc))
                param_loc_avg = self.average(param_loc)
                if not self.use_cuda_rpc:
                    param_loc_avg = param_loc_avg.cpu()
                if param_loc_avg.is_sparse:
                    param_loc_avg = sparse_tensor_to_rpc_format(param_loc_avg)
                for cur_fut in self.futures[param_loc]:
                    cur_fut.set_result(param_loc_avg)
                self.record_batch_end(self.param_key(param_loc))
        return fut


class AverageBatchParameterServer(AverageParameterServer):
    def __init__(self, rank, trainer_count, use_cuda_rpc):
        r"""
        A parameter server that averages the gradients
        from trainers for each training iteration step.
        Gradients are stored and averaged when a gradient
        has been received from each trainer for a param
        location.
        Args:
            rank (int): worker rank
            trainer_count (int): count of trainers sending
                gradients to the server
            use_cuda_rpc (bool): indicator for CUDA RPC
        """
        super().__init__(rank, trainer_count, use_cuda_rpc)

    def process_gradient(self, gradient, param_loc):
        r"""
        Adds the gradient to param_loc bucket stored in
        the gradient_dict.
        Args:
            gradient (torch.Tensor): tensor sent from trainer
            param_loc (int): bucket location sent by the trainer
                containing the gradient
        """
        if param_loc not in self.gradient_dict:
            self.record_straggler_start(self.param_key(param_loc))
            self.record_batch_start(self.param_key(param_loc))
            self.gradient_dict[param_loc] = []
        self.gradient_dict[param_loc].append(gradient)

    @ParameterServerBase.record_method(name="average computation")
    def average(self, param_loc):
        r"""
        Sums the gradients at the param_loc then divides by the
        number of trainers.
        Args:
            param_loc (int): bucket location sent by the trainer
                containing the gradient
        """
        param_loc_avg = self.gradient_dict[param_loc][0]
        for gradient in self.gradient_dict[param_loc][1:]:
            param_loc_avg += gradient
        param_loc_avg / (1.0 * self.trainer_count)
        return param_loc_avg
