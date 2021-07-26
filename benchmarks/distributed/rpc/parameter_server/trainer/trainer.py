import functools
import time
from abc import ABC, abstractmethod

from metrics.MetricsLogger import MetricsLogger

import torch


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


class DdpTrainer(TrainerBase):

    def __init__(
        self,
        process_group,
        use_cuda_rpc,
        server_rref,
        backend,
        epochs,
        preprocess_data,
        create_criterion,
        create_ddp_model,
        hook_state_class,
        hook,
        iteration_step
    ):
        r"""
        A trainer that implements a DDP training algorithm using a simple hook that performs allreduce
        using the process_group implementation.
        Args:
            process_group (ProcessGroup): distributed process group
            use_cuda_rpc (bool): indicator for CUDA RPC
            server_rref (RRef): remote reference to the server
            backend (str): distributed communication backend
            epochs (int): epoch count for training
            preprocess_data (function): preprocesses data passed
                to the trainer before starting training
            create_criterion (function): creates a criterion to calculate loss
            create_ddp_model (function): creates a ddp model for the trainer
            hook_state_class (class): class that will be used to keep tracking of state
                during training.
            hook (function): ddp communication hook
            iteration_step (function): will perform 1 step of training
        """
        super().__init__(process_group.rank())
        self.process_group = process_group
        self.use_cuda_rpc = use_cuda_rpc
        self.server_rref = server_rref
        self.backend = backend
        self.epochs = epochs
        self.preprocess_data = preprocess_data
        self.create_criterion = create_criterion
        self.create_ddp_model = create_ddp_model
        self.hook_state_class = hook_state_class
        self.hook = hook
        self.iteration_step = iteration_step

        self.rank = process_group.rank()
        self.trainer_count = process_group.size()

    def epoch_key(self, epoch, index):
        r"""
        A method that returns an encoded key that represents the current epoch and
        iteration index.
        Args:
            epoch (int): epoch index
            index (int): iteration index
        """
        return f"{epoch},{index}"

    def train(self, model, data):
        r"""
        A method that implements the training algorithm.
        Args:
            model (nn.Module): neural network model
            data (list): training examples
        """
        model = model.cuda(self.rank)
        data = self.preprocess_data(self.rank, data)
        criterion = self.create_criterion(self.rank)
        ddp_model, hook_state = self.create_ddp_model(
            self, self.rank, model, self.process_group, self.hook_state_class, self.hook
        )
        optimizer = torch.optim.SGD(ddp_model.parameters(), 1e-4)

        for epoch in range(self.epochs):
            if epoch % 5 == 0 and self.rank == 0:
                print(f"train epoch={epoch}")
            for index, batch in enumerate(data):
                self.iteration_step(
                    self, ddp_model, criterion, optimizer, hook_state, epoch, index, batch
                )
        torch.cuda.synchronize(self.rank)
