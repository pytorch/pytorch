import threading

import torch
import torch.distributed.rpc as rpc
from utils import sparse_rpc_format_to_tensor, sparse_tensor_to_rpc_format

from .ParameterServerBase import ParameterServerBase


class AverageParameterServer(ParameterServerBase):

    lock = threading.Lock()

    def __init__(
        self,
        rank,
        trainer_count,
        use_cuda_rpc
    ):
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
    def average_gradient(
        server_rref,
        received_batch_number,
        param_loc,
        gradient
    ):
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
