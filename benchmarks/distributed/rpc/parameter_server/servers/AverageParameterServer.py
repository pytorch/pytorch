import threading

import torch
import torch.distributed.rpc as rpc

from .AverageParameterServerBase import AverageParameterServerBase
from .ParameterServerBase import ParameterServerBase


class AverageParameterServer(AverageParameterServerBase):

    lock = threading.Lock()

    def __init__(
        self,
        rank,
        trainer_count,
        use_cuda_rpc
    ):
        super().__init__(rank)

        self.rank = rank
        self.trainer_count = trainer_count
        self.use_cuda_rpc = use_cuda_rpc

        # server state
        self.batch_number = 0
        self.futures = {}
        self.gradient_dict = {}

    def param_key(self, param_loc):
        return f"{self.batch_number},{param_loc}"

    def process_gradient(self, gradient, param_loc):
        if param_loc not in self.gradient_dict:
            self.record_straggler_start(self.param_key(param_loc))
            self.record_batch_start(self.param_key(param_loc))
            self.gradient_dict[param_loc] = gradient
        else:
            self.gradient_dict[param_loc] += gradient

    @ParameterServerBase.record_method(name="average computation")
    def average(self, param_loc):
        param_loc_avg = self.gradient_dict[param_loc]
        param_loc_avg / (1.0 * self.trainer_count)
        return param_loc_avg

    def clear_batch_state(self):
        self.futures.clear()
        self.gradient_dict.clear()

    @staticmethod
    def reset_state(ps_rref):
        self = ps_rref.local_value()
        self.batch_number = 0
        self.futures.clear()
        self.gradient_dict.clear()
        self.clear_metrics()

    @staticmethod
    def get_metrics_rpc(ps_rref):
        self = ps_rref.local_value()
        return self.get_metrics()

    @staticmethod
    @rpc.functions.async_execution
    def average_gradient(
        ps_rref,
        received_batch_number,
        param_loc,
        gradient
    ):
        self = ps_rref.local_value()
        if type(gradient) is list:
            gradient = self.sparse_rpc_format_to_tensor(gradient)
        if not self.use_cuda_rpc:
            gradient = gradient.cuda(self.rank)
        fut = torch.futures.Future()
        with AverageParameterServer.lock:
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
                    param_loc_avg = self.sparse_tensor_to_rpc_format(param_loc_avg)
                for cur_fut in self.futures[param_loc]:
                    cur_fut.set_result(param_loc_avg)
                self.record_batch_end(self.param_key(param_loc))
        return fut
