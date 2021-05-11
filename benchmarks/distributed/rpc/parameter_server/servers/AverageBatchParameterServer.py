import threading

from .AverageParameterServer import AverageParameterServer
from .ParameterServerBase import ParameterServerBase


class AverageBatchParameterServer(AverageParameterServer):

    master_lock = threading.Lock()
    locks = []

    def __init__(
        self,
        rank,
        trainer_count,
        lc,
        use_cuda_rpc
    ):
        super().__init__(rank, trainer_count, lc, use_cuda_rpc)

    def process_gradient(self, gradient, param_loc):
        if param_loc not in self.gradient_dict:
            self.record_straggler_start(self.param_key(param_loc))
            self.record_batch_start(self.param_key(param_loc))
            self.gradient_dict[param_loc] = []
        self.gradient_dict[param_loc].append(gradient)

    @ParameterServerBase.record_method(name="average computation")
    def average(self, param_loc):
        param_loc_avg = self.gradient_dict[param_loc][0]
        for gradient in self.gradient_dict[param_loc][1:]:
            param_loc_avg += gradient
        param_loc_avg / (1.0 * self.trainer_count)
        return param_loc_avg
