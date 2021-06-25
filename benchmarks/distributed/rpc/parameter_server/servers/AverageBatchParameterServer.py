from .AverageParameterServer import AverageParameterServer
from .ParameterServerBase import ParameterServerBase


class AverageBatchParameterServer(AverageParameterServer):

    def __init__(
        self,
        rank,
        trainer_count,
        use_cuda_rpc
    ):
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
