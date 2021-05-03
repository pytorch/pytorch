from abc import abstractmethod

from .TrainerBase import TrainerBase


class DdpTrainerBase(TrainerBase):

    HOOK_FUTURE_METRIC = "hook_future_metric"
    NCCL_ALLREDUCE = "nccl_allreduce"
    GLOO_ALLREDUCE = "gloo_allreduce"

    def __init__(self, rank, metric_class="cuda", overwrite_metrics=False):
        super().__init__(rank, metric_class, overwrite_metrics)

    @staticmethod
    @abstractmethod
    def hook(state, bucket):
        return

    def record_hook_fut_start(self, key, metric_name):
        self.record_start(self.HOOK_FUTURE_METRIC, key, metric_name)

    def record_hook_fut_end(self, key):
        self.record_end(self.HOOK_FUTURE_METRIC, key)

    def bucket_to_parameters(self, bucket):
        parameter_tensors = bucket.get_per_parameter_tensors()
        parameter_tensors_count = len(parameter_tensors)
        if parameter_tensors_count > 0:
            return parameter_tensors
        else:
            return [bucket.get_tensor()]
