import torch

from .MetricBase import MetricBase


class CUDAMetric(MetricBase):
    def __init__(self, name: str):
        self.name = name
        self.start = None
        self.end = None

    def record_start(self, rank):
        self.start = torch.cuda.Event(enable_timing=True)
        with torch.cuda.device(rank):
            self.start.record()

    def record_end(self, rank):
        self.end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.device(rank):
            self.end.record()

    def elapsed_time(self, rank):
        self.start.synchronize()
        self.end.synchronize()
        return self.start.elapsed_time(self.end)
