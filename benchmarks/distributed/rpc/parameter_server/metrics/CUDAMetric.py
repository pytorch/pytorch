import torch

from .MetricBase import MetricBase


class CUDAMetric(MetricBase):
    def __init__(self, rank: int, name: str):
        self.rank = rank
        self.name = name
        self.start = None
        self.end = None

    def record_start(self):
        self.start = torch.cuda.Event(enable_timing=True)
        with torch.cuda.device(self.rank):
            self.start.record()

    def record_end(self):
        self.end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.device(self.rank):
            self.end.record()

    def elapsed_time(self):
        if not self.start.query():
            raise RuntimeError("start event did not complete")
        if not self.end.query():
            raise RuntimeError("end event did not complete")
        return self.start.elapsed_time(self.end)

    def synchronize(self):
        self.start.synchronize()
        self.end.synchronize()
