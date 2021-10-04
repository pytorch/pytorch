import time

from .MetricBase import MetricBase


class CPUMetric(MetricBase):
    def __init__(self, name: str):
        self.name = name
        self.start = None
        self.end = None

    def record_start(self):
        self.start = time.time()

    def record_end(self):
        self.end = time.time()

    def elapsed_time(self):
        if self.start is None:
            raise RuntimeError("start is None")
        if self.end is None:
            raise RuntimeError("end is None")
        return self.end - self.start
