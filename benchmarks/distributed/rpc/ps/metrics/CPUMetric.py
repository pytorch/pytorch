import time

from .MetricBase import MetricBase


class CPUMetric(MetricBase):
    def __init__(self, name: str):
        self.name = name
        self.start = None
        self.end = None

    def record_start(self, rank):
        self.start = time.time()

    def record_end(self, rank):
        self.end = time.time()

    def elapsed_time(self):
        return self.end - self.start
