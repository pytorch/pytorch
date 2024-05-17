from abc import ABC, abstractmethod


class MetricBase(ABC):
    def __init__(self, name):
        self.name = name
        self.start = None
        self.end = None

    @abstractmethod
    def record_start(self):
        return

    @abstractmethod
    def record_end(self):
        return

    @abstractmethod
    def elapsed_time(self):
        return

    def get_name(self):
        return self.name

    def get_end(self):
        return self.end
