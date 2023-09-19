import torch

from abc import ABC, abstractmethod
from typing import Any


class StreamBase(ABC):
    r"""Base stream class abstraction for multi backends Stream to herit from"""
    @abstractmethod
    def wait_event(self, event):
        raise NotImplementedError()

    @abstractmethod
    def wait_stream(self, stream):
        raise NotImplementedError()
    
    @abstractmethod
    def record_event(self, event=None):
        raise NotImplementedError()

    @abstractmethod
    def query(self):
        raise NotImplementedError()
    
    @abstractmethod
    def synchronize(self):
        raise NotImplementedError()
