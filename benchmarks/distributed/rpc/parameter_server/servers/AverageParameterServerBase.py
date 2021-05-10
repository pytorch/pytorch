from abc import abstractmethod

from .ParameterServerBase import ParameterServerBase


class AverageParameterServerBase(ParameterServerBase):

    def __init__(self, rank):
        super().__init__(rank)

    @staticmethod
    @abstractmethod
    def average_gradient():
        return
