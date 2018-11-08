from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from abc import abstractmethod


class Meter(object):

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def Reset(self):
        pass

    @abstractmethod
    def Add(self):
        pass

    @abstractmethod
    def Compute(self):
        pass
