""" Trivial logger which repeatedly logs the same value """
import numpy as np
from .plugin import Plugin


class ConstantMonitor(Plugin):
    def __init__(self, data, stat_name='constant'):
        super(ConstantMonitor, self).__init__([(1, 'iteration'), (1, 'epoch')])
        self.stat_name = stat_name
        self.data = data

    def register(self, trainer):
        self.trainer = trainer
        stats = self.trainer.stats.setdefault(self.stat_name, {})
        stats['data'] = self.data

    def iteration(self, iteration, input, *args):
        pass

    def epoch(self, *args):
        pass


