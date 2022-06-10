




import caffe2.contrib.playground.meter as Meter
from caffe2.python import workspace


class ComputeLoss(Meter.Meter):
    def __init__(self, opts=None, blob_name=''):
        self.blob_name = blob_name
        self.opts = opts
        self.iter = 0
        self.value = 0

    def Reset(self):
        self.iter = 0
        self.value = 0

    def Add(self):
        """Average values of a blob on each gpu"""
        value = 0
        for idx in range(self.opts['distributed']['first_xpu_id'],
                         self.opts['distributed']['first_xpu_id'] +
                         self.opts['distributed']['num_xpus']):
            value += workspace.FetchBlob('{}_{}/{}'.
                format(self.opts['distributed']['device'], idx, self.blob_name))
        self.value += value
        self.iter += 1

    def Compute(self):
        result = self.opts['distributed']['num_shards'] * self.value / self.iter
        self.Reset()
        return result
