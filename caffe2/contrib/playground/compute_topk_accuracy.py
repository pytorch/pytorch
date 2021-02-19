




import caffe2.contrib.playground.meter as Meter
from caffe2.python import workspace
import numpy as np


class ComputeTopKAccuracy(Meter.Meter):
    # Python default arguments are evaluated once when the function is
    # defined, not each time the function is called
    # This means that if you use a mutable default argument and mutate it,
    # you will and have mutated that object for
    # all future calls to the function as well.
    # def __init__(self, blob_name=['softmax', 'label'], opts=None, topk=1):
    def __init__(self, blob_name=None, opts=None, topk=1):
        if blob_name is None:
            blob_name = ['softmax', 'label']
        self.blob_name = blob_name
        self.opts = opts
        self.topk = topk
        self.iter = 0
        self.value = 0

    def Reset(self):
        self.iter = 0
        self.value = 0

    def Add(self):
        for idx in range(self.opts['distributed']['first_xpu_id'],
                         self.opts['distributed']['first_xpu_id'] +
                         self.opts['distributed']['num_xpus']):
            prefix = '{}_{}/'.format(self.opts['distributed']['device'], idx)
            softmax = workspace.FetchBlob(prefix + self.blob_name[0])
            labels = workspace.FetchBlob(prefix + self.blob_name[1])
            output = np.squeeze(softmax)
            target = np.squeeze(labels)
            if len(output.shape) == 1:
                output = output.reshape((1, output.shape[0]))
            else:
                assert len(output.shape) == 2, \
                    'wrong output size (1D or 2D expected)'
            assert len(target.shape) == 1, 'wrong target size (1D expected)'
            assert output.shape[0] == target.shape[0], \
                'target and output do not match'

            N = output.shape[0]
            pred = np.argsort(-output, axis=1)[:, :self.topk]
            correct = pred.astype(target.dtype) == np.repeat(
                target.reshape((N, 1)), [self.topk], axis=1)
            self.value += np.sum(correct[:, :self.topk])
            self.iter += N

    def Compute(self):
        result = self.value / self.iter
        self.Reset()
        return result
