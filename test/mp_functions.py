import torch

class SimpleCustomBatch(object):
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = torch.stack(transposed_data[1], 0)

    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self

    def is_pinned(self):
        return self.inp.is_pinned() and self.tgt.is_pinned()
        

