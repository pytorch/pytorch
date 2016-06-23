import torch


class _StorageBase():
    def __str__(self):
        content = ' ' + '\n '.join(str(self[i]) for i in torch._pyrange(len(self)))
        return content + '\n[torch.RealStorage of size {}]'.format(len(self))

    def __repr__(self):
        return str(self)

    def __iter__(self):
        return iter(map(lambda i: self[i], torch._pyrange(self.size())))

    def tolist(self):
        return [v for v in self]
