import torch


class _StorageBase():
    def __str__(self):
        content = ' ' + '\n '.join(str(self[i]) for i in torch._pyrange(len(self)))
        return content + '\n[{} of size {}]'.format(torch.typename(self), len(self))

    def __repr__(self):
        return str(self)

    def __iter__(self):
        return iter(map(lambda i: self[i], torch._pyrange(self.size())))

    def copy_(self, other):
        torch._C._storageCopy(self, other)
        return self

    def tolist(self):
        return [v for v in self]
