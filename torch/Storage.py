import torch
from ._utils import _type


class _StorageBase(object):
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

    def __copy__(self):
        return self.clone()

    def __deepcopy__(self, memo):
        memo = memo.setdefault('torch', {})
        if self._cdata in memo:
            return memo[self._cdata]
        new_storage = self.clone()
        memo[self._cdata] = new_storage
        return new_storage

    def __reduce__(self):
        return type(self), (self.tolist(),)

    def clone(self):
        return type(self)(self.size()).copy_(self)

    def tolist(self):
        return [v for v in self]

    def double(self):
        return self.type(type(self).__module__ + '.DoubleStorage')

    def float(self):
        return self.type(type(self).__module__ + '.FloatStorage')

    def long(self):
        return self.type(type(self).__module__ + '.LongStorage')

    def int(self):
        return self.type(type(self).__module__ + '.IntStorage')

    def short(self):
        return self.type(type(self).__module__ + '.ShortStorage')

    def char(self):
        return self.type(type(self).__module__ + '.CharStorage')

    def byte(self):
        return self.type(type(self).__module__ + '.ByteStorage')


_StorageBase.type = _type
