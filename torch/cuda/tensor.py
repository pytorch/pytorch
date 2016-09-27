from . import device, _dummy_ctx
from ..tensor import _TensorBase


class _CudaTensorBase(_TensorBase):
    is_cuda = True

    def type(self, *args, **kwargs):
        source_device = self.get_device()
        ctx = device(source_device) if source_device != -1 else _dummy_ctx()
        with ctx:
            return super(_CudaTensorBase, self).type(*args, **kwargs)

    def new(self, *args, **kwargs):
        source_device = self.get_device()
        ctx = device(source_device) if source_device != -1 else _dummy_ctx()
        with ctx:
            return super(_CudaTensorBase, self).new(*args, **kwargs)

