from . import device, _dummy_ctx
from ..tensor import _TensorBase


class _CudaTensorBase(_TensorBase):

    def type(self, *args, **kwargs):
        source_device = self.getDevice()
        ctx = device(source_device) if source_device != -1 else _dummy_ctx()
        with ctx:
            return super(_CudaTensorBase, self).type(*args, **kwargs)

