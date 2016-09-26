from . import device, _dummy_ctx
from ..storage import _StorageBase


class _CudaStorageBase(_StorageBase):
    is_cuda = True

    def type(self, *args, **kwargs):
        source_device = self.get_device()
        ctx = device(source_device) if source_device != -1 else _dummy_ctx()
        with ctx:
            return super(_CudaStorageBase, self).type(*args, **kwargs)

