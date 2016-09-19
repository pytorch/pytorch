from . import device, _dummy_ctx
from ..storage import _StorageBase


class _CudaStorageBase(_StorageBase):

    def type(self, *args, **kwargs):
        source_device = self.getDevice()
        ctx = device(source_device) if source_device != -1 else _dummy_ctx()
        with ctx:
            return super(_CudaStorageBase, self).type(*args, **kwargs)

