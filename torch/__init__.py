import sys
import math
from ._utils import _import_dotted_name

################################################################################
# Load the extension module
################################################################################

# Loading the extension with RTLD_GLOBAL option allows to not link extension
# modules against the _C shared object. Their missing THP symbols will be
# automatically filled by the dynamic loader.
import DLFCN
old_flags = sys.getdlopenflags()
sys.setdlopenflags(DLFCN.RTLD_GLOBAL | DLFCN.RTLD_NOW)
from torch._C import *
sys.setdlopenflags(old_flags)
del DLFCN
del old_flags

################################################################################
# Define basic utilities
################################################################################

def typename(o):
    return o.__module__ + "." + o.__class__.__name__


def is_tensor(obj):
    return obj.__class__ in _tensor_classes


def is_storage(obj):
    return obj.__class__ in _storage_classes


def set_default_tensor_type(t):
    global Tensor
    global Storage
    Tensor = _import_dotted_name(t)
    Storage = _import_dotted_name(t.replace('Tensor', 'Storage'))
    _C._set_default_tensor_type(Tensor)


from .serialization import save, load

################################################################################
# Define Storage and Tensor classes
################################################################################

from .storage import _StorageBase
from .tensor import _TensorBase

class DoubleStorage(_C.DoubleStorageBase, _StorageBase):
    pass
class FloatStorage(_C.FloatStorageBase, _StorageBase):
    pass
class LongStorage(_C.LongStorageBase, _StorageBase):
    pass
class IntStorage(_C.IntStorageBase, _StorageBase):
    pass
class ShortStorage(_C.ShortStorageBase, _StorageBase):
    pass
class CharStorage(_C.CharStorageBase, _StorageBase):
    pass
class ByteStorage(_C.ByteStorageBase, _StorageBase):
    pass

class DoubleTensor(_C.DoubleTensorBase, _TensorBase):
    def is_signed(self):
        return True
class FloatTensor(_C.FloatTensorBase, _TensorBase):
    def is_signed(self):
        return True
class LongTensor(_C.LongTensorBase, _TensorBase):
    def is_signed(self):
        return True
class IntTensor(_C.IntTensorBase, _TensorBase):
    def is_signed(self):
        return True
class ShortTensor(_C.ShortTensorBase, _TensorBase):
    def is_signed(self):
        return True
class CharTensor(_C.CharTensorBase, _TensorBase):
    def is_signed(self):
        # TODO
        return False
class ByteTensor(_C.ByteTensorBase, _TensorBase):
    def is_signed(self):
        return False


_tensor_classes = set()
_storage_classes = set()


_storage_classes.add(DoubleStorage)
_storage_classes.add(FloatStorage)
_storage_classes.add(LongStorage)
_storage_classes.add(IntStorage)
_storage_classes.add(ShortStorage)
_storage_classes.add(CharStorage)
_storage_classes.add(ByteStorage)

_tensor_classes.add(DoubleTensor)
_tensor_classes.add(FloatTensor)
_tensor_classes.add(LongTensor)
_tensor_classes.add(IntTensor)
_tensor_classes.add(ShortTensor)
_tensor_classes.add(CharTensor)
_tensor_classes.add(ByteTensor)


set_default_tensor_type('torch.FloatTensor')

################################################################################
# Initialize extension
################################################################################

# Shared memory manager needs to know the exact location of manager executable
import os
manager_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'lib', 'torch_shm_manager')
if sys.version_info[0] >= 3:
    manager_path = bytes(manager_path, 'ascii')

_C._initExtension(manager_path)

del os
del manager_path

################################################################################
# Remove unnecessary members
################################################################################

del DoubleStorageBase
del FloatStorageBase
del LongStorageBase
del IntStorageBase
del ShortStorageBase
del CharStorageBase
del ByteStorageBase
del DoubleTensorBase
del FloatTensorBase
del LongTensorBase
del IntTensorBase
del ShortTensorBase
del CharTensorBase
del ByteTensorBase

