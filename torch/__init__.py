import sys
import math
from ._utils import _import_dotted_name

################################################################################
# Load the extension module
################################################################################

# Loading the extension with RTLD_GLOBAL option allows to not link extension
# modules against the _C shared object. Their missing THP symbols will be
# automatically filled by the dynamic loader.
import os as _dl_flags

# first check if the os package has the required flags
if not hasattr(_dl_flags, 'RTLD_GLOBAL') or not hasattr(_dl_flags, 'RTLD_NOW'):
    try:
        # next try if DLFCN exists
        import DLFCN as _dl_flags
    except ImportError:
        # as a last attempt, use compile-time constants
        import torch._dl as _dl_flags

old_flags = sys.getdlopenflags()
sys.setdlopenflags(_dl_flags.RTLD_GLOBAL | _dl_flags.RTLD_NOW)
from torch._C import *
sys.setdlopenflags(old_flags)
del _dl_flags
del old_flags

################################################################################
# Define basic utilities
################################################################################

def typename(o):
    module = ''
    class_name = ''
    if hasattr(o, '__module__') and o.__module__ != 'builtins' \
            and o.__module__ != '__builtin__' and o.__module__ is not None:
        module = o.__module__ + '.'

    if hasattr(o, '__qualname__'):
        class_name = o.__qualname__
    elif hasattr(o, '__name__'):
        class_name = o.__name__
    else:
        class_name = o.__class__.__name__

    return module + class_name


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


def set_rng_state(new_state):
    default_generator.set_state(new_state)


def get_rng_state():
    return default_generator.get_state()


def manual_seed(seed):
    return default_generator.manual_seed(seed)


def initial_seed():
    return default_generator.initial_seed()


from .serialization import save, load
from ._tensor_str import set_printoptions

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
    @classmethod
    def storage_type(cls):
        return DoubleStorage
class FloatTensor(_C.FloatTensorBase, _TensorBase):
    def is_signed(self):
        return True
    @classmethod
    def storage_type(cls):
        return FloatStorage
class LongTensor(_C.LongTensorBase, _TensorBase):
    def is_signed(self):
        return True
    @classmethod
    def storage_type(cls):
        return LongStorage
class IntTensor(_C.IntTensorBase, _TensorBase):
    def is_signed(self):
        return True
    @classmethod
    def storage_type(cls):
        return IntStorage
class ShortTensor(_C.ShortTensorBase, _TensorBase):
    def is_signed(self):
        return True
    @classmethod
    def storage_type(cls):
        return ShortStorage
class CharTensor(_C.CharTensorBase, _TensorBase):
    def is_signed(self):
        # TODO
        return False
    @classmethod
    def storage_type(cls):
        return CharStorage
class ByteTensor(_C.ByteTensorBase, _TensorBase):
    def is_signed(self):
        return False
    @classmethod
    def storage_type(cls):
        return ByteStorage


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
