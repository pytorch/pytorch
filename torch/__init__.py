from torch._C import *
import sys
import math

_tensor_classes = set()
_storage_classes = set()

################################################################################
# Define basic utilities
################################################################################

def _import_dotted_name(name):
    components = name.split('.')
    obj = __import__(components[0])
    for component in components[1:]:
        obj = getattr(obj, component)
    return obj

# range gets shadowed by torch.range
def _pyrange(*args, **kwargs):
    return __builtins__['range'](*args, **kwargs)

def typename(o):
    return o.__module__ + "." + o.__class__.__name__

def isTensor(obj):
    return obj.__class__ in _tensor_classes

def isStorage(obj):
    return obj.__class__ in _storage_classes

def isLongStorage(obj):
    return isinstance(obj, LongStorage)

def setDefaultTensorType(t):
    global Tensor
    global Storage
    global _defaultTensorTypeName
    _defaultTensorTypeName = t
    Tensor = _import_dotted_name(t)
    Storage = _import_dotted_name(t.replace('Tensor', 'Storage'))

def getDefaultTensorType():
    return _defaultTensorTypeName

from .Storage import _StorageBase
from .Tensor import _TensorBase

################################################################################
# Define Storage and Tensor classes
################################################################################

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
    pass
class FloatTensor(_C.FloatTensorBase, _TensorBase):
    pass
class LongTensor(_C.LongTensorBase, _TensorBase):
    pass
class IntTensor(_C.IntTensorBase, _TensorBase):
    pass
class ShortTensor(_C.ShortTensorBase, _TensorBase):
    pass
class CharTensor(_C.CharTensorBase, _TensorBase):
    pass
class ByteTensor(_C.ByteTensorBase, _TensorBase):
    pass

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

# This shadows Torch.py and Storage.py
setDefaultTensorType('torch.DoubleTensor')

################################################################################
# Initialize extension
################################################################################

_C._initExtension()

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
