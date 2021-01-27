
import torch
import torch.numpy.random


def wrap_value(v):
    if isinstance(v, torch.Tensor):
        return TensorWrapper(v)
    if type(v)==type and isinstance(object, v):
        return None
    else:
         return v
def unwrap_value(v):
    if isinstance(v, TensorWrapper):
        return v.tensor
    if type(v)==type and isinstance(object, v):
        return None
    else:
        return v


def wrap_args(*args):
    return [wrap_value(v) for v in args]
def wrap_kwargs(**kwargs):
    return dict([(k, wrap_value(kwargs[k])) for k in kwargs])

def unwrap_args(*args):
    return [unwrap_value(v) for v in args]
def unwrap_kwargs(**kwargs):
    result = dict([(k, unwrap_value(kwargs[k])) for k in kwargs])
    return result

#import numpy as orig_np
class TensorWrapper:
    tensor=0

    def __init__(self, t):
        self.tensor=t

    @property
    def dtype(self):
        return self.tensor.dtype

    @property
    def ndim(self):
        return self.tensor.ndim

    def __abs__(*args, **kwargs):
        return wrap_value(args[0].tensor.__abs__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __add__(*args, **kwargs):
        return wrap_value(args[0].tensor.__add__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __and__(*args, **kwargs):
        return wrap_value(args[0].tensor.__and__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __array__(*args, **kwargs):
        return wrap_value(args[0].tensor.__array__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __array_finalize__(*args, **kwargs):
        return wrap_value(args[0].tensor.__array_finalize__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    #def __array_interface__(*args, **kwargs):
    #    return wrap_value(args[0].tensor.__array_interface__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __array_prepare__(*args, **kwargs):
        return wrap_value(args[0].tensor.__array_prepare__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __array_priority__(*args, **kwargs):
        return wrap_value(args[0].tensor.__array_priority__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    #def __array_struct__(*args, **kwargs):
    #    return wrap_value(args[0].tensor.__array_struct__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __array_ufunc__(*args, **kwargs):
        return wrap_value(args[0].tensor.__array_ufunc__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __array_wrap__(*args, **kwargs):
        return wrap_value(args[0].tensor.__array_wrap__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __bool__(*args, **kwargs):
        return wrap_value(args[0].tensor.__bool__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))

    @property
    def __class__(*args, **kwargs):
        return TensorWrapper
        #return wrap_value(args[0].tensor.__class__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __complex__(*args, **kwargs):
        return wrap_value(args[0].tensor.__complex__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __contains__(*args, **kwargs):
        return wrap_value(args[0].tensor.__contains__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __copy__(*args, **kwargs):
        return wrap_value(args[0].tensor.__copy__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __deepcopy__(*args, **kwargs):
        return wrap_value(args[0].tensor.__deepcopy__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __delattr__(*args, **kwargs):
        return wrap_value(args[0].tensor.__delattr__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))

    def __delitem__(*args, **kwargs):
        return wrap_value(args[0].tensor.__delitem__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))

    def __dir__(*args, **kwargs):
        return wrap_value(args[0].tensor.__dir__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __divmod__(*args, **kwargs):
        return wrap_value(args[0].tensor.__divmod__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __doc__(*args, **kwargs):
        return wrap_value(args[0].tensor.__doc__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __eq__(*args, **kwargs):
        return wrap_value(args[0].tensor.__eq__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __float__(*args, **kwargs):
        return wrap_value(args[0].tensor.__float__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __floordiv__(*args, **kwargs):
        return wrap_value(args[0].tensor.__floordiv__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __format__(*args, **kwargs):
        return wrap_value(args[0].tensor.__format__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __ge__(*args, **kwargs):
        return wrap_value(args[0].tensor.__ge__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
#    def __getattribute__(*args, **kwargs):
#        return wrap_value(args[0].tensor.__getattribute__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))


    def __getitem__(*args, **kwargs):
        return wrap_value(args[0].tensor.__getitem__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))


    def __gt__(*args, **kwargs):
        return wrap_value(args[0].tensor.__gt__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __hash__(self):
        return self.tensor.__hash__()
    def __iadd__(*args, **kwargs):
        return wrap_value(args[0].tensor.__iadd__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __iand__(*args, **kwargs):
        return wrap_value(args[0].tensor.__iand__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __ifloordiv__(*args, **kwargs):
        return wrap_value(args[0].tensor.__ifloordiv__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __ilshift__(*args, **kwargs):
        return wrap_value(args[0].tensor.__ilshift__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __imatmul__(*args, **kwargs):
        return wrap_value(args[0].tensor.__imatmul__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __imod__(*args, **kwargs):
        return wrap_value(args[0].tensor.__imod__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __imul__(*args, **kwargs):
        return wrap_value(args[0].tensor.__imul__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __index__(*args, **kwargs):
        return wrap_value(args[0].tensor.__index__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
#    def __init__(*args, **kwargs):
#        return wrap_value(args[0].tensor.__init__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
#    def __init_subclass__(*args, **kwargs):
#        return wrap_value(args[0].tensor.__init_subclass__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
#    def __int__(*args, **kwargs):
#        return wrap_value(args[0].tensor.__int__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __invert__(*args, **kwargs):
        return wrap_value(args[0].tensor.__invert__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __ior__(*args, **kwargs):
        return wrap_value(args[0].tensor.__ior__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __ipow__(*args, **kwargs):
        return wrap_value(args[0].tensor.__ipow__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __irshift__(*args, **kwargs):
        return wrap_value(args[0].tensor.__irshift__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __isub__(*args, **kwargs):
        return wrap_value(args[0].tensor.__isub__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __iter__(self):
        return self.tensor.__iter__()
    def __itruediv__(*args, **kwargs):
        return wrap_value(args[0].tensor.__itruediv__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __ixor__(*args, **kwargs):
        return wrap_value(args[0].tensor.__ixor__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __le__(*args, **kwargs):
        return wrap_value(args[0].tensor.__le__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __len__(*args, **kwargs):
        return wrap_value(args[0].tensor.__len__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __lshift__(*args, **kwargs):
        return wrap_value(args[0].tensor.__lshift__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __lt__(*args, **kwargs):
        return wrap_value(args[0].tensor.__lt__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __matmul__(*args, **kwargs):
        return wrap_value(args[0].tensor.__matmul__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __mod__(*args, **kwargs):
        return wrap_value(args[0].tensor.__mod__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __mul__(*args, **kwargs):
        return wrap_value(args[0].tensor.__mul__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __ne__(*args, **kwargs):
        return wrap_value(args[0].tensor.__ne__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __neg__(*args, **kwargs):
        return wrap_value(args[0].tensor.__neg__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))

    #def __new__(*args, **kwargs):
    #    return wrap_value(args[0].tensor.__new__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __or__(*args, **kwargs):
        return wrap_value(args[0].tensor.__or__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __pos__(*args, **kwargs):
        return wrap_value(args[0].tensor.__pos__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __pow__(*args, **kwargs):
        return wrap_value(args[0].tensor.__pow__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __radd__(*args, **kwargs):
        return wrap_value(args[0].tensor.__radd__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __rand__(*args, **kwargs):
        return wrap_value(args[0].tensor.__rand__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __rdivmod__(*args, **kwargs):
        return wrap_value(args[0].tensor.__rdivmod__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __reduce__(*args, **kwargs):
        return wrap_value(args[0].tensor.__reduce__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __reduce_ex__(*args, **kwargs):
        return wrap_value(args[0].tensor.__reduce_ex__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __repr__(self):
        return self.tensor.__repr__()
    def __rfloordiv__(*args, **kwargs):
        return wrap_value(args[0].tensor.__rfloordiv__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __rlshift__(*args, **kwargs):
        return wrap_value(args[0].tensor.__rlshift__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __rmatmul__(*args, **kwargs):
        return wrap_value(args[0].tensor.__rmatmul__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __rmod__(*args, **kwargs):
        return wrap_value(args[0].tensor.__rmod__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __rmul__(*args, **kwargs):
        return wrap_value(args[0].tensor.__rmul__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __ror__(*args, **kwargs):
        return wrap_value(args[0].tensor.__ror__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __rpow__(*args, **kwargs):
        return wrap_value(args[0].tensor.__rpow__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __rrshift__(*args, **kwargs):
        return wrap_value(args[0].tensor.__rrshift__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __rshift__(*args, **kwargs):
        return wrap_value(args[0].tensor.__rshift__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __rsub__(*args, **kwargs):
        return wrap_value(args[0].tensor.__rsub__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __rtruediv__(*args, **kwargs):
        return wrap_value(args[0].tensor.__rtruediv__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __rxor__(*args, **kwargs):
        return wrap_value(args[0].tensor.__rxor__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
#    def __setattr__(*args, **kwargs):
#        return wrap_value(args[0].tensor.__setattr__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))

    def __setitem__(*args, **kwargs):
        return wrap_value(args[0].tensor.__setitem__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))

#    def __setstate__(*args, **kwargs):
#        return wrap_value(args[0].tensor.__setstate__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __sizeof__(*args, **kwargs):
        return wrap_value(args[0].tensor.__sizeof__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __str__(*args, **kwargs):
        return wrap_value(args[0].tensor.__str__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __sub__(*args, **kwargs):
        return wrap_value(args[0].tensor.__sub__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __subclasshook__(*args, **kwargs):
        return wrap_value(args[0].tensor.__subclasshook__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __truediv__(*args, **kwargs):
        return wrap_value(args[0].tensor.__truediv__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def __xor__(*args, **kwargs):
        return wrap_value(args[0].tensor.__xor__(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def all(*args, **kwargs):
        return wrap_value(args[0].tensor.all(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def any(*args, **kwargs):
        return wrap_value(args[0].tensor.any(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def argmax(*args, **kwargs):
        return wrap_value(args[0].tensor.argmax(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def argmin(*args, **kwargs):
        return wrap_value(args[0].tensor.argmin(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def argpartition(*args, **kwargs):
        return wrap_value(args[0].tensor.argpartition(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def argsort(*args, **kwargs):
        return wrap_value(args[0].tensor.argsort(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def astype(*args, **kwargs):
        return wrap_value(args[0].tensor.astype(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def base(*args, **kwargs):
        return wrap_value(args[0].tensor.base(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def byteswap(*args, **kwargs):
        return wrap_value(args[0].tensor.byteswap(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def choose(*args, **kwargs):
        return wrap_value(args[0].tensor.choose(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def clip(*args, **kwargs):
        return wrap_value(args[0].tensor.clip(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def compress(*args, **kwargs):
        return wrap_value(args[0].tensor.compress(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def conj(*args, **kwargs):
        return wrap_value(args[0].tensor.conj(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def conjugate(*args, **kwargs):
        return wrap_value(args[0].tensor.conjugate(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def copy(*args, **kwargs):
        return wrap_value(args[0].tensor.copy(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def ctypes(*args, **kwargs):
        return wrap_value(args[0].tensor.ctypes(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def cumprod(*args, **kwargs):
        return wrap_value(args[0].tensor.cumprod(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def cumsum(*args, **kwargs):
        return wrap_value(args[0].tensor.cumsum(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def data(*args, **kwargs):
        return wrap_value(args[0].tensor.data(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def diagonal(*args, **kwargs):
        return wrap_value(args[0].tensor.diagonal(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def dot(*args, **kwargs):
        return dot(*args, **kwargs)
        #return wrap_value(args[0].tensor.dot(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))

    def dump(*args, **kwargs):
        return wrap_value(args[0].tensor.dump(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def dumps(*args, **kwargs):
        return wrap_value(args[0].tensor.dumps(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))

    def equal(*args, **kwargs):
        return wrap_value(args[0].tensor.equal(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))

    def fill(*args, **kwargs):
        return wrap_value(args[0].tensor.fill_(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def flags(*args, **kwargs):
        return wrap_value(args[0].tensor.flags(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def flat(*args, **kwargs):
        return wrap_value(args[0].tensor.flat(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def flatten(*args, **kwargs):
        return wrap_value(args[0].tensor.flatten(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def getfield(*args, **kwargs):
        return wrap_value(args[0].tensor.getfield(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def imag(*args, **kwargs):
        return wrap_value(args[0].tensor.imag(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def item(*args, **kwargs):
        return wrap_value(args[0].tensor.item(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def itemset(*args, **kwargs):
        return wrap_value(args[0].tensor.itemset(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def itemsize(*args, **kwargs):
        return wrap_value(args[0].tensor.itemsize(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def max(*args, **kwargs):
        return wrap_value(args[0].tensor.max(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def mean(*args, **kwargs):
        return wrap_value(args[0].tensor.mean(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def min(*args, **kwargs):
        return wrap_value(args[0].tensor.min(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def nbytes(*args, **kwargs):
        return wrap_value(args[0].tensor.nbytes(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    #def ndim(*args, **kwargs):
    #    return wrap_value(args[0].tensor.ndim(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def newbyteorder(*args, **kwargs):
        return wrap_value(args[0].tensor.newbyteorder(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def nonzero(*args, **kwargs):
        return wrap_value(args[0].tensor.nonzero(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def partition(*args, **kwargs):
        return wrap_value(args[0].tensor.partition(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def prod(*args, **kwargs):
        return wrap_value(args[0].tensor.prod(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def ptp(*args, **kwargs):
        return wrap_value(args[0].tensor.ptp(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def put(*args, **kwargs):
        return wrap_value(args[0].tensor.put(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def ravel(*args, **kwargs):
        return wrap_value(args[0].tensor.ravel(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def real(*args, **kwargs):
        return wrap_value(args[0].tensor.real(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def repeat(*args, **kwargs):
        return wrap_value(args[0].tensor.repeat(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def reshape(*args, **kwargs):
        return wrap_value(args[0].tensor.reshape(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def resize(*args, **kwargs):
        return wrap_value(args[0].tensor.resize(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def round(*args, **kwargs):
        return wrap_value(args[0].tensor.round(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def searchsorted(*args, **kwargs):
        return wrap_value(args[0].tensor.searchsorted(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def setfield(*args, **kwargs):
        return wrap_value(args[0].tensor.setfield(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def setflags(*args, **kwargs):
        return wrap_value(args[0].tensor.setflags(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def shape(*args, **kwargs):
        return wrap_value(args[0].tensor.shape(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def size(*args, **kwargs):
        return wrap_value(args[0].tensor.size(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def sort(*args, **kwargs):
        return wrap_value(args[0].tensor.sort(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def squeeze(*args, **kwargs):
        return wrap_value(args[0].tensor.squeeze(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def std(*args, **kwargs):
        return wrap_value(args[0].tensor.std(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def strides(*args, **kwargs):
        return wrap_value(args[0].tensor.strides(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def sum(*args, **kwargs):
        return wrap_value(args[0].tensor.sum(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def swapaxes(*args, **kwargs):
        return wrap_value(args[0].tensor.swapaxes(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def take(*args, **kwargs):
        return wrap_value(args[0].tensor.take(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def tobytes(*args, **kwargs):
        return wrap_value(args[0].tensor.tobytes(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def tofile(*args, **kwargs):
        return wrap_value(args[0].tensor.tofile(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def tolist(*args, **kwargs):
        return wrap_value(args[0].tensor.tolist(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def tostring(*args, **kwargs):
        return wrap_value(args[0].tensor.tostring(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def trace(*args, **kwargs):
        return wrap_value(args[0].tensor.trace(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def transpose(*args, **kwargs):
        if len(args)==1:
            args = [args[0], 0, 1]
        return wrap_value(args[0].tensor.transpose(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def var(*args, **kwargs):
        return wrap_value(args[0].tensor.var(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))
    def view(*args, **kwargs):
        return wrap_value(args[0].tensor.view(*unwrap_args(*args[1:]), **unwrap_kwargs(**kwargs)))

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        f = getattr(self, func)(*args, **kwargs)
        return f

def __all__(*args, **kwargs):
    return wrap_value(torch.__all__(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))

#def __builtins__(*args, **kwargs):
#    return wrap_value(torch.__builtins__((*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
#def __cached__(*args, **kwargs):
#    return wrap_value(torch.__cached__(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
#def __config__(*args, **kwargs):
#    return wrap_value(torch.__config__(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
#def __doc__(*args, **kwargs):
#    return wrap_value(torch.__doc__(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
#def __file__(*args, **kwargs):
#    return wrap_value(torch.__file__(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
#def __git_revision__(*args, **kwargs):
#    return wrap_value(torch.__git_revision__(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
#def __loader__(*args, **kwargs):
#    return wrap_value(torch.__loader__(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def __mkl_version__(*args, **kwargs):
    return wrap_value(torch.__mkl_version__(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
#def __name__(*args, **kwargs):
#    return wrap_value(torch.__name__(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
#def __package__(*args, **kwargs):
#    return wrap_value(torch.__package__(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
#def __path__(*args, **kwargs):
#    return wrap_value(torch.__path__(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
#def __spec__(*args, **kwargs):
#    return wrap_value(torch.__spec__(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
#def __version__(*args, **kwargs):
#    return wrap_value(torch.__version__(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
#def _distributor_init(*args, **kwargs):
#    return wrap_value(torch._distributor_init(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
#def _globals(*args, **kwargs):
#    return wrap_value(torch._globals(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
#def _import_tools(*args, **kwargs):
#    return wrap_value(torch._import_tools(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def _mat(*args, **kwargs):
    return wrap_value(torch._mat(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def _mklinit(*args, **kwargs):
    return wrap_value(torch._mklinit(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def abs(*args, **kwargs):
    return wrap_value(torch.abs(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def absolute(*args, **kwargs):
    return wrap_value(torch.absolute(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def absolute_import(*args, **kwargs):
    return wrap_value(torch.absolute_import(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def add(*args, **kwargs):
    return wrap_value(torch.add(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def add_docstring(*args, **kwargs):
    return wrap_value(torch.add_docstring(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def add_newdoc(*args, **kwargs):
    return wrap_value(torch.add_newdoc(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def add_newdoc_ufunc(*args, **kwargs):
    return wrap_value(torch.add_newdoc_ufunc(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def add_newdocs(*args, **kwargs):
    return wrap_value(torch.add_newdocs(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def alen(*args, **kwargs):
    return wrap_value(torch.alen(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def all(*args, **kwargs):
    return wrap_value(torch.all(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def allclose(*args, **kwargs):
    return wrap_value(torch.allclose(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def alltrue(*args, **kwargs):
    return wrap_value(torch.alltrue(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def amax(*args, **kwargs):
    return wrap_value(torch.amax(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def amin(*args, **kwargs):
    return wrap_value(torch.amin(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def angle(*args, **kwargs):
    return wrap_value(torch.angle(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def any(*args, **kwargs):
    return wrap_value(torch.any(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def append(*args, **kwargs):
    return wrap_value(torch.append(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def apply_along_axis(*args, **kwargs):
    return wrap_value(torch.apply_along_axis(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def apply_over_axes(*args, **kwargs):
    return wrap_value(torch.apply_over_axes(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def arange(*args, **kwargs):
    return wrap_value(torch.arange(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def arccos(*args, **kwargs):
    return wrap_value(torch.arccos(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def arccosh(*args, **kwargs):
    return wrap_value(torch.arccosh(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def arcsin(*args, **kwargs):
    return wrap_value(torch.arcsin(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def arcsinh(*args, **kwargs):
    return wrap_value(torch.arcsinh(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def arctan(*args, **kwargs):
    return wrap_value(torch.arctan(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def arctan2(*args, **kwargs):
    return wrap_value(torch.arctan2(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def arctanh(*args, **kwargs):
    return wrap_value(torch.arctanh(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def argmax(*args, **kwargs):
    return wrap_value(torch.argmax(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def argmin(*args, **kwargs):
    return wrap_value(torch.argmin(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def argpartition(*args, **kwargs):
    return wrap_value(torch.argpartition(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def argsort(*args, **kwargs):
    return wrap_value(torch.argsort(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def argwhere(*args, **kwargs):
    return wrap_value(torch.argwhere(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def around(*args, **kwargs):
    return wrap_value(torch.around(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))

def array(*args, **kwargs):
    ndmin = kwargs.pop('ndmin', None)
    result = wrap_value(torch.tensor(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
    if ndmin:
        while(result.ndim < ndmin):
            result.tensor = result.tensor.unsqueeze(0)
    return result

def array2string(*args, **kwargs):
    return wrap_value(torch.array2string(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def array_equal(*args, **kwargs):
    return wrap_value(torch.array_equal(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def array_equiv(*args, **kwargs):
    return wrap_value(torch.array_equiv(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def array_repr(*args, **kwargs):
    return wrap_value(torch.array_repr(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def array_split(*args, **kwargs):
    return wrap_value(torch.array_split(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def array_str(*args, **kwargs):
    return wrap_value(torch.array_str(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def asanyarray(*args, **kwargs):
    return wrap_value(torch.asanyarray(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def asarray(*args, **kwargs):
    return wrap_value(torch.asarray(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def asarray_chkfinite(*args, **kwargs):
    return wrap_value(torch.asarray_chkfinite(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def ascontiguousarray(*args, **kwargs):
    return wrap_value(torch.ascontiguousarray(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def asfarray(*args, **kwargs):
    return wrap_value(torch.asfarray(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def asfortranarray(*args, **kwargs):
    return wrap_value(torch.asfortranarray(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def asmatrix(*args, **kwargs):
    return wrap_value(torch.asmatrix(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def asscalar(*args, **kwargs):
    return wrap_value(torch.asscalar(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def atleast_1d(*args, **kwargs):
    return wrap_value(torch.atleast_1d(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def atleast_2d(*args, **kwargs):
    return wrap_value(torch.atleast_2d(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def atleast_3d(*args, **kwargs):
    return wrap_value(torch.atleast_3d(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def average(*args, **kwargs):
    return wrap_value(torch.average(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def bartlett(*args, **kwargs):
    return wrap_value(torch.bartlett(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def base_repr(*args, **kwargs):
    return wrap_value(torch.base_repr(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def binary_repr(*args, **kwargs):
    return wrap_value(torch.binary_repr(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def bincount(*args, **kwargs):
    return wrap_value(torch.bincount(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def bitwise_and(*args, **kwargs):
    return wrap_value(torch.bitwise_and(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def bitwise_not(*args, **kwargs):
    return wrap_value(torch.bitwise_not(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def bitwise_or(*args, **kwargs):
    return wrap_value(torch.bitwise_or(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def bitwise_xor(*args, **kwargs):
    return wrap_value(torch.bitwise_xor(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def blackman(*args, **kwargs):
    return wrap_value(torch.blackman(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def block(*args, **kwargs):
    return wrap_value(torch.block(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def bmat(*args, **kwargs):
    return wrap_value(torch.bmat(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
#def bool(*args, **kwargs):
#    return wrap_value(torch.bool(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
#def bool8(*args, **kwargs):
#    return wrap_value(torch.bool8(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
#def bool_(*args, **kwargs):
#    return wrap_value(torch.bool_(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def broadcast(*args, **kwargs):
    return wrap_value(torch.broadcast(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def broadcast_arrays(*args, **kwargs):
    return wrap_value(torch.broadcast_arrays(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def broadcast_to(*args, **kwargs):
    return wrap_value(torch.broadcast_to(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def busday_count(*args, **kwargs):
    return wrap_value(torch.busday_count(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def busday_offset(*args, **kwargs):
    return wrap_value(torch.busday_offset(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def busdaycalendar(*args, **kwargs):
    return wrap_value(torch.busdaycalendar(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def byte(*args, **kwargs):
    return wrap_value(torch.byte(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def byte_bounds(*args, **kwargs):
    return wrap_value(torch.byte_bounds(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def bytes0(*args, **kwargs):
    return wrap_value(torch.bytes0(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def bytes_(*args, **kwargs):
    return wrap_value(torch.bytes_(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def c_(*args, **kwargs):
    return wrap_value(torch.c_(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def can_cast(*args, **kwargs):
    return wrap_value(torch.can_cast(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def cast(*args, **kwargs):
    return wrap_value(torch.cast(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def cbrt(*args, **kwargs):
    return wrap_value(torch.cbrt(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def cdouble(*args, **kwargs):
    return wrap_value(torch.cdouble(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def ceil(*args, **kwargs):
    return wrap_value(torch.ceil(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def cfloat(*args, **kwargs):
    return wrap_value(torch.cfloat(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def char(*args, **kwargs):
    return wrap_value(torch.char(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def character(*args, **kwargs):
    return wrap_value(torch.character(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def chararray(*args, **kwargs):
    return wrap_value(torch.chararray(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def choose(*args, **kwargs):
    return wrap_value(torch.choose(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def clip(*args, **kwargs):
    return wrap_value(torch.clip(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def clongdouble(*args, **kwargs):
    return wrap_value(torch.clongdouble(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def clongfloat(*args, **kwargs):
    return wrap_value(torch.clongfloat(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def column_stack(*args, **kwargs):
    return wrap_value(torch.column_stack(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def common_type(*args, **kwargs):
    return wrap_value(torch.common_type(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def compare_chararrays(*args, **kwargs):
    return wrap_value(torch.compare_chararrays(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def compat(*args, **kwargs):
    return wrap_value(torch.compat(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def complex(*args, **kwargs):
    return wrap_value(torch.complex(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def complex128(*args, **kwargs):
    return wrap_value(torch.complex128(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def complex256(*args, **kwargs):
    return wrap_value(torch.complex256(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def complex64(*args, **kwargs):
    return wrap_value(torch.complex64(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def complex_(*args, **kwargs):
    return wrap_value(torch.complex_(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def complexfloating(*args, **kwargs):
    return wrap_value(torch.complexfloating(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def compress(*args, **kwargs):
    return wrap_value(torch.compress(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def concatenate(*args, **kwargs):
    return wrap_value(torch.concatenate(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def conj(*args, **kwargs):
    return wrap_value(torch.conj(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def conjugate(*args, **kwargs):
    return wrap_value(torch.conjugate(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def convolve(*args, **kwargs):
    return wrap_value(torch.convolve(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def copy(*args, **kwargs):
    return wrap_value(torch.copy(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def copysign(*args, **kwargs):
    return wrap_value(torch.copysign(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def copyto(*args, **kwargs):
    return wrap_value(torch.copyto(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def core(*args, **kwargs):
    return wrap_value(torch.core(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def corrcoef(*args, **kwargs):
    return wrap_value(torch.corrcoef(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def correlate(*args, **kwargs):
    return wrap_value(torch.correlate(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def cos(*args, **kwargs):
    return wrap_value(torch.cos(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def cosh(*args, **kwargs):
    return wrap_value(torch.cosh(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def count_nonzero(*args, **kwargs):
    return wrap_value(torch.count_nonzero(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def cov(*args, **kwargs):
    return wrap_value(torch.cov(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def cross(*args, **kwargs):
    return wrap_value(torch.cross(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def csingle(*args, **kwargs):
    return wrap_value(torch.csingle(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def ctypeslib(*args, **kwargs):
    return wrap_value(torch.ctypeslib(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def cumprod(*args, **kwargs):
    return wrap_value(torch.cumprod(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def cumproduct(*args, **kwargs):
    return wrap_value(torch.cumproduct(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def cumsum(*args, **kwargs):
    return wrap_value(torch.cumsum(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def datetime64(*args, **kwargs):
    return wrap_value(torch.datetime64(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def datetime_as_string(*args, **kwargs):
    return wrap_value(torch.datetime_as_string(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def datetime_data(*args, **kwargs):
    return wrap_value(torch.datetime_data(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def deg2rad(*args, **kwargs):
    return wrap_value(torch.deg2rad(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def degrees(*args, **kwargs):
    return wrap_value(torch.degrees(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def delete(*args, **kwargs):
    return wrap_value(torch.delete(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def deprecate(*args, **kwargs):
    return wrap_value(torch.deprecate(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def deprecate_with_doc(*args, **kwargs):
    return wrap_value(torch.deprecate_with_doc(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def diag(*args, **kwargs):
    return wrap_value(torch.diag(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def diag_indices(*args, **kwargs):
    return wrap_value(torch.diag_indices(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def diag_indices_from(*args, **kwargs):
    return wrap_value(torch.diag_indices_from(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def diagflat(*args, **kwargs):
    return wrap_value(torch.diagflat(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def diagonal(*args, **kwargs):
    return wrap_value(torch.diagonal(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def diff(*args, **kwargs):
    return wrap_value(torch.diff(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def digitize(*args, **kwargs):
    return wrap_value(torch.digitize(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def disp(*args, **kwargs):
    return wrap_value(torch.disp(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def divide(*args, **kwargs):
    return wrap_value(torch.divide(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def division(*args, **kwargs):
    return wrap_value(torch.division(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def divmod(*args, **kwargs):
    return wrap_value(torch.divmod(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def dot(*args, **kwargs):
    t1=args[0]
    t2=args[1]
    if t1.tensor.dim()==0 or t2.tensor.dim()==0:
        return t1 * t2
    if t1.tensor.dim() == 2 and t2.tensor.dim() == 2:
        return matmul(t1, t2)
    raise NotImplementedError("needs work")

    #return wrap_value(torch.dot(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def double(*args, **kwargs):
    return wrap_value(torch.double(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def dsplit(*args, **kwargs):
    return wrap_value(torch.dsplit(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def dstack(*args, **kwargs):
    return wrap_value(torch.dstack(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def dtype(*args, **kwargs):
    return wrap_value(torch.dtype(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def e(*args, **kwargs):
    return wrap_value(torch.e(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def ediff1d(*args, **kwargs):
    return wrap_value(torch.ediff1d(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def einsum(*args, **kwargs):
    return wrap_value(torch.einsum(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def einsum_path(*args, **kwargs):
    return wrap_value(torch.einsum_path(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def emath(*args, **kwargs):
    return wrap_value(torch.emath(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def empty(*args, **kwargs):
    return wrap_value(torch.empty(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def empty_like(*args, **kwargs):
    return wrap_value(torch.empty_like(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def equal(*args, **kwargs):
    return wrap_value(torch.equal(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def erf(*args, **kwargs):
    return wrap_value(torch.erf(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def errstate(*args, **kwargs):
    return wrap_value(torch.errstate(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def euler_gamma(*args, **kwargs):
    return wrap_value(torch.euler_gamma(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def exp(*args, **kwargs):
    return wrap_value(torch.exp(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def exp2(*args, **kwargs):
    return wrap_value(torch.exp2(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def expand_dims(*args, **kwargs):
    return wrap_value(torch.expand_dims(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def expm1(*args, **kwargs):
    return wrap_value(torch.expm1(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def extract(*args, **kwargs):
    return wrap_value(torch.extract(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def eye(*args, **kwargs):
    return wrap_value(torch.eye(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def fabs(*args, **kwargs):
    return wrap_value(torch.fabs(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def fastCopyAndTranspose(*args, **kwargs):
    return wrap_value(torch.fastCopyAndTranspose(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def fft(*args, **kwargs):
    return wrap_value(torch.fft(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def fill_diagonal(*args, **kwargs):
    return wrap_value(torch.fill_diagonal(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def find_common_type(*args, **kwargs):
    return wrap_value(torch.find_common_type(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def finfo(*args, **kwargs):
    return wrap_value(torch.finfo(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def fix(*args, **kwargs):
    return wrap_value(torch.fix(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def flatiter(*args, **kwargs):
    return wrap_value(torch.flatiter(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def flatnonzero(*args, **kwargs):
    return wrap_value(torch.flatnonzero(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def flexible(*args, **kwargs):
    return wrap_value(torch.flexible(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def flip(*args, **kwargs):
    return wrap_value(torch.flip(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def fliplr(*args, **kwargs):
    return wrap_value(torch.fliplr(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def flipud(*args, **kwargs):
    return wrap_value(torch.flipud(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))

float=torch.float
float32=float
double=torch.double
float64=double
int=torch.int
int32=int
long=torch.long
int64=long


#def float(*args, **kwargs):
#    return wrap_value(torch.float(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def float128(*args, **kwargs):
    return wrap_value(torch.float128(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def float16(*args, **kwargs):
    return wrap_value(torch.float16(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def float32(*args, **kwargs):
    return wrap_value(torch.float32(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def float64(*args, **kwargs):
    return wrap_value(torch.float64(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def float_(*args, **kwargs):
    return wrap_value(torch.float_(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def float_power(*args, **kwargs):
    return wrap_value(torch.float_power(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def floating(*args, **kwargs):
    return wrap_value(torch.floating(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def floor(*args, **kwargs):
    return wrap_value(torch.floor(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def floor_divide(*args, **kwargs):
    return wrap_value(torch.floor_divide(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def fmax(*args, **kwargs):
    return wrap_value(torch.fmax(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def fmin(*args, **kwargs):
    return wrap_value(torch.fmin(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def fmod(*args, **kwargs):
    return wrap_value(torch.fmod(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def format_float_positional(*args, **kwargs):
    return wrap_value(torch.format_float_positional(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def format_float_scientific(*args, **kwargs):
    return wrap_value(torch.format_float_scientific(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def format_parser(*args, **kwargs):
    return wrap_value(torch.format_parser(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def frexp(*args, **kwargs):
    return wrap_value(torch.frexp(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def frombuffer(*args, **kwargs):
    return wrap_value(torch.frombuffer(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def fromfile(*args, **kwargs):
    return wrap_value(torch.fromfile(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def fromfunction(*args, **kwargs):
    return wrap_value(torch.fromfunction(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def fromiter(*args, **kwargs):
    return wrap_value(torch.fromiter(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def frompyfunc(*args, **kwargs):
    return wrap_value(torch.frompyfunc(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def fromregex(*args, **kwargs):
    return wrap_value(torch.fromregex(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def fromstring(*args, **kwargs):
    return wrap_value(torch.fromstring(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))

def full(*args, **kwargs):
    if (not isinstance(args[0], tuple) ):
        args = ((args[0],), args[1])
        kwargs = unwrap_kwargs(**kwargs)
        return wrap_value(torch.full(*args, **kwargs))
    return wrap_value(torch.full(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))

def full_like(*args, **kwargs):
    return wrap_value(torch.full_like(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def fv(*args, **kwargs):
    return wrap_value(torch.fv(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def gcd(*args, **kwargs):
    return wrap_value(torch.gcd(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def generic(*args, **kwargs):
    return wrap_value(torch.generic(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def genfromtxt(*args, **kwargs):
    return wrap_value(torch.genfromtxt(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def geomspace(*args, **kwargs):
    return wrap_value(torch.geomspace(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def get_array_wrap(*args, **kwargs):
    return wrap_value(torch.get_array_wrap(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def get_include(*args, **kwargs):
    return wrap_value(torch.get_include(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def get_printoptions(*args, **kwargs):
    return wrap_value(torch.get_printoptions(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def getbufsize(*args, **kwargs):
    return wrap_value(torch.getbufsize(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def geterr(*args, **kwargs):
    return wrap_value(torch.geterr(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def geterrcall(*args, **kwargs):
    return wrap_value(torch.geterrcall(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def geterrobj(*args, **kwargs):
    return wrap_value(torch.geterrobj(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def gradient(*args, **kwargs):
    return wrap_value(torch.gradient(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def greater(*args, **kwargs):
    return wrap_value(torch.greater(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def greater_equal(*args, **kwargs):
    return wrap_value(torch.greater_equal(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def half(*args, **kwargs):
    return wrap_value(torch.half(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def hamming(*args, **kwargs):
    return wrap_value(torch.hamming(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def hanning(*args, **kwargs):
    return wrap_value(torch.hanning(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def heaviside(*args, **kwargs):
    return wrap_value(torch.heaviside(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def histogram(*args, **kwargs):
    return wrap_value(torch.histogram(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def histogram2d(*args, **kwargs):
    return wrap_value(torch.histogram2d(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def histogram_bin_edges(*args, **kwargs):
    return wrap_value(torch.histogram_bin_edges(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def histogramdd(*args, **kwargs):
    return wrap_value(torch.histogramdd(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def hsplit(*args, **kwargs):
    return wrap_value(torch.hsplit(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def hstack(*args, **kwargs):
    return wrap_value(torch.hstack(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def hypot(*args, **kwargs):
    return wrap_value(torch.hypot(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def i0(*args, **kwargs):
    return wrap_value(torch.i0(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def identity(*args, **kwargs):
    return wrap_value(torch.identity(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def iinfo(*args, **kwargs):
    return wrap_value(torch.iinfo(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def imag(*args, **kwargs):
    return wrap_value(torch.imag(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def in1d(*args, **kwargs):
    return wrap_value(torch.in1d(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def index_exp(*args, **kwargs):
    return wrap_value(torch.index_exp(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def indices(*args, **kwargs):
    return wrap_value(torch.indices(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def inexact(*args, **kwargs):
    return wrap_value(torch.inexact(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def inf(*args, **kwargs):
    return wrap_value(torch.inf(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def info(*args, **kwargs):
    return wrap_value(torch.info(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def infty(*args, **kwargs):
    return wrap_value(torch.infty(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def inner(*args, **kwargs):
    return wrap_value(torch.inner(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def insert(*args, **kwargs):
    return wrap_value(torch.insert(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
#def int(*args, **kwargs):
#    return wrap_value(torch.int(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
#def int0(*args, **kwargs):
#    return wrap_value(torch.int0(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
#def int16(*args, **kwargs):
#    return wrap_value(torch.int16(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
#def int32(*args, **kwargs):
#    return wrap_value(torch.int32(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
#def int64(*args, **kwargs):
#    return wrap_value(torch.int64(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
#def int8(*args, **kwargs):
#    return wrap_value(torch.int8(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
#def int_(*args, **kwargs):
#    return wrap_value(torch.int_(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def int_asbuffer(*args, **kwargs):
    return wrap_value(torch.int_asbuffer(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def intc(*args, **kwargs):
    return wrap_value(torch.intc(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def integer(*args, **kwargs):
    return wrap_value(torch.integer(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def interp(*args, **kwargs):
    return wrap_value(torch.interp(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def intersect1d(*args, **kwargs):
    return wrap_value(torch.intersect1d(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def intp(*args, **kwargs):
    return wrap_value(torch.intp(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def invert(*args, **kwargs):
    return wrap_value(torch.invert(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def ipmt(*args, **kwargs):
    return wrap_value(torch.ipmt(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def irr(*args, **kwargs):
    return wrap_value(torch.irr(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def is_busday(*args, **kwargs):
    return wrap_value(torch.is_busday(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def isclose(*args, **kwargs):
    return wrap_value(torch.isclose(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def iscomplex(*args, **kwargs):
    return wrap_value(torch.iscomplex(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def iscomplexobj(*args, **kwargs):
    return wrap_value(torch.iscomplexobj(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def isfinite(*args, **kwargs):
    return wrap_value(torch.isfinite(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def isfortran(*args, **kwargs):
    return wrap_value(torch.isfortran(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def isin(*args, **kwargs):
    return wrap_value(torch.isin(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def isinf(*args, **kwargs):
    return wrap_value(torch.isinf(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def isnan(*args, **kwargs):
    return wrap_value(torch.isnan(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def isnat(*args, **kwargs):
    return wrap_value(torch.isnat(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def isneginf(*args, **kwargs):
    return wrap_value(torch.isneginf(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def isposinf(*args, **kwargs):
    return wrap_value(torch.isposinf(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def isreal(*args, **kwargs):
    return wrap_value(torch.isreal(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def isrealobj(*args, **kwargs):
    return wrap_value(torch.isrealobj(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def isscalar(*args, **kwargs):
    return wrap_value(torch.isscalar(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def issctype(*args, **kwargs):
    return wrap_value(torch.issctype(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def issubclass_(*args, **kwargs):
    return wrap_value(torch.issubclass_(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def issubdtype(*args, **kwargs):
    return wrap_value(torch.issubdtype(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def issubsctype(*args, **kwargs):
    return wrap_value(torch.issubsctype(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def iterable(*args, **kwargs):
    return wrap_value(torch.iterable(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def ix_(*args, **kwargs):
    return wrap_value(torch.ix_(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def kaiser(*args, **kwargs):
    return wrap_value(torch.kaiser(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def kron(*args, **kwargs):
    return wrap_value(torch.kron(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def lcm(*args, **kwargs):
    return wrap_value(torch.lcm(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def ldexp(*args, **kwargs):
    return wrap_value(torch.ldexp(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def left_shift(*args, **kwargs):
    return wrap_value(torch.left_shift(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def less(*args, **kwargs):
    return wrap_value(torch.less(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def less_equal(*args, **kwargs):
    return wrap_value(torch.less_equal(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def lexsort(*args, **kwargs):
    return wrap_value(torch.lexsort(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def lib(*args, **kwargs):
    return wrap_value(torch.lib(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def linalg(*args, **kwargs):
    return wrap_value(torch.linalg(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def linspace(*args, **kwargs):
    return wrap_value(torch.linspace(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def little_endian(*args, **kwargs):
    return wrap_value(torch.little_endian(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def load(*args, **kwargs):
    return wrap_value(torch.load(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def loads(*args, **kwargs):
    return wrap_value(torch.loads(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def loadtxt(*args, **kwargs):
    return wrap_value(torch.loadtxt(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def log(*args, **kwargs):
    return wrap_value(torch.log(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def log10(*args, **kwargs):
    return wrap_value(torch.log10(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def log1p(*args, **kwargs):
    return wrap_value(torch.log1p(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def log2(*args, **kwargs):
    return wrap_value(torch.log2(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def logaddexp(*args, **kwargs):
    return wrap_value(torch.logaddexp(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def logaddexp2(*args, **kwargs):
    return wrap_value(torch.logaddexp2(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def logical_and(*args, **kwargs):
    return wrap_value(torch.logical_and(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def logical_not(*args, **kwargs):
    return wrap_value(torch.logical_not(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def logical_or(*args, **kwargs):
    return wrap_value(torch.logical_or(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def logical_xor(*args, **kwargs):
    return wrap_value(torch.logical_xor(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def logspace(*args, **kwargs):
    return wrap_value(torch.logspace(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def long(*args, **kwargs):
    return wrap_value(torch.long(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def longcomplex(*args, **kwargs):
    return wrap_value(torch.longcomplex(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def longdouble(*args, **kwargs):
    return wrap_value(torch.longdouble(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def longfloat(*args, **kwargs):
    return wrap_value(torch.longfloat(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def longlong(*args, **kwargs):
    return wrap_value(torch.longlong(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def lookfor(*args, **kwargs):
    return wrap_value(torch.lookfor(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def ma(*args, **kwargs):
    return wrap_value(torch.ma(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def mafromtxt(*args, **kwargs):
    return wrap_value(torch.mafromtxt(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def mask_indices(*args, **kwargs):
    return wrap_value(torch.mask_indices(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def mat(*args, **kwargs):
    return wrap_value(torch.mat(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def math(*args, **kwargs):
    return wrap_value(torch.math(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def matmul(*args, **kwargs):
    arg1=args[0]
    arg2=args[1]
    dtype=torch.result_type(arg1.tensor, arg2.tensor)
    return wrap_value(torch.matmul(arg1.tensor.to(dtype), arg2.tensor.to(dtype)))
def matrix(*args, **kwargs):
    return wrap_value(torch.matrix(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def matrixlib(*args, **kwargs):
    return wrap_value(torch.matrixlib(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def max(*args, **kwargs):
    return wrap_value(torch.max(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def maximum(*args, **kwargs):
    return wrap_value(torch.maximum(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def maximum_sctype(*args, **kwargs):
    return wrap_value(torch.maximum_sctype(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def may_share_memory(*args, **kwargs):
    return wrap_value(torch.may_share_memory(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def mean(*args, **kwargs):
    return wrap_value(torch.mean(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def median(*args, **kwargs):
    return wrap_value(torch.median(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def memmap(*args, **kwargs):
    return wrap_value(torch.memmap(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def meshgrid(*args, **kwargs):
    return wrap_value(torch.meshgrid(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def mgrid(*args, **kwargs):
    return wrap_value(torch.mgrid(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def min(*args, **kwargs):
    return wrap_value(torch.min(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def min_scalar_type(*args, **kwargs):
    return wrap_value(torch.min_scalar_type(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def minimum(*args, **kwargs):
    return wrap_value(torch.minimum(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def mintypecode(*args, **kwargs):
    return wrap_value(torch.mintypecode(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def mirr(*args, **kwargs):
    return wrap_value(torch.mirr(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def mod(*args, **kwargs):
    return wrap_value(torch.mod(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def modf(*args, **kwargs):
    return wrap_value(torch.modf(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def moveaxis(*args, **kwargs):
    return wrap_value(torch.moveaxis(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def msort(*args, **kwargs):
    return wrap_value(torch.msort(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def multiply(*args, **kwargs):
    return wrap_value(torch.multiply(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def nan(*args, **kwargs):
    return wrap_value(torch.nan(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def nan_to_num(*args, **kwargs):
    return wrap_value(torch.nan_to_num(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def nanargmax(*args, **kwargs):
    return wrap_value(torch.nanargmax(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def nanargmin(*args, **kwargs):
    return wrap_value(torch.nanargmin(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def nancumprod(*args, **kwargs):
    return wrap_value(torch.nancumprod(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def nancumsum(*args, **kwargs):
    return wrap_value(torch.nancumsum(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def nanmax(*args, **kwargs):
    return wrap_value(torch.nanmax(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def nanmean(*args, **kwargs):
    return wrap_value(torch.nanmean(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def nanmedian(*args, **kwargs):
    return wrap_value(torch.nanmedian(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def nanmin(*args, **kwargs):
    return wrap_value(torch.nanmin(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def nanpercentile(*args, **kwargs):
    return wrap_value(torch.nanpercentile(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def nanprod(*args, **kwargs):
    return wrap_value(torch.nanprod(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def nanquantile(*args, **kwargs):
    return wrap_value(torch.nanquantile(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def nanstd(*args, **kwargs):
    return wrap_value(torch.nanstd(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def nansum(*args, **kwargs):
    return wrap_value(torch.nansum(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def nanvar(*args, **kwargs):
    return wrap_value(torch.nanvar(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def nbytes(*args, **kwargs):
    return wrap_value(torch.nbytes(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))

def ndarray(*args, **kwargs):
    return wrap_value(torch.empty(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))

def ndenumerate(*args, **kwargs):
    return wrap_value(torch.ndenumerate(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def ndfromtxt(*args, **kwargs):
    return wrap_value(torch.ndfromtxt(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
#def ndim(*args, **kwargs):
#    return wrap_value(torch.ndim(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def ndindex(*args, **kwargs):
    return wrap_value(torch.ndindex(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def nditer(*args, **kwargs):
    return wrap_value(torch.nditer(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def negative(*args, **kwargs):
    return wrap_value(torch.negative(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def nested_iters(*args, **kwargs):
    return wrap_value(torch.nested_iters(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def newaxis(*args, **kwargs):
    return wrap_value(torch.newaxis(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def nextafter(*args, **kwargs):
    return wrap_value(torch.nextafter(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def nonzero(*args, **kwargs):
    return wrap_value(torch.nonzero(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def not_equal(*args, **kwargs):
    return wrap_value(torch.not_equal(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def nper(*args, **kwargs):
    return wrap_value(torch.nper(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def npv(*args, **kwargs):
    return wrap_value(torch.npv(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def numarray(*args, **kwargs):
    return wrap_value(torch.numarray(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def number(*args, **kwargs):
    return wrap_value(torch.number(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def obj2sctype(*args, **kwargs):
    return wrap_value(torch.obj2sctype(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def object(*args, **kwargs):
    return wrap_value(torch.object(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def object0(*args, **kwargs):
    return wrap_value(torch.object0(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def object_(*args, **kwargs):
    return wrap_value(torch.object_(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def ogrid(*args, **kwargs):
    return wrap_value(torch.ogrid(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def oldnumeric(*args, **kwargs):
    return wrap_value(torch.oldnumeric(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def ones(*args, **kwargs):
    return wrap_value(torch.ones(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def ones_like(*args, **kwargs):
    return wrap_value(torch.ones_like(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def outer(*args, **kwargs):
    return wrap_value(torch.outer(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def packbits(*args, **kwargs):
    return wrap_value(torch.packbits(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def pad(*args, **kwargs):
    return wrap_value(torch.pad(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def partition(*args, **kwargs):
    return wrap_value(torch.partition(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def percentile(*args, **kwargs):
    return wrap_value(torch.percentile(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def pi(*args, **kwargs):
    return wrap_value(torch.pi(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def piecewise(*args, **kwargs):
    return wrap_value(torch.piecewise(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def pkgload(*args, **kwargs):
    return wrap_value(torch.pkgload(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def place(*args, **kwargs):
    return wrap_value(torch.place(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def pmt(*args, **kwargs):
    return wrap_value(torch.pmt(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def poly(*args, **kwargs):
    return wrap_value(torch.poly(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def poly1d(*args, **kwargs):
    return wrap_value(torch.poly1d(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def polyadd(*args, **kwargs):
    return wrap_value(torch.polyadd(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def polyder(*args, **kwargs):
    return wrap_value(torch.polyder(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def polydiv(*args, **kwargs):
    return wrap_value(torch.polydiv(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def polyfit(*args, **kwargs):
    return wrap_value(torch.polyfit(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def polyint(*args, **kwargs):
    return wrap_value(torch.polyint(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def polymul(*args, **kwargs):
    return wrap_value(torch.polymul(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def polynomial(*args, **kwargs):
    return wrap_value(torch.polynomial(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def polysub(*args, **kwargs):
    return wrap_value(torch.polysub(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def polyval(*args, **kwargs):
    return wrap_value(torch.polyval(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def positive(*args, **kwargs):
    return wrap_value(torch.positive(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def power(*args, **kwargs):
    return wrap_value(torch.pow(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def ppmt(*args, **kwargs):
    return wrap_value(torch.ppmt(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def print_function(*args, **kwargs):
    return wrap_value(torch.print_function(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def printoptions(*args, **kwargs):
    return wrap_value(torch.printoptions(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def prod(*args, **kwargs):
    return wrap_value(torch.prod(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def product(*args, **kwargs):
    return wrap_value(torch.product(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def promote_types(*args, **kwargs):
    return wrap_value(torch.promote_types(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def ptp(*args, **kwargs):
    return wrap_value(torch.ptp(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def put(*args, **kwargs):
    return wrap_value(torch.put(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def put_along_axis(*args, **kwargs):
    return wrap_value(torch.put_along_axis(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def putmask(*args, **kwargs):
    return wrap_value(torch.putmask(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def pv(*args, **kwargs):
    return wrap_value(torch.pv(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def quantile(*args, **kwargs):
    return wrap_value(torch.quantile(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def r_(*args, **kwargs):
    return wrap_value(torch.r_(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def rad2deg(*args, **kwargs):
    return wrap_value(torch.rad2deg(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def radians(*args, **kwargs):
    return wrap_value(torch.radians(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
#def random(*args, **kwargs):
#    return wrap_value(torch.random(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def rank(*args, **kwargs):
    return wrap_value(torch.rank(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def rate(*args, **kwargs):
    return wrap_value(torch.rate(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def ravel(*args, **kwargs):
    return wrap_value(torch.ravel(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def ravel_multi_index(*args, **kwargs):
    return wrap_value(torch.ravel_multi_index(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def real(*args, **kwargs):
    return wrap_value(torch.real(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def real_if_close(*args, **kwargs):
    return wrap_value(torch.real_if_close(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def rec(*args, **kwargs):
    return wrap_value(torch.rec(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def recarray(*args, **kwargs):
    return wrap_value(torch.recarray(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def recfromcsv(*args, **kwargs):
    return wrap_value(torch.recfromcsv(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def recfromtxt(*args, **kwargs):
    return wrap_value(torch.recfromtxt(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def reciprocal(*args, **kwargs):
    return wrap_value(torch.reciprocal(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def record(*args, **kwargs):
    return wrap_value(torch.record(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def remainder(*args, **kwargs):
    return wrap_value(torch.remainder(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def repeat(*args, **kwargs):
    return wrap_value(torch.repeat(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def require(*args, **kwargs):
    return wrap_value(torch.require(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def reshape(*args, **kwargs):
    return wrap_value(torch.reshape(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def resize(*args, **kwargs):
    return wrap_value(torch.resize(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def result_type(*args, **kwargs):
    return wrap_value(torch.result_type(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def right_shift(*args, **kwargs):
    return wrap_value(torch.right_shift(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def rint(*args, **kwargs):
    return wrap_value(torch.rint(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def roll(*args, **kwargs):
    return wrap_value(torch.roll(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def rollaxis(*args, **kwargs):
    return wrap_value(torch.rollaxis(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def roots(*args, **kwargs):
    return wrap_value(torch.roots(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def rot90(*args, **kwargs):
    return wrap_value(torch.rot90(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def round(*args, **kwargs):
    return wrap_value(torch.round(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def round_(*args, **kwargs):
    return wrap_value(torch.round_(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def row_stack(*args, **kwargs):
    return wrap_value(torch.row_stack(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def s_(*args, **kwargs):
    return wrap_value(torch.s_(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def safe_eval(*args, **kwargs):
    return wrap_value(torch.safe_eval(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def save(*args, **kwargs):
    return wrap_value(torch.save(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def savetxt(*args, **kwargs):
    return wrap_value(torch.savetxt(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def savez(*args, **kwargs):
    return wrap_value(torch.savez(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def savez_compressed(*args, **kwargs):
    return wrap_value(torch.savez_compressed(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def sctype2char(*args, **kwargs):
    return wrap_value(torch.sctype2char(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def sctypeDict(*args, **kwargs):
    return wrap_value(torch.sctypeDict(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def sctypeNA(*args, **kwargs):
    return wrap_value(torch.sctypeNA(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def sctypes(*args, **kwargs):
    return wrap_value(torch.sctypes(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def searchsorted(*args, **kwargs):
    return wrap_value(torch.searchsorted(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def select(*args, **kwargs):
    return wrap_value(torch.select(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def set_numeric_ops(*args, **kwargs):
    return wrap_value(torch.set_numeric_ops(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def set_printoptions(*args, **kwargs):
    return wrap_value(torch.set_printoptions(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def set_string_function(*args, **kwargs):
    return wrap_value(torch.set_string_function(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def setbufsize(*args, **kwargs):
    return wrap_value(torch.setbufsize(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def setdiff1d(*args, **kwargs):
    return wrap_value(torch.setdiff1d(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def seterr(*args, **kwargs):
    return wrap_value(torch.seterr(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def seterrcall(*args, **kwargs):
    return wrap_value(torch.seterrcall(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def seterrobj(*args, **kwargs):
    return wrap_value(torch.seterrobj(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def setxor1d(*args, **kwargs):
    return wrap_value(torch.setxor1d(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def shape(*args, **kwargs):
    return wrap_value(torch.shape(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def shares_memory(*args, **kwargs):
    return wrap_value(torch.shares_memory(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def short(*args, **kwargs):
    return wrap_value(torch.short(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def show_config(*args, **kwargs):
    return wrap_value(torch.show_config(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def sign(*args, **kwargs):
    return wrap_value(torch.sign(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def signbit(*args, **kwargs):
    return wrap_value(torch.signbit(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def signedinteger(*args, **kwargs):
    return wrap_value(torch.signedinteger(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def sin(*args, **kwargs):
    return wrap_value(torch.sin(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def sinc(*args, **kwargs):
    return wrap_value(torch.sinc(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def single(*args, **kwargs):
    return wrap_value(torch.single(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def singlecomplex(*args, **kwargs):
    return wrap_value(torch.singlecomplex(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def sinh(*args, **kwargs):
    return wrap_value(torch.sinh(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def size(*args, **kwargs):
    return wrap_value(torch.size(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def sometrue(*args, **kwargs):
    return wrap_value(torch.sometrue(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def sort(*args, **kwargs):
    return wrap_value(torch.sort(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def sort_complex(*args, **kwargs):
    return wrap_value(torch.sort_complex(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def source(*args, **kwargs):
    return wrap_value(torch.source(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def spacing(*args, **kwargs):
    return wrap_value(torch.spacing(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def split(*args, **kwargs):
    return wrap_value(torch.split(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def sqrt(*args, **kwargs):
    return wrap_value(torch.sqrt(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def square(*args, **kwargs):
    return wrap_value(torch.square(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def squeeze(*args, **kwargs):
    return wrap_value(torch.squeeze(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def stack(*args, **kwargs):
    return wrap_value(torch.stack(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def std(*args, **kwargs):
    return wrap_value(torch.std(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def str(*args, **kwargs):
    return wrap_value(torch.str(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def str0(*args, **kwargs):
    return wrap_value(torch.str0(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def str_(*args, **kwargs):
    return wrap_value(torch.str_(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def string_(*args, **kwargs):
    return wrap_value(torch.string_(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def subtract(*args, **kwargs):
    return wrap_value(torch.subtract(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def sum(*args, **kwargs):
    return wrap_value(torch.sum(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def swapaxes(*args, **kwargs):
    return wrap_value(torch.swapaxes(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def sys(*args, **kwargs):
    return wrap_value(torch.sys(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def take(*args, **kwargs):
    return wrap_value(torch.take(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def take_along_axis(*args, **kwargs):
    return wrap_value(torch.take_along_axis(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def tan(*args, **kwargs):
    return wrap_value(torch.tan(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def tanh(*args, **kwargs):
    return wrap_value(torch.tanh(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def tensordot(*args, **kwargs):
    return wrap_value(torch.tensordot(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def test(*args, **kwargs):
    return wrap_value(torch.test(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def testing(*args, **kwargs):
    return wrap_value(torch.testing(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def tile(*args, **kwargs):
    return wrap_value(torch.tile(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def timedelta64(*args, **kwargs):
    return wrap_value(torch.timedelta64(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def trace(*args, **kwargs):
    return wrap_value(torch.trace(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def tracemalloc_domain(*args, **kwargs):
    return wrap_value(torch.tracemalloc_domain(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def transpose(*args, **kwargs):
    return wrap_value(torch.transpose(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def trapz(*args, **kwargs):
    return wrap_value(torch.trapz(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def tri(*args, **kwargs):
    return wrap_value(torch.tri(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def tril(*args, **kwargs):
    return wrap_value(torch.tril(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def tril_indices(*args, **kwargs):
    return wrap_value(torch.tril_indices(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def tril_indices_from(*args, **kwargs):
    return wrap_value(torch.tril_indices_from(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def trim_zeros(*args, **kwargs):
    return wrap_value(torch.trim_zeros(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def triu(*args, **kwargs):
    return wrap_value(torch.triu(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def triu_indices(*args, **kwargs):
    return wrap_value(torch.triu_indices(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def triu_indices_from(*args, **kwargs):
    return wrap_value(torch.triu_indices_from(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def true_divide(*args, **kwargs):
    return wrap_value(torch.true_divide(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def trunc(*args, **kwargs):
    return wrap_value(torch.trunc(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def typeDict(*args, **kwargs):
    return wrap_value(torch.typeDict(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def typeNA(*args, **kwargs):
    return wrap_value(torch.typeNA(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def typecodes(*args, **kwargs):
    return wrap_value(torch.typecodes(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def typename(*args, **kwargs):
    return wrap_value(torch.typename(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def ubyte(*args, **kwargs):
    return wrap_value(torch.ubyte(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def ufunc(*args, **kwargs):
    return wrap_value(torch.ufunc(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def uint(*args, **kwargs):
    return wrap_value(torch.uint(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def uint0(*args, **kwargs):
    return wrap_value(torch.uint0(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))


bool=torch.bool
uint8=torch.uint8
int64=torch.int64
int32=torch.int32
int=torch.int

float16=torch.float16
float32=torch.float32
float=torch.float
float64=torch.double
double=torch.double



"""
def uint16(*args, **kwargs):
    return wrap_value(torch.uint16(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def uint32(*args, **kwargs):
    return wrap_value(torch.uint32(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def uint64(*args, **kwargs):
    return wrap_value(torch.uint64(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def uint8(*args, **kwargs):
    return wrap_value(torch.uint8(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
"""
def uintc(*args, **kwargs):
    return wrap_value(torch.uintc(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def uintp(*args, **kwargs):
    return wrap_value(torch.uintp(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def ulonglong(*args, **kwargs):
    return wrap_value(torch.ulonglong(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def unicode(*args, **kwargs):
    return wrap_value(torch.unicode(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def unicode_(*args, **kwargs):
    return wrap_value(torch.unicode_(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def union1d(*args, **kwargs):
    return wrap_value(torch.union1d(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def unique(*args, **kwargs):
    return wrap_value(torch.unique(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def unpackbits(*args, **kwargs):
    return wrap_value(torch.unpackbits(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def unravel_index(*args, **kwargs):
    return wrap_value(torch.unravel_index(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def unsignedinteger(*args, **kwargs):
    return wrap_value(torch.unsignedinteger(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def unwrap(*args, **kwargs):
    return wrap_value(torch.unwrap(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def ushort(*args, **kwargs):
    return wrap_value(torch.ushort(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def vander(*args, **kwargs):
    return wrap_value(torch.vander(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def var(*args, **kwargs):
    return wrap_value(torch.var(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def vdot(*args, **kwargs):
    return wrap_value(torch.vdot(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def vectorize(*args, **kwargs):
    return wrap_value(torch.vectorize(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def version(*args, **kwargs):
    return wrap_value(torch.version(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def void(*args, **kwargs):
    return wrap_value(torch.void(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def void0(*args, **kwargs):
    return wrap_value(torch.void0(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def vsplit(*args, **kwargs):
    return wrap_value(torch.vsplit(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def vstack(*args, **kwargs):
    return wrap_value(torch.vstack(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def warnings(*args, **kwargs):
    return wrap_value(torch.warnings(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def where(*args, **kwargs):
    return wrap_value(torch.where(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def who(*args, **kwargs):
    return wrap_value(torch.who(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def zeros(*args, **kwargs):
    return wrap_value(torch.zeros(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))
def zeros_like(*args, **kwargs):
    return wrap_value(torch.zeros_like(*unwrap_args(*args), **unwrap_kwargs(**kwargs)))




