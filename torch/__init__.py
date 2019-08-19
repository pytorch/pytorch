# @lint-ignore-every PYTHON3COMPATIMPORTS

r"""
The torch package contains data structures for multi-dimensional
tensors and mathematical operations over these are defined.
Additionally, it provides many utilities for efficient serializing of
Tensors and arbitrary types, and other useful utilities.

It has a CUDA counterpart, that enables you to run your tensor computations
on an NVIDIA GPU with compute capability >= 3.0.
"""

import os
import sys
import platform
import collections
from ._utils import _import_dotted_name
from ._utils_internal import get_file_path, prepare_multiprocessing_environment
from .version import __version__  # noqa: F401
from ._six import string_classes as _string_classes

__all__ = [
    'typename', 'is_tensor', 'is_storage', 'set_default_tensor_type',
    'set_rng_state', 'get_rng_state', 'manual_seed', 'initial_seed', 'seed',
    'save', 'load', 'set_printoptions', 'chunk', 'split', 'stack', 'matmul',
    'no_grad', 'enable_grad', 'rand', 'randn',
    'DoubleStorage', 'FloatStorage', 'LongStorage', 'IntStorage',
    'ShortStorage', 'CharStorage', 'ByteStorage', 'BoolStorage',
    'DoubleTensor', 'FloatTensor', 'LongTensor', 'IntTensor',
    'ShortTensor', 'CharTensor', 'ByteTensor', 'BoolTensor', 'Tensor',
]

################################################################################
# Load the extension module
################################################################################

# Loading the extension with RTLD_GLOBAL option allows to not link extension
# modules against the _C shared object. Their missing THP symbols will be
# automatically filled by the dynamic loader.
import os as _dl_flags

# if we have numpy, it *must* be imported before the call to setdlopenflags()
# or there is risk that later c modules will segfault when importing numpy
try:
    import numpy as _np  # noqa: F401
except ImportError:
    pass

if platform.system() == 'Windows':
    # first get nvToolsExt PATH
    def get_nvToolsExt_path():
        NVTOOLEXT_HOME = _dl_flags.getenv('NVTOOLSEXT_PATH', 'C:\\Program Files\\NVIDIA Corporation\\NvToolsExt')

        if _dl_flags.path.exists(NVTOOLEXT_HOME):
            return _dl_flags.path.join(NVTOOLEXT_HOME, 'bin', 'x64')
        else:
            return ''

    py_dll_path = _dl_flags.path.join(sys.exec_prefix, 'Library', 'bin')
    th_dll_path = _dl_flags.path.join(_dl_flags.path.dirname(__file__), 'lib')

    dll_paths = [th_dll_path, py_dll_path, get_nvToolsExt_path(), _dl_flags.environ['PATH']]

    # then add the path to env
    _dl_flags.environ['PATH'] = ';'.join(dll_paths)

else:
    # first check if the os package has the required flags
    if not hasattr(_dl_flags, 'RTLD_GLOBAL') or not hasattr(_dl_flags, 'RTLD_LAZY'):
        try:
            # next try if DLFCN exists
            import DLFCN as _dl_flags
        except ImportError:
            # as a last attempt, use compile-time constants
            import torch._dl as _dl_flags

    old_flags = sys.getdlopenflags()
    sys.setdlopenflags(_dl_flags.RTLD_GLOBAL | _dl_flags.RTLD_LAZY)

del _dl_flags

from torch._C import *

__all__ += [name for name in dir(_C)
            if name[0] != '_' and
            not name.endswith('Base')]

if platform.system() != 'Windows':
    sys.setdlopenflags(old_flags)
    del old_flags

################################################################################
# Define basic utilities
################################################################################
import types

def ismethod(object):
    """Return true if the object is an instance method.
    Instance method objects provide these attributes:
        __doc__         documentation string
        __name__        name with which this method was defined
        im_class        class object in which this method belongs
        im_func         function object containing implementation of method
        im_self         instance to which this method is bound, or None
    """
    return isinstance(object, types.MethodType)


def isfunction(object):
    """Return true if the object is a user-defined function.
    Function objects provide these attributes:
        __doc__         documentation string
        __name__        name with which this function was defined
        func_code       code object containing compiled function bytecode
        func_defaults   tuple of any default values for arguments
        func_doc        (same as __doc__)
        func_globals    global namespace in which this function was defined
        func_name       (same as __name__)
    """
    return isinstance(object, types.FunctionType)
def getargspec(func):
    """Get the names and default values of a function's arguments.
    A tuple of four things is returned: (args, varargs, varkw, defaults).
    'args' is a list of the argument names (it may contain nested lists).
    'varargs' and 'varkw' are the names of the * and ** arguments or None.
    'defaults' is an n-tuple of the default values of the last n arguments.
    """

    if ismethod(func):
        func = func.__func__
    if not isfunction(func):
        raise TypeError('arg is not a Python function')
    args, varargs, varkw = getargs(func.__code__)
    return args, varargs, varkw, func.__defaults__

def iscode(object):
    """Return true if the object is a code object.
    Code objects provide these attributes:
        co_argcount     number of arguments (not including * or ** args)
        co_code         string of raw compiled bytecode
        co_consts       tuple of constants used in the bytecode
        co_filename     name of file in which this code object was created
        co_firstlineno  number of first line in Python source code
        co_flags        bitmap: 1=optimized | 2=newlocals | 4=*arg | 8=**arg
        co_lnotab       encoded mapping of line numbers to bytecode indices
        co_name         name with which this code object was defined
        co_names        tuple of names of local variables
        co_nlocals      number of local variables
        co_stacksize    virtual machine stack space required
        co_varnames     tuple of names of arguments and local variables

    """
    return isinstance(object, types.CodeType)

CO_OPTIMIZED, CO_NEWLOCALS, CO_VARARGS, CO_VARKEYWORDS = 1, 2, 4, 8

def getargs(co):
    """Get information about the arguments accepted by a code object.
    Three things are returned: (args, varargs, varkw), where 'args' is
    a list of argument names (possibly containing nested lists), and
    'varargs' and 'varkw' are the names of the * and ** arguments or None.
    """

    if not iscode(co):
        raise TypeError('arg is not a code object')

    nargs = co.co_argcount
    names = co.co_varnames
    args = list(names[:nargs])

    # The following acrobatics are for anonymous (tuple) arguments.
    # Which we do not need to support, so remove to avoid importing
    # the dis module.
    for i in range(nargs):
        if args[i][:1] in ['', '.']:
            raise TypeError("tuple function arguments are not supported")
    varargs = None
    if co.co_flags & CO_VARARGS:
        varargs = co.co_varnames[nargs]
        nargs = nargs + 1
    varkw = None
    if co.co_flags & CO_VARKEYWORDS:
        varkw = co.co_varnames[nargs]
    return args, varargs, varkw


ArgSpec = collections.namedtuple('ArgSpec', 'args varargs keywords defaults')

def verify_matching_signatures(implementation, dispatcher):
    """Verify that a dispatcher function has the right signature."""
    implementation_spec = ArgSpec(*getargspec(implementation))
    print("implementation_spec", implementation_spec)
    dispatcher_spec = ArgSpec(*getargspec(dispatcher))
    print("dispatcher_spec", dispatcher_spec)

    if (implementation_spec.args != dispatcher_spec.args or
            implementation_spec.varargs != dispatcher_spec.varargs or
            implementation_spec.keywords != dispatcher_spec.keywords or
            (bool(implementation_spec.defaults) !=
             bool(dispatcher_spec.defaults)) or
            (implementation_spec.defaults is not None and
             len(implementation_spec.defaults) !=
             len(dispatcher_spec.defaults))):
        raise RuntimeError('implementation and dispatcher for %s have '
                           'different function signatures' % implementation)

    if implementation_spec.defaults is not None:
        if dispatcher_spec.defaults != (None,) * len(dispatcher_spec.defaults):
            raise RuntimeError('dispatcher functions can only use None for '
                               'default argument values')


import textwrap
import functools

_wrapped_func_source = textwrap.dedent("""
    @functools.wraps(implementation)
    def {name}(*args, **kwargs):
        relevant_args = dispatcher(*args, **kwargs)
        return implement_torch_function(
            implementation, {name}, relevant_args, args, kwargs)
    """)

TORCH_FUNCTION_ENABLED = True

def torch_function_dispatch(dispatcher, module=None, verify=True,
                            docs_from_dispatcher=False):
    """Decorator for adding dispatch with the __torch_function__ protocol.
    See NEP-18 for example usage.
    Parameters
    ----------
    dispatcher : callable
        Function that when called like ``dispatcher(*args, **kwargs)`` with
        arguments from the NumPy function call returns an iterable of
        array-like arguments to check for ``__torch_function__``.
    module : str, optional
        __module__ attribute to set on new function, e.g., ``module='numpy'``.
        By default, module is copied from the decorated function.
    verify : bool, optional
        If True, verify the that the signature of the dispatcher and decorated
        function signatures match exactly: all required and optional arguments
        should appear in order with the same names, but the default values for
        all optional arguments should be ``None``. Only disable verification
        if the dispatcher's signature needs to deviate for some particular
        reason, e.g., because the function has a signature like
        ``func(*args, **kwargs)``.
    docs_from_dispatcher : bool, optional
        If True, copy docs from the dispatcher function onto the dispatched
        function, rather than from the implementation. This is useful for
        functions defined in C, which otherwise don't have docstrings.
    Returns
    -------dispatcher
    Function suitable for decorating the implementation of a NumPy function.
    """

    if not TORCH_FUNCTION_ENABLED:
        def decorator(implementation):
            if docs_from_dispatcher:
                add_docstring(implementation, dispatcher.__doc__)
            if module is not None:
                implementation.__module__ = module
            return implementation
        return decorator

    def decorator(implementation):
        if verify:
            verify_matching_signatures(implementation, dispatcher)

        if docs_from_dispatcher:
            add_docstring(implementation, dispatcher.__doc__)

        # Equivalently, we could define this function directly instead of using
        # exec. This version has the advantage of giving the helper function a
        # more interpettable name. Otherwise, the original function does not
        # show up at all in many cases, e.g., if it's written in C or if the
        # dispatcher gets an invalid keyword argument.
        source = _wrapped_func_source.format(name=implementation.__name__)
        print("===========source================")
        print(source)

        source_object = compile(
            source, filename='<__torch_function__ internals>', mode='exec')
        scope = {
            'implementation': implementation,
            'dispatcher': dispatcher,
            'functools': functools,
			'implement_torch_function': implement_torch_function,
        }
        exec(source_object, scope)

        public_api = scope[implementation.__name__]

        if module is not None:
            public_api.__module__ = module

        public_api._implementation = implementation

        return public_api

    return decorator

def gemm_dispatcher(input, mat2, out = None):
    return (input, mat2, out)

def typename(o):
    if isinstance(o, torch.Tensor):
        return o.type()

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

@torch_function_dispatch(gemm_dispatcher)
def gemm(input, mat2, out=None):
    return torch.mm(input, mat2, out=None)


def is_tensor(obj):
    r"""Returns True if `obj` is a PyTorch tensor.

    Args:
        obj (Object): Object to test
    """
    return isinstance(obj, torch.Tensor)


def is_storage(obj):
    r"""Returns True if `obj` is a PyTorch storage object.

    Args:
        obj (Object): Object to test
    """
    return type(obj) in _storage_classes


def set_default_tensor_type(t):
    r"""Sets the default ``torch.Tensor`` type to floating point tensor type
    :attr:`t`. This type will also be used as default floating point type for
    type inference in :func:`torch.tensor`.

    The default floating point tensor type is initially ``torch.FloatTensor``.

    Args:
        t (type or string): the floating point tensor type or its name

    Example::

        >>> torch.tensor([1.2, 3]).dtype    # initial default for floating point is torch.float32
        torch.float32
        >>> torch.set_default_tensor_type(torch.DoubleTensor)
        >>> torch.tensor([1.2, 3]).dtype    # a new floating point tensor
        torch.float64

    """
    if isinstance(t, _string_classes):
        t = _import_dotted_name(t)
    _C._set_default_tensor_type(t)


def set_default_dtype(d):
    r"""Sets the default floating point dtype to :attr:`d`. This type will be
    used as default floating point type for type inference in
    :func:`torch.tensor`.

    The default floating point dtype is initially ``torch.float32``.

    Args:
        d (:class:`torch.dtype`): the floating point dtype to make the default

    Example::

        >>> torch.tensor([1.2, 3]).dtype           # initial default for floating point is torch.float32
        torch.float32
        >>> torch.set_default_dtype(torch.float64)
        >>> torch.tensor([1.2, 3]).dtype           # a new floating point tensor
        torch.float64

    """
    _C._set_default_dtype(d)

# If you edit these imports, please update torch/__init__.py.in as well
from .random import set_rng_state, get_rng_state, manual_seed, initial_seed, seed
from .serialization import save, load
from ._tensor_str import set_printoptions

################################################################################
# Define Storage and Tensor classes
################################################################################

from .tensor import Tensor
from .storage import _StorageBase


class DoubleStorage(_C.DoubleStorageBase, _StorageBase):
    pass


class FloatStorage(_C.FloatStorageBase, _StorageBase):
    pass


class HalfStorage(_C.HalfStorageBase, _StorageBase):
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


class BoolStorage(_C.BoolStorageBase, _StorageBase):
    pass


class BFloat16Storage(_C.BFloat16StorageBase, _StorageBase):
    pass


class QUInt8Storage(_C.QUInt8StorageBase, _StorageBase):
    pass

class QInt8Storage(_C.QInt8StorageBase, _StorageBase):
    pass

class QInt32Storage(_C.QInt32StorageBase, _StorageBase):
    pass


_storage_classes = {
    DoubleStorage, FloatStorage, LongStorage, IntStorage, ShortStorage,
    CharStorage, ByteStorage, HalfStorage, BoolStorage, QUInt8Storage, QInt8Storage,
    QInt32Storage, BFloat16Storage
}

# The _tensor_classes set is initialized by the call to _C._initialize_tensor_type_bindings()
_tensor_classes = set()


################################################################################
# Initialize extension
################################################################################

def manager_path():
    if platform.system() == 'Windows':
        return b""
    path = get_file_path('torch', 'bin', 'torch_shm_manager')
    prepare_multiprocessing_environment(get_file_path('torch'))
    if not os.path.exists(path):
        raise RuntimeError("Unable to find torch_shm_manager at " + path)
    return path.encode('utf-8')


# Shared memory manager needs to know the exact location of manager executable
_C._initExtension(manager_path())
del manager_path

for name in dir(_C._VariableFunctions):
    if name.startswith('__'):
        continue
    globals()[name] = getattr(_C._VariableFunctions, name)

################################################################################
# Import interface functions defined in Python
################################################################################

# needs to be after the above ATen bindings so we can overwrite from Python side
from .functional import *


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
del BoolStorageBase
del QUInt8StorageBase
del BFloat16StorageBase

################################################################################
# Import most common subpackages
################################################################################

import torch.cuda
import torch.autograd
from torch.autograd import no_grad, enable_grad, set_grad_enabled  # noqa: F401
import torch.nn
import torch.nn._intrinsic
import torch.nn.quantized
import torch.optim
import torch.multiprocessing
import torch.sparse
import torch.utils.backcompat
import torch.onnx
import torch.jit
import torch.hub
import torch.random
import torch.distributions
import torch.testing
import torch.backends.cuda
import torch.backends.mkl
import torch.backends.openmp
import torch.utils.data
import torch.__config__
import torch.__future__

_C._init_names(list(torch._storage_classes))

# attach docstrings to torch and tensor functions
from . import _torch_docs, _tensor_docs, _storage_docs
del _torch_docs, _tensor_docs, _storage_docs


def compiled_with_cxx11_abi():
    r"""Returns whether PyTorch was built with _GLIBCXX_USE_CXX11_ABI=1"""
    return _C._GLIBCXX_USE_CXX11_ABI


# Import the ops "namespace"
from torch._ops import ops  # noqa: F401
from torch._classes import classes  # noqa: F401

# Import the quasi random sampler
import torch.quasirandom
