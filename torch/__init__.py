
r"""
The torch package contains data structures for multi-dimensional
tensors and defines mathematical operations over these tensors.
Additionally, it provides many utilities for efficient serialization of
Tensors and arbitrary types, and other useful utilities.

It has a CUDA counterpart, that enables you to run your tensor computations
on an NVIDIA GPU with compute capability >= 3.0.
"""

import math
import os
import sys
import platform
import textwrap
import ctypes
import inspect

# multipy/deploy is setting this import before importing torch, this is the most
# reliable way we have to detect if we're running within deploy.
# https://github.com/pytorch/multipy/blob/d60f34ad38c371e441fe7ffdb77a3c3dda5a5d19/multipy/runtime/interpreter/interpreter_impl.cpp#L134-L137
def _running_with_deploy():
    return sys.modules.get("torch._meta_registrations", None) is object

from ._utils import _import_dotted_name, classproperty
from ._utils import _functionalize_sync as _sync
from ._utils_internal import get_file_path, prepare_multiprocessing_environment, \
    USE_RTLD_GLOBAL_WITH_LIBTORCH, USE_GLOBAL_DEPS

# TODO(torch_deploy) figure out how to freeze version.py in fbcode build
if _running_with_deploy():
    __version__ = "torch-deploy-1.8"
else:
    from .torch_version import __version__ as __version__

from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, TYPE_CHECKING, Union, List
import builtins

__all__ = [
    'typename', 'is_tensor', 'is_storage',
    'set_default_tensor_type', 'set_default_device',
    'set_rng_state', 'get_rng_state', 'manual_seed', 'initial_seed', 'seed',
    'save', 'load', 'set_printoptions', 'chunk', 'split', 'stack', 'matmul',
    'no_grad', 'enable_grad', 'rand', 'randn', 'inference_mode',
    'DoubleStorage', 'FloatStorage', 'LongStorage', 'IntStorage',
    'ShortStorage', 'CharStorage', 'ByteStorage', 'BoolStorage',
    'TypedStorage', 'UntypedStorage',
    'DoubleTensor', 'FloatTensor', 'LongTensor', 'IntTensor',
    'ShortTensor', 'CharTensor', 'ByteTensor', 'BoolTensor', 'Tensor',
    'lobpcg', 'use_deterministic_algorithms',
    'are_deterministic_algorithms_enabled',
    'is_deterministic_algorithms_warn_only_enabled',
    'set_deterministic_debug_mode', 'get_deterministic_debug_mode',
    'set_float32_matmul_precision', 'get_float32_matmul_precision',
    'set_warn_always', 'is_warn_always_enabled', 'SymInt', 'SymFloat',
    'SymBool', 'sym_not',
    'sym_int', 'sym_float', 'sym_max', 'sym_min', 'compile', 'vmap',
    'export', 'autocast', 'cond',
]

################################################################################
# Load the extension module
################################################################################

if sys.platform == 'win32':
    pfiles_path = os.getenv('ProgramFiles', 'C:\\Program Files')
    py_dll_path = os.path.join(sys.exec_prefix, 'Library', 'bin')
    th_dll_path = os.path.join(os.path.dirname(__file__), 'lib')

    # When users create a virtualenv that inherits the base environment,
    # we will need to add the corresponding library directory into
    # DLL search directories. Otherwise, it will rely on `PATH` which
    # is dependent on user settings.
    if sys.exec_prefix != sys.base_exec_prefix:
        base_py_dll_path = os.path.join(sys.base_exec_prefix, 'Library', 'bin')
    else:
        base_py_dll_path = ''

    dll_paths = list(filter(os.path.exists, [th_dll_path, py_dll_path, base_py_dll_path]))

    if all(not os.path.exists(os.path.join(p, 'nvToolsExt64_1.dll')) for p in dll_paths):
        nvtoolsext_dll_path = os.path.join(
            os.getenv('NVTOOLSEXT_PATH', os.path.join(pfiles_path, 'NVIDIA Corporation', 'NvToolsExt')), 'bin', 'x64')
    else:
        nvtoolsext_dll_path = ''

    from .version import cuda as cuda_version
    import glob
    if cuda_version and all(not glob.glob(os.path.join(p, 'cudart64*.dll')) for p in dll_paths):
        cuda_version_1 = cuda_version.replace('.', '_')
        cuda_path_var = 'CUDA_PATH_V' + cuda_version_1
        default_path = os.path.join(pfiles_path, 'NVIDIA GPU Computing Toolkit', 'CUDA', 'v' + cuda_version)
        cuda_path = os.path.join(os.getenv(cuda_path_var, default_path), 'bin')
    else:
        cuda_path = ''

    dll_paths.extend(filter(os.path.exists, [nvtoolsext_dll_path, cuda_path]))

    kernel32 = ctypes.WinDLL('kernel32.dll', use_last_error=True)
    with_load_library_flags = hasattr(kernel32, 'AddDllDirectory')
    prev_error_mode = kernel32.SetErrorMode(0x0001)

    kernel32.LoadLibraryW.restype = ctypes.c_void_p
    if with_load_library_flags:
        kernel32.LoadLibraryExW.restype = ctypes.c_void_p

    for dll_path in dll_paths:
        os.add_dll_directory(dll_path)

    try:
        ctypes.CDLL('vcruntime140.dll')
        ctypes.CDLL('msvcp140.dll')
        ctypes.CDLL('vcruntime140_1.dll')
    except OSError:
        print('''Microsoft Visual C++ Redistributable is not installed, this may lead to the DLL load failure.
                 It can be downloaded at https://aka.ms/vs/16/release/vc_redist.x64.exe''')

    dlls = glob.glob(os.path.join(th_dll_path, '*.dll'))
    path_patched = False
    for dll in dlls:
        is_loaded = False
        if with_load_library_flags:
            res = kernel32.LoadLibraryExW(dll, None, 0x00001100)
            last_error = ctypes.get_last_error()
            if res is None and last_error != 126:
                err = ctypes.WinError(last_error)
                err.strerror += f' Error loading "{dll}" or one of its dependencies.'
                raise err
            elif res is not None:
                is_loaded = True
        if not is_loaded:
            if not path_patched:
                os.environ['PATH'] = ';'.join(dll_paths + [os.environ['PATH']])
                path_patched = True
            res = kernel32.LoadLibraryW(dll)
            if res is None:
                err = ctypes.WinError(ctypes.get_last_error())
                err.strerror += f' Error loading "{dll}" or one of its dependencies.'
                raise err

    kernel32.SetErrorMode(prev_error_mode)


def _preload_cuda_deps(lib_folder, lib_name):
    """Preloads cuda deps if they could not be found otherwise."""
    # Should only be called on Linux if default path resolution have failed
    assert platform.system() == 'Linux', 'Should only be called on Linux'
    import glob
    lib_path = None
    for path in sys.path:
        nvidia_path = os.path.join(path, 'nvidia')
        if not os.path.exists(nvidia_path):
            continue
        candidate_lib_paths = glob.glob(os.path.join(nvidia_path, lib_folder, 'lib', lib_name))
        if candidate_lib_paths and not lib_path:
            lib_path = candidate_lib_paths[0]
        if lib_path:
            break
    if not lib_path:
        raise ValueError(f"{lib_name} not found in the system path {sys.path}")
    ctypes.CDLL(lib_path)


# See Note [Global dependencies]
def _load_global_deps() -> None:
    if _running_with_deploy() or platform.system() == 'Windows':
        return

    lib_name = 'libtorch_global_deps' + ('.dylib' if platform.system() == 'Darwin' else '.so')
    here = os.path.abspath(__file__)
    lib_path = os.path.join(os.path.dirname(here), 'lib', lib_name)

    try:
        ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
    except OSError as err:
        # Can only happen for wheel with cuda libs as PYPI deps
        # As PyTorch is not purelib, but nvidia-*-cu12 is
        cuda_libs: Dict[str, str] = {
            'cublas': 'libcublas.so.*[0-9]',
            'cudnn': 'libcudnn.so.*[0-9]',
            'cuda_nvrtc': 'libnvrtc.so.*[0-9]',
            'cuda_runtime': 'libcudart.so.*[0-9]',
            'cuda_cupti': 'libcupti.so.*[0-9]',
            'cufft': 'libcufft.so.*[0-9]',
            'curand': 'libcurand.so.*[0-9]',
            'cusolver': 'libcusolver.so.*[0-9]',
            'cusparse': 'libcusparse.so.*[0-9]',
            'nccl': 'libnccl.so.*[0-9]',
            'nvtx': 'libnvToolsExt.so.*[0-9]',
        }
        is_cuda_lib_err = [lib for lib in cuda_libs.values() if(lib.split('.')[0] in err.args[0])]
        if not is_cuda_lib_err:
            raise err
        for lib_folder, lib_name in cuda_libs.items():
            _preload_cuda_deps(lib_folder, lib_name)
        ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)


if (USE_RTLD_GLOBAL_WITH_LIBTORCH or os.getenv('TORCH_USE_RTLD_GLOBAL')) and \
        (_running_with_deploy() or platform.system() != 'Windows'):
    # Do it the hard way.  You might want to load libtorch with RTLD_GLOBAL in a
    # few circumstances:
    #
    #   1. You're in a build environment (e.g., fbcode) where
    #      libtorch_global_deps is not available, but you still need
    #      to get mkl to link in with RTLD_GLOBAL or it will just
    #      not work.
    #
    #   2. You're trying to run PyTorch under UBSAN and you need
    #      to ensure that only one copy of libtorch is loaded, so
    #      vptr checks work properly
    #
    # If you're using this setting, you must verify that all the libraries
    # you load consistently use the same libstdc++, or you may have
    # mysterious segfaults.
    #
    old_flags = sys.getdlopenflags()
    sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_LAZY)
    from torch._C import *  # noqa: F403
    sys.setdlopenflags(old_flags)
    del old_flags

else:
    # Easy way.  You want this most of the time, because it will prevent
    # C++ symbols from libtorch clobbering C++ symbols from other
    # libraries, leading to mysterious segfaults.
    #
    # If building in an environment where libtorch_global_deps isn't available
    # like parts of fbsource, but where RTLD_GLOBAL causes segfaults, you will
    # want USE_RTLD_GLOBAL_WITH_LIBTORCH = False and USE_GLOBAL_DEPS = False
    #
    # See Note [Global dependencies]
    if USE_GLOBAL_DEPS:
        _load_global_deps()
    from torch._C import *  # noqa: F403

# Appease the type checker; ordinarily this binding is inserted by the
# torch._C module initialization code in C
if TYPE_CHECKING:
    import torch._C as _C

class SymInt:
    """
    Like an int (including magic methods), but redirects all operations on the
    wrapped node. This is used in particular to symbolically record operations
    in the symbolic shape workflow.
    """

    def __init__(self, node):
        # This field MUST be named node; C++ binding code assumes that this
        # class has a field named node that stores SymNode
        self.node = node

    def __bool__(self):
        return builtins.bool(self != 0)

    def __int__(self):
        return self.node.int_()

    def __index__(self):
        return self.node.int_()

    # Magic methods installed by torch.fx.experimental.symbolic_shapes

    def __eq__(self, other: object) -> builtins.bool:
        raise AssertionError("type stub not overridden")

    def __lt__(self, other) -> builtins.bool:
        raise AssertionError("type stub not overridden")

    def __gt__(self, other) -> builtins.bool:
        raise AssertionError("type stub not overridden")

    def __le__(self, other) -> builtins.bool:
        raise AssertionError("type stub not overridden")

    def __ge__(self, other) -> builtins.bool:
        raise AssertionError("type stub not overridden")

    def __sym_max__(self, other):
        raise AssertionError("type stub not overridden")

    def __sym_min__(self, other):
        raise AssertionError("type stub not overridden")

    def __sym_float__(self):
        raise AssertionError("type stub not overridden")

    def __repr__(self):
        return str(self.node)

    def __hash__(self) -> builtins.int:
        ret = self.node.singleton_int()
        if ret is not None:
            return hash(ret)
        else:
            # We could support constant SymInts as well, but not doing it for now
            raise TypeError("unhashable type: non-singleton SymInt")

class SymFloat:
    """
    Like an float (including magic methods), but redirects all operations on the
    wrapped node. This is used in particular to symbolically record operations
    in the symbolic shape workflow.
    """

    def __init__(self, node):
        # This field MUST be named node; C++ binding code assumes that this
        # class has a field named node that stores SymNode
        self.node = node

    def __bool__(self):
        return self.node.bool_()

    # Magic methods installed by torch.fx.experimental.symbolic_shapes

    def __eq__(self, other: object) -> builtins.bool:
        raise AssertionError("type stub not overridden")

    def __lt__(self, other) -> builtins.bool:
        raise AssertionError("type stub not overridden")

    def __gt__(self, other) -> builtins.bool:
        raise AssertionError("type stub not overridden")

    def __le__(self, other) -> builtins.bool:
        raise AssertionError("type stub not overridden")

    def __ge__(self, other) -> builtins.bool:
        raise AssertionError("type stub not overridden")

    def __sym_max__(self, other):
        raise AssertionError("type stub not overridden")

    def __sym_min__(self, other):
        raise AssertionError("type stub not overridden")

    def __sym_int__(self):
        raise AssertionError("type stub not overridden")

    def __repr__(self):
        return self.node.str()

class SymBool:
    """
    Like an bool (including magic methods), but redirects all operations on the
    wrapped node. This is used in particular to symbolically record operations
    in the symbolic shape workflow.

    Unlike regular bools, regular boolean operators will force extra guards instead
    of symbolically evaluate.  Use the bitwise operators instead to handle this.
    """

    def __init__(self, node):
        # This field MUST be named node; C++ binding code assumes that this
        # class has a field named node that stores SymNode
        self.node = node

    def __bool__(self):
        return self.node.bool_()

    def __int__(self):
        return builtins.int(self.node.bool_())

    # Magic methods installed by torch.fx.experimental.symbolic_shapes
    def __and__(self, other) -> "SymBool":
        raise AssertionError("type stub not overridden")

    def __or__(self, other) -> "SymBool":
        raise AssertionError("type stub not overridden")

    # We very carefully define __sym_not__, and not a number of other
    # plausible alternatives:
    #
    #   - We do not override __not__ because this is not a real magic
    #     method; you cannot override the meaning of the not builtin in
    #     Python.  We use the name 'sym_not' to clarify that in user code you
    #     cannot use the builtin not or operator.not_ or operator.__not__ and
    #     hit this magic method; you must use our custom sym_not operator.
    #
    #   - We do not override the __invert__ method because SymBool is
    #     meant to be usable in situations where bool is expected.  However,
    #     bitwise negation ~a does the wrong thing with booleans (because
    #     bool is a subclass of int, so ~1 = -2 which is not falseish.)
    #     This would be a giant footgun, so we get around it by defining
    #     our own operator.  Note that bitwise and/or do the right thing,
    #     so we reuse the conventional operators there for readability.
    #
    def __sym_not__(self) -> "SymBool":
        raise AssertionError("type stub not overridden")

    def __eq__(self, other) -> builtins.bool:
        raise AssertionError("type stub not overridden")

    def __repr__(self):
        return str(self.node)

    def __hash__(self):
        if self.node.is_constant():
            return hash(self.node.bool_())
        else:
            raise TypeError("unhashable type: SymBool")

def sym_not(a):
    r""" SymInt-aware utility for logical negation.

    Args:
        a (SymBool or bool): Object to negate
    """
    if hasattr(a, '__sym_not__'):
        return a.__sym_not__()
    return not a

def sym_float(a):
    r""" SymInt-aware utility for float casting.

    Args:
        a (SymInt, SymFloat, or object): Object to cast
    """
    if isinstance(a, SymFloat):
        return a
    elif hasattr(a, '__sym_float__'):
        return a.__sym_float__()
    return py_float(a)  # type: ignore[operator]


def sym_int(a):
    r""" SymInt-aware utility for int casting.

    Args:
        a (SymInt, SymFloat, or object): Object to cast
    """
    if isinstance(a, SymInt):
        return a
    elif isinstance(a, SymFloat):
        return math.floor(a) if a >= 0 else math.ceil(a)  # type: ignore[arg-type, call-overload]
    return py_int(a)  # type: ignore[operator]

def sym_max(a, b):
    """ SymInt-aware utility for max()."""
    if isinstance(a, (SymInt, SymFloat)):
        return a.__sym_max__(b)
    elif isinstance(b, (SymInt, SymFloat)):
        # NB: If you actually care about preserving output type exactly
        # if you do something like max(0, 0.0), it is NOT sound to treat
        # min/max as commutative
        return b.__sym_max__(a)
    return builtins.max(a, b)  # type: ignore[operator]

def sym_min(a, b):
    """ SymInt-aware utility for max()."""
    if isinstance(a, (SymInt, SymFloat)):
        return a.__sym_min__(b)
    elif isinstance(b, (SymInt, SymFloat)):
        return b.__sym_min__(a)
    return builtins.min(a, b)  # type: ignore[operator]

# Check to see if we can load C extensions, and if not provide some guidance
# on what the problem might be.
try:
    # _initExtension is chosen (arbitrarily) as a sentinel.
    from torch._C import _initExtension
except ImportError:
    import torch._C as _C_for_compiled_check

    # The __file__ check only works for Python 3.7 and above.
    if _C_for_compiled_check.__file__ is None:
        raise ImportError(textwrap.dedent('''
            Failed to load PyTorch C extensions:
                It appears that PyTorch has loaded the `torch/_C` folder
                of the PyTorch repository rather than the C extensions which
                are expected in the `torch._C` namespace. This can occur when
                using the `install` workflow. e.g.
                    $ python setup.py install && python -c "import torch"

                This error can generally be solved using the `develop` workflow
                    $ python setup.py develop && python -c "import torch"  # This should succeed
                or by running Python from a different directory.
            ''').strip()) from None
    raise  # If __file__ is not None the cause is unknown, so just re-raise.

for name in dir(_C):
    if name[0] != '_' and not name.endswith('Base'):
        __all__.append(name)
        obj = getattr(_C, name)
        if (isinstance(obj, Callable) or inspect.isclass(obj)):  # type: ignore[arg-type]
            if (obj.__module__ != 'torch'):
                # TODO: fix their module from C++ side
                if name not in ['DisableTorchFunctionSubclass', 'DisableTorchFunction', 'Generator']:
                    obj.__module__ = 'torch'
    elif name == 'TensorBase':
        # issue 109438 / pr 109940. Prevent TensorBase from being copied into torch.
        delattr(sys.modules[__name__], name)

if not TYPE_CHECKING:
    # issue 38137 and python issue 43367. Submodules of a C extension are
    # non-standard, and attributes of those submodules cannot be pickled since
    # pickle expect to be able to import them as "from _C.sub import attr"
    # which fails with "_C is not a package
    for attr in dir(_C):
        candidate = getattr(_C, attr)
        if type(candidate) is type(_C):
            # submodule
            if f'torch._C.{attr}' not in sys.modules:
                sys.modules[f'torch._C.{attr}'] = candidate


################################################################################
# Define basic utilities
################################################################################


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


def is_tensor(obj):
    r"""Returns True if `obj` is a PyTorch tensor.

    Note that this function is simply doing ``isinstance(obj, Tensor)``.
    Using that ``isinstance`` check is better for typechecking with mypy,
    and more explicit - so it's recommended to use that instead of
    ``is_tensor``.

    Args:
        obj (Object): Object to test
    Example::

        >>> x = torch.tensor([1, 2, 3])
        >>> torch.is_tensor(x)
        True

    """
    return isinstance(obj, torch.Tensor)


def is_storage(obj):
    r"""Returns True if `obj` is a PyTorch storage object.

    Args:
        obj (Object): Object to test
    """
    return type(obj) in _storage_classes


_GLOBAL_DEVICE_CONTEXT = None

def set_default_device(device):
    """Sets the default ``torch.Tensor`` to be allocated on ``device``.  This
    does not affect factory function calls which are called with an explicit
    ``device`` argument.  Factory calls will be performed as if they
    were passed ``device`` as an argument.

    To only temporarily change the default device instead of setting it
    globally, use ``with torch.device(device):`` instead.

    The default device is initially ``cpu``.  If you set the default tensor
    device to another device (e.g., ``cuda``) without a device index, tensors
    will be allocated on whatever the current device for the device type,
    even after :func:`torch.cuda.set_device` is called.

    .. warning::

        This function imposes a slight performance cost on every Python
        call to the torch API (not just factory functions).  If this
        is causing problems for you, please comment on
        https://github.com/pytorch/pytorch/issues/92701

    Args:
        device (device or string): the device to set as default

    Example::

        >>> # xdoctest: +SKIP("requires cuda, changes global state")
        >>> torch.tensor([1.2, 3]).device
        device(type='cpu')
        >>> torch.set_default_device('cuda')  # current device is 0
        >>> torch.tensor([1.2, 3]).device
        device(type='cuda', index=0)
        >>> torch.set_default_device('cuda:1')
        >>> torch.tensor([1.2, 3]).device
        device(type='cuda', index=1)

    """
    global _GLOBAL_DEVICE_CONTEXT
    if _GLOBAL_DEVICE_CONTEXT is not None:
        _GLOBAL_DEVICE_CONTEXT.__exit__(None, None, None)
    if device is None:
        _GLOBAL_DEVICE_CONTEXT = None
        return
    from torch.utils._device import DeviceContext
    _GLOBAL_DEVICE_CONTEXT = DeviceContext(device)
    _GLOBAL_DEVICE_CONTEXT.__enter__()


def set_default_tensor_type(t):
    r"""Sets the default ``torch.Tensor`` type to floating point tensor type
    ``t``. This type will also be used as default floating point type for
    type inference in :func:`torch.tensor`.

    The default floating point tensor type is initially ``torch.FloatTensor``.

    Args:
        t (type or string): the floating point tensor type or its name

    Example::

        >>> # xdoctest: +SKIP("Other tests may have changed the default type. Can we reset it?")
        >>> torch.tensor([1.2, 3]).dtype    # initial default for floating point is torch.float32
        torch.float32
        >>> torch.set_default_tensor_type(torch.DoubleTensor)
        >>> torch.tensor([1.2, 3]).dtype    # a new floating point tensor
        torch.float64

    """
    if isinstance(t, str):
        t = _import_dotted_name(t)
    _C._set_default_tensor_type(t)


def set_default_dtype(d):
    r"""

    Sets the default floating point dtype to :attr:`d`. Supports torch.float32
    and torch.float64 as inputs. Other dtypes may be accepted without complaint
    but are not supported and are unlikely to work as expected.

    When PyTorch is initialized its default floating point dtype is torch.float32,
    and the intent of set_default_dtype(torch.float64) is to facilitate NumPy-like
    type inference. The default floating point dtype is used to:

    1. Implicitly determine the default complex dtype. When the default floating point
       type is float32 the default complex dtype is complex64, and when the default
       floating point type is float64 the default complex type is complex128.
    2. Infer the dtype for tensors constructed using Python floats or complex Python
       numbers. See examples below.
    3. Determine the result of type promotion between bool and integer tensors and
       Python floats and complex Python numbers.

    Args:
        d (:class:`torch.dtype`): the floating point dtype to make the default.
                                  Either torch.float32 or torch.float64.

    Example:
        >>> # xdoctest: +SKIP("Other tests may have changed the default type. Can we reset it?")
        >>> # initial default for floating point is torch.float32
        >>> # Python floats are interpreted as float32
        >>> torch.tensor([1.2, 3]).dtype
        torch.float32
        >>> # initial default for floating point is torch.complex64
        >>> # Complex Python numbers are interpreted as complex64
        >>> torch.tensor([1.2, 3j]).dtype
        torch.complex64

        >>> torch.set_default_dtype(torch.float64)

        >>> # Python floats are now interpreted as float64
        >>> torch.tensor([1.2, 3]).dtype    # a new floating point tensor
        torch.float64
        >>> # Complex Python numbers are now interpreted as complex128
        >>> torch.tensor([1.2, 3j]).dtype   # a new complex tensor
        torch.complex128

    """
    _C._set_default_dtype(d)

def use_deterministic_algorithms(mode: builtins.bool, *, warn_only: builtins.bool = False) -> None:
    r""" Sets whether PyTorch operations must use "deterministic"
    algorithms. That is, algorithms which, given the same input, and when
    run on the same software and hardware, always produce the same output.
    When enabled, operations will use deterministic algorithms when available,
    and if only nondeterministic algorithms are available they will throw a
    :class:`RuntimeError` when called.

    .. note:: This setting alone is not always enough to make an application
        reproducible. Refer to :ref:`reproducibility` for more information.

    .. note:: :func:`torch.set_deterministic_debug_mode` offers an alternative
        interface for this feature.

    The following normally-nondeterministic operations will act
    deterministically when ``mode=True``:

        * :class:`torch.nn.Conv1d` when called on CUDA tensor
        * :class:`torch.nn.Conv2d` when called on CUDA tensor
        * :class:`torch.nn.Conv3d` when called on CUDA tensor
        * :class:`torch.nn.ConvTranspose1d` when called on CUDA tensor
        * :class:`torch.nn.ConvTranspose2d` when called on CUDA tensor
        * :class:`torch.nn.ConvTranspose3d` when called on CUDA tensor
        * :func:`torch.bmm` when called on sparse-dense CUDA tensors
        * :func:`torch.Tensor.__getitem__` when attempting to differentiate a CPU tensor
          and the index is a list of tensors
        * :func:`torch.Tensor.index_put` with ``accumulate=False``
        * :func:`torch.Tensor.index_put` with ``accumulate=True`` when called on a CPU
          tensor
        * :func:`torch.Tensor.put_` with ``accumulate=True`` when called on a CPU
          tensor
        * :func:`torch.Tensor.scatter_add_` when called on a CUDA tensor
        * :func:`torch.gather` when called on a CUDA tensor that requires grad
        * :func:`torch.index_add` when called on CUDA tensor
        * :func:`torch.index_select` when attempting to differentiate a CUDA tensor
        * :func:`torch.repeat_interleave` when attempting to differentiate a CUDA tensor
        * :func:`torch.Tensor.index_copy` when called on a CPU or CUDA tensor
        * :func:`torch.Tensor.scatter` when `src` type is Tensor and called on CUDA tensor
        * :func:`torch.Tensor.scatter_reduce` when ``reduce='sum'`` or ``reduce='mean'`` and called on CUDA tensor
        * :func:`torch.Tensor.resize_`, when called with a tensor that is not
          quantized, sets new elements to a known value.  Floating point or
          complex values are set to NaN. Integer values are set to the maximum
          value.
        * :func:`torch.empty`, :func:`torch.empty_like`, :func:`torch.empty_strided`,
          and :func:`torch.empty_permuted` will fill the output tensor with a known
          value. Floating point or complex dtype tensors are filled with NaN. Integer
          dtype tensors are filled with the maximum value.

    The following normally-nondeterministic operations will throw a
    :class:`RuntimeError` when ``mode=True``:

        * :class:`torch.nn.AvgPool3d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.AdaptiveAvgPool2d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.AdaptiveAvgPool3d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.MaxPool3d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.AdaptiveMaxPool2d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.FractionalMaxPool2d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.FractionalMaxPool3d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.MaxUnpool1d`
        * :class:`torch.nn.MaxUnpool2d`
        * :class:`torch.nn.MaxUnpool3d`
        * :func:`torch.nn.functional.interpolate` when attempting to differentiate a CUDA tensor
          and one of the following modes is used:

          - ``linear``
          - ``bilinear``
          - ``bicubic``
          - ``trilinear``

        * :class:`torch.nn.ReflectionPad1d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.ReflectionPad2d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.ReflectionPad3d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.ReplicationPad1d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.ReplicationPad2d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.ReplicationPad3d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.NLLLoss` when called on a CUDA tensor
        * :class:`torch.nn.CTCLoss` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.EmbeddingBag` when attempting to differentiate a CUDA tensor when
          ``mode='max'``
        * :func:`torch.Tensor.put_` when ``accumulate=False``
        * :func:`torch.Tensor.put_` when ``accumulate=True`` and called on a CUDA tensor
        * :func:`torch.histc` when called on a CUDA tensor
        * :func:`torch.bincount` when called on a CUDA tensor and ``weights``
          tensor is given
        * :func:`torch.kthvalue` with called on a CUDA tensor
        * :func:`torch.median` with indices output when called on a CUDA tensor
        * :func:`torch.nn.functional.grid_sample` when attempting to differentiate a CUDA tensor
        * :func:`torch.cumsum` when called on a CUDA tensor when dtype is floating point or complex
        * :func:`torch.Tensor.scatter_reduce` when ``reduce='prod'`` and called on CUDA tensor
        * :func:`torch.Tensor.resize_` when called with a quantized tensor

    A handful of CUDA operations are nondeterministic if the CUDA version is
    10.2 or greater, unless the environment variable ``CUBLAS_WORKSPACE_CONFIG=:4096:8``
    or ``CUBLAS_WORKSPACE_CONFIG=:16:8`` is set. See the CUDA documentation for more
    details: `<https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility>`_
    If one of these environment variable configurations is not set, a :class:`RuntimeError`
    will be raised from these operations when called with CUDA tensors:

        * :func:`torch.mm`
        * :func:`torch.mv`
        * :func:`torch.bmm`

    Note that deterministic operations tend to have worse performance than
    nondeterministic operations.

    .. note::

        This flag does not detect or prevent nondeterministic behavior caused
        by calling an inplace operation on a tensor with an internal memory
        overlap or by giving such a tensor as the :attr:`out` argument for an
        operation. In these cases, multiple writes of different data may target
        a single memory location, and the order of writes is not guaranteed.

    Args:
        mode (:class:`bool`): If True, makes potentially nondeterministic
            operations switch to a deterministic algorithm or throw a runtime
            error. If False, allows nondeterministic operations.

    Keyword args:
        warn_only (:class:`bool`, optional): If True, operations that do not
            have a deterministic implementation will throw a warning instead of
            an error. Default: ``False``

    Example::

        >>> # xdoctest: +SKIP
        >>> torch.use_deterministic_algorithms(True)

        # Forward mode nondeterministic error
        >>> torch.randn(10, device='cuda').kthvalue(1)
        ...
        RuntimeError: kthvalue CUDA does not have a deterministic implementation...

        # Backward mode nondeterministic error
        >>> torch.nn.AvgPool3d(1)(torch.randn(3, 4, 5, 6, requires_grad=True).cuda()).sum().backward()
        ...
        RuntimeError: avg_pool3d_backward_cuda does not have a deterministic implementation...
    """
    _C._set_deterministic_algorithms(mode, warn_only=warn_only)

def are_deterministic_algorithms_enabled() -> builtins.bool:
    r"""Returns True if the global deterministic flag is turned on. Refer to
    :func:`torch.use_deterministic_algorithms` documentation for more details.
    """
    return _C._get_deterministic_algorithms()

def is_deterministic_algorithms_warn_only_enabled() -> builtins.bool:
    r"""Returns True if the global deterministic flag is set to warn only.
    Refer to :func:`torch.use_deterministic_algorithms` documentation for more
    details.
    """
    return _C._get_deterministic_algorithms_warn_only()

def set_deterministic_debug_mode(debug_mode: Union[builtins.int, str]) -> None:
    r"""Sets the debug mode for deterministic operations.

    .. note:: This is an alternative interface for
        :func:`torch.use_deterministic_algorithms`. Refer to that function's
        documentation for details about affected operations.

    Args:
        debug_mode(str or int): If "default" or 0, don't error or warn on
            nondeterministic operations. If "warn" or 1, warn on
            nondeterministic operations. If "error" or 2, error on
            nondeterministic operations.
    """

    # NOTE: builtins.int is used here because int in this scope resolves
    # to torch.int
    if not isinstance(debug_mode, (builtins.int, str)):
        raise TypeError(f'debug_mode must be str or int, but got {type(debug_mode)}')

    if isinstance(debug_mode, str):
        if debug_mode == 'default':
            debug_mode = 0
        elif debug_mode == 'warn':
            debug_mode = 1
        elif debug_mode == 'error':
            debug_mode = 2
        else:
            raise RuntimeError(
                'invalid value of debug_mode, expected one of `default`, '
                f'`warn`, `error`, but got {debug_mode}')

    if debug_mode == 0:
        _C._set_deterministic_algorithms(False)
    elif debug_mode == 1:
        _C._set_deterministic_algorithms(True, warn_only=True)
    elif debug_mode == 2:
        _C._set_deterministic_algorithms(True)
    else:
        raise RuntimeError(
            'invalid value of debug_mode, expected 0, 1, or 2, '
            f'but got {debug_mode}')

def get_deterministic_debug_mode() -> builtins.int:
    r"""Returns the current value of the debug mode for deterministic
    operations. Refer to :func:`torch.set_deterministic_debug_mode`
    documentation for more details.
    """

    if _C._get_deterministic_algorithms():
        if _C._get_deterministic_algorithms_warn_only():
            return 1
        else:
            return 2
    else:
        return 0

def get_float32_matmul_precision() -> builtins.str:
    r"""Returns the current value of float32 matrix multiplication precision. Refer to
    :func:`torch.set_float32_matmul_precision` documentation for more details.
    """
    return _C._get_float32_matmul_precision()

def set_float32_matmul_precision(precision: str) -> None:
    r"""Sets the internal precision of float32 matrix multiplications.

    Running float32 matrix multiplications in lower precision may significantly increase
    performance, and in some programs the loss of precision has a negligible impact.

    Supports three settings:

        * "highest", float32 matrix multiplications use the float32 datatype (24 mantissa
          bits) for internal computations.
        * "high", float32 matrix multiplications either use the TensorFloat32 datatype (10
          mantissa bits) or treat each float32 number as the sum of two bfloat16 numbers
          (approximately 16 mantissa bits), if the appropriate fast matrix multiplication
          algorithms are available.  Otherwise float32 matrix multiplications are computed
          as if the precision is "highest".  See below for more information on the bfloat16
          approach.
        * "medium", float32 matrix multiplications use the bfloat16 datatype (8 mantissa
          bits) for internal computations, if a fast matrix multiplication algorithm
          using that datatype internally is available. Otherwise float32
          matrix multiplications are computed as if the precision is "high".

    When using "high" precision, float32 multiplications may use a bfloat16-based algorithm
    that is more complicated than simply truncating to some smaller number mantissa bits
    (e.g. 10 for TensorFloat32, 8 for bfloat16).  Refer to [Henry2019]_ for a complete
    description of this algorithm.  To briefly explain here, the first step is to realize
    that we can perfectly encode a single float32 number as the sum of three bfloat16
    numbers (because float32 has 24 mantissa bits while bfloat16 has 8, and both have the
    same number of exponent bits).  This means that the product of two float32 numbers can
    be exactly given by the sum of nine products of bfloat16 numbers.  We can then trade
    accuracy for speed by dropping some of these products.  The "high" precision algorithm
    specifically keeps only the three most significant products, which conveniently excludes
    all of the products involving the last 8 mantissa bits of either input.  This means that
    we can represent our inputs as the sum of two bfloat16 numbers rather than three.
    Because bfloat16 fused-multiply-add (FMA) instructions are typically >10x faster than
    float32 ones, it's faster to do three multiplications and 2 additions with bfloat16
    precision than it is to do a single multiplication with float32 precision.

    .. [Henry2019] http://arxiv.org/abs/1904.06376

    .. note::

        This does not change the output dtype of float32 matrix multiplications,
        it controls how the internal computation of the matrix multiplication is performed.

    .. note::

        This does not change the precision of convolution operations. Other flags,
        like `torch.backends.cudnn.allow_tf32`, may control the precision of convolution
        operations.

    .. note::

        This flag currently only affects one native device type: CUDA.
        If "high" or "medium" are set then the TensorFloat32 datatype will be used
        when computing float32 matrix multiplications, equivalent to setting
        `torch.backends.cuda.matmul.allow_tf32 = True`. When "highest" (the default)
        is set then the float32 datatype is used for internal computations, equivalent
        to setting `torch.backends.cuda.matmul.allow_tf32 = False`.

    Args:
        precision(str): can be set to "highest" (default), "high", or "medium" (see above).

    """
    _C._set_float32_matmul_precision(precision)

def set_warn_always(b: builtins.bool) -> None:
    r"""When this flag is False (default) then some PyTorch warnings may only
    appear once per process. This helps avoid excessive warning information.
    Setting it to True causes these warnings to always appear, which may be
    helpful when debugging.

    Args:
        b (:class:`bool`): If True, force warnings to always be emitted
                           If False, set to the default behaviour
    """
    _C._set_warnAlways(b)

def is_warn_always_enabled() -> builtins.bool:
    r"""Returns True if the global warn_always flag is turned on. Refer to
    :func:`torch.set_warn_always` documentation for more details.
    """
    return _C._get_warnAlways()

################################################################################
# Define error checking functions
################################################################################

# These error checking functions must be kept consistent with their C++
# equivalents. Their C++ equivalents are mentioned where applicable.

def _check_with(error_type, cond: Union[builtins.bool, SymBool], message: Callable[[], str]):  # noqa: F811
    if not isinstance(cond, (builtins.bool, torch.SymBool)):
        raise TypeError(f'cond must be a bool, but got {type(cond)}')

    if torch.fx.experimental.symbolic_shapes.expect_true(cond):
        return

    # error_type must be a subclass of Exception and not subclass of Warning
    assert issubclass(error_type, Exception) and not issubclass(error_type, Warning)

    if message is None:
        message_evaluated = (
            'Expected cond to be True, but got False. (Could this error '
            'message be improved? If so, please report an enhancement request '
            'to PyTorch.)')

    else:
        if not callable(message):
            raise TypeError('message must be a callable')

        message_evaluated = str(message())

    raise error_type(message_evaluated)

def _check(cond, message=None):  # noqa: F811
    r"""Throws error containing an optional message if the specified condition
    is False.

    Error type: ``RuntimeError``

    C++ equivalent: ``TORCH_CHECK``

    Args:
        cond (:class:`bool`): If False, throw error

        message (Callable, optional): Callable that returns either a string or
            an object that has a ``__str__()`` method to be used as the error
            message. Default: ``None``
    """
    _check_with(RuntimeError, cond, message)

def _check_is_size(i, message=None):
    """Checks that a given integer is a valid size (i.e., is non-negative).
    You should use this over _check(i >= 0) because we can use the semantic
    information (that i is a size) to make some further inferences in case
    i is an unbacked SymInt.

    NB: Do NOT use this in contexts where a -1 size would be valid (indicating
    to infer the size from context, or if you should wrap-around or truncate).
    Only use this if the only valid value is an honest to goodness size.
    """
    # This is responsible for the expect_true
    _check(i >= 0, message)
    torch.fx.experimental.symbolic_shapes._advise_is_size(i)

def _check_index(cond, message=None):  # noqa: F811
    r"""Throws error containing an optional message if the specified condition
    is False.

    Error type: ``IndexError``

    C++ equivalent: ``TORCH_CHECK_INDEX``

    Args:
        cond (:class:`bool`): If False, throw error

        message (Callable, optional): Callable that returns either a string or
            an object that has a ``__str__()`` method to be used as the error
            message. Default: ``None``
    """
    _check_with(IndexError, cond, message)

def _check_value(cond, message=None):  # noqa: F811
    r"""Throws error containing an optional message if the specified condition
    is False.

    Error type: ``ValueError``

    C++ equivalent: ``TORCH_CHECK_VALUE``

    Args:
        cond (:class:`bool`): If False, throw error

        message (Callable, optional): Callable that returns either a string or
            an object that has a ``__str__()`` method to be used as the error
            message. Default: ``None``
    """
    _check_with(ValueError, cond, message)

def _check_type(cond, message=None):  # noqa: F811
    r"""Throws error containing an optional message if the specified condition
    is False.

    Error type: ``TypeError``

    C++ equivalent: ``TORCH_CHECK_TYPE``

    Args:
        cond (:class:`bool`): If False, throw error

        message (Callable, optional): Callable that returns either a string or
            an object that has a ``__str__()`` method to be used as the error
            message. Default: ``None``
    """
    _check_with(TypeError, cond, message)

def _check_not_implemented(cond, message=None):  # noqa: F811
    r"""Throws error containing an optional message if the specified condition
    is False.

    Error type: ``NotImplementedError``

    C++ equivalent: ``TORCH_CHECK_NOT_IMPLEMENTED``

    Args:
        cond (:class:`bool`): If False, throw error

        message (Callable, optional): Callable that returns either a string or
            an object that has a ``__str__()`` method to be used as the error
            message. Default: ``None``
    """
    _check_with(NotImplementedError, cond, message)

def _check_tensor_all_with(error_type, cond, message=None):  # noqa: F811
    if not torch.is_tensor(cond):
        raise TypeError(f'cond must be a tensor, but got {type(cond)}')

    if not cond.dtype == torch.bool:
        raise TypeError(
            f'cond tensor must have dtype torch.bool, but got {cond.dtype}')

    _check_with(error_type, cond._is_all_true().item(), message)

# C++ equivalent: `TORCH_CHECK_TENSOR_ALL`
def _check_tensor_all(cond, message=None):  # noqa: F811
    r"""Throws error containing an optional message if the specified condition
    is False.

    Error type: ``RuntimeError``

    C++ equivalent: ``TORCH_CHECK_TENSOR_ALL``

    Args:
        cond (:class:`torch.Tensor`): Tensor of dtype ``torch.bool``. If any
            element is ``False``, throw error

        message (Callable, optional): Callable that returns either a string or
            an object that has a ``__str__()`` method to be used as the error
            message. Default: ``None``
    """
    _check_tensor_all_with(RuntimeError, cond, message)

################################################################################
# Define numeric constants
################################################################################

# For Python Array API (https://data-apis.org/array-api/latest/API_specification/constants.html) and
# NumPy consistency (https://numpy.org/devdocs/reference/constants.html)
from math import e , nan , inf , pi
__all__.extend(['e', 'pi', 'nan', 'inf'])

################################################################################
# Define Storage and Tensor classes
################################################################################

from ._tensor import Tensor
from .storage import _StorageBase, TypedStorage, _LegacyStorage, UntypedStorage, _warn_typed_storage_removal

# NOTE: New <type>Storage classes should never be added. When adding a new
# dtype, use torch.storage.TypedStorage directly.

class ByteStorage(_LegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.uint8

class DoubleStorage(_LegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.double

class FloatStorage(_LegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.float

class HalfStorage(_LegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.half

class LongStorage(_LegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.long

class IntStorage(_LegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.int

class ShortStorage(_LegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.short

class CharStorage(_LegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.int8

class BoolStorage(_LegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.bool

class BFloat16Storage(_LegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.bfloat16

class ComplexDoubleStorage(_LegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.cdouble

class ComplexFloatStorage(_LegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.cfloat

class QUInt8Storage(_LegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.quint8

class QInt8Storage(_LegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.qint8

class QInt32Storage(_LegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.qint32

class QUInt4x2Storage(_LegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.quint4x2

class QUInt2x4Storage(_LegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.quint2x4

_storage_classes = {
    UntypedStorage, DoubleStorage, FloatStorage, LongStorage, IntStorage,
    ShortStorage, CharStorage, ByteStorage, HalfStorage, BoolStorage,
    QUInt8Storage, QInt8Storage, QInt32Storage, BFloat16Storage,
    ComplexFloatStorage, ComplexDoubleStorage, QUInt4x2Storage, QUInt2x4Storage,
    TypedStorage
}

# The _tensor_classes set is initialized by the call to _C._initialize_tensor_type_bindings()
_tensor_classes: Set[Type] = set()

# If you edit these imports, please update torch/__init__.py.in as well
from .random import set_rng_state, get_rng_state, manual_seed, initial_seed, seed
from .serialization import save, load
from ._tensor_str import set_printoptions

################################################################################
# Initialize extension
################################################################################

def manager_path():
    if _running_with_deploy() or platform.system() == 'Windows':
        return b""
    path = get_file_path('torch', 'bin', 'torch_shm_manager')
    prepare_multiprocessing_environment(get_file_path('torch'))
    if not os.path.exists(path):
        raise RuntimeError("Unable to find torch_shm_manager at " + path)
    return path.encode('utf-8')

from torch.amp import autocast

# Initializing the extension shadows the built-in python float / int classes;
# store them for later use by SymInt / SymFloat.
py_float = float
py_int = int

# Shared memory manager needs to know the exact location of manager executable
_C._initExtension(manager_path())
del manager_path

# Appease the type checker: it can't deal with direct setting of globals().
# Note that we will see "too many" functions when reexporting this way; there
# is not a good way to fix this problem.  Perhaps, try to redesign VariableFunctions
# so that this import is good enough
if TYPE_CHECKING:
    # Some type signatures pulled in from _VariableFunctions here clash with
    # signatures already imported. For now these clashes are ignored; see
    # PR #43339 for details.
    from torch._C._VariableFunctions import *  # type: ignore[assignment, misc] # noqa: F403
    # Fixup segment_reduce visibility
    _segment_reduce = segment_reduce
    del segment_reduce

# Ops not to be exposed in `torch` namespace,
# mostly helper ops.
PRIVATE_OPS = (
    'unique_dim',
)

for name in dir(_C._VariableFunctions):
    if name.startswith('__') or name in PRIVATE_OPS:
        continue
    obj = getattr(_C._VariableFunctions, name)
    obj.__module__ = 'torch'
    # Hide some APIs that should not be public
    if name == "segment_reduce":
        # TODO: Once the undocumented FC window is passed, remove the line bellow
        globals()[name] = obj
        name = "_" + name
    globals()[name] = obj
    if not name.startswith("_"):
        __all__.append(name)



################################################################################
# Import TorchDynamo's lazy APIs to avoid circular dependenices
################################################################################

# needs to be before from .functional import * to avoid circular dependencies
from ._compile import _disable_dynamo

################################################################################
# Import interface functions defined in Python
################################################################################

# needs to be after the above ATen bindings so we can overwrite from Python side
from .functional import *  # noqa: F403


################################################################################
# Remove unnecessary members
################################################################################

del _StorageBase
del _LegacyStorage

################################################################################
# Define _assert
################################################################################

# needs to be before the submodule imports to avoid circular dependencies
def _assert(condition, message):
    r"""A wrapper around Python's assert which is symbolically traceable.
    """
    from .overrides import has_torch_function, handle_torch_function

    if type(condition) is not torch.Tensor and has_torch_function((condition,)):
        return handle_torch_function(_assert, (condition,), condition, message)
    assert condition, message

################################################################################
# Import most common subpackages
################################################################################

# Use the redundant form so that type checkers know that these are a part of
# the public API. The "regular" import lines are there solely for the runtime
# side effect of adding to the imported module's members for other users.
from torch import cuda as cuda
from torch import cpu as cpu
from torch import mps as mps
from torch import autograd as autograd
from torch.autograd import (
    no_grad as no_grad,
    enable_grad as enable_grad,
    set_grad_enabled as set_grad_enabled,
    inference_mode as inference_mode,
)
from torch import fft as fft
from torch import futures as futures
from torch import _awaits as _awaits
from torch import nested as nested
from torch import nn as nn
from torch.signal import windows as windows
from torch import optim as optim
import torch.optim._multi_tensor
from torch import multiprocessing as multiprocessing
from torch import sparse as sparse
from torch import special as special
import torch.utils.backcompat
from torch import jit as jit
from torch import linalg as linalg
from torch import hub as hub
from torch import random as random
from torch import distributions as distributions
from torch import testing as testing
from torch import backends as backends
import torch.utils.data
from torch import __config__ as __config__
from torch import __future__ as __future__
from torch import profiler as profiler

# Quantized, sparse, AO, etc. should be last to get imported, as nothing
# is expected to depend on them.
from torch import ao as ao
# nn.quant* depends on ao -- so should be after those.
import torch.nn.quantizable
import torch.nn.quantized
import torch.nn.qat
import torch.nn.intrinsic

_C._init_names(list(torch._storage_classes))

# attach docstrings to torch and tensor functions
from . import _torch_docs, _tensor_docs, _storage_docs
del _torch_docs, _tensor_docs, _storage_docs


def compiled_with_cxx11_abi() -> builtins.bool:
    r"""Returns whether PyTorch was built with _GLIBCXX_USE_CXX11_ABI=1"""
    return _C._GLIBCXX_USE_CXX11_ABI


# Import the ops "namespace"
from torch._ops import ops
from torch._classes import classes
import torch._library

# quantization depends on torch.fx
# Import quantization
from torch import quantization as quantization

# Import the quasi random sampler
from torch import quasirandom as quasirandom

# If you are seeing this, it means that this call site was not checked if
# the memory format could be preserved, and it was switched to old default
# behaviour of contiguous
legacy_contiguous_format = contiguous_format

# Register fork handler to initialize OpenMP in child processes (see gh-28389)
from torch.multiprocessing._atfork import register_after_fork
register_after_fork(torch.get_num_threads)
del register_after_fork

# Import tools that require fully imported torch (for applying
# torch.jit.script as a decorator, for instance):
from ._lobpcg import lobpcg as lobpcg

# These were previously defined in native_functions.yaml and appeared on the
# `torch` namespace, but we moved them to c10 dispatch to facilitate custom
# class usage. We add these lines here to preserve backward compatibility.
quantized_lstm = torch.ops.aten.quantized_lstm
quantized_gru = torch.ops.aten.quantized_gru

from torch.utils.dlpack import from_dlpack, to_dlpack

# Import experimental masked operations support. See
# [RFC-0016](https://github.com/pytorch/rfcs/pull/27) for more
# information.
from . import masked

# Import removed ops with error message about removal
from ._linalg_utils import (  # type: ignore[misc]
    matrix_rank,
    eig,
    solve,
    lstsq,
)
from ._linalg_utils import _symeig as symeig  # type: ignore[misc]

class _TorchCompileInductorWrapper:
    compiler_name = "inductor"

    def __init__(self, mode, options, dynamic):
        self.config: Dict[str, Any] = dict()
        self.dynamic = dynamic
        self.apply_mode(mode)
        self.apply_options(options)

        # FIXME: CUPTI Lazy Re-init and CUDA Graph crashes with CUDA 11.
        if self.config.get("triton.cudagraphs", False):
            os.environ["DISABLE_CUPTI_LAZY_REINIT"] = "1"

    def __eq__(self, other):
        return (isinstance(other, _TorchCompileInductorWrapper) and
                self.config == other.config and
                self.dynamic == other.dynamic)

    def apply_mode(self, mode: Optional[str]):
        if mode is None or mode == "default":
            pass
        elif mode in ("reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"):
            from torch._inductor import list_mode_options
            self.apply_options(list_mode_options(mode, self.dynamic))
        else:
            raise RuntimeError(
                f"Unrecognized mode={mode}, should be one of: default, reduce-overhead, max-autotune, max-autotune-no-cudagraphs"
            )

    def apply_options(self, options: Optional[Dict[str, Any]]):
        if not options:
            return

        from torch._inductor import config
        current_config: Dict[str, Any] = config.to_dict()  # type: ignore[attr-defined]

        for key, val in options.items():
            attr_name = key.replace("-", "_")
            if attr_name not in current_config:
                raise RuntimeError(
                    f"Unexpected optimization option {key}, known options are {list(current_config.keys())}"
                )
            if type(val) is not type(current_config[attr_name]):
                val_type_str = type(val).__name__
                expected_type_str = type(current_config[attr_name]).__name__
                raise RuntimeError(
                    f"Unexpected type of attr {key}, got {val_type_str} should be {expected_type_str}"
                )
            self.config[attr_name] = val

    def __call__(self, model_, inputs_):
        from torch._inductor.compile_fx import compile_fx

        return compile_fx(model_, inputs_, config_patches=self.config)

    def get_compiler_config(self):
        from torch._inductor.compile_fx import get_patched_config_dict
        return get_patched_config_dict(config_patches=self.config)

    def reset(self):
        from torch._inductor import config
        if "triton.cudagraphs" in self.config or config.triton.cudagraphs:
            if self.config.get("triton.cudagraphs", True):
                from torch._inductor.cudagraph_trees import reset_cudagraph_trees
                reset_cudagraph_trees()

class _TorchCompileWrapper:
    def __init__(self, backend, mode, options, dynamic):
        from torch._dynamo.backends.registry import lookup_backend

        if isinstance(backend, str):
            self.compiler_name = backend
        elif hasattr(backend, "__name__"):
            self.compiler_name = backend.__name__
        else:
            self.compiler_name = str(backend)
        self.dynamic = dynamic
        self.compiler_fn = lookup_backend(backend)
        self.kwargs = {}
        # only pass the args if they non-empty
        if mode and mode != "default":
            self.kwargs["mode"] = mode
        if options:
            self.kwargs["options"] = options

    def __eq__(self, other):
        return (isinstance(other, _TorchCompileWrapper) and
                self.compiler_fn == other.compiler_fn and
                self.kwargs == other.kwargs and
                self.dynamic == other.dynamic)

    def __call__(self, model_, inputs_):
        return self.compiler_fn(model_, inputs_, **self.kwargs)


def compile(model: Optional[Callable] = None, *,
            fullgraph: builtins.bool = False,
            dynamic: Optional[builtins.bool] = None,
            backend: Union[str, Callable] = "inductor",
            mode: Union[str, None] = None,
            options: Optional[Dict[str, Union[str, builtins.int, builtins.bool]]] = None,
            disable: builtins.bool = False) -> Callable:
    """
    Optimizes given model/function using TorchDynamo and specified backend.

    Concretely, for every frame executed within the compiled region, we will attempt
    to compile it and cache the compiled result on the code object for future
    use.  A single frame may be compiled multiple times if previous compiled
    results are not applicable for subsequent calls (this is called a "guard
    failure), you can use TORCH_LOGS=guards to debug these situations.
    Multiple compiled results can be associated with a frame up to
    ``torch._dynamo.config.cache_size_limit``, which defaults to 64; at which
    point we will fall back to eager.  Note that compile caches are per
    *code object*, not frame; if you dynamically create multiple copies of a
    function, they will all share the same code cache.

    Args:
       model (Callable): Module/function to optimize
       fullgraph (bool): Whether it is ok to break model into several subgraphs
       dynamic (bool or None): Use dynamic shape tracing.  When this is True, we will up-front attempt
        to generate a kernel that is as dynamic as possible to avoid recompilations when
        sizes change.  This may not always work as some operations/optimizations will
        force specialization; use TORCH_LOGS=dynamic to debug overspecialization.
        When this is False, we will NEVER generate dynamic kernels, we will always specialize.
        By default (None), we automatically detect if dynamism has occurred and compile a more
        dynamic kernel upon recompile.
       backend (str or Callable): backend to be used

        - "inductor" is the default backend, which is a good balance between performance and overhead

        - Non experimental in-tree backends can be seen with `torch._dynamo.list_backends()`

        - Experimental or debug in-tree backends can be seen with `torch._dynamo.list_backends(None)`

        - To register an out-of-tree custom backend: https://pytorch.org/docs/main/compile/custom-backends.html
       mode (str): Can be either "default", "reduce-overhead", "max-autotune" or "max-autotune-no-cudagraphs"

        - "default" is the default mode, which is a good balance between performance and overhead

        - "reduce-overhead" is a mode that reduces the overhead of python with CUDA graphs,
          useful for small batches.  Reduction of overhead can come at the cost of more memory
          usage, as we will cache the workspace memory required for the invocation so that we
          do not have to reallocate it on subsequent runs.  Reduction of overhead is not guaranteed
          to work; today, we only reduce overhead for CUDA only graphs which do not mutate inputs.
          There are other circumstances where CUDA graphs are not applicable; use TORCH_LOG=perf_hints
          to debug.

        - "max-autotune" is a mode that leverages Triton based matrix multiplications and convolutions
          It enables CUDA graphs by default.

        - "max-autotune-no-cudagraphs" is a mode similar to "max-autotune" but without CUDA graphs

        - To see the exact configs that each mode sets you can call `torch._inductor.list_mode_options()`

       options (dict): A dictionary of options to pass to the backend. Some notable ones to try out are

        - `epilogue_fusion` which fuses pointwise ops into templates. Requires `max_autotune` to also be set

        - `max_autotune` which will profile to pick the best matmul configuration

        - `fallback_random` which is useful when debugging accuracy issues

        - `shape_padding` which pads matrix shapes to better align loads on GPUs especially for tensor cores

        - `triton.cudagraphs` which will reduce the overhead of python with CUDA graphs

        - `trace.enabled` which is the most useful debugging flag to turn on

        - `trace.graph_diagram` which will show you a picture of your graph after fusion

        - For inductor you can see the full list of configs that it supports by calling `torch._inductor.list_options()`
       disable (bool): Turn torch.compile() into a no-op for testing

    Example::

        @torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
        def foo(x):
            return torch.sin(x) + torch.cos(x)

    """
    _C._log_api_usage_once("torch.compile")
    # Temporary until we get proper support for python 3.12
    if sys.version_info >= (3, 12):
        raise RuntimeError("Dynamo is not supported on Python 3.12+")

    # Decorator mode
    if model is None:
        def fn(model: Callable):
            if model is None:
                raise RuntimeError("Model can't be None")
            return compile(model,
                           fullgraph=fullgraph,
                           dynamic=dynamic,
                           backend=backend,
                           mode=mode,
                           options=options,
                           disable=disable)
        return fn

    if mode is not None and options is not None:
        raise RuntimeError("Either mode or options can be specified, but both can't be specified at the same time.")
    if mode is None and options is None:
        mode = "default"
    if backend == "inductor":
        backend = _TorchCompileInductorWrapper(mode, options, dynamic)
    else:
        backend = _TorchCompileWrapper(backend, mode, options, dynamic)

    return torch._dynamo.optimize(backend=backend, nopython=fullgraph, dynamic=dynamic, disable=disable)(model)


from torch import export as export

from torch._higher_order_ops import cond

def _register_device_module(device_type, module):
    r"""Register an external runtime module of the specific :attr:`device_type`
    supported by torch.

    After the :attr:`module` is registered correctly, the user can refer
    the external runtime module as part of torch with attribute torch.xxx.
    """
    # Make sure the device_type represent a supported device type for torch.
    device_type = torch.device(device_type).type
    m = sys.modules[__name__]
    if hasattr(m, device_type):
        raise RuntimeError(f"The runtime module of '{device_type}' has already "
                           f"been registered with '{getattr(m, device_type)}'")
    setattr(m, device_type, module)
    torch_module_name = '.'.join([__name__, device_type])
    sys.modules[torch_module_name] = module

# expose return_types
from . import return_types
from . import library
if not TYPE_CHECKING:
    from . import _meta_registrations

# Enable CUDA Sanitizer
if 'TORCH_CUDA_SANITIZER' in os.environ:
    import torch.cuda._sanitizer as csan

    csan.enable_cuda_sanitizer()

# Populate magic methods on SymInt and SymFloat
import torch.fx.experimental.symbolic_shapes

from torch import func as func
from torch.func import vmap


# The function _sparse_coo_tensor_unsafe is removed from PyTorch
# Python API (v. 1.13), here we temporarily provide its replacement
# with a deprecation warning.
# TODO: remove the function for PyTorch v 1.15.
def _sparse_coo_tensor_unsafe(*args, **kwargs):
    import warnings
    warnings.warn('torch._sparse_coo_tensor_unsafe is deprecated, '
                  'use torch.sparse_coo_tensor(..., check_invariants=False) instead.')
    kwargs['check_invariants'] = False
    return torch.sparse_coo_tensor(*args, **kwargs)

# Register MPS specific decomps
torch.backends.mps._init()

if not _running_with_deploy():
    from torch import compiler as compiler

    class _TritonLibrary:
        lib = torch.library.Library("triton", "DEF")
        ops_table: Dict[Tuple[str, str], Callable] = {}

        @classmethod
        def registerOp(cls, op_key, full_schema, op_impl, dispatch_key):
            if (op_key, dispatch_key) not in cls.ops_table:
                cls.lib.define(full_schema)
                cls.lib.impl("triton::" + op_key, op_impl, dispatch_key)
                cls.ops_table[(op_key, dispatch_key)] = op_impl

            return cls.ops_table[(op_key, dispatch_key)]


# Deprecated attributes
_deprecated_attrs = {
    "has_mps": torch.backends.mps.is_built,
    "has_cuda": torch.backends.cuda.is_built,
    "has_cudnn": torch.backends.cudnn.is_available,
    "has_mkldnn": torch.backends.mkldnn.is_available,
}

if TYPE_CHECKING:
    # Import the following modules during type checking to enable code intelligence features,
    # such as auto-completion in tools like pylance, even when these modules are not explicitly
    # imported in user code.
    from torch import _dynamo as _dynamo
    from torch import _inductor as _inductor
    from torch import onnx as onnx

else:
    _lazy_modules = {
        "_dynamo",
        "_inductor",
        "_export",
        # ONNX must be imported after _dynamo, _ops, _subclasses, fx, func and jit
        "onnx",
    }

    def __getattr__(name):
        # Deprecated attrs
        replacement = _deprecated_attrs.get(name)
        if replacement is not None:
            import warnings
            warnings.warn(f"'{name}' is deprecated, please use '{replacement.__module__}.{replacement.__name__}()'", stacklevel=2)
            return replacement()

        # Lazy modules
        if name in _lazy_modules:
            import importlib
            return importlib.import_module(f".{name}", __name__)

        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def _constrain_as_value(symbol, min: Optional[builtins.int] = None, max: Optional[builtins.int] = None):
    """
    Add min/max constraint on the intermediate symbol at tracing time. If called in eager mode,
    it will still check if the input value is within the specified range.
    """
    torch.sym_constrain_range(symbol, min=min, max=max)


def _constrain_as_size(symbol, min: Optional[builtins.int] = None, max: Optional[builtins.int] = None):
    """
    This indicates that a given int is size-like, and can be used in any context where a size is expected.
    You will typically use this when reading out integers from Tensors, e.g., max.item() or lengths.tolist()
    which then need to be used as tensor constructors. Providing these assertions to PyTorch can help resolve
      GuardOnDataDependentSymNode errors upon export, since we cannot guard on unbacked SymInts.

    This function has unusual semantics which distinguish it from constrain_as_value.
    Specifically, at compile-time, we will unsoundly assume that the resulting int is always >= 2.
    As a result, max value you pass in should always be greater than 2.
    This makes it easier to use the unbacked int in size contexts, as we will often attempt to guard on a size being zero/one
    (e.g., when computing the contiguity of a tensor, or testing if broadcasting can occur),
    which will not work on unbacked SymInts. Assuming that the int is >= 2 allows us to
    report False to these tests. Although this is technically unsound,
    in practice we observe that if your program works for all sizes >= 2,
    it probably works for zero and one too. The reason specifically assume size is >= 2 is because
    lot of PyTorch code is specialized for 0 and 1 which could result in not general graphs.
    At runtime, we only assert that the user provided min/max values are respected.

    To demonstrate in a scenario, suppose you do
    ```
    # Case 1
    # This will assume symbol is between [2, inf) at compile time, but [0, inf) at runtime
    constrain_as_size(symbol, min=0)

    # Case 2
    # This will assume symbol is between [2, N] at compile time, but [0, N] at runtime
    constrain_as_size(symbol, min=0, max=N)

    # Case 3
    # This is not valid case as max is <= 2
    constrain_as_size(symbol, min=0, max=1)

    # Case 4
    # This will assume symbol is between [2, inf) at compile time, AND [2, inf) at runtime
    constrain_as_size(symbol, min=2)

    # Case 5
    # This will assume symbol is between [2, inf) at compile time, but [1, inf) at runtime
    constrain_as_size(symbol, min=1)
    ```
    """
    torch.sym_constrain_range_for_size(symbol, min=min, max=max)


from . import _logging
_logging._init_logs()
