
r"""
The torch package contains data structures for multi-dimensional
tensors and defines mathematical operations over these tensors.
Additionally, it provides many utilities for efficient serialization of
Tensors and arbitrary types, and other useful utilities.

It has a CUDA counterpart, that enables you to run your tensor computations
on an NVIDIA GPU with compute capability >= 3.0.
"""

import os
import sys
import platform
import textwrap
import ctypes
import inspect
if sys.version_info < (3,):
    raise Exception("Python 2 has reached end-of-life and is no longer supported by PyTorch.")

from ._utils import _import_dotted_name, classproperty
from ._utils_internal import get_file_path, prepare_multiprocessing_environment, \
    USE_RTLD_GLOBAL_WITH_LIBTORCH, USE_GLOBAL_DEPS
# TODO(torch_deploy) figure out how to freeze version.py in fbcode build
if sys.executable == 'torch_deploy':
    __version__ = "torch-deploy-1.8"
else:
    from .torch_version import __version__ as __version__

from ._six import string_classes as _string_classes

from typing import Set, Type, TYPE_CHECKING, Union, Callable, Any
import builtins

__all__ = [
    'typename', 'is_tensor', 'is_storage', 'set_default_tensor_type',
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

    if all([not os.path.exists(os.path.join(p, 'nvToolsExt64_1.dll')) for p in dll_paths]):
        nvtoolsext_dll_path = os.path.join(
            os.getenv('NVTOOLSEXT_PATH', os.path.join(pfiles_path, 'NVIDIA Corporation', 'NvToolsExt')), 'bin', 'x64')
    else:
        nvtoolsext_dll_path = ''

    from .version import cuda as cuda_version
    import glob
    if cuda_version and all([not glob.glob(os.path.join(p, 'cudart64*.dll')) for p in dll_paths]):
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
        kernel32.AddDllDirectory.restype = ctypes.c_void_p
        kernel32.LoadLibraryExW.restype = ctypes.c_void_p

    for dll_path in dll_paths:
        if sys.version_info >= (3, 8):
            os.add_dll_directory(dll_path)
        elif with_load_library_flags:
            res = kernel32.AddDllDirectory(dll_path)
            if res is None:
                err = ctypes.WinError(ctypes.get_last_error())
                err.strerror += f' Error adding "{dll_path}" to the DLL directories.'
                raise err

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


# See Note [Global dependencies]
def _load_global_deps():
    if platform.system() == 'Windows' or sys.executable == 'torch_deploy':
        return

    lib_name = 'libtorch_global_deps' + ('.dylib' if platform.system() == 'Darwin' else '.so')
    here = os.path.abspath(__file__)
    lib_path = os.path.join(os.path.dirname(here), 'lib', lib_name)

    ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)


if (USE_RTLD_GLOBAL_WITH_LIBTORCH or os.getenv('TORCH_USE_RTLD_GLOBAL')) and \
        platform.system() != 'Windows':
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
        from torch.fx.experimental.symbolic_shapes import SymNode
        assert isinstance(node, SymNode)
        # This field MUST be named node; C++ binding code assumes that this
        # class has a field named node that stores SymNode
        self.node = node

    def __bool__(self):
        return self.node.bool_()

    def __int__(self):
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

    def __sym_float__(self):
        raise AssertionError("type stub not overridden")

    def __repr__(self):
        return self.node.str()

    # For BC; direct access of node is OK too
    def get_pyobj(self):
        return self.node

class SymFloat:
    """
    Like an float (including magic methods), but redirects all operations on the
    wrapped node. This is used in particular to symbolically record operations
    in the symbolic shape workflow.
    """

    def __init__(self, node):
        from torch.fx.experimental.symbolic_shapes import SymNode
        assert isinstance(node, SymNode)
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

    def __repr__(self):
        return self.node.str()

    # For BC; direct access of node is OK too
    def get_pyobj(self):
        return self.node

# Check to see if we can load C extensions, and if not provide some guidance
# on what the problem might be.
try:
    # _initExtension is chosen (arbitrarily) as a sentinel.
    from torch._C import _initExtension
except ImportError:
    import torch._C as _C_for_compiled_check

    # The __file__ check only works for Python 3.7 and above.
    if sys.version_info >= (3, 7) and _C_for_compiled_check.__file__ is None:
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
                if name not in ['DisableTorchFunction', 'Generator']:
                    obj.__module__ = 'torch'

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

        >>> x=torch.tensor([1,2,3])
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
    if isinstance(t, _string_classes):
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

def use_deterministic_algorithms(mode, *, warn_only=False):
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
        * :func:`torch.bincount` when called on a CUDA tensor
        * :func:`torch.kthvalue` with called on a CUDA tensor
        * :func:`torch.median` with indices output when called on a CUDA tensor
        * :func:`torch.nn.functional.grid_sample` when attempting to differentiate a CUDA tensor
        * :func:`torch.cumsum` when called on a CUDA tensor when dtype is floating point or complex

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

        >>> torch.use_deterministic_algorithms(True)

        # Forward mode nondeterministic error
        >>> # xdoctest: +SKIP
        >>> torch.randn(10, device='cuda').kthvalue(0)
        ...
        RuntimeError: kthvalue CUDA does not have a deterministic implementation...

        # Backward mode nondeterministic error
        >>> torch.nn.AvgPool3d(1)(torch.randn(3, 4, 5, 6, requires_grad=True).cuda()).sum().backward()
        ...
        RuntimeError: avg_pool3d_backward_cuda does not have a deterministic implementation...
    """
    _C._set_deterministic_algorithms(mode, warn_only=warn_only)

def are_deterministic_algorithms_enabled():
    r"""Returns True if the global deterministic flag is turned on. Refer to
    :func:`torch.use_deterministic_algorithms` documentation for more details.
    """
    return _C._get_deterministic_algorithms()

def is_deterministic_algorithms_warn_only_enabled():
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

def set_float32_matmul_precision(precision):
    r"""Sets the internal precision of float32 matrix multiplications.

    Running float32 matrix multiplications in lower precision may significantly increase
    performance, and in some programs the loss of precision has a negligible impact.

    Supports three settings:

        * "highest", float32 matrix multiplications use the float32 datatype for
          internal computations.
        * "high", float32 matrix multiplications use the TensorFloat32 or bfloat16_3x
          datatypes for internal computations, if fast matrix multiplication algorithms
          using those datatypes internally are available. Otherwise float32
          matrix multiplications are computed as if the precision is "highest".
        * "medium", float32 matrix multiplications use the bfloat16 datatype for
          internal computations, if a fast matrix multiplication algorithm
          using that datatype internally is available. Otherwise float32
          matrix multiplications are computed as if the precision is "high".

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

def set_warn_always(b):
    r"""When this flag is False (default) then some PyTorch warnings may only
    appear once per process. This helps avoid excessive warning information.
    Setting it to True causes these warnings to always appear, which may be
    helpful when debugging.

    Args:
        b (:class:`bool`): If True, force warnings to always be emitted
                           If False, set to the default behaviour
    """
    _C._set_warnAlways(b)

def is_warn_always_enabled():
    r"""Returns True if the global warn_always flag is turned on. Refer to
    :func:`torch.set_warn_always` documentation for more details.
    """
    return _C._get_warnAlways()

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
    if platform.system() == 'Windows' or sys.executable == 'torch_deploy':
        return b""
    path = get_file_path('torch', 'bin', 'torch_shm_manager')
    prepare_multiprocessing_environment(get_file_path('torch'))
    if not os.path.exists(path):
        raise RuntimeError("Unable to find torch_shm_manager at " + path)
    return path.encode('utf-8')

from torch.amp import autocast

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
    from torch._C._VariableFunctions import *  # type: ignore[misc] # noqa: F403

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
    globals()[name] = obj
    if not name.startswith("_"):
        __all__.append(name)

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
from torch import autograd as autograd
from torch.autograd import (
    no_grad as no_grad,
    enable_grad as enable_grad,
    set_grad_enabled as set_grad_enabled,
    inference_mode as inference_mode,
)
from torch import fft as fft
from torch import futures as futures
from torch import nested as nested
from torch import nn as nn
from torch.signal import windows as windows
from torch import optim as optim
import torch.optim._multi_tensor
from torch import multiprocessing as multiprocessing
from torch import sparse as sparse
from torch import special as special
import torch.utils.backcompat
from torch import onnx as onnx
from torch import jit as jit
from torch import linalg as linalg
from torch import hub as hub
from torch import random as random
from torch import distributions as distributions
from torch import testing as testing
import torch.backends.cuda
import torch.backends.mps
import torch.backends.cudnn
import torch.backends.mkl
import torch.backends.mkldnn
import torch.backends.openmp
import torch.backends.quantized
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


def compiled_with_cxx11_abi():
    r"""Returns whether PyTorch was built with _GLIBCXX_USE_CXX11_ABI=1"""
    return _C._GLIBCXX_USE_CXX11_ABI


# Import the ops "namespace"
from torch._ops import ops
from torch._classes import classes

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

from ._vmap_internals import vmap as vmap

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
        raise RuntimeError("The runtime module of '{}' has already "
                           "been registered with '{}'".format(device_type, getattr(m, device_type)))
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
