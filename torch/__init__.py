"""
The torch package contains data structures for multi-dimensional
tensors and defines mathematical operations over these tensors.
Additionally, it provides many utilities for efficient serialization of
Tensors and arbitrary types, and other useful utilities.

It has a CUDA counterpart, that enables you to run your tensor computations
on an NVIDIA GPU with compute capability >= 3.0.
"""

# mypy: allow-untyped-defs

import builtins
import ctypes
import glob
import importlib
import importlib.util
import inspect
import math
import os
import platform
import sys
import textwrap
import threading
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, TYPE_CHECKING, Union


# multipy/deploy is setting this import before importing torch, this is the most
# reliable way we have to detect if we're running within deploy.
# https://github.com/pytorch/multipy/blob/d60f34ad38c371e441fe7ffdb77a3c3dda5a5d19/multipy/runtime/interpreter/interpreter_impl.cpp#L134-L137
def _running_with_deploy():
    return sys.modules.get("torch._meta_registrations", None) is object


from torch._utils import (
    _functionalize_sync as _sync,
    _import_dotted_name,
    classproperty,
)
from torch._utils_internal import (
    get_file_path,
    prepare_multiprocessing_environment,
    USE_GLOBAL_DEPS,
    USE_RTLD_GLOBAL_WITH_LIBTORCH,
)

# TODO(torch_deploy) figure out how to freeze version.py in fbcode build
if _running_with_deploy():
    __version__ = "torch-deploy-1.8"
else:
    from torch.torch_version import __version__ as __version__


__all__ = [
    "BoolStorage",
    "BoolTensor",
    "ByteStorage",
    "ByteTensor",
    "CharStorage",
    "CharTensor",
    "DoubleStorage",
    "DoubleTensor",
    "FloatStorage",
    "FloatTensor",
    "GradScaler",
    "IntStorage",
    "IntTensor",
    "LongStorage",
    "LongTensor",
    "ShortStorage",
    "ShortTensor",
    "SymBool",
    "SymFloat",
    "SymInt",
    "Tensor",
    "TypedStorage",
    "UntypedStorage",
    "are_deterministic_algorithms_enabled",
    "autocast",
    "chunk",
    "compile",
    "cond",
    "enable_grad",
    "export",
    "get_default_device",
    "get_deterministic_debug_mode",
    "get_device_module",
    "get_float32_matmul_precision",
    "get_rng_state",
    "inference_mode",
    "initial_seed",
    "is_deterministic_algorithms_warn_only_enabled",
    "is_storage",
    "is_tensor",
    "is_warn_always_enabled",
    "load",
    "lobpcg",
    "manual_seed",
    "matmul",
    "no_grad",
    "rand",
    "randn",
    "save",
    "seed",
    "set_default_device",
    "set_default_tensor_type",
    "set_deterministic_debug_mode",
    "set_float32_matmul_precision",
    "set_printoptions",
    "set_rng_state",
    "set_warn_always",
    "split",
    "stack",
    "sym_float",
    "sym_int",
    "sym_ite",
    "sym_max",
    "sym_min",
    "sym_not",
    "typename",
    "unravel_index",
    "use_deterministic_algorithms",
    "vmap",
]

# Please keep this list sorted
assert __all__ == sorted(__all__)

################################################################################
# Load the extension module
################################################################################

if sys.platform == "win32":

    def _load_dll_libraries():
        import sysconfig

        from torch.version import cuda as cuda_version

        pfiles_path = os.getenv("ProgramFiles", r"C:\Program Files")
        py_dll_path = os.path.join(sys.exec_prefix, "Library", "bin")
        th_dll_path = os.path.join(os.path.dirname(__file__), "lib")
        usebase_path = os.path.join(
            sysconfig.get_config_var("userbase"), "Library", "bin"
        )

        # When users create a virtualenv that inherits the base environment,
        # we will need to add the corresponding library directory into
        # DLL search directories. Otherwise, it will rely on `PATH` which
        # is dependent on user settings.
        if sys.exec_prefix != sys.base_exec_prefix:
            base_py_dll_path = os.path.join(sys.base_exec_prefix, "Library", "bin")
        else:
            base_py_dll_path = ""

        dll_paths = [
            p
            for p in (th_dll_path, py_dll_path, base_py_dll_path, usebase_path)
            if os.path.exists(p)
        ]

        if not builtins.any(
            os.path.exists(os.path.join(p, "nvToolsExt64_1.dll")) for p in dll_paths
        ):
            nvtoolsext_dll_path = os.path.join(
                os.getenv(
                    "NVTOOLSEXT_PATH",
                    os.path.join(pfiles_path, "NVIDIA Corporation", "NvToolsExt"),
                ),
                "bin",
                "x64",
            )
        else:
            nvtoolsext_dll_path = ""

        if cuda_version and builtins.all(
            not glob.glob(os.path.join(p, "cudart64*.dll")) for p in dll_paths
        ):
            cuda_version_1 = cuda_version.replace(".", "_")
            cuda_path_var = "CUDA_PATH_V" + cuda_version_1
            default_path = os.path.join(
                pfiles_path, "NVIDIA GPU Computing Toolkit", "CUDA", f"v{cuda_version}"
            )
            cuda_path = os.path.join(os.getenv(cuda_path_var, default_path), "bin")
        else:
            cuda_path = ""

        dll_paths.extend(
            p for p in (nvtoolsext_dll_path, cuda_path) if os.path.exists(p)
        )

        kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
        with_load_library_flags = hasattr(kernel32, "AddDllDirectory")
        prev_error_mode = kernel32.SetErrorMode(0x0001)

        kernel32.LoadLibraryW.restype = ctypes.c_void_p
        if with_load_library_flags:
            kernel32.LoadLibraryExW.restype = ctypes.c_void_p

        for dll_path in dll_paths:
            os.add_dll_directory(dll_path)

        try:
            ctypes.CDLL("vcruntime140.dll")
            ctypes.CDLL("msvcp140.dll")
            ctypes.CDLL("vcruntime140_1.dll")
        except OSError:
            print(
                textwrap.dedent(
                    """
                    Microsoft Visual C++ Redistributable is not installed, this may lead to the DLL load failure.
                    It can be downloaded at https://aka.ms/vs/16/release/vc_redist.x64.exe
                    """
                ).strip()
            )

        dlls = glob.glob(os.path.join(th_dll_path, "*.dll"))
        path_patched = False
        for dll in dlls:
            is_loaded = False
            if with_load_library_flags:
                res = kernel32.LoadLibraryExW(dll, None, 0x00001100)
                last_error = ctypes.get_last_error()
                if res is None and last_error != 126:
                    err = ctypes.WinError(last_error)
                    err.strerror += (
                        f' Error loading "{dll}" or one of its dependencies.'
                    )
                    raise err
                elif res is not None:
                    is_loaded = True
            if not is_loaded:
                if not path_patched:
                    os.environ["PATH"] = ";".join(dll_paths + [os.environ["PATH"]])
                    path_patched = True
                res = kernel32.LoadLibraryW(dll)
                if res is None:
                    err = ctypes.WinError(ctypes.get_last_error())
                    err.strerror += (
                        f' Error loading "{dll}" or one of its dependencies.'
                    )
                    raise err

        kernel32.SetErrorMode(prev_error_mode)

    _load_dll_libraries()
    del _load_dll_libraries


def _preload_cuda_deps(lib_folder, lib_name):
    """Preloads cuda deps if they could not be found otherwise."""
    # Should only be called on Linux if default path resolution have failed
    assert platform.system() == "Linux", "Should only be called on Linux"

    lib_path = None
    for path in sys.path:
        nvidia_path = os.path.join(path, "nvidia")
        if not os.path.exists(nvidia_path):
            continue
        candidate_lib_paths = glob.glob(
            os.path.join(nvidia_path, lib_folder, "lib", lib_name)
        )
        if candidate_lib_paths and not lib_path:
            lib_path = candidate_lib_paths[0]
        if lib_path:
            break
    if not lib_path:
        raise ValueError(f"{lib_name} not found in the system path {sys.path}")
    ctypes.CDLL(lib_path)


# See Note [Global dependencies]
def _load_global_deps() -> None:
    LIBTORCH_PKG_NAME = "libtorchsplit"

    def find_package_path(package_name):
        spec = importlib.util.find_spec(package_name)
        if spec:
            # The package might be a namespace package, so get_data may fail
            try:
                loader = spec.loader
                if loader is not None:
                    file_path = loader.get_filename()  # type: ignore[attr-defined]
                    return os.path.dirname(file_path)
            except AttributeError:
                pass
        return None

    def load_shared_libraries(library_path):
        lib_dir = os.path.join(library_path, "lib")
        if not os.path.exists(lib_dir):
            return

        # Find all shared library files with the appropriate extension
        library_files = [f for f in os.listdir(lib_dir) if f.endswith(lib_ext)]
        if not library_files:
            return

        for lib_file in library_files:
            lib_path = os.path.join(lib_dir, lib_file)
            try:
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            except OSError as err:
                print(f"Failed to load {lib_path}: {err}")

    if _running_with_deploy() or platform.system() == "Windows":
        return

    # Determine the file extension based on the platform
    lib_ext = ".dylib" if platform.system() == "Darwin" else ".so"
    lib_name = f"libtorch_global_deps{lib_ext}"
    here = os.path.abspath(__file__)
    global_deps_lib_path = os.path.join(os.path.dirname(here), "lib", lib_name)

    split_build_lib_name = LIBTORCH_PKG_NAME
    library_path = find_package_path(split_build_lib_name)

    if library_path:
        global_deps_lib_path = os.path.join(library_path, "lib", lib_name)
    try:
        ctypes.CDLL(global_deps_lib_path, mode=ctypes.RTLD_GLOBAL)
    except OSError as err:
        # Can only happen for wheel with cuda libs as PYPI deps
        # As PyTorch is not purelib, but nvidia-*-cu12 is
        cuda_libs: Dict[str, str] = {
            "cublas": "libcublas.so.*[0-9]",
            "cudnn": "libcudnn.so.*[0-9]",
            "cuda_nvrtc": "libnvrtc.so.*[0-9]",
            "cuda_runtime": "libcudart.so.*[0-9]",
            "cuda_cupti": "libcupti.so.*[0-9]",
            "cufft": "libcufft.so.*[0-9]",
            "curand": "libcurand.so.*[0-9]",
            "cusolver": "libcusolver.so.*[0-9]",
            "cusparse": "libcusparse.so.*[0-9]",
            "nccl": "libnccl.so.*[0-9]",
            "nvtx": "libnvToolsExt.so.*[0-9]",
        }
        is_cuda_lib_err = [
            lib for lib in cuda_libs.values() if lib.split(".")[0] in err.args[0]
        ]
        if not is_cuda_lib_err:
            raise err
        for lib_folder, lib_name in cuda_libs.items():
            _preload_cuda_deps(lib_folder, lib_name)
        ctypes.CDLL(global_deps_lib_path, mode=ctypes.RTLD_GLOBAL)

    if library_path:
        # loading libtorch_global_deps first due its special logic
        load_shared_libraries(library_path)


if (USE_RTLD_GLOBAL_WITH_LIBTORCH or os.getenv("TORCH_USE_RTLD_GLOBAL")) and (
    _running_with_deploy() or platform.system() != "Windows"
):
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

    # Magic methods installed by torch.fx.experimental.sym_node

    def __round__(self, ndigits=None):
        return self

    def __truediv__(self, other):
        if isinstance(other, (builtins.float, SymFloat)):
            return sym_float(self).__float_truediv__(other)
        if not isinstance(other, (builtins.int, SymInt)):
            return NotImplemented
        return self.__int_truediv__(other)

    def __rtruediv__(self, other):
        if isinstance(other, (builtins.float, SymFloat)):
            return sym_float(self).__rfloat_truediv__(other)
        if not isinstance(other, (builtins.int, SymInt)):
            return NotImplemented
        return self.__rint_truediv__(other)

    def __floordiv__(self, other):
        if isinstance(other, (builtins.float, SymFloat)):
            return torch.sym_float(math.floor(sym_float(self) / other))
        if not isinstance(other, (builtins.int, SymInt)):
            return NotImplemented
        return self.__int_floordiv__(other)

    def __rfloordiv__(self, other):
        if isinstance(other, (builtins.float, SymFloat)):
            return torch.sym_float(math.floor(other / sym_float(self)))
        if not isinstance(other, (builtins.int, SymInt)):
            return NotImplemented
        return self.__rint_floordiv__(other)

    # nb: complex is impossible to handle correctly lol, with
    # negative base and integral float need to diverge semantics and
    # just always return complex.  Neener neener pretend this problem
    # doesn't exist
    def __pow__(self, other):
        if isinstance(other, (builtins.float, SymFloat)):
            return sym_float(self).__pow__(other)
        if not isinstance(other, (builtins.int, SymInt)):
            return NotImplemented
        # Guards!  This guard is necessary because we need to know it to
        # determine the output type of this operation
        if other >= 0:
            return self.__pow_by_natural__(other)
        else:
            # Mercifully, when the exponent is negative, Python just promotes
            # to doubles and does a float pow:
            #
            #   if (Py_SIZE(b) < 0 && c == NULL) {
            #       /* if exponent is negative and there's no modulus:
            #              return a float.  This works because we know
            #              that this calls float_pow() which converts its
            #              arguments to double. */
            #       Py_DECREF(a);
            #       Py_DECREF(b);
            #       return PyFloat_Type.tp_as_number->nb_power(v, w, x);
            #   }
            return sym_float(self).__pow__(sym_float(other))

    def __rpow__(self, other):
        if isinstance(other, (builtins.float, SymFloat)):
            return sym_float(self).__rpow__(other)
        if not isinstance(other, (builtins.int, SymInt)):
            return NotImplemented
        if self >= 0:  # self is exponent
            return self.__rpow_by_natural__(other)
        else:
            return sym_float(self).__rpow__(sym_float(other))

    def __eq__(self, other: object) -> builtins.bool:
        raise TypeError("type stub not overridden")

    def __lt__(self, other) -> builtins.bool:
        raise TypeError("type stub not overridden")

    def __gt__(self, other) -> builtins.bool:
        raise TypeError("type stub not overridden")

    def __le__(self, other) -> builtins.bool:
        raise TypeError("type stub not overridden")

    def __ge__(self, other) -> builtins.bool:
        raise TypeError("type stub not overridden")

    def __add__(self, other) -> "SymInt":
        raise TypeError("type stub not overridden")

    def __mul__(self, other) -> "SymInt":
        raise TypeError("type stub not overridden")

    def __pow_by_natural__(self, other) -> "SymInt":
        raise TypeError("type stub not overridden")

    def __rpow_by_natural__(self, other) -> "SymInt":
        raise TypeError("type stub not overridden")

    def __int_truediv__(self, other) -> "SymFloat":
        raise TypeError("type stub not overridden")

    def __rint_truediv__(self, other) -> "SymFloat":
        raise TypeError("type stub not overridden")

    def __int_floordiv__(self, other) -> "SymFloat":
        raise TypeError("type stub not overridden")

    def __rint_floordiv__(self, other) -> "SymFloat":
        raise TypeError("type stub not overridden")

    def __sym_max__(self, other):
        raise TypeError("type stub not overridden")

    def __sym_min__(self, other):
        raise TypeError("type stub not overridden")

    def __sym_float__(self):
        raise TypeError("type stub not overridden")

    def __neg__(self):
        raise TypeError("type stub not overridden")

    def __repr__(self):
        return str(self.node)

    def __hash__(self) -> builtins.int:
        if self.node.is_nested_int():
            return hash(self.node.nested_int())
        else:
            # We could support constant SymInts as well, but not doing it for now
            raise TypeError("unhashable type: non-nested SymInt")


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

    def __truediv__(self, other):
        if not isinstance(other, (builtins.int, builtins.float, SymInt, SymFloat)):
            return NotImplemented
        return self.__float_truediv__(sym_float(other))

    def __rtruediv__(self, other):
        if not isinstance(other, (builtins.int, builtins.float, SymInt, SymFloat)):
            return NotImplemented
        return self.__rfloat_truediv__(sym_float(other))

    def __floordiv__(self, other):
        if not isinstance(other, (builtins.int, builtins.float, SymInt, SymFloat)):
            return NotImplemented
        return torch.sym_float(math.floor(self / sym_float(other)))

    def __rfloordiv__(self, other):
        if not isinstance(other, (builtins.int, builtins.float, SymInt, SymFloat)):
            return NotImplemented
        return torch.sym_float(math.floor(sym_float(other) / self))

    def __bool__(self):
        return self.node.bool_()

    # Symbolic power does NOT work with negative base, this is to avoid
    # potential complex outputs
    def __pow__(self, other):
        if not isinstance(other, (builtins.int, builtins.float, SymInt, SymFloat)):
            return NotImplemented
        torch._check(self >= 0)
        return self.__float_pow__(other)

    def __rpow__(self, other):
        if not isinstance(other, (builtins.int, builtins.float, SymInt, SymFloat)):
            return NotImplemented
        torch._check(other >= 0)
        return self.__rfloat_pow__(other)

    # Magic methods installed by torch.fx.experimental.sym_node

    def __eq__(self, other: object) -> builtins.bool:
        raise TypeError("type stub not overridden")

    def __lt__(self, other) -> builtins.bool:
        raise TypeError("type stub not overridden")

    def __gt__(self, other) -> builtins.bool:
        raise TypeError("type stub not overridden")

    def __le__(self, other) -> builtins.bool:
        raise TypeError("type stub not overridden")

    def __ge__(self, other) -> builtins.bool:
        raise TypeError("type stub not overridden")

    def __float_pow__(self, other) -> "SymFloat":
        raise TypeError("type stub not overridden")

    def __rfloat_pow__(self, other) -> "SymFloat":
        raise TypeError("type stub not overridden")

    def __float_truediv__(self, other) -> "SymFloat":
        raise TypeError("type stub not overridden")

    def __rfloat_truediv__(self, other) -> "SymFloat":
        raise TypeError("type stub not overridden")

    def __trunc__(self):
        raise TypeError("type stub not overridden")

    def __sym_max__(self, other):
        raise TypeError("type stub not overridden")

    def __sym_min__(self, other):
        raise TypeError("type stub not overridden")

    def __sym_int__(self):
        raise TypeError("type stub not overridden")

    def is_integer(self):
        """Return True if the float is an integer."""
        raise TypeError("type stub not overridden")

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

    # Magic methods installed by torch.fx.experimental.sym_node
    def __and__(self, other) -> "SymBool":
        raise TypeError("type stub not overridden")

    def __or__(self, other) -> "SymBool":
        raise TypeError("type stub not overridden")

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
        raise TypeError("type stub not overridden")

    def __sym_ite__(self, then_val, else_val):
        raise TypeError("type stub not overridden")

    def __eq__(self, other) -> builtins.bool:
        raise TypeError("type stub not overridden")

    def __repr__(self):
        return str(self.node)

    def __hash__(self):
        if self.node.is_constant():
            return hash(self.node.bool_())
        else:
            raise TypeError("unhashable type: SymBool")


def sym_not(a):
    r"""SymInt-aware utility for logical negation.

    Args:
        a (SymBool or bool): Object to negate
    """
    import sympy

    if overrides.has_torch_function_unary(a):
        return overrides.handle_torch_function(sym_not, (a,), a)
    if hasattr(a, "__sym_not__"):
        return a.__sym_not__()
    if isinstance(a, sympy.Basic):
        return ~a  # type: ignore[operator]
    return not a


def sym_float(a):
    r"""SymInt-aware utility for float casting.

    Args:
        a (SymInt, SymFloat, or object): Object to cast
    """
    if overrides.has_torch_function_unary(a):
        return overrides.handle_torch_function(sym_float, (a,), a)
    if isinstance(a, SymFloat):
        return a
    elif hasattr(a, "__sym_float__"):
        return a.__sym_float__()
    return builtins.float(a)  # type: ignore[operator]


def sym_int(a):
    r"""SymInt-aware utility for int casting.

    Args:
        a (SymInt, SymFloat, or object): Object to cast
    """
    if overrides.has_torch_function_unary(a):
        return overrides.handle_torch_function(sym_int, (a,), a)
    if isinstance(a, SymInt):
        return a
    elif isinstance(a, SymFloat):
        return math.trunc(a)
    return builtins.int(a)  # type: ignore[operator]


def sym_max(a, b):
    """
    SymInt-aware utility for max which avoids branching on a < b.
    Unlike builtins.max(), this only works for int/float, and it always
    promotes to float if any argument is float (unlike builtins.max, which
    will faithfully preserve the type of the input argument).
    """
    if overrides.has_torch_function((a, b)):
        return overrides.handle_torch_function(sym_max, (a, b), a, b)
    if isinstance(a, (SymInt, SymFloat)):
        return a.__sym_max__(b)
    elif isinstance(b, (SymInt, SymFloat)):
        # Due to promotion semantics, this is operator is commutative:
        # max(1, 1.0) === max(1.0, 1) === 1.0
        return b.__sym_max__(a)
    # TODO: Probably can make bool work too, just lazy
    assert isinstance(a, (builtins.int, builtins.float)), type(a)
    assert isinstance(b, (builtins.int, builtins.float)), type(b)
    if isinstance(a, builtins.float) or isinstance(b, builtins.float):
        return builtins.float(builtins.max(a, b))
    else:
        return builtins.max(a, b)


def sym_min(a, b):
    """SymInt-aware utility for min()."""
    if overrides.has_torch_function((a, b)):
        return overrides.handle_torch_function(sym_min, (a, b), a, b)
    if isinstance(a, (SymInt, SymFloat)):
        return a.__sym_min__(b)
    elif isinstance(b, (SymInt, SymFloat)):
        return b.__sym_min__(a)
    assert isinstance(a, (builtins.int, builtins.float)), type(a)
    assert isinstance(b, (builtins.int, builtins.float)), type(b)
    if isinstance(a, builtins.float) or isinstance(b, builtins.float):
        return builtins.float(builtins.min(a, b))
    else:
        return builtins.min(a, b)


# Drop in replacement for math.sqrt, math.sin, math.cos etc
def _get_sym_math_fn(name):
    def fn(a):
        if overrides.has_torch_function_unary(a):
            return overrides.handle_torch_function(fn, (a,), a)
        if hasattr(a, f"__sym_{name}__"):
            return getattr(a, f"__sym_{name}__")()
        return getattr(math, name)(a)

    return fn


__fn, __name, __sym_name = None, "", ""
for __name in (
    "sqrt",
    "cos",
    "cosh",
    "sin",
    "sinh",
    "tan",
    "tanh",
    "asin",
    "acos",
    "atan",
):
    __sym_name = f"_sym_{__name}"
    __fn = _get_sym_math_fn(__name)
    __fn.__qualname__ = __fn.__name__ = __sym_name
    globals()[__sym_name] = __fn

del __fn, __name, __sym_name, _get_sym_math_fn

# Adding temporary shortcut
sym_sqrt = globals()["_sym_sqrt"]
__all__.append("sym_sqrt")


def sym_ite(b, t, f):
    if overrides.has_torch_function((b, t, f)):
        return overrides.handle_torch_function(sym_ite, (b, t, f), b, t, f)
    assert isinstance(b, (SymBool, builtins.bool)) and type(t) == type(f)
    if isinstance(b, SymBool):
        return b.__sym_ite__(t, f)
    return t if b else f


# Check to see if we can load C extensions, and if not provide some guidance
# on what the problem might be.
try:
    # _initExtension is chosen (arbitrarily) as a sentinel.
    from torch._C import _initExtension
except ImportError:
    import torch._C as _C_for_compiled_check

    # The __file__ check only works for Python 3.7 and above.
    if _C_for_compiled_check.__file__ is None:
        raise ImportError(
            textwrap.dedent(
                """
                Failed to load PyTorch C extensions:
                    It appears that PyTorch has loaded the `torch/_C` folder
                    of the PyTorch repository rather than the C extensions which
                    are expected in the `torch._C` namespace. This can occur when
                    using the `install` workflow. e.g.
                        $ python setup.py install && python -c "import torch"

                    This error can generally be solved using the `develop` workflow
                        $ python setup.py develop && python -c "import torch"  # This should succeed
                    or by running Python from a different directory.
                """
            ).strip()
        ) from None
    raise  # If __file__ is not None the cause is unknown, so just re-raise.

# The torch._C submodule is already loaded via `from torch._C import *` above
# Make an explicit reference to the _C submodule to appease linters
from torch import _C as _C

__name, __obj = "", None
for __name in dir(_C):
    if __name[0] != "_" and not __name.endswith("Base"):
        __all__.append(__name)
        __obj = getattr(_C, __name)
        if callable(__obj) or inspect.isclass(__obj):
            if __obj.__module__ != __name__:  # "torch"
                # TODO: fix their module from C++ side
                if __name not in {
                    "DisableTorchFunctionSubclass",
                    "DisableTorchFunction",
                    "Generator",
                }:
                    __obj.__module__ = __name__  # "torch"
    elif __name == "TensorBase":
        # issue 109438 / pr 109940. Prevent TensorBase from being copied into torch.
        delattr(sys.modules[__name__], __name)

del __name, __obj

if not TYPE_CHECKING:
    # issue 38137 and python issue 43367. Submodules of a C extension are
    # non-standard, and attributes of those submodules cannot be pickled since
    # pickle expect to be able to import them as "from _C.sub import attr"
    # which fails with "_C is not a package
    __name, __candidate = "", None
    for __name in dir(_C):
        __candidate = getattr(_C, __name)
        if type(__candidate) is type(_C):
            # submodule
            sys.modules.setdefault(f"{__name__}._C.{__name}", __candidate)

    del __name, __candidate


################################################################################
# Define basic utilities
################################################################################


def typename(o):
    """
    String representation of the type of an object.

    This function returns a fully qualified string representation of an object's type.
    Args:
        o (Object): The object whose type to represent
    Returns:
        str: the type of the object `o`
    Example:
        >>> x = torch.tensor([1,2,3])
        >>> torch.typename(x)
        'torch.LongTensor'
    """
    if isinstance(o, torch.Tensor):
        return o.type()

    module = ""
    class_name = ""
    if (
        hasattr(o, "__module__")
        and o.__module__ != "builtins"
        and o.__module__ != "__builtin__"
        and o.__module__ is not None
    ):
        module = o.__module__ + "."

    if hasattr(o, "__qualname__"):
        class_name = o.__qualname__
    elif hasattr(o, "__name__"):
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


_GLOBAL_DEVICE_CONTEXT = threading.local()


def get_default_device() -> "torch.device":
    r"""Gets the default ``torch.Tensor`` to be allocated on ``device``"""
    global _GLOBAL_DEVICE_CONTEXT
    if hasattr(_GLOBAL_DEVICE_CONTEXT, "device_context"):
        device = _GLOBAL_DEVICE_CONTEXT.device_context.device
        if device.index is not None:
            return device
        else:
            # TODO: Call like get_device_index() method corresponding to
            # each device type
            return torch.tensor([]).device
    else:
        return torch.device("cpu")


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

    .. note::

        This doesn't affect functions that create tensors that share the same memory as the input, like:
        :func:`torch.from_numpy` and :func:`torch.frombuffer`

    Args:
        device (device or string): the device to set as default

    Example::

        >>> # xdoctest: +SKIP("requires cuda, changes global state")
        >>> torch.get_default_device()
        device(type='cpu')
        >>> torch.set_default_device('cuda')  # current device is 0
        >>> torch.get_default_device()
        device(type='cuda', index=0)
        >>> torch.set_default_device('cuda')
        >>> torch.cuda.set_device('cuda:1')  # current device is 1
        >>> torch.get_default_device()
        device(type='cuda', index=1)
        >>> torch.set_default_device('cuda:1')
        >>> torch.get_default_device()
        device(type='cuda', index=1)

    """
    global _GLOBAL_DEVICE_CONTEXT
    if hasattr(_GLOBAL_DEVICE_CONTEXT, "device_context"):
        device_context = _GLOBAL_DEVICE_CONTEXT.device_context
        if device_context is not None:
            device_context.__exit__(None, None, None)

    if device is None:
        device_context = None
    else:
        from torch.utils._device import DeviceContext

        device_context = DeviceContext(device)
        device_context.__enter__()
    _GLOBAL_DEVICE_CONTEXT.device_context = device_context


def set_default_tensor_type(t):
    r"""
    .. warning::

        This function is deprecated as of PyTorch 2.1, please use :func:`torch.set_default_dtype()` and
        :func:`torch.set_default_device()` as alternatives.

    Sets the default ``torch.Tensor`` type to floating point tensor type
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

    Sets the default floating point dtype to :attr:`d`. Supports floating point dtype
    as inputs. Other dtypes will cause torch to raise an exception.

    When PyTorch is initialized its default floating point dtype is torch.float32,
    and the intent of set_default_dtype(torch.float64) is to facilitate NumPy-like
    type inference. The default floating point dtype is used to:

    1. Implicitly determine the default complex dtype. When the default floating type is float16,
       the default complex dtype is complex32. For float32, the default complex dtype is complex64.
       For float64, it is complex128. For bfloat16, an exception will be raised because
       there is no corresponding complex type for bfloat16.
    2. Infer the dtype for tensors constructed using Python floats or complex Python
       numbers. See examples below.
    3. Determine the result of type promotion between bool and integer tensors and
       Python floats and complex Python numbers.

    Args:
        d (:class:`torch.dtype`): the floating point dtype to make the default.

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

        >>> torch.set_default_dtype(torch.float16)
        >>> # Python floats are now interpreted as float16
        >>> torch.tensor([1.2, 3]).dtype    # a new floating point tensor
        torch.float16
        >>> # Complex Python numbers are now interpreted as complex128
        >>> torch.tensor([1.2, 3j]).dtype   # a new complex tensor
        torch.complex32

    """
    _C._set_default_dtype(d)


def use_deterministic_algorithms(
    mode: builtins.bool,
    *,
    warn_only: builtins.bool = False,
) -> None:
    r"""Sets whether PyTorch operations must use "deterministic"
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
        * :class:`torch.nn.ReplicationPad2d` when attempting to differentiate a CUDA tensor
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

    In addition, several operations fill uninitialized memory when this setting
    is turned on and when
    :attr:`torch.utils.deterministic.fill_uninitialized_memory` is turned on.
    See the documentation for that attribute for more information.

    A handful of CUDA operations are nondeterministic if the CUDA version is
    10.2 or greater, unless the environment variable ``CUBLAS_WORKSPACE_CONFIG=:4096:8``
    or ``CUBLAS_WORKSPACE_CONFIG=:16:8`` is set. See the CUDA documentation for more
    details: `<https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility>`_
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
        raise TypeError(f"debug_mode must be str or int, but got {type(debug_mode)}")

    if isinstance(debug_mode, str):
        if debug_mode == "default":
            debug_mode = 0
        elif debug_mode == "warn":
            debug_mode = 1
        elif debug_mode == "error":
            debug_mode = 2
        else:
            raise RuntimeError(
                "invalid value of debug_mode, expected one of `default`, "
                f"`warn`, `error`, but got {debug_mode}"
            )

    if debug_mode == 0:
        _C._set_deterministic_algorithms(False)
    elif debug_mode == 1:
        _C._set_deterministic_algorithms(True, warn_only=True)
    elif debug_mode == 2:
        _C._set_deterministic_algorithms(True)
    else:
        raise RuntimeError(
            "invalid value of debug_mode, expected 0, 1, or 2, " f"but got {debug_mode}"
        )


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
          bits with 23 bits explicitly stored) for internal computations.
        * "high", float32 matrix multiplications either use the TensorFloat32 datatype (10
          mantissa bits explicitly stored) or treat each float32 number as the sum of two bfloat16 numbers
          (approximately 16 mantissa bits with 14 bits explicitly stored), if the appropriate fast matrix multiplication
          algorithms are available.  Otherwise float32 matrix multiplications are computed
          as if the precision is "highest".  See below for more information on the bfloat16
          approach.
        * "medium", float32 matrix multiplications use the bfloat16 datatype (8 mantissa
          bits with 7 bits explicitly stored) for internal computations, if a fast matrix multiplication algorithm
          using that datatype internally is available. Otherwise float32
          matrix multiplications are computed as if the precision is "high".

    When using "high" precision, float32 multiplications may use a bfloat16-based algorithm
    that is more complicated than simply truncating to some smaller number mantissa bits
    (e.g. 10 for TensorFloat32, 7 for bfloat16 explicitly stored).  Refer to [Henry2019]_ for a complete
    description of this algorithm.  To briefly explain here, the first step is to realize
    that we can perfectly encode a single float32 number as the sum of three bfloat16
    numbers (because float32 has 23 mantissa bits while bfloat16 has 7 explicitly stored, and both have the
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


def _check_with(
    error_type,
    cond: Union[builtins.bool, SymBool],
    message: Callable[[], str],
):  # noqa: F811
    if not isinstance(cond, (builtins.bool, torch.SymBool)):
        raise TypeError(f"cond must be a bool, but got {type(cond)}")

    from torch.fx.experimental.symbolic_shapes import expect_true

    if expect_true(cond):
        return

    # error_type must be a subclass of Exception and not subclass of Warning
    assert issubclass(error_type, Exception) and not issubclass(error_type, Warning)

    if message is None:
        message_evaluated = (
            "Expected cond to be True, but got False. (Could this error "
            "message be improved? If so, please report an enhancement request "
            "to PyTorch.)"
        )

    else:
        if not callable(message):
            raise TypeError("message must be a callable")

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
    from torch.fx.experimental.symbolic_shapes import _advise_is_size

    _advise_is_size(i)


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
        raise TypeError(f"cond must be a tensor, but got {type(cond)}")

    if not cond.dtype == torch.bool:
        raise TypeError(f"cond tensor must have dtype torch.bool, but got {cond.dtype}")

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
from math import e, inf, nan, pi

newaxis: None = None

__all__.extend(["e", "pi", "nan", "inf", "newaxis"])

################################################################################
# Define Storage and Tensor classes
################################################################################

from torch._tensor import Tensor  # usort: skip

# needs to be after torch.Tensor is defined to avoid circular dependencies
from torch import storage as storage  # usort: skip
from torch.storage import (
    _LegacyStorage,
    _StorageBase,
    _warn_typed_storage_removal,
    TypedStorage,
    UntypedStorage,
)

# NOTE: New <type>Storage classes should never be added. When adding a new
# dtype, use torch.storage.TypedStorage directly.


class ByteStorage(_LegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.uint8


class DoubleStorage(_LegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.double


class FloatStorage(_LegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.float


class HalfStorage(_LegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.half


class LongStorage(_LegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.long


class IntStorage(_LegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.int


class ShortStorage(_LegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.short


class CharStorage(_LegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.int8


class BoolStorage(_LegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.bool


class BFloat16Storage(_LegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.bfloat16


class ComplexDoubleStorage(_LegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.cdouble


class ComplexFloatStorage(_LegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.cfloat


class QUInt8Storage(_LegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.quint8


class QInt8Storage(_LegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.qint8


class QInt32Storage(_LegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.qint32


class QUInt4x2Storage(_LegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.quint4x2


class QUInt2x4Storage(_LegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.quint2x4


_storage_classes = {
    UntypedStorage,
    DoubleStorage,
    FloatStorage,
    LongStorage,
    IntStorage,
    ShortStorage,
    CharStorage,
    ByteStorage,
    HalfStorage,
    BoolStorage,
    QUInt8Storage,
    QInt8Storage,
    QInt32Storage,
    BFloat16Storage,
    ComplexFloatStorage,
    ComplexDoubleStorage,
    QUInt4x2Storage,
    QUInt2x4Storage,
    TypedStorage,
}

# The _tensor_classes set is initialized by the call to initialize_python_bindings.
_tensor_classes: Set[Type] = set()

# If you edit these imports, please update torch/__init__.py.in as well
from torch import amp as amp, random as random, serialization as serialization
from torch._tensor_str import set_printoptions
from torch.amp import autocast, GradScaler
from torch.random import get_rng_state, initial_seed, manual_seed, seed, set_rng_state
from torch.serialization import load, save

################################################################################
# Initialize extension
################################################################################


# Shared memory manager needs to know the exact location of manager executable
def _manager_path():
    if _running_with_deploy() or platform.system() == "Windows":
        return b""
    path = get_file_path("torch", "bin", "torch_shm_manager")
    prepare_multiprocessing_environment(get_file_path("torch"))
    if not os.path.exists(path):
        raise RuntimeError("Unable to find torch_shm_manager at " + path)
    return path.encode("utf-8")


_C._initExtension(_manager_path())

del _manager_path

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
    del segment_reduce  # noqa: F821

# Ops not to be exposed in `torch` namespace,
# mostly helper ops.
PRIVATE_OPS = ("unique_dim",)

__name, __obj = "", None
for __name in dir(_C._VariableFunctions):
    if __name.startswith("__") or __name in PRIVATE_OPS:
        continue
    __obj = getattr(_C._VariableFunctions, __name)
    __obj.__module__ = __name__  # "torch"
    # Hide some APIs that should not be public
    if __name == "segment_reduce":
        # TODO: Once the undocumented FC window is passed, remove the line bellow
        globals()[__name] = __obj
        __name = "_" + __name
    globals()[__name] = __obj
    if not __name.startswith("_"):
        __all__.append(__name)

del __name, __obj

################################################################################
# Add torch.dtype instances to the public API
################################################################################

import torch

__all__.extend(
    name for name in dir(torch) if isinstance(getattr(torch, name), torch.dtype)
)

################################################################################
# Import TorchDynamo's lazy APIs to avoid circular dependenices
################################################################################

# needs to be before from torch.functional import * to avoid circular dependencies
from torch._compile import _disable_dynamo  # usort: skip

################################################################################
# Import interface functions defined in Python
################################################################################

# needs to be after the above ATen bindings so we can overwrite from Python side
from torch import functional as functional  # usort: skip
from torch.functional import *  # usort: skip # noqa: F403

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
    r"""A wrapper around Python's assert which is symbolically traceable."""
    if type(condition) is not torch.Tensor and overrides.has_torch_function(
        (condition,)
    ):
        return overrides.handle_torch_function(
            _assert, (condition,), condition, message
        )
    assert condition, message


################################################################################
# Import most common subpackages
################################################################################

# Use the redundant form so that type checkers know that these are a part of
# the public API. The "regular" import lines are there solely for the runtime
# side effect of adding to the imported module's members for other users.

# needs to be before import torch.nn as nn to avoid circular dependencies
from torch.autograd import (  # usort: skip
    enable_grad as enable_grad,
    inference_mode as inference_mode,
    no_grad as no_grad,
    set_grad_enabled as set_grad_enabled,
)

from torch import (
    __config__ as __config__,
    __future__ as __future__,
    _awaits as _awaits,
    autograd as autograd,
    backends as backends,
    cpu as cpu,
    cuda as cuda,
    distributions as distributions,
    fft as fft,
    futures as futures,
    hub as hub,
    jit as jit,
    linalg as linalg,
    mps as mps,
    mtia as mtia,
    multiprocessing as multiprocessing,
    nested as nested,
    nn as nn,
    optim as optim,
    overrides as overrides,
    profiler as profiler,
    sparse as sparse,
    special as special,
    testing as testing,
    types as types,
    utils as utils,
    xpu as xpu,
)
from torch.signal import windows as windows

# Quantized, sparse, AO, etc. should be last to get imported, as nothing
# is expected to depend on them.
from torch import ao as ao  # usort: skip

# nn.quant* depends on ao -- so should be after those.
import torch.nn.intrinsic
import torch.nn.qat
import torch.nn.quantizable
import torch.nn.quantized

_C._init_names(list(_storage_classes))

# attach docstrings to torch and tensor functions
from torch import _size_docs, _storage_docs, _tensor_docs, _torch_docs

del _torch_docs, _tensor_docs, _storage_docs, _size_docs


def compiled_with_cxx11_abi() -> builtins.bool:
    r"""Returns whether PyTorch was built with _GLIBCXX_USE_CXX11_ABI=1"""
    return _C._GLIBCXX_USE_CXX11_ABI


import torch._library

# Import the ops "namespace"
from torch._classes import classes as classes
from torch._ops import ops as ops  # usort: skip

# quantization depends on torch.fx and torch.ops
# Import quantization
from torch import quantization as quantization  # usort: skip

# Import the quasi random sampler
from torch import quasirandom as quasirandom  # usort: skip

# If you are seeing this, it means that this call site was not checked if
# the memory format could be preserved, and it was switched to old default
# behaviour of contiguous
legacy_contiguous_format = contiguous_format  # defined by _C._initExtension()

# Register fork handler to initialize OpenMP in child processes (see gh-28389)
from torch.multiprocessing._atfork import register_after_fork

register_after_fork(torch.get_num_threads)
del register_after_fork

# Import tools that require fully imported torch (for applying
# torch.jit.script as a decorator, for instance):
from torch._lobpcg import lobpcg as lobpcg

# These were previously defined in native_functions.yaml and appeared on the
# `torch` namespace, but we moved them to c10 dispatch to facilitate custom
# class usage. We add these lines here to preserve backward compatibility.
quantized_lstm = ops.aten.quantized_lstm
quantized_gru = ops.aten.quantized_gru

# Import experimental masked operations support. See
# [RFC-0016](https://github.com/pytorch/rfcs/pull/27) for more
# information.
from torch import masked as masked

# Import removed ops with error message about removal
from torch._linalg_utils import (  # type: ignore[misc]
    _symeig as symeig,
    eig,
    lstsq,
    matrix_rank,
    solve,
)

from torch.utils.dlpack import from_dlpack, to_dlpack


class _TorchCompileInductorWrapper:
    compiler_name = "inductor"

    def __init__(self, mode, options, dynamic):
        self.config: Dict[str, Any] = dict()
        self.dynamic = dynamic
        self.apply_mode(mode)
        self.apply_options(options)

        # Stash the compiler_fn to be used for backend match guard.
        from torch._inductor.compile_fx import compile_fx

        self.compiler_fn = compile_fx
        if self.config.get("triton.cudagraphs", False):
            os.environ["DISABLE_CUPTI_LAZY_REINIT"] = "1"
            # FIXME: CUDA Graph does not work well with CUPTI teardown.
            #   1) crashes on 1st lazy CUPTI re-init after teardown (CUDA 11)
            #   2) crashes on 2nd non-lazy CUPTI re-init after teardown (CUDA 12)
            # Workaround: turn off CUPTI teardown when using CUDA Graphs.
            os.environ["TEARDOWN_CUPTI"] = "0"

    def __eq__(self, other):
        return (
            isinstance(other, _TorchCompileInductorWrapper)
            and self.config == other.config
            and self.dynamic == other.dynamic
        )

    def apply_mode(self, mode: Optional[str]):
        if mode is None or mode == "default":
            pass
        elif mode in {"reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"}:
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

        current_config: Dict[str, Any] = config.shallow_copy_dict()

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
        return (
            isinstance(other, _TorchCompileWrapper)
            and self.compiler_fn == other.compiler_fn
            and self.kwargs == other.kwargs
            and self.dynamic == other.dynamic
        )

    def __call__(self, model_, inputs_):
        return self.compiler_fn(model_, inputs_, **self.kwargs)

    def reset(self):
        if hasattr(self.compiler_fn, "reset"):
            self.compiler_fn.reset()


def compile(
    model: Optional[Callable] = None,
    *,
    fullgraph: builtins.bool = False,
    dynamic: Optional[builtins.bool] = None,
    backend: Union[str, Callable] = "inductor",
    mode: Union[str, None] = None,
    options: Optional[Dict[str, Union[str, builtins.int, builtins.bool]]] = None,
    disable: builtins.bool = False,
) -> Callable:
    """
    Optimizes given model/function using TorchDynamo and specified backend.
    If you are compiling an :class:`torch.nn.Module`, you can also use :meth:`torch.nn.Module.compile`
    to compile the module inplace without changing its structure.

    Concretely, for every frame executed within the compiled region, we will attempt
    to compile it and cache the compiled result on the code object for future
    use.  A single frame may be compiled multiple times if previous compiled
    results are not applicable for subsequent calls (this is called a "guard
    failure), you can use TORCH_LOGS=guards to debug these situations.
    Multiple compiled results can be associated with a frame up to
    ``torch._dynamo.config.cache_size_limit``, which defaults to 8; at which
    point we will fall back to eager.  Note that compile caches are per
    *code object*, not frame; if you dynamically create multiple copies of a
    function, they will all share the same code cache.

    Args:
       model (Callable): Module/function to optimize
       fullgraph (bool): If False (default), torch.compile attempts to discover compileable regions
        in the function that it will optimize. If True, then we require that the entire function be
        capturable into a single graph. If this is not possible (that is, if there are graph breaks),
        then this will raise an error.
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

        - To register an out-of-tree custom backend:
       https://pytorch.org/docs/main/torch.compiler_custom_backends.html#registering-custom-backends
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
    if sys.version_info >= (3, 13):
        raise RuntimeError("Dynamo is not supported on Python 3.13+")

    # Decorator mode
    if model is None:

        def fn(model: Callable):
            if model is None:
                raise RuntimeError("Model can't be None")
            return compile(
                model,
                fullgraph=fullgraph,
                dynamic=dynamic,
                backend=backend,
                mode=mode,
                options=options,
                disable=disable,
            )

        return fn

    if mode is not None and options is not None:
        raise RuntimeError(
            "Either mode or options can be specified, but both can't be specified at the same time."
        )
    if mode is None and options is None:
        mode = "default"
    if backend == "inductor":
        backend = _TorchCompileInductorWrapper(mode, options, dynamic)
    else:
        backend = _TorchCompileWrapper(backend, mode, options, dynamic)

    return torch._dynamo.optimize(
        backend=backend,
        nopython=fullgraph,
        dynamic=dynamic,
        disable=disable,
    )(model)


from torch import export as export

from torch._higher_order_ops import cond as cond


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
        raise RuntimeError(
            f"The runtime module of '{device_type}' has already "
            f"been registered with '{getattr(m, device_type)}'"
        )
    setattr(m, device_type, module)
    torch_module_name = ".".join([__name__, device_type])
    sys.modules[torch_module_name] = module


# expose return_types
from torch import library as library, return_types as return_types

if not TYPE_CHECKING:
    from torch import _meta_registrations

# Enable CUDA Sanitizer
if "TORCH_CUDA_SANITIZER" in os.environ:
    import torch.cuda._sanitizer as csan

    csan.enable_cuda_sanitizer()

# Populate magic methods on SymInt and SymFloat
import torch.fx.experimental.sym_node

from torch import func as func
from torch.func import vmap as vmap


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
    from torch import _dynamo as _dynamo, _inductor as _inductor, onnx as onnx

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

            warnings.warn(
                f"'{name}' is deprecated, please use '{replacement.__module__}.{replacement.__name__}()'",
                stacklevel=2,
            )
            return replacement()

        # Lazy modules
        if name in _lazy_modules:
            return importlib.import_module(f".{name}", __name__)

        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def get_device_module(device: Optional[Union[torch.device, str]] = None):
    """
    Returns the module associated with a given device(e.g., torch.device('cuda'), "mtia:0", "xpu", ...).
    If no device is given, return the module for the current accelerator or CPU if none is present.
    """
    if isinstance(device, torch.device):
        device_module_name = device.type
    elif isinstance(device, str):
        device_module_name = torch.device(device).type
    elif device is None:
        # Using default accelerator type. If no accelerator is available, it automatically returns CPU device.
        device_module_name = torch._C._get_accelerator().type
    else:
        raise RuntimeError(
            f"Invalid value of device '{device}', expect torch.device, str, or None"
        )
    device_module = getattr(torch, device_module_name, None)
    if device_module is None:
        raise RuntimeError(
            f"Device '{device_module_name}' does not have a corresponding module registered as 'torch.{device_module_name}'."
        )
    return device_module


def _constrain_as_size(
    symbol,
    min: Optional[builtins.int] = None,
    max: Optional[builtins.int] = None,
):
    """
    This indicates that a given int is size-like, and can be used in any context where a size is expected.
    You will typically use this when reading out integers from Tensors, e.g., max.item() or lengths.tolist()
    which then need to be used as tensor constructors. Providing these assertions to PyTorch can help resolve
      GuardOnDataDependentSymNode errors upon export, since we cannot guard on unbacked SymInts.

    This function has unusual semantics in some circumstances in framework
    code, we will treat this int as >= 2 (when we do a size-oblivious guard).
    This makes it easier to use the unbacked int in size contexts,
    as we will often attempt to guard on a size being zero/one
    (e.g., when computing the contiguity of a tensor, or testing if
    broadcasting can occur), which will not work on unbacked SymInts.
    However, if we conservatively assume that the size is not zero/one, we will
    end up with a graph that will still work even if the size is zero/one.

    For more details, see https://docs.google.com/document/d/1HSuTTVvYH1pTew89Rtpeu84Ht3nQEFTYhAX3Ypa_xJs/edit
    ```
    """
    torch.sym_constrain_range_for_size(symbol, min=min, max=max)


from torch import _logging

_logging._init_logs()
