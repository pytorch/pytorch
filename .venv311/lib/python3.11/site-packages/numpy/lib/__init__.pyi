from numpy._core.function_base import add_newdoc
from numpy._core.multiarray import add_docstring, tracemalloc_domain

# all submodules of `lib` are accessible at runtime through `__getattr__`,
# so we implicitly re-export them here
from . import _array_utils_impl as _array_utils_impl
from . import _arraypad_impl as _arraypad_impl
from . import _arraysetops_impl as _arraysetops_impl
from . import _arrayterator_impl as _arrayterator_impl
from . import _datasource as _datasource
from . import _format_impl as _format_impl
from . import _function_base_impl as _function_base_impl
from . import _histograms_impl as _histograms_impl
from . import _index_tricks_impl as _index_tricks_impl
from . import _iotools as _iotools
from . import _nanfunctions_impl as _nanfunctions_impl
from . import _npyio_impl as _npyio_impl
from . import _polynomial_impl as _polynomial_impl
from . import _scimath_impl as _scimath_impl
from . import _shape_base_impl as _shape_base_impl
from . import _stride_tricks_impl as _stride_tricks_impl
from . import _twodim_base_impl as _twodim_base_impl
from . import _type_check_impl as _type_check_impl
from . import _ufunclike_impl as _ufunclike_impl
from . import _utils_impl as _utils_impl
from . import _version as _version
from . import array_utils, format, introspect, mixins, npyio, scimath, stride_tricks
from ._arrayterator_impl import Arrayterator
from ._version import NumpyVersion

__all__ = [
    "Arrayterator",
    "add_docstring",
    "add_newdoc",
    "array_utils",
    "format",
    "introspect",
    "mixins",
    "NumpyVersion",
    "npyio",
    "scimath",
    "stride_tricks",
    "tracemalloc_domain",
]
