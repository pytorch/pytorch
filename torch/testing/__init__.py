from torch._C import FileCheck as FileCheck

from . import _utils
from ._benchmark import benchmark_func as _benchmark_func
from ._comparison import assert_allclose, assert_close as assert_close
from ._creation import make_tensor as make_tensor
