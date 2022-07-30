import functools
from typing import Callable

from torchgen.api.autograd import NativeFunctionWithDifferentiabilityInfo as NFWDI
from torchgen.context import native_function_manager
from torchgen.utils import T

# Like tools.api.context.with_native_function, but for
# NativeFunctionWithDifferentiabilityInfo.
def with_native_function_with_differentiability_info(
    func: Callable[[NFWDI], T]
) -> Callable[[NFWDI], T]:
    @functools.wraps(func)
    def wrapper(f: NFWDI) -> T:
        with native_function_manager(f.func):
            return func(f)

    return wrapper
