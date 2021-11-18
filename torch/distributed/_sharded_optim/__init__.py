from typing import Iterator, Tuple, Union
from .api import ShardedOptimizer

import torch.nn as nn

from torch.distributed._sharded_tensor import (
    ShardedTensor
)

def module_named_params_with_sharded_tensor(
    module: nn.Module,
    prefix: str = '',
    recurse: bool = True,
) -> Iterator[Tuple[str, Union[nn.Parameter, ShardedTensor]]]:

    r"""Returns an iterator over module parameters (together with the
    ShardedTensor parameters), yielding both the name of the parameter
    as well as the parameter itself. This is typically passed to an
    ShardedOptimizer

    Args:
        prefix (str): prefix to prepend to all parameter names.
        recurse (bool): if True, then yields parameters of this module
            and all submodules. Otherwise, yields only parameters that
            are direct members of this module.

    Yields:
        (string, Union[Parameter, ShardedTensor]): Tuple containing
            the name and parameter (or ShardedTensor parameter)

    Example::

        >>> for name, param in self.named_parameters():
        >>>    if name in ['bias']:
        >>>        print(param.size())

    """
    modules = module.named_modules(prefix=prefix) if recurse else [(prefix, module)]

    memo = set()
    for mod_prefix, mod in modules:
        # find all sharded tensor params
        for name, val in vars(mod).items():
            if name.startswith("__"):
                continue

            if val is None:
                continue

            if isinstance(val, ShardedTensor):
                memo.add(val)
                name = mod_prefix + ('.' if mod_prefix else '') + name
                yield name, val

        # find all nn.Parameters
        for name, val in mod._parameters.items():
            if val is None or val in memo:
                continue
            memo.add(val)
            name = mod_prefix + ('.' if mod_prefix else '') + name
            yield name, val
