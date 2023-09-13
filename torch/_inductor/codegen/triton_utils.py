from typing import Dict, List, Union

import torch

from .. import config
from ..utils import instance_descriptor
from ..virtualized import V
from .common import SizeArg, TensorArg


def signature_of(arg: Union[TensorArg, SizeArg], *, size_dtype: str) -> str:
    from triton.runtime.jit import JITFunction

    if isinstance(arg, TensorArg):
        # TODO: Remove fp8 special handling when Triton supports PyTorch fp8 dtypes.
        # Related PR: https://github.com/openai/triton/pull/2279/
        if arg.dtype == torch.float8_e4m3fn:
            tye = "fp8e4nv"
        elif arg.dtype == torch.float8_e5m2:
            tye = "fp8e5"
        else:
            tye = JITFunction._type_of(arg.dtype)
        if V.graph.is_unspec_arg(arg.buffer):
            # had unwrapped 0d tensor as scalar
            new_tye = tye.lstrip("*")
            if new_tye in ["fp16", "bf16"]:
                return "fp32"
            else:
                return new_tye
        else:
            return tye
    if isinstance(arg, SizeArg):
        if size_dtype == "tl.int32":
            return "i32"
        elif size_dtype == "tl.int64":
            return "i64"
        else:
            raise NotImplementedError(f"unhandled size_dtype {size_dtype}")
    raise NotImplementedError(f"unhandled {type(arg)}: {arg}")


def signature_to_meta(
    signature: List[Union[TensorArg, SizeArg]], *, size_dtype: str
) -> Dict[int, str]:
    return {
        i: signature_of(arg, size_dtype=size_dtype) for i, arg in enumerate(signature)
    }


def config_of(args: List[Union[TensorArg, SizeArg]]) -> instance_descriptor:
    def is_aligned(
        x: Union[TensorArg, SizeArg], alignment: int, include_tensor: bool
    ) -> bool:
        """
        Roughly follow triton code here:
        https://github.com/openai/triton/blob/5282ed890d453e10b9ee30076ef89115dd197761/python/triton/runtime/jit.py#L208-L222
        """
        if isinstance(x, TensorArg):
            if include_tensor:
                return not V.graph.scheduler.is_unaligned_buffer(x.buffer)
            else:
                return False
        if isinstance(x, SizeArg):
            # TODO(voz): These are kinda redundant, if we can solve out statically_known_multiple_of with
            # _maybe_evaluate_static...
            if x.name.startswith("load_seed_offset"):
                return False
            else:
                return V.graph.sizevars.statically_known_multiple_of(x.expr, alignment)
        raise NotImplementedError(f"unhandled {type(x)}: {x}")

    if config.triton.divisible_by_16:
        divisible_by_16 = tuple(
            i
            for i, arg in enumerate(args)
            if is_aligned(arg, alignment=16, include_tensor=True)
        )
    else:
        divisible_by_16 = ()
    divisible_by_8 = tuple(
        i
        for i, arg in enumerate(args)
        if is_aligned(arg, alignment=8, include_tensor=False)
    )
    return instance_descriptor(divisible_by_16, (), (), divisible_by_8)
