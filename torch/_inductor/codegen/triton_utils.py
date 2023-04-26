from triton.runtime.jit import JITFunction

from ..._dynamo import config as dynamo_config
from .. import config

from ..utils import instance_descriptor

from ..virtualized import V

from .common import SizeArg, TensorArg


def signature_of(arg):
    if isinstance(arg, TensorArg):
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
        return JITFunction._key_of(V.graph.sizevars.size_hint(arg.expr))
    raise NotImplementedError(f"unhandled {type(arg)}: {arg}")


def config_of(args):
    from ..compile_fx import ALIGNMENT

    def is_aligned(x):
        if isinstance(x, TensorArg):
            return x.buffer not in V.graph.unaligned_buffers
        if isinstance(x, SizeArg):
            return V.graph.sizevars.maybe_guard_multiple_of(x.expr, ALIGNMENT)
        raise NotImplementedError(f"unhandled {type(x)}: {x}")

    if config.triton.divisible_by_16 and not dynamo_config.dynamic_shapes:
        divisible_by_16 = [i for i, arg in enumerate(args) if is_aligned(arg)]
    else:
        divisible_by_16 = []
    return instance_descriptor(tuple(divisible_by_16), ())
