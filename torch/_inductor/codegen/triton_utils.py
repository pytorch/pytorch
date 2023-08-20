from .. import config
from ..utils import instance_descriptor
from ..virtualized import V
from .common import SizeArg, TensorArg


def signature_of(arg, *, size_dtype: str):
    from triton.runtime.jit import JITFunction

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
        if size_dtype == "tl.int32":
            return "i32"
        elif size_dtype == "tl.int64":
            return "i64"
        else:
            raise NotImplementedError(f"unhandled size_dtype {size_dtype}")
    raise NotImplementedError(f"unhandled {type(arg)}: {arg}")


def signature_to_meta(signature, *, size_dtype: str):
    return {
        i: signature_of(arg, size_dtype=size_dtype) for i, arg in enumerate(signature)
    }


def config_of(args):
    from ..compile_fx import ALIGNMENT

    def is_aligned(x):
        if isinstance(x, TensorArg):
            return x.buffer not in V.graph.unaligned_buffers
        if isinstance(x, SizeArg):
            # TODO(voz): These are kinda redundant, if we can solve out statically_known_multiple_of with
            # _maybe_evaluate_static...
            if x.name.startswith("load_seed_offset"):
                return False
            else:
                return V.graph.sizevars.statically_known_multiple_of(x.expr, ALIGNMENT)
        raise NotImplementedError(f"unhandled {type(x)}: {x}")

    if config.triton.divisible_by_16:
        divisible_by_16 = [i for i, arg in enumerate(args) if is_aligned(arg)]
    else:
        divisible_by_16 = []
    return instance_descriptor(tuple(divisible_by_16), ())
