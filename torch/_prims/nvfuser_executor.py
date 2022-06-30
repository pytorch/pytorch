from dataclasses import dataclass
from functools import lru_cache

import torch

from torch.fx import GraphModule
from torch._prims.utils import getnvFuserDtype, Number
import torch.overrides
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten

if torch.cuda.is_available():
    from torch._C._nvfuser import DataType, Fusion, FusionDefinition  # type: ignore[import]


# nvFuserTensorViewTemplate and nvFuserScalarValTemplate are helper objects
# for cached construction of the nvFuser's Fusion
# TODO: change what is stored in the cache for nvFuser's TensorView objects
# https://github.com/pytorch/pytorch/issues/80551
@dataclass(frozen=True)
class nvFuserTensorViewTemplate:
    size: tuple
    stride: tuple
    dtype: DataType


@dataclass(frozen=True)
class nvFuserScalarValTemplate:
    dtype: DataType


def to_nvfuser_template_args(args):
    def to_nvfuser(arg):
        if isinstance(arg, torch.Tensor):
            return nvFuserTensorViewTemplate(
                arg.size(), arg.stride(), getnvFuserDtype(arg.dtype)
            )
        elif isinstance(arg, Number):
            return nvFuserScalarValTemplate(getnvFuserDtype(type(arg)))
        else:
            return arg

    return tree_map(to_nvfuser, args)


# MyPy bug: https://github.com/python/mypy/issues/5107
@lru_cache  # type: ignore[arg-type]
def make_nvfuser_fusion(gm: GraphModule, *nv_args_templates):
    # PROTOTYPE nvfuser executor
    # Everything in the graph must support nvfuser
    for node in gm.graph.nodes:
        if (
            node.op == "call_function"
            and getattr(node.target, "impl_nvfuser", None) is None
        ):
            raise ValueError(
                "All call_function nodes in the graph must support nvfuser. "
                f"Node {node} does not support nvfuser"
            )

    fusion = Fusion()
    with FusionDefinition(fusion) as fd:

        def _to_nvfuser_constant(arg):
            if isinstance(arg, Number):
                return fd.define_constant(arg)
            else:
                return arg

        class FusionInterpreter(torch.fx.Interpreter):
            def call_function(self, target, args, kwargs):
                args = tuple(map(_to_nvfuser_constant, args))
                target = target.impl_nvfuser
                args = (fd,) + args
                return target(*args, **kwargs)

        def templates_to_nvfuser_inputs(arg):
            if isinstance(arg, nvFuserTensorViewTemplate):
                x = fd.define_tensor(arg.size, arg.stride, arg.dtype)
                fd.add_input(x)
                return x
            elif isinstance(arg, nvFuserScalarValTemplate):
                x = fd.define_scalar(arg.dtype)
                fd.add_input(x)
                return x
            else:
                return arg

        # Transforms graph to call nvfuser lowerings
        nv_args = tuple(map(templates_to_nvfuser_inputs, nv_args_templates))
        out = FusionInterpreter(gm).run(*nv_args)
        flat_out, unflatten_spec = tree_flatten(out)
        for o in flat_out:
            fd.add_output(o)

    return fusion, unflatten_spec


def nvfuser_execute(gm: GraphModule, *args):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Attempting to use nvFuser trace executor but CUDA is not available!"
        )

    flat_args, _ = tree_flatten(args)

    # Construction of the fusion is expensive and cached based on the GraphModule
    # and symbolic nvFuser args.
    nv_template_args = to_nvfuser_template_args(flat_args)
    fusion, unflatten_spec = make_nvfuser_fusion(gm, *nv_template_args)  # type: ignore[misc]

    # Inputs to fusion.execute correspond to the same template/symbolic inputs marked with `fd.add_input`
    concrete_fusion_inputs = tuple(
        arg for arg in flat_args if isinstance(arg, (torch.Tensor, Number))
    )

    return tree_unflatten(
        fusion.execute(concrete_fusion_inputs),  # type: ignore[has-type]
        unflatten_spec,  # type: ignore[has-type]
    )
