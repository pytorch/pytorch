import copy
import sys
import functools
from typing import TYPE_CHECKING
import unittest

import torch
import torch._inductor.test_case
import torch.fx.traceback as fx_traceback
import torch.utils.checkpoint
from torch._dynamo.backends.common import aot_autograd
from torch._functorch._aot_autograd.autograd_cache import BundledCompiledForward
from torch._guards import detect_fake_mode
from torch._inductor.output_code import RegionalOutputCode
from torch._inductor.test_case import run_tests
from torch._inductor.utils import run_fw_bw_and_get_code
from torch.fx._graph_pickler import GraphPickler
from torch.fx.passes.regional_inductor import regional_inductor
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.testing import FileCheck
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    skipIfTorchDynamo,
)
from torch.testing._internal.triton_utils import requires_cuda_and_triton


def aot_eager_regional_inductor(on_invoke_subgraph=False):
    def regional_inductor_fn(gm, *args, **kwargs):
        if on_invoke_subgraph:
            from torch.fx.passes.regional_inductor_invoke_subgraph import (
                regional_inductor_invoke_subgraph,
            )
            return regional_inductor_invoke_subgraph(gm, *args, **kwargs)
        else:
            return regional_inductor(gm, *args, **kwargs)

    kwargs = {}

    return aot_autograd(
        fw_compiler=regional_inductor_fn,
        bw_compiler=regional_inductor_fn,
        **kwargs,
    )


def _squared(score, b, h, m, n):
    return score * score


def mask_mod(b, h, q, k):
    return q >= 0


a = 12
b = 64
block_mask = create_block_mask(mask_mod, None, None, a * b, a * b)

x = torch.randn(
    1,
    1,
    a * b,
    b,
    dtype=torch.bfloat16,
    device="cuda",
    requires_grad=True,
)


def main(method):
    layer = 10

    if method == "annotate":

        def fn2(x):
            x = torch.sin(x)
            with fx_traceback.annotate({"compile_with_inductor": 0}):
                x = flex_attention(x, x, x, block_mask=block_mask, score_mod=_squared)
            return torch.cos(x)

        def fn(x):
            for i in range(layer):
                x = fn2(x)
            return x

        opt_fn = torch.compile(
            fn,
            backend=aot_eager_regional_inductor(),
            fullgraph=True,
        )

        _, codes = run_fw_bw_and_get_code(lambda: opt_fn(x))
    else:
        from torch._higher_order_ops.invoke_subgraph import (
            capture_invoke_subgraph_inductor_compile_gms,
            get_invoke_subgraph_compile_options,
        )

        torch._dynamo.config.enable_invoke_subgraph_regional_compile = True
        # must decompose aten.zeros.default, otherwise inductor complain
        # AssertionError: both a fallback and a decomp for same op: aten.zeros.default
        decomp_table = {}
        decomp_table[torch.ops.aten.zeros.default] = torch._decomp.decomposition_table[
            torch.ops.aten.zeros.default
        ]
        nested_config = get_invoke_subgraph_compile_options(decompositions=decomp_table)

        @torch.compiler.nested_compile_region(aot_config=nested_config)
        def f_flex_attention(x, y, z, block_mask, score_mod):
            x = flex_attention(x, y, z, block_mask=block_mask, score_mod=score_mod)
            return x

        def fn2(x):
            x = torch.sin(x)
            x = f_flex_attention(x, x, x, block_mask=block_mask, score_mod=_squared)
            return torch.cos(x)

        def fn(x):
            for i in range(layer):
                x = fn2(x)
            return x

        opt_fn = torch.compile(
            fn,
            backend=aot_eager_regional_inductor(on_invoke_subgraph=True),
            fullgraph=True,
        )

        _, codes = run_fw_bw_and_get_code(lambda: opt_fn(x))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <method>")
        sys.exit(1)
    method = sys.argv[1]
    main(method)
