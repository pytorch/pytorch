import unittest

import jax  # noqa: F401
import numpy as np
import torch
import torch.tpu
from torch._inductor import config as inductor_config
from torch._inductor.utils import run_and_get_code


class TestPallasTPUCodegen(unittest.TestCase):
    """E2E: compile (a + b) * 2.0 with TPUTensor and assert generated code."""

    @inductor_config.patch(
        {"cpu_backend": "pallas", "pallas_tpu_native": True, "fx_graph_cache": False}
    )
    def test_codegen_matches_expected(self):
        self.addCleanup(torch._dynamo.reset)

        x = torch.tpu.zeros((16,), dtype=torch.float32)
        y = torch.tpu.ones((16,), dtype=torch.float32)

        # Inputs must be on TPU
        self.assertEqual(x._jax_array.devices(), {jax.devices("tpu")[0]})
        self.assertEqual(y._jax_array.devices(), {jax.devices("tpu")[0]})

        def fn(a, b):
            return (a + b) * 2.0

        result, (code,) = run_and_get_code(torch.compile(fn), x, y)

        self.assertIsInstance(result, torch.tpu.TPUTensor)
        # Result must be on TPU
        self.assertEqual(result._jax_array.devices(), {jax.devices("tpu")[0]})
        # (0 + 1) * 2.0 = 2.0 for all elements
        np.testing.assert_array_equal(
            np.asarray(result._jax_array), np.full(16, 2.0, dtype=np.float32)
        )

        # Generated code must use TPU allocation, not CPU
        self.assertIn("empty_strided_tpu", code)
        self.assertNotIn("empty_strided_cpu(", code)
        self.assertNotIn("jax.device_put", code)
        self.assertNotIn(".cpu().numpy()", code)

        self.assertEqual(code, """\
# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p
import torch.tpu
empty_strided_tpu = torch.tpu.empty_strided
rand_strided_tpu = torch.tpu.rand_strided


# Topologically Sorted Source Nodes: [add, mul], Original ATen: [aten.add, aten.mul]
# Source node to ATen node mapping:
#   add => add
#   mul => mul
# Graph fragment:
#   %arg0_1 : Tensor "f32[16][1]cpu" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[16][1]cpu" = PlaceHolder[target=arg1_1]
#   %add : Tensor "f32[16][1]cpu"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
#   %mul : Tensor "f32[16][1]cpu"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, 2.0), kwargs = {})
#   return %mul
pallas_fused_add_mul_2c3548d8 = async_compile.pallas('pallas_fused_add_mul_2c3548d8', r\'\'\'
import functools
import math
import torch
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from torch._inductor.runtime.runtime_utils import pallas_partial_reduce, torch_dtype_to_jax_runtime
def pallas_fused_add_mul_2c3548d8_kernel(in_ptr0, in_ptr1, out_ptr0):
    # Define iteration variables as JAX arrays
    x0 = jnp.arange(16)
    tmp0 = in_ptr0[...]
    tmp1 = in_ptr1[...]
    tmp2 = tmp0 + tmp1
    tmp3 = jnp.array(2.0, dtype=jnp.float32)
    tmp4 = tmp2 * tmp3
    out_ptr0[...] = (jnp.full(out_ptr0.shape, tmp4) if jnp.asarray(tmp4).ndim == 0 else (jnp.broadcast_to(jnp.asarray(tmp4), out_ptr0.shape) if jnp.asarray(tmp4).size != out_ptr0.size else jnp.asarray(tmp4).reshape(out_ptr0.shape)))
@functools.partial(jax.jit, static_argnums=(0, 1,), donate_argnums=())
def pallas_fused_add_mul_2c3548d8_jit_wrapper(out_shapes, out_dtypes, in_ptr0, in_ptr1):
    out_specs = tuple(
        jax.ShapeDtypeStruct(shape, dtype)
        for shape, dtype in zip(out_shapes, out_dtypes)
    )
    return pl.pallas_call(
        pallas_fused_add_mul_2c3548d8_kernel,
        out_shape=out_specs,
        interpret=False,
        grid=(1,),
        input_output_aliases={},
    )(
        in_ptr0, in_ptr1,
    )
def pallas_fused_add_mul_2c3548d8_main(in_ptr0, in_ptr1, out_ptr0, stream=None):
    # Convert Torch -> JAX for in-place tensors
    # Convert Torch -> JAX for inputs
    in_ptr0_jax = in_ptr0._jax_array
    in_ptr1_jax = in_ptr1._jax_array
    # Prepare output metadata from PyTorch tensor
    out_shapes = (tuple(out_ptr0.shape),)
    out_dtypes = (torch_dtype_to_jax_runtime(out_ptr0.dtype),)
    res = pallas_fused_add_mul_2c3548d8_jit_wrapper(out_shapes, out_dtypes, in_ptr0_jax, in_ptr1_jax)
    result_values = res if isinstance(res, tuple) else (res,)
    out_ptr0._jax_array = result_values[0]
\'\'\')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1 = args
        args.clear()
        assert_size_stride(arg0_1, (16, ), (1, ))
        assert_size_stride(arg1_1, (16, ), (1, ))
        buf0 = empty_strided_tpu((16, ), (1, ), torch.float32)
        pallas_fused_add_mul_2c3548d8.run(arg0_1, arg1_1, buf0)
        del arg0_1
        del arg1_1
        return (buf0, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided_tpu((16, ), (1, ), torch.float32)
    arg1_1 = rand_strided_tpu((16, ), (1, ), torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
""")  # noqa: B950

    @inductor_config.patch(
        {"cpu_backend": "pallas", "pallas_tpu_native": True, "fx_graph_cache": False}
    )
    def test_graph_break_between_regions(self):
        """Two compiled regions separated by a graph break."""
        self.addCleanup(torch._dynamo.reset)

        x = torch.tpu.zeros((16,), dtype=torch.float32)
        y = torch.tpu.ones((16,), dtype=torch.float32)

        def fn(a, b):
            c = a + b
            torch._dynamo.graph_break()
            return c * 2.0

        result, codes = run_and_get_code(torch.compile(fn), x, y)

        # Graph break produces two compiled regions
        self.assertEqual(len(codes), 2)

        self.assertIsInstance(result, torch.tpu.TPUTensor)
        self.assertEqual(result._jax_array.devices(), {jax.devices("tpu")[0]})
        # (0 + 1) * 2.0 = 2.0 for all elements
        np.testing.assert_array_equal(
            np.asarray(result._jax_array), np.full(16, 2.0, dtype=np.float32)
        )

        # Both regions must use TPU allocation, not CPU
        for code in codes:
            self.assertIn("empty_strided_tpu", code)
            self.assertNotIn("empty_strided_cpu(", code)
            self.assertNotIn("jax.device_put", code)
            self.assertNotIn(".cpu().numpy()", code)

    def test_eager_mode_raises_error(self):
        """Eager compute operations on TPUTensor must raise RuntimeError."""
        x = torch.tpu.zeros((16,), dtype=torch.float32)
        y = torch.tpu.ones((16,), dtype=torch.float32)

        # Eager add should raise
        with self.assertRaises(RuntimeError) as ctx:
            _ = x + y
        self.assertIn("does not support eager operation", str(ctx.exception))
        self.assertIn("torch.compile", str(ctx.exception))

        # Eager mul should raise
        with self.assertRaises(RuntimeError) as ctx:
            _ = x * 2.0
        self.assertIn("does not support eager operation", str(ctx.exception))

        # Metadata access should NOT raise - these only access shape/dtype/device
        # metadata, not the actual tensor data. They go through __torch_function__
        # with __get__ which is explicitly allowed.
        self.assertEqual(x.shape, torch.Size([16]))
        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(x.device, torch.device("cpu"))
        self.assertEqual(x.dim(), 1)
        self.assertEqual(x.numel(), 16)


if __name__ == "__main__":
    unittest.main()
