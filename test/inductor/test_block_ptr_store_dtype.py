# Owner(s): ["module: inductor"]
"""
Unit test for block_ptr store dtype resolution with inplace buffers.

Validates that codegen_block_ptr_store_line casts store values to match the
block pointer element type (the actual tensor dtype), not the graph's
intermediate buffer dtype, when storing into an inplace-mutated buffer.

See T260349710 for the original issue: Inductor codegen for _to_copy ops
generated .to(tl.float32) for block_ptr stores into bf16 gradient buffers,
causing a dtype mismatch assertion in the Triton compiler.

Usage:
    buck test fbcode//caffe2/test/inductor:test_block_ptr_store_dtype
"""

import torch
from torch._inductor.codegen.common import InplacedBuffer, REMOVED
from torch._inductor.codegen.triton import triton_store_type
from torch._inductor.test_case import run_tests, TestCase


class TestBlockPtrStoreDtype(TestCase):
    """Test dtype resolution for block_ptr stores with inplace buffers.

    The logic under test (from codegen_block_ptr_store_line):

        store_dtype = V.graph.get_dtype(name)
        if name in self.args.inplace_buffers:
            buf = self.args.inplace_buffers[name]
            if not isinstance(buf, RemovedArg):
                store_dtype = V.graph.get_dtype(buf.other_names[0])
        value = f"{value}.to({triton_store_type(store_dtype)})"
    """

    @staticmethod
    def _resolve_store_dtype(name, inplace_buffers, get_dtype):
        """Reproduce the dtype resolution logic from codegen_block_ptr_store_line."""
        from torch._inductor.codegen.common import RemovedArg

        store_dtype = get_dtype(name)
        if name in inplace_buffers:
            buf = inplace_buffers[name]
            if not isinstance(buf, RemovedArg):
                store_dtype = get_dtype(buf.other_names[0])
        return store_dtype

    def test_non_inplace_uses_graph_dtype(self):
        """Non-inplace buffer: store dtype matches graph dtype."""
        dtypes = {"buf0": torch.float32}
        result = self._resolve_store_dtype("buf0", {}, lambda n: dtypes[n])
        self.assertEqual(result, torch.float32)
        self.assertEqual(triton_store_type(result), "tl.float32")

    def test_inplace_buffer_uses_input_dtype(self):
        """Inplace buffer with dtype mismatch: store dtype matches the input
        buffer's dtype (the actual tensor), not the graph output dtype.

        This is the key scenario from T260349710: _to_copy produces fp32
        intermediate values stored into a bf16 gradient buffer via inplace
        mutation. The block pointer element type comes from the actual tensor
        (bf16), so the cast must be .to(tl.bfloat16), not .to(tl.float32).
        """
        dtypes = {"buf0": torch.float32, "primals_1": torch.bfloat16}
        inplace_buffers = {
            "buf0": InplacedBuffer("in_out_ptr0", ["primals_1", "buf0"]),
            "primals_1": InplacedBuffer("in_out_ptr0", ["primals_1", "buf0"]),
        }
        result = self._resolve_store_dtype("buf0", inplace_buffers, lambda n: dtypes[n])
        self.assertEqual(result, torch.bfloat16)
        self.assertEqual(triton_store_type(result), "tl.bfloat16")

    def test_removed_inplace_falls_back_to_graph_dtype(self):
        """Removed inplace buffer: falls back to graph dtype."""
        dtypes = {"buf0": torch.float32}
        inplace_buffers = {"buf0": REMOVED}
        result = self._resolve_store_dtype("buf0", inplace_buffers, lambda n: dtypes[n])
        self.assertEqual(result, torch.float32)
        self.assertEqual(triton_store_type(result), "tl.float32")

    def test_same_dtype_inplace_is_unchanged(self):
        """Same-dtype inplace buffer: dtype is unchanged (both bf16)."""
        dtypes = {"buf0": torch.bfloat16, "primals_1": torch.bfloat16}
        inplace_buffers = {
            "buf0": InplacedBuffer("in_out_ptr0", ["primals_1", "buf0"]),
        }
        result = self._resolve_store_dtype("buf0", inplace_buffers, lambda n: dtypes[n])
        self.assertEqual(result, torch.bfloat16)
        self.assertEqual(triton_store_type(result), "tl.bfloat16")

    def test_chained_inplace_uses_original_input_dtype(self):
        """Chained inplace mutations: uses the original input buffer's dtype."""
        dtypes = {
            "buf0": torch.float32,
            "buf1": torch.float16,
            "primals_1": torch.bfloat16,
        }
        # Chain: primals_1 -> buf1 -> buf0
        inplace_buffers = {
            "buf0": InplacedBuffer("in_out_ptr0", ["primals_1", "buf1", "buf0"]),
        }
        result = self._resolve_store_dtype("buf0", inplace_buffers, lambda n: dtypes[n])
        self.assertEqual(result, torch.bfloat16)
        self.assertEqual(triton_store_type(result), "tl.bfloat16")


if __name__ == "__main__":
    run_tests()
