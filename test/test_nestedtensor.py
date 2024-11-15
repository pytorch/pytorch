# Owner(s): ["module: nestedtensor"]

import ast
import contextlib
import io
import itertools
import logging
import math
import random
import sys
import tempfile
import unittest
from functools import partial
from typing import Optional, Tuple

import numpy as np

import torch
import torch._dynamo
import torch._dynamo.testing
import torch.nn
import torch.nn.functional as F
from torch.nested._internal.nested_tensor import (
    buffer_from_jagged,
    jagged_from_list,
    nested_view_from_values_offsets,
    NestedTensor,
    ViewNestedFromBuffer,
)
from torch.nn.attention.flex_attention import create_nested_block_mask, flex_attention
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_FUSED_ATTENTION,
    SM70OrLater,
    SM80OrLater,
)
from torch.testing._internal.common_device_type import (
    dtypes,
    dtypesIfCUDA,
    flex_attention_supported_platform,
    instantiate_device_type_tests,
    onlyCPU,
    onlyCUDA,
    ops,
    PYTORCH_CUDA_MEMCHECK,
    skipCPUIf,
    skipCUDAIf,
    skipCUDAIfRocm,
    skipMeta,
)
from torch.testing._internal.common_dtype import floating_types_and_half
from torch.testing._internal.common_utils import (
    decorateIf,
    freeze_rng_state,
    gradcheck,
    instantiate_parametrized_tests,
    IS_FBCODE,
    IS_WINDOWS,
    markDynamoStrictTest,
    NestedTensorTestCase,
    parametrize,
    run_tests,
    skipIfRocm,
    skipIfSlowGradcheckEnv,
    skipIfTorchDynamo,
    subtest,
    TEST_WITH_ROCM,
    xfailIfTorchDynamo,
)
from torch.testing._internal.opinfo.core import BinaryUfuncInfo, ReductionOpInfo
from torch.testing._internal.opinfo.definitions.nested import (
    njt_op_db,
    SkipRule,
    XFailRule,
)
from torch.utils._pytree import tree_flatten
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts


log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Tests are ported from pytorch/nestedtensor.
# This makes porting as_nested_tensor easier in the future.


def _iter_constructors():
    # yield as_nested_tensor
    yield torch.nested.nested_tensor


# Returns True if the function recompiles between inputs1 and inputs2 with the
# specified dynamic setting.
def _recompiles_for_inputs(fn, inputs1, inputs2, dynamic=True):
    compile_count = [0]

    def counter(gm, example_inputs):
        compile_count[0] += 1
        return gm

    compiled_f = torch.compile(fn, fullgraph=True, backend=counter, dynamic=dynamic)
    compiled_f(*inputs1)
    compiled_f(*inputs2)
    return compile_count[0] > 1


# Helper function to generate a pair of random nested tensors
# one is contiguous, the other is not, but they appear to have same entries
# an output nested tensor consists of
# * `len(ragged_sizes)` matrices
# * matrices[i].shape == (20, ragged_sizes[i])


def random_nt_noncontiguous_pair(ragged_sizes, device="cpu", dtype=torch.float16):
    xs = []
    for size in ragged_sizes:
        xs.append(torch.randn((size, 20), device=device, dtype=dtype))
    # contiguous nested tensor
    ys = []
    for x in xs:
        ys.append(x.transpose(-1, -2))
    nt_contiguous = torch.nested.nested_tensor(ys)
    # noncontiguous nested tensor
    n = len(ragged_sizes)
    nt_noncontiguous = torch.nested.nested_tensor(xs).transpose(-1, -2)
    return nt_contiguous, nt_noncontiguous


# Helper functions to pad a noncontiguous nested tensor
# can be replaced once to_padded_tensor supports noncontiguous memory


def noncontiguous_to_padded_tensor(input, shape=None):
    tensors = input.unbind()
    ntensors = len(tensors)
    assert ntensors > 0
    if shape is None:
        shape = []
        for size in tensors[0].shape:
            shape.append(size)
        for i in range(1, ntensors):
            new_shape = tensors[i].shape
            for j in range(len(shape)):
                shape[j] = max(shape[j], new_shape[j])
        shape = [ntensors] + shape
    result = tensors[0].new_zeros(shape)
    for itensor in range(ntensors):
        tensor = tensors[itensor]
        view = result[itensor]
        for idim in range(tensor.dim()):
            view = view.narrow(idim, 0, tensor.size(idim))
        view.copy_(tensor)
    return result


# Helper function to generate a random nested tensor


def random_nt(
    device,
    dtype,
    num_tensors,
    max_dims,
    min_dims=None,
    layout=torch.strided,
    require_non_empty=True,
):
    if min_dims is None:
        min_dims = tuple([0] * len(max_dims))

    assert len(max_dims) == len(min_dims)
    for min_dim, max_dim in zip(min_dims, max_dims):
        assert max_dim > min_dim, "random_nt: max_dim must be greater than min_dim"
        assert min_dim >= 0, "random_nt: min_dim must be non-negative"
        if require_non_empty:
            assert not (
                min_dim == 0 and max_dim == 1
            ), "random_nt: zero cannot be the only possible value if require_non_empty is True"

    if require_non_empty:
        # Select a random idx that will be required to be non-empty
        non_zero_idx = torch.randint(low=0, high=num_tensors, size=(1,)).item()

    ts1 = []
    for i, _ in enumerate(range(num_tensors)):
        tensor_dims = []
        for min_dim, max_dim in zip(min_dims, max_dims):
            new_min_dim = min_dim
            if require_non_empty and i == non_zero_idx and min_dim == 0:
                new_min_dim = 1
            tensor_dims.append(
                torch.randint(low=new_min_dim, high=max_dim, size=(1,)).item()
            )
        t1 = torch.randn(tensor_dims, device=device, dtype=dtype)
        ts1.append(t1)

    return torch.nested.nested_tensor(ts1, device=device, dtype=dtype, layout=layout)


# Alternate approach to generating a random NT.
# dims should be something like [5, None, 10], with None indicating that a
# random ragged structure should be used
def random_nt_from_dims(
    dims, device=None, dtype=None, layout=torch.strided, requires_grad=False
):
    sizes = [
        [
            d if d is not None else torch.randint(2, 10, size=(1,)).item()
            for d in dims[1:]
        ]
        for d in range(dims[0])
    ]
    return torch.nested.nested_tensor(
        [torch.randn(*size) for size in sizes],
        device=device,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
    )


# Creates an NT matching another NT's number of components and
# shape / ragged structure for all dims specified to be -1.
def random_nt_from_similar(other, dims=None):
    if dims is None:
        return torch.randn_like(other)
    assert len(dims) == other.dim()
    assert dims[0] == -1 or dims[0] == other.size(0)

    ret_sizes = []
    for t in other.unbind():
        other_size = t.shape
        ret_size = []
        for i, d in enumerate(dims[1:]):
            if d == -1:
                ret_size.append(other_size[i])
            else:
                ret_size.append(d)
        ret_sizes.append(ret_size)

    return torch.nested.nested_tensor(
        [torch.randn(*size) for size in ret_sizes], device=other.device
    )


# makes naming nice for tests that parametrize over layout.
def layout_name(layout):
    # e.g. "torch.jagged" -> "jagged"
    return layout.__repr__().split(".")[-1]


def get_op_name(layout):
    # e.g. "<OpOverload(op='aten.sum', overload='dim_IntList')>" -> "sum"
    return layout.__name__.split(".")[0].split("_")[-1]


# Helper function for test_dummy_mha_with_nt
@torch.fx.wrap
def convert_dense_to_nested_tensor_legacy(values):
    offsets = torch.arange(
        0, values.shape[0] * values.shape[1] + 1, values.shape[1], device=values.device
    )
    metadata_cache = {"max_seqlen": values.shape[1], "min_seqlen": 1}
    nt = ViewNestedFromBuffer.apply(
        values.view(-1, values.shape[-1]), offsets, metadata_cache
    )
    return nt


# Helper function for test_dummy_mha_with_nt
@torch.fx.wrap
def convert_jagged_to_nested_tensor_legacy(
    values: torch.Tensor, offsets: torch.Tensor, max_length: int
) -> torch.Tensor:
    metadata_cache = {"max_seqlen": max_length, "min_seqlen": 1}
    nt = ViewNestedFromBuffer.apply(values, offsets, metadata_cache)
    return nt


# Helper function for test_dummy_mha_with_nt
@torch.fx.wrap
def convert_nt_to_jagged_legacy(nt):
    return buffer_from_jagged(nt)


# Helper function for test_dummy_mha_with_nt
@torch.fx.wrap
def convert_dense_to_nested_tensor(values):
    nt = torch.nested.as_nested_tensor(values, layout=torch.jagged)
    return nt


# Helper function for test_dummy_mha_with_nt
@torch.fx.wrap
def convert_jagged_to_nested_tensor(
    values: torch.Tensor, offsets: torch.Tensor, max_length: int
) -> torch.Tensor:
    nt = torch.nested.nested_tensor_from_jagged(
        values, offsets, lengths=None, min_seqlen=1, max_seqlen=max_length
    )
    return nt


# Helper function for test_dummy_mha_with_nt
def convert_nt_to_jagged(nt):
    return nt.values()


@markDynamoStrictTest
class TestNestedTensor(NestedTensorTestCase):
    @parametrize("batch_size", [2, 4])
    @parametrize("max_seq_len", [3, 5])
    @parametrize("vocab_size", [10, 20])
    def test_2d_nested_tensor(self, batch_size, max_seq_len, vocab_size):
        data = []
        nested_tensor_ref_list = []
        for _ in range(batch_size):
            if max_seq_len == 0:
                length = 0
            else:
                length = np.random.randint(low=1, high=max_seq_len)
            row = list(np.random.randint(low=0, high=vocab_size, size=(length,)))
            data.append(row)
            nested_tensor_ref_list.append(torch.Tensor(row))
        nested_tensor = torch.nested.nested_tensor(data, dtype=torch.int64)
        nested_tensor_list = nested_tensor.unbind()
        for id in range(batch_size):
            self.assertEqual(
                nested_tensor_list[id], nested_tensor_ref_list[id].type(torch.int64)
            )

    @parametrize("batch_size", [2, 4])
    @parametrize("max_seq_len", [3, 5])
    @parametrize("vocab_size", [10, 20])
    def test_3d_nested_tensor(self, batch_size, max_seq_len, vocab_size):
        data = []
        nested_tensor_ref_list = []
        for _ in range(batch_size):
            if max_seq_len == 0:
                length = 0
            else:
                length = np.random.randint(low=1, high=max_seq_len)
            row = list(np.random.randint(low=0, high=vocab_size, size=(length,)))
            row = [list(item * np.arange(max_seq_len)) for item in row]
            data.append(row)
            nested_tensor_ref_list.append(torch.Tensor(row))
        nested_tensor = torch.nested.nested_tensor(data, dtype=torch.int64)
        nested_tensor_list = nested_tensor.unbind()
        for id in range(batch_size):
            self.assertEqual(
                nested_tensor_list[id], nested_tensor_ref_list[id].type(torch.int64)
            )

    @parametrize("batch_size", [2, 4])
    @parametrize("max_seq_len", [3, 5])
    @parametrize("vocab_size", [10, 20])
    def test_3d_nested_tensor_float(self, batch_size, max_seq_len, vocab_size):
        data = []
        nested_tensor_ref_list = []
        for _ in range(batch_size):
            if max_seq_len == 0:
                length = 0
            else:
                length = np.random.randint(low=1, high=max_seq_len)
            row = list(
                np.random.randint(low=0, high=vocab_size, size=(length,)).astype(float)
            )
            row = [list(item * np.arange(max_seq_len)) for item in row]
            data.append(row)
            nested_tensor_ref_list.append(torch.Tensor(row))
        nested_tensor = torch.nested.nested_tensor(data, dtype=torch.float)
        nested_tensor_list = nested_tensor.unbind()
        for id in range(batch_size):
            self.assertEqual(
                nested_tensor_list[id], nested_tensor_ref_list[id].type(torch.float)
            )

    @torch.inference_mode()
    def _test_unbind_case(self, a, b):
        nt = torch.nested.nested_tensor([a, b])
        a1, b1 = nt.unbind()
        self.assertTrue(a is not a1)
        self.assertTrue(b is not b1)

        nt = torch.nested.nested_tensor([a, b], dtype=a.dtype)
        a1, b1 = nt.unbind(0)
        self.assertEqual(a, a1)
        self.assertEqual(b, b1)

        a = torch.randn((2, 3)).add_(1)
        nt = torch.nested.nested_tensor([a])
        self.assertEqual(a, nt.unbind(0)[0])

    @torch.inference_mode()
    def test_unbind_0(self):
        self._test_unbind_case(torch.tensor([1, 2]), torch.tensor([7, 8]))

    @torch.inference_mode()
    def test_unbind_1(self):
        self._test_unbind_case(torch.tensor([1]), torch.tensor([7]))

    @torch.inference_mode()
    def test_unbind_3(self):
        self._test_unbind_case(torch.tensor([1.0]), torch.tensor([]))

    @torch.inference_mode()
    def test_unbind_4(self):
        self._test_unbind_case(torch.tensor([]), torch.tensor([]))

    @torch.inference_mode()
    def test_unbind_dim(self):
        def _test_fn(unbind_fn):
            a = torch.rand(3, 2)
            b = torch.rand(2, 3)
            nt = torch.nested.nested_tensor([a, b])
            self.assertRaises(RuntimeError, lambda: unbind_fn(nt, 1))

        # Both of these tests are necessary, because we're using
        # torch_function.
        _test_fn(lambda x, dim: x.unbind(dim))
        # TODO: Re-enable this once using torch_dispatch
        # _test_fn(lambda x, dim: torch.unbind(x, dim))

    @torch.inference_mode()
    def test_nested_tensor(self):
        self.assertRaises(
            TypeError, lambda: torch.nested.nested_tensor(torch.tensor([3.0]))
        )
        self.assertRaises(TypeError, lambda: torch.nested.nested_tensor(4.0))

    @torch.inference_mode()
    def test_nested_tensor_matching_dim(self):
        self.assertRaisesRegex(
            RuntimeError,
            "Found dimension 1 for Tensor at index 1 and dimension 0 for Tensor at index 0.",
            lambda: torch.nested.nested_tensor([torch.tensor(1.0), torch.tensor([])]),
        )
        self.assertRaisesRegex(
            RuntimeError,
            "Found dimension 1 for Tensor at index 2 and dimension 0 for Tensor at index 1.",
            lambda: torch.nested.nested_tensor(
                [torch.tensor(1.0), torch.tensor(2.0), torch.tensor([])]
            ),
        )

    @torch.inference_mode()
    def test_default_nested_tensor(self):
        self.assertRaises(TypeError, lambda: torch.nested.nested_tensor())
        default_nested_tensor = torch.nested.nested_tensor([])
        default_tensor = torch.tensor([])
        # self.assertEqual(default_nested_tensor.nested_dim(), 1)
        # self.assertEqual(default_nested_tensor.nested_size(), ())
        self.assertEqual(default_nested_tensor.dim(), default_tensor.dim())
        self.assertEqual(default_nested_tensor.layout, default_tensor.layout)
        self.assertEqual(default_nested_tensor.device, default_tensor.device)
        self.assertEqual(default_nested_tensor.dtype, default_tensor.dtype)
        self.assertEqual(
            default_nested_tensor.requires_grad, default_tensor.requires_grad
        )
        self.assertIsNone(default_tensor.grad)
        # TODO: Re-enable once we have a performance driven
        # use case and implementation.
        # self.assertEqual(default_nested_tensor.is_pinned(),
        #                  default_tensor.is_pinned())

    @torch.inference_mode()
    def test_dim(self):
        for constructor in _iter_constructors():
            a1 = constructor([])
            self.assertEqual(a1.dim(), 1)
            a1 = constructor([torch.tensor(3.0)])
            self.assertEqual(a1.dim(), 1)
            a1 = constructor([torch.tensor([1, 2, 3, 4])])
            self.assertEqual(a1.dim(), 2)

    @unittest.skipIf(IS_FBCODE, "numel is not virtual in fbcode.")
    @torch.inference_mode()
    def test_numel(self):
        for constructor in _iter_constructors():
            a1 = constructor([])
            self.assertEqual(a1.numel(), 0)
            a1 = constructor([torch.tensor(3.0), torch.tensor(4.0)])
            self.assertEqual(a1.numel(), 2)
            a1 = constructor([torch.randn(2, 2, 2)])
            self.assertEqual(a1.numel(), 8)
            a1 = constructor([torch.randn([1, 2, 3]), torch.randn(3, 2, 1)])
            self.assertEqual(a1.numel(), 12)
            a1 = constructor([torch.randn([1, 1, 3]), torch.randn(3, 2, 4)])
            self.assertEqual(a1.numel(), 27)
            a1 = constructor([torch.randn([5, 5, 5]), torch.randn(6, 6, 6)])
            self.assertEqual(a1.numel(), 341)

            # Interesting edge case
            a1 = constructor([torch.randn([1, 2, 3]), torch.randn(1, 2, 0)])
            self.assertEqual(a1.numel(), 6)

    @torch.inference_mode()
    def test_size(self):
        for constructor in _iter_constructors():
            a1 = constructor([])
            self.assertRaisesRegex(
                RuntimeError,
                "NestedTensorImpl doesn't support sizes",
                lambda: a1.size(),
            )

    def test_size_dim(self):
        a = torch.nested.nested_tensor([])
        self.assertEqual(a.size(0), 0)

        a = torch.nested.nested_tensor([torch.tensor(1)])
        self.assertEqual(a.size(0), 1)

        a = torch.nested.nested_tensor([torch.tensor(1), torch.tensor(2)])
        self.assertEqual(a.size(0), 2)

        a = torch.nested.nested_tensor([torch.rand(1, 2), torch.rand(1, 8)])
        self.assertEqual(a.size(0), 2)
        self.assertEqual(a.size(1), 1)
        self.assertRaisesRegex(
            RuntimeError,
            "Given dimension 2 is irregular and does not have a size",
            lambda: a.size(2),
        )

        a = torch.nested.nested_tensor([torch.rand(3, 4), torch.rand(5, 4)])
        self.assertEqual(a.size(0), 2)
        self.assertRaisesRegex(
            RuntimeError,
            "Given dimension 1 is irregular and does not have a size",
            lambda: a.size(1),
        )
        self.assertEqual(a.size(2), 4)

    @unittest.skipIf(IS_FBCODE, "stride is not virtual in fbcode.")
    @torch.inference_mode()
    def test_stride(self):
        for constructor in _iter_constructors():
            a1 = constructor([])
            self.assertRaisesRegex(
                RuntimeError,
                "NestedTensorImpl doesn't support strides",
                lambda: a1.stride(),
            )

    @unittest.skipIf(IS_FBCODE, "is_contiguous is not virtual in fbcode.")
    @torch.inference_mode()
    def test_is_contiguous(self):
        # Test empty case
        nt_empty = torch.nested.nested_tensor([])
        assert nt_empty.is_contiguous()
        self.assertEqual(nt_empty, nt_empty.contiguous())

        nt_contiguous, nt_noncontiguous = random_nt_noncontiguous_pair((2, 3, 6, 7))

        # Test contiguous case
        assert nt_contiguous.is_contiguous()
        self.assertEqual(nt_contiguous, nt_contiguous.contiguous())

        # Test non_contiguous case
        assert not nt_noncontiguous.is_contiguous()
        self.assertEqual(nt_contiguous, nt_noncontiguous.contiguous())

        # Test querying by memory_format
        self.assertTrue(
            nt_contiguous.is_contiguous(memory_format=torch.contiguous_format)
        )
        self.assertTrue(
            not nt_noncontiguous.is_contiguous(memory_format=torch.contiguous_format)
        )

    @torch.inference_mode()
    def test_repr_string(self):
        a = torch.nested.nested_tensor([])
        expected = "nested_tensor([\n\n])"
        self.assertEqual(str(a), expected)
        self.assertEqual(repr(a), expected)

        a = torch.nested.nested_tensor([torch.tensor(1.0)])
        expected = "nested_tensor([\n  tensor(1.)\n])"
        self.assertEqual(str(a), expected)
        self.assertEqual(repr(a), expected)

        a = torch.nested.nested_tensor([torch.tensor([[1, 2]]), torch.tensor([[4, 5]])])
        expected = "nested_tensor([\n  tensor([[1, 2]]),\n  tensor([[4, 5]])\n])"
        self.assertEqual(str(a), expected)
        self.assertEqual(repr(a), expected)

    def test_to_padded_tensor_on_empty_tensor(self):
        nt = torch.nested.nested_tensor([])
        empty = torch.nested.to_padded_tensor(nt, 4)
        self.assertEqual(empty, torch.tensor([]))

    def test_nested_namespace(self):
        nt = torch.nested.nested_tensor([torch.randn(2, 3), torch.randn(4, 5)])
        result = nt.to_padded_tensor(4)
        nested_namespace_result = torch.nested.to_padded_tensor(nt, 4)
        self.assertEqual(result, nested_namespace_result)

    def test_to(self):
        ntensors = 4
        nt = random_nt(torch.device("cpu"), torch.float32, ntensors, (4, 4))

        def test_copy_behavior(t, non_blocking=False):
            self.assertIs(t, t.to(t, non_blocking=non_blocking))
            self.assertIs(t, t.to(t.dtype, non_blocking=non_blocking))
            self.assertIs(t, t.to(torch.empty_like(t), non_blocking=non_blocking))
            self.assertIsNot(t, t.to(t, non_blocking=non_blocking, copy=True))
            self.assertIsNot(t, t.to(t.dtype, non_blocking=non_blocking, copy=True))
            self.assertIsNot(
                t, t.to(torch.empty_like(t), non_blocking=non_blocking, copy=True)
            )

            devices = [t.device]
            if t.device.type == "cuda":
                if t.device.index == -1:
                    devices.append(f"cuda:{torch.cuda.current_device()}")
                elif t.device.index == torch.cuda.current_device():
                    devices.append("cuda")
            for device in devices:
                self.assertIs(t, t.to(device, non_blocking=non_blocking))
                self.assertIs(t, t.to(device, t.dtype, non_blocking=non_blocking))
                self.assertIsNot(t, t.to(device, non_blocking=non_blocking, copy=True))
                self.assertIsNot(
                    t, t.to(device, t.dtype, non_blocking=non_blocking, copy=True)
                )

        test_copy_behavior(nt)
        self.assertEqual(nt.device, nt.to("cpu").device)
        self.assertEqual(nt.device, nt.to("cpu", dtype=torch.float32).device)
        self.assertIs(torch.float32, nt.to("cpu", dtype=torch.float32).dtype)
        self.assertEqual(nt.device, nt.to(torch.float32).device)
        self.assertIs(torch.float32, nt.to(dtype=torch.float32).dtype)

        def test_data_ptr(getter):
            self.assertEqual(getter(nt), getter(nt.to("cpu")))
            self.assertEqual(
                getter(nt), getter(nt.to(dtype=nt.dtype, device=nt.device, copy=False))
            )
            self.assertEqual(getter(nt), getter(nt.to("cpu", copy=False)))
            self.assertNotEqual(getter(nt), getter(nt.to("cpu", copy=True)))

        test_data_ptr(lambda nt: nt.data_ptr())

        if torch.cuda.is_available():
            for non_blocking in [True, False]:
                for cuda in [
                    "cuda",
                    "cuda:0" if torch.cuda.device_count() == 1 else "cuda:1",
                ]:
                    nt2 = random_nt(cuda, torch.float32, ntensors, (4, 4))
                    test_copy_behavior(nt2, non_blocking)
                    self.assertEqual(
                        nt2.device, nt2.to(cuda, non_blocking=non_blocking).device
                    )
                    self.assertEqual(
                        nt.device, nt2.to("cpu", non_blocking=non_blocking).device
                    )
                    self.assertEqual(
                        nt2.device, nt.to(cuda, non_blocking=non_blocking).device
                    )
                    self.assertIs(
                        torch.int32,
                        nt2.to(
                            "cpu", dtype=torch.int32, non_blocking=non_blocking
                        ).dtype,
                    )
                    self.assertEqual(
                        nt.device,
                        nt2.to(
                            "cpu", dtype=torch.int32, non_blocking=non_blocking
                        ).device,
                    )
                    self.assertIs(torch.int32, nt2.to(dtype=torch.int32).dtype)
                    self.assertEqual(nt2.device, nt2.to(dtype=torch.int32).device)

    def test_copy_(self):
        ntensors = 4
        nt = random_nt(torch.device("cpu"), torch.float32, ntensors, (4, 4))
        nt_copy = torch.empty_like(nt)
        nt_copy.copy_(nt)

        for nt_ub, nt_copy_ub in zip(nt.unbind(), nt_copy):
            self.assertEqual(nt_ub, nt_copy_ub)

        nt_error = torch.nested.nested_tensor([torch.tensor([0, 0])])
        self.assertRaisesRegex(
            RuntimeError,
            "copy_ only supports tensors that are the same size for Nested implementations",
            lambda: nt_error.copy_(nt),
        )

        if torch.cuda.is_available():
            nt = random_nt(torch.device("cuda"), torch.float32, ntensors, (4, 4))
            nt_copy = torch.empty_like(nt, device=torch.device("cpu"))
            nt_copy.copy_(nt, non_blocking=True)
            torch.cuda.current_stream(torch.cuda.current_device()).synchronize()
            for nt_ub, nt_copy_ub in zip(nt.unbind(), nt_copy):
                self.assertEqual(nt_ub, nt_copy_ub)

            nt_copy = torch.empty_like(nt, device=torch.device("cpu"))
            nt_copy.copy_(nt, non_blocking=False)
            for nt_ub, nt_copy_ub in zip(nt.unbind(), nt_copy):
                self.assertEqual(nt_ub, nt_copy_ub)

    def test_fill_(self):
        ntensors = 4
        nt = random_nt(torch.device("cpu"), torch.float32, ntensors, (4, 4))
        nt.fill_(10.0)
        for nt_ub in nt.unbind():
            t = torch.empty_like(nt_ub)
            t.fill_(10.0)
            self.assertEqual(nt_ub, t)

        fill_tensor = torch.tensor([11.0])
        self.assertRaisesRegex(
            RuntimeError,
            "fill_ only supports 0-dimension value tensor",
            lambda: nt.fill_(fill_tensor),
        )

        nt.fill_(fill_tensor[0])
        for nt_ub in nt.unbind():
            t = torch.empty_like(nt_ub)
            t.fill_(11.0)
            self.assertEqual(nt_ub, t)

    def test_zero_(self):
        ntensors = 4
        nt = random_nt(torch.device("cpu"), torch.float32, ntensors, (4, 4))
        nt.zero_()
        for nt_ub in nt.unbind():
            t = torch.empty_like(nt_ub)
            t.fill_(0.0)
            self.assertEqual(nt_ub, t)

    @parametrize(
        "func",
        [torch.ones_like, torch.zeros_like, torch.randn_like],
        name_fn=lambda f: f.__name__,
    )
    def test_like_functions(self, func):
        ntensors = 4
        nt = random_nt(torch.device("cpu"), torch.float32, ntensors, (4, 4))
        torch.manual_seed(1)
        nt_like = func(nt)

        torch.manual_seed(1)
        for nt_ub in nt_like.unbind():
            t_like = func(nt_ub)
            self.assertEqual(nt_ub, t_like)

    def test_cat(self):
        # dim=0 success case
        # No constraints on ragged structures matching.
        x = random_nt_from_dims([5, None, 10])
        y = random_nt_from_dims([3, 4, None])
        output = torch.cat([x, y], dim=0)
        for out_component, xy_component in zip(
            output.unbind(), itertools.chain(x.unbind(), y.unbind())
        ):
            self.assertEqual(out_component, xy_component)

        # dim=-1 success case
        # shape (B, *, D)
        x = random_nt_from_dims([5, None, 10])
        # shape (B, *, D'); same structure as x but dim=-1 differs
        y = random_nt_from_similar(x, dims=[-1, -1, 8])
        # should be shape (B, *, D + D') when supported
        output = torch.cat([x, y], dim=-1)
        for out_component, x_component, y_component in zip(
            output.unbind(), x.unbind(), y.unbind()
        ):
            self.assertEqual(
                out_component, torch.cat([x_component, y_component], dim=-1)
            )

        # dim between 0 and -1 success case
        x = random_nt_from_dims([5, None, 2, 3])
        # same structure as x but dim=2 differs
        y = random_nt_from_similar(x, dims=[-1, -1, 4, -1])
        output = torch.cat([x, y], dim=2)
        for out_component, x_component, y_component in zip(
            output.unbind(), x.unbind(), y.unbind()
        ):
            self.assertEqual(
                out_component, torch.cat([x_component, y_component], dim=1)
            )

        # error case: mixed NT / dense inputs
        x = random_nt_from_dims([5, None, 2])
        y = torch.randn(5, 3, 2)
        with self.assertRaisesRegex(
            RuntimeError, "expected each tensor in given list to be nested"
        ):
            torch.cat([x, y], dim=-1)

        # error case: NTs with different dims
        x = random_nt_from_dims([5, None, 2])
        y = random_nt_from_dims([5, None, 2, 3])
        with self.assertRaisesRegex(
            RuntimeError,
            "expected all nested tensors to have matching ragged structures outside of the concatenated dim",
        ):
            torch.cat([x, y], dim=-1)

        # error case: non-contiguous NT
        x, y = random_nt_noncontiguous_pair((2, 3, 4), dtype=torch.float32)
        # transpose to put ragged dim next to batch dim
        x, y = x.transpose(-2, -1), y.transpose(-2, -1)
        with self.assertRaisesRegex(
            RuntimeError, "only contiguous nested tensors are supported"
        ):
            torch.cat([x, y], dim=-1)

        # error case: multiple ragged dims in inputs
        x = random_nt_from_dims([5, None, None, 2])
        y = random_nt_from_similar(x)
        with self.assertRaisesRegex(
            RuntimeError,
            "only nested tensors with a single ragged dim next to the batch dim are supported",
        ):
            torch.cat([x, y], dim=-1)

        # error case: ragged dim not next to batch dim
        x = random_nt_from_dims([5, 2, None])
        y = random_nt_from_similar(x)
        with self.assertRaisesRegex(
            RuntimeError,
            "only nested tensors with a single ragged dim next to the batch dim are supported",
        ):
            torch.cat([x, y], dim=1)

        # error case: NTs with different batch sizes
        x = random_nt_from_dims([5, None, 2])
        y = random_nt_from_dims([3, None, 2])
        with self.assertRaisesRegex(
            RuntimeError,
            "expected all nested tensors to have matching ragged structures outside of the concatenated dim",
        ):
            torch.cat([x, y], dim=-1)

        # error case: NTs with different ragged structures
        x = torch.nested.nested_tensor(
            [
                torch.randn(2, 6),
                torch.randn(4, 6),
                torch.randn(5, 6),
            ]
        )
        y = torch.nested.nested_tensor(
            [
                torch.randn(5, 6),
                torch.randn(4, 6),
                torch.randn(2, 6),
            ]
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "expected all nested tensors to have matching ragged structures outside of the concatenated dim",
        ):
            torch.cat([x, y], dim=-1)


@markDynamoStrictTest
class TestNestedTensorDeviceType(NestedTensorTestCase):
    # Helper function to generate a pair of random nested tensors
    # the 2 nested tensors have same shapes
    def random_nt_pair(self, device, dtype, num_tensors, max_dims):
        ts1 = []
        ts2 = []
        for _ in range(num_tensors):
            tensor_dims = tuple(
                [
                    torch.randint(low=0, high=max_dim, size=(1,)).item()
                    for max_dim in max_dims
                ]
            )
            t1 = torch.randn(tensor_dims, device=device, dtype=dtype)
            t2 = torch.randn(tensor_dims, device=device, dtype=dtype)
            ts1.append(t1)
            ts2.append(t2)
        return (
            torch.nested.nested_tensor(ts1, device=device, dtype=dtype),
            torch.nested.nested_tensor(ts2, device=device, dtype=dtype),
        )

    @dtypes(*floating_types_and_half())
    def test_detach(self, device, dtype):
        a = torch.randn(2, 4, device=device, dtype=dtype, requires_grad=False)
        b = torch.randn(5, 4, device=device, dtype=dtype, requires_grad=False)
        x = torch.nested.nested_tensor([a, b], requires_grad=True)

        x_detach = x.detach()

        z = x_detach * 4
        self.assertFalse(x_detach.requires_grad)
        self.assertFalse(z.requires_grad)

        a = torch.randn(2, 4, device=device, dtype=dtype, requires_grad=True)
        b = torch.randn(5, 4, device=device, dtype=dtype, requires_grad=True)
        x = torch.nested.as_nested_tensor([a, b])

        y = x * 2
        y = y.detach()
        self.assertFalse(y.requires_grad)
        self.assertIsNone(y.grad_fn)

        z = x + y
        torch.nested.to_padded_tensor(z, 0).sum().backward()
        # This is an incorrect gradient, but we assume that's what the user
        # wanted. detach() is an advanced option.
        self.assertEqual(a.grad, torch.ones(2, 4, device=device, dtype=dtype))
        self.assertEqual(b.grad, torch.ones(5, 4, device=device, dtype=dtype))

    @dtypes(torch.float, torch.double, torch.half)
    @parametrize("requires_grad", [False, True])
    @parametrize("weights_only", [False, True])
    def test_serialization(self, device, dtype, requires_grad, weights_only):
        def compare_metadata(nt1, nt2):
            self.assertEqual(nt1._nested_tensor_size(), nt2._nested_tensor_size())
            self.assertEqual(nt1._nested_tensor_strides(), nt2._nested_tensor_strides())
            self.assertEqual(
                nt1._nested_tensor_storage_offsets(),
                nt2._nested_tensor_storage_offsets(),
            )

        nt_contiguous, nt_noncontiguous = random_nt_noncontiguous_pair((2, 3, 6, 7))
        for a in [nt_contiguous, nt_noncontiguous]:
            buffer = io.BytesIO()
            serialized = torch.save(a, buffer)
            buffer.seek(0)
            b = torch.load(buffer, weights_only=weights_only)
            # should be both conceptually equal and metadata equivalent
            self.assertEqual(a, b)
            compare_metadata(a, b)
            # should be conceptually equal but not necessarily metadata equivalent
            self.assertEqual(b, nt_contiguous)
            self.assertEqual(b, nt_noncontiguous)

    @dtypes(torch.float, torch.float16, torch.double)
    def test_unbind_noncontiguous(self, device, dtype):
        nt_contiguous, nt_noncontiguous = random_nt_noncontiguous_pair(
            (2, 3, 6, 7), device, dtype
        )
        ub_contiguous = nt_contiguous.unbind()
        ub_noncontiguous = nt_noncontiguous.unbind()
        self.assertEqual(len(ub_contiguous), len(ub_noncontiguous))
        n = len(ub_contiguous)
        for i in range(n):
            self.assertEqual(ub_contiguous[i], ub_noncontiguous[i])

    @dtypes(torch.float)
    @skipMeta
    def test_to_then_from_padded_tensor_no_transform0213(self, device, dtype):
        t = torch.randn(4, 4, 4, device=device, dtype=dtype)
        ts = list(torch.unbind(t))
        ts[0] = ts[0][:-1]
        nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
        padded = torch.nested.to_padded_tensor(nt, 0)

        nt_to = torch._nested_from_padded_and_nested_example(padded, nt)

        for t1, t2 in zip(nt.unbind(), nt_to.unbind()):
            self.assertEqual(t1, t2)
        self.assertEqual(nt.device, nt_to.device)

    @dtypes(torch.float)
    @dtypesIfCUDA(torch.float, torch.half)
    @skipMeta
    @torch.inference_mode()
    def test_layer_norm(self, device, dtype):
        def _test(size):
            # Simple shapes test
            t0 = torch.randn(2, size, device=device, dtype=dtype, requires_grad=False)
            t1 = torch.randn(2, size, device=device, dtype=dtype, requires_grad=False)
            ts = [t0, t1, t0, t1]
            nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
            layer_norm = torch.nn.LayerNorm(size, device=device, dtype=dtype)
            nt_result = layer_norm(nt)
            for nt_subresult, t in zip(nt_result.unbind(), ts):
                t_result = layer_norm(t.reshape(1, -1, size).squeeze(0))
                self.assertEqual(nt_subresult, t_result)

            # More complex nt test with different lengths for each tensor
            t0 = torch.randn(4, size, device=device, dtype=dtype, requires_grad=False)
            t1 = torch.randn(10, size, device=device, dtype=dtype, requires_grad=False)
            t2 = torch.randn(7, size, device=device, dtype=dtype, requires_grad=False)
            ts = [t0, t1, t2, t0, t2]
            nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
            layer_norm = torch.nn.LayerNorm(size, device=device, dtype=dtype)
            nt_result = layer_norm(nt)
            for nt_subresult, t in zip(nt_result.unbind(), ts):
                t_result = layer_norm(t.reshape(1, -1, size).squeeze(0))
                self.assertEqual(nt_subresult, t_result)

            if size <= 128:
                # Test with multidimensional tensors after irregular dim
                # (run only with smaller dimensions to ensure fast execution)
                t0 = torch.randn(
                    4, size, size, 4, device=device, dtype=dtype, requires_grad=False
                )
                t1 = torch.randn(
                    10, size, size, 4, device=device, dtype=dtype, requires_grad=False
                )
                t2 = torch.randn(
                    7, size, size, 4, device=device, dtype=dtype, requires_grad=False
                )
                ts = [t0, t1, t2, t0, t2]
                nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
                layer_norm = torch.nn.LayerNorm(
                    (size, size, 4), device=device, dtype=dtype
                )
                nt_result = layer_norm(nt)
                for nt_subresult, t in zip(nt_result.unbind(), ts):
                    t_result = layer_norm(t.reshape(1, -1, size, size, 4).squeeze(0))
                    self.assertEqual(nt_subresult, t_result)

                # Test where the normalizing dimensions are not all
                layer_norm = torch.nn.LayerNorm((size, 4), device=device, dtype=dtype)
                nt_result = layer_norm(nt)
                for nt_subresult, t in zip(nt_result.unbind(), ts):
                    t_result = layer_norm(t.reshape(1, -1, size, size, 4).squeeze(0))
                    self.assertEqual(nt_subresult, t_result)

        for size in (1024, 1023, 513, 512, 256, 128, 2, 4, 32):
            _test(size)

    @dtypes(torch.float)
    @dtypesIfCUDA(torch.float, torch.half)
    @skipMeta
    @torch.inference_mode()
    def test_layer_norm_breaking(self, device, dtype):
        size = 128
        t0 = torch.randn(
            4, size, size, 4, device=device, dtype=dtype, requires_grad=False
        )
        t1 = torch.randn(
            10, size, size, 4, device=device, dtype=dtype, requires_grad=False
        )
        t2 = torch.randn(
            7, size, size, 4, device=device, dtype=dtype, requires_grad=False
        )
        ts = [t0, t1, t2, t0, t2]
        nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
        layer_norm = torch.nn.LayerNorm((4, size, size, 4), device=device, dtype=dtype)
        self.assertRaisesRegex(
            RuntimeError,
            "normalized_shape extends into irregular dimensions for the nested tensor",
            lambda: layer_norm(nt),
        )
        layer_norm = torch.nn.LayerNorm((size + 1, size, 4), device=device, dtype=dtype)
        self.assertRaisesRegex(
            RuntimeError,
            "The shape at dimension 0",
            lambda: layer_norm(nt),
        )

    @parametrize("layout", [torch.strided, torch.jagged], name_fn=layout_name)
    def test_embedding(self, device, layout):
        inputs = [
            torch.randint(100, (L,), device=device, dtype=torch.int64)
            for L in torch.randint(5, 50, (8,))
        ]
        x = torch.nested.nested_tensor(
            inputs, device=device, dtype=torch.int64, layout=layout
        )
        emb = torch.nn.Embedding(100, 8, device=device)
        y = emb(x)
        if layout == torch.jagged:
            y.backward(torch.randn_like(y))

        @torch._dynamo.disable
        def check(inputs, y):
            ys = y.unbind()
            for i, inp in enumerate(inputs):
                self.assertEqual(emb(inp), ys[i])

        check(inputs, y)

    @skipMeta
    @torch.inference_mode()
    @dtypes(*floating_types_and_half())
    def test_masked_fill(self, device, dtype):
        # nested tensor * nested tensor
        (nt, mask) = self.random_nt_pair(device, dtype, 4, (4, 4))
        mask = torch.nested.nested_tensor([m < 0 for m in mask.unbind()])
        ref = torch.nested.nested_tensor(
            [t.masked_fill(m, 0) for (t, m) in zip(nt.unbind(), mask.unbind())]
        )
        out = nt.masked_fill(mask, 0)
        self.assertEqual(ref, out)

    @dtypes(torch.float, torch.float16)
    def test_to_padded_tensor_simple(self, device, dtype):
        t = torch.randn(4, 4, 4, device=device, dtype=dtype)
        ts = list(torch.unbind(t))
        ts[0] = ts[0][:-1]
        nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
        for padding_value in (0, 1):
            padded = torch.nested.to_padded_tensor(nt, padding_value)

            correct_output = t.clone()
            if padding_value == 0:
                correct_output[0][-1] = torch.zeros_like(correct_output[0][-1])
            else:
                correct_output[0][-1] = torch.ones_like(correct_output[0][-1])

            self.assertEqual(padded, correct_output)
            self.assertEqual(padded.device, torch.device(device))
            self.assertEqual(padded.dtype, dtype)

    @dtypes(torch.float, torch.float16)
    def test_to_padded_tensor_output_size(self, device, dtype):
        t = torch.randn(4, 4, 4, device=device, dtype=dtype)
        output_size = (4, 6, 5)
        ts = list(torch.unbind(t))
        ts[0] = ts[0][:-1]
        nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
        for padding_value in (0, 1):
            padded = torch.nested.to_padded_tensor(
                nt, padding_value, output_size=output_size
            )
            correct_output = (
                torch.ones(output_size, device=device, dtype=dtype) * padding_value
            )
            correct_output[:4:, :4, :4] = t.clone()
            if padding_value == 0:
                correct_output[0][3] = torch.zeros_like(correct_output[0][3])
            else:
                correct_output[0][3] = torch.ones_like(correct_output[0][3])

            self.assertEqual(padded, correct_output)
            self.assertEqual(padded.device, torch.device(device))
            self.assertEqual(padded.dtype, dtype)

    @dtypes(torch.float, torch.float16, torch.double)
    def test_to_padded_tensor_dim2(self, device, dtype):
        ts = [
            torch.randn(160, device=device, dtype=dtype),
            torch.randn(1240, device=device, dtype=dtype),
            torch.randn(2400, device=device, dtype=dtype),
        ]
        nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
        pad = 42
        correct_output = []
        for t in ts:
            next_output = torch.ones_like(ts[2]) * pad
            correct_output.append(next_output)
            next_output[: t.size(0)].copy_(t)
        correct_output = torch.stack(correct_output)
        padded = torch.nested.to_padded_tensor(nt, pad)
        self.assertEqual(padded, correct_output)

    @dtypes(torch.float, torch.float16, torch.double)
    def test_to_padded_tensor_dim3(self, device, dtype):
        ts = [
            torch.randn(16, 21, device=device, dtype=dtype),
            torch.randn(24, 32, device=device, dtype=dtype),
            torch.randn(40, 53, device=device, dtype=dtype),
        ]
        nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
        pad = 42
        correct_output = []
        for t in ts:
            next_output = torch.ones_like(ts[2]) * pad
            correct_output.append(next_output)
            next_output[: t.size(0), : t.size(1)].copy_(t)
        correct_output = torch.stack(correct_output)
        padded = torch.nested.to_padded_tensor(nt, pad)
        self.assertEqual(padded, correct_output)

    @dtypes(torch.float, torch.float16, torch.double)
    def test_to_padded_tensor_dim4(self, device, dtype):
        ts = [
            torch.randn(16, 21, 13, device=device, dtype=dtype),
            torch.randn(24, 32, 14, device=device, dtype=dtype),
            torch.randn(40, 53, 16, device=device, dtype=dtype),
        ]
        nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
        pad = 42
        correct_output = []
        for t in ts:
            next_output = torch.ones_like(ts[2]) * pad
            correct_output.append(next_output)
            next_output[: t.size(0), : t.size(1), : t.size(2)].copy_(t)
        correct_output = torch.stack(correct_output)
        padded = torch.nested.to_padded_tensor(nt, pad)
        self.assertEqual(padded, correct_output)

    # TODO: test noncontiguous to_padded_tensor
    # For now this tests the functionality of noncontiguous_to_padded_tensor
    # and the error message of to_padded_tensor
    # since to_padded_tensor does not support noncontiguous buffer yet
    @dtypes(torch.float, torch.float16, torch.double)
    @torch.inference_mode()
    def test_to_padded_tensor_noncontiguous(self, device, dtype):
        nt_contiguous, nt_noncontiguous = random_nt_noncontiguous_pair(
            (2, 3, 6, 7), device, dtype
        )
        # test noncontiguous_to_padded_tensor functionality
        self.assertEqual(
            torch.nested.to_padded_tensor(nt_contiguous, 0.0),
            noncontiguous_to_padded_tensor(nt_noncontiguous),
        )
        # test to_padded_tensor error message
        self.assertRaisesRegex(
            RuntimeError,
            r"for now to_padded_tensor only supports contiguous nested tensor",
            lambda: torch.nested.to_padded_tensor(nt_noncontiguous, 0.0),
        )

    @skipMeta
    def test_device_checks(self, device):
        nt = torch.nested.nested_tensor([], device=device)
        is_cuda = "cuda" in str(device)
        self.assertEqual(nt.is_cuda, is_cuda)

    @dtypes(torch.float, torch.float16, torch.double)
    def test_nested_tensor_indexing(self, device, dtype):
        # edge case: empty nested tensor
        nt0 = torch.nested.nested_tensor([])
        self.assertRaises(IndexError, lambda: nt0[0])
        # normal case
        x0 = torch.randn((2, 5), device=device, dtype=dtype)
        x1 = torch.randn((3, 4), device=device, dtype=dtype)
        nt = torch.nested.nested_tensor([x0, x1])
        # single index: only support integer in the batch dimension
        self.assertEqual(nt[0], x0)
        self.assertEqual(nt[-1], x1)
        self.assertRaises(IndexError, lambda: nt[2])
        self.assertRaises(IndexError, lambda: nt[-3])
        self.assertRaises(NotImplementedError, lambda: nt[:])
        self.assertEqual(nt[...], nt)
        # tuple of indices: only support integer in the batch dimension
        #                 + all possible indexing in the original tensor dimensions
        self.assertEqual(nt[0, 0, 0], x0[0, 0])
        self.assertEqual(nt[0, 1, :], x0[1, :])
        self.assertEqual(nt[1, ...], x1)
        self.assertRaises(IndexError, lambda: nt[1, 4, 2])
        self.assertRaises(NotImplementedError, lambda: nt[:, 1, 1])
        # test select on non-batch dimensions
        self.assertEqual(nt.select(1, 0)[0], x0.select(0, 0))
        self.assertEqual(nt.select(1, 0)[1], x1.select(0, 0))
        self.assertRaises(IndexError, lambda: nt.select(1, 3))
        self.assertEqual(nt.select(2, 0)[0], x0.select(1, 0))
        self.assertEqual(nt.select(2, 0)[1], x1.select(1, 0))
        self.assertRaises(IndexError, lambda: nt.select(2, 5))
        # make sure indexing returns a view
        nt[0].fill_(100.0)
        answer = torch.tensor(100.0, device=device, dtype=dtype).expand((2, 5))
        self.assertEqual(nt[0], answer)
        nt[1, 1, :].fill_(200.0)
        answer = torch.tensor(200.0, device=device, dtype=dtype).expand(4)
        self.assertEqual(nt[1, 1, :], answer)

        # Test that indexing works when requires_grad_(True)
        # previously this was failing because the backward kernel for select.int uses .sizes()
        nt = torch.nested.nested_tensor([x0, x1]).requires_grad_(True)
        self.assertEqual(nt[0], x0)
        self.assertEqual(nt[-1], x1)
        grad_x0 = torch.randn((2, 5), device=device, dtype=dtype)
        nt[0].backward(grad_x0)
        expected_grad = torch.nested.nested_tensor(
            [grad_x0, torch.zeros((3, 4), device=device, dtype=dtype)]
        )
        self.assertEqual(nt.grad, expected_grad)

    @parametrize(
        "func",
        [
            subtest(torch.nn.functional.relu, name="relu"),
            subtest(torch.nn.functional.relu_, name="relu_"),
            subtest(torch.nn.functional.gelu, name="gelu"),
            subtest(torch._C._nn.gelu_, name="gelu_"),
            subtest(torch.tanh, name="tanh"),
            subtest(torch.tanh_, name="tanh_"),
            subtest(torch.neg, name="neg"),
            subtest(torch.nn.functional.silu, name="silu"),
            subtest(partial(torch.nn.functional.silu, inplace=True), name="silu_"),
            subtest(torch.abs, name="abs"),
            subtest(torch.abs_, name="abs_"),
            subtest(torch.sgn, name="sgn"),
            subtest(torch.logical_not, name="logical_not"),
            subtest(torch.sin, name="sin"),
            subtest(torch.cos, name="cos"),
            subtest(torch.isinf, name="isinf"),
            subtest(torch.isposinf, name="isposinf"),
            subtest(torch.isneginf, name="isneginf"),
            subtest(torch.isnan, name="isnan"),
            subtest(torch.sqrt, name="sqrt"),
        ],
    )
    def test_unary_funcs(self, device, func):
        nt, nt_noncontiguous = random_nt_noncontiguous_pair(
            (2, 3, 6, 7), device=device, dtype=torch.float32
        )
        nested_result = func(nt)
        self.assertTrue(nested_result.is_nested)
        for t, t_res in zip(nt.unbind(), nested_result.unbind()):
            self.assertEqual(func(t), t_res)
        self.assertRaisesRegex(
            RuntimeError,
            "NestedTensor must be contiguous to get buffer.",
            lambda: func(nt_noncontiguous),
        )

    @parametrize("func", [subtest(torch.ge, name="ge"), subtest(torch.eq, name="eq")])
    def test_binary_ops_with_scalar(self, device, func):
        nt_contiguous, nt_noncontiguous = random_nt_noncontiguous_pair(
            (2, 3, 6, 7), device=device, dtype=torch.float32
        )
        scalar = 0.0

        # should work regardless of contiguity
        for nt in (nt_contiguous, nt_noncontiguous):
            nested_result = func(nt, scalar)
            self.assertTrue(nested_result.is_nested)
            for t, t_res in zip(nt.unbind(), nested_result.unbind()):
                self.assertEqual(func(t, scalar), t_res)

    @dtypes(*floating_types_and_half())
    def test_nested_tensor_chunk(self, device, dtype):
        # Transformer use case
        a = torch.randn(3, 3 * 4, device=device, dtype=dtype)
        b = torch.randn(2, 3 * 4, device=device, dtype=dtype)
        c = torch.randn(1, 3 * 4, device=device, dtype=dtype)
        a_chunks = a.chunk(3, dim=-1)
        b_chunks = b.chunk(3, dim=-1)
        c_chunks = c.chunk(3, dim=-1)

        a_nt = [a_chunks[0], b_chunks[0], c_chunks[0]]
        b_nt = [a_chunks[1], b_chunks[1], c_chunks[1]]
        c_nt = [a_chunks[2], b_chunks[2], c_chunks[2]]

        nt = torch.nested.nested_tensor([a, b, c])
        chunked = nt.chunk(3, dim=-1)

        self.assertEqual(chunked[0], torch.nested.nested_tensor(a_nt))
        self.assertEqual(chunked[1], torch.nested.nested_tensor(b_nt))
        self.assertEqual(chunked[2], torch.nested.nested_tensor(c_nt))

        for chunk in chunked:
            self.assertFalse(chunk.is_contiguous())

        # Failure chunking on ragged dimensions
        self.assertRaisesRegex(
            RuntimeError,
            "Chunk for nested tensors is currently only supported for the last dimension.",
            lambda: torch.chunk(nt, 5, dim=1),
        )
        self.assertRaisesRegex(
            RuntimeError,
            "Chunk for nested tensors is currently only supported for the last dimension.",
            lambda: torch.chunk(nt, 5, dim=0),
        )

        # Failure on non-contiguous nt
        _, nt_noncontiguous = random_nt_noncontiguous_pair((2, 3), device, dtype)
        self.assertRaisesRegex(
            RuntimeError,
            "chunk expects `self` to be contiguous.",
            lambda: torch.chunk(nt_noncontiguous, 5, dim=-1),
        )

        # Failure when calling non divisible n_chunks
        self.assertRaisesRegex(
            RuntimeError,
            "Chunk for nested tensors is only supported for "
            "nested tensors with trailing dimension divisible by chunks.",
            lambda: torch.chunk(nt, 5, dim=-1),
        )

        # Failure when calling backward on a chunk
        a = torch.randn(3, 3 * 4, device=device, dtype=dtype, requires_grad=True)
        b = torch.randn(2, 3 * 4, device=device, dtype=dtype, requires_grad=True)
        nt_grad = torch.nested.as_nested_tensor([a, b])
        chunked = torch.chunk(nt_grad, 2, dim=-1)
        self.assertRaisesRegex(
            RuntimeError,
            "Nested Strided Tensor doesn't support chunk backward.",
            lambda: chunked[0].backward(chunked[0].clone()),
        )

    @dtypes(*floating_types_and_half())
    def test_nested_tensor_split_with_sizes(self, device, dtype):
        a = torch.randn(3, 20, device=device, dtype=dtype)
        b = torch.randn(2, 20, device=device, dtype=dtype)
        c = torch.randn(1, 20, device=device, dtype=dtype)

        split_sizes = [4, 6, 10]
        a_splits = a.split_with_sizes(split_sizes, dim=-1)
        b_splits = b.split_with_sizes(split_sizes, dim=-1)
        c_splits = c.split_with_sizes(split_sizes, dim=-1)

        nt = torch.nested.nested_tensor([a, b, c])
        nt_splits = nt.split_with_sizes(split_sizes, dim=-1)

        for i, nt_split in enumerate(nt_splits):
            self.assertEqual(
                nt_split,
                torch.nested.nested_tensor([a_splits[i], b_splits[i], c_splits[i]]),
            )
            dense_strides = torch.stack(
                [
                    torch.tensor(a_splits[i].stride()),
                    torch.tensor(b_splits[i].stride()),
                    torch.tensor(c_splits[i].stride()),
                ]
            )
            self.assertEqual(nt_split._nested_tensor_strides(), dense_strides)
            self.assertFalse(nt_split.is_contiguous())

        # Failure calling on ragged dimensions
        self.assertRaisesRegex(
            RuntimeError,
            "split_with_sizes for nested tensors is currently only supported for the last dimension.",
            lambda: torch.split_with_sizes(nt, split_sizes, dim=1),
        )

        # Failure calling on non-last dimension
        self.assertRaisesRegex(
            RuntimeError,
            "split_with_sizes for nested tensors is currently only supported for the last dimension.",
            lambda: torch.split_with_sizes(nt, split_sizes, dim=0),
        )

        # Failure on non-contiguous nt
        _, nt_noncontiguous = random_nt_noncontiguous_pair((2, 3), device, dtype)
        self.assertRaisesRegex(
            RuntimeError,
            "split_with_sizes expects `self` to be contiguous.",
            lambda: torch.split_with_sizes(nt_noncontiguous, split_sizes, dim=-1),
        )

        # Failure when calling with split_sizes that don't cover the full dim size
        bad_split_sizes = [4, 6, 9]  # don't add up to 20
        self.assertRaisesRegex(
            RuntimeError,
            "split_with_sizes expects split_sizes to sum exactly to 20",
            lambda: torch.split_with_sizes(nt, bad_split_sizes, dim=-1),
        )

    @dtypes(torch.float, torch.float16, torch.double)
    @torch.inference_mode()
    def test_nested_tensor_indexing_noncontiguous(self, device, dtype):
        nt_contiguous, nt_noncontiguous = random_nt_noncontiguous_pair(
            (2, 3, 6, 7), device, dtype
        )
        self.assertEqual(nt_contiguous.size(0), nt_noncontiguous.size(0))
        n = nt_contiguous.size(0)
        for i in range(n):
            self.assertEqual(nt_contiguous[i], nt_noncontiguous[i])

    @dtypes(torch.float, torch.float16)
    @skipMeta
    @torch.inference_mode()
    @parametrize("transpose", [True, False])
    def test_nested_tensor_add(self, device, dtype, transpose):
        if transpose:
            a = torch.randn(2, 2, 2, device=device, dtype=dtype)
            b = torch.rand(2, 2, 2, device=device, dtype=dtype)
            c = a.transpose(-1, -2).contiguous()
            d = b.transpose(-1, -2).contiguous()
            nt1 = torch.nested.nested_tensor([a, b, a, b])
            nt2 = torch.nested.nested_tensor([c, d, c, d]).transpose(-1, -2)
        else:
            (nt1, nt2) = self.random_nt_pair(device, dtype, 4, (4, 4))
        ref = torch.nested.nested_tensor(
            [t1 + t2 for (t1, t2) in zip(nt1.unbind(), nt2.unbind())]
        )
        out = nt1 + nt2
        self.assertEqual(ref, out)

    @dtypes(torch.float, torch.float16)
    @skipMeta
    @torch.inference_mode()
    @parametrize("transpose", [True, False])
    def test_nested_tensor_sub(self, device, dtype, transpose):
        if transpose:
            a = torch.randn(2, 2, 2, device=device, dtype=dtype)
            b = torch.rand(2, 2, 2, device=device, dtype=dtype)
            c = a.transpose(-1, -2).contiguous()
            d = b.transpose(-1, -2).contiguous()
            nt1 = torch.nested.nested_tensor([a, b, a, b])
            nt2 = torch.nested.nested_tensor([c, d, c, d]).transpose(-1, -2)
        else:
            (nt1, nt2) = self.random_nt_pair(device, dtype, 4, (4, 4))
        ref = torch.nested.nested_tensor(
            [t1 - t2 for (t1, t2) in zip(nt1.unbind(), nt2.unbind())]
        )
        out = nt1 - nt2
        self.assertEqual(ref, out)

    @onlyCUDA
    @dtypes(torch.float, torch.float16)
    @torch.inference_mode()
    @parametrize("embedding_dim", [8, 128, 256, 384])
    def test_nested_tensor_dense_elementwise(self, device, dtype, embedding_dim):
        def _test_add_mul(nt, t):
            ref_add = torch.nested.nested_tensor(
                [t1 + t2 for (t1, t2) in zip(nt.unbind(), t.unbind())]
            )
            ref_mul = torch.nested.nested_tensor(
                [t1 * t2 for (t1, t2) in zip(nt.unbind(), t.unbind())]
            )
            self.assertEqual(nt.add(t), ref_add)
            self.assertEqual(nt.mul(t), ref_mul)

        batch_size = 32
        seq_lens = torch.randint(low=0, high=10, size=(batch_size,))

        # [B, *, D], [B, 1, D] case
        ts = [torch.randn((seq_len, embedding_dim)) for seq_len in seq_lens]
        nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
        t = torch.randn((batch_size, 1, embedding_dim), device=device, dtype=dtype)
        _test_add_mul(nt, t)

        # [B, *], [B, 1] case
        ts = [torch.randn(seq_len) for seq_len in seq_lens]
        nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
        t = torch.randn((batch_size, 1), device=device, dtype=dtype)
        _test_add_mul(nt, t)

    @dtypes(torch.float, torch.float16)
    @skipMeta
    @torch.inference_mode()
    def test_nested_tensor_mul(self, device, dtype):
        # nested tensor * nested tensor
        (nt1, nt2) = self.random_nt_pair(device, dtype, 4, (4, 4))
        ref = torch.nested.nested_tensor(
            [t1 * t2 for (t1, t2) in zip(nt1.unbind(), nt2.unbind())]
        )
        out = nt1 * nt2
        self.assertEqual(ref, out)
        # nested tensor * scalar
        number = 10.0
        scalar = torch.tensor(number).to(dtype).to(device)
        ref = torch.nested.nested_tensor([t * number for t in nt1.unbind()])
        out_number0 = nt1 * number
        out_number1 = number * nt1
        out_scalar0 = nt1 * scalar
        out_scalar1 = scalar * nt1
        self.assertEqual(out_number0, ref)
        self.assertEqual(out_number1, ref)
        self.assertEqual(out_scalar0, ref)
        self.assertEqual(out_scalar1, ref)
        # error case: numel == 1 but dim > 0
        vector = torch.tensor([number]).to(dtype).to(device)
        self.assertRaisesRegex(
            RuntimeError,
            "Expected both self and other to be nested, but got a nested self and non-nested other",
            lambda: nt1.mul(vector),
        )
        self.assertRaisesRegex(
            RuntimeError,
            "Expected both self and other to be nested, but got a non-nested self and nested other",
            lambda: vector.mul(nt1),
        )

    @dtypes(torch.float, torch.float16)
    @skipMeta
    @torch.inference_mode()
    def test_nested_tensor_div(self, device, dtype):
        nt, nt2 = self.random_nt_pair(device, dtype, 4, (4, 4))
        scale = 4.0
        ref = torch.nested.nested_tensor([t / scale for t in nt.unbind()])
        out = nt / 4.0
        self.assertEqual(ref, out)
        ref_transposed = ref.transpose(1, 2)
        out = nt.transpose(1, 2) / 4.0
        self.assertEqual(ref_transposed, out)

        ref = torch.nested.nested_tensor(
            [t / t2 for (t, t2) in zip(nt.unbind(), nt2.unbind())]
        )
        out = nt / nt2
        self.assertEqual(ref, out)

        out = nt.transpose(1, 2) / nt2.transpose(1, 2)
        self.assertEqual(ref.transpose(1, 2), out)

        nt_transpose_copy = torch.nested.nested_tensor(
            [t.transpose(0, 1) for t in nt.unbind()]
        )

        self.assertRaisesRegex(
            RuntimeError,
            "div requires strides to match when given NestedTensors",
            lambda: nt_transpose_copy.transpose(1, 2) / nt2,
        )

        nt = torch.nested.nested_tensor(
            [torch.randn(i, 4) for i in [3, 4, 5]], device=device, dtype=dtype
        )
        nt_chunks = nt.chunk(2, -1)
        self.assertRaisesRegex(
            RuntimeError,
            "div requires offsets to match when given NestedTensors",
            lambda: nt_chunks[0] / nt_chunks[1],
        )

    @dtypes(torch.float, torch.float16)
    @skipMeta
    @torch.inference_mode()
    def test_nested_tensor_add_in_place(self, device, dtype):
        (nt1, nt2) = self.random_nt_pair(device, dtype, 4, (4, 4))
        ref = torch.nested.nested_tensor(
            [t1 + t2 for (t1, t2) in zip(nt1.unbind(), nt2.unbind())]
        )
        nt1 += nt2
        self.assertEqual(ref, nt1)

    @dtypes(torch.float, torch.float16)
    @skipMeta
    @torch.inference_mode()
    def test_nested_tensor_mul_in_place(self, device, dtype):
        # nested tensor * nested tensor
        (nt1, nt2) = self.random_nt_pair(device, dtype, 4, (4, 4))
        ref = torch.nested.nested_tensor(
            [t1 * t2 for (t1, t2) in zip(nt1.unbind(), nt2.unbind())]
        )
        nt1 *= nt2
        self.assertEqual(ref, nt1)
        # nested tensor * scalar
        number = 10.0
        scalar = torch.tensor(number).to(dtype).to(device)
        ref = torch.nested.nested_tensor([t * number for t in nt1.unbind()])
        out_number = nt1.clone()
        out_number *= number
        out_scalar = nt1.clone()
        out_scalar *= scalar
        self.assertEqual(out_number, ref)
        self.assertEqual(out_scalar, ref)
        self.assertRaisesRegex(
            RuntimeError,
            r"output with shape \[.*\] doesn't match the broadcast shape \[.*\]",
            lambda: scalar.mul_(nt1),
        )
        # error case: numel == 1 but dim > 0
        vector = torch.tensor([number]).to(dtype).to(device)
        self.assertRaisesRegex(
            RuntimeError,
            "Expected both self and other to be nested, but got a nested self and non-nested other",
            lambda: nt1.mul_(vector),
        )
        self.assertRaisesRegex(
            RuntimeError,
            "Expected both self and other to be nested, but got a non-nested self and nested other",
            lambda: vector.mul_(nt1),
        )

    @onlyCPU
    @skipMeta
    @dtypes(torch.float)
    def test_nested_tensor_sum_dim(self, device, dtype):
        params = ((2, (1, 1)), ((4), (4, 4)), (10, (3, 5, 7)))

        def test_sum(device, dtype, ntensors, max_sizes, dim, keepdim=True):
            nt = random_nt(device, dtype, ntensors, max_sizes, require_non_empty=False)
            nt2 = nt.clone()
            ub2 = nt2.unbind()
            nt.requires_grad_(True)
            [t.requires_grad_(True) for t in ub2]
            nt_sum = nt.sum(dim=dim, keepdim=keepdim)
            ub2_sum = [t.sum(-1, keepdim=keepdim) for t in ub2]
            self.assertEqual(nt_sum, torch.nested.nested_tensor(ub2_sum))

            # test backward
            # generate gradient tensor that has the same size as the output
            size = nt_sum._nested_tensor_size()
            gt2 = []
            for i in range(ntensors):
                gt2.append(torch.randn(size[i].tolist(), device=device, dtype=dtype))
            gt = torch.nested.nested_tensor(gt2).clone()
            nt_sum.backward(gt)
            for t2, g2 in zip(ub2_sum, gt2):
                t2.backward(g2)
            self.assertEqual(nt.grad, torch.nested.nested_tensor([t.grad for t in ub2]))
            return

        for ntensors, max_sizes in params:
            test_sum(device, dtype, ntensors, max_sizes, len(max_sizes))

        # Test error inputs
        with self.assertRaisesRegex(
            RuntimeError, "NestedTensor can only be reduced across the last"
        ):
            torch.nested.nested_tensor(
                [torch.tensor([3, 4, 5]), torch.tensor([1, 2])]
            ).sum(0, keepdim=True)

        with self.assertRaisesRegex(
            RuntimeError, "NestedTensor only allows reduction of a single"
        ):
            torch.nested.nested_tensor(
                [torch.tensor([[3, 4, 5]]), torch.tensor([[1, 2]])]
            ).sum([0, 1], keepdim=True)

        with self.assertRaisesRegex(
            RuntimeError, "NestedTensor always requires keepdim=True for now."
        ):
            torch.nested.nested_tensor(
                [torch.tensor([3, 4, 5]), torch.tensor([1, 2])]
            ).sum(-1)

    @dtypes(torch.float, torch.float16)
    def test_contiguous(self, device, dtype):
        # Since we don't have access to the buffer in python this is harder to show what
        # we are testing for. When we call chunk on a consistent dim of a NT
        # for chunk_size > 1 the resulting tensors are views of the original NT
        # whose numels is now less than the size of the buffer. Clone was
        # previously creating a new NT with a buffer that was the same size as the
        # original.
        nt_contiguous = torch.nested.nested_tensor(
            [
                torch.randn(2, 20, device=device, dtype=dtype),
                torch.randn(4, 20, device=device, dtype=dtype),
            ]
        )
        # Split up the last dimension which has a consistent size of 20 into 5 chunks
        chunks = nt_contiguous.chunk(5, dim=-1)

        # # Check chunks are contiguous after calling contiguous
        for chunk in chunks:
            self.assertFalse(chunk.is_contiguous())
            self.assertTrue(chunk.contiguous().is_contiguous())

    @dtypes(torch.float, torch.float16)
    @skipMeta
    def test_clone(self, device, dtype):
        nt1 = random_nt(device, dtype, 4, (4, 4), (1, 1))
        nt2 = nt1.clone()
        # Verify the values match
        self.assertEqual(nt1, nt2)
        # Verify modifying nt2 doesn't affect nt1
        nt2.mul_(nt1)
        ub1 = nt1.unbind()
        ub2 = nt2.unbind()
        for i in range(len(ub1)):
            self.assertNotEqual(ub1[i], ub2[i])

        nt1.clone(memory_format=torch.preserve_format)
        msg = "Nested tensor clone supports Preserve and Contiguous memory formats, called clone with memory format: ChannelsLast"
        with self.assertRaisesRegex(RuntimeError, msg):
            nt1.clone(memory_format=torch.channels_last)

    # cannot test torch.float16 because: RuntimeError: "bernoulli_scalar_cpu_" not implemented for 'Half'
    @decorateIf(xfailIfTorchDynamo, lambda params: params["layout"] == torch.jagged)
    @dtypes(torch.float, torch.double)
    @parametrize("layout", [torch.strided, torch.jagged], name_fn=layout_name)
    def test_dropout(self, device, dtype, layout):
        # edge case: empty nested tensor
        # TODO: support empty NT in jagged layout
        if layout == torch.strided:
            nt0 = torch.nested.nested_tensor([], layout=layout)
            y = torch.nn.functional.dropout(nt0, 0.5)
            self.assertEqual(nt0, y)
        # normal nested tensor
        ntensors = 4
        if layout == torch.jagged:
            nt = random_nt(device, dtype, ntensors, (4, 4), (0, 3), layout=layout)
        else:
            nt = random_nt(device, dtype, ntensors, (4, 4), layout=layout)
        # edge case: invalid dropout
        self.assertRaises(ValueError, lambda: torch.nn.Dropout(-0.1))
        self.assertRaises(ValueError, lambda: torch.nn.Dropout(1.1))
        self.assertRaises(ValueError, lambda: torch.nn.functional.dropout(nt, -0.1))
        self.assertRaises(ValueError, lambda: torch.nn.functional.dropout(nt, 1.1))
        # edge case: no dropout
        dropouter = torch.nn.Dropout(0.0)
        y0 = dropouter(nt)
        y1 = torch.nn.functional.dropout(nt, 0.0)
        self.assertEqual(nt, y0)
        self.assertEqual(nt, y1)
        # edge case: all dropout
        dropouter = torch.nn.Dropout(1.0)
        y0 = dropouter(nt)
        y1 = torch.nn.functional.dropout(nt, 1.0)
        nt0 = torch.zeros_like(nt)
        self.assertEqual(nt0, y0)
        self.assertEqual(nt0, y1)
        # normal case: normal dropout
        p = 0.2
        y = torch.nn.functional.dropout(nt, p)
        expect = nt.clone()
        if layout == torch.jagged:
            expect = torch.where(y == 0.0, y, nt)
            expect /= 1.0 - p
            self.assertEqual(y, expect)
        else:
            expect = nt.clone()
            for i in range(ntensors):
                actual_tensor = y[i].view(-1)
                expect_tensor = expect[i].view(-1)
                for j in range(actual_tensor.shape[0]):
                    if actual_tensor[j].item() == 0.0:
                        expect_tensor[j] = 0.0
                    else:
                        expect_tensor[j] /= 1.0 - p
            self.assertEqual(y, expect)
        with freeze_rng_state():
            dropouter = torch.nn.Dropout(p)
            y0 = dropouter(nt)
        with freeze_rng_state():
            y1 = torch.nn.functional.dropout(nt, p)
        self.assertEqual(y0, y1)

    @dtypes(torch.float, torch.double)
    def test_dropout_noncontiguous(self, device, dtype):
        ntensors = 4
        nt0 = random_nt(device, dtype, ntensors, (4, 4))
        nt1 = nt0.transpose(-1, -2)
        p = 0.3
        with freeze_rng_state():
            dropouter = torch.nn.Dropout(p)
            y0 = dropouter(nt0)
        with freeze_rng_state():
            y1 = torch.nn.functional.dropout(nt1, p).transpose(-1, -2)
        self.assertEqual(y0, y1)

    # cannot test torch.float16 because: RuntimeError: "softmax_kernel_impl" not implemented for 'Half'
    @dtypes(torch.float, torch.double)
    def test_softmax(self, device, dtype):
        # normal nested tensor
        ntensors = 4
        nt = random_nt(device, dtype, ntensors, (4, 4))
        # error case: softmax across nested dimension
        self.assertRaisesRegex(
            RuntimeError,
            "Cannot apply softmax across nested dimension 0",
            lambda: torch.nn.functional.softmax(nt, 0),
        )
        self.assertRaisesRegex(
            RuntimeError,
            "Cannot apply softmax across nested dimension 0",
            lambda: torch.nn.functional.softmax(nt, -3),
        )
        # error case: dimension out of range
        self.assertRaises(IndexError, lambda: torch.nn.functional.softmax(nt, 3))
        self.assertRaises(IndexError, lambda: torch.nn.functional.softmax(nt, -4))
        # normal case: should equal to padding -inf
        softmaxer = torch.nn.Softmax(1)
        y0 = softmaxer(nt)
        y1 = torch.nn.functional.softmax(nt, 1)
        self.assertEqual(y0, y1)
        pt = torch.nested.to_padded_tensor(nt, float("-inf"))
        # if an entire slice is padded, then softmax will return 0.0 / 0.0 = nan
        # however, physically speaking that should be 0.0
        expect = torch.nn.functional.softmax(pt, 1).nan_to_num_(0.0)
        self.assertEqual(torch.nested.to_padded_tensor(y0, 0.0), expect)
        # edge case: empty nested tensor
        nt0 = torch.nested.nested_tensor([])
        y = torch.nn.functional.softmax(nt0, 1)
        self.assertEqual(nt0, y)
        # edge case: nesting scalars
        nt1 = torch.nested.nested_tensor([torch.tensor(0.0), torch.tensor(1.0)])
        self.assertRaises(RuntimeError, lambda: torch.nn.functional.softmax(nt1, 0))
        self.assertRaises(IndexError, lambda: torch.nn.functional.softmax(nt1, 1))

    @dtypes(torch.float, torch.double)
    @torch.inference_mode()
    def test_softmax_noncontiguous(self, device, dtype):
        nt_contiguous, nt_noncontiguous = random_nt_noncontiguous_pair(
            (2, 3, 6, 7), device, dtype
        )
        self.assertEqual(
            torch.nn.functional.softmax(nt_contiguous, -1),
            torch.nn.functional.softmax(nt_noncontiguous, -1),
        )

    def _test_bmm(self, device, dtype):
        # error case: not 3D tensors
        nt0 = torch.nested.nested_tensor([], device=device, dtype=dtype)
        nt1 = torch.nested.nested_tensor(
            [torch.randn(2), torch.randn(3)], device=device, dtype=dtype
        )
        nt2 = torch.nested.nested_tensor(
            [torch.randn((2, 4)), torch.randn((3, 4))], device=device, dtype=dtype
        )
        self.assertRaisesRegex(
            RuntimeError, "batch1 must be a 3D tensor", lambda: nt0.bmm(nt0)
        )
        self.assertRaisesRegex(
            RuntimeError, "batch1 must be a 3D tensor", lambda: nt0.bmm(nt1)
        )
        self.assertRaisesRegex(
            RuntimeError, "batch1 must be a 3D tensor", lambda: nt0.bmm(nt2)
        )
        self.assertRaisesRegex(
            RuntimeError, "batch1 must be a 3D tensor", lambda: nt1.bmm(nt0)
        )
        self.assertRaisesRegex(
            RuntimeError, "batch1 must be a 3D tensor", lambda: nt1.bmm(nt1)
        )
        self.assertRaisesRegex(
            RuntimeError, "batch1 must be a 3D tensor", lambda: nt1.bmm(nt2)
        )
        self.assertRaisesRegex(
            RuntimeError, "batch2 must be a 3D tensor", lambda: nt2.bmm(nt0)
        )
        self.assertRaisesRegex(
            RuntimeError, "batch2 must be a 3D tensor", lambda: nt2.bmm(nt1)
        )
        # error case: incompatible batch size
        nt0 = torch.nested.nested_tensor(
            [torch.randn((2, 4)), torch.randn((3, 4))], device=device, dtype=dtype
        )
        nt1 = torch.nested.nested_tensor(
            [torch.randn((4, 6)), torch.randn((4, 5)), torch.randn((4, 7))],
            device=device,
            dtype=dtype,
        )
        self.assertRaisesRegex(
            RuntimeError,
            "Expected size for the 1st dimension of batch2 tensor to be: 2 but got: 3.",
            lambda: nt0.bmm(nt1),
        )
        self.assertRaisesRegex(
            RuntimeError,
            "Expected size for the 1st dimension of batch2 tensor to be: 3 but got: 2.",
            lambda: nt1.bmm(nt0),
        )
        # error case: underlying matrices cannot be multiplied
        nt0 = torch.nested.nested_tensor(
            [torch.randn((2, 4)), torch.randn((3, 4))], device=device, dtype=dtype
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"0-th nested matrices in batch cannot be multiplied \(2x4 and 2x4\)",
            lambda: nt0.bmm(nt0),
        )
        # normal nested tensor
        nt0 = torch.nested.nested_tensor(
            [torch.randn((2, 4)), torch.randn((3, 7))], device=device, dtype=dtype
        )
        nt1 = torch.nested.nested_tensor(
            [torch.randn((4, 6)), torch.randn((7, 5))], device=device, dtype=dtype
        )
        actual = torch.nested.to_padded_tensor(nt0.bmm(nt1), 0.0)
        expect = torch.nested.to_padded_tensor(nt0, 0.0).bmm(
            torch.nested.to_padded_tensor(nt1, 0.0)
        )
        if dtype == torch.float16:
            self.assertEqual(actual, expect, rtol=1e-3, atol=1e-3)
        else:
            self.assertEqual(actual, expect)

        # nested tensor bmm normal tensor
        nt0 = torch.nested.nested_tensor(
            [torch.randn((2, 7)), torch.randn((3, 7))], device=device, dtype=dtype
        )
        nt1 = torch.rand(2, 7, 5, dtype=dtype, device=device)
        actual = torch.nested.to_padded_tensor(nt0.bmm(nt1), 0.0)
        expect = torch.nested.to_padded_tensor(nt0, 0.0).bmm(nt1)
        if dtype == torch.float16:
            self.assertEqual(actual, expect, rtol=1e-3, atol=1e-3)
        else:
            self.assertEqual(actual, expect)

        # nested tensor bmm normal tensor with non-contiguous view
        nt1 = torch.rand(2, 5, 7, dtype=dtype, device=device)
        nt1 = nt1.transpose(1, 2)
        actual = torch.nested.to_padded_tensor(nt0.bmm(nt1), 0.0)
        expect = torch.nested.to_padded_tensor(nt0, 0.0).bmm(nt1)
        if dtype == torch.float16:
            self.assertEqual(actual, expect, rtol=1e-3, atol=1e-3)
        else:
            self.assertEqual(actual, expect)

        # normal tensor bmm nested tensor
        nt0 = torch.rand(2, 5, 7, dtype=dtype, device=device)
        nt1 = torch.nested.nested_tensor(
            [torch.randn((7, 6)), torch.randn((7, 5))], device=device, dtype=dtype
        )
        actual = torch.nested.to_padded_tensor(nt0.bmm(nt1), 0.0)
        expect = nt0.bmm(torch.nested.to_padded_tensor(nt1, 0.0))
        if dtype == torch.float16:
            self.assertEqual(actual, expect, rtol=1e-3, atol=1e-3)
        else:
            self.assertEqual(actual, expect)

        # test tensorcore path
        nt0 = torch.nested.nested_tensor(
            [torch.randn((2, 8)), torch.randn((3, 16))], device=device, dtype=dtype
        )
        nt1 = torch.nested.nested_tensor(
            [torch.randn((8, 8)), torch.randn((16, 8))], device=device, dtype=dtype
        )
        actual = torch.nested.to_padded_tensor(nt0.bmm(nt1), 0.0)
        expect = torch.nested.to_padded_tensor(nt0, 0.0).bmm(
            torch.nested.to_padded_tensor(nt1, 0.0)
        )
        if dtype == torch.float16:
            self.assertEqual(actual, expect, rtol=1e-3, atol=1e-3)
        else:
            self.assertEqual(actual, expect)

    @onlyCUDA
    @dtypes(torch.float, torch.double, torch.float16)
    def test_bmm_cuda(self, device, dtype):
        self._test_bmm(device, dtype)

    @onlyCPU
    # cannot test torch.float16 because: RuntimeError: "addmm_impl_cpu_" not implemented for 'Half'
    @dtypes(torch.float, torch.double)
    def test_bmm_cpu(self, device, dtype):
        self._test_bmm(device, dtype)

    # cannot test torch.float16 because: RuntimeError: "addmm_impl_cpu_" not implemented for 'Half'
    @dtypes(torch.float, torch.double)
    def test_bmm_noncontiguous(self, device, dtype):
        nt0_contiguous, nt0_noncontiguous = random_nt_noncontiguous_pair(
            (2, 3), device, dtype
        )
        nt1_contiguous, nt1_noncontiguous = random_nt_noncontiguous_pair(
            (6, 7), device, dtype
        )
        self.assertEqual(
            nt0_contiguous.transpose(-1, -2).bmm(nt1_contiguous),
            nt0_noncontiguous.transpose(-1, -2).bmm(nt1_noncontiguous),
        )

    @dtypes(torch.float, torch.double)
    def test_matmul_with_bmm_path(self, device, dtype):
        def unbind_rebind_matmul(nt1, nt2):
            t1s = nt1.unbind()
            t2s = nt2.unbind()
            out_ts = [t1.matmul(t2) for t1, t2 in zip(t1s, t2s)]
            return torch.nested.nested_tensor(out_ts)

        # [N, n_head, *, head_dim], [N, n_head, head_dim, *]
        Ns = [1, 2, 5]
        n_heads = np.random.randint(2, 5)
        head_dim = 3
        t1s = []
        t2s = []
        for N in Ns:
            for _ in range(N):
                seq_len1 = np.random.randint(2, 5)
                seq_len2 = np.random.randint(2, 5)
                t1s.append(torch.randn(n_heads, seq_len1, head_dim))
                t2s.append(torch.randn(n_heads, head_dim, seq_len2))
            nt1 = torch.nested.nested_tensor(t1s, device=device, dtype=dtype)
            nt2 = torch.nested.nested_tensor(t2s, device=device, dtype=dtype)
            self.assertEqual(torch.matmul(nt1, nt2), unbind_rebind_matmul(nt1, nt2))

        # test with noncontiguous
        t3s = []
        t4s = []
        for _ in range(N):
            seq_len = np.random.randint(2, 5)
            t3s.append(torch.randn(seq_len, n_heads, head_dim))
            t4s.append(torch.randn(seq_len, n_heads, head_dim))
        nt3 = torch.nested.nested_tensor(t3s, device=device, dtype=dtype).transpose(
            1, 2
        )
        nt4 = (
            torch.nested.nested_tensor(t4s, device=device, dtype=dtype)
            .transpose(1, 2)
            .transpose(2, 3)
        )
        self.assertEqual(torch.matmul(nt3, nt4), unbind_rebind_matmul(nt3, nt4))

    # cannot test torch.float16 because: RuntimeError: "bmm" not implemented for 'Half'
    @dtypes(torch.float, torch.double)
    def test_matmul(self, device, dtype):
        # error case: one is nested but the other is not
        nt = torch.nested.nested_tensor(
            [torch.randn(2), torch.randn(3)], device=device, dtype=dtype
        )
        t = torch.randn(4, device=device, dtype=dtype)
        self.assertRaisesRegex(
            RuntimeError,
            "Expected both to be nested, but got a nested self and non-nested other",
            lambda: torch.matmul(nt, t),
        )
        self.assertRaisesRegex(
            RuntimeError,
            "Expected both to be nested, but got a non-nested self and nested other",
            lambda: torch.matmul(t, nt),
        )
        # error case: not 3+D tensors
        nt0 = torch.nested.nested_tensor([], device=device, dtype=dtype)
        nt1 = torch.nested.nested_tensor(
            [torch.randn(2), torch.randn(3)], device=device, dtype=dtype
        )
        nt2 = torch.nested.nested_tensor(
            [torch.randn((2, 4)), torch.randn((3, 4))], device=device, dtype=dtype
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"matmul: For nested tensors, only inputs with >= 3 dims are currently supported. 1st input has rank: [0-9]+",
            lambda: torch.matmul(nt0, nt0),
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"matmul: For nested tensors, only inputs with >= 3 dims are currently supported. 1st input has rank: [0-9]+",
            lambda: torch.matmul(nt0, nt1),
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"matmul: For nested tensors, only inputs with >= 3 dims are currently supported. 1st input has rank: [0-9]+",
            lambda: torch.matmul(nt0, nt2),
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"matmul: For nested tensors, only inputs with >= 3 dims are currently supported. 1st input has rank: [0-9]+",
            lambda: torch.matmul(nt1, nt0),
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"matmul: For nested tensors, only inputs with >= 3 dims are currently supported. 1st input has rank: [0-9]+",
            lambda: torch.matmul(nt1, nt1),
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"matmul: For nested tensors, only inputs with >= 3 dims are currently supported. 1st input has rank: [0-9]+",
            lambda: torch.matmul(nt1, nt2),
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"matmul: For nested tensors, only inputs with >= 3 dims are currently supported. 2nd input has rank: [0-9]+",
            lambda: torch.matmul(nt2, nt0),
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"matmul: For nested tensors, only inputs with >= 3 dims are currently supported. 2nd input has rank: [0-9]+",
            lambda: torch.matmul(nt2, nt1),
        )
        # error case: incompatible batch size
        nt0 = torch.nested.nested_tensor(
            [torch.randn((2, 4)), torch.randn((3, 4))], device=device, dtype=dtype
        )
        nt1 = torch.nested.nested_tensor(
            [torch.randn((4, 6)), torch.randn((4, 5)), torch.randn((4, 7))],
            device=device,
            dtype=dtype,
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"matmul: Expected size for the 1st dimension of 2nd input tensor to be: [0-9]+ but got: [0-9]+.",
            lambda: torch.matmul(nt0, nt1),
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"matmul: Expected size for the 1st dimension of 2nd input tensor to be: [0-9]+ but got: [0-9]+.",
            lambda: torch.matmul(nt1, nt0),
        )
        # error case: incompatible (wrong) batch sizes that shouldn't even broadcast?
        nt0 = torch.nested.nested_tensor(
            [torch.randn((2, 2, 4)), torch.randn((2, 3, 4))], device=device, dtype=dtype
        )
        nt1 = torch.nested.nested_tensor(
            [torch.randn((3, 4, 6)), torch.randn((3, 4, 5))], device=device, dtype=dtype
        )
        self.assertRaisesRegex(
            RuntimeError,
            "matmul(): For nested tensors, batch dimensions must have the same sizes,",
            lambda: torch.matmul(nt0, nt1),
        )
        # error case: incompatible batch sizes that should technically broadcast
        nt0 = torch.nested.nested_tensor(
            [torch.randn((2, 2, 4)), torch.randn((1, 3, 4))], device=device, dtype=dtype
        )
        nt1 = torch.nested.nested_tensor(
            [torch.randn((1, 4, 6)), torch.randn((3, 4, 5))], device=device, dtype=dtype
        )
        self.assertRaisesRegex(
            RuntimeError,
            "matmul(): For nested tensors, batch dimensions must have the same sizes,",
            lambda: torch.matmul(nt0, nt1),
        )
        # error case: underlying matrices cannot be multiplied
        nt0 = torch.nested.nested_tensor(
            [torch.randn((2, 4)), torch.randn((3, 4))], device=device, dtype=dtype
        )
        self.assertRaisesRegex(
            RuntimeError,
            "matmul(): Nested tensors cannot be matrix multiplied",
            lambda: torch.matmul(nt0, nt0),
        )
        # normal nested tensor: 3D
        nt0 = torch.nested.nested_tensor(
            [torch.randn((2, 4)), torch.randn((3, 7))], device=device, dtype=dtype
        )
        nt1 = torch.nested.nested_tensor(
            [torch.randn((4, 6)), torch.randn((7, 5))], device=device, dtype=dtype
        )
        actual = torch.nested.to_padded_tensor(torch.matmul(nt0, nt1), 0.0)
        expect = torch.matmul(
            torch.nested.to_padded_tensor(nt0, 0.0),
            torch.nested.to_padded_tensor(nt1, 0.0),
        )
        self.assertEqual(actual, expect)
        # normal nested tensor: 4D (with testing for batch_size=1)
        nt0 = torch.nested.nested_tensor(
            [torch.randn((1, 2, 4)), torch.randn((8, 3, 7))], device=device, dtype=dtype
        )
        nt1 = torch.nested.nested_tensor(
            [torch.randn((1, 4, 6)), torch.randn((8, 7, 5))], device=device, dtype=dtype
        )
        actual = torch.nested.to_padded_tensor(torch.matmul(nt0, nt1), 0.0)
        expect = torch.matmul(
            torch.nested.to_padded_tensor(nt0, 0.0),
            torch.nested.to_padded_tensor(nt1, 0.0),
        )
        self.assertEqual(actual, expect)
        # normal nested tensor: 5D
        nt0 = torch.nested.nested_tensor(
            [torch.randn((8, 9, 2, 4)), torch.randn((8, 9, 3, 7))],
            device=device,
            dtype=dtype,
        )
        nt1 = torch.nested.nested_tensor(
            [torch.randn((8, 9, 4, 6)), torch.randn((8, 9, 7, 5))],
            device=device,
            dtype=dtype,
        )
        actual = torch.nested.to_padded_tensor(torch.matmul(nt0, nt1), 0.0)
        expect = torch.matmul(
            torch.nested.to_padded_tensor(nt0, 0.0),
            torch.nested.to_padded_tensor(nt1, 0.0),
        )
        self.assertEqual(actual, expect)

    # only supported on CUDA for now
    @dtypes(torch.float, torch.double)
    def test_matmul_nt_with_broadcasted_t(self, device, dtype):
        # NT (B, *, C, D) with T (D, E) broadcasting case
        nt = random_nt_from_dims([3, None, 4, 5], device=device, dtype=dtype)
        t = torch.randn(5, 6, device=device, dtype=dtype)
        output = torch.matmul(nt, t)

        # should be equivalent to matmul-ing each component with the dense tensor
        self.assertEqual(nt.size(0), output.size(0))
        for component, out_component in zip(nt, output):
            self.assertEqual(out_component, torch.matmul(component, t))

    # cannot test torch.float16 because: RuntimeError: "bmm" not implemented for 'Half'
    @dtypes(torch.float, torch.double)
    def test_matmul_noncontiguous(self, device, dtype):
        nt0_contiguous, nt0_noncontiguous = random_nt_noncontiguous_pair(
            (2, 3), device, dtype
        )
        nt1_contiguous, nt1_noncontiguous = random_nt_noncontiguous_pair(
            (6, 7), device, dtype
        )
        self.assertEqual(
            torch.matmul(nt0_contiguous.transpose(-1, -2), nt1_contiguous),
            torch.matmul(nt0_noncontiguous.transpose(-1, -2), nt1_noncontiguous),
        )

    @dtypes(torch.float, torch.double)
    def test_linear(self, device, dtype):
        a = torch.randn(1, 2, device=device, dtype=dtype)
        b = torch.randn(2, 2, device=device, dtype=dtype)
        c = torch.randn(3, 2, device=device, dtype=dtype)
        nt = torch.nested.nested_tensor([a, b, c])

        weight = torch.randn(2, 2, device=device, dtype=dtype)
        bias = torch.randn(2, device=device, dtype=dtype)
        # success case
        torch.functional.F.linear(nt, weight, bias)

        # invalid nested tensor dimension
        msg = r"Linear requires nested_tensor.dim == 3 and dense_matrix.dim == 2. Nested tensor dim: 2. Dense tensor dim: 2"
        nt1 = torch.nested.nested_tensor(
            [
                torch.randn(1, device=device, dtype=dtype),
                torch.randn(2, device=device, dtype=dtype),
            ]
        )
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.functional.F.linear(nt1, weight, bias)

        # invalid weight shape
        msg = r"Linear requires nested_tensor.dim == 3 and dense_matrix.dim == 2. Nested tensor dim: 3. Dense tensor dim: 3"
        weight1 = torch.randn(2, 2, 3, device=device, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.functional.F.linear(nt, weight1, bias)

        # inconsistent last dim of nested tensor
        msg = r"Expected all tensors in nested tensor to have the same trailing dimension, instead last dimension equals:"
        nt2 = torch.nested.nested_tensor(
            [
                torch.randn(1, 2, device=device, dtype=dtype),
                torch.randn(2, 3, device=device, dtype=dtype),
            ]
        )
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.functional.F.linear(nt2, weight, bias)

        # Mismatch of nested tensor last dim and weight dimension
        weight2 = torch.randn(2, 4, device=device, dtype=dtype)
        msg = (
            r"Shape mismatch for NestedTensor Linear: Expected input's \(a nested tensor\) 'last_dim'"
            r" to equal 'weight.size\(1\), but got: last_dim = 2, and weight.size\(1\) = 4"
        )
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.functional.F.linear(nt, weight2, bias)

        # Nested tensor input and nested weight
        nt_weight = nt.clone()
        msg = r"Linear does not support nested weight when input is a nested tensor."
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.functional.F.linear(nt, nt_weight, bias)

    # TODO: test noncontiguous linear
    # For now this tests the error message of linear
    # since linear does not support noncontiguous buffer yet
    @dtypes(torch.float, torch.double)
    def test_linear_noncontiguous(self, device, dtype):
        nt_contiguous, nt_noncontiguous = random_nt_noncontiguous_pair(
            (2, 3, 6, 7), device, dtype
        )
        weight = torch.randn((8, 5), device=device, dtype=dtype)
        self.assertRaisesRegex(
            RuntimeError,
            r"for now linear only supports contiguous nested tensor",
            lambda: torch.nn.functional.linear(nt_noncontiguous, weight),
        )

    @dtypes(torch.float, torch.float16, torch.double)
    def test_to_padded_tensor_zero_numel_errors(self, device, dtype):
        ts = [torch.ones(1, 0), torch.ones(0, 0)]
        nt = torch.nested.nested_tensor(
            ts, device=device, dtype=dtype, layout=torch.strided
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"at least one constituent tensor should have non-zero numel",
            lambda: torch.nested.to_padded_tensor(nt, 0.0),
        )

    @dtypes(torch.float, torch.float16, torch.double)
    def test_transpose(self, device, dtype):
        nt = random_nt(device, dtype, 4, (4, 4))
        # error case: transpose nested dimension
        self.assertRaisesRegex(
            RuntimeError,
            "Nested tensor dimension 0 cannot be transposed",
            lambda: nt.transpose(0, 1),
        )
        self.assertRaisesRegex(
            RuntimeError,
            "Nested tensor dimension 0 cannot be transposed",
            lambda: nt.transpose(1, -3),
        )
        # error case: dimension out of range
        self.assertRaises(IndexError, lambda: nt.transpose(1, 3))
        self.assertRaises(IndexError, lambda: nt.transpose(-4, -1))
        # normal case
        ntT = nt.transpose(-1, -2)
        ptT_from_ntT = noncontiguous_to_padded_tensor(ntT)
        pt = torch.nested.to_padded_tensor(nt, 0.0)
        ptT = pt.transpose(-1, -2)
        self.assertEqual(ptT, ptT_from_ntT)

    @dtypes(torch.float, torch.float16, torch.double)
    def test_squeeze_unsqueeze(self, device, dtype):
        a = torch.arange(6).reshape(2, 3)
        b = torch.arange(15).reshape(5, 3)
        nt = torch.nested.nested_tensor([a, b], device=device, dtype=dtype)
        # error case: squeeze no dimension
        self.assertRaisesRegex(
            RuntimeError,
            "For nested tensors, squeeze without the dim argument",
            lambda: nt.squeeze(),
        )
        # error case: squeeze nested dimension
        self.assertRaisesRegex(
            RuntimeError,
            "For nested tensors, squeezing dimension 0",
            lambda: nt.squeeze(0),
        )
        # error case: dimension out of range
        self.assertRaises(IndexError, lambda: nt.squeeze(3))
        # error case: squeeze nested tensor of singleton tensors
        c = torch.ones(1)
        nt_singleton = torch.nested.nested_tensor([c, c], device=device, dtype=dtype)
        self.assertRaisesRegex(
            RuntimeError,
            "For nested tensors, squeezing a nested tensor of singleton",
            lambda: nt_singleton.squeeze(1),
        )

        # squeezing a dim which does not have size 1 should be a no-op
        nt2 = nt.squeeze(-1)
        self.assertEqual(nt, nt2)

        # test cases that should work
        nt_sizes = nt._nested_tensor_size()
        nt_strides = nt._nested_tensor_strides()
        for i in range(-2, 4):
            if i == 0:
                # cannot unsqueeze batch dim
                continue
            nt_unsqueezed = nt.unsqueeze(i)
            # negative dim will correspond to unsqueeze() applied at dim = dim + nt.dim() + 1
            wrapped_i = i + nt.dim() + 1 if i < 0 else i
            # col_index into nt size tensor is requires subtraction of 1 to ignore batch dim
            size_idx = wrapped_i - 1
            self.assertEqual(
                nt_unsqueezed._nested_tensor_size()[:, size_idx],
                torch.ones(2, dtype=torch.long),
            )
            unsqueezed_stride = nt_unsqueezed._nested_tensor_strides()[:, size_idx]
            if i == nt.ndim or i == -1:
                self.assertEqual(unsqueezed_stride, torch.ones(2, dtype=torch.long))
            else:
                stride_col_after = nt_strides[:, size_idx]
                size_col_after = nt_sizes[:, size_idx]
                self.assertEqual(unsqueezed_stride, stride_col_after * size_col_after)
            nt_squeezed = nt_unsqueezed.squeeze(i)
            self.assertEqual(nt_squeezed, nt)
            self.assertEqual(nt_squeezed._nested_tensor_size(), nt_sizes)
            self.assertEqual(nt_squeezed._nested_tensor_strides(), nt_strides)

    @dtypes(torch.float, torch.float16, torch.double)
    def test_transpose_inference_mode_interaction(self, device, dtype):
        nt = random_nt(device, dtype, 4, (4, 4))
        # Construct in default mode and transpose while in inference mode
        with torch.inference_mode():
            ntT = nt.transpose(-1, -2)
            ptT_from_ntT = noncontiguous_to_padded_tensor(ntT)
            pt = torch.nested.to_padded_tensor(nt, 0.0)
            ptT = pt.transpose(-1, -2)
            self.assertEqual(ptT, ptT_from_ntT)

        # Construct and transpose while in inference mode
        with torch.inference_mode():
            nt = random_nt(device, dtype, 4, (4, 4))
            ntT = nt.transpose(-1, -2)
            ptT_from_ntT = noncontiguous_to_padded_tensor(ntT)
            pt = torch.nested.to_padded_tensor(nt, 0.0)
            ptT = pt.transpose(-1, -2)
            self.assertEqual(ptT, ptT_from_ntT)

    @dtypes(torch.float, torch.float16, torch.double)
    def test_view(self, device, dtype):
        nt = random_nt(device, dtype, 4, (4, 4))
        # error case: empty shape
        self.assertRaisesRegex(
            RuntimeError,
            r"shape '\[\]' is invalid for a nested tensor",
            lambda: nt.view(()),
        )
        # error case: empty nested tensor
        nt_empty = torch.nested.nested_tensor([])
        self.assertRaisesRegex(
            RuntimeError,
            "empty nested tensor cannot be reshaped",
            lambda: nt_empty.view(-1),
        )
        # error case: -1 for batch size
        self.assertRaisesRegex(
            RuntimeError,
            r"view: For now nested view cannot change or infer the implicit batch dimension",
            lambda: nt.view(-1, 2, 3),
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"shape '\[.*\]' is invalid for input of size [0-9]+",
            lambda: nt.view(4, 2, 3),
        )
        # normal case
        x0 = torch.randn((2, 20), device=device, dtype=dtype)
        x1 = torch.randn((3, 20), device=device, dtype=dtype)
        nt = torch.nested.nested_tensor([x0, x1])
        pt = torch.nested.to_padded_tensor(nt, 0.0)
        # error case, trying to reshape batch dim to a legit shape
        self.assertRaisesRegex(
            RuntimeError,
            r"For now nested view cannot change or infer the implicit batch dimension",
            lambda: nt.transpose(-1, -2).view(40, -1),
        )
        # inherit only the ragged dimension
        # (2, 20) -> (2, 5, 4)
        # (3, 20) -> (3, 5, 4)
        nt1 = nt.view(2, -1, 5, 4)
        # (2, 3, 20) -> (2, 3, 5, 4) -> (2, 4, 5, 4)
        pt1 = pt.view(2, -1, 5, 4)
        self.assertEqual(noncontiguous_to_padded_tensor(nt1), pt1)

        # more than one -1 (even for "old" dims), should fail
        # this attempts to do # (2, (2, 3), 5, 4) -> (2, (2, 3), 5, 2, 2)
        # but we ban "inherit old behavior" for >1 dimension
        self.assertRaisesRegex(
            RuntimeError,
            r"only one dimension can be inferred",
            lambda: nt1.view(2, -1, -1, 2, 2),
        )

    @dtypes(torch.float, torch.float16, torch.double)
    def test_view_inference_mode_interaction(self, device, dtype):
        # Construct in default mode and view while in inference mode
        nt = torch.nested.nested_tensor(
            [torch.randn((2, 20)), torch.randn((3, 20))], device=device, dtype=dtype
        )
        with torch.inference_mode():
            ntT = nt.view(2, -1, 4, 5)
            ptT_from_ntT = noncontiguous_to_padded_tensor(ntT)
            pt = torch.nested.to_padded_tensor(nt, 0.0)
            ptT = pt.view(2, -1, 4, 5)
            self.assertEqual(ptT, ptT_from_ntT)
        # Construct and view while in inference mode
        with torch.inference_mode():
            nt = torch.nested.nested_tensor(
                [torch.randn((2, 20)), torch.randn((3, 20))], device=device, dtype=dtype
            )
            ntT = nt.view(2, -1, 4, 5)
            ptT_from_ntT = noncontiguous_to_padded_tensor(ntT)
            pt = torch.nested.to_padded_tensor(nt, 0.0)
            ptT = pt.view(2, -1, 4, 5)
            self.assertEqual(ptT, ptT_from_ntT)

    @dtypes(torch.float, torch.float16, torch.double)
    def test_reshape(self, device, dtype):
        nt = random_nt(device, dtype, 4, (4, 4))
        # error case: empty shape
        self.assertRaisesRegex(
            RuntimeError,
            r"shape '\[\]' is invalid for a nested tensor",
            lambda: nt.reshape(()),
        )
        # error case: empty nested tensor
        nt_empty = torch.nested.nested_tensor([])
        self.assertRaisesRegex(
            RuntimeError,
            "empty nested tensor cannot be reshaped",
            lambda: nt_empty.reshape(-1),
        )
        # error case: -1 for batch size
        self.assertRaisesRegex(
            RuntimeError,
            r"reshape: For now nested reshape cannot change or infer the implicit batch dimension",
            lambda: nt.reshape(-1, 2, 3),
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"shape '\[.*\]' is invalid for input of size [0-9]+",
            lambda: nt.reshape(4, 2, 3),
        )
        # normal case
        x0 = torch.randn((2, 20), device=device, dtype=dtype)
        x1 = torch.randn((3, 20), device=device, dtype=dtype)
        nt = torch.nested.nested_tensor([x0, x1])  # (2, (2, 3), 20)
        pt = torch.nested.to_padded_tensor(nt, 0.0)
        # error case, trying to reshape batch dim to a legit shape
        self.assertRaisesRegex(
            RuntimeError,
            r"reshape: For now nested reshape cannot change or infer the implicit batch dimension",
            lambda: nt.transpose(-1, -2).reshape(40, -1),
        )
        # inherit only the ragged dimension
        # (2, 20) -> (2, 5, 4)
        # (3, 20) -> (3, 5, 4)
        nt1 = nt.reshape(2, -1, 5, 4)
        # (2, 3, 20) -> (2, 3, 5, 4) -> (2, 4, 5, 4)
        pt1 = pt.reshape(2, -1, 5, 4)
        self.assertEqual(noncontiguous_to_padded_tensor(nt1), pt1)

        # more than one -1 (even for "old" dims), should fail
        # this attempts to do # (2, (2, 3), 5, 4) -> (2, (2, 3), 5, 2, 2)
        # but we ban "inherit old behavior" for >1 dimension
        self.assertRaisesRegex(
            RuntimeError,
            r"only one dimension can be inferred",
            lambda: nt1.reshape(2, -1, -1, 2, 2),
        )

    def test_nested_masked_select(self, device):
        t = torch.randn([3, 3], device=device)
        mask = torch.tensor([False], device=device)

        njt = torch.nested.masked_select(t, mask)
        self.assertEqual(njt.values(), torch.tensor([], device=device))
        self.assertEqual(njt.offsets(), torch.tensor([0, 0, 0, 0], device=device))

        mask = torch.tensor([[False], [False], [True]], device=device)
        njt = torch.nested.masked_select(t, mask)
        self.assertEqual(njt.values(), t[-1], atol=0.1, rtol=0.1)
        self.assertEqual(njt.offsets(), torch.tensor([0, 0, 0, 3], device=device))

        mask = torch.tensor(
            [[False, False, True], [True, False, True], [False, False, True]],
            device=device,
        )
        njt = torch.nested.masked_select(t, mask)
        self.assertEqual(njt.values(), t.masked_select(mask))
        self.assertEqual(njt.offsets(), torch.tensor([0, 1, 3, 4], device=device))

        t = torch.randn([2, 3, 3, 1], device=device)
        mask = torch.tensor(
            [
                [
                    [[True], [False], [True]],
                    [[True], [False], [True]],
                    [[True], [False], [True]],
                ],
                [
                    [[False], [True], [True]],
                    [[False], [True], [True]],
                    [[True], [True], [True]],
                ],
            ],
            device=device,
        )
        njt = torch.nested.masked_select(t, mask)
        self.assertEqual(njt.values(), t.masked_select(mask))
        self.assertEqual(
            njt.offsets(),
            torch.tensor(
                [0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 8, 8, 9, 10, 11, 12, 13],
                device=device,
            ),
        )

    @dtypes(torch.float, torch.float16, torch.double)
    def test_narrow(self, device, dtype):
        nt = random_nt_from_dims([5, None, None, None], device=device, dtype=dtype)

        # narrow on dim=0 from start to end
        bounds = [(0, 5), (0, 3), (1, 2), (1, 5), (2, 4)]
        for start, end in bounds:
            length = end - start
            narrowed = nt.narrow(dim=0, start=start, length=length)
            # ensure output is a view
            self.assertTrue(narrowed._base is nt)
            for nc, c in zip(narrowed.unbind(), nt.unbind()[start:end]):
                self.assertEqual(nc, c)

        # dim != 0 is not supported
        for dim in range(1, nt.dim()):
            with self.assertRaisesRegex(
                RuntimeError, "only dim=0 supported for nested tensors"
            ):
                nt.narrow(dim=dim, start=0, length=1)

        # error case: non-contiguous NT
        _, nt_noncont = random_nt_noncontiguous_pair((2, 3, 4))
        with self.assertRaisesRegex(
            RuntimeError, "only contiguous nested tensors supported"
        ):
            nt_noncont.narrow(dim=0, start=0, length=1)

    @parametrize("input_dim", [3, 4])
    def test_scaled_dot_product_attention(self, device, input_dim):
        def rand_tensor(*shape):
            return torch.randn(shape, device=device)

        E = 8
        if input_dim == 3:
            # Shape: (N, L, E); ragged L
            query = torch.nested.nested_tensor(
                [rand_tensor(2, E), rand_tensor(3, E), rand_tensor(4, E)]
            )

            # Shape: (N, S, E); ragged S
            key = torch.nested.nested_tensor(
                [rand_tensor(3, E), rand_tensor(4, E), rand_tensor(5, E)]
            )
            value = torch.nested.nested_tensor(
                [rand_tensor(3, E), rand_tensor(4, E), rand_tensor(5, E)]
            )
        elif input_dim == 4:
            # In the 4D case the L and S is ragged
            # Shape: (N, N', L, E); ragged N' and L
            query = torch.nested.nested_tensor(
                [rand_tensor(2, 2, E), rand_tensor(3, 3, E), rand_tensor(4, 4, E)]
            )
            # Shape: (N, N', S, E); ragged N' and S
            key = torch.nested.nested_tensor(
                [rand_tensor(2, 3, E), rand_tensor(3, 4, E), rand_tensor(4, 5, E)]
            )
            value = torch.nested.nested_tensor(
                [rand_tensor(2, 3, E), rand_tensor(3, 4, E), rand_tensor(4, 5, E)]
            )
        else:
            self.fail(f"Invalid input_dim {input_dim} encountered in SDP test")

        def rand_mask(size):
            return torch.randint(0, 2, size=size, dtype=torch.bool, device=device)

        # Shape: (N, L, S); ragged L and S matching above
        attn_mask = torch.nested.nested_tensor(
            [rand_mask((2, 3)), rand_mask((3, 4)), rand_mask((4, 5))]
        )

        dropout_p = 0.0  # no dropout for reproducibility

        # Success case: no attn_mask set and is_causal=False.
        actual = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=None, is_causal=False, dropout_p=dropout_p
        )

        expected_outputs = []
        for q, k, v in zip(query.unbind(), key.unbind(), value.unbind()):
            output = torch.nn.functional.scaled_dot_product_attention(
                q.unsqueeze(0),
                k.unsqueeze(0),
                v.unsqueeze(0),
                attn_mask=None,
                dropout_p=dropout_p,
            )
            expected_outputs.append(output.squeeze(0))
        expected_output_nested = torch.nested.nested_tensor(expected_outputs)
        self.assertEqual(actual, expected_output_nested)

        # Error case: explicit attn_mask set.
        with self.assertRaisesRegex(
            RuntimeError, "not supported when an explicit attn_mask is set"
        ):
            torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=attn_mask, dropout_p=dropout_p
            )

        # Error case: is_causal=True.
        with self.assertRaisesRegex(RuntimeError, "not supported when is_causal=True"):
            torch.nn.functional.scaled_dot_product_attention(
                query, key, value, dropout_p=dropout_p, is_causal=True
            )

    @dtypes(torch.float, torch.float16, torch.double)
    def test_empty_like(self, device, dtype):
        ntensors = 4
        nt = random_nt(device, dtype, ntensors, (4, 4))

        # Create empty on same device as original nested tensor
        nt_empty = torch.empty_like(nt)
        assert nt.is_same_size(nt_empty)
        self.assertEqual(nt.dtype, nt_empty.dtype)
        self.assertEqual(nt.device, nt_empty.device)
        self.assertEqual(nt.layout, nt_empty.layout)

        if torch.cuda.is_available():
            if device == "cpu":
                nt_cuda = torch.empty_like(nt, device="cuda")
                self.assertEqual(torch.device("cuda").type, nt_cuda.device.type)
            else:
                nt_cpu = torch.empty_like(nt, device="cpu")
                self.assertEqual(torch.device("cpu").type, nt_cpu.device.type)

        # Check changing dtype of empty_like nested tensor output
        dtype_set = {torch.float, torch.float16, torch.double}
        for other_dtype in dtype_set - {dtype}:
            nt_empty_other_dtype = torch.empty_like(nt, dtype=other_dtype)
            self.assertEqual(nt.dtype, dtype)
            self.assertEqual(nt_empty_other_dtype.dtype, other_dtype)
            self.assertEqual(nt.device, nt_empty.device)
            self.assertEqual(nt.layout, nt_empty.layout)

        # Create tensor for autograd
        nt_empty_req_grad = torch.empty_like(nt, requires_grad=True)
        self.assertEqual(nt_empty_req_grad.requires_grad, True)

        # Test noncontiguous tensor does not fail to copy
        nt_cont, nt_noncont = random_nt_noncontiguous_pair((2, 3, 6, 7))
        nt_empty = torch.empty_like(nt_cont)
        assert nt_cont.is_same_size(nt_empty)
        nt_empty_non_contig = torch.empty_like(nt_noncont)
        assert nt_noncont.is_same_size(nt_empty_non_contig)

        # Test the contiguous memory format option
        nt_empty_contig = torch.empty_like(
            nt_cont, memory_format=torch.contiguous_format
        )
        assert nt_cont.is_same_size(nt_empty_contig)
        assert nt_empty_contig.is_contiguous()

        nt_empty_non_contig = torch.empty_like(
            nt_noncont, memory_format=torch.contiguous_format
        )
        assert nt_noncont.is_same_size(nt_empty_non_contig)
        assert nt_empty_non_contig.is_contiguous()

        # Test other memory formats fail
        self.assertRaises(
            RuntimeError,
            lambda: torch.empty_like(nt_cont, memory_format=torch.channels_last),
        )
        self.assertRaises(
            RuntimeError,
            lambda: torch.empty_like(nt_noncont, memory_format=torch.channels_last),
        )
        self.assertRaises(
            RuntimeError,
            lambda: torch.empty_like(nt_cont, memory_format=torch.channels_last_3d),
        )
        self.assertRaises(
            RuntimeError,
            lambda: torch.empty_like(nt_noncont, memory_format=torch.channels_last_3d),
        )


@markDynamoStrictTest
class TestNestedTensorAutograd(NestedTensorTestCase):
    # Note [Gradcheck args check_batched_grad=False] the common_utils testing version of gradcheck
    # includes the default parameters used for testing ops with gradcheck. However nested tensor
    # does not support the stack op therefore we turn it off for these tests
    def _create_leaf_nested_tensor_from_list(self, tensor_device, requires_grad=False):
        return torch.nested.nested_tensor(
            [torch.randn(1, 2), torch.randn(7, 8)],
            requires_grad=requires_grad,
            device=tensor_device,
        )

    def _create_nested_tensor_from_list(self, tensor_device, requires_grad=False):
        return torch.nested.as_nested_tensor(
            [
                torch.randn(1, 2, requires_grad=requires_grad),
                torch.randn(7, 8, requires_grad=requires_grad),
            ],
            device=tensor_device,
        )

    def _create_nested_tensor_from_mask(self, tensor_device, requires_grad=False):
        data = torch.randn(2, 3, 4, requires_grad=requires_grad, device=tensor_device)
        mask = torch.ones_like(data[:, :, 0]).bool()
        return torch._nested_tensor_from_mask(data, mask)

    def test_as_nested_tensor_propagates_gradients(self, device):
        a = torch.arange(3, dtype=torch.float, device=device)
        b = torch.arange(5, dtype=torch.float, device=device)
        nt = torch.nested.as_nested_tensor([a, b])
        # tensors with requires_grad=False are leaves
        self.assertTrue(nt.is_leaf)
        self.assertTrue(not nt.requires_grad)

        a = torch.arange(3, dtype=torch.float, requires_grad=True, device=device)
        b = torch.arange(5, dtype=torch.float, requires_grad=True, device=device)
        nt2 = torch.nested.as_nested_tensor([a, b])
        fake_grad = torch.nested.nested_tensor(
            [torch.ones_like(a), torch.zeros_like(b)], device=device
        )
        nt2.backward(fake_grad)
        self.assertEqual(a.grad, fake_grad[0])
        self.assertEqual(b.grad, fake_grad[1])

    def test_nested_tensor_generates_leaf(self, device):
        a = torch.arange(3, dtype=torch.float, requires_grad=True, device=device)
        b = torch.arange(5, dtype=torch.float, requires_grad=True, device=device)

        nt = torch.nested.nested_tensor([a, b], requires_grad=False)
        self.assertTrue(nt.is_leaf)
        self.assertTrue(not nt.requires_grad)

        nt2 = torch.nested.nested_tensor([a, b], requires_grad=True)
        self.assertTrue(nt2.is_leaf)
        self.assertTrue(nt2.requires_grad)

        fake_grad = torch.nested.nested_tensor(
            [torch.ones_like(a), torch.zeros_like(b)], device=device
        )
        nt2.backward(fake_grad)
        self.assertEqual(nt2.grad, fake_grad)
        self.assertEqual(a.grad, None)
        self.assertEqual(b.grad, None)

    def test_set_requires_grad_from_list(self, device):
        nt = self._create_nested_tensor_from_list(device)
        nt.requires_grad_()
        assert nt.requires_grad

    def test_set_requires_grad_from_mask(self, device):
        nt = self._create_nested_tensor_from_mask(device)
        nt.requires_grad_()
        assert nt.requires_grad

    def test_backward_for_add_op(self, device):
        nt_1 = self._create_nested_tensor_from_mask(device)
        nt_2 = self._create_nested_tensor_from_mask(device)

        nt_1.requires_grad_()
        c = nt_1 + nt_2

        assert nt_1.requires_grad
        assert c.requires_grad
        grad_output = self._create_nested_tensor_from_mask(device)
        c.backward(grad_output)

        #  Grad check doesn't work with nested yet.
        # d/dnt_1 (nt + nt_1) = 1*grad_output
        self.assertEqual(nt_1.grad, grad_output)

    def test_backward_for_sub_op(self, device):
        nt_1 = self._create_nested_tensor_from_mask(device)
        nt_2 = self._create_nested_tensor_from_mask(device)

        nt_1.requires_grad_()
        nt_2.requires_grad_()
        c = nt_1 - nt_2

        assert nt_1.requires_grad
        assert nt_2.requires_grad
        assert c.requires_grad
        grad_output = self._create_nested_tensor_from_mask(device)
        c.backward(grad_output)

        self.assertEqual(nt_1.grad, grad_output)
        self.assertEqual(nt_2.grad, -1 * grad_output)

    def test_backward_sub_strided(self, device):
        a = torch.nested.nested_tensor(
            [torch.randn(9, 2, 4), torch.randn(12, 2, 4)],
            requires_grad=True,
            device=device,
        )
        b = torch.nested.nested_tensor(
            [torch.randn(9, 4, 2), torch.randn(12, 4, 2)],
            requires_grad=True,
            device=device,
        )
        c = a - b.transpose(-1, -2)
        grad_output = c.clone()
        c.backward(grad_output)
        self.assertEqual(a.grad, grad_output)
        self.assertEqual(b.grad, -1 * grad_output.transpose(-1, -2))

    def test_backward_add_strided(self, device):
        a = torch.nested.nested_tensor(
            [torch.randn(9, 2, 4), torch.randn(12, 2, 4)],
            requires_grad=True,
            device=device,
        )
        b = torch.nested.nested_tensor(
            [torch.randn(9, 4, 2), torch.randn(12, 4, 2)],
            requires_grad=True,
            device=device,
        )
        c = a + b.transpose(-1, -2)
        grad_output = c.clone()
        c.backward(grad_output)
        self.assertEqual(a.grad, grad_output)
        self.assertEqual(b.grad, grad_output.transpose(-1, -2))

    # Test Factory Functions
    def test_nested_tensor_to_padded_tensor(self, device):
        for padding_val in [0, 1]:
            nt = self._create_leaf_nested_tensor_from_list(
                tensor_device=device, requires_grad=True
            )

            out = torch.nested.to_padded_tensor(nt, padding_val)
            grad_output = torch.ones(out.shape, device=device)
            out.backward(grad_output)

            self.assertEqual(
                nt.grad,
                torch.nested.nested_tensor(
                    [torch.ones(1, 2), torch.ones(7, 8)], device=device
                ),
            )

    def test_nested_tensor_from_mask_and_to_padded(self, device):
        N, L, D = 2, 4, 4
        mask = torch.ones(N, L, device=device)
        for i in range(1, N):
            end = torch.randint(1, L - 1, (1,), device=device)
            mask[i, end:] = 0

        mask[0, :] = 1
        mask = mask.bool()

        data = torch.randn(
            N, L, D, requires_grad=True, dtype=torch.float64, device=device
        )

        def grad_test_func(inpt):
            nt = torch._nested_tensor_from_mask(inpt, mask)
            # This implicitly tests to_padded_tensor grads
            return torch.nested.to_padded_tensor(nt, 0)

        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

    def test_nested_tensor_from_padded(self, device):
        nested_size = torch.tensor([[1, 2], [2, 2]])
        padded_tensor = torch.randn(2, 2, 2, dtype=torch.float64, device=device)
        padded_tensor[0, 1, :] = 0
        padded_tensor.requires_grad_()

        def grad_test_func(tensor, nested_size):
            nt = torch._nested_from_padded(
                tensor, nested_size, fuse_transform_0213=False
            )
            # This implicitly tests to_padded_tensor grads
            return torch.nested.to_padded_tensor(nt, 0)

        data = (padded_tensor, nested_size)
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

    def test_nested_tensor_from_padded_fused(self, device):
        nested_size = torch.tensor([[1, 8], [2, 8]])
        padded_tensor = torch.randn(2, 2, 2, 4, dtype=torch.float64, device=device)
        padded_tensor[0, 1, :] = 0
        padded_tensor.requires_grad_()

        def grad_test_func(tensor, nested_size):
            nt = torch._nested_from_padded(
                tensor, nested_size, fuse_transform_0213=True
            )
            # This implicitly tests to_padded_tensor grads
            return torch.nested.to_padded_tensor(nt, 0)

        data = (padded_tensor, nested_size)
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

    def test_nested_tensor_from_list(self, device):
        a = torch.randn(1, 2, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(2, 2, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(10, 2, requires_grad=True, dtype=torch.float64, device=device)

        def grad_test_func(a, b, c):
            c = torch.nested.as_nested_tensor([a, b, c])
            # This implictily tests to_padded_tensor grads
            return torch.nested.to_padded_tensor(c, 0)

        data = (a, b, c)
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

    @parametrize("layout", [torch.strided, torch.jagged], name_fn=layout_name)
    def test_dropout_backward(self, layout):
        if layout == torch.jagged:
            nt = torch.nested.nested_tensor(
                [torch.randn((2, 5)), torch.randn((3, 5))],
                requires_grad=True,
                layout=layout,
            )
        else:
            nt = torch.nested.nested_tensor(
                [torch.randn((2, 5)), torch.randn((3, 4))],
                requires_grad=True,
                layout=layout,
            )
        p = 0.2
        y = torch.nn.functional.dropout(nt, p)
        y.backward(nt.detach().clone())
        self.assertEqual(nt.grad, y)

    def test_nested_tensor_bmm_gradcheck(self, device):
        a = torch.randn(2, 6, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(3, 6, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(6, 4, requires_grad=True, dtype=torch.float64, device=device)
        d = torch.randn(6, 5, requires_grad=True, dtype=torch.float64, device=device)

        def grad_test_func(a, b, c, d):
            nt0 = torch.nested.as_nested_tensor([a, b])
            nt1 = torch.nested.as_nested_tensor([c, d])
            result = nt0.bmm(nt1)
            return torch.nested.to_padded_tensor(result, 0.0)

        data = (a, b, c, d)
        assert torch.autograd.gradcheck(grad_test_func, inputs=data)

    def test_nested_tensor_bmm_backward(self, device):
        nt0 = torch.nested.nested_tensor(
            [torch.randn((2, 6)), torch.randn((3, 6))],
            requires_grad=True,
            device=device,
        )
        nt1 = torch.nested.nested_tensor(
            [torch.randn((6, 4)), torch.randn((6, 5))],
            requires_grad=True,
            device=device,
        )
        with torch.no_grad():
            pt0 = torch.nested.to_padded_tensor(nt0, 0.0).requires_grad_(True)
            pt1 = torch.nested.to_padded_tensor(nt1, 0.0).requires_grad_(True)

        ynt = nt0.bmm(nt1)
        ypt = pt0.bmm(pt1)
        ynt.backward(ynt.clone())
        ypt.backward(ypt.clone())

        self.assertEqual(torch.nested.to_padded_tensor(nt0.grad, 0.0), pt0.grad)
        self.assertEqual(torch.nested.to_padded_tensor(nt1.grad, 0.0), pt1.grad)

    def test_nested_tensor_matmul_gradcheck(self, device):
        a = torch.randn(2, 6, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(3, 6, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(6, 4, requires_grad=True, dtype=torch.float64, device=device)
        d = torch.randn(6, 5, requires_grad=True, dtype=torch.float64, device=device)

        def grad_test_func(a, b, c, d):
            nt0 = torch.nested.as_nested_tensor([a, b])
            nt1 = torch.nested.as_nested_tensor([c, d])
            result = torch.matmul(nt0, nt1)
            return torch.nested.to_padded_tensor(result, 0.0)

        data = (a, b, c, d)
        assert torch.autograd.gradcheck(grad_test_func, inputs=data)

    def test_nested_tensor_matmul_backward(self, device):
        nt0 = torch.nested.nested_tensor(
            [torch.randn((7, 2, 6)), torch.randn((7, 3, 6))],
            requires_grad=True,
            device=device,
        )
        nt1 = torch.nested.nested_tensor(
            [torch.randn((7, 6, 4)), torch.randn((7, 6, 5))],
            requires_grad=True,
            device=device,
        )
        with torch.no_grad():
            pt0 = torch.nested.to_padded_tensor(nt0, 0.0).requires_grad_(True)
            pt1 = torch.nested.to_padded_tensor(nt1, 0.0).requires_grad_(True)

        ynt = torch.matmul(nt0, nt1)
        ypt = torch.matmul(pt0, pt1)
        ynt.backward(ynt.clone())
        ypt.backward(ypt.clone())

        self.assertEqual(torch.nested.to_padded_tensor(nt0.grad, 0.0), pt0.grad)
        self.assertEqual(torch.nested.to_padded_tensor(nt1.grad, 0.0), pt1.grad)

    def test_nested_tensor_transpose_gradcheck(self, device):
        a = torch.randn(2, 5, requires_grad=True, device=device)
        b = torch.randn(3, 4, requires_grad=True, device=device)

        def grad_test_func(a, b):
            nt = torch.nested.as_nested_tensor([a, b])
            result = nt.transpose(-2, -1).transpose(-2, -1)
            return torch.nested.to_padded_tensor(result, 0.0)

        data = (a, b)
        assert torch.autograd.gradcheck(grad_test_func, inputs=data, eps=1e-3)

    def test_nested_tensor_transpose_backward(self, device):
        nt = torch.nested.nested_tensor(
            [torch.randn((2, 5)), torch.randn((3, 4))],
            requires_grad=True,
            device=device,
        )
        with torch.no_grad():
            pt = torch.nested.to_padded_tensor(nt, 0.0).requires_grad_(True)

        ynt = nt.transpose(-2, -1)
        ypt = pt.transpose(-2, -1)
        ynt.backward(ynt.clone())
        ypt.backward(ypt.clone())

        self.assertEqual(torch.nested.to_padded_tensor(nt.grad, 0.0), pt.grad)

    def test_nested_tensor_reshape_gradcheck(self, device):
        a = torch.randn(2, 6, requires_grad=True, device=device)
        b = torch.randn(3, 6, requires_grad=True, device=device)

        def grad_test_func(a, b):
            nt = torch.nested.as_nested_tensor([a, b])
            result = nt.reshape(2, -1, 2, 3)
            return torch.nested.to_padded_tensor(result, 0.0)

        data = (a, b)
        assert torch.autograd.gradcheck(grad_test_func, inputs=data, eps=1e-3)

    def test_nested_tensor_reshape_backward(self):
        nt = torch.nested.nested_tensor(
            [torch.randn((2, 6)), torch.randn((3, 6))], requires_grad=True
        )
        with torch.no_grad():
            pt = torch.nested.to_padded_tensor(nt, 0.0).requires_grad_(True)

        ynt = nt.reshape(2, -1, 2, 3)
        ypt = pt.reshape(2, -1, 2, 3)
        ynt.backward(ynt.clone())
        ypt.backward(ypt.clone())

        self.assertEqual(torch.nested.to_padded_tensor(nt.grad, 0.0), pt.grad)

    def test_nested_tensor_squeeze_backward(self, device):
        nt = torch.nested.nested_tensor(
            [torch.randn((2, 6, 1)), torch.randn((3, 6, 1))],
            requires_grad=True,
            device=device,
        )
        with torch.no_grad():
            pt = torch.nested.to_padded_tensor(nt, 0.0).requires_grad_(True)

        ynt = nt.squeeze(-1)
        ypt = pt.squeeze(-1)
        ynt.backward(ynt.clone())
        ypt.backward(ypt.clone())

        self.assertEqual(torch.nested.to_padded_tensor(nt.grad, 0.0), pt.grad)

    def test_nested_tensor_squeeze_gradcheck(self, device):
        a = torch.randn(
            (2, 6, 1), dtype=torch.float64, requires_grad=True, device=device
        )
        b = torch.randn(
            (3, 6, 1), dtype=torch.float64, requires_grad=True, device=device
        )

        def grad_test_func(a, b):
            nt = torch.nested.as_nested_tensor([a, b])
            result = nt.squeeze(-1)
            return torch.nested.to_padded_tensor(result, 0.0)

        assert torch.autograd.gradcheck(grad_test_func, inputs=(a, b), eps=1e-3)

    def test_nested_tensor_unsqueeze_backward(self, device):
        nt = torch.nested.nested_tensor(
            [torch.randn((2, 6)), torch.randn((3, 6))],
            requires_grad=True,
            device=device,
        )
        with torch.no_grad():
            pt = torch.nested.to_padded_tensor(nt, 0.0).requires_grad_(True)

        ynt = nt.unsqueeze(2)
        ypt = pt.unsqueeze(2)
        ynt.backward(ynt.clone())
        ypt.backward(ypt.clone())

        self.assertEqual(torch.nested.to_padded_tensor(nt.grad, 0.0), pt.grad)

    def test_nested_tensor_unsqueeze_gradcheck(self, device):
        a = torch.randn((2, 6), dtype=torch.float64, requires_grad=True, device=device)
        b = torch.randn((3, 6), dtype=torch.float64, requires_grad=True, device=device)

        def grad_test_func(a, b):
            nt = torch.nested.as_nested_tensor([a, b])
            result = nt.unsqueeze(-1)
            return torch.nested.to_padded_tensor(result, 0.0)

        assert torch.autograd.gradcheck(grad_test_func, inputs=(a, b), eps=1e-3)

    def test_nested_tensor_linear(self, device):
        a = torch.randn(1, 2, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(2, 2, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(3, 2, requires_grad=True, dtype=torch.float64, device=device)

        weight = torch.randn(
            2, 2, requires_grad=True, dtype=torch.float64, device=device
        )
        bias = torch.randn(2, requires_grad=True, dtype=torch.float64, device=device)

        def grad_test_func(a, b, c, weight, bias=None):
            nt = torch.nested.as_nested_tensor([a, b, c])
            # This implicitly tests to_padded_tensor grads
            d = torch.functional.F.linear(nt, weight, bias)
            return torch.nested.to_padded_tensor(d, 0)

        data = (a, b, c, weight, bias)
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

        # Test linear with no bias added
        data = (a, b, c, weight)
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

    def test_nested_tensor_linear_plus_transpose(self, device):
        a = torch.randn(1, 2, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(2, 2, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(3, 2, requires_grad=True, dtype=torch.float64, device=device)

        weight = torch.randn(
            2, 2, requires_grad=True, dtype=torch.float64, device=device
        )
        bias = torch.randn(2, requires_grad=True, dtype=torch.float64, device=device)

        def grad_test_func(a, b, c, weight, bias=None):
            nt = torch.nested.as_nested_tensor([a, b, c])
            # This implicitly tests to_padded_tensor grads
            d = torch.functional.F.linear(nt, weight, bias)
            d = d.transpose(-1, -2).contiguous()
            return torch.nested.to_padded_tensor(d, 0)

        data = (a, b, c, weight, bias)
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

        # Test linear with no bias added
        data = (a, b, c, weight)
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

    def test_nested_tensor_softmax(self, device):
        a = torch.randn(1, 2, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(2, 2, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(3, 2, requires_grad=True, dtype=torch.float64, device=device)

        def grad_test_func(a, b, c, dim):
            nt = torch.nested.as_nested_tensor([a, b, c])
            # This implicitly tests to_padded_tensor grads
            d = torch.functional.F.softmax(nt, dim=dim)
            return torch.nested.to_padded_tensor(d, 0)

        # softmax over last dim
        data = (a, b, c, -1)
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

    def test_nested_tensor_linear_backward(self, device):
        a = torch.randn(1, 2, requires_grad=False, device=device)
        b = torch.randn(2, 2, requires_grad=False, device=device)
        c = torch.randn(3, 2, requires_grad=False, device=device)

        weight = torch.randn(2, 2, requires_grad=True, device=device)
        bias = torch.randn(2, requires_grad=True, device=device)
        nt = torch.nested.as_nested_tensor([a, b, c], device=device)

        out = torch.functional.F.linear(nt, weight, bias)

        out.backward(out.clone())

        assert weight.grad is not None
        assert bias.grad is not None

        assert a.grad is None
        assert b.grad is None
        assert c.grad is None

    def test_values_grad_with_broadcast(self, device):
        a = torch.randn(1, 2, 4, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(2, 2, 4, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(3, 2, 4, requires_grad=True, dtype=torch.float64, device=device)

        def grad_test_func(a, b, c):
            nt = torch.nested.as_nested_tensor([a, b, c])
            buffer = nt.values()
            return buffer.sum()

        data = (a, b, c)
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

    def test_to_buffer_series_ops_grad_with_broadcast(self, device):
        a = torch.randn(1, 1, 2, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(1, 1, 2, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(1, 1, 2, requires_grad=True, dtype=torch.float64, device=device)

        def grad_test_func(a, b, c):
            nt = torch.nested.as_nested_tensor([a, b, c])
            buffer = nt.values()
            buffer = buffer * 2
            return buffer.exp()

        data = (a, b, c)
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

    def test_unbind_flow_through(self, device):
        a = torch.randn(1, 2, 4, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(2, 2, 4, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(3, 2, 4, requires_grad=True, dtype=torch.float64, device=device)

        def grad_test_func(a, b, c):
            nt = torch.nested.as_nested_tensor([a, b, c])
            ntT = nt.transpose(-1, -2)
            unbound = ntT.unbind()
            d = unbound[0]
            d = torch.pow(d, 2)
            return d

        data = (a, b, c)
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

    def test_split_with_sizes_flow_through(self, device):
        a = torch.randn(2, 5, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(3, 5, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(4, 5, requires_grad=True, dtype=torch.float64, device=device)

        def grad_test_func(a, b, c):
            nt = torch.nested.as_nested_tensor([a, b, c])
            splits = nt.split_with_sizes([2, 3], dim=-1)
            unbound = splits[1].unbind()
            d = unbound[0]
            d = torch.pow(d, 2)
            return d

        data = (a, b, c)
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

    def test_indexing_backward(self, device):
        x0 = torch.randn((2, 5))
        x1 = torch.randn((3, 4))
        nt = torch.nested.nested_tensor([x0, x1], device=device, requires_grad=True)
        self.assertEqual(nt[0], x0)
        self.assertEqual(nt[-1], x1)
        grad_x0 = torch.randn((2, 5), device=device)
        nt[0].backward(grad_x0)
        expected_grad = torch.nested.nested_tensor(
            [grad_x0, torch.zeros((3, 4), device=device)]
        )
        self.assertEqual(nt.grad, expected_grad)

    def test_masked_fill_backward(self, device):
        a = torch.randn(1, 2, 4, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(2, 2, 4, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(3, 2, 4, requires_grad=True, dtype=torch.float64, device=device)

        def grad_test_func(a, b, c):
            nt = torch.nested.as_nested_tensor([a, b, c])
            mask = nt.detach().clone().to(bool)
            out = nt.masked_fill(mask, 0)
            out = torch.nested.to_padded_tensor(out, 0)
            return out

        data = (a, b, c)
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

    def test_gelu_backward(self, device):
        a = torch.randn(1, 2, 4, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(2, 2, 4, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(3, 2, 4, requires_grad=True, dtype=torch.float64, device=device)

        def grad_test_func(a, b, c):
            nt = torch.nested.as_nested_tensor([a, b, c])
            nt_gelu = torch.nn.functional.gelu(nt)
            return torch.nested.to_padded_tensor(nt_gelu, 0)

        data = (a, b, c)
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

    def test_relu_backward(self, device):
        a = torch.randn(1, 2, 4, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(2, 2, 4, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(3, 2, 4, requires_grad=True, dtype=torch.float64, device=device)

        def grad_test_func(a, b, c):
            nt = torch.nested.as_nested_tensor([a, b, c])
            nt_relu = torch.nn.functional.relu(nt)
            return torch.nested.to_padded_tensor(nt_relu, 0)

        data = (a, b, c)
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

    def test_selu_backward(self, device):
        a = torch.randn(1, 2, 4, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(2, 2, 4, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(3, 2, 4, requires_grad=True, dtype=torch.float64, device=device)

        def grad_test_func(a, b, c):
            nt = torch.nested.as_nested_tensor([a, b, c])
            nt_relu = torch.nn.functional.silu(nt)
            return torch.nested.to_padded_tensor(nt_relu, 0)

        data = (a, b, c)
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

    def test_abs_backward(self, device):
        a = torch.randn(1, 2, 4, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(2, 2, 4, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(3, 2, 4, requires_grad=True, dtype=torch.float64, device=device)

        def grad_test_func(a, b, c):
            nt = torch.nested.as_nested_tensor([a, b, c])
            nt_abs = torch.abs(nt)
            return torch.nested.to_padded_tensor(nt_abs, 0)

        data = (a, b, c)
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

    # Previously would error when input NT doesn't require grad
    # NotImplementedError: Cannot access storage of UndefinedTensorImpl
    def test_layer_norm_backward_edge_case(self, device):
        size = 4
        a = torch.randn(
            1, 2, size, requires_grad=False, dtype=torch.float64, device=device
        )
        nt = torch.nested.nested_tensor([a])
        nt_layer_norm = torch.nn.LayerNorm(
            nt.size(-1), device=device, dtype=torch.float64
        )
        out = nt_layer_norm(nt)
        out.backward(out.clone())

    def test_accumulate_grad_different_strides(self, device):
        a = torch.rand(1, 4, 2, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.rand(1, 8, 2, requires_grad=True, dtype=torch.float64, device=device)

        def grad_test_func(a, b):
            nt_1 = torch.nested.as_nested_tensor([a, b])
            nt_2 = nt_1.clone()
            out = torch.nn.functional.scaled_dot_product_attention(nt_1, nt_2, nt_2)
            return torch.nested.to_padded_tensor(out, 0)

        data = (a, b)
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

    # https://github.com/pytorch/pytorch/issues/95562
    @skipIfSlowGradcheckEnv
    @parametrize("size", [1024, 1023, 513, 512, 256, 128, 32, 4, 2])
    def test_layer_norm_backward(self, device, size):
        a = torch.randn(
            1, 2, size, requires_grad=True, dtype=torch.float64, device=device
        )
        b = torch.randn(
            2, 2, size, requires_grad=True, dtype=torch.float64, device=device
        )
        c = torch.randn(
            3, 2, size, requires_grad=True, dtype=torch.float64, device=device
        )

        def grad_test_func(a, b, c):
            nt = torch.nested.as_nested_tensor([a, b, c])
            layer_norm = torch.nn.LayerNorm(
                nt.size(-1), device=device, dtype=torch.float64
            )
            nt_layer_norm = layer_norm(nt)
            return torch.nested.to_padded_tensor(nt_layer_norm, 0)

        data = (a, b, c)
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

    # https://github.com/pytorch/pytorch/issues/95562
    @skipIfSlowGradcheckEnv
    # Could either mark slow or reduce size
    @parametrize("size", [128, 32, 4, 2])
    def test_layer_norm_backward_5d(self, device, size):
        a = torch.randn(
            4, size, size, 4, requires_grad=True, dtype=torch.float64, device=device
        )
        b = torch.randn(
            7, size, size, 4, requires_grad=True, dtype=torch.float64, device=device
        )
        c = torch.randn(
            10, size, size, 4, requires_grad=True, dtype=torch.float64, device=device
        )

        def grad_test_func(a, b, c):
            nt = torch.nested.as_nested_tensor([a, b, c])
            layer_norm = torch.nn.LayerNorm(
                (size, size, nt.size(-1)), device=device, dtype=torch.float64
            )
            nt_layer_norm = layer_norm(nt)
            return torch.nested.to_padded_tensor(nt_layer_norm, 0)

        data = (a, b, c)
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)


# Found in torch/testing/_comparison.py
default_atol = {torch.float16: 1e-3, torch.bfloat16: 1e-3, torch.float32: 1e-5}
default_rtol = {torch.float16: 1e-3, torch.bfloat16: 1.6e-2, torch.float32: 1.3e-6}


def get_rtol(true_value: torch.Tensor, computed_value: torch.Tensor) -> float:
    deviation = true_value - computed_value
    deviation = torch.abs(deviation / true_value)
    # Fill in the nans with the default rtol
    torch.nan_to_num_(deviation, nan=default_rtol[computed_value.dtype])
    return deviation.max().item()


def get_atol(true_value: torch.Tensor, computed_value: torch.Tensor) -> float:
    deviation = true_value - computed_value
    atol = torch.abs(deviation).max().item()
    return atol


def get_tolerances(
    true_value: torch.Tensor,
    computed_value: torch.Tensor,
    fudge_factor: Optional[float] = None,
) -> Tuple[float, float]:
    """Returns the absolute and relative tolerances for comparing two tensors."""
    fudge_factor = fudge_factor if fudge_factor is not None else 1.0
    atol = get_atol(true_value, computed_value)
    rtol = get_rtol(true_value, computed_value)

    atol = fudge_factor * max(atol, default_atol[computed_value.dtype])
    rtol = fudge_factor * max(rtol, default_rtol[computed_value.dtype])
    # torch.isclose() has weird behavior around see:
    # https://github.com/pytorch/pytorch/issues/102400
    if rtol > 1e30:
        rtol = default_rtol[computed_value.dtype]
    return atol, rtol


# We can probably parametrizing existing tests instead of having a separate
# test class as we begin to support more ops. Also maybe rewrite with OpInfos.
@markDynamoStrictTest
class TestNestedTensorSubclass(NestedTensorTestCase):
    # TODO: consolidate with the below
    def _get_list_for_jagged_tensor(self, nested_size, device, requires_grad=True):
        Ds = nested_size[1:]
        out = []
        for s in nested_size[0]:
            out.append(
                torch.randn(
                    s,
                    *Ds,
                    requires_grad=requires_grad,
                    device=device,
                    dtype=torch.float64,
                )
            )
        return out

    def _get_example_tensor_lists(
        self,
        include_list_of_lists=True,
        include_requires_grad=True,
        include_inner_dim_size_1=False,
        include_2d_tensor=False,
    ):
        def _make_tensor(
            *shape, include_requires_grad=include_requires_grad, requires_grad=True
        ):
            return torch.randn(
                *shape,
                requires_grad=(requires_grad if include_requires_grad else False),
            )

        # Purposefully introduce mixed requires_grad settings for the components
        # when include_requires_grad=True.
        example_lists = [
            # (B, *, D) with B=4
            [
                _make_tensor(2, 5),
                _make_tensor(3, 5, requires_grad=False),
                _make_tensor(4, 5, requires_grad=False),
                _make_tensor(6, 5),
            ],
            # (B, *, D_0, D_1) with B=5
            [
                _make_tensor(2, 5, 6),
                _make_tensor(3, 5, 6),
                _make_tensor(4, 5, 6, requires_grad=False),
                _make_tensor(5, 5, 6),
                _make_tensor(6, 5, 6),
            ],
            # (B, *, D_0, D_1, D_2) with B=6
            [
                _make_tensor(2, 5, 6, 7),
                _make_tensor(3, 5, 6, 7),
                _make_tensor(4, 5, 6, 7, requires_grad=False),
                _make_tensor(5, 5, 6, 7),
                _make_tensor(6, 5, 6, 7),
                _make_tensor(7, 5, 6, 7),
            ],
        ]

        if include_list_of_lists:
            example_lists.append(
                # (B, *, D) with B=3 in list form
                [
                    _make_tensor(2, 5, requires_grad=False).tolist(),
                    _make_tensor(3, 5).tolist(),
                    _make_tensor(4, 5).tolist(),
                ]
            )

        if include_inner_dim_size_1:
            example_lists.append(
                [
                    _make_tensor(2, 1),
                    _make_tensor(3, 1, requires_grad=False),
                    _make_tensor(4, 1, requires_grad=False),
                    _make_tensor(6, 1),
                ]  # (B, *, 1)
            )
            example_lists.append(
                [
                    _make_tensor(2, 5, 1),
                    _make_tensor(3, 5, 1, requires_grad=False),
                    _make_tensor(4, 5, 1, requires_grad=False),
                    _make_tensor(6, 5, 1),
                ]  # (B, *, 5, 1)
            )

        if include_2d_tensor:
            example_lists.append(
                [
                    _make_tensor(2),
                    _make_tensor(3, requires_grad=False),
                    _make_tensor(4, requires_grad=False),
                    _make_tensor(6),
                ]  # (B, *)
            )

        return example_lists

    @dtypes(torch.float32)
    @parametrize(
        "contiguity",
        ["contig", "noncontig_transposed", "noncontig_with_holes"],
        name_fn=lambda c: c,
    )
    @parametrize("weights_only", [True, False])
    def test_serialization(self, device, dtype, contiguity, weights_only):
        # Test with 3 cases:
        # 1. contiguous
        # 2. non-contiguous transposed
        # 3. non-contiguous with holes
        if contiguity == "contig":
            nt = random_nt_from_dims(
                [4, None, 10],
                device=device,
                dtype=dtype,
                layout=torch.jagged,
            )
        elif contiguity == "noncontig_transposed":
            nt = random_nt_from_dims(
                [3, None, 5, 2],
                device=device,
                dtype=dtype,
                layout=torch.jagged,
            ).transpose(-3, -2)
        elif contiguity == "noncontig_with_holes":
            nt = torch.nested.nested_tensor_from_jagged(
                values=torch.randn(10, 3, device=device, dtype=dtype),
                offsets=torch.tensor([0, 3, 7, 10], device=device, dtype=torch.int64),
                # these lengths specify holes
                lengths=torch.tensor([1, 2, 3], device=device, dtype=torch.int64),
            )
        else:
            raise ValueError("invalid contiguity specified for test_serialization()")

        # Access sizes / strides to ensure cache doesn't break serialization.
        # See https://github.com/pytorch/pytorch/issues/129366
        nt.size()
        nt.stride()

        with tempfile.TemporaryFile() as f:
            torch.save(nt, f)
            safe_globals = [
                torch.nested._internal.nested_tensor.NestedTensor,
                torch.nested._internal.nested_tensor._rebuild_njt,
                set,
                torch._dynamo.decorators._DimRange,
            ]
            f.seek(0)
            ctx = (
                torch.serialization.safe_globals(safe_globals)
                if weights_only
                else contextlib.nullcontext()
            )

            with ctx:
                nt_loaded = torch.load(f, weights_only=weights_only)

            self.assertIsNot(nt, nt_loaded)
            # we expect a new offsets tensor -> different nested int upon load
            self.assertEqualIgnoringNestedInts(nt, nt_loaded)
            self.assertEqual(nt._ragged_idx, nt_loaded._ragged_idx)
            # ensure shapes are equal except nested int
            nt_rest_of_shape = (
                *nt.shape[: nt._ragged_idx],
                *nt.shape[nt._ragged_idx + 1 :],
            )
            nt_loaded_rest_of_shape = (
                *nt_loaded.shape[: nt_loaded._ragged_idx],
                *nt_loaded.shape[nt_loaded._ragged_idx + 1 :],
            )
            self.assertEqual(nt_rest_of_shape, nt_loaded_rest_of_shape)
            # ensure metadata cache is carried through serialization
            self.assertEqual(nt._metadata_cache, nt_loaded._metadata_cache)
            # ensure lengths are carried through if present
            self.assertEqual(nt._lengths, nt_loaded._lengths)

    def test_tensor_attributes(self, device):
        a = torch.randn(2, 3, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(3, 3, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(4, 3, requires_grad=True, dtype=torch.float64, device=device)
        nt = torch.nested.as_nested_tensor([a, b, c], layout=torch.jagged)
        _offsets = nt.offsets()

        for op in (
            torch.ops.aten.is_non_overlapping_and_dense.default,
            torch.ops.aten.sym_size.default,
            torch.ops.aten.dim.default,
            torch.ops.aten.numel.default,
            torch.ops.aten.sym_numel.default,
            torch.ops.aten.sym_stride.default,
            torch.ops.aten.sym_storage_offset.default,
        ):
            op(nt)

        with self.assertRaisesRegex(
            RuntimeError, "directly calling torch.ops.aten.size"
        ):
            torch.ops.aten.size.default(nt)

        nested_int = torch.nested._internal.nested_tensor.get_tensor_symint(
            _offsets, coeff=1
        )
        self.assertEqual(nt.size(), (3, nested_int, 3))
        self.assertEqual(nt.shape, (3, nested_int, 3))
        self.assertEqual(nt.dim(), 3)
        self.assertEqual(nt.numel(), 27)

    @parametrize("nt_dim", [3, 4, 5])
    def test_linear(self, device, nt_dim):
        if nt_dim == 3:
            fixed_shape = (3,)
        elif nt_dim == 4:
            fixed_shape = (4, 3)
        elif nt_dim == 5:
            fixed_shape = (5, 4, 3)

        a = torch.randn(
            2, *fixed_shape, requires_grad=True, dtype=torch.float64, device=device
        )
        b = torch.randn(
            3, *fixed_shape, requires_grad=True, dtype=torch.float64, device=device
        )
        c = torch.randn(
            4, *fixed_shape, requires_grad=True, dtype=torch.float64, device=device
        )
        weight = torch.randn(
            4, 3, requires_grad=True, dtype=torch.float64, device=device
        )
        bias = torch.randn(4, requires_grad=True, dtype=torch.float64, device=device)

        def grad_test_func(a, b, c, weight, bias):
            nt = torch.nested.as_nested_tensor([a, b, c], layout=torch.jagged)
            out = torch.nn.functional.linear(nt, weight, bias)
            return out.values()

        gradcheck(
            grad_test_func, inputs=(a, b, c, weight, bias), check_batched_grad=False
        )

    def test_unary_pointwise(self, device):
        a = torch.randn(2, 3, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(3, 3, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(4, 3, requires_grad=True, dtype=torch.float64, device=device)

        def grad_test_func(a, b, c):
            nt = torch.nested.as_nested_tensor([a, b, c], layout=torch.jagged)
            out = torch.nn.functional.silu(nt.sin().cos())
            return out.values()

        gradcheck(grad_test_func, inputs=(a, b, c), check_batched_grad=False)

    def test_unary_pointwise_transposed_inputs(self, device):
        a, b, c = (
            torch.randn(
                i + 2, 5, requires_grad=True, dtype=torch.float64, device=device
            )
            for i in range(3)
        )

        nt = torch.nested.nested_tensor(
            [a.detach(), b.detach(), c.detach()], layout=torch.jagged
        )
        nt_t = nt.transpose(1, 2)
        self.assertFalse(nt_t.is_contiguous())
        out = torch.nn.functional.silu(nt_t.sin().cos())
        self.assertEqual(
            out.is_contiguous(),
            torch.nn.functional.silu(b.transpose(-1, -2).sin().cos()).is_contiguous(),
        )

        self.assertEqual(nt_t.shape, out.shape)

        a, b, c = (
            torch.randn(
                i + 2, 5, requires_grad=True, dtype=torch.float64, device=device
            )
            for i in range(3)
        )

        def grad_test_func(a, b, c):
            nt = torch.nested.as_nested_tensor([a, b, c], layout=torch.jagged)
            nt_t = nt.transpose(1, 2)
            out = torch.nn.functional.silu(nt_t.sin().cos())
            return out.values()

        gradcheck(grad_test_func, inputs=(a, b, c), check_batched_grad=False)

    def test_binary_pointwise(self, device):
        a = torch.randn(2, 3, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(3, 3, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(4, 3, requires_grad=True, dtype=torch.float64, device=device)

        # Incorrect usage: shape check will fail if the offsets tensor are not
        #                  the same exact tensor object
        nt1 = torch.nested.as_nested_tensor([a, b, c], layout=torch.jagged)
        nt2 = torch.nested.as_nested_tensor([a, b, c], layout=torch.jagged)

        self.assertRaisesRegex(
            RuntimeError,
            "cannot call binary pointwise function .* with inputs of shapes",
            lambda: nt1 * nt2,
        )

        # Correct usage: chain the calls using the same offsets tensor object
        def grad_test_func(a, b, c):
            nt1 = torch.nested.as_nested_tensor([a, b, c], layout=torch.jagged)
            # TODO: Switch to public API that takes in (values, offsets) once it exists
            nt2, offsets = jagged_from_list([a, b, c], nt1.offsets())
            out = nt1 * nt2
            return out.values()

        gradcheck(grad_test_func, inputs=(a, b, c), check_batched_grad=False)

    def test_binary_pointwise_transposed(self, device):
        a, b, c = (
            torch.randn(i + 2, 5, dtype=torch.float64, device=device) for i in range(3)
        )

        nt1, offsets = jagged_from_list([a, b, c], None)
        nt2, offsets = jagged_from_list([a, b, c], offsets)

        nt1_t = nt1.transpose(1, 2)
        nt2_t = nt2.transpose(1, 2)

        # out = nt1_t * nt2_t
        # self.assertFalse(nt1_t.is_contiguous())
        # self.assertEqual(out.is_contiguous(), (b.transpose(-1, -2) * b.transpose(-1, -2)).is_contiguous())
        # self.assertEqual(out.shape, nt1_t.shape)

        self.assertRaisesRegex(
            RuntimeError,
            "cannot call binary pointwise function mul.Tensor with inputs of shapes",
            lambda: nt1 * nt2_t,
        )

        a, b, c = (
            torch.randn(
                i + 2, 5, requires_grad=True, dtype=torch.float64, device=device
            )
            for i in range(3)
        )

        # Correct usage: chain the calls using the same offsets tensor object
        def grad_test_func(a, b, c):
            nt1, offsets = jagged_from_list([a, b, c], None)
            nt2, offsets = jagged_from_list([a, b, c], offsets)
            nt1_t = nt1.transpose(1, 2)
            nt2_t = nt2.transpose(1, 2)
            out = nt1_t * nt2_t
            return out.values()

        gradcheck(grad_test_func, inputs=(a, b, c), check_batched_grad=False)

    def test_binary_pointwise_with_nested_int_second_arg(self, device):
        # See https://github.com/pytorch/pytorch/issues/138496
        nt = random_nt_from_dims(
            [3, None, 5],
            device=device,
            dtype=torch.float32,
            layout=torch.jagged,
        )

        with self.assertRaisesRegex(RuntimeError, "invalid argument"):
            nt * nt.size(1)

        with self.assertRaisesRegex(RuntimeError, "invalid argument"):
            nt + nt.size(1)

    def test_split(self, device):
        a = torch.randn(2, 3, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(3, 3, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(4, 3, requires_grad=True, dtype=torch.float64, device=device)

        nt = torch.nested.as_nested_tensor([a, b, c], layout=torch.jagged)
        out = torch.split(nt, 2, -1)
        self.assertEqual(len(out), 2)
        self.assertEqualIgnoringNestedInts(
            out[0],
            torch.nested.as_nested_tensor(
                [a[:, 0:2], b[:, 0:2], c[:, 0:2]], layout=torch.jagged
            ),
        )
        self.assertEqualIgnoringNestedInts(
            out[1],
            torch.nested.as_nested_tensor(
                [a[:, 2:], b[:, 2:], c[:, 2:]], layout=torch.jagged
            ),
        )

        with self.assertRaisesRegex(
            RuntimeError,
            r"split\(\): not supported for NestedTensor on dim=1",
        ):
            torch.split(nt, 2, 1)

    def test_split_with_sizes(self, device):
        a = torch.randn(2, 3, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(3, 3, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(4, 3, requires_grad=True, dtype=torch.float64, device=device)

        nt = torch.nested.as_nested_tensor([a, b, c], layout=torch.jagged)
        out = torch.split(nt, [1, 2], -1)
        self.assertEqual(len(out), 2)
        self.assertEqualIgnoringNestedInts(
            out[0],
            torch.nested.as_nested_tensor(
                [a[:, 0:1], b[:, 0:1], c[:, 0:1]], layout=torch.jagged
            ),
        )
        self.assertEqualIgnoringNestedInts(
            out[1],
            torch.nested.as_nested_tensor(
                [a[:, 1:], b[:, 1:], c[:, 1:]], layout=torch.jagged
            ),
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"split_with_sizes\(\): not supported for NestedTensor on dim=1",
        ):
            torch.split(nt, [1, 2], 1)

    def test_softmax(self, device):
        nt = random_nt_from_dims(
            [3, None, 5],
            device=device,
            dtype=torch.float32,
            layout=torch.jagged,
            requires_grad=True,
        )

        # operate on dim=2
        output = nt.softmax(dim=2)

        @torch._dynamo.disable
        def _compare_to_ref(nt, output, dim):
            for in_component, out_component in zip(nt.unbind(), output.unbind()):
                self.assertEqual(in_component.softmax(dim=dim), out_component)

        # dim=2 -> dim=1 after unbind
        _compare_to_ref(nt, output, dim=1)

        # operate on dim=-1
        output2 = nt.softmax(dim=-1)
        torch._dynamo.disable(self.assertEqual)(output, output2)
        _compare_to_ref(nt, output2, dim=-1)

        def grad_test_func(a, b):
            nt = torch.nested.as_nested_tensor([a, b], layout=torch.jagged)
            out = nt.softmax(dim=-1)
            return out.values()

        a = torch.rand(4, 5, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.rand(8, 5, requires_grad=True, dtype=torch.float64, device=device)
        gradcheck(grad_test_func, inputs=(a, b), check_batched_grad=False)

    def test_views_inherit_ragged_dim(self, device):
        # view
        nt = random_nt_from_dims(
            [4, None, 8, 10], device=device, dtype=torch.float32, layout=torch.jagged
        )
        # inherit ragged dim via -1
        view = nt.view(4, -1, 80)
        self.assertEqual(nt.shape[1], view.shape[1])
        # inherit batch and ragged dims via -1
        view2 = nt.view(-1, -1, 80)
        self.assertEqual(nt.shape[:2], view2.shape[:2])

        # expand
        nt = random_nt_from_dims(
            [3, None, 1], device=device, dtype=torch.float32, layout=torch.jagged
        )
        # inherit batch and ragged dims via -1
        view = nt.expand(-1, -1, 5)
        self.assertEqual(nt.shape[:2], view.shape[:2])

    def test_view_ragged_idx_not_one(self, device):
        nt = random_nt_from_dims(
            [2, None, 20], device=device, dtype=torch.float32, layout=torch.jagged
        )

        view_transposed = nt.transpose(1, 2).view(2, 20, nt.size(1))
        self.assertEqual((2, 20, nt.size(1)), (view_transposed.size()))
        self.assertEqual(view_transposed._base, nt._base)

    def test_unsafe_view(self, device):
        nt = random_nt_from_dims(
            [4, None, 8, 10], device=device, dtype=torch.float32, layout=torch.jagged
        )
        # basic view
        view1 = torch.ops.aten._unsafe_view(nt, (4, -1, 80))
        self.assertEqual((4, nt.size(1), 80), tuple(view1.size()))
        # _unsafe_view differs from view in that the view information is not tracked
        self.assertTrue(view1._base is None)

        # test an unsafe_view when ragged_idx != 1, currently only supports identity view
        nt_t = nt.transpose(1, 2)
        view2 = torch.ops.aten._unsafe_view(nt_t, (4, 8, nt.size(1), 10))
        self.assertEqual((4, 8, nt.size(1), 10), tuple(view2.size()))
        self.assertTrue(view2._base is None)

    @xfailIfTorchDynamo
    @parametrize("requires_grad", [False, True])
    def test_reshape_decomp(self, device, requires_grad):
        # contiguous NT should result in view.
        nt = (
            random_nt_from_dims(
                [3, None, 10],
                device=device,
                dtype=torch.float32,
                layout=torch.jagged,
            )
            .detach()
            .requires_grad_(requires_grad)
        )
        view = nt.reshape(-1, -1, 5, 2)
        self.assertEqual(view.shape[:2], nt.shape[:2])
        self.assertTrue(view._is_view() and view._base is nt)
        # make sure gradients flow back
        if requires_grad:
            view.backward(torch.ones_like(view))
            self.assertEqual(nt.grad, torch.ones_like(nt))

        # non-contiguous NT should result in contiguous copy
        nt = random_nt_from_dims(
            [3, None, 5, 2],
            device=device,
            dtype=torch.float32,
            layout=torch.jagged,
            requires_grad=requires_grad,
        )
        nt_noncontig = nt.transpose(-1, -2)
        self.assertFalse(nt_noncontig.is_contiguous())
        copy = nt_noncontig.reshape(-1, -1, 10)
        self.assertTrue(copy.is_contiguous())
        self.assertEqual(copy.shape[:2], nt.shape[:2])
        # make sure gradients flow back
        if requires_grad:
            copy.backward(torch.ones_like(copy))
            self.assertEqual(nt.grad, torch.ones_like(nt))

    def test_flatten_decomp(self, device):
        nt = random_nt_from_dims(
            [3, None, 5, 2], device=device, dtype=torch.float32, layout=torch.jagged
        )
        flattened = nt.flatten(-2, -1)
        self.assertEqual(flattened.shape, nt.view(3, -1, 10).shape)

        nt = random_nt_from_dims(
            [3, None, 5, 2, 6], device=device, dtype=torch.float32, layout=torch.jagged
        )
        flattened = nt.flatten(-3, -2)
        self.assertEqual(flattened.shape, nt.view(3, -1, 10, 6).shape)

    def test_chunk(self, device):
        # none NJT case
        t = torch.randn(10, 4, 5, requires_grad=True)
        t_list = t.chunk(3, dim=0)
        loss = t_list[0].sum() + t_list[2].sum()
        loss.backward()

        # normal case
        D = 30
        B = 8
        nt = random_nt_from_dims(
            [B, None, D],
            device=device,
            dtype=torch.float32,
            layout=torch.jagged,
            requires_grad=True,
        )
        NUM_CHUNKS = 3
        chunks = nt.chunk(NUM_CHUNKS, dim=-1)
        self.assertEqual(len(chunks), NUM_CHUNKS)
        for i in range(NUM_CHUNKS):
            self.assertEqual(chunks[i].shape[-1], D // NUM_CHUNKS)

        # test chunk_backward
        values = torch.randn(
            5, 11, dtype=torch.float64, device=device, requires_grad=True
        )
        offsets = torch.tensor([0, 2, 3, 5], device=device)

        def grad_test_func(values, offsets):
            nt = torch.nested.nested_tensor_from_jagged(values, offsets)
            chunks = nt.chunk(3, dim=-1)
            return chunks[0].values().sum()

        assert gradcheck(
            grad_test_func,
            inputs=(values, offsets),
            check_batched_grad=False,
        )

        # chunk on batch dim
        chunks = nt.chunk(NUM_CHUNKS, dim=0)
        self.assertEqual(len(chunks), NUM_CHUNKS)
        chunk_size = math.ceil(B / NUM_CHUNKS)
        for i in range(NUM_CHUNKS):
            if i < NUM_CHUNKS - 1:
                self.assertEqual(chunks[i].shape[0], chunk_size)
            else:
                self.assertEqual(chunks[i].shape[0], B - chunk_size * (NUM_CHUNKS - 1))
            offsets_expected = (
                nt._offsets[i * chunk_size + 1 : (i + 1) * chunk_size + 1]
                - nt._offsets[i * chunk_size]
            )
            self.assertEqual(chunks[i]._offsets[1:], offsets_expected)
        self.assertEqual(nt._values, torch.cat([x._values for x in chunks], dim=0))

        with self.assertRaisesRegex(
            RuntimeError,
            "dim != 0 INTERNAL ASSERT FAILED .* Nested Tensor doesn't support chunk backward on dim=0 yet.",
        ):
            # doesn't support backward for chunk (dim=0) yet
            loss = (
                chunks[0].values().sum()
                + chunks[1].values().sum()
                + chunks[2].values().sum()
            )
            loss.backward()

        # chunk on ragged dim not supported
        with self.assertRaisesRegex(
            RuntimeError, "chunk.* not supported for NestedTensor on dim=1"
        ):
            nt.chunk(2, dim=1)

    def test_squeeze(self, device):
        B = 4
        D = 6
        # squeeze middle dim
        nt = random_nt_from_dims(
            [B, None, 1, D], device=device, dtype=torch.float32, layout=torch.jagged
        )
        j0 = nt.shape[1]

        for dim_arg in [-2, 2]:
            out = nt.squeeze(dim_arg)
            self.assertEqual(out.shape, (B, j0, D))
            self.assertEqual(out.unsqueeze(-2), nt)

        # squeeze last dim
        nt = random_nt_from_dims(
            [B, None, 1], device=device, dtype=torch.float32, layout=torch.jagged
        )
        j1 = nt.shape[1]

        for dim_arg in [-1, 2]:
            out = nt.squeeze(dim_arg)
            self.assertEqual(out.shape, (B, j1))
            self.assertEqual(out.unsqueeze(-1), nt)

        # squeeze on batch dim not supported
        with self.assertRaisesRegex(
            RuntimeError, "squeeze.* not supported for NestedTensor on dim=0"
        ):
            nt.squeeze(0)

        # squeeze on ragged dim not supported
        with self.assertRaisesRegex(
            RuntimeError, "squeeze.* not supported for NestedTensor on dim=1"
        ):
            nt.squeeze(1)

    def test_binary_pointwise_broadcasting(self, device):
        # (B, j0, 3, 4)
        ts = self._get_list_for_jagged_tensor(
            ((2, 3, 4), 3, 4), device, requires_grad=True
        )
        # (B, j0, ?, ?) + (?) -> (B, j0, ?, ?)
        # (B, j0, ?, ?) + (?, ?) -> (B, j0, ?, ?)
        # (B, j0, ?, ?) + (1, ?, ?) -> (B, j0, ?, ?)
        # Unsupported: (B, j0, ?, ?) + (1, 1, 1, ?, ?) -> (1, B, j0, ?, ?)
        t_sizes = (
            (4,),
            (1, 4),
            (3, 1),
            (1, 3, 1),
            (1, 1, 1, 4),
            # (1, 1, 1, 1, 4), (unsupported today)
        )

        def grad_test_func(t, *ts):
            nt = torch.nested.as_nested_tensor(list(ts), layout=torch.jagged)
            out = nt + t
            return out.values()

        for t_size in t_sizes:
            t = torch.rand(
                t_size, requires_grad=True, device=device, dtype=torch.float64
            )
            gradcheck(grad_test_func, inputs=(t, *ts), check_batched_grad=False)

    def test_threshold_backward(self, device):
        ts1 = self._get_list_for_jagged_tensor(
            ((2, 3, 4), 16), device=device, requires_grad=False
        )
        ts2 = self._get_list_for_jagged_tensor(
            ((2, 3, 4), 16), device=device, requires_grad=False
        )

        nt1, offsets = jagged_from_list(ts1, None)
        nt2, offsets = jagged_from_list(ts2, offsets)
        buf1 = nt1.values().detach().clone()
        buf2 = nt2.values().detach().clone()

        res_nt = torch.ops.aten.threshold_backward(nt1, nt2, 0.0)
        res_dense = torch.ops.aten.threshold_backward(buf1, buf2, 0.0)

        self.assertEqual(res_dense, res_nt.values())

    @onlyCUDA
    @dtypes(torch.float32)
    def test_record_stream(self, device, dtype):
        def _create_nt():
            values = torch.ones(1024, 4 * 1024, device="cuda")
            offsets = torch.tensor([0, 500, 1024], device="cuda", dtype=torch.int64)
            lengths = offsets.diff()
            nt = torch.nested.nested_tensor_from_jagged(values, offsets, lengths)
            data_ptrs = {
                nt._values.data_ptr(),
                nt._offsets.data_ptr(),
                nt._lengths.data_ptr(),
            }
            return nt, data_ptrs

        def fn(record_stream):
            nt, data_ptrs = _create_nt()
            s = torch.cuda.Stream()

            with torch.cuda.stream(s):
                # emulate doing something long via sleep
                per_ms = 2e7
                torch.cuda._sleep(int(per_ms * 100))
                if record_stream:
                    nt.record_stream(s)
            return data_ptrs

        # expect memory reuse when record_stream() is not run
        data_ptrs = fn(record_stream=False)
        nt, nt_data_ptrs = _create_nt()
        self.assertEqual(data_ptrs, nt_data_ptrs)
        del nt
        torch.cuda.synchronize()

        # expect memory to be preserved (no reuse) when record_stream() is run
        data_ptrs = fn(record_stream=True)
        nt, nt_data_ptrs = _create_nt()
        self.assertEqual(len(data_ptrs.intersection(nt_data_ptrs)), 0)

    @dtypes(torch.float32)
    @parametrize(
        "func",
        [torch.ops.aten.sum.dim_IntList, torch.ops.aten.mean.dim],
        name_fn=get_op_name,
    )
    @parametrize("keepdim", [False, True])
    @parametrize("requires_grad", [False, True])
    @parametrize("components_require_grad", [False, True])
    def test_jagged_op_different_output_shape_dim(
        self, device, dtype, keepdim, requires_grad, components_require_grad, func
    ):
        """
        Operator passes when reducing on valid reduction dimensions.
        This test is for operators which return an output tensor with a shape different from the input tensor.
        """
        if get_op_name(func) == "mean" and not keepdim:
            return

        op_name = get_op_name(func)

        ts = self._get_list_for_jagged_tensor(
            ((2, 3, 4), 3, 4), device=device, requires_grad=True
        )  # (B, j0, 3, 4)

        # verify correctness of shapes (assuming that ragged_idx == 1)
        if op_name == "sum":
            reduce_dims = (
                ((0, 1), (3, 4), (1, 1, 3, 4), (0,)),  # batch, ragged
                ((2, 3), (3, None), (3, None, 1, 1), (1, 2)),  # non-batch, non-batch
                ((0, 1, 3), (3,), (1, 1, 3, 1), (0, 2)),  # batch, ragged, non-batch
                ((0, 1, 2), (4,), (1, 1, 1, 4), (0, 1)),  # batch, ragged, non-batch
                (
                    (0, 1, 2, 3),
                    (),
                    (1, 1, 1, 1),
                    (0, 1, 2),
                ),  # batch, ragged, non-batch, non-batch
                ((2,), (3, None, 4), (3, None, 1, 4), (1,)),  # non-batch
            )  # (dims, expected shape, expected keepdim shape, reduce_dim_expected), where j0 is represented as None
        elif op_name == "mean":
            reduce_dims = (
                ((2,), (3, None, 4), (3, None, 1, 4), (1,)),
                ((3,), (3, None, 3), (3, None, 3, 1), (2,)),
            )

        for rd, ref_shape_no_keepdim, ref_shape_keepdim, _ in reduce_dims:
            nt = torch.nested.as_nested_tensor(ts, layout=torch.jagged)
            out = func(nt, dim=rd, keepdim=keepdim)
            ref_shape = ref_shape_keepdim if keepdim else ref_shape_no_keepdim
            if not torch.compiler.is_compiling:  # if not using torch dynamo
                self.assertEqual(len(out.shape), len(ref_shape))
                for o, r in zip(out.shape, ref_shape):
                    if r is not None:
                        self.assertEqual(o, r)
                    else:
                        self.assertTrue(isinstance(o, torch.SymInt))

        # verify correctness of values
        tensor_lists = self._get_example_tensor_lists(
            include_list_of_lists=False,
            include_requires_grad=components_require_grad,
            include_inner_dim_size_1=True,
        )
        for tensor_list, reduce_dim_tuple in itertools.product(
            tensor_lists, reduce_dims
        ):
            nt = torch.nested.nested_tensor(
                tensor_list,
                device=device,
                dtype=dtype,
                layout=torch.jagged,
                requires_grad=requires_grad,
            )

            reduce_dim, _, _, reduce_dim_expected = reduce_dim_tuple

            if nt.dim() > reduce_dim[-1]:
                out_actual = func(nt, dim=reduce_dim, keepdim=keepdim)
                if nt._ragged_idx in reduce_dim:  # raggedness reduced away
                    out_expected = func(
                        nt.values(), dim=reduce_dim_expected, keepdim=keepdim
                    )
                    self.assertTrue(torch.allclose(out_actual, out_expected))
                else:  # raggedness preserved
                    out_expected = func(nt.values(), dim=reduce_dim_expected)
                    self.assertTrue(
                        torch.allclose(
                            out_actual.values().view(-1), out_expected.view(-1)
                        )
                    )

    @dtypes(torch.float32)
    @parametrize("requires_grad", [False, True])
    @parametrize("components_require_grad", [False, True])
    def test_softmax_dim(
        self,
        device,
        dtype,
        requires_grad,
        components_require_grad,
    ):
        """
        Softmax passes when reducing on valid reduction dimensions.
        """
        ts = self._get_list_for_jagged_tensor(
            ((2, 3, 4), 3, 4), device=device, requires_grad=True
        )  # (B, j0, 3, 4)

        output_shape = (3, None, 3, 4)

        # verify correctness of shapes (assuming that ragged_idx == 1)
        reduce_dims = (
            (2, 1),
            (3, 2),
        )  # (reduction dimension, effective reduction dimension for baseline)

        for reduce_dim, _ in reduce_dims:
            nt = torch.nested.as_nested_tensor(ts, layout=torch.jagged)
            out_actual = torch.nn.functional.softmax(nt, dim=reduce_dim)
            torch._dynamo.disable(self.assertEqual)(
                len(out_actual.shape), len(output_shape)
            )  # disable if running on dynamo
            for dim_actual, dim_expected in zip(out_actual.shape, output_shape):
                if dim_expected is not None:
                    self.assertEqual(dim_actual, dim_expected)
                else:
                    self.assertTrue(isinstance(dim_actual, torch.SymInt))

        # verify correctness of values
        tensor_lists = self._get_example_tensor_lists(
            include_list_of_lists=False,
            include_requires_grad=components_require_grad,
            include_inner_dim_size_1=True,
        )
        for tensor_list, reduce_dim_tuple in itertools.product(
            tensor_lists, reduce_dims
        ):
            nt = torch.nested.nested_tensor(
                tensor_list,
                device=device,
                dtype=dtype,
                layout=torch.jagged,
                requires_grad=requires_grad,
            )

            reduce_dim, reduce_dim_expected = reduce_dim_tuple

            if nt.dim() > reduce_dim:
                out_actual = torch.nn.functional.softmax(
                    nt, dim=reduce_dim
                )  # nested tensor
                out_expected = torch.nn.functional.softmax(
                    nt.values(), dim=reduce_dim_expected
                )  # dense tensor of dimensions 1 less than out_actual
                self.assertTrue(
                    torch.allclose(out_actual.values().view(-1), out_expected.view(-1))
                )

    @dtypes(torch.float32)
    @parametrize(
        "func",
        [torch.ops.aten.sum.dim_IntList, torch.ops.aten.mean.dim],
        name_fn=get_op_name,
    )
    @parametrize("keepdim", [False, True])
    @parametrize("requires_grad", [False, True])
    @parametrize("components_require_grad", [False, True])
    def test_op_dim_reduce_ragged_idx_1_different_output_shape(
        self, device, dtype, keepdim, requires_grad, components_require_grad, func
    ):
        """
        Operator on NestedTensor passes when trying to reduce across ragged dimension, where ragged_idx == 1.
        This test is for operators which return an output tensor with a shape different from the input tensor.
        """
        if get_op_name(func) == "mean" and not keepdim:
            return

        op_name = get_op_name(func)

        tensor_lists = self._get_example_tensor_lists(
            include_list_of_lists=False,
            include_requires_grad=components_require_grad,
            include_inner_dim_size_1=True,  # (B, *, 1)
        )
        reduce_dim = (1,)  # ragged

        for tensor_list in tensor_lists:
            nt = torch.nested.nested_tensor(
                tensor_list,
                device=device,
                dtype=dtype,
                layout=torch.jagged,
                requires_grad=requires_grad,
            )

            out_actual = func(nt, dim=reduce_dim, keepdim=keepdim)
            out_expected = torch.cat(
                [func(t, dim=(reduce_dim[0] - 1)).unsqueeze(0) for t in nt.unbind()]
            )
            if keepdim:
                out_expected = out_expected.unsqueeze(reduce_dim[0])

            self.assertFalse(
                out_actual.is_nested,
                f"{op_name}(): the result of reducing a nested tensor along the ragged dimension is a dense tensor",
            )  # output is a dense tensor
            self.assertEqual(out_actual, out_expected)

    @dtypes(torch.float32)
    @parametrize("requires_grad", [False, True])
    @parametrize("components_require_grad", [False, True])
    def test_softmax_dim_reduce_ragged_idx_1(
        self, device, dtype, requires_grad, components_require_grad
    ):
        """
        Softmax on NestedTensor passes when trying to reduce across ragged dimension, where ragged_idx == 1.
        """
        tensor_lists = self._get_example_tensor_lists(
            include_list_of_lists=False,
            include_requires_grad=components_require_grad,
            include_inner_dim_size_1=True,  # (B, *, 1)
            include_2d_tensor=True,  # (B, *)
        )
        reduce_dim = 1  # ragged

        for tensor_list in tensor_lists:
            nt = torch.nested.nested_tensor(
                tensor_list,
                device=device,
                dtype=dtype,
                layout=torch.jagged,
                requires_grad=requires_grad,
            )

            out_actual = torch.nn.functional.softmax(nt, dim=reduce_dim)
            out_expected = torch.cat(
                [
                    torch.nn.functional.softmax(t, dim=reduce_dim - 1)
                    for t in nt.unbind()
                ]
            )

            self.assertTrue(
                out_actual.is_nested,
                "softmax(): the result of reducing a nested tensor along the ragged dimension is a nested tensor",
            )  # output is a nested tensor
            self.assertTrue(torch.allclose(out_actual.values(), out_expected))

    @dtypes(torch.float32)
    @parametrize("requires_grad", [False, True])
    @parametrize("components_require_grad", [False, True])
    def test_softmax_reduce_batch_dim(
        self, device, dtype, requires_grad, components_require_grad
    ):
        """
        Softmax on NestedTensor fails when trying to reduce across batch dimension.
        """
        tensor_lists = self._get_example_tensor_lists(
            include_list_of_lists=False,
            include_requires_grad=components_require_grad,
            include_inner_dim_size_1=True,  # (B, *, 1)
        )
        reduce_dim = 0  # batch

        for tensor_list in tensor_lists:
            nt = torch.nested.nested_tensor(
                tensor_list,
                device=device,
                dtype=dtype,
                layout=torch.jagged,
                requires_grad=requires_grad,
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "not supported when reducing across the batch dimension for NestedTensor",
            ):
                out = torch.nn.functional.softmax(nt, dim=reduce_dim)

    @dtypes(torch.float32)
    @parametrize("requires_grad", [False, True])
    @parametrize("components_require_grad", [False, True])
    def test_layer_norm_reduce_ragged_idx_1(
        self, device, dtype, requires_grad, components_require_grad
    ):
        """
        Layer normalization on NestedTensor passes when trying to normalize across ragged dimension, where ragged_idx == 1.
        """

        # requires_grad = False does not currently work with dynamo tests and throws this error:
        #   AssertionError: SymInts must use SymNodeVariable.
        #   If the underlying value is static, we will create a ConstantVariable and specialize.
        if torch._dynamo.is_compiling() and not requires_grad:
            return

        tensor_lists = self._get_example_tensor_lists(
            include_list_of_lists=False,
            include_requires_grad=components_require_grad,
            include_inner_dim_size_1=True,  # (B, *, 1)
        )

        for tensor_list in tensor_lists:
            nt = torch.nested.nested_tensor(
                tensor_list,
                device=device,
                dtype=dtype,
                layout=torch.jagged,
                requires_grad=requires_grad,
            )

            if (
                nt.dim() >= 3
            ):  # layer norm only works for tensors with 3 or more dimensions
                normalized_shape = nt.shape[nt._ragged_idx :]

                out_actual = torch.nn.functional.layer_norm(
                    nt, normalized_shape=normalized_shape
                )
                out_expected = torch.cat(
                    [
                        torch.nn.functional.layer_norm(t, normalized_shape=t.shape)
                        for t in nt.unbind()
                    ]
                )  # e.g. in 3D tensor (B, *, M), performs layer normalization on B 2D tensors (*, M)

                self.assertTrue(
                    out_actual.is_nested,
                    "layer_norm(): the result of reducing a nested tensor along the ragged dimension is a nested tensor",
                )  # output is a nested tensor
                self.assertEqual(out_actual._values.shape, out_expected.shape)
                self.assertTrue(torch.allclose(out_actual.values(), out_expected))

    @dtypes(torch.float32)
    @parametrize("requires_grad", [False, True])
    @parametrize("components_require_grad", [False, True])
    def test_layer_norm_2d_input(
        self,
        device,
        dtype,
        requires_grad,
        components_require_grad,
    ):
        """
        Layer normalization on NestedTensor fails when trying to operate on a 2-dimensional tensor
        """
        tensor_lists = self._get_example_tensor_lists(
            include_list_of_lists=False,
            include_requires_grad=components_require_grad,
            include_inner_dim_size_1=True,  # (B, *, 1)
            include_2d_tensor=True,  # (B, *)
        )

        for tensor_list in tensor_lists:
            nt = torch.nested.nested_tensor(
                tensor_list,
                device=device,
                dtype=dtype,
                layout=torch.jagged,
                requires_grad=requires_grad,
            )

            if nt.dim() <= 2:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "not supported for NestedTensor objects with 2 or fewer dimensions",
                ):
                    out = torch.nn.functional.layer_norm(
                        nt, normalized_shape=(nt.shape[nt._ragged_idx],)
                    )

    @dtypes(torch.float32)
    @parametrize("requires_grad", [False, True])
    @parametrize("components_require_grad", [False, True])
    def test_layer_norm_operate_on_batch_dim(
        self,
        device,
        dtype,
        requires_grad,
        components_require_grad,
    ):
        """
        Layer normalization on NestedTensor fails when trying to operate on the batch dimension
        """
        tensor_lists = self._get_example_tensor_lists(
            include_list_of_lists=False,
            include_requires_grad=components_require_grad,
            include_inner_dim_size_1=True,  # (B, *, 1)
            include_2d_tensor=True,  # (B, *)
        )

        for tensor_list in tensor_lists:
            nt = torch.nested.nested_tensor(
                tensor_list,
                device=device,
                dtype=dtype,
                layout=torch.jagged,
                requires_grad=requires_grad,
            )

            if nt.dim() > 2:  # cannot perform layer normalization on 2D tensors
                with self.assertRaisesRegex(
                    RuntimeError,
                    "not supported when normalizing over the batch dimension for NestedTensor",
                ):
                    out = torch.nn.functional.layer_norm(nt, normalized_shape=nt.shape)

    @dtypes(torch.float32)
    @parametrize(
        "func",
        [torch.ops.aten.sum.dim_IntList, torch.ops.aten.mean.dim],
        name_fn=get_op_name,
    )
    @parametrize(
        "transpose_offset", [1, 2]
    )  # [transpose consecutive dimensions, transpose nonconsecutive dimensions]
    @parametrize("keepdim", [False, True])
    @parametrize("requires_grad", [False, True])
    @parametrize("components_require_grad", [False, True])
    def test_op_dim_reduce_ragged_idx_greater_than_1_different_output_shape(
        self,
        device,
        dtype,
        keepdim,
        requires_grad,
        components_require_grad,
        func,
        transpose_offset,
    ):
        """
        Operator on NestedTensor passes when trying to reduce across a transposed ragged dimension, i.e. ragged_idx > 1
        This test is for operators which return an output tensor with a shape different from the input tensor.
        """
        if get_op_name(func) == "mean" and not keepdim:
            return

        op_name = get_op_name(func)

        tensor_lists = self._get_example_tensor_lists(
            include_list_of_lists=False,
            include_requires_grad=components_require_grad,
            include_inner_dim_size_1=True,  # (B, *, 1)
            include_2d_tensor=True,  # (B, *)
        )

        for tensor_list in tensor_lists:
            nt = torch.nested.nested_tensor(
                tensor_list,
                device=device,
                dtype=dtype,
                layout=torch.jagged,
                requires_grad=requires_grad,
            )

            if nt.dim() > nt._ragged_idx + transpose_offset:
                nt_transposed = nt.transpose(
                    nt._ragged_idx, nt._ragged_idx + transpose_offset
                )
                reduce_dim = (nt_transposed._ragged_idx,)  # ragged

                out_actual = func(nt_transposed, dim=reduce_dim, keepdim=keepdim)
                out_expected = torch.cat(
                    [
                        func(t, dim=(reduce_dim[0] - 1)).unsqueeze(0)
                        for t in nt_transposed.unbind()
                    ]
                )
                if keepdim:
                    out_expected = out_expected.unsqueeze(reduce_dim[0])

                self.assertFalse(
                    out_actual.is_nested,
                    f"{op_name}(): the result of reducing a nested tensor along the ragged dimension is a dense tensor",
                )  # output is a dense tensor
                self.assertEqual(out_actual, out_expected)

    @dtypes(torch.float32)
    @parametrize(
        "transpose_offset", [1, 2]
    )  # [transpose consecutive dimensions, transpose nonconsecutive dimensions]
    @parametrize("requires_grad", [False, True])
    @parametrize("components_require_grad", [False, True])
    def test_softmax_dim_reduce_ragged_idx_greater_than_1_same_output_shape(
        self,
        device,
        dtype,
        requires_grad,
        components_require_grad,
        transpose_offset,
    ):
        """
        Softmax on NestedTensor fails when trying to reduce across a transposed ragged dimension, i.e. ragged_idx > 1
        This test is for operators which return an output tensor with the same shape as the input tensor.
        """
        tensor_lists = self._get_example_tensor_lists(
            include_list_of_lists=False,
            include_requires_grad=components_require_grad,
            include_inner_dim_size_1=True,  # (B, *, 1)
        )

        for tensor_list in tensor_lists:
            nt = torch.nested.nested_tensor(
                tensor_list,
                device=device,
                dtype=dtype,
                layout=torch.jagged,
                requires_grad=requires_grad,
            )

            if nt.dim() > nt._ragged_idx + transpose_offset:
                nt_transposed = nt.transpose(
                    nt._ragged_idx, nt._ragged_idx + transpose_offset
                )
                reduce_dim = nt_transposed._ragged_idx  # ragged

                with self.assertRaisesRegex(
                    RuntimeError,
                    "not supported when reducing along the ragged dimension for ragged_idx > 1 for NestedTensor",
                ):
                    out = torch.nn.functional.softmax(nt_transposed, dim=reduce_dim)

    @dtypes(torch.float32)
    @parametrize(
        "func",
        [torch.ops.aten.sum.dim_IntList, torch.ops.aten.mean.dim],
        name_fn=get_op_name,
    )
    @parametrize("keepdim", [False, True])
    @parametrize("requires_grad", [False, True])
    @parametrize("components_require_grad", [False, True])
    def test_op_dim_transpose_non_ragged_dim_different_output_shape(
        self, device, dtype, keepdim, requires_grad, components_require_grad, func
    ):
        """
        Operator passes when reducing transposed nested tensors on valid reduction dimensions.
        This test is for operators which return an output tensor with a shape different from the input tensor.
        """
        if get_op_name(func) == "mean" and not keepdim:
            return

        # verify correctness of shapes (assuming that ragged_idx == 1)
        if get_op_name(func) == "sum":
            reduce_dims = (
                ((0, 1), (3, 4), (1, 1, 3, 4), (0,)),  # batch, ragged
                ((2, 3), (3, None), (3, None, 1, 1), (1, 2)),  # non-batch, non-batch
                ((0, 1, 3), (3,), (1, 1, 3, 1), (0, 2)),  # batch, ragged, non-batch
                ((0, 1, 2), (4,), (1, 1, 1, 4), (0, 1)),  # batch, ragged, non-batch
                (
                    (0, 1, 2, 3),
                    (),
                    (1, 1, 1, 1),
                    (0, 1, 2),
                ),  # batch, ragged, non-batch, non-batch
                ((2,), (3, None, 4), (3, None, 1, 4), (1,)),  # non-batch
            )  # (dims, expected shape, expected keepdim shape, reduce_dim_expected), where j0 is represented as None
        elif get_op_name(func) == "mean":
            reduce_dims = (
                ((2,), (3, None, 4), (3, None, 1, 4), (1,)),
                ((3,), (3, None, 3), (3, None, 3, 1), (2,)),
            )

        # verify correctness of values
        tensor_lists = self._get_example_tensor_lists(
            include_list_of_lists=False,
            include_requires_grad=components_require_grad,
        )
        for tensor_list, reduce_dim_tuple in itertools.product(
            tensor_lists, reduce_dims
        ):
            nt = torch.nested.nested_tensor(
                tensor_list,
                device=device,
                dtype=dtype,
                layout=torch.jagged,
                requires_grad=requires_grad,
            ).transpose(-1, -2)

            reduce_dim, _, _, reduce_dim_expected = reduce_dim_tuple

            if nt.dim() > max(
                reduce_dim[-1], nt._ragged_idx + 2
            ):  # ensure that transposed dimensions are non-batch, non-ragged dimensions
                out_actual = func(nt, dim=reduce_dim, keepdim=keepdim)
                if nt._ragged_idx in reduce_dim:  # raggedness reduced away
                    out_expected = func(
                        nt.values(), dim=reduce_dim_expected, keepdim=keepdim
                    )
                    self.assertTrue(torch.allclose(out_actual, out_expected))
                else:  # raggedness preserved
                    out_expected = func(nt.values(), dim=reduce_dim_expected)
                    self.assertTrue(
                        torch.allclose(
                            out_actual.values().view(-1), out_expected.view(-1)
                        )
                    )

    @dtypes(torch.float32)
    @parametrize("requires_grad", [False, True])
    @parametrize("components_require_grad", [False, True])
    def test_softmax_dim_transpose_non_ragged_dim(
        self,
        device,
        dtype,
        requires_grad,
        components_require_grad,
    ):
        """
        Softmax passes when reducing transposed nested tensors on valid reduction dimensions.
        This test is for operators which return an output tensor with the same shape as the input tensor.
        """
        # verify correctness of shapes (assuming that ragged_idx == 1)
        reduce_dims = (
            (2, 1),
            (3, 2),
        )  # (reduction dimension, effective reduction dimension for baseline)

        # verify correctness of values
        tensor_lists = self._get_example_tensor_lists(
            include_list_of_lists=False,
            include_requires_grad=components_require_grad,
            include_inner_dim_size_1=True,  # (B, *, 1)
        )
        for tensor_list, reduce_dim_tuple in itertools.product(
            tensor_lists, reduce_dims
        ):
            nt = torch.nested.nested_tensor(
                tensor_list,
                device=device,
                dtype=dtype,
                layout=torch.jagged,
                requires_grad=requires_grad,
            ).transpose(-1, -2)

            reduce_dim, reduce_dim_expected = reduce_dim_tuple

            if nt.dim() > max(reduce_dim, nt._ragged_idx + 2):
                out_actual = torch.nn.functional.softmax(
                    nt, dim=reduce_dim
                )  # nested tensor
                out_expected = torch.nn.functional.softmax(
                    nt.values(), dim=reduce_dim_expected
                )  # dense tensor of dimensions 1 less than out_actual

                self.assertTrue(
                    torch.allclose(out_actual.values().view(-1), out_expected.view(-1))
                )

    @dtypes(torch.float32)
    @parametrize("keepdim", [False, True])
    @parametrize("requires_grad", [False, True])
    @parametrize("components_require_grad", [False, True])
    def test_sum_dim_reduce_ragged_and_non_batch(
        self,
        device,
        dtype,
        keepdim,
        requires_grad,
        components_require_grad,
    ):
        """
        Sum on NestedTensor fails when trying to reduce across ragged and non-batch dimensions
        """
        tensor_lists = self._get_example_tensor_lists(
            include_list_of_lists=False, include_requires_grad=components_require_grad
        )
        reduce_dims = (
            (1, 2),  # ragged, non-batch
            (1, 3),  # ragged, non-batch
        )

        for tensor_list, reduce_dim in itertools.product(tensor_lists, reduce_dims):
            nt = torch.nested.nested_tensor(
                tensor_list,
                device=device,
                dtype=dtype,
                layout=torch.jagged,
                requires_grad=requires_grad,
            )

            if nt.dim() > reduce_dim[-1]:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "reducing along a ragged and non-batch dimension is not supported",
                ):
                    out = torch.sum(nt, dim=reduce_dim, keepdim=keepdim)

    @dtypes(torch.float32)
    @parametrize("keepdim", [False, True])
    @parametrize("requires_grad", [False, True])
    @parametrize("components_require_grad", [False, True])
    def test_sum_dim_reduce_batch_and_non_batch(
        self,
        device,
        dtype,
        keepdim,
        requires_grad,
        components_require_grad,
    ):
        """
        Sum on NestedTensor fails when trying to reduce across batch and non-batch dimensions
        """
        tensor_lists = self._get_example_tensor_lists(
            include_list_of_lists=False, include_requires_grad=components_require_grad
        )
        reduce_dims = (
            (0, 2),  # batch, non-batch
            (0, 3),  # batch, non-batch
        )

        for tensor_list, reduce_dim in itertools.product(tensor_lists, reduce_dims):
            nt = torch.nested.nested_tensor(
                tensor_list,
                device=device,
                dtype=dtype,
                layout=torch.jagged,
                requires_grad=requires_grad,
            )

            if nt.dim() > reduce_dim[-1]:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "reducing along the batch dimension but not the ragged dimension "
                    + "is not supported",
                ):
                    out = torch.sum(nt, dim=reduce_dim, keepdim=keepdim)

    @dtypes(torch.float32)
    @parametrize(
        "func",
        [torch.ops.aten.sum.dim_IntList, torch.ops.aten.mean.dim],
        name_fn=get_op_name,
    )
    @parametrize("keepdim", [False, True])
    @parametrize("requires_grad", [False, True])
    @parametrize("components_require_grad", [False, True])
    def test_op_dim_reduce_batch_only_different_output_shape(
        self, device, dtype, keepdim, requires_grad, components_require_grad, func
    ):
        """
        Operator on NestedTensor fails when trying to reduce across batch dimension
        """
        if get_op_name(func) == "mean" and not keepdim:
            return

        tensor_lists = self._get_example_tensor_lists(
            include_list_of_lists=False, include_requires_grad=components_require_grad
        )
        reduce_dim = (0,)  # batch

        for tensor_list in tensor_lists:
            nt = torch.nested.nested_tensor(
                tensor_list,
                device=device,
                dtype=dtype,
                layout=torch.jagged,
                requires_grad=requires_grad,
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "reducing along the batch dimension but not the ragged dimension "
                + "is not supported",
            ):
                out = func(nt, dim=reduce_dim, keepdim=keepdim)

    @dtypes(torch.float32)
    @parametrize(
        "func",
        [torch.ops.aten.sum.dim_IntList, torch.ops.aten.mean.dim],
        name_fn=get_op_name,
    )
    @parametrize("keepdim", [False, True])
    @parametrize("requires_grad", [False, True])
    @parametrize("components_require_grad", [False, True])
    def test_op_dim_with_lengths_different_output_shape(
        self,
        device,
        dtype,
        keepdim,
        requires_grad,
        components_require_grad,
        func,
    ):
        """
        Operator on NestedTensor fails when trying to reduce a nested tensor with lengths,
        i.e. a nested tensor with holes, if reducing on the ragged dimension.
        This test is for operators which return an output tensor with different shape than the input tensor.
        """
        if get_op_name(func) == "mean" and not keepdim:
            return

        reduce_dims = ((1,), (2,), (2, 3))

        lengths = torch.randint(5, 10, (20,), device=device)
        offsets = torch.zeros((21,), device=device, dtype=torch.int)
        torch.cumsum(lengths, dim=0, out=offsets[1:])

        values = torch.randn(
            (offsets[-1].item(), 20),
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )

        nt_with_holes = torch.nested.nested_tensor_from_jagged(
            values,
            offsets,
            lengths=offsets.diff() - 2,  # arbitrary subtraction to create holes
        )

        for reduce_dim in reduce_dims:
            if nt_with_holes.dim() > reduce_dim[-1]:
                if nt_with_holes._ragged_idx in reduce_dim:
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "reducing across the ragged dimension is not supported for "
                        + "non-contiguous nested tensors with holes",
                    ):
                        out = func(nt_with_holes, dim=reduce_dim, keepdim=keepdim)
                else:
                    out = func(nt_with_holes, dim=reduce_dim, keepdim=keepdim)

    @dtypes(torch.float32)
    @parametrize("requires_grad", [False, True])
    @parametrize("components_require_grad", [False, True])
    def test_softmax_dim_with_lengths(
        self,
        device,
        dtype,
        requires_grad,
        components_require_grad,
    ):
        """
        Softmax on NestedTensor fails when trying to reduce a nested tensor with lengths,
        i.e. a nested tensor with holes, if reducing on the ragged dimension.
        """
        reduce_dims = (1, 2, 3)

        lengths = torch.randint(5, 10, (20,), device=device)
        offsets = torch.zeros((21,), device=device, dtype=torch.int)
        torch.cumsum(lengths, dim=0, out=offsets[1:])

        values = torch.randn(
            (offsets[-1].item(), 20),
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )

        nt_with_holes = torch.nested.nested_tensor_from_jagged(
            values,
            offsets,
            lengths=offsets.diff() - 2,  # arbitrary subtraction to create holes
        )

        for reduce_dim in reduce_dims:
            if nt_with_holes.dim() > reduce_dim:
                if nt_with_holes._ragged_idx == reduce_dim:
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "not supported where lengths is not None "
                        + "if reducing across the ragged dimension for NestedTensor",
                    ):
                        out = torch.nn.functional.softmax(nt_with_holes, dim=reduce_dim)
                else:
                    out = torch.nn.functional.softmax(nt_with_holes, dim=reduce_dim)

    @skipIfTorchDynamo(
        "ragged_size = nt_with_holes.shape[nt_with_holes._ragged_idx] does not currently work "
        + "with dynamo tests and throws this error: `AssertionError: SymInts must use SymNodeVariable. "
        + "If the underlying value is static, we will create a ConstantVariable and specialize.`"
    )
    @dtypes(torch.float32)
    @parametrize("requires_grad", [False, True])
    @parametrize("components_require_grad", [False, True])
    def test_layer_norm_with_lengths(
        self,
        device,
        dtype,
        requires_grad,
        components_require_grad,
    ):
        """
        Layer normalization on NestedTensor fails when trying to operate on a nested tensor with lengths,
        i.e. a nested tensor with holes, if operating on the ragged dimension.
        """

        # create components for nested tensor
        lengths = torch.randint(5, 10, (20,), device=device)
        offsets = torch.zeros((21,), device=device, dtype=torch.int)
        torch.cumsum(lengths, dim=0, out=offsets[1:])
        values = torch.randn(
            (offsets[-1].item(), 10, 30),
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )

        nt_with_holes = torch.nested.nested_tensor_from_jagged(
            values,
            offsets,
            lengths=offsets.diff() - 2,  # arbitrary subtraction to create holes
        )

        ragged_size = nt_with_holes.shape[nt_with_holes._ragged_idx]

        normalized_shapes = (
            (10, 30),  # normalization on non-ragged dimension passes
            (ragged_size, 10, 30),  # normalization on ragged dimension fails
        )

        for normalized_shape in normalized_shapes:
            if ragged_size in normalized_shape:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "not supported where lengths is not None if operating on the ragged dimension for NestedTensor",
                ):
                    out = torch.nn.functional.layer_norm(
                        nt_with_holes, normalized_shape=normalized_shape
                    )
            else:
                out = torch.nn.functional.layer_norm(
                    nt_with_holes, normalized_shape=normalized_shape
                )

    @unittest.skipIf(
        PYTORCH_CUDA_MEMCHECK, "is_pinned uses failure to detect pointer property"
    )
    @onlyCUDA
    def test_pin_memory(self, device):
        nt_contiguous, nt_noncontiguous = random_nt_noncontiguous_pair((2, 3, 6, 7))
        for nt in [nt_contiguous, nt_noncontiguous]:
            self.assertFalse(nt.is_pinned())
            pinned = nt.pin_memory(device)
            self.assertTrue(pinned.is_pinned())
            self.assertEqual(nt, pinned)
            self.assertNotEqual(nt.data_ptr(), pinned.data_ptr())
            # test that pin_memory on already pinned tensor has no effect
            self.assertIs(pinned, pinned.pin_memory())
            self.assertEqual(pinned.data_ptr(), pinned.pin_memory().data_ptr())

    @torch.compiler.disable
    def _validate_nt(
        self,
        nt,
        device,
        dtype,
        layout,
        requires_grad,
        dim,
        batch_size,
        contiguous,
        cached_min_seqlen=None,
        cached_max_seqlen=None,
        base=None,
        ref_nt=None,
    ):
        # Validate a bunch of properties after NT construction.
        device = torch.device(device)
        self.assertEqual(nt.dim(), dim)
        self.assertEqual(nt.device, device)
        self.assertEqual(nt.dtype, dtype)
        self.assertEqual(nt.layout, layout)
        self.assertEqual(nt.requires_grad, requires_grad)
        self.assertEqual(nt.is_contiguous(), contiguous)

        if layout == torch.jagged:
            self.assertEqual(nt._values.device, device)
            self.assertEqual(nt._offsets.device, device)
            self.assertEqual(nt.shape[0], batch_size)
            self.assertTrue(isinstance(nt.shape[1], torch.SymInt))

            if base is not None:
                self.assertTrue(nt._is_view() and nt._base is base)
                replay_cache = nt._view_func(torch.randn_like(nt._base))._metadata_cache
                self.assertEqual(
                    "min_seqlen" in replay_cache, cached_min_seqlen is not None
                )
                self.assertEqual(
                    "max_seqlen" in replay_cache, cached_max_seqlen is not None
                )

            self.assertEqual(
                "min_seqlen" in nt._metadata_cache, cached_min_seqlen is not None
            )
            self.assertEqual(
                "max_seqlen" in nt._metadata_cache, cached_max_seqlen is not None
            )

            if cached_min_seqlen is not None:
                self.assertEqual(nt._min_seqlen, cached_min_seqlen)

            if cached_max_seqlen is not None:
                self.assertEqual(nt._max_seqlen, cached_max_seqlen)

        if ref_nt is not None:
            self.assertEqual(nt.size(0), ref_nt.size(0))
            for n1, n2 in zip(nt.unbind(), ref_nt.unbind()):
                self.assertEqual(n1, n2)

    @dtypes(torch.float, torch.double, torch.half)
    @parametrize("requires_grad", [False, True])
    @parametrize("components_require_grad", [False, True])
    def test_jagged_layout_construction_nested_tensor(
        self, device, dtype, requires_grad, components_require_grad
    ):
        for tensor_list in self._get_example_tensor_lists(
            include_list_of_lists=True, include_requires_grad=components_require_grad
        ):
            nt = torch.nested.nested_tensor(
                tensor_list,
                device=device,
                dtype=dtype,
                layout=torch.jagged,
                requires_grad=requires_grad,
            )

            expected_dim = torch.as_tensor(tensor_list[0]).dim() + 1
            expected_batch_size = len(tensor_list)
            expected_contiguous = True
            expected_min_seqlen = min(
                (torch.tensor(t) if isinstance(t, list) else t).shape[0]
                for t in tensor_list
            )
            expected_max_seqlen = max(
                (torch.tensor(t) if isinstance(t, list) else t).shape[0]
                for t in tensor_list
            )
            self._validate_nt(
                nt,
                device,
                dtype,
                torch.jagged,
                requires_grad,
                expected_dim,
                expected_batch_size,
                expected_contiguous,
                expected_min_seqlen,
                expected_max_seqlen,
            )

            # Make sure grads -don't- flow back into original tensors for nested_tensor()
            if requires_grad:
                (nt * 2).backward(torch.ones_like(nt))
            for t in tensor_list:
                t = t if isinstance(t, torch.Tensor) else torch.as_tensor(t)
                self.assertTrue(t.grad is None)

    @dtypes(torch.float, torch.double, torch.half)
    @parametrize("components_require_grad", [False, True])
    def test_jagged_layout_construction_as_nested_tensor(
        self, device, dtype, components_require_grad
    ):
        # NB: as_nested_tensor(tensor_list) doesn't support lists of lists for tensor_list
        for tensor_list in self._get_example_tensor_lists(
            include_list_of_lists=False, include_requires_grad=components_require_grad
        ):
            nt = torch.nested.as_nested_tensor(
                tensor_list, device=device, dtype=dtype, layout=torch.jagged
            )

            # nt.requires_grad=True should be set if at least one component requires grad
            expected_dim = tensor_list[0].dim() + 1
            expected_batch_size = len(tensor_list)
            expected_contiguous = True
            expected_min_seqlen = min(
                (torch.tensor(t) if isinstance(t, list) else t).shape[0]
                for t in tensor_list
            )
            expected_max_seqlen = max(
                (torch.tensor(t) if isinstance(t, list) else t).shape[0]
                for t in tensor_list
            )
            self._validate_nt(
                nt,
                device,
                dtype,
                torch.jagged,
                components_require_grad,
                expected_dim,
                expected_batch_size,
                expected_contiguous,
                expected_min_seqlen,
                expected_max_seqlen,
            )

            # Make sure grads flow back into original tensors for as_nested_tensor()
            if components_require_grad:
                (nt * 2).backward(torch.ones_like(nt))
                for t in tensor_list:
                    if t.requires_grad:
                        self.assertEqual(t.grad, torch.ones_like(t) * 2)
                    else:
                        self.assertTrue(t.grad is None)

    @xfailIfTorchDynamo
    @unittest.skipIf(
        PYTORCH_CUDA_MEMCHECK, "is_pinned uses failure to detect pointer property"
    )
    @onlyCUDA
    def test_jagged_layout_construction_with_pinned_memory(self, device):
        for tensor_list in self._get_example_tensor_lists():
            nt = torch.nested.nested_tensor(
                tensor_list, layout=torch.jagged, device="cpu", pin_memory=True
            )

            expected_dim = torch.as_tensor(tensor_list[0]).dim() + 1
            expected_batch_size = len(tensor_list)
            expected_min_seqlen = min(
                (torch.tensor(t) if isinstance(t, list) else t).shape[0]
                for t in tensor_list
            )
            expected_max_seqlen = max(
                (torch.tensor(t) if isinstance(t, list) else t).shape[0]
                for t in tensor_list
            )
            self._validate_nt(
                nt,
                device="cpu",
                dtype=torch.float32,
                layout=torch.jagged,
                requires_grad=False,
                dim=expected_dim,
                batch_size=expected_batch_size,
                contiguous=True,
                cached_min_seqlen=expected_min_seqlen,
                cached_max_seqlen=expected_max_seqlen,
            )
            self.assertTrue(nt.is_pinned())

    @dtypes(torch.float, torch.double, torch.half)
    @parametrize("requires_grad", [False, True])
    @parametrize("values_is_view", [False, True])
    def test_jagged_view_from_values_offsets(
        self, device, dtype, requires_grad, values_is_view
    ):
        if values_is_view:
            # make values a view of base
            base = torch.randn(
                2, 3, 4, 5, 6, device=device, dtype=dtype, requires_grad=requires_grad
            )
            values = base.flatten(0, -2)
        else:
            values = torch.randn(
                10, 5, device=device, dtype=dtype, requires_grad=requires_grad
            )
        offsets = torch.tensor([0, 2, 4, 6, 10], device=device, dtype=torch.int64)

        nt = nested_view_from_values_offsets(values, offsets)

        expected_dim = values.dim() + 1
        expected_batch_size = offsets.shape[0] - 1
        expected_base = base if values_is_view else values
        lengths = offsets.diff()
        self._validate_nt(
            nt,
            device,
            dtype,
            torch.jagged,
            requires_grad,
            expected_dim,
            expected_batch_size,
            # ensure NT is a proper view
            base=expected_base,
            contiguous=True,
            # if no min / max are passed, expect the metadata cache to be empty
            cached_min_seqlen=None,
            cached_max_seqlen=None,
        )

        if requires_grad:
            # Make sure grads flow back
            (nt * 2).backward(torch.ones_like(nt))

            @torch.compiler.disable
            def _check_grad(t):
                self.assertTrue(t.grad is not None)
                self.assertEqual(t.grad, torch.ones_like(t) * 2)

            _check_grad(base if values_is_view else values)

    @dtypes(torch.float)
    @parametrize("pass_min_max", [False, True])
    def test_nested_tensor_from_jagged(self, device, dtype, pass_min_max):
        # === construct from (values, offsets) ===
        values = torch.randn(10, 5, device=device, dtype=dtype)
        offsets = torch.tensor([0, 2, 4, 6, 10], device=device, dtype=torch.int64)

        # compute min / max seqlen
        lengths = offsets.diff()
        min_seqlen = lengths.min().item()
        max_seqlen = lengths.max().item()

        if pass_min_max:
            nt = torch.nested.nested_tensor_from_jagged(
                values, offsets=offsets, min_seqlen=min_seqlen, max_seqlen=max_seqlen
            )
        else:
            nt = torch.nested.nested_tensor_from_jagged(values, offsets=offsets)
        self._validate_nt(
            nt,
            device,
            dtype,
            torch.jagged,
            requires_grad=False,
            dim=3,
            batch_size=4,
            contiguous=True,
            cached_min_seqlen=(min_seqlen if pass_min_max else None),
            cached_max_seqlen=(max_seqlen if pass_min_max else None),
            base=values,
        )

        # === construct from (values, offsets, lengths) ===
        lengths = torch.tensor([2, 1, 1, 2], device=device)

        # compute min / max seqlen
        min_seqlen = lengths.min().item()
        max_seqlen = lengths.max().item()

        if pass_min_max:
            nt = torch.nested.nested_tensor_from_jagged(
                values,
                offsets=offsets,
                lengths=lengths,
                min_seqlen=min_seqlen,
                max_seqlen=max_seqlen,
            )
        else:
            nt = torch.nested.nested_tensor_from_jagged(
                values, offsets=offsets, lengths=lengths
            )

        # when both offsets / lengths are specified, expect non-contiguous
        self._validate_nt(
            nt,
            device,
            dtype,
            torch.jagged,
            requires_grad=False,
            dim=3,
            batch_size=4,
            contiguous=False,
            cached_min_seqlen=(min_seqlen if pass_min_max else None),
            cached_max_seqlen=(max_seqlen if pass_min_max else None),
            base=values,
        )
        self.assertIs(nt.lengths(), lengths)

        # === construct from (values, lengths) ===
        values = torch.randn(14, 5, device=device, dtype=dtype)
        lengths = torch.tensor([2, 3, 4, 5], device=device)

        # compute min / max seqlen
        min_seqlen = lengths.min().item()
        max_seqlen = lengths.max().item()

        if pass_min_max:
            nt = torch.nested.nested_tensor_from_jagged(
                values, lengths=lengths, min_seqlen=min_seqlen, max_seqlen=max_seqlen
            )
        else:
            nt = torch.nested.nested_tensor_from_jagged(values, lengths=lengths)

        # for now, if only lengths is specified, convert to offsets to integrate best with the
        # existing kernels
        expected_offsets = torch.tensor([0, 2, 5, 9, 14], device=device)
        expected_nt = torch.nested.nested_tensor_from_jagged(
            values, offsets=expected_offsets
        )
        self._validate_nt(
            nt,
            device,
            dtype,
            torch.jagged,
            requires_grad=False,
            dim=3,
            batch_size=4,
            contiguous=True,
            cached_min_seqlen=(min_seqlen if pass_min_max else None),
            cached_max_seqlen=(max_seqlen if pass_min_max else None),
            base=values,
            ref_nt=expected_nt,
        )

        # error case: no offsets or lengths
        with self.assertRaisesRegex(
            RuntimeError, "At least one of offsets or lengths is required"
        ):
            torch.nested.nested_tensor_from_jagged(values, offsets=None, lengths=None)

    @onlyCPU
    def test_nested_tensor_from_jagged_fx_trace(self, device):
        def fn(x, y):
            return torch.nested.nested_tensor_from_jagged(x, y)

        def user_unwrapped(x, y):
            return fn(x, y)

        with self.assertRaisesRegex(
            RuntimeError,
            "torch.nested.nested_tensor_from_jagged does not support tracing with fx.symbolic_trace",
        ):
            torch.fx.symbolic_trace(user_unwrapped)

    @dtypes(torch.float, torch.double, torch.half)
    @parametrize("dim", range(5))
    @parametrize(
        "layout",
        [torch.strided, torch.jagged],
        name_fn=lambda l: f"layout_{str(l).split('.')[1]}",
    )
    @parametrize("requires_grad", [False, True])
    @parametrize("contiguous", [False, True])
    def test_as_nested_tensor_from_tensor(
        self, device, dtype, dim, layout, requires_grad, contiguous
    ):
        if dim == 0:
            t = torch.tensor(3.0, requires_grad=requires_grad)
        else:
            t = torch.randn(*(3 for _ in range(dim)), requires_grad=requires_grad)
        assert t.dim() == dim

        if dim < 2:
            # 0-1 dim tensors can't be converted to NTs
            with self.assertRaisesRegex(
                RuntimeError, "Expected tensor argument to have dim"
            ):
                nt = torch.nested.as_nested_tensor(
                    t, device=device, dtype=dtype, layout=layout
                )
            return

        orig_t = t
        if not contiguous:
            t = t.transpose(0, 1)

        nt = torch.nested.as_nested_tensor(t, device=device, dtype=dtype, layout=layout)
        expected_dim = t.dim()
        expected_batch_size = t.size(0)
        expected_seqlen = t.size(1) if layout == torch.jagged else None
        self._validate_nt(
            nt,
            device,
            dtype,
            layout,
            requires_grad=requires_grad,
            dim=dim,
            batch_size=expected_batch_size,
            contiguous=True,
            cached_min_seqlen=expected_seqlen,
            cached_max_seqlen=expected_seqlen,
        )

        if torch.device(device) == t.device and dtype == t.dtype and contiguous:
            # should be the non-copying (view) case
            self.assertTrue(nt._is_view() and nt._base is t)

        # should have equivalent components to construction from unbound tensor list
        nt_from_unbind = torch.nested.as_nested_tensor(
            list(t.unbind(0)), device=device, dtype=dtype, layout=layout
        )
        self.assertEqualIgnoringNestedInts(nt, nt_from_unbind)

        # ensure call on a NT with the same properties returns the NT directly
        nt2 = torch.nested.as_nested_tensor(
            nt, device=device, dtype=dtype, layout=layout
        )
        self.assertTrue(nt is nt2)

        # ensure call with device=None uses input tensor device
        nt3 = torch.nested.as_nested_tensor(
            t.to(device=device, dtype=dtype),
            device=None,
            dtype=None,
            layout=layout,
        )
        self._validate_nt(
            nt3,
            device,
            dtype,
            layout,
            requires_grad=requires_grad,
            dim=dim,
            batch_size=expected_batch_size,
            contiguous=True,
            cached_min_seqlen=expected_seqlen,
            cached_max_seqlen=expected_seqlen,
        )

        # we don't support conversion between layouts this way atm
        other_layout = torch.strided if layout == torch.jagged else torch.jagged
        with self.assertRaisesRegex(
            RuntimeError, "Converting between nested tensor layouts is not supported"
        ):
            torch.nested.as_nested_tensor(
                nt, device=device, dtype=dtype, layout=other_layout
            )

        if requires_grad:
            # make sure gradients flow back into inputs
            (nt * 2).backward(torch.ones_like(nt))
            self.assertEqual(orig_t.grad, torch.ones_like(orig_t) * 2)

    @dtypes(torch.float32)
    def test_construction_from_list(self, device, dtype):
        from torch.fx.experimental.symbolic_shapes import is_nested_int

        # success case: single ragged dim anywhere but the batch dim
        for nt_dim in [2, 3, 4]:
            for ragged_dim in range(1, nt_dim):
                B = 6
                shapes = [list(range(3, 3 + nt_dim - 1)) for _ in range(B)]
                for b in range(B):
                    # subtract 1 to convert to component dim space
                    shapes[b][ragged_dim - 1] = torch.randint(
                        2, 9, (1,), device=device, dtype=torch.int64
                    ).item()

                components = [
                    torch.randn(shape, device=device, dtype=dtype) for shape in shapes
                ]
                nt = torch.nested.nested_tensor(components, layout=torch.jagged)

                self.assertEqual(nt.dim(), nt_dim)
                self.assertEqual(nt._ragged_idx, ragged_dim)
                for d in range(nt_dim):
                    self.assertEqual(d == ragged_dim, is_nested_int(nt.shape[d]))

        # error case: empty list
        with self.assertRaisesRegex(
            RuntimeError, "Cannot construct a nested tensor from an empty tensor list"
        ):
            torch.nested.nested_tensor([], layout=torch.jagged)

        # error case: list of zero-dim tensors
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot construct a nested tensor from a list of zero-dim tensors",
        ):
            torch.nested.nested_tensor(
                [
                    torch.tensor(3.0, device=device, dtype=dtype),
                    torch.tensor(4.0, device=device, dtype=dtype),
                    torch.tensor(5.0, device=device, dtype=dtype),
                ],
                layout=torch.jagged,
            )

        # error case: multiple ragged dims
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot represent given tensor list as a nested tensor with the jagged layout",
        ):
            torch.nested.nested_tensor(
                [
                    torch.randn(2, 3, device=device, dtype=dtype),
                    torch.randn(4, 5, device=device, dtype=dtype),
                ],
                layout=torch.jagged,
            )

        # error case: components on multiple devices
        if "cuda" in device:
            with self.assertRaisesRegex(
                RuntimeError,
                "When constructing a nested tensor, all tensors in list must be on the same device",
            ):
                torch.nested.nested_tensor(
                    [
                        torch.randn(2, 3, device=device, dtype=dtype),
                        torch.randn(2, 4, device="cpu", dtype=dtype),
                    ],
                    layout=torch.jagged,
                )

        # error case: components with multiple dtypes
        with self.assertRaisesRegex(
            RuntimeError,
            "When constructing a nested tensor, all tensors in list must have the same dtype",
        ):
            torch.nested.nested_tensor(
                [
                    torch.randn(2, 3, device=device, dtype=dtype),
                    torch.randn(2, 4, device=device, dtype=torch.float64),
                ],
                layout=torch.jagged,
            )

        # error case: components with multiple dims
        with self.assertRaisesRegex(
            RuntimeError,
            "When constructing a nested tensor, all tensors in list must have the same dim",
        ):
            torch.nested.nested_tensor(
                [
                    torch.randn(2, 3, device=device, dtype=dtype),
                    torch.randn(2, 3, 4, device=device, dtype=dtype),
                ],
                layout=torch.jagged,
            )

    @dtypes(torch.double, torch.half)
    @onlyCUDA
    def test_device_dtype_transfer_updates_offsets(self, device, dtype):
        for tensor_list in self._get_example_tensor_lists():
            orig_device = torch.device("cpu")
            orig_dtype = torch.float32
            nt = torch.nested.nested_tensor(
                tensor_list, layout=torch.jagged, device=orig_device, dtype=orig_dtype
            )

            self.assertEqual(torch.int64, nt.offsets().dtype)
            nt = nt.to(device=device).to(dtype=dtype)

            # offsets should still be int64 on the new device
            self.assertEqual(nt.values().device, nt.offsets().device)
            self.assertEqual(torch.int64, nt.offsets().dtype)

    def test_unbind(self, device):
        for tensor_list in self._get_example_tensor_lists():
            nt = torch.nested.nested_tensor(
                tensor_list, layout=torch.jagged, device=device
            )  # ragged_idx = 1
            out = nt.unbind()
            self.assertEqual(len(out), len(tensor_list))
            for i, t in enumerate(out):
                self.assertEqual(t, tensor_list[i])

    @parametrize("ragged_idx", [2, 3])
    def test_unbind_transpose(self, device, ragged_idx):
        for tensor_list in self._get_example_tensor_lists():
            nt = torch.nested.nested_tensor(
                tensor_list, layout=torch.jagged, device=device
            )
            if ragged_idx < nt.dim():
                nt = nt.transpose(1, ragged_idx)  # set ragged_idx
                out = nt.unbind()
                self.assertEqual(len(out), len(tensor_list))
                for i, t in enumerate(out):
                    self.assertEqual(
                        t.transpose(0, ragged_idx - 1), tensor_list[i]
                    )  # transpose back each element of result

    def test_unbind_transpose_ragged_idx_last_dim(self, device):
        for tensor_list in self._get_example_tensor_lists():
            nt = torch.nested.nested_tensor(
                tensor_list, layout=torch.jagged, device=device
            ).transpose(1, -1)  # set ragged_idx = last dimension
            out = nt.unbind()
            self.assertEqual(len(out), len(tensor_list))
            for i, t in enumerate(out):
                self.assertEqual(
                    t.transpose(0, -1), tensor_list[i]
                )  # transpose back each element of result

    def test_unbind_lengths(self, device):
        values = torch.randn(16, 128, device=device)
        offsets = torch.tensor([0, 8, 12, 13, 16], device=device)
        lengths = torch.tensor([6, 2, 1, 2], device=device)
        nt = torch.nested.nested_tensor_from_jagged(
            values, offsets=offsets, lengths=lengths
        )  # 3D nested tensor

        tensor_list = []
        for i in range(offsets.shape[0] - 1):
            tensor_list.append(values[offsets[i] : (offsets[i] + lengths[i])])

        out = nt.unbind()
        self.assertEqual(len(out), len(tensor_list))
        for i, t in enumerate(out):
            self.assertEqual(t, tensor_list[i])

    def test_unbind_lengths_ragged_idx_1(self, device):
        values = torch.randn(16, 8, 128, device=device)
        offsets = torch.tensor([0, 8, 12, 13, 16], device=device)
        lengths = torch.tensor([6, 2, 1, 2], device=device)
        ragged_idx = 1
        nt = torch.nested._internal.nested_tensor.NestedTensor(
            values, offsets=offsets, lengths=lengths, _ragged_idx=ragged_idx
        )  # 4D nested tensor

        tensor_list = []
        for i in range(offsets.shape[0] - 1):
            tensor_list.append(values[offsets[i] : (offsets[i] + lengths[i]), :, :])

        out = nt.unbind()

        self.assertEqual(len(out), len(tensor_list))
        for i, t in enumerate(out):
            self.assertEqual(t, tensor_list[i])

    def test_unbind_lengths_ragged_idx_equals_2_bad_dim(self, device):
        values = torch.randn(16, 8, 128, device=device)
        offsets = torch.tensor([0, 8, 12, 13, 16], device=device)
        lengths = torch.tensor([6, 2, 1, 2], device=device)
        ragged_idx = 2
        nt = torch.nested._internal.nested_tensor.NestedTensor(
            values, offsets=offsets, lengths=lengths, _ragged_idx=ragged_idx
        )  # 4D nested tensor

        self.assertRaisesRegex(
            RuntimeError,
            r"unbind\(\): nested tensor offsets and lengths.*",
            lambda: nt.unbind(),
        )

    def test_unbind_lengths_ragged_idx_2(self, device):
        values = torch.randn(16, 8, 128, device=device)
        offsets = torch.tensor([0, 2, 4, 8], device=device)
        lengths = torch.tensor([2, 1, 3], device=device)
        ragged_idx = 2
        nt = torch.nested._internal.nested_tensor.NestedTensor(
            values, offsets=offsets, lengths=lengths, _ragged_idx=ragged_idx
        )  # 4D nested tensor

        tensor_list = []
        for i in range(offsets.shape[0] - 1):
            tensor_list.append(values[:, offsets[i] : (offsets[i] + lengths[i]), :])

        out = nt.unbind()

        self.assertEqual(len(out), len(tensor_list))
        for i, t in enumerate(out):
            self.assertEqual(t, tensor_list[i])

    def test_unbind_lengths_ragged_idx_3(self, device):
        values = torch.randn(16, 8, 128, device=device)
        offsets = torch.tensor([0, 100, 128], device=device)
        lengths = torch.tensor([50, 28], device=device)
        ragged_idx = 3
        nt = torch.nested._internal.nested_tensor.NestedTensor(
            values, offsets=offsets, lengths=lengths, _ragged_idx=ragged_idx
        )  # 4D nested tensor

        tensor_list = []
        for i in range(offsets.shape[0] - 1):
            tensor_list.append(values[:, :, offsets[i] : (offsets[i] + lengths[i])])

        out = nt.unbind()

        self.assertEqual(len(out), len(tensor_list))
        for i, t in enumerate(out):
            self.assertEqual(t, tensor_list[i])

    @skipIfTorchDynamo(
        "TorchDynamo raises an error for ragged_idx == 0 earlier than Torch"
    )
    def test_unbind_lengths_ragged_idx_0(self, device):
        values = torch.randn(16, 8, 128, device=device)
        offsets = torch.tensor([0, 100, 128], device=device)
        lengths = torch.tensor([50, 28], device=device)
        ragged_idx = 0
        nt = torch.nested._internal.nested_tensor.NestedTensor(
            values, offsets=offsets, lengths=lengths, _ragged_idx=ragged_idx
        )  # 4D nested tensor

        tensor_list = []
        for i in range(offsets.shape[0] - 1):
            tensor_list.append(values[:, :, offsets[i] : (offsets[i] + lengths[i])])

        self.assertRaisesRegex(
            RuntimeError,
            r"unbind\(\): nested tensor.*out of bounds",
            lambda: nt.unbind(),
        )

    def test_narrow(self, device):
        starts = torch.tensor([0, 1, 2, 3, 4], device=device, dtype=torch.int64)
        lengths = torch.tensor([3, 2, 2, 1, 5], device=device, dtype=torch.int64)
        buffer = (
            torch.arange(0, 10, device=device, dtype=torch.int64)
            .unsqueeze(0)
            .expand(5, -1)
            .clone()
            .detach()
        )
        nt = torch.nested.narrow(buffer, 1, starts, lengths, layout=torch.jagged)

        self.assertTrue(nt._is_view() and nt._base is buffer)

        # TODO: Use this approach when unbind is functional
        # unbinded_nt = nt.unbind()
        # for i in range(starts.shape[0]):
        #     self.assertEqual(torch.arange(starts[i], starts[i] + lengths[i], device=device, dtype=torch.int64), unbinded_nt[i])
        for i in range(starts.shape[0]):
            self.assertEqual(
                torch.arange(
                    starts[i], starts[i] + lengths[i], device=device, dtype=torch.int64
                ),
                nt.values()[nt.offsets()[i] : (nt.offsets()[i] + nt.lengths()[i])],
            )

    def test_njt_cat(self, device):
        offsets = torch.tensor([0, 2, 3], device=device, dtype=torch.int64)
        values_1 = torch.randn(
            3, 2, dtype=torch.float64, device=device, requires_grad=True
        )
        values_2 = torch.randn(
            3, 4, dtype=torch.float64, device=device, requires_grad=True
        )

        def grad_test_func(values_1, values_2, offsets):
            nt_1 = torch.nested.nested_tensor_from_jagged(values_1, offsets)
            nt_2 = torch.nested.nested_tensor_from_jagged(values_2, offsets)
            nt_3 = torch.cat([nt_1, nt_2], dim=-1)
            return nt_3.values()

        assert gradcheck(
            grad_test_func,
            inputs=(values_1, values_2, offsets),
            check_batched_grad=False,
        )

    def test_is_contiguous(self, device):
        a = torch.randn(2, 3, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(3, 3, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(4, 3, requires_grad=True, dtype=torch.float64, device=device)
        nt_contiguous = torch.nested.as_nested_tensor([a, b, c], layout=torch.jagged)

        starts_nc = torch.tensor([0, 1, 2, 3, 4], device=device, dtype=torch.int64)
        lengths_nc = torch.tensor([3, 2, 2, 1, 5], device=device, dtype=torch.int64)
        narrow_base = (
            torch.arange(0, 10, device=device, dtype=torch.int64)
            .unsqueeze(0)
            .expand(5, -1)
            .clone()
        )
        nt_noncontiguous = torch.nested.narrow(
            narrow_base, 1, starts_nc, lengths_nc, layout=torch.jagged
        )

        starts_c = torch.tensor([1, 0, 0, 0, 0], device=device, dtype=torch.int64)
        lengths_c = torch.tensor([9, 10, 10, 10, 8], device=device, dtype=torch.int64)
        nt_contiguous_narrow = torch.nested.narrow(
            narrow_base, 1, starts_c, lengths_c, layout=torch.jagged
        )

        # Test contiguous case
        assert nt_contiguous.is_contiguous()

        # Test narrow case
        assert not nt_noncontiguous.is_contiguous()
        assert nt_contiguous_narrow.is_contiguous()

        # Test querying by memory_format
        self.assertTrue(
            nt_contiguous.is_contiguous(memory_format=torch.contiguous_format)
        )
        self.assertTrue(
            not nt_noncontiguous.is_contiguous(memory_format=torch.contiguous_format)
        )
        self.assertTrue(
            nt_contiguous_narrow.is_contiguous(memory_format=torch.contiguous_format)
        )

    def test_layout_under_torch_dispatch_mode(self):
        from torch.testing._internal.logging_tensor import (
            capture_logs_with_logging_tensor_mode,
        )

        nt = random_nt_from_dims(
            [2, None, 3], torch.device("cpu"), torch.float32, layout=torch.jagged
        )

        with capture_logs_with_logging_tensor_mode():
            self.assertEqual(nt.layout, torch.jagged)

    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    @parametrize(
        "func", [torch.empty_like, torch.randn_like], name_fn=lambda f: f.__name__
    )
    def test_like_shape(self, func):
        nt = random_nt_from_dims(
            [2, None, 3], torch.device("cpu"), torch.float32, layout=torch.jagged
        )
        nt_like = func(nt)

        for nt_ub in nt_like.unbind():
            t_like = func(nt_ub)
            self.assertEqual(nt_ub.shape, t_like.shape)

    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    @parametrize(
        "func", [torch.ones_like, torch.zeros_like], name_fn=lambda f: f.__name__
    )
    def test_like_value(self, func):
        nt = random_nt_from_dims(
            [2, None, 3], torch.device("cpu"), torch.float32, layout=torch.jagged
        )
        nt_like = func(nt)

        for nt_ub in nt_like.unbind():
            t_like = func(nt_ub)
            self.assertEqual(nt_ub, t_like)

    def test_noncontiguous_pointwise(self, device):
        a = torch.randn(2, 3, 4, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(3, 3, 4, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(4, 3, 4, requires_grad=True, dtype=torch.float64, device=device)
        nt = torch.nested.nested_tensor([a, b, c], layout=torch.jagged)
        # transpose ragged dim
        transposed = nt.transpose(1, 2)
        self.assertFalse(transposed.is_contiguous())
        clone = transposed.clone()

        def check_nt_equality(x, y):
            self.assertEqual(x.values(), y.values())
            self.assertEqual(x.offsets(), y.offsets())
            self.assertEqual(x._ragged_idx, y._ragged_idx)
            self.assertEqual(x.shape, y.shape)

        self.assertFalse(clone.is_contiguous())
        check_nt_equality(clone, transposed)

        clone_contig = transposed.clone(memory_format=torch.contiguous_format)
        self.assertTrue(clone_contig.is_contiguous())
        check_nt_equality(clone_contig, transposed)

        detached = transposed.detach()
        self.assertFalse(clone.is_contiguous())
        check_nt_equality(detached, transposed)

    def test_permute(self, device):
        nt = random_nt_from_dims(
            [2, None, 3, 5], device, torch.float32, layout=torch.jagged
        )
        nt_shape = nt.shape
        nt_inner_shape = nt.values().shape
        with self.assertRaisesRegex(
            ValueError,
            r"permute\(\): number of dimensions in the tensor input \(4\) "
            + r"does not match the length of the desired ordering of dimensions \(3\).",
        ):
            nt.permute(0, 2, 1)
        with self.assertRaisesRegex(
            ValueError, r"permute\(\): duplicate dims are not allowed."
        ):
            nt.permute(0, 2, -2, 3)
        with self.assertRaisesRegex(
            ValueError, "Permute is not supported on the batch dimension for jagged NT"
        ):
            nt.permute(1, 0, 2, 3)
        nt_permute = nt.permute(0, 2, 1, -1)
        self.assertEqual(
            nt_permute.shape, (nt_shape[0], nt_shape[2], nt_shape[1], nt_shape[3])
        )
        self.assertEqual(
            nt_permute.values().shape,
            (nt_inner_shape[1], nt_inner_shape[0], nt_inner_shape[2]),
        )
        self.assertEqual(nt_permute._ragged_idx, 2)
        self.assertEqual(nt_permute.permute(0, 2, 1, 3), nt)

    def test_to_dtype(self, device):
        nt = random_nt_from_dims(
            [2, None, 3], device, torch.float32, layout=torch.jagged
        )
        nt_after = nt.to(torch.float64)
        self.assertEqual(torch.float32, nt.dtype)
        self.assertEqual(torch.float64, nt_after.dtype)
        self.assertEqual(torch.float64, nt_after.values().dtype)
        self.assertEqual(torch.int64, nt_after.offsets().dtype)

        noncontiguous_nt = nt.transpose(1, 2)
        noncontiguous_nt_after = noncontiguous_nt.to(torch.bfloat16)
        self.assertEqual(torch.bfloat16, noncontiguous_nt_after.dtype)
        self.assertEqual(torch.bfloat16, noncontiguous_nt_after.values().dtype)
        self.assertEqual(torch.int64, noncontiguous_nt_after.offsets().dtype)

    def test_to_copy(self, device):
        nt = torch.nested.nested_tensor(
            [
                torch.randn(
                    i + 2, 3, 4, requires_grad=True, dtype=torch.float64, device=device
                )
                for i in range(3)
            ],
            layout=torch.jagged,
        )

        nt_copy_dtype = torch.ops.aten._to_copy(nt, dtype=torch.float16)
        self.assertEqual(torch.float16, nt_copy_dtype.dtype)

        nt_t = nt.transpose(1, 2)
        nt_t_copy_dtype = torch.ops.aten._to_copy(nt_t, dtype=torch.float16)
        self.assertEqual(torch.float16, nt_t_copy_dtype.dtype)

    def test_copy_(self, device):
        offsets = torch.tensor([0, 2, 4], device=device)
        a = torch.nested.nested_tensor_from_jagged(
            torch.zeros(4, 3, device=device), offsets
        )
        b = torch.nested.nested_tensor_from_jagged(
            torch.ones(4, 3, device=device), offsets
        )
        a.copy_(b)
        torch._dynamo.disable(self.assertEqual)(a, b)

        offsets_2 = torch.tensor([0, 2, 4], device=device)
        c = torch.nested.nested_tensor_from_jagged(
            torch.ones(4, 3, device=device), offsets_2
        )
        # fail when tensors have the same size but not the exact same offset tensor.
        with self.assertRaisesRegex(
            RuntimeError,
            "copy_ only supports Nested Tensors that have same size and the exact same offset tensor.",
        ):
            a.copy_(c)

        # fail when tensors have different sizes
        a = a.transpose(1, 2)
        with self.assertRaisesRegex(
            RuntimeError,
            "copy_ only supports Nested Tensors that have same size and the exact same offset tensor.",
        ):
            a.copy_(b)

    # This can't happen in the opinfo tests due to subprocess creation
    @unittest.skipIf(
        TEST_WITH_ROCM,
        "In ROCm, kernel asserts are disabled due to performance overhead",
    )
    def test_index_put_error(self, device):
        import subprocess

        with self.subTest():
            r = subprocess.call(
                [
                    sys.executable,
                    "-c",
                    """\
import torch
offsets = torch.tensor([0, 2, 5, 7], device='cuda')
lengths = torch.tensor([2, 2, 2], device='cuda')
indices = [
    torch.tensor([0, 1, 2], device='cuda'),
    torch.tensor([0, 2, 1], device='cuda'),
    torch.tensor([0, 0, 0], device='cuda'),
]
a = torch.nested.nested_tensor_from_jagged(
    torch.zeros(7, 3, device='cuda'), offsets, lengths
)
a[indices] = 1.0
torch.cuda.synchronize()
""",
                ]
            )
            self.assertTrue(r != 0)

    @skipIfTorchDynamo("Dynamo doesn't know how to trace prof.events()")
    def test_profiler_sequence_nr(self):
        with torch.profiler.profile() as prof:
            values = torch.randn(4, 6, requires_grad=True)
            offsets = torch.tensor([0, 2, 4])
            values = values * 2
            l = torch.nn.Linear(6, 8)
            nt = torch.nested.nested_tensor_from_jagged(values, offsets)

            nt = l(nt)
            val = nt.values()

            loss = val.sum()
            loss.backward()

        fwd_seq_nrs = []
        for evt in prof.events():
            if (
                "linear" in evt.name.lower()
                and "backward" not in evt.name.lower()
                and evt.sequence_nr != -1
            ):
                fwd_seq_nrs.append(evt.sequence_nr)

        bwd_seq_nrs = []
        for evt in prof.events():
            if (
                "linear" in evt.name.lower()
                and "backward" in evt.name.lower()
                and "evaluate_function" not in evt.name.lower()
                and evt.sequence_nr != -1
            ):
                bwd_seq_nrs.append(evt.sequence_nr)

        # There should only be one such event with a sequence number:
        # the PythonTLSSnapshot event - but, note that it's not terrible if
        # we end up with multiple events with the same sequence number - so we
        # could relax this check if it becomes inconvenient to maintain this
        # property.
        self.assertEqual(len(fwd_seq_nrs), 1)
        self.assertEqual(len(bwd_seq_nrs), 1)
        self.assertEqual(fwd_seq_nrs[0], bwd_seq_nrs[0])

    def test_is_same_size(self, device):
        def get_3_tensors():
            return [
                torch.randn(
                    i + 2, 3, 4, requires_grad=True, dtype=torch.float64, device=device
                )
                for i in range(3)
            ]

        nt1, offsets1 = jagged_from_list(get_3_tensors(), None)
        nt2, offsets1 = jagged_from_list(get_3_tensors(), offsets1)

        nt3, offsets2 = jagged_from_list(get_3_tensors(), None)
        nt4, offsets2 = jagged_from_list(get_3_tensors(), offsets2)

        def check_size(nt1, nt2, nt3, nt4):
            self.assertTrue(torch.ops.aten.is_same_size(nt1, nt2))
            self.assertTrue(torch.ops.aten.is_same_size(nt3, nt4))
            self.assertFalse(torch.ops.aten.is_same_size(nt1, nt3))

        check_size(nt1, nt2, nt3, nt4)

        nt1_t, nt2_t, nt3_t, nt4_t = (x.transpose(1, 2) for x in (nt1, nt2, nt3, nt4))
        check_size(nt1_t, nt2_t, nt3_t, nt4_t)

    @skipIfTorchDynamo("compiles internally")
    @unittest.skipIf(IS_WINDOWS, reason="Windows not yet supported for torch.compile")
    @skipCUDAIf(not SM70OrLater, "GPU capability is < SM70")
    def test_specialize_dynamic_shape(self, device):
        values = torch.randn((18, 16), device=device)
        offsets = torch.tensor([0, 2, 3, 6, 15, 18], device=device)
        like_values = torch.randn_like(values)

        # this marks values as dynamic
        nt = torch.nested.nested_tensor_from_jagged(values, offsets)

        def fn(values, same_size):
            # here, the dynamic shape is specialized by same_size's shape
            # https://github.com/pytorch/pytorch/issues/127097
            # make sure this doesn't error out in torch.compile
            return values + same_size

        self.assertEqual(
            fn(values, like_values),
            torch.compile(fn)(values, like_values),
        )

    @skipIfTorchDynamo("compiles internally")
    @unittest.skipIf(IS_WINDOWS, reason="Windows not yet supported for torch.compile")
    @skipCUDAIf(not SM70OrLater, "GPU capability is < SM70")
    def test_specialize_dynamic_shape_recompile(self, device):
        def generate_inp(total_len):
            values = torch.randn((total_len, 16), device=device)
            offsets = torch.tensor([0, 2, 3, 6, 15, total_len], device=device)
            like_values = torch.randn_like(values)
            return values, offsets, like_values

        def check_results(ref_fn, res_fn, args):
            values, offsets, like_values = args
            # this may add dynamic shape markings
            # goal of this test is to make sure that whatever markings are there,
            # we eventually stop recompiling as shape changes.
            nt = torch.nested.nested_tensor_from_jagged(values, offsets)

            self.assertEqual(ref_fn(values, like_values), res_fn(values, like_values))

        def fn(values, same_size):
            return values + same_size

        compile_counter = torch._dynamo.testing.CompileCounter()

        compiled_fn = torch.compile(fn, backend=compile_counter, fullgraph=True)
        check_results(fn, compiled_fn, generate_inp(18))
        self.assertEqual(compile_counter.frame_count, 1)

        check_results(fn, compiled_fn, generate_inp(19))
        # we'll probably recompile here with dynamic shapes - it's okay if not though.
        frame_count_2 = compile_counter.frame_count
        self.assertIn(frame_count_2, [1, 2])

        # make sure that by now we've already compiled with dynamic shapes, so additional
        # shapes should not trigger additional recompiles.
        check_results(fn, compiled_fn, generate_inp(20))
        self.assertEqual(compile_counter.frame_count, frame_count_2)

    # Note 1: Math fallback doesn't work with bfloat16 on CUDA
    # Note 2: ROCm doesn't support flash attention or mem_efficient attention for NT
    @unittest.skipIf(
        TEST_WITH_ROCM,
        "ROCm doesn't support flash attention or mem_efficient attention for NT",
    )
    @dtypes(
        *(
            [torch.float16, torch.bfloat16, torch.float32]
            if SM80OrLater
            else [torch.float16, torch.float32]
        )
    )
    def test_sdpa(self, device, dtype):
        batch_size = 1
        emb_dims = 128
        n_heads = 8
        head_dims = emb_dims // n_heads

        sen1 = torch.randn(11, emb_dims, dtype=dtype, device=device)
        sen2 = torch.randn(13, emb_dims, dtype=dtype, device=device)

        query = torch.nn.Linear(
            emb_dims, emb_dims, bias=False, device=device, dtype=dtype
        )
        key = torch.nn.Linear(
            emb_dims, emb_dims, bias=False, device=device, dtype=dtype
        )
        value = torch.nn.Linear(
            emb_dims, emb_dims, bias=False, device=device, dtype=dtype
        )

        # Simplest case: 1 sentence, no batching
        x_d1 = sen1.unsqueeze(0)
        x_nt = torch.nested.as_nested_tensor([sen1], layout=torch.jagged)

        # See note below for why we detach here.
        q_d1 = (
            query(x_d1)
            .view(batch_size, -1, n_heads, head_dims)
            .detach()
            .requires_grad_(True)
        )
        q_d1_t = q_d1.transpose(1, 2)
        k_d1 = (
            key(x_d1)
            .view(batch_size, -1, n_heads, head_dims)
            .detach()
            .requires_grad_(True)
        )
        k_d1_t = k_d1.transpose(1, 2)
        v_d1 = (
            value(x_d1)
            .view(batch_size, -1, n_heads, head_dims)
            .detach()
            .requires_grad_(True)
        )
        v_d1_t = v_d1.transpose(1, 2)

        q_nt = (
            query(x_nt)
            .view(*x_nt.size()[0:2], n_heads, head_dims)
            .detach()
            .requires_grad_(True)
        )
        q_nt_t = q_nt.transpose(1, 2)
        k_nt = (
            key(x_nt)
            .view(*x_nt.size()[0:2], n_heads, head_dims)
            .detach()
            .requires_grad_(True)
        )
        k_nt_t = k_nt.transpose(1, 2)
        v_nt = (
            value(x_nt)
            .view(*x_nt.size()[0:2], n_heads, head_dims)
            .detach()
            .requires_grad_(True)
        )
        v_nt_t = v_nt.transpose(1, 2)

        # High Precision Math Reference
        q_d1_f32 = q_d1.to(torch.float32)
        k_d1_f32 = k_d1.to(torch.float32)
        v_d1_f32 = v_d1.to(torch.float32)
        q_d1_f32_t = q_d1_f32.transpose(1, 2)
        k_d1_f32_t = k_d1_f32.transpose(1, 2)
        v_d1_f32_t = v_d1_f32.transpose(1, 2)
        out_ref = torch.ops.aten._scaled_dot_product_attention_math(
            q_d1_f32_t, k_d1_f32_t, v_d1_f32_t
        )[0]
        grads_ref = torch.autograd.grad(out_ref.sum(), (q_d1_f32, k_d1_f32, v_d1_f32))

        # Low Precision Math Reference
        out_lp_ref = torch.ops.aten._scaled_dot_product_attention_math(
            q_d1_t, k_d1_t, v_d1_t
        )[0]
        grads_lp_ref = torch.autograd.grad(out_lp_ref.sum(), (q_d1, k_d1, v_d1))

        # Compute tolerances
        output_ref_atol, output_ref_rtol = get_tolerances(out_ref, out_lp_ref)
        # fudge factor of 1.7 for smaller GPUs e.g., A2, A16
        grad_q_ref_atol, grad_q_ref_rtol = get_tolerances(
            grads_ref[0], grads_lp_ref[0], 1.7
        )
        grad_k_ref_atol, grad_k_ref_rtol = get_tolerances(grads_ref[1], grads_lp_ref[1])
        grad_v_ref_atol, grad_v_ref_rtol = get_tolerances(grads_ref[2], grads_lp_ref[2])
        grad_atols = [grad_q_ref_atol, grad_k_ref_atol, grad_v_ref_atol]
        grad_rtols = [grad_q_ref_rtol, grad_k_ref_rtol, grad_v_ref_rtol]

        attn_d1 = torch.nn.functional.scaled_dot_product_attention(
            q_d1_t, k_d1_t, v_d1_t
        ).transpose(1, 2)
        attn_nt = torch.nn.functional.scaled_dot_product_attention(
            q_nt_t, k_nt_t, v_nt_t
        ).transpose(1, 2)

        self.assertEqual(
            attn_d1,
            attn_nt.unbind()[0].unsqueeze(0),
            atol=output_ref_atol,
            rtol=output_ref_rtol,
        )

        # Simple case: 2 sentences, no extra params
        x_d2 = sen2.unsqueeze(0)
        x_nt = torch.nested.as_nested_tensor([sen1, sen2], layout=torch.jagged)

        # NB: we make sure the leaf tensor we compute gradients for is the view-ed tensor before
        # it is transposed. This is because today we cannot backward through view or unbind a
        # transposed tensor.
        q_d2 = (
            query(x_d2)
            .view(batch_size, -1, n_heads, head_dims)
            .detach()
            .requires_grad_(True)
        )
        q_d2_t = q_d2.transpose(1, 2)
        k_d2 = (
            key(x_d2)
            .view(batch_size, -1, n_heads, head_dims)
            .detach()
            .requires_grad_(True)
        )
        k_d2_t = k_d2.transpose(1, 2)
        v_d2 = (
            value(x_d2)
            .view(batch_size, -1, n_heads, head_dims)
            .detach()
            .requires_grad_(True)
        )
        v_d2_t = v_d2.transpose(1, 2)

        q_nt = (
            query(x_nt)
            .view(*x_nt.size()[0:2], n_heads, head_dims)
            .detach()
            .requires_grad_(True)
        )
        q_nt_t = q_nt.transpose(1, 2)
        k_nt = (
            key(x_nt)
            .view(*x_nt.size()[0:2], n_heads, head_dims)
            .detach()
            .requires_grad_(True)
        )
        k_nt_t = k_nt.transpose(1, 2)
        v_nt = (
            value(x_nt)
            .view(*x_nt.size()[0:2], n_heads, head_dims)
            .detach()
            .requires_grad_(True)
        )
        v_nt_t = v_nt.transpose(1, 2)

        attn_d2 = torch.nn.functional.scaled_dot_product_attention(
            q_d2_t, k_d2_t, v_d2_t
        ).transpose(1, 2)
        d1_grads = torch.autograd.grad(attn_d1.sum(), (q_d1, k_d1, v_d1))
        d2_grads = torch.autograd.grad(attn_d2.sum(), (q_d2, k_d2, v_d2))

        # Simple case 3: batch_size = 1, seq_len = 1
        q_3 = torch.randn(1, 8, 16, dtype=dtype, device=device)
        q_nt_3 = torch.nested.as_nested_tensor([q_3], layout=torch.jagged)
        q_nt_3 = q_nt_3.transpose(1, 2)
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q_nt_3, q_nt_3, q_nt_3
        )
        self.assertEqual(attn_out.shape, q_nt_3.shape)

        def check_forward_backward():
            attn_nt = torch.nn.functional.scaled_dot_product_attention(
                q_nt_t, k_nt_t, v_nt_t
            ).transpose(1, 2)

            attn_nts = attn_nt.unbind()
            self.assertEqual(
                attn_d1,
                attn_nts[0].unsqueeze(0),
                atol=output_ref_atol,
                rtol=output_ref_rtol,
            )
            self.assertEqual(
                attn_d2,
                attn_nts[1].unsqueeze(0),
                atol=output_ref_atol,
                rtol=output_ref_rtol,
            )

            nt_grads = torch.autograd.grad(attn_nt.values().sum(), (q_nt, k_nt, v_nt))
            for nt_grad, d1_grad, d2_grad, grad_atol, grad_rtol in zip(
                nt_grads, d1_grads, d2_grads, grad_atols, grad_rtols
            ):
                unbound_nt_grads = nt_grad.unbind()
                self.assertEqual(
                    d1_grad,
                    unbound_nt_grads[0].unsqueeze(0),
                    atol=grad_atol,
                    rtol=grad_rtol,
                )
                self.assertEqual(
                    d2_grad,
                    unbound_nt_grads[1].unsqueeze(0),
                    atol=grad_atol,
                    rtol=grad_rtol,
                )

        # Default
        check_forward_backward()

        # Test dispatcher works by calling only mem-effn and math (as they are safe for all devices)
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=True, enable_math=True
        ):
            check_forward_backward()

        # Test math fallback
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):
            # Math fallback doesn't work with bfloat16 on CUDA because
            # "group_gemm_dispatch" not implemented for 'BFloat16'
            if not (str(device).startswith("cuda") and dtype == torch.bfloat16):
                check_forward_backward()

    @skipIfTorchDynamo("SDPA test compiles internally")
    @unittest.skipIf(IS_WINDOWS, reason="Windows not yet supported for torch.compile")
    @skipCUDAIf(not SM70OrLater, "GPU capability is < SM70")
    # Guarding with sqrt() doesn't work on ROCm?
    @skipCUDAIfRocm
    @onlyCUDA
    @dtypes(
        *(
            [torch.float16, torch.bfloat16, torch.float32]
            if SM80OrLater
            else [torch.float16, torch.float32]
        )
    )
    def test_sdpa_compile(self, device, dtype):
        batch_size = 1
        emb_dims = 1024
        n_heads = 8
        head_dims = emb_dims // n_heads

        sen1 = torch.randn(11, emb_dims, dtype=dtype, device=device)
        sen2 = torch.randn(13, emb_dims, dtype=dtype, device=device)

        query = torch.nn.Linear(
            emb_dims, emb_dims, bias=False, device=device, dtype=dtype
        )
        key = torch.nn.Linear(
            emb_dims, emb_dims, bias=False, device=device, dtype=dtype
        )
        value = torch.nn.Linear(
            emb_dims, emb_dims, bias=False, device=device, dtype=dtype
        )

        # Simplest case: 1 sentence, no batching
        x_d1 = sen1.unsqueeze(0)
        x_d2 = sen2.unsqueeze(0)
        x_nt = torch.nested.as_nested_tensor([sen1, sen2], layout=torch.jagged)

        q_d1 = query(x_d1).view(batch_size, -1, n_heads, head_dims).transpose(1, 2)
        k_d1 = key(x_d1).view(batch_size, -1, n_heads, head_dims).transpose(1, 2)
        v_d1 = value(x_d1).view(batch_size, -1, n_heads, head_dims).transpose(1, 2)
        q_d2 = query(x_d2).view(batch_size, -1, n_heads, head_dims).transpose(1, 2)
        k_d2 = key(x_d2).view(batch_size, -1, n_heads, head_dims).transpose(1, 2)
        v_d2 = value(x_d2).view(batch_size, -1, n_heads, head_dims).transpose(1, 2)

        q_nt = (
            query(x_nt)
            .view(*x_nt.size()[0:2], n_heads, head_dims)
            .detach()
            .transpose(1, 2)
        )
        k_nt = (
            key(x_nt)
            .view(*x_nt.size()[0:2], n_heads, head_dims)
            .detach()
            .transpose(1, 2)
        )
        v_nt = (
            value(x_nt)
            .view(*x_nt.size()[0:2], n_heads, head_dims)
            .detach()
            .transpose(1, 2)
        )

        # High Precision Math Reference
        q_d1_f32 = q_d1.to(torch.float32)
        k_d1_f32 = k_d1.to(torch.float32)
        v_d1_f32 = v_d1.to(torch.float32)
        out_ref = torch.ops.aten._scaled_dot_product_attention_math(
            q_d1_f32, k_d1_f32, v_d1_f32
        )[0]
        # Low Precision Math Reference
        out_lp_ref = torch.ops.aten._scaled_dot_product_attention_math(
            q_d1, k_d1, v_d1
        )[0]
        output_ref_atol, output_ref_rtol = get_tolerances(out_ref, out_lp_ref)

        attn_d1 = torch.nn.functional.scaled_dot_product_attention(
            q_d1, k_d1, v_d1
        ).transpose(1, 2)
        attn_d2 = torch.nn.functional.scaled_dot_product_attention(
            q_d2, k_d2, v_d2
        ).transpose(1, 2)

        compiled_sdpa = torch.compile(torch.nn.functional.scaled_dot_product_attention)
        attn_nt = compiled_sdpa(q_nt, k_nt, v_nt).transpose(1, 2)

        attn_nts = attn_nt.unbind()
        self.assertEqual(
            attn_d1,
            attn_nts[0].unsqueeze(0),
            atol=output_ref_atol,
            rtol=output_ref_rtol,
        )
        self.assertEqual(
            attn_d2,
            attn_nts[1].unsqueeze(0),
            atol=output_ref_atol,
            rtol=output_ref_rtol,
        )

    @dtypes(torch.float32, torch.double, torch.half)
    def test_sdpa_with_constant_sequence_length(self, device, dtype):
        # shape (B, P*, S, D)
        # B: batch size
        # P*: ragged number of prompts
        # S: (constant) sequence length
        # D: embedding size
        query = random_nt_from_dims(
            [4, None, 8, 10],
            device=device,
            dtype=dtype,
            layout=torch.jagged,
            requires_grad=True,
        )
        key = random_nt_from_similar(query)
        value = random_nt_from_similar(query)
        output = F.scaled_dot_product_attention(query, key, value)
        self.assertTrue(isinstance(output, NestedTensor))
        output.values().sum().backward()

        query_dense = query.detach().clone().requires_grad_(True)
        # should be equivalent to just running the buffers through
        output_dense = F.scaled_dot_product_attention(
            query_dense.values(), key.values(), value.values()
        )
        torch._dynamo.disable(self.assertEqual)(output._values, output_dense)
        output_dense.sum().backward()
        torch._dynamo.disable(self.assertEqual)(query.grad, query_dense.grad)

    @onlyCUDA
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FUSED_ATTENTION,
        "Platform doesn't support flash or mem-efficient attention",
    )
    @dtypes(
        *(
            [torch.float16, torch.bfloat16, torch.float32]
            if SM80OrLater
            else [torch.float16, torch.float32]
        )
    )
    def test_sdpa_with_packed_in_proj(self, device, dtype):
        # shape (B, *, D)
        input_packed = random_nt_from_dims(
            [5, None, 10], device=device, dtype=dtype, layout=torch.jagged
        )

        # Do input projection.
        num_heads = 2
        # should be multiple of 4 for efficient kernels (e.g. flash / mem-efficient)
        head_dim = 8
        qkv_linear = torch.nn.Linear(10, num_heads * head_dim * 3).to(
            device=device, dtype=dtype
        )

        def in_proj(input_packed, qkv_linear=qkv_linear):
            qkv_post_proj = qkv_linear(input_packed)
            # these are non-contiguous to trigger _is_safe_to_get_storage_as_tensor()
            q, k, v = qkv_post_proj.chunk(3, dim=-1)
            q = q.unflatten(-1, [num_heads, head_dim]).transpose(-2, -3)
            k = k.unflatten(-1, [num_heads, head_dim]).transpose(-2, -3)
            v = v.unflatten(-1, [num_heads, head_dim]).transpose(-2, -3)
            return q, k, v

        q, k, v = in_proj(input_packed)
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=None)

        # compare to individually running unbound components through
        for in_component, out_component in zip(
            input_packed.unbind(), output.transpose(-2, -3).unbind()
        ):
            q, k, v = in_proj(in_component)
            out = F.scaled_dot_product_attention(q, k, v).transpose(-2, -3)

            # Low Precision Math Reference
            out_lp_ref = torch.ops.aten._scaled_dot_product_attention_math(q, k, v)[
                0
            ].transpose(-2, -3)
            output_ref_atol, output_ref_rtol = get_tolerances(
                out, out_lp_ref, fudge_factor=2
            )

            self.assertEqual(
                out, out_component, atol=output_ref_atol, rtol=output_ref_rtol
            )

    @skipIfTorchDynamo("SDPA test compiles internally")
    @unittest.skipIf(IS_WINDOWS, reason="Windows not yet supported for torch.compile")
    @skipCUDAIf(not SM70OrLater, "GPU capability is < SM70")
    # mha_varlen_fwd not supported on ROCm
    @skipCUDAIfRocm
    @onlyCUDA
    @dtypes(
        *(
            [torch.float16, torch.bfloat16, torch.float32]
            if SM80OrLater
            else [torch.float16, torch.float32]
        )
    )
    def test_sdpa_backwards(self, device, dtype):
        values = torch.randn(9, 3, 256, requires_grad=True, device=device, dtype=dtype)
        offsets = torch.tensor([0, 1, 3, 5, 9], device=device, dtype=torch.int64)

        @torch.compile
        def f(values, offsets):
            nt = convert_jagged_to_nested_tensor(values, offsets, max_length=4)
            nt = nt.transpose(-2, -3)
            # purposefully graph break to trigger view replay for subclass view input
            torch.tensor(1).item()
            output = F.scaled_dot_product_attention(nt, nt, nt).transpose(-2, -3)
            return convert_nt_to_jagged(output)

        output = f(values, offsets)
        output.sum().backward()
        self.assertEqual(values.grad, torch.ones_like(values))

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FUSED_ATTENTION,
        "Platform doesn't support flash or mem-efficient attention",
    )
    @skipCUDAIf(not SM70OrLater, "GPU capability is < SM70")
    @skipCUDAIfRocm
    @onlyCUDA
    @skipIfTorchDynamo()
    @unittest.skipIf(IS_WINDOWS, reason="Windows not yet supported for torch.compile")
    def test_sdpa_autocast(self, device):
        def fn_nt(values32, values16, offsets):
            nt32 = convert_jagged_to_nested_tensor(values32, offsets, max_length=16)
            nt16 = convert_jagged_to_nested_tensor(values16, offsets, max_length=16)
            nt32 = nt32.transpose(1, 2)
            nt16 = nt16.transpose(1, 2)
            return F.scaled_dot_product_attention(nt32, nt16, nt32)

        def fn_dense(x32, x16):
            x32 = x32.view(8, 16, 4, 16).transpose(1, 2)
            x16 = x16.view(8, 16, 4, 16).transpose(1, 2)
            return F.scaled_dot_product_attention(x32, x16, x32)

        values32 = torch.randn((8 * 16, 4, 16), device=device, dtype=torch.float32)
        values16 = torch.randn((8 * 16, 4, 16), device=device, dtype=torch.float16)
        offsets = torch.arange(0, 8 * 16 + 1, 16, device=device, dtype=torch.int32)

        x32 = values32.clone()
        x16 = values16.clone()

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out_dense_eager = fn_dense(x32, x16)
            out_dense_compiled = torch.compile(fn_dense)(x32, x16)
            out_nt_eager = fn_nt(values32, values16, offsets)
            out_nt_compiled = torch.compile(fn_nt)(values32, values16, offsets)

        self.assertEqual(out_dense_eager, out_dense_compiled)
        self.assertEqual(
            out_dense_eager.transpose(1, 2),
            out_nt_eager.values().transpose(0, 1).view(8, 16, 4, 16),
        )
        self.assertEqual(
            out_dense_eager.transpose(1, 2),
            out_nt_compiled.values().transpose(0, 1).view(8, 16, 4, 16),
        )

        def get_values():
            return tuple(
                x.detach().clone().requires_grad_(True) for x in (values32, values16)
            )

        v32_dense_eager, v16_dense_eager = get_values()
        v32_dense_compile, v16_dense_compile = get_values()
        v32_nt_eager, v16_nt_eager = get_values()
        v32_nt_compile, v16_nt_compile = get_values()

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss_dense_eager = fn_dense(v32_dense_eager, v16_dense_eager).sum()
            loss_dense_compile = torch.compile(fn_dense)(
                v32_dense_compile, v16_dense_compile
            ).sum()
            loss_nt_eager = fn_nt(v32_nt_eager, v16_nt_eager, offsets).values().sum()
            loss_nt_compile = (
                torch.compile(fn_nt)(v32_nt_compile, v16_nt_compile, offsets)
                .values()
                .sum()
            )

        loss_dense_eager.backward()
        loss_dense_compile.backward()
        loss_nt_eager.backward()
        loss_nt_compile.backward()

        self.assertEqual(v32_dense_eager.grad, v32_dense_compile.grad)
        self.assertEqual(v32_dense_eager.grad, v32_nt_eager.grad, atol=1e-4, rtol=1e-4)
        self.assertEqual(
            v32_dense_eager.grad, v32_nt_compile.grad, atol=1e-4, rtol=1e-4
        )

        self.assertEqual(v16_dense_eager.grad, v16_dense_compile.grad)
        self.assertEqual(v16_dense_eager.grad, v16_nt_eager.grad, atol=1e-5, rtol=5e-3)
        self.assertEqual(
            v16_dense_eager.grad, v16_nt_compile.grad, atol=1e-5, rtol=5e-3
        )

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FUSED_ATTENTION,
        "Platform doesn't support flash or mem-efficient attention",
    )
    @skipCUDAIf(not SM70OrLater, "GPU capability is < SM70")
    @skipCUDAIfRocm
    @onlyCUDA
    @skipIfTorchDynamo()
    def test_sdpa_flop_counter(self, device):
        from torch.utils.flop_counter import FlopCounterMode

        def get_flops(nt):
            flop_counter = FlopCounterMode(display=False)
            with flop_counter:
                ret = torch.nn.functional.scaled_dot_product_attention(nt, nt, nt)
                ret.values().sum().backward()
            return flop_counter.get_total_flops()

        values = torch.randn(
            (8 * 16, 4, 16), requires_grad=True, device=device, dtype=torch.float16
        )
        offsets = torch.arange(0, 8 * 16 + 1, 16, device=device, dtype=torch.int32)
        nt = convert_jagged_to_nested_tensor(values, offsets, max_length=16)

        values_meta = torch.randn(
            (8 * 16, 4, 16), requires_grad=True, device="meta", dtype=torch.float16
        )
        offsets_meta = torch.arange(0, 8 * 16 + 1, 16, device="meta", dtype=torch.int32)
        nt_meta = convert_jagged_to_nested_tensor(values, offsets, max_length=16)

        self.assertEqual(get_flops(nt), get_flops(nt_meta))

    @skipIfTorchDynamo()
    def test_nested_tensor_activation_checkpoint(self, device):
        values = torch.randn(
            9, 3, 256, requires_grad=True, device=device, dtype=torch.float32
        )
        lengths = torch.tensor([1, 2, 3, 3], device=device, dtype=torch.int64)
        offsets = F.pad(lengths, pad=(1, 0)).cumsum(dim=0)

        def fn(values, offsets):
            nt = convert_jagged_to_nested_tensor(values, offsets, max_length=4)
            return convert_nt_to_jagged(nt).sum()

        checkpoint(fn, values, offsets, use_reentrant=False).backward()
        self.assertIsNotNone(values.grad)

        context_fn = partial(
            create_selective_checkpoint_contexts, [torch.ops.aten.cumsum.default]
        )

        values.grad = None

        def fn(values, lengths):
            offsets = F.pad(lengths, pad=(1, 0)).cumsum(dim=0)
            nt = convert_jagged_to_nested_tensor(values, offsets, max_length=4)
            return convert_nt_to_jagged(nt).sum()

        checkpoint(
            fn, values, lengths, use_reentrant=False, context_fn=context_fn
        ).backward()
        self.assertIsNotNone(values.grad)

    # Internally-defined NT use cases are lifted to here for maximum test realism.
    # TODO: Remove these when ViewNestedFromBuffer, etc. are deprecated.
    @skipCUDAIfRocm  # not needed
    @skipIfTorchDynamo("compiles internally")
    @unittest.skipIf(IS_WINDOWS, reason="Windows not yet supported for torch.compile")
    @skipCUDAIf(not SM70OrLater, "GPU capability is < SM70")
    @parametrize("use_legacy_api", [True, False])
    @skipCPUIf(True, "SPDA Math NT fallback causes failure: see issue #133644")
    def test_dummy_mha_with_nt(self, device, use_legacy_api):
        bs = 3
        d1 = 2
        d2 = 4
        d3 = 16
        n_heads = 2
        d_head = d3 // n_heads
        max_length_1 = 10
        max_length_2 = 20
        torch.manual_seed(0)

        class mha(torch.nn.Module):
            def __init__(self, use_legacy_api) -> None:
                super().__init__()
                torch.manual_seed(0)
                self.linear = torch.nn.Linear(d2, d3, device=device)
                self.use_legacy_api = use_legacy_api

            def forward(self, query, value, offsets):
                value = self.linear(value)
                if self.use_legacy_api:
                    key = convert_jagged_to_nested_tensor_legacy(
                        value, offsets, max_length_1
                    )
                    value = convert_jagged_to_nested_tensor_legacy(
                        value, offsets, max_length_2
                    )
                    query = convert_dense_to_nested_tensor_legacy(query)
                else:
                    key = convert_jagged_to_nested_tensor(value, offsets, max_length_1)
                    value = convert_jagged_to_nested_tensor(
                        value, offsets, max_length_2
                    )
                    query = convert_dense_to_nested_tensor(query)
                q = query.view(bs, -1, n_heads, d_head).transpose(1, 2)
                k = key.view(bs, -1, n_heads, d_head).transpose(1, 2)
                v = value.view(bs, -1, n_heads, d_head).transpose(1, 2)

                with torch.nn.attention.sdpa_kernel(
                    [
                        torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                        torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                    ]
                ):
                    attn_output = torch.nn.functional.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        attn_mask=None,
                        dropout_p=0.0,
                        is_causal=False,
                    )
                attn_output = attn_output.transpose(1, 2)
                if self.use_legacy_api:
                    attn_output = convert_nt_to_jagged_legacy(attn_output)
                else:
                    attn_output = convert_nt_to_jagged(attn_output)
                return attn_output, key._max_seqlen, value._max_seqlen

        query = torch.rand(bs, d1, d3, device=device)
        value = torch.rand(30, d2, requires_grad=True, device=device)
        # total_length must > than max_length otherwise flash_attn backwark will fail
        offsets = torch.tensor([0, 2, 3, 30], device=device)

        m = mha(use_legacy_api)
        symbolic_traced: torch.fx.GraphModule = torch.fx.symbolic_trace(m)
        m = torch.compile(symbolic_traced)
        attn_output, cached_key_max_seqlen, cached_value_max_seqlen = m(
            query, value, offsets
        )
        loss = attn_output.sum()
        # Check that NT can be fx traced and torch.compile, and backward works
        loss.backward()

        # Check that value.requires_grad is not lost after tracing and compiling
        value_grad = value.grad  # save for comparison later
        self.assertIsNotNone(value_grad)
        # check that max_seqlen is cached properly
        self.assertEqual(cached_key_max_seqlen, max_length_1)
        self.assertEqual(cached_value_max_seqlen, max_length_2)

        # check if the output is numerically equivalent with the eager mode
        m_eager = mha(use_legacy_api)

        value.grad = None
        attn_output_eager, _, _ = m_eager(query, value, offsets)
        attn_output_eager.sum().backward()
        self.assertTrue(torch.allclose(attn_output_eager, attn_output))
        self.assertTrue(torch.allclose(value_grad, value.grad))

    # Helper function to generate random query, key, value NJTs in (B, n_heads, *, D) format.
    # If noncontig_with_holes is True, the results will be non-contiguous with holes (i.e. have
    # both offsets and lengths specified).
    def _rand_qkv(self, device, dtype, noncontig_with_holes=False):
        batch_size = 8
        n_heads = 8
        D = 16

        sentence_lengths = [random.randint(2, 1023) for _ in range(batch_size - 1)]
        total = sum(sentence_lengths)

        # shape (B, *, D_total) where D_total = n_heads * D
        query = torch.nested.nested_tensor(
            [
                torch.randn(l, n_heads * D, device=device, dtype=dtype)
                for l in sentence_lengths
            ],
            layout=torch.jagged,
        )
        if noncontig_with_holes:
            query = torch.nested.nested_tensor_from_jagged(
                query._values,
                query._offsets,
                # -1 to introduce holes
                lengths=query._offsets.diff() - 1,
                jagged_dim=query._ragged_idx,
                min_seqlen=query._min_seqlen,
                max_seqlen=query._max_seqlen,
            )
        # NB: randn_like() doesn't propagate lengths so this doesn't preserve non-contiguity
        key = torch.randn_like(query)
        value = torch.randn_like(query)

        # shape (B, *, D_total) -> (B, n_heads, *, D)
        query = (
            query.unflatten(-1, [n_heads, D]).transpose(1, 2).detach().requires_grad_()
        )
        key = key.unflatten(-1, [n_heads, D]).transpose(1, 2).detach().requires_grad_()
        value = (
            value.unflatten(-1, [n_heads, D]).transpose(1, 2).detach().requires_grad_()
        )

        return query, key, value

    @onlyCUDA
    @flex_attention_supported_platform
    @dtypes(torch.float32)
    # non-contiguous with holes not supported yet
    @decorateIf(unittest.skip, lambda params: params["noncontig_with_holes"])
    @parametrize("noncontig_with_holes", [False, True])
    @skipIfRocm
    def test_flex_attention(self, device, dtype, noncontig_with_holes):
        query, key, value = self._rand_qkv(device, dtype, noncontig_with_holes)

        # Run FlexAttention with a causal mask
        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        block_mask = create_nested_block_mask(causal_mask, 1, 1, query, _compile=True)
        out_flex = flex_attention(query, key, value, block_mask=block_mask)
        grad_out = torch.randn_like(out_flex)
        grads_flex = torch.autograd.grad(
            out_flex, inputs=(query, key, value), grad_outputs=(grad_out,)
        )
        flex_outs = [out_flex, *grads_flex]

        # Run FlexAttention with a score_mod that represents causal attention
        def causal_score_mod(score, b, h, q_idx, kv_idx):
            return torch.where(q_idx >= kv_idx, score, float("-inf"))

        out_flex2 = flex_attention(query, key, value, score_mod=causal_score_mod)
        grads_flex2 = torch.autograd.grad(
            out_flex2, inputs=(query, key, value), grad_outputs=(grad_out,)
        )
        flex_outs2 = [out_flex2, *grads_flex2]

        # Run causal SDPA for comparison
        out_sdpa = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        grads_sdpa = torch.autograd.grad(
            out_sdpa, inputs=(query, key, value), grad_outputs=(grad_out,)
        )
        sdpa_outs = [out_sdpa, *grads_sdpa]

        # Compare flex vs. SDPA output and grads
        for flex, flex2, sdpa in zip(flex_outs, flex_outs2, sdpa_outs):
            self.assertTrue(flex.is_nested and flex2.is_nested and sdpa.is_nested)
            self.assertEqual(flex, sdpa, atol=1e-2, rtol=1e-2)
            self.assertEqual(flex2, sdpa, atol=1e-2, rtol=1e-2)

    @onlyCUDA
    @flex_attention_supported_platform
    @dtypes(torch.float32)
    def test_flex_attention_converts_stacked_seq_indices(self, device, dtype):
        # This test verifies that a score_mod function written to operate within
        # NJT sequence index space, such as a lookup table, works correctly. This
        # validates that FlexAttention properly converts indices within the
        # "stacked sequence" space used for NJT -> sequence-relative indices.
        query, key, value = self._rand_qkv(device, dtype)

        # Test with score_mod
        score_mod_table = torch.randn(query._max_seqlen, device=device, dtype=dtype)

        def my_score_mod(score, b, h, q_idx, kv_idx):
            return score_mod_table[q_idx]

        flex_attention(query, key, value, score_mod=my_score_mod)

        # Test with mask_mod
        mask_mod_table = score_mod_table > 0.0

        def my_mask_mod(b, h, q_idx, kv_idx):
            return mask_mod_table[q_idx]

        block_mask = create_nested_block_mask(my_mask_mod, 1, 1, query, _compile=True)
        flex_attention(query, key, value, block_mask=block_mask)

    @dtypes(torch.float32)
    def test_apply_(self, device, dtype):
        nt = random_nt_from_dims(
            [5, None, 10],
            device=device,
            dtype=dtype,
            layout=torch.jagged,
            requires_grad=True,
        )

        def f(x):
            return x * 2

        if device != "cpu":
            with self.assertRaisesRegex(
                TypeError, "apply_ is only implemented on CPU tensors"
            ):
                nt.apply_(f)
            return

        before = nt._values.detach().clone()

        nt.apply_(f)
        expected = f(before)
        self.assertEqual(expected, nt._values)
        # apply_ should swap values in-place without appending to autograd graph
        self.assertIsNone(nt.grad)
        self.assertIsNone(nt._values.grad_fn)

    @onlyCUDA
    @dtypes(torch.float64, torch.float32, torch.half)
    @parametrize(
        "contiguity",
        ["noncontig_transposed", "noncontig_with_holes"],
        name_fn=lambda c: c,
    )
    def test_noncontiguous_to(self, device, dtype, contiguity):
        # Dense tensors preserve non-contiguity through to() calls (i.e. strides are
        # preserved). Test for the analogous behavior for NJTs:
        # 1. non-contiguous transposed
        # 2. non-contiguous with holes
        if contiguity == "noncontig_transposed":
            nt = random_nt_from_dims(
                [3, None, 5, 2],
                device=device,
                dtype=dtype,
                layout=torch.jagged,
            ).transpose(-3, -2)
        elif contiguity == "noncontig_with_holes":
            nt = torch.nested.nested_tensor_from_jagged(
                values=torch.randn(10, 3, device=device, dtype=dtype),
                offsets=torch.tensor([0, 3, 7, 10], device=device, dtype=torch.int64),
                # these lengths specify holes
                lengths=torch.tensor([1, 2, 3], device=device, dtype=torch.int64),
            )
        else:
            raise ValueError("invalid contiguity specified for test_noncontiguous_to()")

        # test dtype conversion
        dtype_conversions = {
            torch.float32: torch.half,
            torch.float64: torch.float32,
            torch.half: torch.float32,
        }
        other_dtype = dtype_conversions[dtype]
        nt2 = nt.to(dtype=other_dtype)
        self.assertEqual(nt2.dtype, other_dtype)
        self.assertEqual(nt.is_contiguous(), nt2.is_contiguous())
        self.assertEqual(nt._values.is_contiguous(), nt2._values.is_contiguous())
        self.assertEqual(nt.shape, nt2.shape)
        # expect no change for offsets / lengths
        self.assertEqual(nt._offsets, nt2._offsets)
        self.assertEqual(nt._lengths, nt2._lengths)

        # test device conversion
        other_device = torch.device("cpu")
        nt3 = nt.to(device=other_device)
        self.assertEqual(nt3.device, other_device)
        self.assertEqual(nt.is_contiguous(), nt3.is_contiguous())
        self.assertEqual(nt._values.is_contiguous(), nt3._values.is_contiguous())
        self.assertEqual(nt.shape, nt3.shape)
        # expect device change for offsets / lengths
        self.assertEqual(nt3._offsets.device, other_device)
        if nt._lengths is not None:
            self.assertEqual(nt3._lengths.device, other_device)

    @dtypes(torch.float32)
    def test_autograd_function_with_None_grad(self, device, dtype):
        class MyFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inp):
                ctx.save_for_backward(inp)
                out1 = inp + 1
                out2 = inp * 2
                return out1, out2

            @staticmethod
            def backward(ctx, grad_out1, grad_out2):
                (inp,) = ctx.saved_tensors
                return grad_out1 + grad_out2

        f = MyFunction.apply
        nt = random_nt_from_dims(
            [5, None, 10],
            device=device,
            dtype=dtype,
            layout=torch.jagged,
            requires_grad=True,
        )

        # Only use one of the autograd.Function outputs downstream so that the grad
        # for the other output is None. We're testing that the engine can allocate
        # correctly-shaped (NJT) zeros for the grad of the other output in this case.
        (out1, _) = f(nt)
        out1.backward(torch.ones_like(out1))

    @dtypes(torch.float64, torch.float32, torch.half)
    def test_jagged_padded_dense_conversion_kernels(self, device, dtype):
        values = torch.randn(10, 5, device=device, dtype=dtype)
        offsets = torch.tensor([0, 1, 3, 8, 10], device=device, dtype=torch.int64)
        max_length = offsets.diff().max().item()
        padding_value = 1.3

        # convert jagged -> padded dense
        padded = torch.ops.aten._jagged_to_padded_dense_forward(
            values, [offsets], [max_length], padding_value
        )

        batch_size = offsets.shape[0] - 1
        expected_padded_shape = (batch_size, max_length, values.shape[-1])
        self.assertEqual(padded.shape, expected_padded_shape)

        # convert padded dense -> jagged
        total_L = values.shape[0]
        output_jagged = torch.ops.aten._padded_dense_to_jagged_forward(
            padded, [offsets], total_L
        )

        # should be equivalent to the original values
        self.assertEqual(values, output_jagged)

        # success case: truncate to max length as needed
        trunc_max_length = max_length - 1
        trunc_padded = torch.ops.aten._jagged_to_padded_dense_forward(
            values, [offsets], [trunc_max_length], padding_value
        )
        self.assertEqual(padded[:, :trunc_max_length, :], trunc_padded)

        # specific to CPU impls
        if device == "cpu":
            # error case: multiple offsets on cpu since CPU kernels don't support more now
            with self.assertRaisesRegex(
                RuntimeError, "only a single jagged dim is supported"
            ):
                torch.ops.aten._jagged_to_padded_dense_forward(
                    values, [offsets, offsets], [max_length, max_length], padding_value
                )

            with self.assertRaisesRegex(
                RuntimeError, "only a single jagged dim is supported"
            ):
                torch.ops.aten._padded_dense_to_jagged_forward(
                    padded, [offsets, offsets], total_L
                )

            # error case: > 1D offsets
            offsets2d = offsets.unsqueeze(-1)
            with self.assertRaisesRegex(RuntimeError, "expected 1D offsets"):
                torch.ops.aten._jagged_to_padded_dense_forward(
                    values, [offsets2d], [max_length], padding_value
                )

            with self.assertRaisesRegex(RuntimeError, "expected 1D offsets"):
                torch.ops.aten._padded_dense_to_jagged_forward(
                    padded, [offsets2d], total_L
                )

            # error case: final offset != total_L
            offsets_wrong = offsets.detach().clone()
            offsets_wrong[-1] = total_L + 1
            with self.assertRaisesRegex(
                RuntimeError, "final offset should match total_L value"
            ):
                torch.ops.aten._padded_dense_to_jagged_forward(
                    padded, [offsets_wrong], total_L
                )

            # error case: 1D padded input
            padded_wrong = padded.flatten().detach().clone()
            with self.assertRaisesRegex(RuntimeError, "expected padded dim >= 2"):
                torch.ops.aten._padded_dense_to_jagged_forward(
                    padded_wrong, [offsets], total_L
                )

            # error case: batch item has length > max length
            # max_length is 5 above; 7 here
            offsets_wrong = torch.tensor(
                [0, 1, 8, 9, 10], device=device, dtype=torch.int64
            )
            with self.assertRaisesRegex(RuntimeError, "found batch item of length"):
                torch.ops.aten._padded_dense_to_jagged_forward(
                    padded, [offsets_wrong], total_L
                )

    @dtypes(torch.float32)
    @skipIfTorchDynamo("Test compiles internally")
    @unittest.skipIf(
        sys.version_info >= (3, 12), "torch.compile is not supported on python 3.12+"
    )
    @unittest.skipIf(IS_WINDOWS, reason="Windows not yet supported for torch.compile")
    @skipCUDAIf(not SM70OrLater, "GPU capability is < SM70")
    @skipCUDAIfRocm
    def test_compile_preserves_metadata_cache(self, device, dtype):
        # shape (B, *, D)
        nt = random_nt_from_dims(
            [4, None, 3, 16],
            device=device,
            dtype=dtype,
            layout=torch.jagged,
            requires_grad=True,
        )

        # expect min / max seqlen to be stored here
        cache = dict(nt._metadata_cache)

        @torch.compile
        def f(nt):
            q = nt.transpose(-3, -2)
            output = F.scaled_dot_product_attention(q, q, q).transpose(-3, -2)
            return output

        output = f(nt)
        output.backward(torch.ones_like(output))
        self.assertEqual(output._metadata_cache, cache)

    @dtypes(torch.float32)
    @skipIfTorchDynamo("Test compiles internally")
    @unittest.skipIf(
        sys.version_info >= (3, 12), "torch.compile is not supported on python 3.12+"
    )
    @unittest.skipIf(IS_WINDOWS, reason="Windows not yet supported for torch.compile")
    @skipCUDAIf(not SM70OrLater, "GPU capability is < SM70")
    @skipCUDAIfRocm
    def test_compile_with_dynamic_max_seq_len(self, device, dtype):
        # shape (B, *, D)
        # max seq len: 18
        nt = torch.nested.nested_tensor(
            [
                torch.randn(2, 5),
                torch.randn(3, 5),
                torch.randn(18, 5),
            ],
            layout=torch.jagged,
        )

        # max seq len: 19
        nt2 = torch.nested.nested_tensor(
            [
                torch.randn(2, 5),
                torch.randn(3, 5),
                torch.randn(19, 5),
            ],
            layout=torch.jagged,
        )

        def f(nt):
            # TODO: Replace with public API when we can use @properties
            return torch.ones_like(nt) * nt._get_max_seqlen()

        for dynamic in [False, True, None]:
            self.assertFalse(_recompiles_for_inputs(f, (nt,), (nt2,), dynamic=dynamic))

    @dtypes(torch.float32)
    @skipIfTorchDynamo("Test compiles internally")
    @unittest.skipIf(
        sys.version_info >= (3, 12), "torch.compile is not supported on python 3.12+"
    )
    @unittest.skipIf(IS_WINDOWS, reason="Windows not yet supported for torch.compile")
    @skipCUDAIf(not SM70OrLater, "GPU capability is < SM70")
    @skipCUDAIfRocm
    def test_compile_with_dynamic_min_seq_len(self, device, dtype):
        # shape (B, *, D)
        # min seq len: 7
        nt = torch.nested.nested_tensor(
            [
                torch.randn(7, 5),
                torch.randn(8, 5),
                torch.randn(9, 5),
            ],
            layout=torch.jagged,
        )

        # min seq len: 8
        nt2 = torch.nested.nested_tensor(
            [
                torch.randn(8, 5),
                torch.randn(9, 5),
                torch.randn(10, 5),
            ],
            layout=torch.jagged,
        )

        def f(nt):
            # TODO: Replace with public API when we can use @properties
            return torch.ones_like(nt) * nt._get_min_seqlen()

        for dynamic in [False, True, None]:
            self.assertFalse(_recompiles_for_inputs(f, (nt,), (nt2,), dynamic=dynamic))

    @dtypes(torch.float32)
    @skipIfTorchDynamo("Test compiles internally")
    @unittest.skipIf(
        sys.version_info >= (3, 12), "torch.compile is not supported on python 3.12+"
    )
    @unittest.skipIf(IS_WINDOWS, reason="Windows not yet supported for torch.compile")
    @skipCUDAIf(not SM70OrLater, "GPU capability is < SM70")
    @skipCUDAIfRocm
    def test_compile_with_propagated_dynamic_max_seq_len(self, device, dtype):
        # shape (B, *, D)
        # max seq len: 18
        nt = torch.nested.nested_tensor(
            [
                torch.randn(2, 5),
                torch.randn(3, 5),
                torch.randn(18, 5),
            ],
            layout=torch.jagged,
        )

        # max seq len: 19
        nt2 = torch.nested.nested_tensor(
            [
                torch.randn(2, 5),
                torch.randn(3, 5),
                torch.randn(19, 5),
            ],
            layout=torch.jagged,
        )

        def f(nt):
            nt2 = nt.sin() + 1
            # TODO: Replace with public API when we can use @properties
            return torch.ones_like(nt2) * nt2._get_max_seqlen()

        ref = f(nt)
        output = torch.compile(f, fullgraph=True, dynamic=False)(nt)
        self.assertEqual(ref, output)

        for dynamic in [False, True, None]:
            self.assertFalse(_recompiles_for_inputs(f, (nt,), (nt2,), dynamic=dynamic))

    @dtypes(torch.float32, torch.double, torch.half)
    def test_unbind_backward(self, device, dtype):
        nt = torch.nested.nested_tensor(
            [
                torch.randn(2, 4, device=device),
                torch.randn(5, 4, device=device),
                torch.randn(3, 4, device=device),
            ],
            layout=torch.jagged,
            requires_grad=True,
        )

        a, b, c = nt.unbind()
        b.sum().backward()

        @torch._dynamo.disable
        def check(nt):
            expected_grad = torch.zeros_like(nt)
            expected_grad.unbind()[1].add_(1.0)
            self.assertEqual(nt.grad, expected_grad)

        check(nt)

    @dtypes(torch.float32, torch.double, torch.half, torch.bool)
    @parametrize("nt_dim", [2, 3, 4])
    @parametrize("requires_grad", [False, True])
    def test_to_padded_tensor(self, device, dtype, nt_dim, requires_grad):
        if dtype is torch.bool and requires_grad:
            # grads not supported for bool
            return

        if nt_dim == 2:
            post_seq_len_shape = ()
        elif nt_dim == 3:
            post_seq_len_shape = (10,)
        elif nt_dim == 4:
            post_seq_len_shape = (9, 10)

        nt = torch.nested.nested_tensor(
            [
                torch.randint(2, (n, *post_seq_len_shape), device=device, dtype=dtype)
                if dtype is torch.bool
                else torch.randn(n, *post_seq_len_shape, device=device, dtype=dtype)
                for n in range(2, 9)
            ],
            layout=torch.jagged,
            requires_grad=requires_grad,
        )

        PADDING_VAL = 4.2
        expected_padded = nt._values.new_full((7, 8, *post_seq_len_shape), PADDING_VAL)
        for i, component in enumerate(nt.unbind()):
            expected_padded[i, : component.shape[0]].copy_(component)

        padded = nt.to_padded_tensor(PADDING_VAL)
        self.assertEqual(expected_padded, padded)

        # convert padded dense -> NJT
        from torch.nested._internal.nested_tensor import nested_from_padded

        nt2 = nested_from_padded(padded, nt.offsets())
        self.assertEqual(nt, nt2)

        if requires_grad and dtype is not torch.bool:
            # ensure gradients flow through conversions
            nt2.backward(torch.ones_like(nt2))
            self.assertEqual(nt.grad, torch.ones_like(nt))

    # blows up due to test parametrization otherwise
    @torch._dynamo.utils.disable_cache_limit()
    @skipIfTorchDynamo("SDPA test compiles internally")
    @unittest.skipIf(IS_WINDOWS, reason="Windows not yet supported for torch.compile")
    @skipCUDAIf(not SM70OrLater, "GPU capability is < SM70")
    @skipCUDAIfRocm
    @dtypes(torch.float32, torch.double, torch.half)
    @parametrize("nt_dim", [2, 3, 4])
    @parametrize("requires_grad", [False, True])
    def test_to_padded_tensor_compile(self, device, dtype, nt_dim, requires_grad):
        if dtype is torch.bool and requires_grad:
            # grads not supported for bool
            return

        if nt_dim == 2:
            post_seq_len_shape = ()
        elif nt_dim == 3:
            post_seq_len_shape = (10,)
        elif nt_dim == 4:
            post_seq_len_shape = (9, 10)

        nt = torch.nested.nested_tensor(
            [
                torch.randint(2, (n, *post_seq_len_shape), device=device, dtype=dtype)
                if dtype is torch.bool
                else torch.randn(n, *post_seq_len_shape, device=device, dtype=dtype)
                for n in range(2, 9)
            ],
            layout=torch.jagged,
            requires_grad=requires_grad,
        )

        def f(x):
            return x.sin() + 1

        from torch.nested._internal.nested_tensor import nested_from_padded

        @torch.compile(fullgraph=True)
        def g(nt):
            def _g(nt):
                PADDING_VAL = 4.2
                padded = nt.to_padded_tensor(PADDING_VAL)
                padded = f(padded)
                # NB: sum_S must be specified to use the lowering for dense -> jagged
                # and get full fusion
                return nested_from_padded(
                    padded, nt.offsets(), sum_S=nt.values().shape[0]
                )

            # NB: use checkpointing to force fusion
            return torch.utils.checkpoint.checkpoint(_g, nt, use_reentrant=False)

        expected_output = f(nt)
        if requires_grad:
            expected_output.backward(torch.ones_like(expected_output))
            expected_grad = nt.grad.detach().clone()
            nt.grad = None

        from torch._inductor.utils import run_and_get_code

        compiled_output, generated_code = run_and_get_code(g, nt)
        if requires_grad:
            compiled_output.backward(torch.ones_like(compiled_output))
            compiled_grad = nt.grad.detach().clone()
            self.assertEqual(compiled_grad, expected_grad, rtol=1e-3, atol=1e-3)

        self.assertEqual(compiled_output, expected_output, rtol=1e-3, atol=1e-3)

        # === Verify that computation fusion happens. ===
        # Fallback op call -> fusion didn't happen.
        fallback_op_calls_present = any(
            "torch.ops.aten._padded_dense_to_jagged_forward.default("
            in generated_code[i]
            or "torch.ops.aten._jagged_to_padded_dense_forward.default("
            in generated_code[i]
            for i in range(len(generated_code))
        )

        # NB: Fusion isn't supported on CPU.
        self.assertEqual("cuda" in device, not fallback_op_calls_present)

        for i in range(len(generated_code)):
            # Examine buffer construction lines in the generated code to determine
            # whether fusion occurred. If fusion happens, a 3D buffer with shape
            # (B, max_seqlen, D) should never be materialized.
            buffer_constructions = [
                line.strip()
                for line in generated_code[i].split("\n")
                if "empty_strided_cuda(" in line
            ]

            buffer_dims = [
                # buffer dim == number of elements in the tensor size tuple arg
                len(ast.parse(t).body[0].value.args[0].elts)
                for t in buffer_constructions
            ]

            if "cuda" in device:
                self.assertFalse(any(d == 3 for d in buffer_dims))

    @dtypes(torch.float32)
    @skipIfTorchDynamo("Test compiles internally")
    @unittest.skipIf(
        sys.version_info >= (3, 12), "torch.compile is not supported on python 3.12+"
    )
    @unittest.skipIf(IS_WINDOWS, reason="Windows not yet supported for torch.compile")
    @skipCUDAIf(not SM70OrLater, "GPU capability is < SM70")
    @skipCUDAIfRocm
    def test_compile_padded_dense_conversion_preserves_metadata_cache(
        self, device, dtype
    ):
        # shape (B, *, D)
        nt = random_nt_from_dims(
            [4, None, 3, 16],
            device=device,
            dtype=dtype,
            layout=torch.jagged,
            requires_grad=True,
        )

        # expect min / max seqlen to be stored here
        cache = dict(nt._metadata_cache)

        @torch.compile
        def g(nt):
            padded = nt.to_padded_tensor(0.3)
            intermediate = padded.sin() + 1

            from torch.nested._internal.nested_tensor import nested_from_padded

            return nested_from_padded(
                intermediate,
                nt.offsets(),
                min_seqlen=nt._min_seqlen,
                max_seqlen=nt._max_seqlen,
                sum_S=nt.values().shape[0],
            )

        output = g(nt)
        output.backward(torch.ones_like(output))
        self.assertEqual(output._metadata_cache, cache)

    # See https://github.com/pytorch/pytorch/issues/128649
    @xfailIfTorchDynamo
    @dtypes(torch.float32)
    def test_composite_op_in_inference_mode(self, device, dtype):
        # expect view
        nt = random_nt_from_dims(
            [4, None, 48],
            device=device,
            dtype=dtype,
            layout=torch.jagged,
            requires_grad=True,
        )

        with torch.inference_mode():
            output = nt.reshape([4, -1, 3, 16])
            self.assertEqual(output.shape, (4, nt.shape[1], 3, 16))
            self.assertTrue(output._is_view())

        # expect copy
        nt = random_nt_from_dims(
            [4, None, 3, 16],
            device=device,
            dtype=dtype,
            layout=torch.jagged,
            requires_grad=True,
        ).transpose(-1, -2)

        with torch.inference_mode():
            output = nt.reshape([4, -1, 48])
            self.assertEqual(output.shape, (4, nt.shape[1], 48))
            self.assertFalse(output._is_view())

    @dtypes(torch.float32)
    def test_composite_op_with_custom_mode(self, device, dtype):
        from torch.utils._python_dispatch import TorchDispatchMode

        # simple passthrough TorchDispatchMode
        class CustomDispatchMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=..., kwargs=None):
                return func(*args, **kwargs)

        nt = random_nt_from_dims(
            [4, None, 2, 3],
            device=device,
            dtype=dtype,
            layout=torch.jagged,
            requires_grad=True,
        )
        with CustomDispatchMode():
            res = nt.reshape(4, -1, 6)

        self.assertEqual(res.shape, (4, nt.shape[1], 6))


# The following lists specify rules indicating how to handle particular SampleInputs: are they
# expected to fail, should they be skipped, etc. Note that rules are attempted to be matched
# from top to bottom and only one rule at most will be matched, so order matters! The guiding
# general principle here should be one xfail / skip per bug if at all possible :)
FORWARD_FAILURES = [
    # not implemented
    XFailRule(
        error_type=NotImplementedError,
        match_fn=lambda device, dtype, op, sample: op.full_name
        in {
            # unary
            "nn.functional.celu",
            "nn.functional.elu",
            "nn.functional.hardshrink",
            "nn.functional.hardsigmoid",
            "nn.functional.hardtanh",
            "nn.functional.logsigmoid",
            "nn.functional.mish",
            "nn.functional.relu6",
            "nn.functional.rrelu",
            "nn.functional.selu",
            "nn.functional.softplus",
            "nn.functional.softshrink",
            "nn.functional.threshold",
            # binary
            "__rsub__",
            "complex",
            "floor_divide",
            "polar",
            "rsub",
            # reduction
            "count_nonzero",
            "linalg.vector_norm",
            "nansum",
            "std",
            "std.unbiased",
            "var",
            "var.unbiased",
        },
        name="not_implemented",
    ),
    # expected: masked ops don't support jagged layout
    XFailRule(
        error_type=ValueError,
        error_msg="expects strided",
        match_fn=lambda device, dtype, op, sample: op.full_name
        in {
            "masked.amax",
            "masked.amin",
            "masked.argmax",
            "masked.argmin",
            "masked.logsumexp",
            "masked.mean",
            "masked.norm",
            "masked.prod",
            "masked.std",
            "masked.sum",
            "masked.var",
        },
        name="no_masked_jagged_support",
    ),
    # Need to adjust sample input func to pass the right thing
    XFailRule(
        error_type=TypeError,
        error_msg="missing 1 required positional arguments",
        match_fn=lambda device, dtype, op, sample: op.full_name
        == "nn.functional.prelu",
        name="invalid_prelu_sample_input_func",
    ),
    # Op doesn't support lengths being present
    XFailRule(
        error_type=ValueError,
        error_msg="expected input to be a contiguous jagged layout NestedTensor",
        match_fn=lambda device, dtype, op, sample: (
            op.full_name == "nn.functional.linear" and sample.input._lengths is not None
        ),
        name="no_linear_noncontig_holes_support",
    ),
    # Some kinda reduction bug that needs to be fixed!
    XFailRule(
        error_type=IndexError,
        error_msg="tuple index out of range",
        match_fn=lambda device, dtype, op, sample: (
            # min.reduction_with_dim and max.reduction_with_dim aren't associated with
            # ReductionOpInfo entries sadly even though they're reductions
            (isinstance(op, ReductionOpInfo) or "reduction_with_dim" in op.full_name)
            and (
                sample.name == "3D_noncontig_transposed_with_seqlen_cache: "
                "normal dim reduction with keepdim=False"
            )
        ),
        name="transposed_reduction_bug",
    ),
    # nanmean sometimes hits an unimplemented nansum() path and other times hits an
    # unimplemented sum() path
    XFailRule(
        error_type=NotImplementedError,
        match_fn=lambda device, dtype, op, sample: (
            op.full_name == "nanmean"
            and not (
                "noncontig_holes" in sample.name
                and "dim" in sample.kwargs
                and (
                    (
                        isinstance(sample.kwargs["dim"], int)
                        and sample.kwargs["dim"] == sample.input._ragged_idx
                    )
                    or (
                        isinstance(sample.kwargs["dim"], (tuple, list))
                        and sample.input._ragged_idx in sample.kwargs["dim"]
                    )
                )
            )
        ),
        name="nansum_unimplemented",
    ),
    # expected: reducing across the ragged dimension is not supported for non-contiguous
    # nested tensors with holes
    XFailRule(
        error_type=RuntimeError,
        error_msg=(
            "reducing across the ragged dimension is not supported for non-contiguous "
            "nested tensors with holes"
        ),
        match_fn=lambda device, dtype, op, sample: (
            # min.reduction_with_dim and max.reduction_with_dim aren't associated with
            # ReductionOpInfo entries sadly even though they're reductions
            (isinstance(op, ReductionOpInfo) or "reduction_with_dim" in op.full_name)
            and "noncontig_holes" in sample.name
            and "dim" in sample.kwargs
            and (
                (
                    isinstance(sample.kwargs["dim"], int)
                    and sample.kwargs["dim"] == sample.input._ragged_idx
                )
                or (
                    isinstance(sample.kwargs["dim"], (tuple, list))
                    and sample.input._ragged_idx in sample.kwargs["dim"]
                )
            )
        ),
        name="ragged_dim_reduction_noncontig_holes",
    ),
    # expected: index_put() doesn't work on non-contiguous NJTs without ragged dimension indices
    XFailRule(
        error_type=RuntimeError,
        error_msg="If ragged dimension is not part of indices, this only works on contiguous NJTs",
        match_fn=lambda device, dtype, op, sample: (
            op.full_name == "index_put"
            and not sample.input.is_contiguous()
            and len(sample.kwargs["indices"]) - 1 < sample.input._ragged_idx
        ),
        name="index_put_noncontig_holes_no_ragged_dim_indices",
    ),
    # expected: masked_select() doesn't work on non-contiguous NJTs
    XFailRule(
        error_type=ValueError,
        error_msg="expected self to be a contiguous jagged layout NestedTensor",
        match_fn=lambda device, dtype, op, sample: (
            op.full_name == "masked_select" and not sample.input.is_contiguous()
        ),
        name="masked_select_noncontig",
    ),
    # expected: bmm / matmul sometimes use a to_padded_tensor() fallback which isn't
    # supported for non-contig NJTs with holes
    XFailRule(
        error_type=RuntimeError,
        error_msg="not supported for nested tensors with holes",
        match_fn=lambda device, dtype, op, sample: (
            op.full_name in {"bmm", "matmul"}
            and "noncontig_holes" in sample.name
            and
            # "other" is the name for the matmul arg and "mat2" is the name for the bmm arg
            sample.input.dim()
            == sample.kwargs.get("other", sample.kwargs.get("mat2")).dim()
        ),
        name="mm_noncontig_holes",
    ),
    # some jiterator op failures due to unsupported jagged layout
    XFailRule(
        error_type=RuntimeError,
        error_msg="unsupported tensor layout",
        match_fn=lambda device, dtype, op, sample: op.full_name
        in {
            "jiterator_binary",
            "jiterator_binary_return_by_ref",
            "jiterator_unary",
        },
        name="no_jiterator_jagged_support",
    ),
    # Bug when broadcasting a binary op with non-contiguous with holes NJT + dense
    # tensor with 1 in ragged dim.
    XFailRule(
        error_type=RuntimeError,
        error_msg="cannot call binary pointwise function .* with inputs of shapes",
        match_fn=lambda device, dtype, op, sample: (
            isinstance(op, BinaryUfuncInfo)
            and "noncontig_holes" in sample.name
            and "broadcasting 1 over ragged" in sample.name
        ),
        name="binary_noncontig_holes_broadcasting_1_over_ragged",
    ),
    # Bug: this op returns a tuple of Tensors so it doesn't work with NJT's unary
    # pointwise logic
    XFailRule(
        error_type=AttributeError,
        error_msg="'tuple' object has no attribute 'device'",
        match_fn=lambda device, dtype, op, sample: op.full_name == "frexp",
        name="frexp_tuple_return",
    ),
    # Bug: fill doesn't work with NJTs at all for some reason
    XFailRule(
        error_type=TypeError,
        error_msg="received an invalid combination of arguments",
        match_fn=lambda device, dtype, op, sample: op.full_name == "fill",
        name="fill_bug",
    ),
]

BACKWARD_FAILURES = [
    *FORWARD_FAILURES,
    # I don't know why these fail in CI only and I just want to land this; investigate this later.
    SkipRule(
        match_fn=lambda device, dtype, op, sample: (
            op.full_name
            in {
                "__rpow__",
                "clamp_max",
                "clamp_min",
                "float_power",
                "pow",
                "sinc",
                "special.i1",
                "special.i1e",
            }
        ),
        name="skip_things_that_break_in_ci_but_not_locally",
    ),
    # Bug: Something is wrongly creating an empty tensor with the jagged layout on the C++ side
    # for these binary ops
    XFailRule(
        error_type=RuntimeError,
        error_msg="== Layout::Strided INTERNAL ASSERT FAILED",
        match_fn=lambda device, dtype, op, sample: (
            op.full_name
            in {
                "__rpow__",
                "clamp_min",
                "clamp_max",
                "float_power",
                "pow",
            }
            and (
                (
                    "(NT, T) broadcasting 1 over ragged" in sample.name
                    and "noncontig_holes" not in sample.name
                )
                or "(NT, T) broadcasting all 1s" in sample.name
                or "(NT, T) mixed broadcasting" in sample.name
            )
        ),
        name="binary_empty_with_jagged_layout",
    ),
    # Bug: Something is wrongly creating an empty tensor with the jagged layout on the C++ side
    # for this op when cached seqlen metadata is present
    XFailRule(
        error_type=RuntimeError,
        error_msg="== Layout::Strided INTERNAL ASSERT FAILED",
        match_fn=lambda device, dtype, op, sample: (
            op.full_name
            in {
                "special.i1",
                "special.i1e",
                "sinc",
            }
            and "with_seqlen_cache" in sample.name
        ),
        name="binary_empty_with_jagged_layout_with_cached_seqlens",
    ),
    XFailRule(
        error_type=RuntimeError,
        error_msg="reducing across the ragged dimension is not supported for non-contiguous",
        match_fn=lambda device, dtype, op, sample: (
            isinstance(op, BinaryUfuncInfo)
            # doesn't happen for these ops for some reason
            and op.full_name
            not in {"copysign", "max.binary", "maximum", "min.binary", "minimum"}
            and "(NT, T) broadcasting all 1s" in sample.name
            and "noncontig_holes" in sample.name
        ),
        name="binary_noncontig_holes_ragged_dim_reduction",
    ),
    XFailRule(
        error_type=RuntimeError,
        error_msg="reducing across the ragged dimension is not supported for non-contiguous",
        match_fn=lambda device, dtype, op, sample: (
            op.full_name == "nn.functional.rms_norm"
            and sample.input._lengths is not None
        ),
        name="rms_norm_noncontig_holes_ragged_dim_reduction",
    ),
    # not implemented
    XFailRule(
        error_type=NotImplementedError,
        match_fn=lambda device, dtype, op, sample: op.full_name
        in {
            # uses fill_ which isn't implemented
            "atanh",
        }
        and "with_seqlen_cache" in sample.name,
        name="atanh_unimplemented_fill",
    ),
    # expected: autodiff on complex dtype is not supported
    XFailRule(
        error_type=RuntimeError,
        error_msg=(
            "_nested_view_from_jagged does not support automatic differentiation "
            "for outputs with complex dtype"
        ),
        match_fn=lambda device, dtype, op, sample: (
            op.full_name in {"cdouble", "cfloat", "chalf"}
            and "with_seqlen_cache" in sample.name
        ),
        name="no_complex_autodiff",
    ),
    # bad derivative formula or something
    XFailRule(
        error_type=RuntimeError,
        error_msg="NestedTensor does not support directly calling torch.ops.aten.size",
        match_fn=lambda device, dtype, op, sample: (
            op.full_name in {"sgn", "masked_select"}
            and "with_seqlen_cache" in sample.name
        ),
        name="direct_size_call_with_seqlen_cache",
    ),
    # Bug: need to use the correct nested int in the return shape
    XFailRule(
        error_type=RuntimeError,
        error_msg="Function CloneBackward0 returned an invalid gradient",
        match_fn=lambda device, dtype, op, sample: (
            op.full_name == "clone"
            and sample.kwargs.get("memory_format", None) == torch.contiguous_format
        ),
        name="clone_wrong_nested_int_for_gradient",
    ),
    # some min / max ops use masked_fill_ underneath sometimes, which isn't implemented
    XFailRule(
        error_type=NotImplementedError,
        error_msg="aten.masked_fill_.Scalar",
        match_fn=lambda device, dtype, op, sample: (
            op.full_name in {"max.binary", "min.binary", "minimum", "maximum"}
            and (
                (
                    (
                        "(NT, T) broadcasting all 1s" in sample.name
                        or "(NT, T) broadcasting 1 over ragged" in sample.name
                        or "(NT, T) mixed broadcasting" in sample.name
                    )
                    and "noncontig" not in sample.name
                )
                or sample.name
                in {
                    "4D_noncontig_with_seqlen_cache: (NT, T) broadcasting 1 over ragged",
                    "4D_noncontig_with_seqlen_cache: (NT, T) broadcasting all 1s",
                }
            )
        ),
        name="unimplemented_masked_fill",
    ),
    XFailRule(
        error_type=ValueError,
        error_msg="expected condition to be a contiguous jagged layout NestedTensor",
        match_fn=lambda device, dtype, op, sample: (
            op.full_name in {"max.binary", "min.binary", "minimum", "maximum"}
            and (
                "(NT, T) broadcasting all 1s" in sample.name
                or "(NT, T) broadcasting 1 over ragged" in sample.name
            )
        ),
        name="no_where_noncontig_support",
    ),
]

COMPILE_FORWARD_FAILURES = [
    *FORWARD_FAILURES,
    # Bug: cross-device conversions with to() result in new nested ints within compile only
    XFailRule(
        error_type=AssertionError,
        error_msg="The values for attribute 'shape' do not match",
        match_fn=lambda device, dtype, op, sample: (
            op.full_name == "to" and "-> cpu" in sample.name
        ),
        name="cross_device_transfer_wrong_nested_int_in_compile",
    ),
    # clone() -> contiguous format on an non-contiguous NJT with holes currently uses
    # unbind(), leading to data-dependent error in torch.compile
    XFailRule(
        error_type=torch._dynamo.exc.Unsupported,
        error_msg="data dependent operator: aten._local_scalar_dense.default",
        match_fn=lambda device, dtype, op, sample: (
            op.full_name == "clone"
            and "noncontig_holes" in sample.name
            and sample.kwargs.get("memory_format", None) == torch.contiguous_format
        ),
        name="clone_unbind_data_dependency",
    ),
    # Bug: no idea what's going on here; needs investigation within AOTAutograd
    XFailRule(
        error_type=ValueError,
        error_msg="has length 1 but the spec refers to a pytree that holds 3 items",
        match_fn=lambda device, dtype, op, sample: (
            op.full_name == "nan_to_num" and "noncontig_transposed" in sample.name
        ),
        name="crazy_aot_autograd_bug1",
    ),
    # Bug: also no idea what's going on here: needs investigation within AOTAutograd
    XFailRule(
        error_type=AssertionError,
        error_msg="Expected 5 == 4",
        match_fn=lambda device, dtype, op, sample: (
            op.full_name == "isreal" and "noncontig_transposed" in sample.name
        ),
        name="crazy_aot_autograd_bug2",
    ),
]

COMPILE_BACKWARD_FAILURES = [
    # Bug: Something is wrongly creating an empty tensor with the jagged layout on the C++ side
    # for these binary ops
    XFailRule(
        error_type=NotImplementedError,
        error_msg="non-strided meta tensors not supported yet",
        match_fn=lambda device, dtype, op, sample: (
            op.full_name
            in {
                "__rpow__",
                "clamp_max",
                "clamp_min",
                "float_power",
                "pow",
                "sinc",
            }
            and "noncontig_holes" not in sample.name
            and (
                "(NT, T) broadcasting 1 over ragged" in sample.name
                or "(NT, T) broadcasting all 1s" in sample.name
                or "(NT, T) mixed broadcasting" in sample.name
            )
        ),
        name="empty_with_jagged_layout_for_some_binary_ops",
    ),
    # Bug: Something is wrongly creating an empty tensor with the jagged layout on the C++ side
    # for this op when cached seqlen metadata is present
    XFailRule(
        error_type=NotImplementedError,
        error_msg="non-strided meta tensors not supported yet",
        match_fn=lambda device, dtype, op, sample: (
            op.full_name
            in {
                "special.i1",
                "special.i1e",
                "sinc",
            }
            and "with_seqlen_cache" in sample.name
        ),
        name="empty_with_jagged_layout_with_cached_seqlens",
    ),
    # in compile, these complex ops use view_as_real(), which isn't implemented
    XFailRule(
        error_type=NotImplementedError,
        error_msg="aten.view_as_real.default",
        match_fn=lambda device, dtype, op, sample: (
            op.full_name in {"cdouble", "cfloat", "chalf"}
            and "with_seqlen_cache" in sample.name
        ),
        name="unimplemented_view_as_real",
    ),
    # torch._subclasses.fake_tensor.DataDependentOutputException: aten._local_scalar_dense.default
    # from item call in clone() -> unbind()
    XFailRule(
        error_type=torch._dynamo.exc.Unsupported,
        error_msg="Backend compiler failed with a fake tensor exception",
        match_fn=lambda device, dtype, op, sample: (
            (
                (
                    isinstance(op, BinaryUfuncInfo)
                    and
                    # don't include unimplemented ops
                    op.full_name
                    not in {
                        "__rsub__",
                        "complex",
                        "floor_divide",
                        "polar",
                        "rsub",
                    }
                )
                or op.full_name
                in {
                    "__rpow__",
                    "clamp_max",
                    "clamp_min",
                    "float_power",
                    "pow",
                    "sinc",
                }
            )
            and "(NT, T) broadcasting all 1s" in sample.name
            and "noncontig_holes" in sample.name
        ),
        name="backward_unbind_data_dependency",
    ),
    # ditto
    XFailRule(
        error_type=torch._dynamo.exc.Unsupported,
        error_msg="Backend compiler failed with a fake tensor exception",
        match_fn=lambda device, dtype, op, sample: (
            op.full_name == "nn.functional.rms_norm"
            and sample.input._lengths is not None
        ),
        name="rms_norm_backward_unbind_data_dependency",
    ),
    # clone() -> preserve format on an non-contiguous NJT with holes currently uses
    # unbind(), leading to data-dependent error in torch.compile
    XFailRule(
        error_type=torch._dynamo.exc.Unsupported,
        error_msg="Backend compiler failed with a fake tensor exception",
        match_fn=lambda device, dtype, op, sample: (
            op.full_name == "clone"
            and "noncontig_holes" in sample.name
            and sample.kwargs.get("memory_format", None) == torch.preserve_format
        ),
        name="clone_unbind_data_dependency_backward",
    ),
    *COMPILE_FORWARD_FAILURES,
    *BACKWARD_FAILURES,
]

COMPARE_TENSOR_COMPONENT_EQUALITY = {
    # masked_select is expected to output a different shape
    "masked_select",
}


# OpInfo-based NJT tests. These tests utilize an NJT-specific op_db generated from the standard
# op_db. Note that certain tradeoffs were made wrt coverage vs. time spent running tests:
#   * All tests run with dtype=torch.float32 only
class TestNestedTensorOpInfo(NestedTensorTestCase):
    # TODO: move this
    def _gen_grad_outputs(self, out_val):
        if isinstance(out_val, (list, tuple)):
            return tuple(torch.ones_like(c) for c in out_val)
        else:
            return (torch.ones_like(out_val),)

    # Returns a context manager xfailing / skipping only for expected errors.
    def maybe_skip_or_xfail(self, rules, device, dtype, op, sample):
        if rules is None or len(rules) == 0:
            return contextlib.nullcontext()

        for rule in rules:
            if rule.match(device, dtype, op, sample):
                log.debug(
                    "matched %s rule '%s': %s %s %s %s",
                    rule.type,
                    rule.name,
                    op.full_name,
                    device,
                    dtype,
                    sample,
                )
                return rule.get_context(self)

        log.debug("matched no rules: %s %s %s %s", op.full_name, device, dtype, sample)
        return contextlib.nullcontext()

    @ops([op for op in njt_op_db if op.supports_njt], allowed_dtypes=(torch.float32,))
    def test_forward(self, device, dtype, op):
        for i, sample in enumerate(
            op.sample_inputs(device=device, dtype=dtype, requires_grad=False)
        ):
            maybe_skip_or_xfail = self.maybe_skip_or_xfail(
                FORWARD_FAILURES, device, dtype, op, sample
            )
            with self.subTest(sample=sample, i=i), maybe_skip_or_xfail:
                # compare to reference, but expect different nested int
                out = op.op(sample.input, *sample.args, **sample.kwargs)
                out_ref = op.ref(op, sample)
                self.assertEqualIgnoringNestedInts(out, out_ref)

                # TODO: Revisit once https://github.com/pytorch/pytorch/pull/138369 lands
                # TODO: Add xfails for other inplace ops instead of hardcoding
                if op.inplace_variant and "index_put" in op.full_name:
                    op.inplace_variant(sample.input, *sample.args, **sample.kwargs)
                    self.assertEqualIgnoringNestedInts(sample.input, out_ref)

    @ops(
        [op for op in njt_op_db if op.supports_njt and op.supports_autograd],
        allowed_dtypes=(torch.float32,),
    )
    def test_backward(self, device, dtype, op):
        for i, sample in enumerate(
            op.sample_inputs(device=device, dtype=dtype, requires_grad=True)
        ):
            maybe_skip_or_xfail = self.maybe_skip_or_xfail(
                BACKWARD_FAILURES, device, dtype, op, sample
            )
            with self.subTest(sample=sample, i=i), maybe_skip_or_xfail:
                # compare to reference, but expect different nested int
                out = op.op(sample.input, *sample.args, **sample.kwargs)
                out_ref = op.ref(op, sample)
                self.assertEqualIgnoringNestedInts(out, out_ref)

                inps, _ = tree_flatten((sample.input, sample.args, sample.kwargs))
                g_inps = [
                    inp
                    for inp in inps
                    if isinstance(inp, torch.Tensor) and inp.requires_grad
                ]
                if len(g_inps) > 0:
                    grads = torch.autograd.grad(
                        out, inputs=g_inps, grad_outputs=self._gen_grad_outputs(out)
                    )

                    grads_ref = torch.autograd.grad(
                        out_ref,
                        inputs=g_inps,
                        grad_outputs=self._gen_grad_outputs(out_ref),
                    )

                    self.assertEqualNoncontigAware(grads, grads_ref)

    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    @ops([op for op in njt_op_db if op.supports_njt], allowed_dtypes=(torch.float32,))
    def test_compile_forward(self, device, dtype, op):
        for i, sample in enumerate(
            op.sample_inputs(device=device, dtype=dtype, requires_grad=False)
        ):
            maybe_skip_or_xfail = self.maybe_skip_or_xfail(
                COMPILE_FORWARD_FAILURES, device, dtype, op, sample
            )
            with self.subTest(sample=sample, i=i), maybe_skip_or_xfail:
                torch.compiler.reset()

                op_fn = op.op

                def f(*args, **kwargs):
                    return op_fn(*args, **kwargs)

                compiled_f = torch.compile(
                    f, fullgraph=True, backend="aot_eager_decomp_partition"
                )

                out_ref = f(sample.input, *sample.args, **sample.kwargs)
                out_compile = compiled_f(sample.input, *sample.args, **sample.kwargs)

                if op.full_name in COMPARE_TENSOR_COMPONENT_EQUALITY:
                    self.assertEqualIgnoringNestedInts(out_compile, out_ref)
                else:
                    self.assertEqual(out_compile, out_ref)

                # TODO: Revisit once https://github.com/pytorch/pytorch/pull/138369 lands
                # TODO: Add xfails for other inplace ops instead of hardcoding
                if op.inplace_variant and "index_put" in op.full_name:
                    op_fn = op.inplace_variant

                    def in_f(*args, **kwargs):
                        return op_fn(*args, **kwargs)

                    compiled_in_f = torch.compile(
                        in_f, fullgraph=True, backend="aot_eager_decomp_partition"
                    )

                    compiled_in_f(sample.input, *sample.args, **sample.kwargs)
                    if op.full_name in COMPARE_TENSOR_COMPONENT_EQUALITY:
                        self.assertEqualIgnoringNestedInts(sample.input, out_ref)
                    else:
                        self.assertEqual(sample.input, out_ref)

    @ops(
        [op for op in njt_op_db if op.supports_njt and op.supports_autograd],
        allowed_dtypes=(torch.float32,),
    )
    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_compile_backward(self, device, dtype, op):
        for i, sample in enumerate(
            op.sample_inputs(device=device, dtype=dtype, requires_grad=True)
        ):
            maybe_skip_or_xfail = self.maybe_skip_or_xfail(
                COMPILE_BACKWARD_FAILURES, device, dtype, op, sample
            )
            with self.subTest(sample=sample, i=i), maybe_skip_or_xfail:
                torch.compiler.reset()

                op_fn = op.op

                def f(*args, **kwargs):
                    return op_fn(*args, **kwargs)

                compiled_f = torch.compile(
                    f, fullgraph=True, backend="aot_eager_decomp_partition"
                )

                out_ref = f(sample.input, *sample.args, **sample.kwargs)
                out_compile = compiled_f(sample.input, *sample.args, **sample.kwargs)

                if op.full_name in COMPARE_TENSOR_COMPONENT_EQUALITY:
                    self.assertEqualIgnoringNestedInts(out_compile, out_ref)
                else:
                    self.assertEqual(out_compile, out_ref)

                inps, _ = tree_flatten((sample.input, sample.args, sample.kwargs))
                g_inps = [
                    inp
                    for inp in inps
                    if isinstance(inp, torch.Tensor) and inp.requires_grad
                ]
                if len(g_inps) > 0:
                    grads_compile = torch.autograd.grad(
                        out_compile,
                        inputs=g_inps,
                        grad_outputs=self._gen_grad_outputs(out_compile),
                    )

                    grads_ref = torch.autograd.grad(
                        out_ref,
                        inputs=g_inps,
                        grad_outputs=self._gen_grad_outputs(out_ref),
                    )

                    self.assertEqualNoncontigAware(grads_compile, grads_ref)

    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    @skipIfTorchDynamo(
        "Dynamo fails on pending unbacked symints at assertEqual(ref_y[0][0][0].item(), 2)"
    )
    def test_nested_tensor_non_contiguous_mutation(self):
        def fn(x, x0):
            x[0, 0, 0] = 2
            return x

        def _inp():
            base = torch.zeros(32, 3)
            v = base.t()
            return torch.nested.nested_tensor_from_jagged(
                v,
                offsets=torch.tensor([0, 2, 3]),
                # lengths=torch.tensor([2, 1]),
            ), torch.ones(2, 32)

        ref_x, ref_x0 = _inp()
        ref_y = fn(ref_x, ref_x0)

        self.assertEqual(ref_y[0][0][0].item(), 2)

        y = torch.compile(fn, fullgraph=True, backend="aot_eager")(*_inp())
        self.assertEqual(y[0][0][0], 2)


instantiate_parametrized_tests(TestNestedTensor)
instantiate_device_type_tests(TestNestedTensorDeviceType, globals())
instantiate_device_type_tests(TestNestedTensorAutograd, globals())
instantiate_device_type_tests(TestNestedTensorSubclass, globals())
instantiate_device_type_tests(TestNestedTensorOpInfo, globals())

if __name__ == "__main__":
    run_tests()
