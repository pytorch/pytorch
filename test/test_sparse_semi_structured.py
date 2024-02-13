# Owner(s): ["module: sparse"]
import itertools
import random
import unittest

import torch
from torch import nn
import torch.nn.functional as F

from torch.sparse import (
    SparseSemiStructuredTensor,
    SparseSemiStructuredTensorCUSPARSELT,
    SparseSemiStructuredTensorCUTLASS,
    to_sparse_semi_structured,
)

from torch.testing import make_tensor

from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)

from torch.testing._internal.common_dtype import all_types_and_complex
import torch._dynamo.test_case

from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    subtest,
    TestCase,
    TEST_WITH_ROCM,
    IS_WINDOWS,
)

import pytest

from torch.utils._triton import has_triton

SEMI_STRUCTURED_SUPPORTED_DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.int8]
SEMI_STRUCTURED_SUPPORTED_BACKENDS = {}

_IS_SM8X = False

if torch.cuda.is_available():
    _IS_SM8X = torch.cuda.get_device_capability(0)[0] == 8
    SEMI_STRUCTURED_SUPPORTED_BACKENDS["cutlass"] = SparseSemiStructuredTensorCUTLASS

    # check if cslt is available for now using this:
    # TODO when we add cusparselt as a backend, we can update this to be use torch.cusparselt.is_available()
    try:
        torch._cslt_compress(torch.ones(128, 256).cuda())
        SEMI_STRUCTURED_SUPPORTED_BACKENDS["cusparselt"] = SparseSemiStructuredTensorCUSPARSELT
    except Exception:
        pass

inference_dtypes = dtypes(torch.float16, torch.bfloat16, torch.float32, torch.int8)
training_dtypes = dtypes(torch.float16, torch.bfloat16)
parametrize_backends = parametrize("backend", SEMI_STRUCTURED_SUPPORTED_BACKENDS)

atol_rtol_kw = {
    torch.float16: {
        "rtol": 2e-3,
        "atol": 1e-4,
    },
    torch.bfloat16: {
        "rtol": 1e-1,
        "atol": 1e-1,
    },
}

def sparse24_largest_mask_2d(original):
    return to_sparse_semi_structured(original, training=True).to_dense().bool()

def sparsify24_dense(original):
    return sparse24_largest_mask_2d(original) * original

def rand_sparse_semi_structured_mask(
    r, c, dtype=torch.float16, device="cuda", choice=None
):
    """
    This function returns a 1:2 sparse matrix of size (r, c).
    Note that this means this matrix will also be 2:4 and 4:8 sparse as well.
    """

    choices = [[0, 1], [1, 0]]
    mask_entries = [choice or random.choice(choices) for i in range(r * c // 2)]

    return (
        torch.tensor(mask_entries, dtype=dtype, device=device)
        .reshape(r, c)
        .contiguous()
    )

def rand_sparse_semi_structured(r, c, dtype, device, choice=None):
    pattern = '2by4' if dtype != torch.float32 else '1by2'
    if pattern == '1by2':
        ksparse = 2
        choices = [
            [0, 1],
            [1, 0]
        ]
    elif pattern == '2by4':
        ksparse = 4
        choices = [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 1]
        ]
    mask_entries = [choice or random.choice(choices) for i in range(r * c // ksparse)]
    mask = torch.tensor(mask_entries, dtype=torch.bool).view(r, c).to(device)
    dense = make_tensor(r, c, dtype=dtype, device=device)
    dense[dense == 0] = 1  # To prevent zeros except where mask applied.
    dense = dense.masked_fill(~mask, 0)
    return dense


def rand_sparse_semi_structured_all_patterns(r, c, dtype, device):
    pattern = '2by4' if dtype != torch.float32 else '1by2'
    if pattern == '1by2':
        ksparse = 2
        choices = [
            [[0, 0], [0, 1]],
            [[0, 1], [0, 1]],
            [[1, 0], [1, 0]],
            [[1, 1], [1, 0]]
        ]
    elif pattern == '2by4':
        ksparse = 4
        choices = [
            [[0, 0, 0, 0], [0, 0, 1, 1]],
            [[0, 0, 0, 1], [0, 0, 1, 1]],
            [[0, 0, 1, 0], [0, 0, 1, 1]],
            [[0, 0, 1, 1], [0, 0, 1, 1]],
            [[0, 1, 0, 0], [0, 1, 1, 0]],
            [[0, 1, 0, 1], [0, 1, 0, 1]],
            [[0, 1, 1, 0], [0, 1, 1, 0]],
            [[0, 1, 1, 1], [0, 1, 0, 1]],
            [[1, 0, 0, 0], [1, 0, 1, 0]],
            [[1, 0, 0, 1], [1, 0, 0, 1]],
            [[1, 0, 1, 0], [1, 0, 1, 0]],
            [[1, 0, 1, 1], [1, 0, 0, 1]],
            [[1, 1, 0, 0], [1, 1, 0, 0]],
            [[1, 1, 0, 1], [1, 1, 0, 0]],
            [[1, 1, 1, 0], [1, 1, 0, 0]],
            [[1, 1, 1, 1], [1, 1, 0, 0]],
        ]
    mask_rows = [random.randint(0, len(choices) - 1) for i in range(r * c // ksparse)]

    COL_INV, COL_VAL = 0, 1
    mask_entries_inv = [choices[i][COL_INV] for i in mask_rows]
    mask_entries_val = [choices[i][COL_VAL] for i in mask_rows]
    mask_inv = torch.tensor(mask_entries_inv, dtype=torch.bool).view(r, c).to(device)
    mask_val = torch.tensor(mask_entries_val, dtype=torch.bool).view(r, c).to(device)
    dense = make_tensor(r, c, dtype=dtype, device=device)
    dense[dense == 0] = 1   # To prevent zeros except where mask below applied.
    dense_inv = dense.masked_fill(~mask_inv, 0)
    dense_val = dense_inv.masked_fill(~mask_val, 0)

    return dense_inv, dense_val


class SparseSemiStructuredTensorCompileTest(torch._dynamo.test_case.TestCase):

    def setUp(self):
        if not _IS_SM8X:
            self.skipTest('Only runs on SM80')
        super().setUp()

    def tearDown(self):
        super().tearDown()

    @staticmethod
    def _test_mlp_contiguous_relu_compile(backend, dense_input_shape):
        """
        Test nn.Linear + .contiguous() + nn.ReLU with SparseSemiStructuredTensor + torch.compile
        We expect:
            (1) The sparse tensor subclass should turn nn.Linear into `aten._structured_sparse_linear` + `aten.contiguous()`
            (2) Inductor should fuse the .contiguous() call into the relu
        """

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(128, 128)

            def forward(self, x):
                x = self.linear(x)
                x = x.contiguous()
                return torch.nn.functional.relu(x)

        input = torch.rand(dense_input_shape, device="cuda").half()
        model = Model().eval().cuda().half()
        mod_linear = model.linear
        m, n = mod_linear.weight.shape
        mask = torch.Tensor([1, 0, 0, 1]).tile((m, n // 4)).bool().cuda()
        # set masked weight
        mod_linear.weight = nn.Parameter(mod_linear.weight * mask)

        dense_result = model(input)
        mod_linear.weight = nn.Parameter(to_sparse_semi_structured(mod_linear.weight, backend=backend))
        sparse_result = model(input)

        model = torch.compile(model, backend="inductor", fullgraph=True)
        sparse_compile_result = model(input)

        # test that sparse_compile_result and dense_result are numerically close
        assert torch.allclose(dense_result, sparse_compile_result, rtol=1e-3, atol=1e-3)
        # assert sparse and sparse_compile have the same strides,
        # as meta registrations may return contiguous tensors when the output is transposed
        # https://github.com/pytorch/pytorch/pull/114477
        assert sparse_result.stride() == sparse_compile_result.stride()

    @unittest.skipIf(IS_WINDOWS, "torch.compile not supported on windows")
    @unittest.skipIf("cusparselt" not in SEMI_STRUCTURED_SUPPORTED_BACKENDS, "cusparselt not supported on this machine")
    def test_mlp_contiguous_relu_compile_cusparselt(self):
        """
        test for cuSPASRELt meta registrations (_cslt_sparse_mm) + torch.compile
        """
        for dense_input_shape in [(1, 128), (64, 128), (128, 128), (64, 128, 128)]:
            SparseSemiStructuredTensorCompileTest._test_mlp_contiguous_relu_compile("cusparselt", dense_input_shape)

    @unittest.skipIf(IS_WINDOWS, "torch.compile not supported on windows")
    def test_mlp_contiguous_relu_compile_cutlass(self):
        """
        test for CUTLASS meta registrations (_sparse_semi_structured_linear) + torch.compile
        """
        for dense_input_shape in [(1, 128), (64, 128), (128, 128), (64, 128, 128)]:
            SparseSemiStructuredTensorCompileTest._test_mlp_contiguous_relu_compile("cutlass", dense_input_shape)

    @unittest.skipIf(IS_WINDOWS, "torch.compile not supported on windows")
    def test_sp24_meta(self) -> None:
        x = torch.randn([1024, 512], device="meta", dtype=torch.float16)
        x_s = to_sparse_semi_structured(x, training=True, backend="cusparselt")
        assert x_s.shape == x.shape
        x_s_t = x_s.t()
        assert x_s_t.shape == x.t().shape


    # @parametrize_backends
    def test_sp24_compile(self) -> None:
        x = torch.randn([1024, 512], device="cuda", dtype=torch.float16, requires_grad=True)
        e = torch.eye(x.shape[0], x.shape[0], device="cuda", dtype=torch.float16)

        def fn(x, e):
            y = to_sparse_semi_structured(x, backend="cusparselt", training=True)
            y = y.t()
            return x @ y

        # Eager
        output = fn(x, e)
        output.backward(output)
        # Torch compile
        output = torch.compile(fn)(x, e)
        output.backward(output)

class TestSparseSemiStructured(TestCase):

    def setUp(self):
        if not _IS_SM8X:
            self.skipTest('Only runs on SM80')

    @inference_dtypes
    @parametrize_backends
    def test_to_sparse_semi_structured(self, dtype, backend):
        A = rand_sparse_semi_structured_mask(128, 256, dtype=dtype)
        A_sparse = to_sparse_semi_structured(A, backend=backend)

        assert A.shape == A_sparse.shape
        assert A.device == A_sparse.device
        assert A.dtype == A_sparse.dtype

        assert isinstance(A, torch.Tensor)
        assert isinstance(A_sparse, SparseSemiStructuredTensor)

    @inference_dtypes
    @parametrize_backends
    @parametrize("dense_input_shape", [(128, 1), (128, 64), (128, 128)])
    def test_mm_sparse_first_NN(self, dense_input_shape, dtype, device, backend):
        """
        Ensure torch.mm(A_sparse, B) is correct for float16 and will throw error for int8
        """
        A = rand_sparse_semi_structured_mask(256, 128, dtype=dtype)
        A_sparse = to_sparse_semi_structured(A, backend=backend)

        B = torch.rand(dense_input_shape, device=A_sparse.device).to(dtype)

        # Currently we don't support int matmul on GPU, so evaluate on CPU and copy over
        if dtype is torch.int8:
            regex = "two_four_sgemm_cutlass_dispatch_layouts" if backend == "cutlass" else "CUDA error: operation not supported when calling `cusparseLtMatmulDescriptorInit"

            with self.assertRaisesRegex(RuntimeError, regex):
                sparse_result = torch.mm(A_sparse, B)
        else:
            dense_result = torch.mm(A, B)
            sparse_result = torch.mm(A_sparse, B)
            assert torch.allclose(dense_result, sparse_result, rtol=1e-3, atol=1e-3)

    @inference_dtypes
    @parametrize_backends
    @parametrize("dense_input_shape", [(1, 128), (64, 128), (128, 128)])
    def test_mm_sparse_first_NT(self, dense_input_shape, dtype, device, backend):
        """
        Ensure torch.mm(A_sparse, B.t()) is correct for float16/bfloat16
        and will throw an error for int8 + padding
        """
        A = rand_sparse_semi_structured_mask(256, 128, dtype=dtype)
        A_sparse = to_sparse_semi_structured(A, backend=backend)

        B = torch.rand(dense_input_shape, device=A_sparse.device).to(dtype)

        # Currently we don't support int matmul on GPU, so evaluate on CPU and copy over
        if dtype is torch.int8 and dense_input_shape in {(1, 128)}:
            # padding with int8 throws an error because transposing B yields a contiguous output
            # and row-row 2:4 sparse @ dense with NN is not supported by cuSPARSELt or CUTLASS.
            if backend == "cutlass":
                with self.assertRaisesRegex(RuntimeError, "two_four_sgemm_cutlass_dispatch_layouts"):
                    sparse_result = torch.mm(A_sparse, B.t())
            else:
                with self.assertRaisesRegex(RuntimeError,
                                            "CUDA error: operation not supported when calling `cusparseLtMatmulDescriptorInit"):
                    sparse_result = torch.mm(A_sparse, B.t())
        elif dtype is torch.int8:
            # test transpose
            dense_result = torch.mm(A.cpu(), B.t().cpu()).to(device, dtype=torch.int8)
            sparse_result = torch.mm(A_sparse, B.t())
            assert torch.allclose(dense_result, sparse_result, rtol=1e-3, atol=1e-3)
        else:
            # test transpose
            dense_result = torch.mm(A, B.t())
            sparse_result = torch.mm(A_sparse, B.t())
            assert torch.allclose(dense_result, sparse_result, rtol=1e-3, atol=1e-3)

    @inference_dtypes
    @parametrize("dense_input_shape", [(1, 128), (64, 128), (128, 128)])
    @parametrize_backends
    def test_mm_sparse_first_TN(self, dtype, dense_input_shape, device, backend):
        """
        Ensure torch.mm(A_sparse.t(), B) throws error
        """
        A = rand_sparse_semi_structured_mask(128, 256, dtype=dtype)
        A_sparse = to_sparse_semi_structured(A, backend=backend)

        B = torch.rand(dense_input_shape, device=A_sparse.device).to(dtype)

        with self.assertRaisesRegex(
            NotImplementedError,
            r"`SparseSemiStructuredTensor.*` matmul: operation is not supported",
        ):
            torch.mm(A_sparse.t(), B)

    @inference_dtypes
    @parametrize("dense_input_shape", [(1, 128), (64, 128), (128, 128)])
    @parametrize_backends
    def test_mm_sparse_second_NT(self, dense_input_shape, dtype, device, backend):
        """
        Ensure torch.mm(A, B_sparse.t()) is correct
        """
        B = rand_sparse_semi_structured_mask(256, 128, dtype=dtype)
        B_sparse = to_sparse_semi_structured(B, backend=backend)

        A = torch.rand(dense_input_shape, device=B_sparse.device).to(dtype)

        # Currently we don't support int matmul on GPU, so evaluate on CPU and copy over
        if dtype is torch.int8:
            dense_result = torch.mm(A.cpu(), B.t().cpu()).to(device, dtype=torch.int8)
            sparse_result = torch.mm(A, B_sparse.t())
        else:
            dense_result = torch.mm(A, B.t())
            sparse_result = torch.mm(A, B_sparse.t())

        assert torch.allclose(dense_result, sparse_result, rtol=1e-3, atol=1e-3)

    @inference_dtypes
    @parametrize("dense_input_shape", [(1, 128), (64, 128), (128, 128)])
    @parametrize_backends
    def test_mm_sparse_second_NN(self, dense_input_shape, dtype, device, backend):
        """
        Ensure torch.mm(A, B_sparse) throws error
        """
        B = rand_sparse_semi_structured_mask(256, 128, dtype=dtype)
        B_sparse = to_sparse_semi_structured(B, backend=backend)

        A = torch.rand(dense_input_shape, device=B_sparse.device).to(dtype)

        with self.assertRaisesRegex(
            NotImplementedError,
            r"`SparseSemiStructuredTensor.*` matmul: operation is not supported",
        ):
            sparse_result = torch.mm(A, B_sparse)

    @parametrize("dense_input_shape", [(1, 128), (64, 128), (128, 128), (64, 128, 128)])
    @parametrize("inference_mode", [subtest(True), subtest(False)])
    @parametrize_backends
    def test_linear(self, dense_input_shape, inference_mode, device, backend):
        """
        Test nn.Linear has the same numerics
        """
        input = torch.rand((dense_input_shape), device=device).half()
        model = nn.Linear(128, 256).to(device).half()
        m, n = model.weight.shape
        mask = rand_sparse_semi_structured_mask(m, n, device=device, dtype=torch.bool)
        # set masked weight
        model.weight = nn.Parameter(model.weight * mask)

        dense_result = model(input)

        model.weight = nn.Parameter(to_sparse_semi_structured(model.weight, backend=backend))

        if inference_mode:
            with torch.inference_mode():
                sparse_result = model(input)
        else:
            sparse_result = model(input)

        assert torch.allclose(dense_result, sparse_result, rtol=1e-3, atol=1e-3)

    @parametrize("dense_input_shape", [(1, 128), (64, 128), (128, 128), (64, 128, 128)])
    @parametrize_backends
    def test_mlp(self, device, dense_input_shape, backend):
        input = torch.rand(dense_input_shape, device=device).half()
        model = (
            nn.Sequential(
                nn.Linear(128, 256),
                nn.Linear(256, 128),
            )
            .half()
            .to(device)
        )

        for i in range(2):
            m, n = model[i].weight.shape
            mask = rand_sparse_semi_structured_mask(
                m, n, device=device, dtype=torch.bool
            )
            # set masked weight
            model[i].weight = nn.Parameter(model[i].weight * mask)

        dense_result = model(input)

        for i in range(2):
            model[i].weight = nn.Parameter(to_sparse_semi_structured(model[i].weight, backend=backend))

        sparse_result = model(input)

        assert torch.allclose(dense_result, sparse_result, rtol=1e-3, atol=1e-3)

    @parametrize_backends
    def test_values(self, backend):
        A = rand_sparse_semi_structured_mask(128, 128)
        A_sparse = to_sparse_semi_structured(A, backend=backend)
        assert A_sparse.values().shape == (128, 64)
        assert (A_sparse.values() == 1).all()

    @parametrize_backends
    def test_indices(self, backend):
        A = rand_sparse_semi_structured_mask(128, 128)
        A_sparse = to_sparse_semi_structured(A, backend=backend)
        assert A_sparse.indices().shape == (128, 8)

    @inference_dtypes
    @parametrize_backends
    def test_min_sparse_shape(self, dtype, device, backend):
        config = SEMI_STRUCTURED_SUPPORTED_BACKENDS[backend]._DTYPE_SHAPE_CONSTRAINTS[dtype]
        A = rand_sparse_semi_structured_mask(config.sparse_min_rows, config.sparse_min_cols, dtype=dtype, device=device)
        A_sparse = to_sparse_semi_structured(A, backend=backend)
        B = torch.rand((config.sparse_min_cols, config.dense_min_cols), device=device).to(dtype)
        if dtype == torch.int8:
            dense_res = torch.mm(A.cpu(), B.cpu()).to(device, dtype=torch.int8)
            # int8 sparse matmul not supported for R/R -> R layout, so we transpose one of the arguments to get R/C -> R
            B_t = B.t().contiguous()
            sparse_res = torch.mm(A_sparse, B_t.t())
        else:
            dense_res = torch.mm(A, B)
            sparse_res = torch.mm(A_sparse, B)
        assert torch.allclose(sparse_res, dense_res, rtol=1e-3, atol=1e-3)

    @inference_dtypes
    @parametrize_backends
    def test_unsupported_shape(self, dtype, device, backend):
        A = rand_sparse_semi_structured_mask(2, 2, dtype=dtype, device=device)
        with self.assertRaisesRegex(RuntimeError, "Error original_tensor.shape"):
            A_sparse = to_sparse_semi_structured(A, backend=backend)

    @dtypes(*all_types_and_complex())
    @parametrize_backends
    def test_unsupported_dtype(self, dtype, device, backend):
        A = rand_sparse_semi_structured_mask(128, 128, dtype=dtype, device=device)

        if dtype not in SEMI_STRUCTURED_SUPPORTED_DTYPES:
            with self.assertRaisesRegex(RuntimeError, "Error original_tensor.dtype"):
                A_sparse = to_sparse_semi_structured(A, backend=backend)
        else:
            A_sparse = to_sparse_semi_structured(A, backend=backend)

    @parametrize_backends
    def test_unsupported_dim(self, device, backend):
        A = torch.rand(128, 128, 128, device=device, dtype=torch.float16)

        with self.assertRaisesRegex(RuntimeError, "Error original_tensor.dim"):
            A_sparse = to_sparse_semi_structured(A, backend=backend)


def create_random_mask(shape) -> torch.Tensor:
    r = random.Random(0)
    mask = torch.zeros(shape, dtype=torch.bool)
    for line in range(mask.shape[0]):
        for col in range(0, mask.shape[1], 4):
            sparsity = r.choice(
                [
                    [False, False, True, True],
                    [False, True, False, True],
                    [True, False, False, True],
                    [False, True, True, False],
                    [True, False, True, False],
                    [True, True, False, False],
                ]
            )
            mask[line, col : col + 4] = torch.tensor(sparsity, dtype=torch.bool)
    return mask

class TestSparseSemiStructuredTraining(TestCase):

    def setUp(self):
        if not _IS_SM8X:
            self.skipTest('Only runs on SM80')

    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("backend", SEMI_STRUCTURED_SUPPORTED_BACKENDS)
    def test_autocast(self, dtype, backend) -> None:
        N = 128
        inp = torch.randn([N, N], dtype=torch.float32, device="cuda")
        W = torch.randn([N, N], dtype=torch.float32, device="cuda")
        sInp = to_sparse_semi_structured(inp.to(dtype=dtype), backend=backend, training=True)
        y = sInp @ W.to(dtype=dtype)
        with torch.autocast("cuda", dtype=dtype):
            sInp_ac = to_sparse_semi_structured(inp, backend=backend, training=True)
            y_ac = sInp_ac @ W

        assert torch.allclose(
            sInp.to_dense(),
            sInp_ac.to_dense(),
            **atol_rtol_kw[dtype],
        )
        assert torch.allclose(y, y_ac, **atol_rtol_kw[dtype])

    @training_dtypes
    def test_pruning_algo_causal1122(self, dtype) -> None:
        inp = torch.tensor(
            [[4, 3, 2, 1], [-1, -3, 0.6, 0.5], [1, 2, 3, 4], [10, 2, -1, 5]],
            device="cuda",
            dtype=dtype,
        )
        inp = F.pad(inp, (0, 128 - 4, 0, 128 - 4), "constant", 1)
        sInp = SparseSemiStructuredTensorCUTLASS.from_dense_fast(inp, algo="causal1122")

        mask = sInp.to_dense() / inp
        assert mask[:4, :4].int().tolist() == [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 1],
        ]

    @training_dtypes
    @parametrize_backends
    def test_pruning_algo_largest_abs_values_greedy(self, dtype, backend) -> None:
        inp = torch.tensor(
            [[4, 3, 2, 1], [-1, -3, 0.6, 0.5], [1, 2, 3, 4], [10, 2, -1, 5]],
            device="cuda",
            dtype=dtype,
        )
        inp = F.pad(inp, (0, 128 - 4, 0, 128 - 4), "constant", 1)
        sInp = SEMI_STRUCTURED_SUPPORTED_BACKENDS[backend].from_dense_fast(inp, algo="largest_abs_values_greedy")

        mask = sInp.to_dense() / inp
        assert mask[:4, :4].int().tolist() == [
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 1],
        ]

    def test_detach_requires_grad(self) -> None:
        x = torch.randn([128, 64], device="cuda", dtype=torch.float16, requires_grad=True)
        xs = to_sparse_semi_structured(x, training=True)
        assert xs.requires_grad

        # `detach` behavior
        xs2 = xs.detach()
        assert not xs2.requires_grad
        assert not (xs2 * 2).requires_grad

        xs2.requires_grad_(True)
        assert xs2.requires_grad
        ys = xs2 * 2
        assert ys.requires_grad
        ys.backward(ys)

    def test_detach2(self) -> None:
        x = torch.randn([128, 64], device="cuda", dtype=torch.float16, requires_grad=False)
        assert not to_sparse_semi_structured(x, training=True).requires_grad
        x.requires_grad_(True)
        xs = to_sparse_semi_structured(x, training=True)
        assert xs.requires_grad
        xs2 = xs.detach()
        xs2.requires_grad_(True)
        xs3 = xs2 * 2
        assert xs3.requires_grad
        xs3.backward(xs3)
        assert xs2.grad is not None
        assert x.grad is None

    @training_dtypes
    def test_gemm(self, dtype) -> None:
        M, N, K = 32, 32, 64
        a = torch.randn([M, K], device="cuda", dtype=dtype)
        b = torch.randn([K, N], device="cuda", dtype=dtype)
        mask = rand_sparse_semi_structured_mask(M, K, dtype=torch.bool)

        a.masked_fill_(~mask, 0)

        a_sparse = to_sparse_semi_structured(a, backend="cutlass")

        masked_a = a * mask
        ref_out = masked_a @ b
        sp24_out = a_sparse @ b
        assert torch.allclose(ref_out, sp24_out, **atol_rtol_kw[dtype])


    @training_dtypes
    @parametrize_backends
    def test_pack_both_ways_meta_correctness(self, dtype, backend) -> None:
        M, N = 128, 256
        # Construct x to make sure we always have exactly 8 elements per 4x4 tile
        a = (4 * torch.arange(8))[:, None] + torch.arange(8)[None, :]
        a = a.repeat(M // 8, N // 8)
        assert a.shape == (M, N)
        a = a.cuda().to(dtype)
        b = torch.randn([a.shape[1], 128], device="cuda", dtype=dtype)
        a_sparse = to_sparse_semi_structured(a, backend=backend, training=True)

        mask_dense = sparse24_largest_mask_2d(a).to(dtype)

        if backend == "cutlass":
            assert isinstance(a_sparse, SparseSemiStructuredTensorCUTLASS)
            sparse_mask = to_sparse_semi_structured(mask_dense, backend=backend, training=True)
            assert torch.allclose(a_sparse.meta.view(torch.short), sparse_mask.meta)

        ref_gemm = (mask_dense * a) @ b
        pack_gemm = a_sparse @ b
        assert torch.allclose(ref_gemm, pack_gemm, **atol_rtol_kw[dtype])

    @training_dtypes
    def test_pack_both_ways_id(self, dtype) -> None:
        N = 512
        torch.manual_seed(0)
        a = torch.randn([N, N], dtype=dtype, device="cuda")
        b = torch.eye(N, dtype=dtype, device="cuda")

        packed, meta, packed_t, meta_t = torch.ops.sparse.sparse24_sparsify_both_ways(a)[
            :4
        ]
        # Heuristic to ensure we pack the same values
        assert torch.allclose(
            packed.to(torch.float64).sum(), packed_t.to(torch.float64).sum()
        )

        mask_dense = sparse24_largest_mask_2d(a.to(dtype))

        ref_gemm = mask_dense * a
        # Test A@B
        pack_gemm = torch._sparse_semi_structured_linear(b.t(), packed, meta).t()
        max_diff = (ref_gemm - pack_gemm).abs().argmax()
        assert torch.allclose(
            ref_gemm, pack_gemm,
            **atol_rtol_kw[dtype]
        ), f"packed is wrong at pos: ({max_diff // N}, {max_diff % N})"
        # Test A.t@B
        pack_gemm = torch._sparse_semi_structured_linear(b.t(), packed_t, meta_t)
        max_diff = (ref_gemm - pack_gemm).abs().argmax()

        assert torch.allclose(
            ref_gemm, pack_gemm,
            **atol_rtol_kw[dtype]
        ), f"packed_t is wrong at pos: ({max_diff // N}, {max_diff % N})"

    @training_dtypes
    def test_pack_both_ways_edge_case1(self, dtype) -> None:
        # In this case, the heuristic will keep 7 values out of 16
        # instead of 8. let's see how the kernel handles this
        quad = torch.tensor(
            [
                [2, -1, -2, -3],  # Should be packed as `2 <null>`
                [-1, 8, -1, 6],
                [-1, -1, 4, 5],
                [-1, 3, 7, -1],
            ],
            dtype=dtype,
            device="cuda",
        )
        a = torch.randn([32, 64], dtype=dtype, device="cuda")
        a[:4, :4] = quad
        packed, meta, packed_t, meta_t = torch.ops.sparse.sparse24_sparsify_both_ways(a)[:4]
        # Check first line in A
        assert packed[0, 0].item() == 2
        assert packed[0, 1].item() == 0
        # And first column in A.t
        assert packed_t[0, 0].item() == 2
        assert packed_t[0, 1].item() == 0

    @training_dtypes
    def test_sp24_apply(self, dtype) -> None:
        M, N = 256, 1024
        x = torch.randn([M, N], dtype=dtype, device="cuda")
        (
            packed,
            meta,
            packed_t,
            meta_t,
            threads_masks,
        ) = torch.ops.sparse.sparse24_sparsify_both_ways(x)
        packed2, packed_t2 = torch.ops.sparse.sparse24_apply(x, threads_masks)
        assert torch.allclose(packed, packed2)
        assert torch.allclose(packed_t, packed_t2)

    @training_dtypes
    def test_sp24_api_different_pattern(self, dtype) -> None:
        M, N = 256, 256
        x = torch.randn([M, N], dtype=dtype, device="cuda")
        y = torch.randn([M, N], dtype=dtype, device="cuda")
        sx = to_sparse_semi_structured(x, training=True)
        sy = to_sparse_semi_structured(y, training=True)
        # Can't add with different sparsity pattern
        with pytest.raises(ValueError):
            sx + sy
        # Ok, same sparsity pattern
        assert isinstance(sx + sx, SparseSemiStructuredTensor)
        # Ok, sharing sparsity pattern of x
        sy2 = sx.from_dense_like(y)
        assert isinstance(sx + sy2, SparseSemiStructuredTensor)

    @training_dtypes
    def test_sp24_api_different_pattern_transposed(self, dtype) -> None:
        N = 256
        x = torch.randn([N, N], dtype=dtype, device="cuda")
        sx = to_sparse_semi_structured(x, training=True)
        sxt = sx.t()
        assert isinstance(sxt, SparseSemiStructuredTensor)
        # Can't add with different sparsity pattern
        with pytest.raises(ValueError):
            sx + sxt
        # But this should work
        sx + sxt.t()
        # And we should be able to sparsify with transposed pattern
        sxt2 = sxt.from_dense_like(x.t())
        assert torch.allclose(sxt2.packed, sxt.packed)
        assert torch.allclose(sxt2.packed_t, sxt.packed_t)

    @training_dtypes
    @parametrize_backends
    def test_sp24_transpose_invariant(self, dtype, backend) -> None:
        M, N = 128, 256

        torch.manual_seed(0)
        r = random.Random(0)

        def gen4x4():
            # Create a 4x4 tile that can be 24 sparsified perfectly
            values = [
                [1, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 1, 1],
                [1, 0, 0, 1],
            ]
            c1, c2 = r.sample([0, 1, 2, 3], 2)
            r1, r2 = r.sample([0, 1, 2, 3], 2)
            values[r1], values[r2] = values[r2], values[r1]
            for i in range(4):
                values[i][c1], values[i][c2] = values[i][c2], values[i][c1]
            return values

        a = torch.zeros([M, N], device="cuda", dtype=torch.float16)
        assert M % 4 == 0 and N % 4 == 0
        for m in range(0, M, 4):
            for n in range(0, N, 4):
                a[m : m + 4, n : n + 4] = torch.tensor(
                    gen4x4(), device="cuda", dtype=torch.float16
                )
        a = a * torch.randn_like(a).abs()

        # Sparsify `a`` and `a.t()`
        a_s = to_sparse_semi_structured(a, training=True, backend=backend)
        a_t_s = to_sparse_semi_structured(a.t().contiguous(), training=True, backend=backend)
        assert torch.allclose(a_s.to_dense(), a)
        assert torch.allclose(a_t_s.t().to_dense(), a)  # type: ignore
        assert torch.allclose(a_t_s.to_dense().t(), a)

    @training_dtypes
    def test_sp24_matmuls(self, dtype) -> None:
        M, N, K = 64, 256, 1024
        a = torch.randn([M, K], device="cuda", dtype=dtype)
        b = torch.randn([K, N], device="cuda", dtype=dtype)
        a_m = sparse24_largest_mask_2d(a)
        b_m = sparse24_largest_mask_2d(b)
        a_s = to_sparse_semi_structured(a, training=True)
        b_s = to_sparse_semi_structured(b, training=True)

        assert torch.allclose(a_s @ b, (a * a_m) @ b, **atol_rtol_kw[dtype])
        assert torch.allclose(a @ b_s, a @ (b * b_m), **atol_rtol_kw[dtype])
        assert torch.allclose(
            a @ a_s.t(), a @ (a * a_m).t(), **atol_rtol_kw[dtype]
        )
        assert torch.allclose(
            a_s.t() @ a, (a * a_m).t() @ a, **atol_rtol_kw[dtype]
        )

    def test_sp24_matmuls_mat_vec(self) -> None:
        a = torch.randn([64, 128], device="cuda", dtype=torch.float16)
        b = torch.randn([128], device="cuda", dtype=torch.float16)
        a_m = sparse24_largest_mask_2d(a)
        a_s = to_sparse_semi_structured(a, training=True)

        with pytest.raises(NotImplementedError):
            assert torch.allclose(a_s @ b, (a * a_m) @ b, **atol_rtol_kw[a.dtype])


    def test_sp24_matmuls_bmm(self) -> None:
        a = torch.randn([64, 128], device="cuda", dtype=torch.float16)
        b = torch.randn([5, 6, 128], device="cuda", dtype=torch.float16)
        a_m = sparse24_largest_mask_2d(a)
        a_s = to_sparse_semi_structured(a, training=True)

        with pytest.raises(NotImplementedError):
            assert torch.allclose(a_s @ b, (a * a_m) @ b, **atol_rtol_kw[a.dtype])

    @training_dtypes
    @parametrize("act", [F.gelu, F.relu])
    def test_sp24_api_mlp_act24_correctness(self, dtype, act) -> None:
        B, in_ft, hid_ft, out_ft = 256, 2048, 6144, 2048
        torch.manual_seed(0)
        x = torch.randn([B, in_ft], dtype=dtype, device="cuda", requires_grad=True)
        w1 = (
            torch.randn([in_ft, hid_ft], dtype=dtype, device="cuda", requires_grad=False)
            * 0.01
        )
        w2 = (
            torch.randn([hid_ft, out_ft], dtype=dtype, device="cuda", requires_grad=False)
            * 0.01
        )
        grad = (
            torch.randn([B, out_ft], dtype=dtype, device="cuda", requires_grad=False) * 0.1
        )
        w1.requires_grad_(True)
        w2.requires_grad_(True)

        params_with_grads = [x, w1, w2]

        # Run baseline
        x1 = x @ w1
        x1 = sparsify24_dense(x1)
        x1 = act(x1)
        out = x1 @ w2
        out.backward(grad)

        grads_ref = [t.grad for t in params_with_grads]
        for t in params_with_grads:
            t.grad = None

        # Run with sparsity
        x1 = x @ w1
        x1 = to_sparse_semi_structured(x1, training=True)
        x1 = act(x1)
        out = x1 @ w2
        out.backward(grad)

        for grad_name, grad_ref, grad_calc in zip(
            ["x", "w1", "w2"], grads_ref, [t.grad for t in params_with_grads]
        ):
            assert grad_calc is not None, grad_name
            assert grad_ref is not None, grad_name
            assert torch.allclose(grad_calc, grad_ref, **atol_rtol_kw[dtype])

    @training_dtypes
    def test_sp24_api_swiglu_correctness(self, dtype) -> None:
        B, in_ft, hid_ft, out_ft = 256, 2048, 6144 // 2, 2048
        torch.manual_seed(0)
        x = torch.randn([B, in_ft], dtype=dtype, device="cuda", requires_grad=True)
        w1 = (
            torch.randn([in_ft, hid_ft], dtype=dtype, device="cuda", requires_grad=False)
            * 0.01
        )
        w2 = (
            torch.randn([in_ft, hid_ft], dtype=dtype, device="cuda", requires_grad=False)
            * 0.01
        )
        w3 = (
            torch.randn([hid_ft, out_ft], dtype=dtype, device="cuda", requires_grad=False)
            * 0.01
        )
        grad = (
            torch.randn([B, out_ft], dtype=dtype, device="cuda", requires_grad=False) * 0.1
        )
        w1.requires_grad_(True)
        w2.requires_grad_(True)
        w3.requires_grad_(True)

        params_with_grads = [x, w1, w2, w3]

        # Run baseline
        x1 = x @ w1
        x2 = x @ w2
        x1s = sparsify24_dense(F.silu(x1))
        hid = x1s * x2
        out = hid @ w3
        out.backward(grad)

        grads_ref = [t.grad for t in params_with_grads]
        for t in params_with_grads:
            t.grad = None

        # Run with sparsity
        x1 = x @ w1
        x2 = x @ w2
        x1s = to_sparse_semi_structured(F.silu(x1), training=True)
        hid = x1s * x2
        out = hid @ w3
        out.backward(grad)

        for grad_name, grad_ref, grad_calc in zip(
            ["x", "w1", "w2", "w3"], grads_ref, [t.grad for t in params_with_grads]
        ):
            assert grad_calc is not None, grad_name
            assert grad_ref is not None, grad_name
            assert torch.allclose(grad_calc, grad_ref, **atol_rtol_kw[dtype])


    @training_dtypes
    @parametrize("input_rowmajor", [subtest(True), subtest(False)])
    def test_sparsify24_like_dense(self, dtype, input_rowmajor):
        M, N = 128, 256
        if input_rowmajor:
            x = torch.randn([M, N], dtype=dtype, device="cuda")
        else:
            x = torch.randn([N, M], dtype=dtype, device="cuda").t()
        sx = to_sparse_semi_structured(x.contiguous(), training=True)
        sx_like = sx.from_dense_like(x, out_dense=True)
        assert torch.allclose(
            sx_like, sx.to_dense(), **atol_rtol_kw[dtype]
        )


    @training_dtypes
    @parametrize_backends
    def test_sparsify24_weights(self, dtype, backend):
        x = torch.randn([128, 512], dtype=dtype, device="cuda", requires_grad=True)
        w = torch.randn([1024, 512], dtype=dtype, device="cuda", requires_grad=True)

        flat_w = w.flatten()  # FSDP-like processing
        w = flat_w.reshape(w.shape)

        subclass = SEMI_STRUCTURED_SUPPORTED_BACKENDS[backend]
        sw = subclass.from_dense_fast(w, gradient="24dense")
        y = x @ sw.t()

        y.backward(y)


    @parametrize_backends
    @parametrize("with_bias", [False, True])
    def test_linear_dispatch_inference_mode(self, backend: str, with_bias: bool) -> None:
        B, ft_in, ft_out = 128, 256, 512
        x = torch.randn([B, ft_in], device="cuda", dtype=torch.float16)
        weight = torch.randn([ft_out, ft_in], device="cuda", dtype=torch.float16)
        bias = (
            torch.randn([ft_out], device="cuda", dtype=torch.float16) if with_bias else None
        )

        w_sparse = to_sparse_semi_structured(
            weight,
            backend=backend,
            training = True
        )
        # NOTE: When in `inference_mode`, PyTorch no longer dispatches to `addmm`, but to `linear`
        # so we need to support that as well
        with torch.inference_mode():
            out = F.linear(x, w_sparse, bias)
        out_ref = F.linear(x, w_sparse.to_dense(), bias)
        assert torch.allclose(out, out_ref, **atol_rtol_kw[x.dtype])

class TestSparseSemiStructuredCUTLASS(TestCase):
    """
    This contains CUTLASS specific tests for
         - torch._sparse_semi_structured_linear
    """
    def setUp(self):
        if not _IS_SM8X:
            self.skipTest('Only runs on SM80')
        if "cutlass" not in SEMI_STRUCTURED_SUPPORTED_BACKENDS:
            self.skipTest('CUTLASS not enabled')

    @unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support CUTLASS")
    @inference_dtypes
    def test_linear_cutlass(self, device, dtype):

        def run_test(batch_shape, m, n, k, device, dtype, dtype_out, add_bias, activation, rtol, atol):
            weight = rand_sparse_semi_structured(m, k, dtype, device)
            input = make_tensor((*batch_shape, n, k), dtype=dtype, device=device)
            bias = make_tensor((m,), dtype=dtype_out, device=device) if add_bias else None

            dtype_dense = torch.float32
            input_dense = input.to(dtype_dense)
            weight_dense = weight.to(dtype_dense)
            bias_dense = bias.to(dtype_dense) if add_bias else None
            output0 = torch.nn.functional.linear(input_dense, weight_dense, bias=bias_dense)
            if activation == "relu":
                relu = torch.nn.ReLU()
                output0 = relu(output0)
            elif activation == "silu":
                silu = torch.nn.SiLU()
                output0 = silu(output0)

            compressed = to_sparse_semi_structured(weight)

            weight_sparse = compressed.values()
            meta = compressed.indices()

            output1 = torch._sparse_semi_structured_linear(input, weight_sparse, meta, bias=bias, activation=activation,
                                                           out_dtype=dtype_out if dtype == torch.int8 else None)
            torch.testing.assert_close(output1.to(dtype_dense), output0, rtol=rtol, atol=atol)

        if dtype == torch.float32:
            # Inputs are converted to TF32 internally for sparse GEMM,
            # so make dense GEMM to do the same for matching results.
            orig = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = True

        batch_shapes = [[], [3], [3, 1]]
        dtype_out = {torch.int8: torch.int32, torch.half: torch.half, torch.bfloat16: torch.bfloat16, torch.float32: torch.float32}
        activations = [None, "relu", "silu"]
        rtol, atol = 1e-3, 1e-3
        if dtype == torch.bfloat16:
            rtol, atol = 5e-3, 5e-3
        elif dtype == torch.float32:
            rtol, atol = 1e-3, 75e-2
        for batch_shape, m, n, k, add_bias, activation in \
                itertools.product(batch_shapes, range(3), range(3), range(3), (False, True), activations):
            if activation == "silu" and dtype == torch.int8:
                continue  # SiLU not supported for integer inputs

            m = 2 ** m * 32
            n = 2 ** n * 32
            k = 2 ** k * 128
            run_test(batch_shape, m, n, k, device, dtype, dtype_out[dtype], add_bias, activation, rtol, atol)

        if dtype == torch.float32:
            torch.backends.cuda.matmul.allow_tf32 = orig


    @unittest.skipIf(not has_triton(), "Test needs triton and recent GPU arch")
    @inference_dtypes
    def test_conversions(self, device, dtype):

        def run_test(r, c, device, dtype):
            dense_ref = rand_sparse_semi_structured(r, c, dtype, device)

            compressed = to_sparse_semi_structured(dense_ref)

            # The torch.ops.aten._to_sparse_semi_structured operator
            # uses CUTLASS to perform conversion from given dense
            # matrix to the pair of corresponding sparse and metadata
            # matrices, with the later used here as a reference to
            # compare the metadata matrix produced by conversion
            # performed by SparseSemiStructuredTensor class
            # constructor against.
            _, meta_ref = torch.ops.aten._to_sparse_semi_structured(dense_ref)

            meta = compressed.indices()
            torch.testing.assert_close(meta, meta_ref, rtol=0, atol=0)

            dense = compressed.to_dense()
            torch.testing.assert_close(dense, dense_ref, rtol=0, atol=0)

        shapes = [[32, 128], [32, 256], [64, 128], [64, 256]]
        for r, c in shapes:
            run_test(r, c, device, dtype)

    @unittest.skipIf(not has_triton(), "Test needs triton and recent GPU arch")
    @inference_dtypes
    def test_conversions_all_patterns(self, device, dtype):
        r, c = 32, 128

        dense_inv, dense_val = rand_sparse_semi_structured_all_patterns(r, c, dtype, device)

        compressed = to_sparse_semi_structured(dense_inv)
        dense = compressed.to_dense()

        torch.testing.assert_close(dense, dense_val, rtol=0, atol=0)



CUSPARSELT_NUM_ALG_IDS = 4
CUSPARSELT_MIXED_DTYPE_SUPPORT = [torch.float16, torch.bfloat16, torch.int32]


class TestSparseSemiStructuredCUSPARSELT(TestCase):
    """
    This contains cuSPARSELt specific tests for
        torch._cslt_compress
        torch._cslt_sparse_mm
    """
    def setUp(self):
        if not _IS_SM8X:
            self.skipTest('Only runs on SM80')
        if "cusparselt" not in SEMI_STRUCTURED_SUPPORTED_BACKENDS:
            self.skipTest('cuSPARSELt not enabled')

    @parametrize("out_dtype", CUSPARSELT_MIXED_DTYPE_SUPPORT)
    @parametrize("dense_input_shape", [(128, 128)])
    def test_cslt_sparse_mm_mixed_dtype(self, dense_input_shape, out_dtype, device):
        A = rand_sparse_semi_structured_mask(128, 128, dtype=torch.int8)
        A_compressed = torch._cslt_compress(A)

        B = torch.rand(dense_input_shape, device=device).to(torch.int8)

        dense_result = torch.mm(A.cpu().to(torch.int64), B.t().cpu().to(torch.int64)).to(device, dtype=out_dtype)
        sparse_result = torch._cslt_sparse_mm(A_compressed, B.t(), out_dtype=out_dtype)
        assert torch.allclose(dense_result, sparse_result, rtol=1e-3, atol=1e-3)

    @training_dtypes
    def test_cslt_sparse_mm_alpha(self, dtype, device):
        A = torch.Tensor([0, 0, 1, 1]).tile((128, 64)).to(dtype).cuda()
        B = torch.ones((256, 128), device=device).to(dtype)
        alpha = torch.Tensor([2**(-i) for i in range(128)]).cuda()

        A_compressed = torch._cslt_compress(A)
        sparse_result = torch._cslt_sparse_mm(A_compressed, B, alpha=alpha)

        alpha_scaled = torch.stack([alpha] * 128).t()
        dense_result = alpha_scaled * torch.mm(A.to(torch.float32), B.to(torch.float32))
        dense_result = dense_result.to(dtype)

        assert torch.allclose(sparse_result, dense_result, rtol=1e-3, atol=1e-3)

    @parametrize("out_dtype", CUSPARSELT_MIXED_DTYPE_SUPPORT)
    def test_cslt_sparse_mm_alpha_mixed_dtype(self, out_dtype, device):
        A = torch.Tensor([0, 0, 10, 10]).tile((128, 64)).to(torch.int8).cuda()
        B = torch.ones((128, 256), device=device).to(torch.int8).t()
        alpha = torch.Tensor([2**(-i) if out_dtype is not torch.int32 else 1
                              for i in range(128)]).cuda()

        A_compressed = torch._cslt_compress(A)
        sparse_result = torch._cslt_sparse_mm(A_compressed, B, alpha=alpha, out_dtype=out_dtype).cpu()

        alpha_scaled = torch.stack([alpha] * 128).t()
        dense_result = alpha_scaled.cpu() * torch.mm(A.to(torch.int64).cpu(), B.to(torch.int64).cpu())
        dense_result = dense_result.to(out_dtype)

        assert torch.allclose(sparse_result, dense_result, rtol=1e-3, atol=1e-3)

    @parametrize("alg_id", range(CUSPARSELT_NUM_ALG_IDS))
    @inference_dtypes
    def test_cslt_sparse_mm_alg_id(self, device, dtype, alg_id):
        # alg_id=3 not supported for float32 dtype
        if dtype == torch.float32 and alg_id == 3:
            return
        A = rand_sparse_semi_structured_mask(128, 128, dtype=dtype)
        A_compressed = torch._cslt_compress(A)
        B = torch.ones((128, 128), device=device).to(dtype)

        A_compressed = torch._cslt_compress(A)
        sparse_result = torch._cslt_sparse_mm(A_compressed, B.t(), alg_id=alg_id)

        dense_result = torch.mm(A.to(torch.float32), B.to(torch.float32))
        dense_result = dense_result.to(dtype)

        assert torch.allclose(sparse_result, dense_result, rtol=1e-3, atol=1e-3)

    @inference_dtypes
    def test_cslt_sparse_mm_search(self, device, dtype):
        A = rand_sparse_semi_structured_mask(128, 128, dtype=dtype)
        A_compressed = torch._cslt_compress(A)
        B = torch.ones((128, 128), device=device).to(dtype)

        A_compressed = torch._cslt_compress(A)
        alg_id = torch._cslt_sparse_mm_search(A_compressed, B.t())
        # for cuSPARSELt v0.4.0 there is a bug where although there are 5 alg_ids, we run into an error
        # when setting using the last one (4)
        # in cuSPARSELt v0.5.0 there are only 4 alg_ids total, so we should remove the +1 here when we update.
        assert alg_id in range(CUSPARSELT_NUM_ALG_IDS + 1)

    @training_dtypes
    @parametrize("bias", [False, True])
    @parametrize("aligned", [True, False])
    @parametrize("amp", [True, False])
    def test_linearw24(self, dtype, bias: bool, aligned: bool, amp: bool) -> None:

        class LinearW24(torch.nn.Linear):
            def forward(self, input: torch.Tensor) -> torch.Tensor:
                sparse_weight = to_sparse_semi_structured(self.weight, training=True, backend="cusparselt")
                return F.linear(input, sparse_weight, self.bias)

        B, ft_in, ft_out = 64, 128, 256
        if not aligned:
            B = 65
        model_dtype = torch.float32 if amp else dtype
        x = torch.randn([B, ft_in], device="cuda", dtype=model_dtype, requires_grad=True)
        grad = torch.randn([B, ft_out], device="cuda", dtype=model_dtype)
        m = torch.nn.Linear(ft_in, ft_out, bias=bias).cuda().to(model_dtype)

        m24 = LinearW24(ft_in, ft_out, bias=bias).cuda().to(model_dtype)

        with torch.autocast("cuda", dtype=dtype, enabled=amp):
            # Make weights sparse
            state_dict = m.state_dict()
            weight_sp24 = SparseSemiStructuredTensorCUTLASS.from_dense_fast(state_dict["weight"].abs())
            state_dict["weight"] = weight_sp24.to_dense().to(model_dtype).detach()
            m.load_state_dict(state_dict)
            m24.load_state_dict(state_dict)

            # FW with dense weights
            out = m(x)

            # FW with sparsity
            x24 = x.detach().requires_grad_()
            out24 = m24(x24)

        # Backward passes outside autocast
        out.backward(grad)
        out24.backward(grad)

        assert out24.is_contiguous()
        assert x24.grad is not None
        assert x24.grad.is_contiguous()
        assert m24.weight.grad is not None
        assert m24.weight.grad.is_contiguous()
        if bias:
            assert m24.bias.grad is not None

        assert torch.allclose(out24, out, **atol_rtol_kw[dtype])
        assert x.grad is not None and x24.grad is not None
        assert torch.allclose(x24.grad, x.grad, **atol_rtol_kw[dtype])
        assert m.weight.grad is not None
        assert torch.allclose(
            m24.weight.grad.to(dtype),
            weight_sp24.from_dense_like(
                m.weight.grad.to(dtype), out_dense=True
            ),
            **atol_rtol_kw[dtype],
        )
        if bias:
            assert m.bias.grad is not None
            assert m24.bias.grad is not None
            assert torch.allclose(
                m24.bias.grad.to(dtype),
                m.bias.grad.to(dtype),
                **atol_rtol_kw[dtype],
            )


instantiate_device_type_tests(TestSparseSemiStructured, globals(), only_for="cuda")
instantiate_device_type_tests(TestSparseSemiStructuredCUTLASS, globals(), only_for="cuda")
instantiate_device_type_tests(TestSparseSemiStructuredCUSPARSELT, globals(), only_for="cuda")
instantiate_device_type_tests(TestSparseSemiStructuredTraining, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()



# class _TransformerFFN(nn.Module):
#     def __init__(
#         self,
#         in_features: int,
#         hidden_features=None,
#         out_features=None,
#         act_layer=nn.GELU,
#         bias: bool = True,
#         linear_cls=nn.Linear,
#     ) -> None:
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = linear_cls(in_features, hidden_features, bias=bias)
#         self.act = act_layer()
#         self.fc2 = linear_cls(hidden_features, out_features, bias=bias)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.fc2(x)
#         return x


# @cuda_sm80_only
# @torch_compile_tests
# @pytest.mark.skipif(not sp24._has_cusparseLt(), reason="requires cusparselt")
# def test_linearw24_block_compile() -> None:
#     # TODO: Parametrize on `dtype` when torch.compile gets faster
#     # currently takes ~5s per test
#     dtype = torch.bfloat16
#     B, FT_IN, FT_HIDDEN = 31, 512, 2048

#     _workaround_cusparselt_internal_error()
#     m = _TransformerFFN(FT_IN, FT_HIDDEN, linear_cls=LinearW24).to("cuda").to(dtype)
#     m_c = _TransformerFFN(FT_IN, FT_HIDDEN, linear_cls=LinearW24).to("cuda").to(dtype)
#     m_c.load_state_dict(m.state_dict())
#     m_c = cast(_TransformerFFN, torch.compile(m_c))

#     x, grad = [torch.randn([B, FT_IN], dtype=dtype, device="cuda") for _ in range(2)]
#     x = x.requires_grad_()
#     out = m(x)
#     out.backward(grad)

#     x_c = x.detach().requires_grad_()
#     out_c = m_c(x_c)
#     out_c.backward(grad)

#     assert_allclose(out_c, out, "output", **atol_rtol_kw[dtype])
#     assert x_c.grad is not None and x.grad is not None
#     assert_allclose(x_c.grad, x.grad, "output", **atol_rtol_kw[dtype])
#     for param_name, param_ref, param_c in [
#         ["fc1.weight", m.fc1.weight, m_c.fc1.weight],
#         ["fc1.bias", m.fc1.bias, m_c.fc1.bias],
#         ["fc2.weight", m.fc2.weight, m_c.fc2.weight],
#         ["fc2.bias", m.fc2.bias, m_c.fc2.bias],
#     ]:
#         assert param_ref.grad is not None and param_c.grad is not None, param_name
#         assert_allclose(param_c.grad, param_ref.grad, param_name, **atol_rtol_kw[dtype])
