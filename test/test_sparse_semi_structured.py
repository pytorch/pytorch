# Owner(s): ["module: sparse"]
import itertools
import random
import unittest
import sys

import torch
from torch import nn

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

from torch.utils._triton import has_triton

CUSPARSELT_NUM_ALG_IDS = 4
CUSPARSELT_MIXED_DTYPE_SUPPORT = [torch.float16, torch.bfloat16, torch.int32]

SEMI_STRUCTURED_SUPPORTED_DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.int8]
SEMI_STRUCTURED_SUPPORTED_BACKENDS = []

_IS_SM8X = False
if torch.cuda.is_available():
    _IS_SM8X = torch.cuda.get_device_capability(0)[0] == 8
    SEMI_STRUCTURED_SUPPORTED_BACKENDS.append("cutlass")

    # check if cslt is available for now using this:
    # TODO when we add cusparselt as a backend, we can update this to be use torch.cusparselt.is_available()
    try:
        torch._cslt_compress(torch.ones(128, 256).cuda())
        SEMI_STRUCTURED_SUPPORTED_BACKENDS.append("cusparselt")
    except Exception:
        pass



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

        SparseSemiStructuredTensor._FORCE_CUTLASS = backend == "cutlass"

        input = torch.rand(dense_input_shape, device="cuda").half()
        model = Model().eval().cuda().half()
        mod_linear = model.linear
        m, n = mod_linear.weight.shape
        mask = torch.Tensor([1, 0, 0, 1]).tile((m, n // 4)).bool().cuda()
        # set masked weight
        mod_linear.weight = nn.Parameter(mod_linear.weight * mask)

        dense_result = model(input)
        mod_linear.weight = nn.Parameter(to_sparse_semi_structured(mod_linear.weight))
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
    @unittest.skipIf(sys.version_info >= (3, 12), "torch.compile is not supported on python 3.12+")
    @unittest.skipIf("cusparselt" not in SEMI_STRUCTURED_SUPPORTED_BACKENDS, "cusparselt not supported on this machine")
    def test_mlp_contiguous_relu_compile_cusparselt(self):
        """
        test for cuSPASRELt meta registrations (_cslt_sparse_mm) + torch.compile
        """
        for dense_input_shape in [(1, 128), (64, 128), (128, 128), (64, 128, 128)]:
            SparseSemiStructuredTensorCompileTest._test_mlp_contiguous_relu_compile("cusparselt", dense_input_shape)

    @unittest.skipIf(IS_WINDOWS, "torch.compile not supported on windows")
    @unittest.skipIf(sys.version_info >= (3, 12), "torch.compile is not supported on python 3.12+")
    def test_mlp_contiguous_relu_compile_cutlass(self):
        """
        test for CUTLASS meta registrations (_sparse_semi_structured_linear) + torch.compile
        """
        for dense_input_shape in [(1, 128), (64, 128), (128, 128), (64, 128, 128)]:
            SparseSemiStructuredTensorCompileTest._test_mlp_contiguous_relu_compile("cutlass", dense_input_shape)


class TestSparseSemiStructured(TestCase):

    def setUp(self):
        if not _IS_SM8X:
            self.skipTest('Only runs on SM80')

    @dtypes(*SEMI_STRUCTURED_SUPPORTED_DTYPES)
    @parametrize("backend", SEMI_STRUCTURED_SUPPORTED_BACKENDS)
    def test_to_sparse_semi_structured(self, dtype, backend):
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")

        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")

        A = rand_sparse_semi_structured_mask(128, 256, dtype=dtype)
        A_sparse = to_sparse_semi_structured(A)

        assert A.shape == A_sparse.shape
        assert A.device == A_sparse.device
        assert A.dtype == A_sparse.dtype

        assert isinstance(A, torch.Tensor)
        assert isinstance(A_sparse, SparseSemiStructuredTensor)


    @dtypes(*SEMI_STRUCTURED_SUPPORTED_DTYPES)
    @parametrize("dense_input_shape", [(128, 1), (128, 64), (128, 128)])
    @parametrize("backend", SEMI_STRUCTURED_SUPPORTED_BACKENDS)
    def test_mm_sparse_first_NN(self, dense_input_shape, dtype, device, backend):
        """
        Ensure torch.mm(A_sparse, B) is correct for float16 and will throw error for int8
        """
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")

        A = rand_sparse_semi_structured_mask(256, 128, dtype=dtype)
        A_sparse = to_sparse_semi_structured(A)

        B = torch.rand(dense_input_shape, device=A_sparse.device).to(dtype)

        # Currently we don't support int matmul on GPU, so evaluate on CPU and copy over
        if dtype is torch.int8:
            # This should fail
            if backend == "cutlass":
                with self.assertRaisesRegex(RuntimeError, "two_four_sgemm_dispatch_layouts"):
                    sparse_result = torch.mm(A_sparse, B)
            else:
                with self.assertRaisesRegex(RuntimeError,
                                            "CUDA error: operation not supported when calling `cusparseLtMatmulDescriptorInit"):
                    sparse_result = torch.mm(A_sparse, B)
        else:
            dense_result = torch.mm(A, B)
            sparse_result = torch.mm(A_sparse, B)
            assert torch.allclose(dense_result, sparse_result, rtol=1e-3, atol=1e-3)

    @dtypes(*SEMI_STRUCTURED_SUPPORTED_DTYPES)
    @parametrize("dense_input_shape", [(1, 128), (64, 128), (128, 128)])
    @parametrize("backend", SEMI_STRUCTURED_SUPPORTED_BACKENDS)
    def test_mm_sparse_first_NT(self, dense_input_shape, dtype, device, backend):
        """
        Ensure torch.mm(A_sparse, B.t()) is correct for float16/bfloat16
        and will throw an error for int8 + padding
        """
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")

        A = rand_sparse_semi_structured_mask(256, 128, dtype=dtype)
        A_sparse = to_sparse_semi_structured(A)

        B = torch.rand(dense_input_shape, device=A_sparse.device).to(dtype)

        # Currently we don't support int matmul on GPU, so evaluate on CPU and copy over
        if dtype is torch.int8 and dense_input_shape in {(1, 128)}:
            # padding with int8 throws an error because transposing B yields a contiguous output
            # and row-row 2:4 sparse @ dense with NN is not supported by cuSPARSELt or CUTLASS.
            if backend == "cutlass":
                with self.assertRaisesRegex(RuntimeError, "two_four_sgemm_dispatch_layouts"):
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

    @dtypes(*SEMI_STRUCTURED_SUPPORTED_DTYPES)
    @parametrize("dense_input_shape", [(1, 128), (64, 128), (128, 128)])
    @parametrize("backend", SEMI_STRUCTURED_SUPPORTED_BACKENDS)
    def test_mm_sparse_first_TN(self, dtype, dense_input_shape, device, backend):
        """
        Ensure torch.mm(A_sparse.t(), B) throws error
        """
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")
        A = rand_sparse_semi_structured_mask(128, 256, dtype=dtype)
        A_sparse = to_sparse_semi_structured(A)

        B = torch.rand(dense_input_shape, device=A_sparse.device).to(dtype)

        with self.assertRaisesRegex(
            NotImplementedError,
            r"`SparseSemiStructuredTensor.*` matmul: operation is not supported",
        ):
            torch.mm(A_sparse.t(), B)

    @dtypes(*SEMI_STRUCTURED_SUPPORTED_DTYPES)
    @parametrize("dense_input_shape", [(1, 128), (64, 128), (128, 128)])
    @parametrize("backend", SEMI_STRUCTURED_SUPPORTED_BACKENDS)
    def test_mm_sparse_second_NT(self, dense_input_shape, dtype, device, backend):
        """
        Ensure torch.mm(A, B_sparse.t()) is correct
        """
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")
        B = rand_sparse_semi_structured_mask(256, 128, dtype=dtype)
        B_sparse = to_sparse_semi_structured(B)

        A = torch.rand(dense_input_shape, device=B_sparse.device).to(dtype)

        # Currently we don't support int matmul on GPU, so evaluate on CPU and copy over
        if dtype is torch.int8:
            dense_result = torch.mm(A.cpu(), B.t().cpu()).to(device, dtype=torch.int8)
            sparse_result = torch.mm(A, B_sparse.t())
        else:
            dense_result = torch.mm(A, B.t())
            sparse_result = torch.mm(A, B_sparse.t())

        assert torch.allclose(dense_result, sparse_result, rtol=1e-3, atol=1e-3)

    @dtypes(*SEMI_STRUCTURED_SUPPORTED_DTYPES)
    @parametrize("dense_input_shape", [(1, 128), (64, 128), (128, 128)])
    @parametrize("backend", SEMI_STRUCTURED_SUPPORTED_BACKENDS)
    def test_mm_sparse_second_NN(self, dense_input_shape, dtype, device, backend):
        """
        Ensure torch.mm(A, B_sparse) throws error
        """
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")
        B = rand_sparse_semi_structured_mask(256, 128, dtype=dtype)
        B_sparse = to_sparse_semi_structured(B)

        A = torch.rand(dense_input_shape, device=B_sparse.device).to(dtype)

        with self.assertRaisesRegex(
            NotImplementedError,
            r"`SparseSemiStructuredTensor.*` matmul: operation is not supported",
        ):
            sparse_result = torch.mm(A, B_sparse)

    @parametrize("dense_input_shape", [(1, 128), (64, 128), (128, 128), (64, 128, 128)])
    @parametrize("inference_mode", [subtest(True), subtest(False)])
    @parametrize("backend", SEMI_STRUCTURED_SUPPORTED_BACKENDS)
    def test_linear(self, dense_input_shape, inference_mode, device, backend):
        """
        Test nn.Linear has the same numerics
        """
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")
        input = torch.rand((dense_input_shape), device=device).half()
        model = nn.Linear(128, 256).to(device).half()
        m, n = model.weight.shape
        mask = rand_sparse_semi_structured_mask(m, n, device=device, dtype=torch.bool)
        # set masked weight
        model.weight = nn.Parameter(model.weight * mask)

        dense_result = model(input)

        model.weight = nn.Parameter(to_sparse_semi_structured(model.weight))

        if inference_mode:
            with torch.inference_mode():
                sparse_result = model(input)
        else:
            sparse_result = model(input)

        assert torch.allclose(dense_result, sparse_result, rtol=1e-3, atol=1e-3)

    @parametrize("dense_input_shape", [(1, 128), (64, 128), (128, 128), (64, 128, 128)])
    @parametrize("backend", SEMI_STRUCTURED_SUPPORTED_BACKENDS)
    def test_mlp(self, device, dense_input_shape, backend):
        SparseSemiStructuredTensor._FORCE_CUTLASS = backend == "cutlass"
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")
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
            model[i].weight = nn.Parameter(to_sparse_semi_structured(model[i].weight))

        sparse_result = model(input)

        assert torch.allclose(dense_result, sparse_result, rtol=1e-3, atol=1e-3)

    @parametrize("backend", SEMI_STRUCTURED_SUPPORTED_BACKENDS)
    def test_values(self, backend):
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")
        A = rand_sparse_semi_structured_mask(128, 128)
        A_sparse = to_sparse_semi_structured(A)
        assert A_sparse.values().shape == (128, 64)
        assert (A_sparse.values() == 1).all()

    @parametrize("backend", SEMI_STRUCTURED_SUPPORTED_BACKENDS)
    def test_indices(self, backend):
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")
        A = rand_sparse_semi_structured_mask(128, 128)
        A_sparse = to_sparse_semi_structured(A)
        assert A_sparse.indices().shape == (128, 8)

    @dtypes(*SEMI_STRUCTURED_SUPPORTED_DTYPES)
    @parametrize("backend", SEMI_STRUCTURED_SUPPORTED_BACKENDS)
    def test_min_sparse_shape(self, dtype, device, backend):
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")
        if backend == "cutlass":
            config = SparseSemiStructuredTensorCUTLASS._DTYPE_SHAPE_CONSTRAINTS[dtype]
        elif backend == "cusparselt":
            config = SparseSemiStructuredTensorCUSPARSELT._DTYPE_SHAPE_CONSTRAINTS[dtype]
        A = rand_sparse_semi_structured_mask(config.sparse_min_rows, config.sparse_min_cols, dtype=dtype, device=device)
        A_sparse = to_sparse_semi_structured(A)
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

    @dtypes(*SEMI_STRUCTURED_SUPPORTED_DTYPES)
    @parametrize("backend", SEMI_STRUCTURED_SUPPORTED_BACKENDS)
    def test_unsupported_shape(self, dtype, device, backend):
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")
        A = rand_sparse_semi_structured_mask(2, 2, dtype=dtype, device=device)
        with self.assertRaisesRegex(RuntimeError, "Error original_tensor.shape"):
            A_sparse = to_sparse_semi_structured(A)

    @dtypes(*all_types_and_complex())
    @parametrize("backend", SEMI_STRUCTURED_SUPPORTED_BACKENDS)
    def test_unsupported_dtype(self, dtype, device, backend):
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")
        A = rand_sparse_semi_structured_mask(128, 128, dtype=dtype, device=device)

        if dtype not in SEMI_STRUCTURED_SUPPORTED_DTYPES:
            with self.assertRaisesRegex(RuntimeError, "Error original_tensor.dtype"):
                A_sparse = to_sparse_semi_structured(A)
        else:
            A_sparse = to_sparse_semi_structured(A)

    @parametrize("backend", SEMI_STRUCTURED_SUPPORTED_BACKENDS)
    def test_unsupported_dim(self, device, backend):
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")
        A = torch.rand(128, 128, 128, device=device, dtype=torch.float16)

        with self.assertRaisesRegex(RuntimeError, "Error original_tensor.dim"):
            A_sparse = to_sparse_semi_structured(A)

    @unittest.skipIf(TEST_WITH_ROCM or IS_WINDOWS, "ROCm and Windows doesn't support CUTLASS")
    @parametrize("backend", ["cutlass"])
    @dtypes(*SEMI_STRUCTURED_SUPPORTED_DTYPES)
    def test_linear_cutlass(self, device, dtype, backend):
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")

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
    @parametrize("backend", ["cutlass"])
    @dtypes(*SEMI_STRUCTURED_SUPPORTED_DTYPES)
    def test_conversions(self, device, dtype, backend):
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")

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
    @parametrize("backend", ["cutlass"])
    @dtypes(*SEMI_STRUCTURED_SUPPORTED_DTYPES)
    def test_conversions_all_patterns(self, device, dtype, backend):
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")
        r, c = 32, 128

        dense_inv, dense_val = rand_sparse_semi_structured_all_patterns(r, c, dtype, device)

        compressed = to_sparse_semi_structured(dense_inv)
        dense = compressed.to_dense()

        torch.testing.assert_close(dense, dense_val, rtol=0, atol=0)

class TestCUSPARSELT(TestCase):
    """
    This contains cuSPARSELt specific tests.
    """

    def setUp(self):
        if not _IS_SM8X:
            self.skipTest('Only runs on SM80')
        if "cusparselt" not in SEMI_STRUCTURED_SUPPORTED_BACKENDS:
            self.skipTest('cuSPARSELt not enabled')
        else:
            SparseSemiStructuredTensor._FORCE_CUTLASS = False

    @parametrize("out_dtype", CUSPARSELT_MIXED_DTYPE_SUPPORT)
    @parametrize("dense_input_shape", [(128, 128)])
    def test_cslt_sparse_mm_mixed_dtype(self, dense_input_shape, out_dtype, device):
        A = rand_sparse_semi_structured_mask(128, 128, dtype=torch.int8)
        A_compressed = torch._cslt_compress(A)

        B = torch.rand(dense_input_shape, device=device).to(torch.int8)

        dense_result = torch.mm(A.cpu().to(torch.int64), B.t().cpu().to(torch.int64)).to(device, dtype=out_dtype)
        sparse_result = torch._cslt_sparse_mm(A_compressed, B.t(), out_dtype=out_dtype)
        assert torch.allclose(dense_result, sparse_result, rtol=1e-3, atol=1e-3)

    @dtypes(torch.float16, torch.bfloat16)
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
    @dtypes(*SEMI_STRUCTURED_SUPPORTED_DTYPES)
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

    @dtypes(*SEMI_STRUCTURED_SUPPORTED_DTYPES)
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


instantiate_device_type_tests(TestSparseSemiStructured, globals(), only_for="cuda")
instantiate_device_type_tests(TestCUSPARSELT, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
