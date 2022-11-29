# Owner(s): ["module: onnx"]
import copy

import functorch

import onnx_test_common
import torch
from torch.onnx._globals import GLOBALS
from torch.testing._internal import common_utils
from torch.testing._internal.common_methods_invocations import op_db

SKIPPED_OPS = {
    "linalg.householder_product",
    "cholesky_inverse",
    "linalg.matrix_rank",
    "signal.windows.cosine",
    "linalg.pinv",
    "linalg.solve_ex",
    "cholesky_solve",
    "jiterator_4inputs_with_extra_args",
    "pca_lowrank",
    "lu_solve",
    "geqrf",
    "linalg.svdvals",
    "linalg.solve_triangular",
    "pinverse",
    "linalg.cholesky",
    "linalg.ldl_factor_ex",
    "linalg.matrix_norm",
    "lu_unpack",
    "linalg.cholesky_ex",
    "linalg.eigvals",
    "cholesky",
    "linalg.ldl_solve",
    "linalg.cond",
    "signal.windows.kaiser",
    "linalg.lstsq",
    "linalg.eig",
    "triangular_solve",
    "logdet",
    "linalg.lu_solve",
    "linalg.tensorinv",
    "signal.windows.exponential",
    "qr",
    "linalg.lu",
    "symeig",
    "linalg.inv",
    "linalg.qr",
    "jiterator_2inputs_2outputs",
    "linalg.solve",
    "svd_lowrank",
    "svd",
    "lu",
    "linalg.eigh",
    "jiterator_binary_return_by_ref",
    "norm",
    "linalg.lu_factor_ex",
    "linalg.eigvalsh",
    "linalg.slogdet",
    "linalg.tensorsolve",
    "linalg.lu_factor",
    "jiterator_unary",
    "linalg.inv_ex",
    "linalg.svd",
    "linalg.norm",
    "linalg.matrix_power",
    "jiterator_binary",
    "linalg.det",
    "linalg.ldl_factor",
    "ormqr",
}


class TestFxToOnnxWithOnnxRuntimeOnOperators(onnx_test_common._TestONNXRuntime):
    def setUp(self):
        super().setUp()
        self._onnx_shape_inference = GLOBALS.onnx_shape_inference
        GLOBALS.onnx_shape_inference = False
        self.op_db = [op for op in op_db if op.name not in SKIPPED_OPS]

    def tearDown(self):
        GLOBALS.onnx_shape_inference = self._onnx_shape_inference

    def test_cpu_float_op_without_kwargs(self):
        selected_op_names = {
            "nn.functional.alpha_dropout",
            "neg",
            "flatten",
            "dot",
            "diagonal",
            "nn.functional.elu",
            "view_as",
            "lt",
            "sigmoid",
            "sum",
            "ceil",
            "where",
            "nn.functional.hardshrink",
            "ne",
            "broadcast_tensors",
            "erf",
            "le",
            "logical_and",
            "exp",
            "acos",
            "minimum",
            "ldexp",
            "mean",
            "T",
            "nn.functional.relu6",
            "nn.functional.hardswish",
            "nn.functional.hardtanh",
            "resolve_neg",
            "clone",
            "mH",
            "nn.functional.hardsigmoid",
            "fmod",
            "nn.functional.tanhshrink",
            "rsqrt",
            "mm",
            "conj_physical",
            "clamp_max",
            "argmax",
            "__rmul__",
            "special.ndtr",
            "fft.ifftshift",
            "nn.functional.mse_loss",
            "mv",
            "real",
            "H",
            "__rdiv__",
            "fft.fftshift",
            "nn.functional.silu",
            "div",
            "isnan",
            "asin",
            "mul",
            "log1p",
            "remainder",
            "square",
            "tan",
            "logical_xor",
            "nn.functional.soft_margin_loss",
            "__rmod__",
            "__radd__",
            "mT",
            "__rsub__",
            "floor",
            "sub",
            "tanh",
            "clamp_min",
            "logical_or",
            "positive",
            "gt",
            "expand_as",
            "abs",
            "round",
            "msort",
            "reciprocal",
            "atan",
            "log2",
            "nn.functional.pairwise_distance",
            "sqrt",
            "reshape_as",
            "nn.functional.celu",
            "nn.functional.l1_loss",
            "nn.functional.relu",
            "eq",
            "fliplr",
            "true_divide",
            "sign",
            "rsub",
            "log",
            "nn.functional.feature_alpha_dropout",
            "cos",
            "contiguous",
            "nn.functional.leaky_relu",
            "nn.functional.softshrink",
            "isinf",
            "bmm",
            "flipud",
            "argmin",
            "sin",
            "add",
            "float",
            "kron",
            "ge",
            "pow",
            "resolve_conj",
            "maximum",
            "nn.functional.embedding",
            "conj",
            "nn.functional.softsign",
            "nn.functional.mish",
            "outer",
        }
        target_dtype = torch.float
        target_device = "cpu"

        failed_op_names = set()
        for op in self.op_db:
            if op.name not in selected_op_names:
                continue

            samples = op.sample_inputs(target_device, target_dtype)
            for sample_input in samples:
                # Use the correct schema from get_signature_for_torch_op
                # Then schema.bind(sample_input.input, *sample_input.args, **sample_input.kwargs).
                args = [sample_input.input] + list(sample_input.args)
                kwargs = sample_input.kwargs

                if len(kwargs) > 0:
                    continue

                try:
                    gm = functorch.make_fx(op)(*copy.deepcopy(args))
                    self.run_test_with_fx_to_onnx_exporter(
                        gm, args, rtol=1e-3, atol=1e-7
                    )
                except Exception as e:
                    failed_op_names.add(op.name)

        self.assertEqual(len(failed_op_names), 0, f"Failed op names: {failed_op_names}")

    def test_cpu_float_op_with_kwargs(self):
        selected_op_names = {
            "diagonal",
            "nn.functional.l1_loss",
            "addmm",
            "nn.functional.hardshrink",
            "nn.functional.softshrink",
            "fft.ifftshift",
            "add",
            "nn.functional.mse_loss",
            "nn.functional.soft_margin_loss",
            "fft.fftshift",
            "sub",
            "nn.functional.hardtanh",
            "flatten",
            "rsub",
        }
        target_dtype = torch.float
        target_device = "cpu"

        failed_op_names = set()
        for op in self.op_db:
            if op.name not in selected_op_names:
                continue

            samples = op.sample_inputs(target_device, target_dtype)
            for sample_input in samples:
                # Use the correct schema from get_signature_for_torch_op
                # Then schema.bind(sample_input.input, *sample_input.args, **sample_input.kwargs).
                args = [sample_input.input] + list(sample_input.args)
                kwargs = sample_input.kwargs

                if len(kwargs) == 0:
                    continue

                def wrapped(*args_):
                    return op(*args_, **kwargs)

                try:
                    gm = functorch.make_fx(wrapped)(*copy.deepcopy(args))
                    self.run_test_with_fx_to_onnx_exporter(
                        gm, args, rtol=1e-3, atol=1e-7
                    )
                except Exception as e:
                    failed_op_names.add(op.name)

        self.assertEqual(len(failed_op_names), 0, f"Failed op names: {failed_op_names}")


if __name__ == "__main__":
    common_utils.run_tests()
