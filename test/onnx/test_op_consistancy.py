# Owner(s): ["module: onnx"]

"""Test consistency between torch.onnx exported operators and aten operators."""
import itertools

import torch
from torch.onnx import _constants, verification
from torch.testing._internal import (
    common_device_type,
    common_methods_invocations,
    common_utils,
)

# The min onnx opset version to test for
MIN_ONNX_OPSET_VERSION = 9
# The max onnx opset version to test for
MAX_ONNX_OPSET_VERSION = _constants.ONNX_MAX_OPSET

TESTED_OPSETS = range(MIN_ONNX_OPSET_VERSION, MAX_ONNX_OPSET_VERSION + 1)

# TODO(justinchuby): We need a way to annotate symbolic functions with their supported dtypes
# TODO(justinchuby): Add a way to include expected failures for specific opsets

# Ops and the dtypes to be tested for consistency
# NOTES: Incrementally add / uncomment ops and add types to this list as they are supported

ALLOWLIST_OP = {
    # "__getitem__": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "__radd__": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "__rand__": frozenset(["b8", "i16", "i32", "i64", "u8"]),
    # "__rdiv__": frozenset(["f16", "f32", "i16", "i32", "u8"]),
    # "__rmatmul__": frozenset(["f32"]),
    # "__rmul__": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "__ror__": frozenset(["b8", "i16", "i32", "i64", "u8"]),
    # "__rpow__": frozenset(["f16"]),
    # "__rxor__": frozenset(["b8", "i16", "i32", "i64", "u8"]),
    # "masked.argmax": frozenset(["i16", "i64", "u8"]),
    # "masked.argmin": frozenset(["i16", "i64", "u8"]),
    # "masked.log_softmax": frozenset(["f32"]),
    # "masked.logaddexp": frozenset(["f32"]),
    # "masked.norm": frozenset(["f16", "f32"]),
    # "masked.normalize": frozenset(["f16", "f32"]),
    # "masked.softmax": frozenset(["f32"]),
    # "masked.softmin": frozenset(["f32"]),
    # "masked.std": frozenset(["f32"]),
    # "masked.var": frozenset(["f32"]),
    # "abs": frozenset(["f16", "f32", "i16", "i32", "u8"]),
    # "acos": frozenset(["f32", "i16", "i32", "u8"]),
    # "acosh": frozenset(["f32", "i16", "i32", "u8"]),
    # "add": frozenset(["b8", "f16", "f32", "i16", "i32", "i64"]),
    # "addbmm": frozenset(["f32"]),
    # "addcdiv": frozenset(["f32"]),
    # "addcmul": frozenset(["f32", "i16", "i32", "i64", "u8"]),
    # "addmm": frozenset(["f32"]),
    # "addmv": frozenset(["f32"]),
    # "addr": frozenset(["b8", "f32", "i16", "i32", "i64", "u8"]),
    # "all": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "allclose": frozenset(["f16", "f32"]),
    # "any": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "arange": frozenset(["f16", "f32", "i16", "i32", "i64", "u8"]),
    # "argmax": frozenset(["f16", "f32", "i16", "i32", "i64", "u8"]),
    # "argmin": frozenset(["f16", "f32", "i16", "i32", "i64", "u8"]),
    # "amax": frozenset(["f32"]),
    # "amix": frozenset(["f32"]),
    # "logsumexp": frozenset(["f32"]),
    # "mean": frozenset(["f32"]),
    # "sum": frozenset(["f32"]),
    # "asin": frozenset(["f32", "i16", "i32", "u8"]),
    # "asinh": frozenset(["f32", "i16", "i32", "u8"]),
    # "atan": frozenset(["f32", "i16", "i32", "u8"]),
    # "atan2": frozenset(["f32"]),
    # "atanh": frozenset(["f32", "i16", "i32", "u8"]),
    # "atleast_1d": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "atleast_2d": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "atleast_3d": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "baddbmm": frozenset(["f32"]),
    # "bitwise_and": frozenset(["b8", "i16", "i32", "i64", "u8"]),
    # "bitwise_left_shift": frozenset(["i16", "i32", "i64", "u8"]),
    # "bitwise_not": frozenset(["b8", "i16", "i32", "i64", "u8"]),
    # "bitwise_or": frozenset(["b8", "i16", "i32", "i64", "u8"]),
    # "bitwise_right_shift": frozenset(["i16", "i32", "i64", "u8"]),
    # "bitwise_xor": frozenset(["b8", "i16", "i32", "i64", "u8"]),
    # "block_diag": frozenset(["b8", "f16", "f32", "i16", "i32", "i64"]),
    # "bmm": frozenset(["f32"]),
    # "broadcast_shapes": frozenset(["f32"]),
    # "cat": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    "ceil": frozenset(
        ["bf16", "f16", "f32"]
    ),  # Ceil not implemented for f64 in onnxruntime
    # "char": frozenset(["b8", "u8"]),
    # "chunk": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "clone": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "column_stack": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "combinations": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "conj": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "conj_physical": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "contiguous": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "corrcoef": frozenset(["f32"]),
    # "cos": frozenset(["f32", "i16", "i32", "u8"]),
    # "cosh": frozenset(["f32", "i16", "i32", "u8"]),
    # "cov": frozenset(["f32"]),
    # "deg2rad": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "diag": frozenset(["f32", "i32"]),
    # "diag_embed": frozenset(["b8", "f16", "f32", "i16", "i32", "i64"]),
    # "diagflat": frozenset(["f32", "i32"]),
    # "diagonal_scatter": frozenset(["b8", "f16", "f32", "i16", "i32", "i64"]),
    # "diff": frozenset(["f16", "f32", "i16", "i32", "i64"]),
    # "dist": frozenset(["f32"]),
    # "dot": frozenset(["f32", "i16", "i32", "i64", "u8"]),
    # "einsum": frozenset(["f32"]),
    # "equal": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "erf": frozenset(["f32", "i16", "i32", "u8"]),
    # "exp": frozenset(["f32", "i16", "i32", "u8"]),
    # "exp2": frozenset(["f16", "f32", "i16", "i32", "u8"]),
    # "eye": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "fill": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "flatten": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "flip": frozenset(["f16", "f32", "i16", "i32", "i64", "u8"]),
    # "fliplr": frozenset(["f16", "f32", "i16", "i32", "i64", "u8"]),
    # "flipud": frozenset(["f16", "f32", "i16", "i32", "i64", "u8"]),
    # "float": frozenset(["f32"]),
    # "floor": frozenset(["f32"]),
    # "gradient": frozenset(["f16", "f32", "i16"]),
    # "half": frozenset(["f16"]),
    # "hstack": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "index_select": frozenset(["f32", "i16", "i32", "i64"]),
    # "int": frozenset(["i32"]),
    # "isclose": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "isfinite": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "isinf": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "isnan": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "isreal": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "kron": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "linalg.matrix_norm": frozenset(["f16"]),
    # "linalg.svd": frozenset(["f32"]),
    # "linalg.vector_norm": frozenset(["f16", "f32"]),
    # "linspace": frozenset(["f16", "f32", "i16", "i32", "i64", "u8"]),
    # "log": frozenset(["f32", "i16", "i32", "u8"]),
    # "log10": frozenset(["f32", "i16", "i32", "u8"]),
    # "log2": frozenset(["f32", "i16", "i32", "u8"]),
    # "log_softmax": frozenset(["f32"]),
    # "logaddexp": frozenset(["f32"]),
    # "logaddexp2": frozenset(["f32"]),
    # "logical_not": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "logspace": frozenset(["f32", "i16", "i32", "i64", "u8"]),
    # "masked_fill": frozenset(["f16", "i16", "i32", "i64"]),
    # "masked_select": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "matmul": frozenset(["f32"]),
    # "mm": frozenset(["f32"]),
    # "mv": frozenset(["f32"]),
    # "neg": frozenset(["f16", "f32", "i16", "i32"]),
    # "nn.functional.adaptive_max_pool1d": frozenset(["f32"]),
    # "nn.functional.adaptive_max_pool2d": frozenset(["f32"]),
    # "nn.functional.binary_cross_entropy": frozenset(["f32"]),
    # "nn.functional.binary_cross_entropy_with_logits": frozenset(["f32"]),
    # "nn.functional.celu": frozenset(["f32"]),
    # "nn.functional.conv1d": frozenset(["f32"]),
    # "nn.functional.conv2d": frozenset(["f32"]),
    # "nn.functional.conv_transpose1d": frozenset(["f32"]),
    # "nn.functional.cosine_embedding_loss": frozenset(
    #     ["b8", "f32", "i16", "i32", "i64"]
    # ),
    # "nn.functional.elu": frozenset(["f32"]),
    # "nn.functional.feature_alpha_dropout": frozenset(
    #     [
    #         "b8",
    #         "f16",
    #         "f32",
    #         "i16",
    #         "i32",
    #         "i64",
    #         "u8",
    #     ]
    # ),
    # "nn.functional.gaussian_nll_loss": frozenset(["f32"]),
    # "nn.functional.glu": frozenset(["f32"]),
    # "nn.functional.group_norm": frozenset(["f32"]),
    # "nn.functional.hardtanh": frozenset(["f32", "i16", "i32", "i64"]),
    # "nn.functional.hinge_embedding_loss": frozenset(["f32"]),
    # "nn.functional.huber_loss": frozenset(["f32"]),
    # "nn.functional.instance_norm": frozenset(["f32"]),
    # "nn.functional.kl_div": frozenset(["f32"]),
    # "nn.functional.l1_loss": frozenset(["f16", "f32"]),
    # "nn.functional.leaky_relu": frozenset(["f32"]),
    # "nn.functional.linear": frozenset(["f32"]),
    # "nn.functional.local_response_norm": frozenset(["f32"]),
    # "nn.functional.margin_ranking_loss": frozenset(["f32", "i16", "i32"]),
    # "nn.functional.mse_loss": frozenset(["f16", "f32"]),
    # "nn.functional.pad": frozenset(["b8", "f16", "f32", "i16", "i32", "i64"]),
    # "nn.functional.pairwise_distance": frozenset(["f16", "f32", "i16", "i32", "i64"]),
    # "nn.functional.poisson_nll_loss": frozenset(["f32", "i16", "i32", "u8"]),
    # "nn.functional.prelu": frozenset(["f32"]),
    # "nn.functional.relu": frozenset(["f32", "i16", "i32", "i64", "u8"]),
    # "nn.functional.relu6": frozenset(["f32", "i16", "i32", "i64", "u8"]),
    # "nn.functional.selu": frozenset(["f32"]),
    # "nn.functional.silu": frozenset(["f32"]),
    # "nn.functional.smooth_l1_loss": frozenset(["f16", "f32"]),
    # "nn.functional.soft_margin_loss": frozenset(["f32"]),
    # "nn.functional.softmin": frozenset(["f32"]),
    # "nn.functional.softsign": frozenset(["f16", "f32", "i16", "u8"]),
    # "nn.functional.tanhshrink": frozenset(["f32", "i16", "i32", "u8"]),
    # "nn.functional.threshold": frozenset(["f32", "i16", "i32", "i64", "u8"]),
    # "nn.functional.triplet_margin_loss": frozenset(["f32", "i16", "i32", "i64"]),
    # "nn.functional.triplet_margin_with_distance_loss": frozenset(
    #     ["f32", "i16", "i32", "i64"]
    # ),
    # "nn.functional.upsample_bilinear": frozenset(["f32"]),
    # "norm": frozenset(["f32", "f16"]),
    # "positive": frozenset(["f16", "f32", "i16", "i32", "i64", "u8"]),
    # "pow": frozenset(["f16"]),
    # "rad2deg": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "real": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "reciprocal": frozenset(["f16", "f32", "i16", "i32", "u8"]),
    # "repeat": frozenset(["f16", "f32", "i16", "i32", "i64", "u8"]),
    # "repeat_interleave": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "resize_": frozenset(["b8", "i16", "i32", "i64", "u8"]),
    # "resize_as_": frozenset(["b8", "i16", "i32", "i64", "u8"]),
    # "resolve_conj": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "resolve_neg": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "rot90": frozenset(["f16", "f32", "i16", "i32", "i64", "u8"]),
    # "round": frozenset(["f32"]),
    # "rsqrt": frozenset(["f32", "i16", "i32", "u8"]),
    # "select_scatter": frozenset(["b8", "f16", "f32", "i16", "i32", "i64"]),
    # "sgn": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "short": frozenset(["i16"]),
    # "sigmoid": frozenset(["f32"]),
    # "sign": frozenset(["b8", "f16", "f32", "i16", "i32", "u8"]),
    # "sin": frozenset(["f32", "i16", "i32", "u8"]),
    # "sinh": frozenset(["f32", "i16", "i32", "u8"]),
    # "slice_scatter": frozenset(["b8", "f16", "f32", "i16", "i32", "i64"]),
    # "softmax": frozenset(["f32"]),
    # "special.ndtr": frozenset(["b8", "f32", "i16", "i32", "i64", "u8"]),
    # "split": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    "sqrt": frozenset(["f32", "f64"]),
    # "square": frozenset(["f16", "f32"]),
    # "squeeze": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "stack": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "sub": frozenset(["f32", "i16", "i32", "i64"]),
    # "sum_to_size": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "svd": frozenset(["f32"]),
    "t": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "tan": frozenset(["i16", "i32", "u8"]),
    # "tanh": frozenset(["f32", "i16", "i32", "u8"]),
    # "tensordot": frozenset(["f32"]),
    # "tile": frozenset(["f16", "f32", "i16", "i32", "i64", "u8"]),
    # "topk": frozenset(["f32"]),
    # "trapz": frozenset(["f16", "f32", "i16", "i32", "i64"]),
    # "tril": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "tril_indices": frozenset(["i32", "i64"]),
    # "triu": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "triu_indices": frozenset(["i32", "i64"]),
    # "true_divide": frozenset(["b8", "f16", "f32", "i16", "u8"]),
    # "trunc": frozenset(["f32"]),
    # "unbind": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "unflatten": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "unsqueeze": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "view": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "view_as": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "vsplit": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "vstack": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "zero_": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "clamp": frozenset(["f32", "i16", "i32", "i64", "u8"]),
    # "clamp_max": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "clamp_min": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "logical_and": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "logical_or": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "logical_xor": frozenset(["b8", "f16", "f32", "i16", "i32", "i64", "u8"]),
    # "where": frozenset(["f16", "f32", "i16", "i32", "i64", "u8"]),
}


SUPPORTED_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    # Floating types
    torch.float32,
    torch.float64,
    torch.bfloat16,
    # QInt types
    torch.qint8,
    torch.quint8,
    # Complex types
    torch.complex32,
    torch.complex64,
    torch.complex128,
)


_ORT_PROVIDERS = ("CPUExecutionProvider",)


class SingleOpModel(torch.nn.Module):
    """Test model to wrap around a single op for export."""

    def __init__(self, op, kwargs):
        super().__init__()
        self.operator = op
        self.kwargs = kwargs

    def forward(self, *args):
        return self.operator(*args, **self.kwargs)


class TestConsistency(common_utils.TestCase):
    @common_device_type.ops(
        common_methods_invocations.op_db, allowed_dtypes=SUPPORTED_DTYPES
    )
    def test_output_match(self, device: str, dtype: torch.dtype, op):
        assert device == "cpu"

        if op.name not in ALLOWLIST_OP:
            self.skipTest(f"'{op.name}' is not in the allow list for test on ONNX")
        else:
            if common_utils.dtype_abbrs[dtype] not in ALLOWLIST_OP[op.name]:
                self.skipTest(
                    f"'{op.name}' is in the allow list for ONNX but dtype '{dtype}' is excluded"
                )

        samples = op.sample_inputs(
            device,
            dtype,
            requires_grad=False,
        )

        for (cpu_sample, opset) in itertools.product(samples, TESTED_OPSETS):
            model = SingleOpModel(op, cpu_sample.kwargs)

            with self.subTest(sample=cpu_sample, opset=opset):
                verification.verify(
                    model,
                    (cpu_sample.input, *cpu_sample.args),
                    input_kwargs={},
                    opset_version=opset,
                    keep_initializers_as_inputs=True,
                    ort_providers=_ORT_PROVIDERS,
                    check_shape=True,
                    check_dtype=True,
                    flatten=False,
                )


common_device_type.instantiate_device_type_tests(
    TestConsistency, globals(), only_for="cpu"
)


if __name__ == "__main__":
    common_utils.run_tests()
