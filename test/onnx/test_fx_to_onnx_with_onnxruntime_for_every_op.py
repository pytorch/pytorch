# Owner(s): ["module: onnx"]
import onnx_test_common
import torch
from torch.onnx._internal._fx import _OP_OVERLOAD_TO_EXPORTER_KEY_TABLE
from torch.testing._internal import common_utils
from torch.testing._internal.common_methods_invocations import op_db
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map

not_recorded_ops = {torch.ops.aten.detach.default}


class RecordExampleMode(TorchDispatchMode):
    def __init__(self, skipped_ops):
        self.skipped_ops = skipped_ops
        self.inputs = []
        self.kw_inputs = []
        self.outputs = []
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if (
            any(arg.kwarg_only for arg in func._schema.arguments)
            or func in self.skipped_ops
        ):
            # Only record examples for functions that don't have kwarg-only arguments.
            # Dynamo FX exporter doesn't support kwarg so does FX-to-ONNX exporter.
            return func(*args, **kwargs)
        detached_args = tree_map(
            lambda x: x.detach().to("cpu") if isinstance(x, torch.Tensor) else x, args
        )
        detached_kwargs = tree_map(
            lambda x: x.detach().to("cpu") if isinstance(x, torch.Tensor) else x, kwargs
        )
        self.inputs.append(detached_args)
        self.kw_inputs.append(detached_kwargs)
        self.ops.append(func)
        if kwargs is None:
            kwargs = {}
        out = func(*args, **kwargs)
        detached_outputs = tree_map(
            lambda x: x.detach().to("cpu") if isinstance(x, torch.Tensor) else x, out
        )
        self.outputs.append(detached_outputs)
        return out


missing_implementation_cases = {
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
    "signal.windows.gaussian",
}


allowed_test_dtypes = {torch.float}


class TestFxToOnnxWithOnnxRuntimeOnOperators(onnx_test_common._TestONNXRuntime):
    def test_op(self):
        for op in op_db:
            # Two kinds of ops are skipped.
            #  1. Their implementation is not always built with PyTorch.
            #  2. Non-aten ops.
            #
            # Reason of skipping case 1:
            #  For example, when linear algebra is disabled, torch.linalg.* are not runnable
            #  and the following
            #   op(*args, **kwargs)
            #  will throw an error.
            #  Therefore, we skip those missing operators. Please do NOT extend this list
            #  for other reasons; if a op is added, it only means PyTorch can't run it.
            # Reason of skipping case 2:
            #  We don't have FX-to-ONNX exporter for those ops.
            if op.aten_name is None or op.name in missing_implementation_cases:
                continue
            mode = RecordExampleMode(not_recorded_ops)
            with mode:
                selected_dtypes = [
                    dtype
                    for dtype in allowed_test_dtypes
                    if op.supports_dtype(dtype, "cpu")
                ]
                for dtype in selected_dtypes:
                    samples = op.sample_inputs("cpu", dtype)
                    for sample_input in samples:
                        args = [sample_input.input] + list(sample_input.args)
                        kwargs = sample_input.kwargs
                        op(*args, **kwargs)

            for inputs, kw_inputs, outputs, op in zip(
                mode.inputs, mode.kw_inputs, mode.outputs, mode.ops
            ):
                if op not in _OP_OVERLOAD_TO_EXPORTER_KEY_TABLE:
                    print(
                        f"[SkipTest] Missing exporter. {op}, {','.join([str(type(x)) for x in inputs])}, {str(kw_inputs)}"
                    )
                    continue
                if any(not isinstance(value, torch.Tensor) for value in inputs):
                    print(
                        f"[SkipTest] Non-tensor inputs. {op}, {','.join([str(type(x)) for x in inputs])}, {str(kw_inputs)}"
                    )
                    continue
                if any(arg.kwarg_only for arg in op._schema.arguments):
                    print(
                        "[SkipTest] Key-word argument generally not supported yet."
                        f" {op}, {','.join([str(type(x)) for x in inputs])}, {str(kw_inputs)}"
                    )
                    continue
                try:
                    self.run_test_with_positional_args(op, args, **kwargs)
                    print(
                        f"[PassTest] {op}, {','.join([str(type(x)) for x in inputs])}, {str(kw_inputs)}"
                    )
                except Exception as e:
                    print(
                        f"[FailTest] {op}, {','.join([str(type(x)) for x in inputs])}, {str(kw_inputs)}"
                    )


if __name__ == "__main__":
    common_utils.run_tests()
