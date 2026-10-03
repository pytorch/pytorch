# Owner(s): ["module: unknown"]

import platform
from functools import partial
from unittest import skipIf as skipif

import torch
from torch.autograd.gradcheck import GradcheckError
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    OpDTypes,
    ops,
    skip,
    skipOps,
    xfail,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import (
    IS_MACOS,
    run_tests,
    skipIfTorchInductor,
    TEST_WITH_ROCM,
    TEST_WITH_SLOW_GRADCHECK,
    TestCase,
    TestGradients,
    unMarkDynamoStrictTest,
)
from torch.testing._internal.opinfo.core import (
    sample_skips_and_xfails,
    SkipRule,
    XFailRule,
)


# TODO: mitigate flaky issue on macOS https://github.com/pytorch/pytorch/issues/66033
# AFAIK, c10::ThreadPool looks correct in the way it uses condition_variable wait. The
# issue seems to point to macOS itself https://github.com/graphia-app/graphia/issues/33
if IS_MACOS:
    torch.set_num_threads(1)

# gradcheck requires double precision
_gradcheck_ops = partial(
    ops, dtypes=OpDTypes.supported, allowed_dtypes=[torch.double, torch.cdouble]
)


# Device-agnostic skips migrated from OpInfo definitions (see #177259).
_fwd_grad_all = {
    skip("as_strided"),
    skip("as_strided_copy"),
    skip("round", variant_name="decimals_3"),
    skip("round", variant_name="decimals_neg_3"),
    skip("polygamma", variant_name="polygamma_n_1"),
    skip("polygamma", variant_name="polygamma_n_2"),
    skip("polygamma", variant_name="polygamma_n_3"),
    skip("polygamma", variant_name="polygamma_n_4"),
    xfail("bfloat16"),
    xfail("float"),
    xfail("half"),
    xfail("cfloat"),
    xfail("chalf"),
    skip("normal"),
    skip("normal", variant_name="number_mean"),
    skip("linalg.lstsq"),
}


@unMarkDynamoStrictTest
class TestFwdGradients(TestGradients):
    @_gradcheck_ops([op for op in op_db if op.supports_forward_ad])
    @sample_skips_and_xfails(
        [
            XFailRule(
                op_match_fn=lambda device, op: op.name == "cumprod",
                sample_match_fn=lambda device, sample: bool((sample.input == 0).any()),
                error_type=GradcheckError,
                error_msg="Forward-over-forward AD: .*mismatch",
                name="cumprod zero factors: #196701",
            ),
            XFailRule(
                op_match_fn=lambda device, op: op.name == "linalg.householder_product",
                sample_match_fn=lambda device, sample: sample.input.numel() > 0
                and sample.args[0].numel() > 0,
                error_type=GradcheckError,
                error_msg="Forward-over-forward AD: .*mismatch",
                name="householder_product nested JVP: #196698",
            ),
            XFailRule(
                op_match_fn=lambda device, op: op.name
                in ("native_batch_norm", "_native_batch_norm_legit"),
                sample_match_fn=lambda device, sample: sample.input.numel() > 0
                and sample.args[-3] is True,
                error_type=GradcheckError,
                error_msg="Forward-over-forward AD: .*mismatch",
                name="batch_norm training nested JVP: #196699",
            ),
            XFailRule(
                op_match_fn=lambda device, op: op.name
                in ("_batch_norm_with_update", "nn.functional.layer_norm"),
                sample_match_fn=lambda device, sample: sample.input.numel() > 0,
                error_type=GradcheckError,
                error_msg="Forward-over-forward AD: .*mismatch",
                name="normalization nested JVP: #196699, #196700",
            ),
            XFailRule(
                op_match_fn=lambda device, op: op.name == "nn.functional.batch_norm",
                sample_match_fn=lambda device, sample: sample.input.numel() > 0
                and sample.kwargs.get("training", False),
                error_type=GradcheckError,
                error_msg="Forward-over-forward AD: .*mismatch",
                name="batch_norm training nested JVP: #196699",
            ),
            XFailRule(
                op_match_fn=lambda device, op: op.name == "nn.functional.instance_norm",
                sample_match_fn=lambda device, sample: sample.input.numel() > 0
                and sample.kwargs.get("use_input_stats", True),
                error_type=GradcheckError,
                error_msg="Forward-over-forward AD: .*mismatch",
                name="instance_norm uses batch_norm: #196699",
            ),
            SkipRule(
                op_match_fn=lambda device, op: op.name == "linalg.norm"
                and op.variant_test_name == "subgradients_at_zero",
                sample_match_fn=lambda device, sample: sample.input.numel() > 0
                and sample.args[0] != 0
                and bool((sample.input == 0).all()),
                name="No classical second derivative at zero; numerical reference is inapplicable",
            ),
            SkipRule(
                op_match_fn=lambda device, op: op.name
                in (
                    "nn.functional.max_unpool1d",
                    "nn.functional.max_unpool2d",
                    "nn.functional.max_unpool3d",
                )
                and not op.variant_test_name,
                sample_match_fn=lambda device, sample: sample.args[0].numel() > 0
                and bool(
                    sample.args[0]
                    .flatten(2)
                    .sort(dim=-1)
                    .values.diff(dim=-1)
                    .eq(0)
                    .any()
                ),
                name="Duplicate writes within an unpool plane have no deterministic numerical reference",
            ),
            XFailRule(
                op_match_fn=lambda device, op: device == "cuda"
                and not TEST_WITH_ROCM
                and (
                    op.name in ("svd", "linalg.svd", "linalg.svdvals", "linalg.cond")
                    or op.name == "norm"
                    and op.variant_test_name == "nuc"
                ),
                sample_match_fn=lambda device, sample: sample.input.dtype
                == torch.cdouble
                and sample.input.numel() > 0,
                error_type=RuntimeError,
                error_msg="Expected the output of forward differentiable view operations to have the tangent have the same layout as primal",
                name="CUDA complex SVD tangent layout",
            ),
            XFailRule(
                op_match_fn=lambda device, op: device == "cuda"
                and not TEST_WITH_ROCM
                and op.name in ("svd_lowrank", "pca_lowrank"),
                sample_match_fn=lambda device, sample: sample.input.dtype
                == torch.cdouble
                and sample.input.numel() > 0
                and sample.kwargs.get("q", 6) > 0,
                error_type=RuntimeError,
                error_msg="Expected the output of forward differentiable view operations to have the tangent have the same layout as primal",
                name="CUDA complex low-rank SVD tangent layout",
            ),
            XFailRule(
                op_match_fn=lambda device, op: device == "cuda"
                and not TEST_WITH_ROCM
                and op.name == "linalg.matrix_norm",
                sample_match_fn=lambda device, sample: sample.input.dtype
                == torch.cdouble
                and sample.input.numel() > 0
                and (sample.args[0] if sample.args else sample.kwargs.get("ord"))
                in ("nuc", 2, -2),
                error_type=RuntimeError,
                error_msg="Expected the output of forward differentiable view operations to have the tangent have the same layout as primal",
                name="CUDA complex spectral matrix norm uses SVD",
            ),
            XFailRule(
                op_match_fn=lambda device, op: device == "cuda"
                and not TEST_WITH_ROCM
                and op.name == "linalg.norm",
                sample_match_fn=lambda device, sample: sample.input.dtype
                == torch.cdouble
                and sample.input.numel() > 0
                and sample.input.ndim == 2
                and (sample.args[0] if sample.args else sample.kwargs.get("ord"))
                in ("nuc", 2, -2),
                error_type=RuntimeError,
                error_msg="Expected the output of forward differentiable view operations to have the tangent have the same layout as primal",
                name="CUDA complex spectral matrix norm uses SVD",
            ),
            XFailRule(
                op_match_fn=lambda device, op: op.name == "linalg.lu_solve",
                sample_match_fn=lambda device, sample: (
                    sample.input.dtype == torch.cdouble
                    and not sample.kwargs["left"]
                    and sample.input.ndim > 2
                    and sample.input.numel() > 0
                    and sample.args[1].numel() > 0
                ),
                error_type=RuntimeError,
                error_msg="Expected the output of forward differentiable view operations to have the tangent have the same layout as primal",
                name="Batched complex right-sided lu_solve tangent layout",
            ),
            XFailRule(
                op_match_fn=lambda device, op: op.name == "logcumsumexp",
                sample_match_fn=lambda device, sample: (
                    sample.input.dtype == torch.cdouble
                    and sample.input.ndim > 0
                    and sample.input.abs().max() > 1000
                ),
                error_type=GradcheckError,
                error_msg="Forward-over-forward AD: .*mismatch",
                name="Complex logcumsumexp underflow: #196705",
            ),
            XFailRule(
                op_match_fn=lambda device, op: (
                    device == "cpu" or device == "cuda" and not TEST_WITH_ROCM
                )
                and op.name
                in (
                    "nn.functional.dropout",
                    "nn.functional.dropout2d",
                    "nn.functional.dropout3d",
                ),
                sample_match_fn=lambda device, sample: (
                    sample.input.dtype == torch.cdouble
                    and sample.kwargs.get("training", True)
                    and 0 < sample.kwargs.get("p", 0.5) < 1
                ),
                error_type=NotImplementedError,
                error_msg=r"(bernoulli_scalar_(cpu|cuda)_|fused_dropout).*ComplexDouble",
                name="Bernoulli and fused dropout have no complex kernel",
            ),
            XFailRule(
                op_match_fn=lambda device, op: (
                    device == "cpu" or device == "cuda" and not TEST_WITH_ROCM
                )
                and op.name
                in (
                    "nn.functional.alpha_dropout",
                    "nn.functional.feature_alpha_dropout",
                ),
                sample_match_fn=lambda device, sample: (
                    sample.input.dtype == torch.cdouble
                    and sample.kwargs.get("training", False)
                    and 0 < sample.kwargs.get("p", 0.5) < 1
                ),
                error_type=NotImplementedError,
                error_msg=r"bernoulli_scalar_(cpu|cuda)_.*ComplexDouble",
                name="Bernoulli has no complex kernel",
            ),
            XFailRule(
                op_match_fn=lambda device, op: op.name == "nn.functional.alpha_dropout",
                sample_match_fn=lambda device, sample: (
                    sample.input.dtype == torch.double
                    and sample.input.ndim == 0
                    and sample.kwargs.get("training", False)
                    and 0 < sample.kwargs.get("p", 0.5) < 1
                ),
                error_type=RuntimeError,
                error_msg="ZeroTensors are immutable",
                name="Scalar alpha_dropout mutates a zero tangent",
            ),
            XFailRule(
                op_match_fn=lambda device, op: op.name == "tensor_split",
                sample_match_fn=lambda device, sample: isinstance(
                    sample.args[0], torch.Tensor
                )
                and sample.args[0].ndim > 0,
                error_type=RuntimeError,
                error_msg="Cannot access data pointer of Tensor that doesn't have storage",
                name="Tensor split indices are storage-less under nested JVP",
            ),
            XFailRule(
                op_match_fn=lambda device, op: op.name
                in (
                    "nn.functional.glu",
                    "nn.functional.hardsigmoid",
                    "nn.functional.huber_loss",
                    "nn.functional.soft_margin_loss",
                ),
                sample_match_fn=lambda device, sample: sample.input.numel() > 0,
                error_type=NotImplementedError,
                error_msg=r"Trying to use forward AD with (aten::glu_jvp|aten::hardsigmoid_backward|huber_loss_backward|soft_margin_loss_backward) that does not support it",
                name="The first JVP calls an operator without a forward-AD rule",
            ),
        ]
    )
    def test_fn_fwgrad_fwgrad(self, device, dtype, op):
        self._skip_helper(op, device, dtype)
        samples = op.sample_inputs(
            device,
            dtype,
            requires_grad=True,
            use_subtests=True,
            small_inputs_only=TEST_WITH_SLOW_GRADCHECK,
        )
        for sample, subtest, expectation in samples:
            with subtest(self), expectation(self):
                self._check_helper(
                    device, dtype, op, op.get_op(), "fwgrad_fwgrad", samples=(sample,)
                )

    @_gradcheck_ops([op for op in op_db if op.derivative_inputs_func is not None])
    @sample_skips_and_xfails(
        [
            XFailRule(
                op_match_fn=lambda device, op: op.name
                in (
                    "cholesky_inverse",
                    "cholesky_solve",
                    "logcumsumexp",
                    "scatter_reduce",
                ),
                error_type=GradcheckError,
                error_msg="Jacobian computed with forward mode mismatch",
                name="Forward formulas: #196682, #196694, #196705, #196702",
            ),
            XFailRule(
                op_match_fn=lambda device, op: op.name == "triangular_solve",
                sample_match_fn=lambda device, sample: sample.kwargs["unitriangular"]
                or sample.kwargs["transpose"],
                error_type=GradcheckError,
                error_msg="Jacobian computed with forward mode mismatch",
                name="triangular_solve flags: #196707, #198602",
            ),
        ]
    )
    def test_forward_mode_AD_boundary(self, device, dtype, op):
        for sample, subtest, expectation in op.derivative_inputs(
            device, dtype, requires_grad=True, use_subtests=True
        ):
            with subtest(self), expectation(self):
                self._check_helper(
                    device,
                    dtype,
                    op,
                    op.get_op(),
                    "gradcheck",
                    samples=(sample,),
                    check_forward_ad=True,
                    check_backward_ad=False,
                    check_batched_grad=False,
                )

    @_gradcheck_ops([op for op in op_db if op.derivative_inputs_func is not None])
    @sample_skips_and_xfails(
        [
            XFailRule(
                op_match_fn=lambda device, op: op.name
                in ("cholesky_inverse", "cholesky_solve", "cumprod", "logaddexp"),
                error_type=GradcheckError,
                error_msg="Forward-over-forward AD: .*mismatch",
                name="Nested formulas: #196682, #196694, #196701, #196704",
            ),
            XFailRule(
                op_match_fn=lambda device, op: op.name == "sinc",
                sample_match_fn=lambda device, sample: sample.name == "zero",
                error_type=GradcheckError,
                error_msg="Forward-over-forward AD: .*mismatch",
                name="sinc removable singularity: #196703",
            ),
            XFailRule(
                op_match_fn=lambda device, op: op.name == "triangular_solve",
                sample_match_fn=lambda device, sample: sample.kwargs["unitriangular"]
                or sample.kwargs["transpose"],
                error_type=GradcheckError,
                error_msg="Forward-over-forward AD: .*mismatch",
                name="triangular_solve flags: #196707, #198602",
            ),
        ]
    )
    def test_fn_fwgrad_fwgrad_boundary(self, device, dtype, op):
        for sample, subtest, expectation in op.derivative_inputs(
            device, dtype, requires_grad=True, use_subtests=True
        ):
            with subtest(self), expectation(self):
                self._check_helper(
                    device, dtype, op, op.get_op(), "fwgrad_fwgrad", samples=(sample,)
                )

    # Test that forward-over-reverse gradgrad is computed correctly
    @skipOps(
        _fwd_grad_all
        | {
            skip("sparse.sampled_addmm"),
            skip("sparse.mm", variant_name="reduce"),
            xfail(
                "as_strided",
                variant_name="partial_views",
                dtypes=(torch.complex64, torch.complex128),
            ),
            skip("as_strided_scatter"),
            xfail("triangular_solve"),
            skip("svd_lowrank", dtypes=(torch.complex128,)),
            skip("pca_lowrank", dtypes=(torch.complex128,)),
            xfail("polar"),
            xfail("logcumsumexp", dtypes=(torch.complex128,)),
            xfail("scatter_reduce", variant_name="prod"),
            xfail("linalg.norm", variant_name="subgradients_at_zero"),
        }
    )
    @_gradcheck_ops(op_db)
    def test_fn_fwgrad_bwgrad(self, device, dtype, op):
        self._skip_helper(op, device, dtype)

        if op.supports_fwgrad_bwgrad:
            self._check_helper(device, dtype, op, op.get_op(), "fwgrad_bwgrad")
        else:
            err_msg = r"Trying to use forward AD with .* that does not support it"
            hint_msg = (
                "Running forward-over-backward gradgrad for an OP that has does not support it did not "
                "raise any error. If your op supports forward AD, you should set supports_fwgrad_bwgrad=True."
            )
            with self.assertRaisesRegex(NotImplementedError, err_msg, msg=hint_msg):
                self._check_helper(device, dtype, op, op.get_op(), "fwgrad_bwgrad")

    def _forward_grad_helper(self, device, dtype, op, variant, is_inplace):
        # TODO: clean up how attributes are passed to gradcheck from OpInfos
        def call_grad_test_helper():
            check_batched_forward_grad = (
                op.check_batched_forward_grad and not is_inplace
            ) or (op.check_inplace_batched_forward_grad and is_inplace)
            self._grad_test_helper(
                device,
                dtype,
                op,
                variant,
                check_forward_ad=True,
                check_backward_ad=False,
                check_batched_grad=False,
                check_batched_forward_grad=check_batched_forward_grad,
            )

        if op.supports_forward_ad:
            call_grad_test_helper()
        else:
            err_msg = r"Trying to use forward AD with .* that does not support it"
            hint_msg = (
                "Running forward AD for an OP that has does not support it did not "
                "raise any error. If your op supports forward AD, you should set supports_forward_ad=True"
            )
            with self.assertRaisesRegex(NotImplementedError, err_msg, msg=hint_msg):
                call_grad_test_helper()

    @skipif(
        platform.machine() == "s390x",
        reason="Different precision of openblas functions: https://github.com/OpenMathLib/OpenBLAS/issues/4194",
    )
    @skipOps(
        _fwd_grad_all
        | {
            xfail("cov", dtypes=(torch.cdouble,)),
            skip("sparse.sampled_addmm"),
            skip("sparse.mm", variant_name="reduce"),
            xfail("as_strided", variant_name="partial_views"),
            xfail("as_strided_scatter"),
            skip("native_layer_norm"),
            skip("native_batch_norm"),
            skip("_native_batch_norm_legit"),
            skip("_batch_norm_with_update"),
            skip("nn.functional.scaled_dot_product_attention"),
            xfail("bernoulli"),
            xfail("logcumsumexp", dtypes=(torch.complex128,)),
            xfail("nn.functional.feature_alpha_dropout", variant_name="with_train"),
            skip("nn.functional.multi_head_attention_forward"),
            xfail("scatter_reduce", variant_name="prod"),
            xfail("linalg.norm", variant_name="subgradients_at_zero"),
        }
    )
    @_gradcheck_ops(op_db)
    def test_forward_mode_AD(self, device, dtype, op):
        self._skip_helper(op, device, dtype)

        self._forward_grad_helper(device, dtype, op, op.get_op(), is_inplace=False)

    @skipIfTorchInductor("to be fixed")
    @skipOps(
        _fwd_grad_all
        | {
            skip("abs", dtypes=(torch.cdouble,)),
            xfail("as_strided", variant_name="partial_views"),
            xfail("nn.functional.rrelu"),
            xfail("nn.functional.feature_alpha_dropout", variant_name="with_train"),
            xfail("scatter_reduce", variant_name="prod"),
        }
    )
    @_gradcheck_ops(op_db)
    def test_inplace_forward_mode_AD(self, device, dtype, op):
        self._skip_helper(op, device, dtype)

        if not op.inplace_variant or not op.supports_inplace_autograd:
            self.skipTest("Skipped! Operation does not support inplace autograd.")

        self._forward_grad_helper(
            device, dtype, op, self._get_safe_inplace(op.get_inplace()), is_inplace=True
        )


instantiate_device_type_tests(TestFwdGradients, globals(), allow_xpu=True)

if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()
