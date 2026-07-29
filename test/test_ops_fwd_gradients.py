# Owner(s): ["module: unknown"]

import platform
from functools import partial
from unittest import skipIf as skipif

import torch
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
    TestCase,
    TestGradients,
    unMarkDynamoStrictTest,
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
    skip("__rpow__"),
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
    # Test that forward-over-reverse gradgrad is computed correctly
    @skipOps(
        _fwd_grad_all
        | {
            skip("cov"),
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
            xfail("cov"),
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


instantiate_device_type_tests(TestFwdGradients, globals())

if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()
