# Owner(s): ["oncall: pt2"]

"""
Test suite for OpInfo ops with unbacked symints.

This test marks tensor dimensions as unbacked and verifies that ops
can be compiled with fullgraph=True without data-dependent errors (DDEs).
"""

import copy
import unittest

import torch
import torch._dynamo
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
)
from torch.testing._internal.common_methods_invocations import DecorateInfo, op_db
from torch.testing._internal.common_utils import run_tests, suppress_warnings, TestCase
from torch.utils._pytree import tree_flatten, tree_map_


DEVICE_TYPE = "cpu"


# Copied from test_dtensor_ops.py
def xfail(op_name, variant_name="", *, device_type=None, dtypes=None):
    return (op_name, variant_name, device_type, dtypes, True)


def skip(op_name, variant_name="", *, device_type=None, dtypes=None):
    return (op_name, variant_name, device_type, dtypes, False)


def apply_skip_decorators(all_opinfos, test_case_name, base_test_name, to_skip):
    # Build lookup dict for O(n) performance
    opinfo_by_name = {}
    for o in all_opinfos:
        key = (o.name, o.variant_test_name)
        opinfo_by_name.setdefault(key, []).append(o)

    for xfail_entry in to_skip:
        op_name, variant_name, device_type, dtypes, expected_failure = xfail_entry
        matching_opinfos = opinfo_by_name.get((op_name, variant_name), [])
        # Some ops may not exist in op_db, skip silently
        for opinfo in matching_opinfos:
            decorators = list(opinfo.decorators)
            if expected_failure:
                decorator = DecorateInfo(
                    unittest.expectedFailure,
                    test_case_name,
                    base_test_name,
                    device_type=device_type,
                    dtypes=dtypes,
                )
                decorators.append(decorator)
            else:
                decorator = DecorateInfo(
                    unittest.skip("Skipped!"),
                    test_case_name,
                    base_test_name,
                    device_type=device_type,
                    dtypes=dtypes,
                )
                decorators.append(decorator)
            opinfo.decorators = tuple(decorators)


# Ops that have data-dependent errors with unbacked dimensions.
# These are base tensor issues (not DTensor-related).
ops_dde_xfail = {
    xfail("_chunk_cat"),
    xfail("_unsafe_masked_index_put_accumulate"),
    xfail("_upsample_bilinear2d_aa"),
    xfail("addmv"),
    xfail("allclose"),
    xfail("as_strided_scatter"),
    xfail("baddbmm"),
    xfail("bernoulli"),
    xfail("cauchy"),
    xfail("cdist"),
    xfail("cholesky"),
    xfail("cholesky_inverse"),
    xfail("chunk"),
    xfail("combinations"),
    xfail("corrcoef"),
    xfail("cov"),
    xfail("cross"),
    xfail("cummax"),
    xfail("cummin"),
    xfail("cumulative_trapezoid"),
    xfail("diag"),
    xfail("diagonal"),
    xfail("diagonal_copy"),
    xfail("diagonal_scatter"),
    xfail("diff"),
    xfail("dist"),
    xfail("dsplit"),
    xfail("equal"),
    xfail("exponential"),
    xfail("fft.fft"),
    xfail("fft.fft2"),
    xfail("fft.fftn"),
    xfail("fft.fftshift"),
    xfail("fft.hfft"),
    xfail("fft.hfft2"),
    xfail("fft.hfftn"),
    xfail("fft.ifft"),
    xfail("fft.ifft2"),
    xfail("fft.ifftn"),
    xfail("fft.ifftshift"),
    xfail("fft.ihfft"),
    xfail("fft.ihfft2"),
    xfail("fft.ihfftn"),
    xfail("fft.irfft"),
    xfail("fft.irfft2"),
    xfail("fft.irfftn"),
    xfail("fft.rfft"),
    xfail("fft.rfft2"),
    xfail("fft.rfftn"),
    xfail("float"),
    xfail("geometric"),
    xfail("geqrf"),
    xfail("gradient"),
    xfail("grid_sampler_2d"),
    xfail("hash_tensor"),
    xfail("histogram"),
    xfail("histogramdd"),
    xfail("hsplit"),
    xfail("index_fill"),
    xfail("inner"),
    xfail("kron"),
    xfail("linalg.cross"),
    xfail("linalg.cholesky"),
    xfail("linalg.cholesky_ex"),
    xfail("linalg.cond"),
    xfail("linalg.det"),
    xfail("linalg.diagonal"),
    xfail("linalg.eig"),
    xfail("linalg.eigh"),
    xfail("linalg.eigvals"),
    xfail("linalg.eigvalsh"),
    xfail("linalg.householder_product"),
    xfail("linalg.inv"),
    xfail("linalg.inv_ex"),
    xfail("linalg.ldl_factor"),
    xfail("linalg.ldl_factor_ex"),
    xfail("linalg.ldl_solve"),
    xfail("linalg.lstsq"),
    xfail("linalg.lstsq", "grad_oriented"),
    xfail("linalg.lu"),
    xfail("linalg.lu_factor"),
    xfail("linalg.lu_factor_ex"),
    xfail("linalg.lu_solve"),
    xfail("linalg.matrix_norm"),
    xfail("linalg.matrix_power"),
    xfail("linalg.matrix_rank"),
    xfail("linalg.matrix_rank", "hermitian"),
    xfail("linalg.multi_dot"),
    xfail("linalg.norm"),
    xfail("linalg.norm", "subgradients_at_zero"),
    xfail("linalg.pinv"),
    xfail("linalg.pinv", "hermitian"),
    xfail("linalg.qr"),
    xfail("linalg.slogdet"),
    xfail("linalg.solve"),
    xfail("linalg.solve_ex"),
    xfail("linalg.solve_triangular"),
    xfail("linalg.svd"),
    xfail("linalg.svdvals"),
    xfail("linalg.tensorinv"),
    xfail("linalg.tensorsolve"),
    xfail("linalg.vander"),
    xfail("linalg.vector_norm"),
    xfail("log_normal"),
    xfail("logdet"),
    xfail("logsumexp"),
    xfail("lu"),
    xfail("lu_solve"),
    xfail("lu_unpack"),
    xfail("masked.amax"),
    xfail("masked.amin"),
    xfail("masked.argmax"),
    xfail("masked.argmin"),
    xfail("masked.cumprod"),
    xfail("masked.cumsum"),
    xfail("masked.log_softmax"),
    xfail("masked.logaddexp"),
    xfail("masked.logsumexp"),
    xfail("masked.mean"),
    xfail("masked.median"),
    xfail("masked.norm"),
    xfail("masked.prod"),
    xfail("masked.softmax"),
    xfail("masked.softmin"),
    xfail("masked.std"),
    xfail("masked.sum"),
    xfail("masked.var"),
    xfail("matrix_exp"),
    xfail("max_pool2d_with_indices_backward"),
    xfail("multinomial"),
    xfail("nanquantile"),
    xfail("nn.functional.adaptive_avg_pool1d"),
    xfail("nn.functional.adaptive_avg_pool2d"),
    xfail("nn.functional.adaptive_avg_pool3d"),
    xfail("nn.functional.adaptive_max_pool1d"),
    xfail("nn.functional.adaptive_max_pool2d"),
    xfail("nn.functional.adaptive_max_pool3d"),
    xfail("nn.functional.alpha_dropout"),
    xfail("nn.functional.avg_pool1d"),
    xfail("nn.functional.avg_pool2d"),
    xfail("nn.functional.avg_pool3d"),
    xfail("nn.functional.batch_norm"),
    xfail("nn.functional.bilinear"),
    xfail("nn.functional.binary_cross_entropy"),
    xfail("nn.functional.binary_cross_entropy_with_logits"),
    xfail("nn.functional.channel_shuffle"),
    xfail("nn.functional.conv1d"),
    xfail("nn.functional.conv2d"),
    xfail("nn.functional.conv3d"),
    xfail("nn.functional.conv_transpose1d"),
    xfail("nn.functional.conv_transpose2d"),
    xfail("nn.functional.conv_transpose3d"),
    xfail("nn.functional.cosine_similarity"),
    xfail("nn.functional.cross_entropy"),
    xfail("nn.functional.ctc_loss"),
    xfail("nn.functional.dropout"),
    xfail("nn.functional.dropout2d"),
    xfail("nn.functional.dropout3d"),
    xfail("nn.functional.embedding"),
    xfail("nn.functional.embedding_bag"),
    xfail("nn.functional.feature_alpha_dropout"),
    xfail("nn.functional.feature_alpha_dropout", "with_train"),
    xfail("nn.functional.feature_alpha_dropout", "without_train"),
    xfail("nn.functional.fractional_max_pool2d"),
    xfail("nn.functional.fractional_max_pool3d"),
    xfail("nn.functional.gaussian_nll_loss"),
    xfail("nn.functional.glu"),
    xfail("nn.functional.grid_sample"),
    xfail("nn.functional.group_norm"),
    xfail("nn.functional.huber_loss"),
    xfail("nn.functional.instance_norm"),
    xfail("nn.functional.interpolate", "area"),
    xfail("nn.functional.interpolate", "bicubic"),
    xfail("nn.functional.interpolate", "bilinear"),
    xfail("nn.functional.interpolate", "linear"),
    xfail("nn.functional.interpolate", "trilinear"),
    xfail("nn.functional.l1_loss"),
    xfail("nn.functional.local_response_norm"),
    xfail("nn.functional.max_pool1d"),
    xfail("nn.functional.max_pool2d"),
    xfail("nn.functional.max_pool3d"),
    xfail("nn.functional.max_unpool1d"),
    xfail("nn.functional.max_unpool1d", "grad"),
    xfail("nn.functional.max_unpool2d"),
    xfail("nn.functional.max_unpool2d", "grad"),
    xfail("nn.functional.max_unpool3d"),
    xfail("nn.functional.max_unpool3d", "grad"),
    xfail("nn.functional.mse_loss"),
    xfail("nn.functional.multi_head_attention_forward"),
    xfail("nn.functional.multilabel_margin_loss"),
    xfail("nn.functional.nll_loss"),
    xfail("nn.functional.pad"),
    xfail("nn.functional.pad", "circular"),
    xfail("nn.functional.pad", "reflect"),
    xfail("nn.functional.pad", "replicate"),
    xfail("nn.functional.pad", "replicate_negative"),
    xfail("nn.functional.pdist"),
    xfail("nn.functional.pixel_shuffle"),
    xfail("nn.functional.prelu"),
    xfail("nn.functional.rrelu"),
    xfail("nn.functional.scaled_dot_product_attention"),
    xfail("nn.functional.smooth_l1_loss"),
    xfail("nn.functional.unfold"),
    xfail("nn.functional.upsample_bilinear"),
    xfail("norm"),
    xfail("norm", "fro"),
    xfail("norm", "nuc"),
    xfail("normal"),
    xfail("normal", "in_place"),
    xfail("normal", "number_mean"),
    xfail("ormqr"),
    xfail("pca_lowrank"),
    xfail("pinverse"),
    xfail("quantile"),
    xfail("qr"),
    xfail("rand_like"),
    xfail("randint_like"),
    xfail("randn_like"),
    xfail("repeat_interleave"),
    xfail("resize_"),
    xfail("resize_as_"),
    xfail("roll"),
    xfail("scatter"),
    xfail("scatter_add"),
    xfail("scatter_reduce"),
    xfail("scatter_reduce", "amax"),
    xfail("scatter_reduce", "amin"),
    xfail("scatter_reduce", "mean"),
    xfail("scatter_reduce", "prod"),
    xfail("scatter_reduce", "sum"),
    xfail("searchsorted"),
    xfail("sparse.mm", "reduce"),
    xfail("split"),
    xfail("stft"),
    xfail("svd"),
    xfail("svd_lowrank"),
    xfail("sum_to_size"),
    xfail("take"),
    xfail("take_along_dim"),
    xfail("tensordot"),
    xfail("tensor_split"),
    xfail("to_sparse"),
    xfail("trace"),
    xfail("trapezoid"),
    xfail("trapz"),
    xfail("triangular_solve"),
    xfail("unbind"),
    xfail("unbind_copy"),
    xfail("uniform"),
    xfail("unsafe_chunk"),
    xfail("unsafe_split"),
    xfail("view_as_complex"),
    xfail("vsplit"),
}

# Ops that skip (no valid sample with markable dims, or can't be deepcopied)
ops_skip = {
    skip("arange"),
    skip("broadcast_shapes"),
    skip("empty"),
    skip("empty_permuted"),
    skip("empty_strided"),
    skip("eye"),
    skip("full"),
    skip("item"),
    skip("linspace"),
    skip("linspace", "tensor_overload"),
    skip("logspace"),
    skip("logspace", "tensor_overload"),
    skip("ones"),
    skip("randint"),
    skip("randn"),
    skip("scalar_tensor"),
    skip("signal.windows.bartlett"),
    skip("signal.windows.blackman"),
    skip("signal.windows.cosine"),
    skip("signal.windows.exponential"),
    skip("signal.windows.gaussian"),
    skip("signal.windows.general_cosine"),
    skip("signal.windows.general_hamming"),
    skip("signal.windows.hamming"),
    skip("signal.windows.hann"),
    skip("signal.windows.kaiser"),
    skip("signal.windows.nuttall"),
    skip("zeros"),
    # Sparse ops that can't be deepcopied
    skip("sparse.mm"),
    skip("sparse.sampled_addmm"),
}


apply_skip_decorators(
    op_db, "TestOpsUnbacked", "test_unbacked_op_db", ops_dde_xfail | ops_skip
)


class TestOpsUnbacked(TestCase):
    def _has_valid_unbacked_dims(self, t: torch.Tensor) -> bool:
        """Check if tensor has dimensions that can be marked as unbacked."""
        return t.ndim > 0 and any(s >= 2 for s in t.shape)

    def _mark_unbacked(self, t: torch.Tensor) -> None:
        """Mark all eligible dimensions of a tensor as unbacked."""
        for i in range(t.ndim):
            if t.shape[i] >= 2:
                torch._dynamo.decorators.mark_unbacked(t, i)

    def _run_with_unbacked_compile(self, func, args, kwargs):
        """
        Mark tensor dims as unbacked and run with fullgraph compile.
        Raises if a DDE occurs.
        """
        torch._dynamo.reset()

        def mark_unbacked_tree(x):
            if isinstance(x, torch.Tensor) and self._has_valid_unbacked_dims(x):
                self._mark_unbacked(x)
            return x

        tree_map_(mark_unbacked_tree, (args, kwargs))

        @torch.compile(backend="eager", fullgraph=True)
        def compiled_func(*a, **kw):
            return func(*a, **kw)

        compiled_func(*args, **kwargs)

    @suppress_warnings
    @ops(op_db, allowed_dtypes=(torch.float,))
    def test_unbacked_op_db(self, device, dtype, op):
        samples = list(op.sample_inputs(device, dtype, requires_grad=False))

        any_tested = False

        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs
            all_tensors = [
                x
                for x in tree_flatten((args, kwargs))[0]
                if isinstance(x, torch.Tensor)
            ]
            if not any(self._has_valid_unbacked_dims(t) for t in all_tensors):
                continue

            # # First verify the sample passes in eager mode
            op.op(*copy.deepcopy(args), **copy.deepcopy(kwargs))

            any_tested = True
            args_copy, kwargs_copy = copy.deepcopy((args, kwargs))
            self._run_with_unbacked_compile(op.op, args_copy, kwargs_copy)

        if not any_tested:
            self.fail("Should have skipped; no valid samples found")


instantiate_device_type_tests(TestOpsUnbacked, globals(), only_for=(DEVICE_TYPE,))

if __name__ == "__main__":
    run_tests()
