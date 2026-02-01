# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import copy
import re
import unittest
import warnings

import torch
import torch.distributed as dist
import torch.testing._internal.common_methods_invocations as common_ops
from torch.distributed._local_tensor import LocalTensorMode, reconcile_args
from torch.distributed.tensor import (
    distribute_tensor,
    DTensor,
    init_device_mesh,
    Replicate,
    Shard,
)
from torch.distributed.tensor._ops.single_dim_strategy import _ShardingPlaceholder
from torch.overrides import resolve_name
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
)
from torch.testing._internal.common_methods_invocations import DecorateInfo, op_db
from torch.testing._internal.common_utils import run_tests, suppress_warnings, TestCase
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorConverter,
    DTensorOpTestBase,
    validate_sharding_rule_sample,
)
from torch.utils import _pytree as pytree
from torch.utils._debug_mode import _OpCall, DebugMode
from torch.utils._pytree import tree_map


# rewrite common size variables to sth can be sharded evenly
# we can enable uneven shards later, but need to adjust more on
# sample inputs (i.e. view/reshape need to adjust shape size as well)
common_ops.L = 24
common_ops.M = 12
common_ops.S = 4
common_ops.XS = 2


# Copied from functorch
def xfail(op_name, variant_name="", *, device_type=None, dtypes=None):
    return (op_name, variant_name, device_type, dtypes, True)


def skip(op_name, variant_name="", *, device_type=None, dtypes=None):
    return (op_name, variant_name, device_type, dtypes, False)


def skipOps(op_db, test_case_name, base_test_name, to_skip):
    all_opinfos = op_db
    for xfail in to_skip:
        op_name, variant_name, device_type, dtypes, expected_failure = xfail
        matching_opinfos = [
            o
            for o in all_opinfos
            if o.name == op_name and o.variant_test_name == variant_name
        ]
        assert len(matching_opinfos) >= 1, f"Couldn't find OpInfo for {xfail}"
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

    # This decorator doesn't modify fn in any way
    def wrapped(fn):
        return fn

    return wrapped


def repurpose_ops(op_db, base_test_name, derived_test_name):
    """
    Copies op info database and for the decorators that applied to base test class updates
    them to apply to derived test class. The class update is required because decorators are applied
    only if the class name matches (it doesn't consider base classes).

    Specifically we use this function to create two test classes (one for multi-threaded and one for
    local tensor flavors) that share common test body but different rules for skip or fail.

    Args:
        op_db: List of OpInfo objects to be repurposed.
        base_test_name: The original test class name to be replaced.
        derived_test_name: The new test class name to set in decorators.

    Returns:
        list: A new list of OpInfo objects with updated target class names for the
        decorator.
    """
    repurposed_ops = []
    for opinfo in op_db:
        opinfo_copy = copy.deepcopy(opinfo)
        for decorator in list(opinfo_copy.decorators):
            if hasattr(decorator, "cls_name") and decorator.cls_name == base_test_name:
                decorator.cls_name = derived_test_name
        repurposed_ops.append(opinfo_copy)
    return repurposed_ops


# Re-generate this failed list, turn on dry_run of the below func
# check_dtensor_func(self, test, op, dry_run=True), then run sth
# like python test/distributed/tensor/test_dtensor_ops.py > failed.expect
dtensor_fails = {
    # these sometimes pass and sometimes fail
    # we need to remove many of them from list once op
    # get full support with varying sharding specs
    xfail("__getitem__"),
    xfail("__rsub__"),
    xfail("_chunk_cat"),
    xfail("_native_batch_norm_legit"),
    xfail("_upsample_bilinear2d_aa"),
    xfail("addbmm"),
    xfail("addmv"),
    xfail("addr"),
    xfail("all"),
    xfail("allclose"),
    xfail("alias_copy"),
    xfail("aminmax"),
    xfail("any"),
    xfail("arange"),
    xfail("as_strided"),
    xfail("as_strided", "partial_views"),
    xfail("as_strided_copy"),
    xfail("as_strided_scatter"),
    xfail("bernoulli"),
    xfail("_batch_norm_with_update"),
    xfail("block_diag"),
    xfail("broadcast_shapes"),
    xfail("cartesian_prod"),
    xfail("cauchy"),
    xfail("cdist"),
    xfail("cholesky"),
    xfail("cholesky_inverse"),
    xfail("cholesky_solve"),
    xfail("combinations"),
    xfail("complex"),
    xfail("count_nonzero"),
    xfail("cross"),
    xfail("cummax"),
    xfail("cummin"),
    xfail("diagonal_scatter"),
    xfail("dist"),
    xfail("expand_copy"),
    xfail("exponential"),
    xfail("eye"),
    xfail("fft.fft2"),
    xfail("fft.fft"),
    xfail("fft.fftn"),
    xfail("fft.fftshift"),
    xfail("fft.ifft2"),
    xfail("fft.ifft"),
    xfail("fft.ifftshift"),
    xfail("fft.ihfft2"),
    xfail("fft.ihfft"),
    xfail("fft.ihfftn"),
    xfail("fft.irfft2"),
    xfail("fft.irfftn"),
    xfail("fft.rfft2"),
    xfail("fft.rfft"),
    xfail("fft.rfftn"),
    xfail("fill"),
    xfail("flatten"),
    xfail("flip"),
    xfail("fliplr"),
    xfail("flipud"),
    xfail("floor_divide"),
    xfail("frexp"),
    xfail("full"),
    xfail("geometric"),
    xfail("geqrf"),
    xfail("grid_sampler_2d"),
    xfail("histogram"),
    xfail("histogramdd"),
    xfail("index_add"),
    xfail("index_copy"),
    xfail("index_fill"),
    xfail("index_put"),
    xfail("index_reduce", "prod"),
    xfail("index_reduce", "mean"),
    xfail("index_reduce", "amax"),
    xfail("index_reduce", "amin"),
    xfail("index_select"),
    xfail("isin"),
    xfail("kthvalue"),
    xfail("kron"),
    xfail("linalg.cholesky"),
    xfail("linalg.cholesky_ex"),
    xfail("linalg.cross"),
    xfail("linalg.det"),
    xfail("linalg.eig"),
    xfail("linalg.eigvals"),
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
    xfail("linalg.matrix_power"),
    xfail("linalg.pinv"),
    xfail("linalg.pinv", "hermitian"),
    xfail("linalg.slogdet"),
    xfail("linalg.solve"),
    xfail("linalg.solve_ex"),
    xfail("linalg.solve_triangular"),
    xfail("linalg.tensorinv"),
    xfail("linalg.tensorsolve"),
    xfail("linalg.vander"),
    xfail("linalg.vecdot"),
    xfail("linspace"),
    xfail("linspace", "tensor_overload"),
    xfail("log_normal"),
    xfail("logcumsumexp"),
    xfail("logdet"),
    xfail("logspace"),
    xfail("logspace", "tensor_overload"),
    xfail("logsumexp"),
    xfail("lu"),
    xfail("lu_solve"),
    xfail("lu_unpack"),
    xfail("masked_fill"),
    xfail("masked_scatter"),
    xfail("masked_select"),
    xfail("masked.logsumexp"),
    xfail("masked.median"),
    xfail("matrix_exp"),
    xfail("median"),
    xfail("mode"),
    xfail("multinomial"),
    xfail("mv"),
    xfail("max_pool2d_with_indices_backward", ""),
    xfail("nanmean"),
    xfail("nanmedian"),
    xfail("nanquantile"),
    xfail("nansum"),
    xfail("native_batch_norm"),
    xfail("narrow_copy"),
    xfail("ne"),
    xfail("transpose"),
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
    xfail("nn.functional.batch_norm", "without_cudnn"),
    xfail("nn.functional.bilinear"),
    xfail("nn.functional.binary_cross_entropy"),
    xfail("nn.functional.binary_cross_entropy_with_logits"),
    xfail("nn.functional.celu"),
    xfail("nn.functional.conv1d"),
    xfail("nn.functional.conv2d"),
    xfail("nn.functional.conv3d"),
    xfail("nn.functional.conv_transpose1d"),
    xfail("nn.functional.conv_transpose2d"),
    xfail("nn.functional.conv_transpose3d"),
    xfail("nn.functional.cosine_similarity"),
    xfail("nn.functional.ctc_loss"),
    xfail("nn.functional.dropout"),
    xfail("nn.functional.elu"),
    xfail("nn.functional.fractional_max_pool2d"),
    xfail("nn.functional.fractional_max_pool3d"),
    xfail("nn.functional.glu"),
    xfail("nn.functional.grid_sample"),
    xfail("nn.functional.group_norm"),
    xfail("nn.functional.hardshrink"),
    xfail("nn.functional.hardsigmoid"),
    xfail("nn.functional.hardswish"),
    xfail("nn.functional.hardtanh"),
    xfail("nn.functional.huber_loss"),
    xfail("nn.functional.instance_norm"),
    xfail("nn.functional.interpolate", "area"),
    xfail("nn.functional.interpolate", "nearest"),
    xfail("nn.functional.interpolate", "nearest-exact"),
    xfail("nn.functional.leaky_relu"),
    xfail("nn.functional.local_response_norm"),
    xfail("nn.functional.logsigmoid"),
    xfail("nn.functional.margin_ranking_loss"),
    xfail("nn.functional.max_pool1d"),
    xfail("nn.functional.max_pool2d"),
    xfail("nn.functional.max_pool3d"),
    xfail("nn.functional.max_unpool1d"),
    xfail("nn.functional.max_unpool1d", "grad"),
    xfail("nn.functional.max_unpool2d"),
    xfail("nn.functional.max_unpool2d", "grad"),
    xfail("nn.functional.max_unpool3d"),
    xfail("nn.functional.max_unpool3d", "grad"),
    xfail("nn.functional.mish"),
    xfail("nn.functional.mse_loss"),
    xfail("nn.functional.multi_margin_loss"),
    xfail("nn.functional.multilabel_margin_loss"),
    xfail("nn.functional.multilabel_soft_margin_loss"),
    xfail("nn.functional.pad", "reflect"),
    xfail("nn.functional.pad", "replicate"),
    xfail("nn.functional.pad", "replicate_negative"),
    xfail("nn.functional.pairwise_distance"),
    xfail("nn.functional.pdist"),
    xfail("nn.functional.pixel_shuffle"),
    xfail("nn.functional.pixel_unshuffle"),
    xfail("nn.functional.prelu"),
    xfail("nn.functional.relu6"),
    xfail("nn.functional.rrelu"),
    xfail("nn.functional.selu"),
    xfail("nn.functional.smooth_l1_loss"),
    xfail("nn.functional.soft_margin_loss"),
    xfail("nn.functional.softplus"),
    xfail("nn.functional.softshrink"),
    xfail("nn.functional.threshold"),
    xfail("nn.functional.triplet_margin_loss"),
    xfail("nn.functional.triplet_margin_with_distance_loss"),
    xfail("nn.functional.unfold"),
    xfail("nn.functional.upsample_nearest"),
    xfail("nonzero"),
    xfail("normal"),
    xfail("normal", "number_mean"),
    xfail("normal", "in_place"),
    xfail("ormqr"),
    xfail("ones"),
    xfail("pca_lowrank"),
    xfail("permute_copy"),
    xfail("pinverse"),
    xfail("polar"),
    xfail("put"),
    xfail("quantile"),
    xfail("rand_like"),
    xfail("randint_like"),
    xfail("randint"),
    xfail("randn"),
    xfail("randn_like"),
    xfail("ravel"),
    xfail("renorm"),
    xfail("repeat_interleave"),
    xfail("resize_"),
    xfail("resize_as_"),
    xfail("reshape"),
    xfail("reshape_as"),
    xfail("roll"),
    xfail("rot90"),
    xfail("rsub"),
    xfail("scalar_tensor"),
    xfail("scatter_reduce", "amax"),
    xfail("scatter_reduce", "amin"),
    xfail("scatter_reduce", "mean"),
    xfail("scatter_reduce", "prod"),
    xfail("scatter_reduce", "sum"),
    xfail("searchsorted"),
    xfail("select_scatter"),
    xfail("sparse.sampled_addmm"),
    xfail("sparse.mm", "reduce"),
    xfail("special.airy_ai"),
    xfail("special.bessel_j0"),
    xfail("special.bessel_j1"),
    xfail("special.bessel_y0"),
    xfail("special.bessel_y1"),
    xfail("special.chebyshev_polynomial_t"),
    xfail("special.chebyshev_polynomial_u"),
    xfail("special.chebyshev_polynomial_v"),
    xfail("special.chebyshev_polynomial_w"),
    xfail("special.entr"),
    xfail("special.erfcx"),
    xfail("special.hermite_polynomial_h"),
    xfail("special.hermite_polynomial_he"),
    xfail("special.i0e"),
    xfail("special.i1"),
    xfail("special.i1e"),
    xfail("special.laguerre_polynomial_l"),
    xfail("special.legendre_polynomial_p"),
    xfail("special.log_ndtr"),
    xfail("special.modified_bessel_i0"),
    xfail("special.modified_bessel_i1"),
    xfail("special.modified_bessel_k0"),
    xfail("special.modified_bessel_k1"),
    xfail("special.ndtri"),
    xfail("special.scaled_modified_bessel_k0"),
    xfail("special.scaled_modified_bessel_k1"),
    xfail("special.shifted_chebyshev_polynomial_t"),
    xfail("special.shifted_chebyshev_polynomial_u"),
    xfail("special.shifted_chebyshev_polynomial_v"),
    xfail("special.shifted_chebyshev_polynomial_w"),
    xfail("special.spherical_bessel_j0"),
    xfail("special.xlog1py"),
    xfail("special.zeta"),
    xfail("squeeze", "multiple"),
    xfail("squeeze_copy"),
    xfail("signal.windows.bartlett"),
    xfail("signal.windows.blackman"),
    xfail("signal.windows.cosine"),
    xfail("signal.windows.exponential"),
    xfail("signal.windows.gaussian"),
    xfail("signal.windows.general_cosine"),
    xfail("signal.windows.general_hamming"),
    xfail("signal.windows.hamming"),
    xfail("signal.windows.hann"),
    xfail("signal.windows.nuttall"),
    xfail("signal.windows.kaiser"),
    xfail("stack"),
    xfail("std_mean"),
    xfail("std_mean", "unbiased"),
    xfail("stft"),
    xfail("svd_lowrank"),
    xfail("t_copy"),
    xfail("take"),
    xfail("take_along_dim"),
    xfail("tensor_split"),
    xfail("to_sparse"),
    xfail("trace"),
    xfail("triangular_solve"),
    xfail("unbind"),
    xfail("unbind_copy"),
    xfail("unfold"),
    xfail("unfold_copy"),
    xfail("uniform"),
    xfail("unflatten"),
    xfail("unique_consecutive"),
    xfail("unique"),
    xfail("unsafe_split"),
    xfail("unsafe_chunk"),
    xfail("_unsafe_masked_index"),
    xfail("_unsafe_masked_index_put_accumulate"),
    xfail("var_mean"),
    xfail("var_mean", "unbiased"),
    xfail("vdot"),
    xfail("view"),
    xfail("view_as"),
    xfail("view_copy"),
    xfail("zeros"),
    # /TODO(whc) debug/triage
    # ops inside this might even fail without dtensor
    # tests, as we rescale op db common test size factor (i.e. L, M, S)
    # which triggered the original function run failures with input
    # generation becomes wrong, we skip them for now but should enable later.
    # TODO: need to clean this list and remove all cases
    skip("argwhere"),
    skip("cumprod"),
    skip("__rmatmul__"),
    skip("meshgrid", "list_of_tensors"),
    skip("meshgrid", "variadic_tensors"),
    skip("nn.functional.scaled_dot_product_attention"),
    skip("nn.functional.softmin"),
    skip("nn.functional.embedding"),
    skip("nn.functional.embedding_bag"),
    skip("nn.functional.feature_alpha_dropout", "with_train"),
    skip("nn.functional.feature_alpha_dropout", "without_train"),
    skip("nn.functional.hinge_embedding_loss"),
    skip("nn.functional.cosine_embedding_loss"),
    skip("fft.hfft"),
    skip("fft.hfft2"),
    skip("fft.hfft2"),
    skip("fft.hfftn"),
    skip("fft.ifftn"),
    skip("fft.irfft"),
    skip("istft"),
    skip("isclose"),
    skip("isreal"),
    skip("matmul"),
    skip("masked.mean"),
    skip("masked.var"),
    skip("masked.std"),
    skip("masked.normalize"),
    skip("prod"),
    skip("_segment_reduce", "lengths"),
    skip("_segment_reduce", "offsets"),
    # TODO: fix the following ops
    skip("squeeze"),
    skip("empty"),
    skip("empty_strided"),
    skip("empty_like"),
    skip("empty_permuted"),
    skip("new_empty"),
    skip("new_empty_strided"),
}

dtensor_multi_threaded_fails = {
    xfail("full_like"),
    xfail("nn.functional.dropout2d"),
    xfail("nn.functional.dropout3d"),
    xfail("masked.cumprod"),
    skip("nn.functional.multi_head_attention_forward"),
}

# Add a list of ops that are currently failing BW pass
skip_bw = [
    None,  # corresponds to the transpose ops 'H' and 'T'
    "torch.bucketize",
    "torch.conj_physical",
    "torch.eq",
    "torch.isfinite",
    "torch.isnan",
]


OP_DB_WORLD_SIZE = 4
# DEVICE_TYPE = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() >= OP_DB_WORLD_SIZE else "cpu"
# TODO: debug cuda illegal memory access issue and re-enable cuda tests
DEVICE_TYPE = "cpu"


class TestDTensorOps(TestCase):
    __test__ = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__test__ = True

    @property
    def world_size(self) -> int:
        return OP_DB_WORLD_SIZE

    def run_opinfo_test(
        self, dtype, op, requires_grad=True, sample_inputs_filter=lambda s: True
    ):
        self.mesh = init_device_mesh(DEVICE_TYPE, (self.world_size,))

        # test each op with dist tensor inputs and normal inputs
        def test():
            samples = op.sample_inputs(DEVICE_TYPE, dtype, requires_grad=requires_grad)
            for sample_input in samples:
                if not sample_inputs_filter(sample_input):
                    continue
                args = [sample_input.input] + list(sample_input.args)
                kwargs = sample_input.kwargs

                self.run_dtensor_crossref(op.op, args, kwargs)
                # we need to figure out a way to test the out variant, out variant testing
                # is tricky, as we need to pre allocate the dtensor out, some of them rely
                # on sharding placements to be pre-known (i.e. mm.out)
                # if isinstance(expected, torch.Tensor) and op.supports_out:
                #     func(*args, **kwargs, out=expected)

        self.check_dtensor_func(test, op)

    def assert_ref_dtensor_equal(self, dtensor_rs, rs):
        flat_dtensor_rs = pytree.tree_leaves(dtensor_rs)
        flat_rs = pytree.tree_leaves(rs)
        self.assertEqual(len(flat_dtensor_rs), len(flat_rs))
        for dtensor_r, r in zip(flat_dtensor_rs, flat_rs):
            if not isinstance(r, torch.Tensor):
                continue

            self.assertIsInstance(dtensor_r, torch.Tensor)
            self.assertEqualOnRank(
                dtensor_r.shape,
                r.shape,
                f"Shape mismatch! original shape:{r.shape}, dtensor shape: {dtensor_r.shape}",
            )
            self.assertEqualOnRank(
                dtensor_r.requires_grad,
                r.requires_grad,
                "op result requires_grad mismatch!"
                f"original requires_grad: {r.requires_grad}, "
                f"dtensor requires_grad: {dtensor_r.requires_grad}",
            )

            self.assertEqualOnRank(dtensor_r, r)

    def assertEqualOnRank(self, x, y, msg=None, *, rank=0) -> None:
        raise NotImplementedError

    def run_dtensor_crossref(self, func, args, kwargs):
        to_dtensor = DTensorConverter(self.mesh, args, kwargs)

        def concat_res_if_necessary(func, res: object) -> object:
            # concat the result on corresponding dim for ops like
            # split, so that we can call backward on a single tensor
            if (resolve_name(func) is not None) and ("split" in resolve_name(func)):
                dim = args[2] if len(args) == 3 else 0
                return torch.cat(res, dim=dim)
            else:
                return res

        # TODO: also handle cases where func raise an exception
        op_args, op_kwargs = reconcile_args(args, kwargs)
        rs = func(*op_args, **op_kwargs)
        rs = concat_res_if_necessary(func, rs)

        def to_replicate(e: object) -> object:
            return e.full_tensor() if isinstance(e, DTensor) else e

        # Suppress warnings, this doesn't matter for test_meta.py
        # but it does matter if you want to use this decorator
        # for cross-ref testing, as some tests may be looking at
        # errors
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # for every comb of sharding choices, we test if it works
            for dtensor_args, dtensor_kwargs in to_dtensor:
                # Only attempt if we managed to convert all tensors to DTensor
                # (if any of them failed, we're in a mixed tensor situation and
                # this is not allowed in DTensor)
                try:
                    if to_dtensor.successful():
                        # Handle special cases first if there's any
                        # Suppress warnings, this doesn't matter for test_meta.py
                        # but it does matter if you want to use this decorator
                        # for cross-ref testing, as some tests may be looking at
                        # errors
                        dtensor_rs = func(*dtensor_args, **dtensor_kwargs)

                        # we need to skip tests containing tensors of zero elements for now.
                        # see issue: https://github.com/pytorch/PiPPy/issues/470
                        # TODO remove this once issue above fixed.
                        flat_args = pytree.tree_leaves(dtensor_rs)
                        if any(
                            isinstance(e, torch.Tensor) and e.numel() == 0
                            for e in flat_args
                        ):
                            continue

                        # redistribute/all_gather the results to compare with normal output
                        dtensor_rs = tree_map(to_replicate, dtensor_rs)
                        dtensor_rs = concat_res_if_necessary(func, dtensor_rs)
                        try:
                            if resolve_name(func) not in skip_bw:
                                if isinstance(dtensor_rs, DTensor):
                                    dtensor_rs.to_local().sum().backward()
                                elif isinstance(dtensor_rs, tuple):
                                    dtensor_rs[0].to_local().sum().backward()

                        except Exception as e:
                            # TODO(anj): Remove this guard exception after gaining more confidence.
                            if torch.distributed.get_rank() == 0:
                                print(
                                    f"failed to run BW: {resolve_name(func)}, {func}, {str(e)})"
                                )
                        self.assert_ref_dtensor_equal(dtensor_rs, rs)
                    else:
                        raise RuntimeError(
                            f"Failed to convert args to DTensor; "
                            f"originally (*{args}, **{kwargs})"
                        )
                except Exception as e:
                    raise RuntimeError(
                        f"{str(e)}\n\nFailed to run: {resolve_name(func)}, with (*{dtensor_args}, **{dtensor_kwargs})"
                    ) from e
        return rs

    def check_dtensor_func(self, test_func, opinfo, dry_run=False):
        try:
            test_func()
        except Exception:
            if not dry_run:
                raise
            if dist.get_rank() == 0:
                if opinfo.variant_test_name:
                    print(f"xfail('{opinfo.name}', '{opinfo.variant_test_name}'),")
                else:
                    print(f"xfail('{opinfo.name}'),")

    def run_one_hot(self):
        ops = [op for op in op_db if op.name == "nn.functional.one_hot"]
        assert len(ops) == 1
        op = ops[0]
        # num_classes = -1 appears to have a bug with dtensor.max().item()
        self.run_opinfo_test(
            torch.int64,
            op,
            requires_grad=False,
            sample_inputs_filter=lambda s: s.kwargs["num_classes"] != -1,
        )

    def run_mean(self):
        self.mesh = init_device_mesh(DEVICE_TYPE, (self.world_size,))

        shape = [2 * self.world_size + 1, 2 * self.world_size]
        tensor = (
            torch.arange(shape[0] * shape[1], dtype=torch.float32)
            .reshape(shape)
            .to(DEVICE_TYPE)
        )

        for is_evenly_shardable in [True, False]:
            if is_evenly_shardable:
                placement = [Shard(1)]
                reduce_dim = 1
            else:
                placement = [Shard(0)]
                reduce_dim = 0
            dtensor = distribute_tensor(tensor, self.mesh, placement)

            with DebugMode(record_torchfunction=False) as debug_mode:
                mean = dtensor.mean(dim=reduce_dim)
                full_tensor = mean.full_tensor()

            self.assertEqual(full_tensor, tensor.mean(dim=reduce_dim))

            if is_evenly_shardable:
                self.assertTrue("P(avg)->R" in debug_mode.debug_string())
            else:
                self.assertTrue("S(0)->R" in debug_mode.debug_string())

    def test_embedding_error_msg(self):
        self.mesh_2d = init_device_mesh(
            DEVICE_TYPE, (2, self.world_size // 2), mesh_dim_names=("dp", "tp")
        )
        self.mesh_1d = self.mesh_2d["tp"]

        weight_global = torch.randn(2048, 256, device=DEVICE_TYPE)
        weight_dtensor = distribute_tensor(weight_global, self.mesh_1d, [Shard(0)])

        input_global = torch.randint(0, 2048, (16, 2048), device=DEVICE_TYPE)
        input_dtensor = distribute_tensor(
            input_global, self.mesh_2d, [Shard(0), Replicate()]
        )

        expected_error_msg = (
            "Sharding propagation failed for aten.embedding.default"
            "(Spec(f32[2048, 256](S(0))), Spec(i64[16, 2048](S(0)R))) "
            "on DeviceMesh((dp=2, tp=2), "
        )
        with self.assertRaisesRegex(RuntimeError, re.escape(expected_error_msg)):
            _ = torch.ops.aten.embedding.default(weight_dtensor, input_dtensor)


class TestMultiThreadedDTensorOps(DTensorOpTestBase, TestDTensorOps):
    _op_db = repurpose_ops(op_db, "TestDTensorOps", "TestMultiThreadedDTensorOps")

    @suppress_warnings
    @ops(_op_db, allowed_dtypes=(torch.float,))
    @skipOps(
        _op_db,
        "TestMultiThreadedDTensorOps",
        "test_dtensor_op_db",
        dtensor_fails | dtensor_multi_threaded_fails,
    )
    def test_dtensor_op_db(self, dtype, op):
        self.run_opinfo_test(dtype, op)

    def test_mean(self):
        self.run_mean()

    def test_one_hot(self):
        self.run_one_hot()


class TestLocalDTensorOps(TestDTensorOps):
    _op_db = repurpose_ops(op_db, "TestDTensorOps", "TestLocalDTensorOps")

    def setUp(self) -> None:
        super().setUp()
        torch.distributed.init_process_group("fake", rank=0, world_size=self.world_size)
        self.fake_pg = torch.distributed.distributed_c10d._get_default_group()

    def tearDown(self):
        super().tearDown()
        # Clear sharding propagation cache to avoid stale mesh references
        # between tests that destroy and recreate process groups
        from torch.distributed.tensor.debug import _clear_sharding_prop_cache

        _clear_sharding_prop_cache()
        try:
            dist.destroy_process_group()
        except AssertionError:
            pass

    @suppress_warnings
    @ops(_op_db, allowed_dtypes=(torch.float,))
    @skipOps(
        _op_db,
        "TestLocalDTensorOps",
        "test_dtensor_op_db",
        dtensor_fails,
    )
    def test_dtensor_op_db(self, dtype, op):
        self.run_opinfo_test(dtype, op)

    def test_mean(self):
        with LocalTensorMode(frozenset(range(self.world_size))):
            self.run_mean()

    def test_one_hot(self):
        self.run_one_hot()

    def run_opinfo_test(
        self, dtype, op, requires_grad=True, sample_inputs_filter=lambda s: True
    ):
        with LocalTensorMode(frozenset(range(self.world_size))):
            super().run_opinfo_test(dtype, op, requires_grad, sample_inputs_filter)

    def assertEqualOnRank(self, x, y, msg=None, *, rank=0):
        self.assertEqual(x, y, msg)


class TestSingleDimStrategies(DTensorOpTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def _extract_aten_op_and_args(self, torch_op, args, kwargs):
        with DebugMode(store_original_args=True) as debug_mode:
            try:
                torch_op(*args, **kwargs)
            except Exception:
                self.skipTest(f"Op {torch_op} failed on replicated DTensors")

        for op in debug_mode.operators:
            if isinstance(op, _OpCall) and "aten" in str(op.op):
                return op.op, op.args, op.kwargs

        self.skipTest(f"Op {torch_op} failed to extract aten op")

    @suppress_warnings
    @ops(op_db, allowed_dtypes=(torch.float,))
    def test_single_dim_strategy(self, dtype, op):
        torch.manual_seed(42)
        mesh = init_device_mesh(DEVICE_TYPE, (self.world_size,))
        sharding_prop = DTensor._op_dispatcher.sharding_propagator

        try:
            samples = list(op.sample_inputs(DEVICE_TYPE, dtype, requires_grad=False))
        except Exception:
            self.skipTest(f"Failed to get sample inputs for {op.name}")
        if not samples:
            self.skipTest(f"No sample inputs for {op.name}")

        sample = samples[0]
        args = (sample.input,) + tuple(sample.args)

        # create Replicated DTensors
        try:
            dtensor_args, dtensor_kwargs = pytree.tree_map_only(
                torch.Tensor,
                lambda t: distribute_tensor(t, mesh, (Replicate(),)),
                (args, sample.kwargs),
            )
        except Exception:
            self.skipTest(f"Failed to create replicate DTensors for {op.name}")

        # extract aten op/args/kwargs
        aten_op, aten_args, aten_kwargs = self._extract_aten_op_and_args(
            op.op, dtensor_args, dtensor_kwargs
        )

        single_dim_strats = sharding_prop.op_single_dim_strategy_funcs
        if aten_op not in single_dim_strats:
            self.skipTest(f"No single-dim strategy for {op.name}: {aten_op}")

        # extract tensor_meta, full tensors
        all_tensor_meta = []

        def _collect_tensor_meta(dt):
            meta = dt._spec.tensor_meta
            all_tensor_meta.append(meta)
            return meta

        args_meta, kwargs_meta = pytree.tree_map_only(
            DTensor, _collect_tensor_meta, (aten_args, aten_kwargs)
        )
        full_args, full_kwargs = pytree.tree_map_only(
            torch.Tensor, lambda t: t.full_tensor(), (aten_args, aten_kwargs)
        )

        # enumerate strategies, replace placeholders with Shard
        strategies = pytree.tree_map_only(
            _ShardingPlaceholder,
            lambda s: Shard(s.dim),
            single_dim_strats[aten_op](aten_op, args_meta, kwargs_meta),
        )
        # TODO(pianpwk): handle multi-output once that lands for single-dim
        for output_placement, *input_placements in strategies:
            # skip strategies with invalid shards
            def is_invalid_shard(meta, p):
                ndim = len(meta.shape)
                if (
                    not isinstance(p, Shard)
                    or ndim == 0
                    or p.dim >= ndim
                    or meta.shape[p.dim] == 0
                    or meta.shape[p.dim] % self.world_size != 0
                ):
                    return True
                return False

            if any(
                is_invalid_shard(t, p)
                for t, p in zip(all_tensor_meta, input_placements)
            ):
                continue

            # add the validate_sharding_rule function
            self.assertTrue(
                validate_sharding_rule_sample(
                    aten_op,
                    full_args,
                    full_kwargs,
                    input_placements,
                    (output_placement,),
                    mesh,
                ),
                f"{op.name}: {input_placements} -> {(output_placement,)} failed",
            )


# only instantiate tests for DEVICE_TYPE alone (i.e. either CPU or GPU)
instantiate_device_type_tests(
    TestMultiThreadedDTensorOps, globals(), only_for=(DEVICE_TYPE,)
)

instantiate_device_type_tests(TestLocalDTensorOps, globals(), only_for=(DEVICE_TYPE,))

instantiate_device_type_tests(
    TestSingleDimStrategies, globals(), only_for=(DEVICE_TYPE,)
)

if __name__ == "__main__":
    run_tests()
