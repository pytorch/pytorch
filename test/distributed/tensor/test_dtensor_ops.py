# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import copy
import re
import unittest
import warnings

import torch
import torch._dynamo
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
from torch.testing._internal.common_ops_unbacked import ops_dde_xfail, ops_unbacked_skip
from torch.testing._internal.common_utils import run_tests, suppress_warnings, TestCase
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorConverter,
    DTensorOpTestBase,
    validate_sharding_rule_sample,
)
from torch.utils import _pytree as pytree
from torch.utils._debug_mode import _OpCall, DebugMode
from torch.utils._pytree import tree_flatten, tree_map


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
        if not (len(matching_opinfos) >= 1):
            raise AssertionError(f"Couldn't find OpInfo for {xfail}")
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
    # view/reshape ops: rejects flatten/split of sharded dims without redistribution
    xfail("cartesian_prod"),
    xfail("flatten"),
    xfail("kron"),
    xfail("ravel"),
    xfail("repeat_interleave"),
    xfail("reshape"),
    xfail("reshape_as"),
    xfail("take_along_dim"),
    xfail("unbind"),
    xfail("unflatten"),
    xfail("view"),
    xfail("view_as"),
    # factory/creation ops: test harness can't convert non-tensor args to DTensor
    xfail("arange"),
    xfail("broadcast_shapes"),
    xfail("eye"),
    xfail("full"),
    xfail("linspace"),
    xfail("logspace"),
    xfail("ones"),
    xfail("scalar_tensor"),
    xfail("signal.windows.bartlett"),
    xfail("signal.windows.blackman"),
    xfail("signal.windows.cosine"),
    xfail("signal.windows.exponential"),
    xfail("signal.windows.gaussian"),
    xfail("signal.windows.general_cosine"),
    xfail("signal.windows.general_hamming"),
    xfail("signal.windows.hamming"),
    xfail("signal.windows.hann"),
    xfail("signal.windows.kaiser"),
    xfail("signal.windows.nuttall"),
    xfail("zeros"),
    # random/stochastic ops: different RNG states between DTensor and reference
    xfail("bernoulli"),
    xfail("cauchy"),
    xfail("nn.functional.alpha_dropout"),
    xfail("nn.functional.dropout"),
    xfail("normal"),
    xfail("normal", "in_place"),
    xfail("normal", "number_mean"),
    xfail("rand_like"),
    xfail("randint"),
    xfail("randint_like"),
    xfail("randn"),
    xfail("randn_like"),
    xfail("uniform"),
    # mixed Tensor/DTensor inputs: op creates plain Tensors mixed with DTensor args
    xfail("__getitem__"),
    xfail("nn.functional.fractional_max_pool2d"),
    xfail("nn.functional.fractional_max_pool3d"),
    xfail("pca_lowrank"),
    xfail("quantile"),
    xfail("svd_lowrank"),
    # dynamic output shape: output shape depends on data values
    xfail("combinations"),
    xfail("linalg.lstsq"),
    xfail("linalg.lstsq", "grad_oriented"),
    xfail("masked_select"),
    xfail("nn.functional.ctc_loss"),
    # 0-dim tensor edge cases: strategies don't handle scalar tensors
    xfail("logsumexp"),
    xfail("masked.logsumexp"),
    xfail("transpose"),
    # conv stride+padding: TP convolution rejects stride != 1 with padding
    xfail("nn.functional.conv1d"),
    xfail("nn.functional.conv2d"),
    xfail("nn.functional.conv3d"),
    xfail("nn.functional.conv_transpose1d"),
    xfail("nn.functional.conv_transpose2d"),
    xfail("nn.functional.conv_transpose3d"),
    # in-place op requires placement change during decomposition
    xfail("nn.functional.cosine_similarity"),
    # "cannot resize variables that require grad" from test harness
    xfail("resize_"),
    xfail("resize_as_"),
    # DTensorConverter can't convert sparse tensor inputs
    xfail("sparse.sampled_addmm"),
    xfail("sparse.mm", "reduce"),
    # bug in squeeze.dims strategy: TypeError with empty dims arg
    xfail("squeeze", "multiple"),
    # meta tensor data not allocated yet during tensor_split
    xfail("tensor_split"),
    # output_specs count mismatch in unsafe_split strategy
    xfail("unsafe_split"),
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
    xfail("masked.cumprod"),
    xfail("nn.functional.dropout2d"),
    xfail("nn.functional.dropout3d"),
    xfail("nn.functional.huber_loss"),
    skip("nn.functional.multi_head_attention_forward"),
}

# Ops that fail to compile with DTensor + torch.compile(fullgraph=True).
# These are compile-time failures, NOT numeric correctness issues.
dtensor_compiled_fails = {
    # View-type ops that decompose into as_strided (at autograd level).
    # DTensor doesn't have a sharding strategy for as_strided.
    xfail("atleast_1d"),
    xfail("atleast_2d"),
    xfail("atleast_3d"),
    xfail("broadcast_tensors"),
    xfail("broadcast_to"),
    xfail("diagonal"),
    xfail("dsplit"),
    xfail("expand"),
    xfail("expand_as"),
    xfail("hsplit"),
    xfail("linalg.diagonal"),
    xfail("max", "reduction_with_dim"),
    xfail("min", "reduction_with_dim"),
    xfail("movedim"),
    xfail("narrow"),
    xfail("permute"),
    xfail("select"),
    xfail("slice"),
    xfail("t"),
    xfail("transpose_copy"),
    xfail("unsqueeze"),
    xfail("vsplit"),
    # Decompositions that use plain tensor constructors (e.g. arange),
    # causing mixed tensor/DTensor errors during Dynamo's fake prop.
    xfail("corrcoef"),
    xfail("cov"),
    xfail("nn.functional.interpolate", "bicubic"),
    xfail("nn.functional.interpolate", "bilinear"),
    xfail("nn.functional.interpolate", "linear"),
    xfail("nn.functional.interpolate", "trilinear"),
    xfail("nn.functional.upsample_bilinear"),
    # Data-dependent outputs (SymBool, unbacked shapes) that raise
    # during DTensor's fake prop.
    xfail("equal"),
    xfail("hash_tensor"),
    xfail("item"),
    xfail("nonzero_static"),
    # Decompositions with .is_cuda checks that fail during sharding
    # propagation for aten.is_cuda / prim::device.
    xfail("nn.functional.binary_cross_entropy"),
    xfail("nn.functional.binary_cross_entropy_with_logits"),
    xfail("nn.functional.gaussian_nll_loss"),
    xfail("nn.functional.logsigmoid"),
    # Miscellaneous runtime crashes (e.g. index out of bounds).
    xfail("gather"),
    xfail("index_select"),
    xfail("scatter"),
    xfail("scatter_add"),
    # False positives: these have no sharding strategy and their
    # eager DTensor failure is registered elsewhere.
    xfail("nn.functional.margin_ranking_loss"),
    xfail("nn.functional.multilabel_soft_margin_loss"),
}

# Ops that compile successfully but fail numeric checks in eager DTensor tests.
# These are excluded from TestCompiledDTensorOps skip list since we don't check numerics.
dtensor_numeric_only_fails = {
    xfail("arange"),
    xfail("broadcast_shapes"),
    xfail("eye"),
    xfail("full"),
    xfail("full_like"),
    xfail("linspace"),
    xfail("logspace"),
    xfail("nn.functional.hardshrink"),
    xfail("nn.functional.huber_loss"),
    xfail("nn.functional.smooth_l1_loss"),
    xfail("nn.functional.softshrink"),
    xfail("ones"),
    xfail("randint"),
    xfail("randn"),
    xfail("scalar_tensor"),
    xfail("signal.windows.bartlett"),
    xfail("signal.windows.blackman"),
    xfail("signal.windows.cosine"),
    xfail("signal.windows.exponential"),
    xfail("signal.windows.gaussian"),
    xfail("signal.windows.general_cosine"),
    xfail("signal.windows.general_hamming"),
    xfail("signal.windows.hamming"),
    xfail("signal.windows.hann"),
    xfail("signal.windows.kaiser"),
    xfail("signal.windows.nuttall"),
    xfail("sparse.mm", "reduce"),
    xfail("sparse.sampled_addmm"),
    xfail("squeeze_copy"),
    xfail("stack"),
    xfail("unsafe_chunk"),
    xfail("zeros"),
}

# Ops in dtensor_fails that have no sharding strategy (NotImplementedError).
# These will error during sharding propagation and affect unbacked tests too.
dtensor_fails_no_strategy = {
    xfail("_batch_norm_with_update"),
    xfail("_chunk_cat"),
    xfail("_native_batch_norm_legit"),
    xfail("_unsafe_masked_index"),
    xfail("_unsafe_masked_index_put_accumulate"),
    xfail("_upsample_bilinear2d_aa"),
    xfail("addbmm"),
    xfail("allclose"),
    xfail("as_strided"),
    xfail("as_strided", "partial_views"),
    xfail("as_strided_copy"),
    xfail("as_strided_scatter"),
    xfail("block_diag"),
    xfail("cdist"),
    xfail("cholesky"),
    xfail("cholesky_inverse"),
    xfail("cholesky_solve"),
    xfail("complex"),
    xfail("cross"),
    xfail("cummax"),
    xfail("cummin"),
    xfail("diagonal_scatter"),
    xfail("exponential"),
    xfail("fft.ihfft2"),
    xfail("fft.ihfftn"),
    xfail("geometric"),
    xfail("geqrf"),
    xfail("grid_sampler_2d"),
    xfail("histogram"),
    xfail("histogramdd"),
    xfail("index_add"),
    xfail("index_copy"),
    xfail("index_fill"),
    xfail("index_reduce", "prod"),
    xfail("index_reduce", "mean"),
    xfail("index_reduce", "amax"),
    xfail("index_reduce", "amin"),
    xfail("isin"),
    xfail("kthvalue"),
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
    xfail("linspace", "tensor_overload"),
    xfail("log_normal"),
    xfail("logcumsumexp"),
    xfail("logdet"),
    xfail("logspace", "tensor_overload"),
    xfail("lu"),
    xfail("lu_solve"),
    xfail("lu_unpack"),
    xfail("masked.median"),
    xfail("masked_scatter"),
    xfail("matrix_exp"),
    xfail("max_pool2d_with_indices_backward"),
    xfail("median"),
    xfail("mode"),
    xfail("multinomial"),
    xfail("nanmean"),
    xfail("nanmedian"),
    xfail("nanquantile"),
    xfail("nansum"),
    xfail("native_batch_norm"),
    xfail("nn.functional.adaptive_avg_pool1d"),
    xfail("nn.functional.adaptive_avg_pool2d"),
    xfail("nn.functional.adaptive_avg_pool3d"),
    xfail("nn.functional.adaptive_max_pool1d"),
    xfail("nn.functional.adaptive_max_pool2d"),
    xfail("nn.functional.adaptive_max_pool3d"),
    xfail("nn.functional.avg_pool1d"),
    xfail("nn.functional.avg_pool2d"),
    xfail("nn.functional.avg_pool3d"),
    xfail("nn.functional.batch_norm"),
    xfail("nn.functional.bilinear"),
    xfail("nn.functional.grid_sample"),
    xfail("nn.functional.group_norm"),
    xfail("nn.functional.hardshrink"),
    xfail("nn.functional.instance_norm"),
    xfail("nn.functional.interpolate", "area"),
    xfail("nn.functional.interpolate", "nearest"),
    xfail("nn.functional.interpolate", "nearest-exact"),
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
    xfail("nn.functional.multi_margin_loss"),
    xfail("nn.functional.multilabel_margin_loss"),
    xfail("nn.functional.pad", "reflect"),
    xfail("nn.functional.pad", "replicate"),
    xfail("nn.functional.pad", "replicate_negative"),
    xfail("nn.functional.pdist"),
    xfail("nn.functional.rrelu"),
    xfail("nn.functional.threshold"),
    xfail("nn.functional.unfold"),
    xfail("nn.functional.upsample_nearest"),
    xfail("nonzero"),
    xfail("ormqr"),
    xfail("pinverse"),
    xfail("polar"),
    xfail("put"),
    xfail("renorm"),
    xfail("scatter_reduce", "amax"),
    xfail("scatter_reduce", "amin"),
    xfail("scatter_reduce", "mean"),
    xfail("scatter_reduce", "prod"),
    xfail("scatter_reduce", "sum"),
    xfail("searchsorted"),
    xfail("select_scatter"),
    xfail("special.airy_ai"),
    xfail("special.bessel_y0"),
    xfail("special.bessel_y1"),
    xfail("special.chebyshev_polynomial_t"),
    xfail("special.chebyshev_polynomial_u"),
    xfail("special.chebyshev_polynomial_v"),
    xfail("special.chebyshev_polynomial_w"),
    xfail("special.entr"),
    xfail("special.hermite_polynomial_h"),
    xfail("special.hermite_polynomial_he"),
    xfail("special.laguerre_polynomial_l"),
    xfail("special.legendre_polynomial_p"),
    xfail("special.modified_bessel_i0"),
    xfail("special.modified_bessel_i1"),
    xfail("special.modified_bessel_k0"),
    xfail("special.modified_bessel_k1"),
    xfail("special.scaled_modified_bessel_k0"),
    xfail("special.scaled_modified_bessel_k1"),
    xfail("special.shifted_chebyshev_polynomial_t"),
    xfail("special.shifted_chebyshev_polynomial_u"),
    xfail("special.shifted_chebyshev_polynomial_v"),
    xfail("special.shifted_chebyshev_polynomial_w"),
    xfail("special.xlog1py"),
    xfail("squeeze_copy"),
    xfail("stft"),
    xfail("take"),
    xfail("to_sparse"),
    xfail("triangular_solve"),
    xfail("unfold"),
    xfail("unfold_copy"),
    xfail("unique"),
    xfail("unique_consecutive"),
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

    def iter_valid_samples(
        self,
        op,
        dtype,
        requires_grad=False,
        sample_filter=None,
        needs_deepcopy=False,
    ):
        """
        Iterate over valid samples for an op, yielding (args, kwargs) tuples.

        Args:
            op: The OpInfo object
            dtype: The dtype to use for sample inputs
            requires_grad: Whether tensors should require grad
            sample_filter: Optional callable(args, kwargs) -> bool to filter samples
            needs_deepcopy: If True, yields deepcopied args/kwargs and skips
                            samples that can't be deepcopied
        """
        samples = op.sample_inputs(DEVICE_TYPE, dtype, requires_grad=requires_grad)
        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs

            if sample_filter and not sample_filter(args, kwargs):
                continue

            if needs_deepcopy:
                try:
                    args = copy.deepcopy(args)
                    kwargs = copy.deepcopy(kwargs)
                except NotImplementedError:
                    continue

            yield args, kwargs

    def run_opinfo_test(self, dtype, op, requires_grad=True, sample_filter=None):
        self.mesh = init_device_mesh(DEVICE_TYPE, (self.world_size,))

        def test():
            for args, kwargs in self.iter_valid_samples(
                op, dtype, requires_grad=requires_grad, sample_filter=sample_filter
            ):
                self.run_dtensor_crossref(op.op, args, kwargs)

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
        if len(ops) != 1:
            raise AssertionError(f"Expected 1 op, got {len(ops)}")
        op = ops[0]
        # num_classes = -1 appears to have a bug with dtensor.max().item()
        self.run_opinfo_test(
            torch.int64,
            op,
            requires_grad=False,
            sample_filter=lambda args, kwargs: kwargs.get("num_classes") != -1,
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
        dtensor_fails | dtensor_multi_threaded_fails | dtensor_fails_no_strategy,
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
        dtensor_fails | dtensor_fails_no_strategy,
    )
    def test_dtensor_op_db(self, dtype, op):
        self.run_opinfo_test(dtype, op)

    def test_mean(self):
        with LocalTensorMode(frozenset(range(self.world_size))):
            self.run_mean()

    def test_one_hot(self):
        self.run_one_hot()

    def run_opinfo_test(self, dtype, op, requires_grad=True, sample_filter=None):
        with LocalTensorMode(frozenset(range(self.world_size))):
            super().run_opinfo_test(dtype, op, requires_grad, sample_filter)

    def assertEqualOnRank(self, x, y, msg=None, *, rank=0):
        self.assertEqual(x, y, msg)


# Ops where DTensor shard prop has DDEs with unbacked (base tensor passes).
# This list only contains ops NOT in ops_dde_xfail - those are base tensor issues.
ops_unbacked_dtensor_dde = {
    xfail("__getitem__"),
    xfail("__radd__"),
    xfail("__rdiv__"),
    xfail("__rmatmul__"),
    xfail("__rmod__"),
    xfail("__rmul__"),
    xfail("__rpow__"),
    xfail("__rsub__"),
    xfail("_segment_reduce", "lengths"),
    xfail("_segment_reduce", "offsets"),
    xfail("_unsafe_masked_index"),
    xfail("add"),
    xfail("addcdiv"),
    xfail("addcmul"),
    xfail("addmm"),
    xfail("addmm", "decomposed"),
    xfail("addr"),
    xfail("alias_copy"),
    xfail("aminmax"),
    xfail("argsort"),
    xfail("argwhere"),
    xfail("as_strided"),
    xfail("as_strided", "partial_views"),
    xfail("as_strided_copy"),
    xfail("atan2"),
    xfail("block_diag"),
    xfail("broadcast_tensors"),
    skip("broadcast_to"),
    xfail("bucketize"),
    xfail("cartesian_prod"),
    xfail("cholesky_solve"),
    xfail("clamp"),
    xfail("clamp_max"),
    xfail("clamp_min"),
    xfail("column_stack"),
    xfail("copysign"),
    xfail("cumprod"),
    xfail("diagflat"),
    xfail("dist"),
    xfail("div", "floor_rounding"),
    xfail("div", "no_rounding_mode"),
    xfail("div", "trunc_rounding"),
    xfail("einsum"),
    xfail("eq"),
    xfail("expand"),
    xfail("expand_as"),
    xfail("expand_copy"),
    xfail("fill"),
    xfail("flatten"),
    xfail("flip"),
    xfail("fliplr"),
    xfail("flipud"),
    xfail("float_power"),
    xfail("floor_divide"),
    xfail("fmax"),
    xfail("fmin"),
    xfail("fmod"),
    xfail("frexp"),
    xfail("gather"),
    xfail("ge"),
    xfail("gt"),
    xfail("heaviside"),
    xfail("histc"),
    xfail("hypot"),
    xfail("igamma"),
    xfail("igammac"),
    xfail("index_put"),
    xfail("index_select"),
    xfail("isclose"),
    xfail("ldexp"),
    xfail("le"),
    xfail("lerp"),
    xfail("logaddexp"),
    xfail("logical_and"),
    xfail("logical_or"),
    xfail("logical_xor"),
    xfail("lt"),
    xfail("masked.normalize"),
    xfail("masked_fill"),
    xfail("masked_scatter"),
    xfail("masked_select"),
    xfail("matmul"),
    xfail("max", "binary"),
    xfail("max", "reduction_with_dim"),
    xfail("maximum"),
    xfail("meshgrid", "list_of_tensors"),
    xfail("meshgrid", "variadic_tensors"),
    xfail("min", "binary"),
    xfail("min", "reduction_with_dim"),
    xfail("minimum"),
    xfail("msort"),
    xfail("mul"),
    xfail("mv"),
    xfail("narrow"),
    xfail("narrow_copy"),
    xfail("ne"),
    xfail("new_empty"),
    xfail("new_empty_strided"),
    xfail("new_full"),
    xfail("new_ones"),
    xfail("new_zeros"),
    xfail("nextafter"),
    xfail("nn.functional.celu"),
    xfail("nn.functional.conv1d"),
    xfail("nn.functional.conv2d"),
    xfail("nn.functional.conv3d"),
    xfail("nn.functional.conv_transpose1d"),
    xfail("nn.functional.conv_transpose2d"),
    xfail("nn.functional.conv_transpose3d"),
    xfail("nn.functional.cosine_embedding_loss"),
    xfail("nn.functional.elu"),
    xfail("nn.functional.hardsigmoid"),
    xfail("nn.functional.hardtanh"),
    xfail("nn.functional.hinge_embedding_loss"),
    xfail("nn.functional.interpolate", "nearest"),
    xfail("nn.functional.interpolate", "nearest-exact"),
    xfail("nn.functional.linear"),
    xfail("nn.functional.logsigmoid"),
    xfail("nn.functional.margin_ranking_loss"),
    xfail("nn.functional.mish"),
    xfail("nn.functional.multilabel_soft_margin_loss"),
    xfail("nn.functional.normalize"),
    xfail("nn.functional.pixel_unshuffle"),
    xfail("nn.functional.poisson_nll_loss"),
    xfail("nn.functional.relu6"),
    xfail("nn.functional.selu"),
    xfail("nn.functional.softplus"),
    xfail("nn.functional.soft_margin_loss"),
    xfail("nn.functional.triplet_margin_loss"),
    xfail("nn.functional.triplet_margin_with_distance_loss"),
    xfail("nonzero_static"),
    xfail("outer"),
    xfail("permute_copy"),
    xfail("pow"),
    xfail("prod"),
    xfail("ravel"),
    xfail("remainder"),
    xfail("reshape"),
    xfail("reshape_as"),
    xfail("rsub"),
    xfail("rot90"),
    xfail("scatter"),
    xfail("scatter_add"),
    xfail("slice"),
    xfail("sort"),
    xfail("special.bessel_j0"),
    xfail("special.bessel_j1"),
    xfail("special.log_ndtr"),
    xfail("special.ndtri"),
    xfail("special.spherical_bessel_j0"),
    xfail("special.zeta"),
    xfail("squeeze"),
    xfail("squeeze", "multiple"),
    xfail("std_mean"),
    xfail("sub"),
    xfail("topk"),
    xfail("transpose_copy"),
    xfail("true_divide"),
    xfail("unflatten"),
    xfail("unsqueeze_copy"),
    xfail("vdot"),
    xfail("view"),
    xfail("view_as"),
    xfail("view_as_complex"),
    xfail("view_copy"),
    xfail("where"),
    xfail("xlogy"),
}


class TestUnbackedDTensorOps(TestDTensorOps):
    """
    Test suite for DTensor ops with unbacked symints.

    This runs correctness tests with tensor dimensions marked as unbacked
    and the op compiled with fullgraph=True to catch DDEs during tracing.
    """

    _op_db = repurpose_ops(op_db, "TestDTensorOps", "TestUnbackedDTensorOps")

    def setUp(self) -> None:
        super().setUp()
        torch.distributed.init_process_group("fake", rank=0, world_size=self.world_size)

    def tearDown(self):
        super().tearDown()
        from torch.distributed.tensor.debug import _clear_sharding_prop_cache

        _clear_sharding_prop_cache()
        torch._dynamo.reset()
        try:
            dist.destroy_process_group()
        except AssertionError:
            pass

    def assertEqualOnRank(self, x, y, msg=None, *, rank=0):
        self.assertEqual(x, y, msg)

    def _has_valid_unbacked_dims(self, t: torch.Tensor) -> bool:
        """Check if tensor has dimensions that can be marked as unbacked."""
        return t.ndim > 0 and any(s >= 2 for s in t.shape)

    def _sample_has_valid_unbacked_dims(self, args, kwargs) -> bool:
        """Check if any tensor in args/kwargs has valid unbacked dimensions."""
        all_tensors = [
            x for x in tree_flatten((args, kwargs))[0] if isinstance(x, torch.Tensor)
        ]
        return any(self._has_valid_unbacked_dims(t) for t in all_tensors)

    def _mark_unbacked(self, t: torch.Tensor) -> None:
        """Mark all eligible dimensions of a tensor as unbacked."""
        for i in range(t.ndim):
            if t.shape[i] >= 2:
                torch._dynamo.decorators.mark_unbacked(t, i)

    def run_dtensor_crossref(self, func, args, kwargs):
        """
        Override to add unbacked marking and fullgraph compilation.

        Same as parent but:
        1. Marks DTensor dimensions as unbacked before running
        2. Wraps the op in @torch.compile(backend="eager", fullgraph=True)
        """
        to_dtensor = DTensorConverter(self.mesh, args, kwargs)

        def concat_res_if_necessary(func, res: object) -> object:
            if (resolve_name(func) is not None) and ("split" in resolve_name(func)):
                dim = args[2] if len(args) == 3 else 0
                return torch.cat(res, dim=dim)
            return res

        op_args, op_kwargs = reconcile_args(args, kwargs)
        rs = func(*op_args, **op_kwargs)
        rs = concat_res_if_necessary(func, rs)

        def to_replicate(e: object) -> object:
            return e.full_tensor() if isinstance(e, DTensor) else e

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for dtensor_args, dtensor_kwargs in to_dtensor:
                try:
                    if not to_dtensor.successful():
                        raise RuntimeError(
                            f"Failed to convert args to DTensor; "
                            f"originally (*{args}, **{kwargs})"
                        )

                    pytree.tree_map_only(
                        lambda x: isinstance(x, DTensor)
                        and self._has_valid_unbacked_dims(x),
                        self._mark_unbacked,
                        (dtensor_args, dtensor_kwargs),
                    )

                    # Compile with fullgraph=True to catch DDEs
                    torch._dynamo.reset()

                    @torch.compile(backend="eager", fullgraph=True)
                    def compiled_func(*a, **kw):
                        return func(*a, **kw)

                    compiled_func(*dtensor_args, **dtensor_kwargs)

                except Exception as e:
                    raise RuntimeError(
                        f"{str(e)}\n\nFailed to run: {resolve_name(func)}, "
                        f"with (*{dtensor_args}, **{dtensor_kwargs})"
                    ) from e
        return rs

    @suppress_warnings
    @ops(_op_db, allowed_dtypes=(torch.float,))
    @skipOps(
        _op_db,
        "TestUnbackedDTensorOps",
        "test_unbacked_dtensor_op_db",
        ops_dde_xfail
        | ops_unbacked_dtensor_dde
        | dtensor_fails_no_strategy
        | ops_unbacked_skip,
    )
    def test_unbacked_dtensor_op_db(self, dtype, op):
        # Filter to samples with valid unbacked dimensions
        self.run_opinfo_test(
            dtype,
            op,
            requires_grad=False,
            sample_filter=self._sample_has_valid_unbacked_dims,
        )


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


class TestCompiledDTensorOps(TestDTensorOps):
    """
    Test DTensor ops compile successfully with aot_eager backend.
    Uses fake PG for speed - focuses on compilation, not output correctness.
    """

    _op_db = repurpose_ops(op_db, "TestDTensorOps", "TestCompiledDTensorOps")

    def setUp(self) -> None:
        super().setUp()
        torch.distributed.init_process_group("fake", rank=0, world_size=self.world_size)

    def tearDown(self):
        super().tearDown()
        from torch.distributed.tensor.debug import _clear_sharding_prop_cache

        _clear_sharding_prop_cache()
        torch._dynamo.reset()
        try:
            dist.destroy_process_group()
        except AssertionError:
            pass

    def assertEqualOnRank(self, x, y, msg=None, *, rank=0):
        # Skip output comparison - we only care that compilation succeeds
        pass

    def run_dtensor_crossref(self, func, args, kwargs):
        """
        Override to compile with aot_eager and verify compilation succeeds.
        Does not check output correctness.
        """
        to_dtensor = DTensorConverter(self.mesh, args, kwargs)

        for dtensor_args, dtensor_kwargs in to_dtensor:
            if not to_dtensor.successful():
                continue

            torch._dynamo.reset()

            @torch.compile(backend="aot_eager", fullgraph=True)
            def compiled_func(*a, **kw):
                return func(*a, **kw)

            # Just run - if it compiles and runs without error, we pass
            compiled_func(*dtensor_args, **dtensor_kwargs)

    @suppress_warnings
    @ops(_op_db, allowed_dtypes=(torch.float,))
    @skipOps(
        _op_db,
        "TestCompiledDTensorOps",
        "test_compiled_dtensor_op_db",
        (
            dtensor_fails
            | dtensor_fails_no_strategy
            | dtensor_multi_threaded_fails
            | dtensor_compiled_fails
        )
        - dtensor_numeric_only_fails,
    )
    def test_compiled_dtensor_op_db(self, dtype, op):
        self.run_opinfo_test(dtype, op, requires_grad=False)


# only instantiate tests for DEVICE_TYPE alone (i.e. either CPU or GPU)
instantiate_device_type_tests(
    TestMultiThreadedDTensorOps, globals(), only_for=(DEVICE_TYPE,)
)

instantiate_device_type_tests(TestLocalDTensorOps, globals(), only_for=(DEVICE_TYPE,))

instantiate_device_type_tests(
    TestUnbackedDTensorOps, globals(), only_for=(DEVICE_TYPE,)
)

instantiate_device_type_tests(
    TestSingleDimStrategies, globals(), only_for=(DEVICE_TYPE,)
)

instantiate_device_type_tests(
    TestCompiledDTensorOps, globals(), only_for=(DEVICE_TYPE,)
)

if __name__ == "__main__":
    run_tests()
