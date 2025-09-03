# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import unittest
import warnings

import torch
import torch.distributed as dist
import torch.testing._internal.common_methods_invocations as common_ops
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.overrides import resolve_name
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
)
from torch.testing._internal.common_methods_invocations import DecorateInfo, op_db
from torch.testing._internal.common_utils import run_tests, suppress_warnings
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorConverter,
    DTensorOpTestBase,
)
from torch.utils import _pytree as pytree
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


def skipOps(test_case_name, base_test_name, to_skip):
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


# Re-generate this failed list, turn on dry_run of the below func
# check_dtensor_func(self, test, op, dry_run=True), then run sth
# like python test/distributed/tensor/test_dtensor_ops.py > failed.expect
dtensor_fails = {
    # these sometimes pass and sometimes fail
    # we need to remove many of them from list once op
    # get full support with varying sharding specs
    xfail('__getitem__'),
    xfail('__rsub__'),
    xfail('_batch_norm_with_update'),
    xfail('_chunk_cat'),
    xfail('_native_batch_norm_legit'),
    xfail('_softmax_backward_data'),
    xfail('_unsafe_masked_index'),
    xfail('_unsafe_masked_index_put_accumulate'),
    xfail('_upsample_bilinear2d_aa'),
    xfail('addbmm'),
    xfail('addmv'),
    xfail('addr'),
    xfail('alias_copy'),
    xfail('all'),
    xfail('allclose'),
    xfail('aminmax'),
    xfail('any'),
    xfail('arange'),
    xfail('argmax'),
    xfail('argmin'),
    xfail('argsort'),
    xfail('as_strided_copy'),
    xfail('as_strided'),
    xfail('as_strided', 'partial_views'),
    xfail('as_strided_scatter'),
    xfail('bernoulli'),
    xfail('block_diag'),
    xfail('broadcast_shapes'),
    xfail('cartesian_prod'),
    xfail('cauchy'),
    xfail('cdist'),
    xfail('cholesky'),
    xfail('cholesky_inverse'),
    xfail('cholesky_solve'),
    xfail('chunk'),
    xfail('clamp'),
    xfail('clamp_max'),
    xfail('clamp_min'),
    xfail('combinations'),
    xfail('complex'),
    xfail('constant_pad_nd'),
    xfail('count_nonzero'),
    xfail('cross'),
    xfail('cummax'),
    xfail('cummin'),
    xfail('diagonal_scatter'),
    xfail('dist'),
    xfail('empty'),
    xfail('empty_like'),
    xfail('empty_permuted'),
    xfail('empty_strided'),
    xfail('equal'),
    xfail('expand_copy'),
    xfail('exponential'),
    xfail('eye'),
    xfail('fft.fft2'),
    xfail('fft.fft'),
    xfail('fft.fftn'),
    xfail('fft.fftshift'),
    xfail('fft.ifft2'),
    xfail('fft.ifft'),
    xfail('fft.ifftshift'),
    xfail('fft.ihfft2'),
    xfail('fft.ihfft'),
    xfail('fft.ihfftn'),
    xfail('fft.irfft2'),
    xfail('fft.irfftn'),
    xfail('fft.rfft2'),
    xfail('fft.rfft'),
    xfail('fft.rfftn'),
    xfail('fill'),
    xfail('flatten'),
    xfail('flip'),
    xfail('fliplr'),
    xfail('flipud'),
    xfail('floor_divide'),
    xfail('fmax'),
    xfail('fmin'),
    xfail('frexp'),
    xfail('full'),
    xfail('full_like'),
    xfail('gather'),
    xfail('geometric'),
    xfail('geqrf'),
    xfail('gradient'),
    xfail('grid_sampler_2d'),
    xfail('heaviside'),
    xfail('histc'),
    xfail('index_add'),
    xfail('index_copy'),
    xfail('index_fill'),
    xfail('index_put'),
    xfail('index_reduce', 'amax'),
    xfail('index_reduce', 'amin'),
    xfail('index_reduce', 'mean'),
    xfail('index_reduce', 'prod'),
    xfail('index_select'),
    xfail('isin'),
    xfail('jiterator_2inputs_2outputs'),
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
DEVICE_TYPE = (
    "cuda"
    if torch.cuda.is_available() and torch.cuda.device_count() >= OP_DB_WORLD_SIZE
    else "cpu"
)


class TestDTensorOps(DTensorOpTestBase):
    @property
    def world_size(self) -> int:
        return OP_DB_WORLD_SIZE

    # only allow float dytpe for now, we can relax this constraint
    # when feel necessary later (i.e when adding quantization support).
    @suppress_warnings
    @ops(op_db, allowed_dtypes=(torch.float,))
    @skipOps("TestDTensorOps", "test_dtensor_op_db", dtensor_fails)
    def test_dtensor_op_db(self, dtype, op):
        self.mesh = DeviceMesh(DEVICE_TYPE, torch.arange(self.world_size))

        # test each op with dist tensor inputs and normal inputs
        def test():
            samples = op.sample_inputs(DEVICE_TYPE, dtype, requires_grad=True)
            for sample_input in samples:
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
        rs = func(*args, **kwargs)
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
                            f"failed to convert args to DTensor; "
                            f"originally (*{args}, **{kwargs})"
                        )
                except Exception as e:
                    raise RuntimeError(
                        f"failed to run: {resolve_name(func)}, with (*{dtensor_args}, **{dtensor_kwargs})"
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


# only instantiate tests for DEVICE_TYPE alone (i.e. either CPU or GPU)
instantiate_device_type_tests(TestDTensorOps, globals(), only_for=(DEVICE_TYPE,))


if __name__ == "__main__":
    run_tests()
