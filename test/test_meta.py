# Owner(s): ["module: primTorch"]

import itertools
import torch
import os
from enum import Enum
from torch.overrides import resolve_name
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten
from torch._subclasses.meta_utils import MetaConverter, assert_metadata_eq
import torch.utils._python_dispatch
from torch._dispatch.python import enable_python_dispatcher
from torch.testing._internal.common_utils import (
    TestCase,
    skipIfCrossRef,
    suppress_warnings,
    TEST_WITH_ASAN,
    run_tests,
    dtype_abbrs
)
from torch.testing._internal.common_device_type import (
    ops,
    instantiate_device_type_tests,
    onlyCUDA,
    OpDTypes,
)
from torch.testing._internal.common_methods_invocations import op_db
from torchgen.utils import YamlLoader
from torchgen.model import OperatorName

import sys
import yaml
import atexit
import re
from collections import defaultdict
import unittest
import warnings
import weakref
from functools import wraps

bf16 = torch.bfloat16
f64 = torch.float64
f32 = torch.float32
f16 = torch.float16
c32 = torch.complex32
c64 = torch.complex64
c128 = torch.complex128
i8 = torch.int8
i16 = torch.int16
i32 = torch.int32
i64 = torch.int64
b8 = torch.bool
u8 = torch.uint8


class TestMetaConverter(TestCase):
    def assertSameVersionCounter(self, m1, m2):
        # Cannot easily test m1 and m2 have same storage due to
        # lack of Storage bindings.  Use version counter.
        vc = m1._version
        self.assertEqual(m2._version, vc)
        # Doing it this way ensures that we get VC bump even with leaves
        with torch.no_grad():
            m1._base.add_(3)
        self.assertNotEqual(m1._version, vc)
        self.assertEqual(m2._version, m1._version)

    def assertMetadataMatches(self, m1, m2):
        assert_metadata_eq(self.assertEqual, m1, m2)

    def test_view_of_non_leaf(self):
        x = torch.randn(4, requires_grad=True)
        y = x.neg()
        z1 = y[:]
        z2 = y[:]
        to_meta = MetaConverter()
        m1 = to_meta(z1)
        m2 = to_meta(z2)

        # check the test is actually testing what it claims
        self.assertTrue(m1._is_view())
        self.assertFalse(m1._base.is_leaf)

        self.assertIsNot(m1, m2)
        self.assertMetadataMatches(m1, z1)
        self.assertMetadataMatches(m2, z2)
        self.assertSameVersionCounter(m1, m2)

    def test_view_of_leaf(self):
        x = torch.randn(4, requires_grad=True)
        z1 = x[:]
        z2 = x[:]
        to_meta = MetaConverter()
        m1 = to_meta(z1)
        m2 = to_meta(z2)

        # check the test is actually testing what it claims
        self.assertTrue(m1._is_view())
        self.assertTrue(m1._base.is_leaf)

        self.assertIsNot(m1, m2)
        self.assertMetadataMatches(m1, z1)
        self.assertMetadataMatches(m2, z2)
        self.assertSameVersionCounter(m1, m2)

    def test_view_of_view_of_leaf(self):
        x = torch.randn(8)
        y = x.view(2, 4)
        y.requires_grad = True
        z = y.view(2, 2, 2)

        to_meta = MetaConverter()
        mx = to_meta(x)
        mz = to_meta(z)

        self.assertFalse(z.is_leaf)

        self.assertMetadataMatches(mx, x)
        self.assertMetadataMatches(mz, z)

    def test_leaf(self):
        x = torch.randn(4, requires_grad=True)
        to_meta = MetaConverter()
        m = to_meta(x)

        # check the test is actually testing what it claims
        self.assertTrue(m.is_leaf)
        self.assertTrue(m.requires_grad)

        self.assertMetadataMatches(m, x)

    def test_non_leaf(self):
        x = torch.randn(4, requires_grad=True)
        y = x.neg()
        to_meta = MetaConverter()
        m = to_meta(y)

        # check the test is actually testing what it claims
        self.assertFalse(m.is_leaf)
        self.assertTrue(m.requires_grad)

        self.assertMetadataMatches(m, y)

    def test_requires_grad_false(self):
        x = torch.randn(4, requires_grad=False)
        to_meta = MetaConverter()
        m = to_meta(x)

        # check the test is actually testing what it claims
        self.assertFalse(m.requires_grad)

        self.assertMetadataMatches(m, x)

    def test_channels_last(self):
        x = torch.empty(2, 3, 4, 5, memory_format=torch.channels_last)
        to_meta = MetaConverter()
        m = to_meta(x)

        # check the test is actually testing what it claims
        self.assertTrue(m.is_leaf)

        self.assertMetadataMatches(m, x)

    def test_channels_last_leaf(self):
        x = torch.empty(2, 3, 4, 5, memory_format=torch.channels_last, requires_grad=True)
        to_meta = MetaConverter()
        m = to_meta(x)

        # check the test is actually testing what it claims
        self.assertTrue(m.requires_grad)
        self.assertTrue(m.is_leaf)

        self.assertMetadataMatches(m, x)

    def test_channels_last_non_leaf(self):
        x = torch.empty(2, 3, 4, 5, memory_format=torch.channels_last, requires_grad=True)
        y = x + 2

        # sanity
        self.assertEqual(x.stride(), y.stride())
        self.assertFalse(y.is_leaf)

        to_meta = MetaConverter()
        m = to_meta(y)

        # check the test is actually testing what it claims
        self.assertTrue(m.requires_grad)
        self.assertFalse(m.is_leaf)

        self.assertMetadataMatches(m, y)

        # Check that we can autograd with m as input without erroring;
        # see https://github.com/pytorch/pytorch/issues/87956
        loss = m.sum()
        torch.autograd.grad(loss, m)

    def test_empty_strided_non_dense_leaf(self):
        x = torch.empty_strided((2, 2), (4, 2), requires_grad=True)

        to_meta = MetaConverter()
        m = to_meta(x)

        # check the test is actually testing what it claims
        self.assertTrue(m.requires_grad)
        self.assertTrue(m.is_leaf)

        self.assertMetadataMatches(m, x)

    def test_non_leaf_torture(self):
        x = torch.empty(20, requires_grad=True)
        with torch.no_grad():
            x.set_(x.storage(), 10, (2,), (2,))

        to_meta = MetaConverter()
        m = to_meta(x)

        # check the test is actually testing what it claims
        self.assertTrue(m.requires_grad)
        self.assertTrue(m.is_leaf)

        self.assertMetadataMatches(m, x)

    # NB: complex stuff is not actually exercised right now because
    # we have a blanket exclusion for complex conversion

    def test_view_as_real(self):
        x = torch.randn(4, dtype=torch.complex64)
        y = torch.view_as_real(x)
        m = MetaConverter()(y)
        self.assertMetadataMatches(m, y)

    def test_complex_noncontiguous_bug(self):
        x = torch.randn((2, 2, 4, 9), dtype=torch.complex32)[:, 0, :, :]
        m = MetaConverter()(x)
        self.assertMetadataMatches(m, x)

    def test_view_as_complex(self):
        x = torch.randn((4, 2), dtype=torch.float32)
        y = torch.view_as_complex(x)
        m = MetaConverter()(y)
        self.assertMetadataMatches(m, y)

    def test_view_dtype(self):
        x = torch.randn(4, dtype=torch.float32)
        y = x.view(dtype=torch.int32)
        m = MetaConverter()(y)
        self.assertMetadataMatches(m, y)

    def test_imag(self):
        x = torch.randn(4, dtype=torch.complex64)
        y = x.imag
        m = MetaConverter()(y)
        self.assertMetadataMatches(m, y)

    def test_weakref(self):
        x = torch.randn(4, 4, 4)
        m = MetaConverter()
        y = m(x)
        z = m(x)
        self.assertIs(y, z)
        self.assertEqual(len(m.tensor_memo), 1)
        self.assertEqual(len(m.storage_memo), 1)
        del x
        self.assertEqual(len(m.tensor_memo), 0)
        m.check_for_expired_weak_storages()
        self.assertEqual(len(m.storage_memo), 0)
        li = []
        r = []
        for i in range(4):
            li.append(torch.rand([i]))
            r.append(m(li[-1]))
        self.assertEqual(len(m.tensor_memo), 4)
        del li
        self.assertEqual(len(m.tensor_memo), 0)
        m.check_for_expired_weak_storages()
        self.assertEqual(len(m.storage_memo), 0)

    def test_tensor_outlives_converter(self):
        m = MetaConverter()
        ref = weakref.ref(m)
        x = torch.randn([4, 4])
        y = m(x)
        del m
        self.assertIs(ref(), None)

aten = torch.ops.aten

CHECK_STRIDES = {
    torch.Tensor.__getitem__,
}

CHECK_STRIDES_SKIPS = {
    aten._conj_physical.default,
    aten._fft_c2c.default,
    aten._fft_c2r.default,
    aten._fft_r2c.default,
    aten._linalg_svd.default,
    aten._scaled_dot_product_attention_forward.default,
    aten.binary_cross_entropy.default,
    aten.complex.default,
    aten.copysign.Tensor,
    aten.div.Tensor_mode,
    aten.floor_divide.default,
    aten.heaviside.default,
    aten.lerp.Scalar,
    aten.lerp.Tensor,
    aten.logical_and.default,
    aten.logical_or.default,
    aten.logical_xor.default,
    aten.pow.Scalar,
    aten.prelu.default,
    aten.special_xlog1py.default,
    aten.xlogy.Tensor,

    # channel_last and channel_last_3d related failures
    aten.convolution.default,

    # following ops fails if include_storage_offset = True, but these are a bit edge casey
    # we should still fix them, leaving them here for tracking.
    # aten._reshape_alias.default,  # repro with test_dispatch_symbolic_meta_outplace_all_strides_matmul_cuda_float32
    # aten.view.default,  # repro with test_dispatch_symbolic_meta_outplace_all_strides_unflatten_cuda_float32
}

def should_check_strides(func):
    if func in CHECK_STRIDES:
        return True
    if func in CHECK_STRIDES_SKIPS:
        return False
    if not isinstance(func, torch._ops.OpOverload):
        return False
    # Prims are expected to model strides correctly
    if func.namespace == "prims":
        return True
    # Check if it's a view, by testing if any of the returns have
    # a non-empty alias set
    if any(r.alias_info.before_set for r in func._schema.returns if r.alias_info):
        return True
    # TODO: check for TensorIterator
    return True

def assert_ref_meta_equal(test_case, func, meta_rs, rs, msg_callable):
    flat_meta_rs, _ = tree_flatten(meta_rs)
    flat_rs, _ = tree_flatten(rs)
    test_case.assertEqual(len(flat_meta_rs), len(flat_rs))
    for i, meta_r, r in zip(range(len(flat_rs)), flat_meta_rs, flat_rs):
        def test_assert(cond, msg):
            if not cond:
                raise RuntimeError(f"output {i}: {msg_callable(msg)}")
        if not isinstance(r, torch.Tensor):
            continue
        test_assert(isinstance(meta_r, torch.Tensor), f"but real {i}th result is Tensor")
        test_assert(meta_r.dtype == r.dtype, f"but real dtype was {r.dtype}")
        test_assert(meta_r.shape == r.shape, f"but real shape was {r.shape}")
        # See https://github.com/pytorch/pytorch/issues/78050
        if should_check_strides(func):
            same_strides, _ = torch._prims_common.check_significant_strides(meta_r, r)
            test_assert(same_strides, f"but real stride was {r.stride()}")
        test_assert(
            meta_r.storage_offset() == r.storage_offset(),
            f"but real storage_offset was {r.storage_offset()}")
        test_assert(meta_r.requires_grad == r.requires_grad, f"but real requires_grad was {r.requires_grad}")
        test_assert(meta_r.is_conj() == r.is_conj(), f"but real is_conj was {r.is_conj()}")
        test_assert(meta_r.is_neg() == r.is_neg(), f"but real is_neg was {r.is_neg()}")


# This environment variable controls whether or not we print expected failure
# lists at the end of a test suite run.  The intended usage looks like this:
#
# 1. Run `PYTORCH_COLLECT_EXPECT=1 python test/test_meta.py` on a CUDA build
#    of PyTorch that has LAPACK/MAGMA installed.  You can filter `-k test_meta`
#    or `-k test_dispatch_meta` to only focus on one or another list
# 2. Given the printed skip/xfail list, add them to the corresponding lists;
#    torch.* entries go in meta_function and aten.* entries go in meta_dispatch.
#    If there are preexisting entries, you need to merge in the entries.
#
# This is somewhat manual but typically you shouldn't need to do this, unless
# you've made a major change (e.g., added a new dtype to PyTorch) and need to
# refresh the lists.  If you want to do it from scratch, just clear out the
# preexisting lists before running.
#
# WARNING: Python dict literals will silently ignore duplicate keys
COLLECT_EXPECT = os.getenv('PYTORCH_COLLECT_EXPECT', '0') == '1'

seen_succeeded = {}
seen_failed = {}
failed_reasons = defaultdict(set)
def print_seen():
    expected_failures = []
    skips = []

    def fmt_dtypes(dtypes):
        r = ', '.join(sorted(dtype_abbrs[d] for d in dtypes))
        return '{' + r + '}'

    for op, failed_dtypes in seen_failed.items():
        ops = resolve_name(op)
        succeeded_dtypes = seen_succeeded.get(op, set())
        expected_failures_dtypes = failed_dtypes - succeeded_dtypes
        skips_dtypes = failed_dtypes & succeeded_dtypes
        reasons = ""
        if failed_reasons[op]:
            reasons = "  # " + ", ".join(sorted(failed_reasons[op]))
        if expected_failures_dtypes:
            expected_failures.append(f"    {ops}: {fmt_dtypes(expected_failures_dtypes)},{reasons}")
        if skips_dtypes:
            skips.append(f"    {ops}: {fmt_dtypes(skips_dtypes)},")
    expected_failures.sort()
    skips.sort()
    nl = '\n'
    print(f"""\
expected_failures = {{
{nl.join(expected_failures)}
}}

skips = {{
{nl.join(skips)}
}}
""")
if COLLECT_EXPECT:
    atexit.register(print_seen)

# Success forces pass; failure forces fail; skip unconditionally skips testing
TestExpect = Enum("TestExpect", ("SUCCESS", "XFAILURE", "SKIP"))

# unlike print produce strides
def verbose_print(e):
    class Lit:
        def __init__(self, s):
            self.s = s

        def __repr__(self):
            return self.s

    def go(t):
        if isinstance(t, torch.Tensor):
            return Lit(f"{t} stride={t.stride()}")
        else:
            return t

    return repr(tree_map(go, e))

def run_meta_crossref(
    test_case,
    test_expect,
    func,
    args,
    kwargs,
    *,
    dtype,
    device_type,
    run_symbolic_meta: bool
):
    to_meta = MetaConverter()
    do_meta = test_expect is not TestExpect.SKIP

    if do_meta:
        try:
            meta_args = tree_map(to_meta, args)
            meta_kwargs = tree_map(to_meta, kwargs)
        except Exception as e:
            raise RuntimeError(
                f"failed to convert args to meta; "
                f"originally (*{args}, **{kwargs})") from e

    try:
        rs = func(*args, **kwargs)
    except Exception as e:
        # A lot of OpInfo for inplace are actually broken because
        # they're not tested outside of gradcheck which only checks
        # torch.float64 and torch.complex128 (which this second one
        # often skipped as well).
        raise unittest.SkipTest("Original OpInfo is broken")


    # TODO: also handle cases where func raise an exception

    # For now, only attempt if we managed to convert all tensor types
    # (if any of them failed, we're in a mixed device situation and
    # this isn't well supported)
    if do_meta and to_meta.successful():
        # Special cases
        if func is torch.tensor_split:
            # Use original indices_or_sections, this argument is data dependent
            meta_args = (meta_args[0], args[1]) + meta_args[2:]
        elif func is torch.Tensor.__getitem__:
            # Ensure boolean tensors use original
            assert len(args) == 2
            flat_args, _ = tree_flatten(args[1])
            flat_meta_args, spec = tree_flatten(meta_args[1])
            flat_new_args = []
            for a, ma in zip(flat_args, flat_meta_args):
                flat_new_args.append(a if isinstance(a, torch.Tensor) and a.dtype in [torch.int8, torch.bool] else ma)
            meta_args = (meta_args[0], tree_unflatten(flat_new_args, spec))
        elif func is torch.ops.aten.repeat_interleave.Tensor:
            if kwargs.get("output_size", None) is None:
                meta_args = args
        elif func is torch.ops.aten.index.Tensor:
            # Don't convert boolean tensors to meta as they will have nonzero
            # called on them
            indices = []
            for meta_index, real_index in zip(meta_args[1], args[1]):
                if meta_index is not None and meta_index.dtype in [torch.int8, torch.bool]:
                    indices.append(real_index)
                else:
                    indices.append(meta_index)
            meta_args = (meta_args[0], indices)

        if kwargs.get("device", None) is not None:
            meta_kwargs["device"] = "meta"

        try:
            # Suppress warnings, this doesn't matter for test_meta.py
            # but it does matter if you want to use this decorator
            # for cross-ref testing, as some tests may be looking at
            # errors
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if run_symbolic_meta:
                    # Run the decomps and meta kernels registered
                    # to the python dispatcher instead of the regular dispatcher.
                    # This should be the same set of kernels
                    # that fake tensor runs in dynamic shapes mode.
                    with enable_python_dispatcher():
                        meta_rs = func(*meta_args, **meta_kwargs)
                else:
                    meta_rs = func(*meta_args, **meta_kwargs)
        except Exception as e:
            if test_expect is TestExpect.XFAILURE:
                return rs
            seen_failed.setdefault(func, set()).add(dtype)
            if isinstance(e, NotImplementedError):
                m = RE_NOT_IMPLEMENTED_MSG.search(e.args[0])
                if m:
                    failed_reasons[func].add(m.group(1))
            if COLLECT_EXPECT:
                return rs
            raise RuntimeError(f"""\
failed to run: {resolve_name(func)}(
*{verbose_print(meta_args)},
**{verbose_print(meta_kwargs)}
)""") from e
        else:
            try:
                delim = ',\n  '
                assert_ref_meta_equal(test_case, func, meta_rs, rs, lambda msg: f"""\
meta disagrees with real impl:
{resolve_name(func)}(
  {delim.join(map(verbose_print, meta_args))},
  {delim.join(k + ": " + verbose_print(v) for k, v in meta_kwargs.items())}
) = (
  {verbose_print(meta_rs)}
)
{msg}
""")
            except Exception:
                if test_expect is TestExpect.XFAILURE:
                    return rs
                seen_failed.setdefault(func, set()).add(dtype)
                if COLLECT_EXPECT:
                    return rs
                raise
            else:
                seen_succeeded.setdefault(func, set()).add(dtype)
                if test_expect is TestExpect.XFAILURE and not COLLECT_EXPECT:
                    raise RuntimeError(f"unexpected success {resolve_name(func)}")

    return rs



RE_NOT_IMPLEMENTED_MSG = re.compile(r"Could not run '([^']+)' with arguments ")

meta_function_expected_failures = {
    torch.Tensor.to_sparse : {f64, i32, c128, i64, i16, f16, u8, c64, bf16, b8, i8, f32},
    torch.allclose : {f64, f16, c128, c64, bf16, f32},
    torch.argwhere : {f64, i32, c128, i64, i16, f16, u8, c64, bf16, b8, i8, f32},
    torch.combinations : {f64, i32, c128, i64, i16, f16, u8, c64, bf16, b8, i8, f32},
    torch.corrcoef : {f64, i32, c128, i64, i16, u8, c64, bf16, i8, f32},
    torch.count_nonzero : {f64, i32, c128, i64, i16, f16, u8, c64, bf16, b8, i8, f32},
    torch.cov : {f64, i32, c128, i64, i16, u8, c64, bf16, i8, f32},
    torch.functional.istft : {f64, c64, c128, f32},
    torch.geqrf : {f64, c64, c128, f32},
    torch.linalg.householder_product : {f64, c64, c128, f32},
    torch.linalg.solve_triangular : {f64, c64, c128, f32},
    torch.masked_select : {f64, i32, c128, i64, i16, f16, u8, c64, bf16, b8, i8, f32},
    torch.matrix_exp : {f64, c128, c64, bf16, f32},
    torch.nonzero : {f64, i32, c128, i64, i16, c32, f16, u8, c64, bf16, b8, i8, f32},
    torch.Tensor.nonzero : {f64, i32, c128, i64, i16, c32, f16, u8, c64, bf16, b8, i8, f32},
    torch.ormqr : {f64, c64, c128, f32},
    torch.repeat_interleave : {f64, i32, c128, i64, i16, c32, f16, u8, c64, bf16, b8, i8, f32},
    torch.take : {f64, i32, c128, i64, i16, f16, u8, c64, bf16, b8, i8, f32},
    torch.Tensor.item : {f64, i32, c128, i64, i16, f16, u8, c64, bf16, b8, i8, f32},
    torch.bincount : {i32, i64, u8, i16, i8},
    torch.frexp : {f64, f16, bf16, f32},
    torch.functional.unique : {f64, i32, i64, u8, i16, bf16, b8, i8, f32},
    torch.functional.unique_consecutive : {f64, i32, i64, u8, i16, bf16, b8, i8, f32},
    torch.histc : {f64, bf16, f32},
    torch.histogram : {f64, f32},
    torch.histogramdd : {f64, f32},
    torch.kthvalue : {f64, i32, i64, u8, i16, bf16, i8, f32},
    torch.logcumsumexp : {f64, bf16, f32},
    torch.median : {f64, i32, i64, u8, i16, bf16, i8, f32},
    torch.mode : {f64, i32, i64, f16, u8, i16, bf16, b8, i8, f32},
    torch.multinomial : {f64, bf16, f32},
    torch.nn.functional.ctc_loss : {f64, f32},
    torch.nn.functional.gaussian_nll_loss : {f64, bf16, f32},
    torch.nn.functional.max_pool3d : {f64, f32},
    torch.nn.functional.max_pool3d_with_indices : {f64, f32},
    torch.nn.functional.max_unpool1d : {f64, f32},
    torch.nn.functional.max_unpool2d : {f64, f32},
    torch.nn.functional.max_unpool3d : {f64, f32},
    torch.nn.functional.multi_margin_loss : {f64, f32},
    torch.nn.functional.multilabel_margin_loss : {f64, f32},
    torch.nn.functional.one_hot : {i64},
    torch.nn.functional.pdist : {f64, f32},
    torch.polar : {f64, f32},
    torch.segment_reduce : {f64, f16, bf16, f32},
    torch.searchsorted : {f64, i32, i64, f16, u8, i16, bf16, i8, f32},
    torch.symeig : {f64, f32, c128, c64},
    torch.cholesky : {f64, f32, c128, c64},
    torch.cholesky_inverse : {f64, f32, c128, c64},
    torch.cholesky_solve : {f64, f32, c128, c64},
    torch.linalg.eig : {f64, f32, c128, c64},
    torch.linalg.eigvals : {f64, f32, c128, c64},
    torch.linalg.lstsq : {f64, f32, c128, c64},
    torch.Tensor.conj_physical_: {c128, c32, c64},
}

meta_function_expected_failures_only_outplace = {
    torch.nn.functional.rrelu : {f64, bf16, f32},
}

"""
# This is some sample code for how we could dump these dicts into YAML
# file for easier reading/writing
import yaml
print(yaml.dump(
  {resolve_name(k): [dtype_abbrs[d] for d in v]
   for k, v in meta_function_expected_failures.items()}, default_flow_style=None))
import sys
sys.exit()
"""

meta_function_skips = {
    torch.Tensor.__rmatmul__ : {bf16, c128, f64, f32, f16, c64},
    torch.Tensor.matmul : {f64, f32, c128, c64},
    torch.fft.fft2 : {i8, i64, u8, c128, b8, f64, i16, f32, i32, c64, c32, f16},
    torch.fft.fft : {i8, i64, u8, c128, b8, f64, i16, f32, i32, c64, c32, f16},
    torch.fft.fftn : {i8, i64, u8, c128, b8, f64, i16, f32, i32, c64, c32, f16},
    torch.fft.ifft2 : {i8, i64, u8, c128, b8, f64, i16, f32, i32, c64, c32, f16, c32},
    torch.fft.ifft : {c128, c64, c32, f16},
    torch.fft.ifftn : {i8, i64, u8, c128, b8, f64, i16, f32, i32, c64, c32, f16},
    torch.fft.hfft: {f16},
    torch.fft.hfftn: {f16},
    torch.fft.hfft2: {f16},
    torch.fft.ihfft: {f16},
    torch.fft.ihfft2 : {i8, i64, u8, f64, b8, f32, i32, i16, f16, c32, f16},
    torch.fft.ihfftn : {i8, i64, u8, f64, b8, f32, i32, i16, c32, f16},
    torch.fft.irfft2 : {f16},
    torch.fft.irfft : {f16},
    torch.fft.irfftn : {f16},
    torch.fft.rfft2 : {i8, i64, u8, f64, b8, f32, i32, i16, c32, f16},
    torch.fft.rfft : {i8, i64, u8, f64, b8, f32, i32, i16, c32, f16},
    torch.fft.rfftn : {i8, i64, u8, f64, b8, f32, i32, i16, c32, f16},
    torch.functional.atleast_2d : {bf16, i8, c32, i64, u8, c128, b8, f64, i16, i32, f32, f16, c64},
    torch.functional.atleast_3d : {bf16, i8, c32, i64, u8, c128, b8, f64, i16, i32, f32, f16, c64},
    torch.functional.cartesian_prod : {bf16, i8, i64, u8, c128, b8, f64, i16, i32, f32, f16, c64},
    torch.functional.einsum : {bf16, c128, f64, f32, f16, c64},
    torch.functional.stft : {c128, f32, c64, f64},
    torch.functional.tensordot : {bf16, i8, i64, u8, c128, f64, i16, f32, i32, c64},
    torch.inner : {bf16, i8, i64, u8, c128, f64, i16, f32, i32, c64},
    torch.linalg.lu_solve : {c128, c64},
    torch.linalg.matrix_norm : {c128, f32, c64, f64},
    torch.linalg.matrix_power : {c128, c64},
    torch.linalg.matrix_rank : {c128, c64},
    torch.linalg.svd : {c128, c64},
    torch.matmul : {bf16, c128, f64, f32, f16, c64},
    torch.nanquantile : {f64, f32},
    torch.narrow : {bf16, i8, i64, u8, c128, b8, f64, i16, i32, f32, f16, c32, c64},
    torch.nn.functional.batch_norm : {f64, f32},
    torch.nn.functional.binary_cross_entropy : {bf16, f64, f32, f16},
    torch.nn.functional.dropout3d : {bf16, f64, f32, f16},
    torch.nn.functional.local_response_norm : {bf16, f64, f32, f16},
    torch.svd : {c128, c64},
    torch.take_along_dim : {bf16, i8, i64, u8, c128, b8, f64, i16, i32, f32, f16, c64},
    torch.vstack : {bf16, i8, c32, i64, u8, c128, b8, f64, i16, i32, f32, f16, c64},
    torch.aminmax : {i8, i64, u8, f64, b8, f32, i32, i16},
    torch.cummax : {bf16, i8, i64, u8, f64, b8, f32, i32, i16},
    torch.cummin : {bf16, i8, i64, u8, f64, b8, f32, i32, i16},
    torch.diff : {b8},
    torch.equal : {bf16, i8, c32, i64, u8, c128, b8, f64, i16, i32, f32, f16, c64},
    torch.functional.cdist : {f64, f32},
    torch.nanmean : {bf16, f64, f32, f16},
    torch.nn.functional.cross_entropy : {bf16, f64, f32},
    torch.nn.functional.interpolate : {bf16, f64, f32, u8},
    torch.nn.functional.nll_loss : {bf16, f64, f32},
    torch.linalg.pinv : {f64, f32},
    torch.linalg.cond : {c128, c64, f32, f64},
    torch.linalg.vander: {c128, c64, f32, f64, i16, i32, i64, i8, u8},
    torch.linalg.vecdot : {bf16, f64, f32, f16},
    torch.empty : {bf16, i8, c32, i64, u8, c128, b8, f64, i16, i32, f32, f16, c64},
    # This fails for arguments dispatched to grid_sampler_3d, but succeeds
    # for grid_sampler_2d, so we can't just xfail it
    torch.nn.functional.grid_sample : {f64, f32},
    torch.bucketize : {f64, i32, i64, f16, u8, i16, bf16, i8, f32},
    torch.Tensor.addbmm_: {bf16, c128, c64, f32, f64, i16, i32, i64, i8, u8},
}


meta_function_device_expected_failures = defaultdict(dict)
meta_function_device_expected_failures_only_outplace = defaultdict(dict)
meta_function_device_skips = defaultdict(dict)

meta_function_device_expected_failures['cpu'] = {
    torch.native_batch_norm: {bf16},
    torch.native_layer_norm: {bf16},
}

meta_function_device_expected_failures['cuda'] = {
    torch.corrcoef: {bf16, f16},  # aten::_local_scalar_dense
    torch.cov: {f16},  # aten::_local_scalar_dense
    torch.functional.unique: {f16},  # aten::_unique2, aten::unique_dim
    torch.functional.unique_consecutive: {f16},  # aten::unique_consecutive
    torch.geqrf: {f32, f64},  # aten::geqrf
    torch.histc: {i16, i32, i64, i8},  # aten::histc, aten::histc.out
    torch.kthvalue: {f16},  # aten::kthvalue.values
    torch.linalg.householder_product: {f32, f64},  # aten::linalg_householder_product, aten::linalg_householder_product.out
    torch.linalg.solve_triangular: {f32, f64},  # aten::linalg_solve_triangular, aten::linalg_solve_triangular.out
    torch.logcumsumexp: {bf16, f16},  # aten::_logcumsumexp, aten::_logcumsumexp.out
    torch.matrix_exp: {f16},  # aten::linalg_matrix_exp
    torch.median: {f16},  # aten::median, aten::median.dim_values
    torch.multinomial: {f16},  # aten::multinomial, aten::multinomial.out
    torch.nn.functional.gaussian_nll_loss: {f16},  # aten::_local_scalar_dense
    torch.nn.functional.max_pool3d: {bf16, f16},  # aten::max_pool3d_with_indices
    torch.nn.functional.max_pool3d_with_indices: {bf16, f16},  # aten::max_pool3d_with_indices
    torch.nn.functional.max_unpool1d: {f16},  # aten::max_unpool2d
    torch.nn.functional.max_unpool2d: {f16},  # aten::max_unpool2d
    torch.nn.functional.max_unpool3d: {f16},  # aten::max_unpool3d
    torch.nn.functional.multi_margin_loss: {bf16, f16},  # aten::multi_margin_loss
    torch.nn.functional.multilabel_margin_loss: {bf16, f16},  # aten::multilabel_margin_loss_forward
    torch.ormqr: {f32, f64},  # aten::ormqr, aten::ormqr.out
}

meta_function_device_expected_failures_only_outplace['cuda'] = {
    torch.nn.functional.rrelu: {f16},  # aten::rrelu_with_noise
}

meta_function_device_skips['cpu'] = {
    torch.narrow_copy: {b8, bf16, c128, c32, c64, f16, f32, f64, i16, i32, i64, i8, u8},
    torch.native_batch_norm: {f32, f64},
}

meta_function_device_skips['cuda'] = {
    torch.cummax: {f16},
    torch.cummin: {f16},
    torch.functional.tensordot: {f16},
    torch.inner: {f16},
    torch.linalg.matrix_power: {f32, f64},
    torch.linalg.matrix_rank: {f32, f64},
    torch.linalg.svd: {f32, f64},
    torch.nn.functional.cross_entropy: {f16},
    torch.nn.functional.interpolate: {f16},
    torch.nn.functional.nll_loss: {f16},
    torch.svd: {f32, f64},
    # This fails for arguments dispatched to grid_sampler_3d, but succeeds
    # for grid_sampler_2d, so we can't just xfail it
    torch.nn.functional.grid_sample : {f16},
}

# This is a __torch_function__ mode that, when enabled, interposes every
# Torch API call and runs the operator as normal, and then reruns it
# with meta inputs, and then checks that everything about the output agrees.
# Most of the logic deals with faithfully replicating the original tensor
# as a meta tensor, which is nontrivial because there are a lot of subsystems
# that may potentially be exercised.
#
# That being said, this class is a little overkill for what it is doing in
# this test file (since I could have just inlined __torch_function__ on the
# OpInfo call, and OpInfos generally have very regular inputs), but it will be
# useful for more comprehensive testing e.g., as seen in
# https://github.com/pytorch/pytorch/pull/75994  The big benefit is it is
# A LOT more efficient that torch dispatch mode (at the cost of less coverage)
class MetaCrossRefFunctionMode(torch.overrides.TorchFunctionMode):
    test_case: TestCase
    device_type: str
    dtype: torch.dtype

    def __init__(self, test_case, *, device, dtype, inplace):
        self.test_case = test_case
        self.device_type = torch.device(device).type
        self.dtype = dtype
        self.inplace = inplace

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        if (
            torch.jit.is_tracing() or isinstance(func, torch.ScriptMethod) or
            # meta converter doesn't work correctly when no_dispatch() is on, so
            # skip running the crossref test in this case
            torch._C._dispatch_tls_local_exclude_set().has(torch._C.DispatchKey.Python)
        ):
            return func(*args, **kwargs)

        if self.dtype in meta_function_skips.get(func, set()):
            test_expect = TestExpect.SKIP
        elif self.dtype in meta_function_device_skips[self.device_type].get(func, set()):
            test_expect = TestExpect.SKIP
        elif self.dtype in meta_function_expected_failures.get(func, set()):
            test_expect = TestExpect.XFAILURE
        elif not self.inplace and self.dtype in meta_function_expected_failures_only_outplace.get(func, set()):
            test_expect = TestExpect.XFAILURE
        elif self.dtype in meta_function_device_expected_failures[self.device_type].get(func, set()):
            test_expect = TestExpect.XFAILURE
        elif not self.inplace and \
                self.dtype in meta_function_device_expected_failures_only_outplace[self.device_type].get(func, set()):
            test_expect = TestExpect.XFAILURE
        else:
            test_expect = TestExpect.SUCCESS

        return run_meta_crossref(
            self.test_case, test_expect, func, args,
            kwargs, dtype=self.dtype, device_type=self.device_type, run_symbolic_meta=False
        )

# these always fail
meta_dispatch_expected_failures = {
    aten.allclose.default: {f16, bf16, f32, f64, c64, c128},  # NotImplementedError: 'aten::_local_scalar_dense'
    aten._fft_c2c.out : {f16, c64, i8, f64, c128, i32, i64, f32, c32, b8, i16, u8},
    aten._fft_r2c.out : {f16, i8, f64, i32, i64, f32, b8, i16, u8},
    aten.cholesky.default : {c64, c128, f64, f32},
    aten.cholesky.out : {c64, c128, f64, f32},
    aten.cholesky_inverse.default : {c64, c128, f64, f32},
    aten.cholesky_inverse.out : {c64, c128, f64, f32},
    aten.cholesky_solve.default : {c64, c128, f64, f32},
    aten.cholesky_solve.out : {c64, c128, f64, f32},
    aten.count_nonzero.default : {c64, f16, i8, f64, c128, i64, bf16, f32, i32, b8, i16, u8},
    aten.count_nonzero.dim_IntList : {c64, f16, i8, f64, c128, i64, bf16, f32, i32, b8, i16, u8},
    aten.geqrf.default : {c64, c128, f64, f32},
    aten.linalg_eig.default : {c64, c128, f64, f32},
    aten.linalg_householder_product.default : {c64, c128, f64, f32},
    aten.linalg_householder_product.out : {c64, c128, f64, f32},
    aten.linalg_lstsq.default : {c64, c128, f64, f32},
    aten.linalg_matrix_exp.default : {c64, bf16, f32, f64, c128},
    aten.linalg_solve_triangular.default : {c64, c128, f64, f32},
    aten.linalg_solve_triangular.out : {c64, c128, f64, f32},
    aten.masked_select.default : {c64, f16, i8, f64, c128, i64, bf16, f32, i32, b8, i16, u8},
    aten.masked_select.out : {c64, f16, i8, f64, c128, i64, bf16, f32, i32, b8, i16, u8},
    aten.nonzero.default : {c64, f16, i8, f64, c128, i64, bf16, f32, i32, c32, b8, i16, u8},
    aten.nonzero.out : {c64, f16, i8, f64, c128, i64, bf16, f32, i32, c32, b8, i16, u8},
    aten.ormqr.default : {c64, c128, f64, f32},
    aten.ormqr.out : {c64, c128, f64, f32},
    aten.polar.out : {f32, f64},
    aten.symeig.default : {c64, c128, f64, f32},
    aten.take.default : {c64, f16, i8, f64, c128, i64, bf16, f32, i32, b8, i16, u8},
    aten.take.out : {c64, f16, i8, f64, c128, i64, bf16, f32, i32, b8, i16, u8},
    aten.tensordot.out : {c64, i8, f64, c128, i64, bf16, f32, i32, i16, u8},
    aten.to_sparse.default : {c64, f16, i8, f64, c128, i64, bf16, f32, i32, b8, i16, u8},
    aten.to_sparse.sparse_dim : {c64, f16, i8, f64, c128, i64, bf16, f32, i32, b8, i16, u8},
    aten._ctc_loss.default : {f32, f64},  # Shape of second output depends on data.
    aten._ctc_loss.Tensor : {f32, f64},  # Shape of second output depends on data.
    aten._histogramdd_bin_edges.default : {f32, f64},
    aten._histogramdd_from_bin_cts.default : {f32, f64},
    aten._histogramdd_from_bin_tensors.default : {f32, f64},
    aten._local_scalar_dense.default : {c32, c64, f16, i8, f64, c128, i64, bf16, f32, i32, b8, i16, u8},
    aten._pdist_forward.default : {f32, f64},
    aten._unique2.default : {i8, f64, i64, bf16, f32, i32, b8, i16, u8},
    aten.bincount.default : {i64, i8, i32, i16, u8},
    aten.equal.default : {c64, f16, i8, f64, c128, i64, bf16, f32, i32, b8, i16, u8},
    aten.frexp.Tensor : {bf16, f32, f16, f64},
    aten.grid_sampler_3d.default : {f32, f64},
    aten.histc.default : {bf16, f32, f64},
    aten.histc.out : {bf16, f32, f64},
    aten.histogram.bin_ct : {f32, f64},
    aten.histogram.bins_tensor : {f32, f64},
    aten.kthvalue.default : {i8, f64, i64, bf16, f32, i32, i16, u8},
    aten.logcumsumexp.default : {bf16, f32, f64},
    aten.logcumsumexp.out : {bf16, f32, f64},
    aten.max_pool3d_with_indices.default : {f32, f64},
    aten.max_unpool2d.default : {f32, f64},
    aten.max_unpool3d.default : {f32, f64},
    aten.median.default : {i8, f64, i64, bf16, f32, i32, i16, u8},
    aten.median.dim : {i8, f64, i64, bf16, f32, i32, i16, u8},
    aten.mode.default : {f16, i8, f64, i64, bf16, f32, i32, b8, i16, u8},
    aten.multi_margin_loss.default : {f32, f64},
    aten.multilabel_margin_loss_forward.default : {f32, f64},
    aten.multinomial.default : {bf16, f32, f64},
    aten.multinomial.out : {bf16, f32, f64},
    aten.nll_loss2d_forward.default : {bf16, f32, f64},
    aten.polar.default : {f32, f64},
    aten.rrelu_with_noise.default : {bf16, f32, f64},
    aten.searchsorted.Tensor : {f16, i8, f64, i64, bf16, f32, i32, i16, u8},
    aten.searchsorted.Tensor_out : {f16, i8, f64, i64, bf16, f32, i32, i16, u8},
    aten.segment_reduce.default : {bf16, f32, f16, f64},
    aten.unique_consecutive.default : {i8, f64, i64, bf16, f32, i32, b8, i16, u8},
    aten.unique_dim.default : {i8, f64, i64, bf16, f32, i32, b8, i16, u8},
    aten.upsample_nearest3d.vec : {bf16, f32, f64, u8},
    aten.conj_physical_.default: {c128, c32, c64},
}

# these sometimes pass and sometimes fail
meta_dispatch_skips = {
    aten.index.Tensor: {i64, bf16, f16, u8, b8, f32, i8, f64, i16, i32, c32, c64, c128},  # at::nonzero doesn't have a Meta function
    aten._to_copy.default: {i64, bf16, f16, u8, b8, f32, i8, f64, i16, i32, c32, c64, c128},
    aten.aminmax.default: {i64, u8, b8, f32, i8, f64, i16, i32},
    aten.cummax.default: {i64, bf16, u8, b8, f32, i8, f64, i16, i32},
    aten.cummin.default: {i64, bf16, u8, b8, f32, i8, f64, i16, i32},
    aten.linalg_lu_solve.default: {c32, c64, c128},
    aten.linalg_lu_solve.out: {c32, c64, c128},
    aten.linalg_pinv.atol_rtol_tensor: {f32, f64},
    aten.linalg_pinv.atol_rtol_tensor_out: {f32, f64},
    aten.empty.memory_format: {b8, bf16, c128, c64, c32, f16, f32, f64, i16, i32, i64, i8, u8},
    aten.bucketize.Tensor : {f16, i8, f64, i64, bf16, f32, i32, i16, u8},
    aten.bucketize.Tensor_out : {f16, i8, f64, i64, bf16, f32, i32, i16, u8},
    aten.addbmm_.default: {bf16, c128, c64, f32, f64, i16, i32, i64, i8, u8},
}

# For CompositeImplicitAutograd functions that fail before hitting the Mode
meta_dispatch_early_skips = set({
    torch.Tensor.float_power_,
})

meta_dispatch_device_expected_failures = defaultdict(dict)
meta_dispatch_device_skips = defaultdict(dict)

meta_dispatch_device_expected_failures['cpu'] = {
    aten.native_batch_norm.default: {bf16},
    aten.native_layer_norm.default: {bf16},
}

meta_dispatch_device_expected_failures['cuda'] = {
    aten._unique2.default: {f16},  # aten::_unique2
    aten._use_cudnn_ctc_loss.default: {f32, f64},  # aten::_use_cudnn_ctc_loss
    aten._use_cudnn_ctc_loss.Tensor: {f32, f64},  # aten::_use_cudnn_ctc_loss.Tensor
    aten.cudnn_grid_sampler.default: {f16, f32, f64},  # aten::cudnn_grid_sampler
    aten.geqrf.default: {f32, f64},  # aten::geqrf
    aten.grid_sampler_3d.default: {f16},  # aten::grid_sampler_3d
    aten.histc.default: {i16, i32, i64, i8},  # aten::histc
    aten.histc.out: {i16, i32, i64, i8},  # aten::histc.out
    aten.kthvalue.default: {f16},  # aten::kthvalue.values
    aten.linalg_eigvalsh.out: {f32, f64},  # aten::linalg_eigvalsh.out
    aten.linalg_householder_product.default: {f32, f64},  # aten::linalg_householder_product
    aten.linalg_householder_product.out: {f32, f64},  # aten::linalg_householder_product.out
    aten.linalg_matrix_exp.default: {f16},  # aten::linalg_matrix_exp
    aten.linalg_solve_triangular.default: {f32, f64},  # aten::linalg_solve_triangular
    aten.linalg_solve_triangular.out: {f32, f64},  # aten::linalg_solve_triangular.out
    aten.log_sigmoid_forward.default: {bf16, f16, f64, f32},
    aten.log_sigmoid_forward.output : {bf16, f16, f64, f32},  # aten::log_sigmoid_forward.output
    aten.logcumsumexp.default: {bf16, f16},  # aten::_logcumsumexp
    aten.logcumsumexp.out: {bf16, f16},  # aten::_logcumsumexp.out
    aten.max_pool3d_with_indices.default: {bf16, f16},  # aten::max_pool3d_with_indices
    aten.max_unpool2d.default: {f16},  # aten::max_unpool2d
    aten.max_unpool3d.default: {f16},  # aten::max_unpool3d
    aten.median.default: {f16},  # aten::median
    aten.median.dim: {f16},  # aten::median.dim_values
    aten.multi_margin_loss.default: {bf16, f16},  # aten::multi_margin_loss
    aten.multilabel_margin_loss_forward.default: {bf16, f16},  # aten::multilabel_margin_loss_forward
    aten.multinomial.default: {f16},  # aten::multinomial
    aten.multinomial.out: {f16},  # aten::multinomial.out
    aten.nll_loss2d_forward.default: {f16},  # aten::nll_loss2d_forward
    aten.ormqr.default: {f32, f64},  # aten::ormqr
    aten.ormqr.out: {f32, f64},  # aten::ormqr.out
    aten.rrelu_with_noise.default: {f16},  # aten::rrelu_with_noise
    aten.tensordot.out: {f16},  # aten::tensordot.out
    aten.unique_consecutive.default: {f16},  # aten::unique_consecutive
    aten.unique_dim.default: {f16},  # aten::unique_dim
    aten.upsample_nearest3d.vec: {f16},  # aten::upsample_nearest3d.vec
}

meta_dispatch_device_skips['cpu'] = {
    aten._embedding_bag_forward_only.default: {bf16, f16, f32, f64},
    aten.native_batch_norm.default: {f32, f64},
}

meta_dispatch_device_skips['cuda'] = {
    aten._conj.default: {c32, f16},  # file issue
    aten._linalg_svd.default: {c64, c128},  # aten::linalg_eigvalsh.out
    aten.cudnn_batch_norm.default: {f32, f64},
    aten.log_softmax.int : {c32, c64},
    aten.softmax.int : {c32, c64},
    aten.softmax.int : {c32, c64},

    aten.cummax.default: {f16},
    aten.cummin.default: {f16},
    # ROCm stuff; technically this should be expected failure but it's
    # not worth it; these should get unified anyway
    aten.miopen_batch_norm.default: {f32},
}

def get_strided_args(args):

    def get_strided_variants(t, include_storage_offset=False):
        variants = []

        # contiguous
        variants.append(t)

        # transposed
        if t.ndim > 1:
            perm = list(reversed(range(t.ndim)))
            transposed = torch.empty(
                t.shape[::-1], device=t.device, dtype=t.dtype, requires_grad=t.requires_grad
            ).permute(perm).copy_(t)
            variants.append(transposed)

        # nondense
        if t.ndim > 0:
            nondense = torch.repeat_interleave(t, 2, dim=-1)[..., ::2]
            variants.append(nondense)

        # channel_last
        if t.ndim == 4:
            variants.append(t.contiguous(memory_format=torch.channels_last))

        # channel_last_3d
        if t.ndim == 5:
            variants.append(t.contiguous(memory_format=torch.channels_last_3d))

        # storage_offset
        if include_storage_offset:
            buffer = torch.empty(t.numel() + 1, device=t.device, dtype=t.dtype, requires_grad=t.requires_grad)
            buffer = buffer.as_strided(t.shape, t.stride(), storage_offset=1)
            buffer.copy_(t)
            variants.append(buffer)

        return variants

    strided_args = []
    for arg in args:
        if isinstance(arg, torch.Tensor) and not arg.is_sparse_csr and arg.is_contiguous():
            strided_arg_variants = get_strided_variants(arg)
        else:
            strided_arg_variants = [arg]
        strided_args.append(strided_arg_variants)

    for result in itertools.product(*strided_args):
        yield result

class MetaCrossRefDispatchMode(torch.utils._python_dispatch.TorchDispatchMode):
    test_case: TestCase
    device: torch.device
    dtype: torch.dtype

    def __init__(self, test_case, *, device, dtype, symbolic_meta: bool):
        self.test_case = test_case
        # save TLS
        self.precision = test_case.precision
        self.rel_tol = test_case.rel_tol
        self.device_type = torch.device(device).type
        self.dtype = dtype
        self.symbolic_meta = symbolic_meta

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        self.test_case.precision = self.precision
        self.test_case.rel_tol = self.rel_tol

        if self.dtype in meta_dispatch_skips.get(func, set()):
            test_expect = TestExpect.SKIP
        elif self.dtype in meta_dispatch_device_skips[self.device_type].get(func, set()):
            test_expect = TestExpect.SKIP
        elif self.dtype in meta_dispatch_expected_failures.get(func, set()):
            test_expect = TestExpect.XFAILURE
        elif self.dtype in meta_dispatch_device_expected_failures[self.device_type].get(func, set()):
            test_expect = TestExpect.XFAILURE
        else:
            test_expect = TestExpect.SUCCESS

        return run_meta_crossref(
            self.test_case,
            test_expect,
            func,
            args,
            kwargs,
            dtype=self.dtype,
            device_type=self.device_type,
            run_symbolic_meta=self.symbolic_meta,
        )

# NB: we're running these tests only on CUDA because there are some
# inconsistencies between CUDA and CPU, and running on CUDA makes it easier
# to ignore the CPU case when inconsistencies arise.  Ideally we deal
# with the inconsistencies but this takes time.
class TestMeta(TestCase):
    # Copies inputs to inplace operations to avoid inplace modifications
    #   to leaves requiring gradient
    def _get_safe_inplace(self, inplace_variant):
        @wraps(inplace_variant)
        def _fn(t, *args, **kwargs):
            return inplace_variant(t.clone(), *args, **kwargs)

        return _fn

    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @skipIfCrossRef
    @suppress_warnings
    @ops(op_db)
    def test_meta_outplace(self, device, dtype, op):
        # run the OpInfo sample inputs, cross-referencing them with the
        # meta implementation and check the results are the same.  All
        # the heavy lifting happens in MetaCrossRefFunctionMode
        func = op.get_op()
        samples = op.sample_inputs(device, dtype, requires_grad=False)
        for sample_input in samples:
            args = [sample_input.input] + list(sample_input.args)
            kwargs = sample_input.kwargs
            with MetaCrossRefFunctionMode(self, dtype=dtype, device=device, inplace=False):
                expected = func(*args, **kwargs)
                if isinstance(expected, torch.Tensor) and op.supports_out:
                    func(*args, **kwargs, out=expected)

    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @skipIfCrossRef
    @suppress_warnings
    @ops(op_db)
    def test_meta_inplace(self, device, dtype, op):
        func = op.get_inplace()
        if not func:
            self.skipTest("No inplace variable for this op")
        func = self._get_safe_inplace(func)
        samples = op.sample_inputs(device, dtype, requires_grad=False)
        for sample_input in samples:
            if sample_input.broadcasts_input:
                continue
            args = [sample_input.input] + list(sample_input.args)
            kwargs = sample_input.kwargs
            with MetaCrossRefFunctionMode(self, dtype=dtype, device=device, inplace=True):
                expected = func(*args, **kwargs)

    def _run_dispatch_meta_test(self, device, dtype, op, symbolic_meta, inplace, all_stride_variants=False):
        if inplace:
            func = op.get_inplace()
            if not func:
                self.skipTest("No inplace variable for this op")
        else:
            func = op.get_op()

        if func in meta_dispatch_early_skips:
            self.skipTest("Function is in dispatch early skips")

        if inplace:
            func = self._get_safe_inplace(func)

        samples = op.sample_inputs(device, dtype, requires_grad=False)
        for sample_input in samples:
            if inplace and sample_input.broadcasts_input:
                continue

            sample_args = [sample_input.input] + list(sample_input.args)
            kwargs = sample_input.kwargs

            if all_stride_variants and sum(isinstance(arg, torch.Tensor) for arg in sample_args) <= 5:
                # test inputs <= 5 tensors to avoid combinatorial explosion
                strided_args = get_strided_args(sample_args)
            else:
                strided_args = [sample_args]

            for args in strided_args:
                with MetaCrossRefDispatchMode.push(self, dtype=dtype, device=device, symbolic_meta=symbolic_meta):
                    expected = func(*args, **kwargs)

                    if not inplace and isinstance(expected, torch.Tensor) and op.supports_out:
                        func(*args, **kwargs, out=expected)


    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @skipIfCrossRef
    @suppress_warnings
    @ops(op_db)
    def test_dispatch_meta_outplace(self, device, dtype, op):
        self._run_dispatch_meta_test(device, dtype, op, symbolic_meta=False, inplace=False)


    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @skipIfCrossRef
    @suppress_warnings
    @ops(op_db)
    def test_dispatch_meta_inplace(self, device, dtype, op):
        self._run_dispatch_meta_test(device, dtype, op, symbolic_meta=False, inplace=True)

    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @skipIfCrossRef
    @suppress_warnings
    @ops(op_db)
    def test_dispatch_symbolic_meta_outplace(self, device, dtype, op):
        self._run_dispatch_meta_test(device, dtype, op, symbolic_meta=True, inplace=False)


    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @skipIfCrossRef
    @suppress_warnings
    @ops(op_db)
    def test_dispatch_symbolic_meta_inplace(self, device, dtype, op):
        self._run_dispatch_meta_test(device, dtype, op, symbolic_meta=True, inplace=True)

    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @skipIfCrossRef
    @suppress_warnings
    # only test one dtype, as output stride behavior is the same for all dtypes
    @ops(op_db, dtypes=OpDTypes.any_common_cpu_cuda_one)
    # Only test on CUDA, as CUDA kernel's stride is the reference
    @onlyCUDA
    def test_dispatch_symbolic_meta_outplace_all_strides(self, device, dtype, op):
        self._run_dispatch_meta_test(device, dtype, op, symbolic_meta=True, inplace=False, all_stride_variants=True)

    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @skipIfCrossRef
    @suppress_warnings
    # only test one dtype, as output stride behavior is the same for all dtypes
    @ops(op_db, dtypes=OpDTypes.any_common_cpu_cuda_one)
    # Only test on CUDA, as CUDA kernel's stride is the reference
    @onlyCUDA
    def test_dispatch_symbolic_meta_inplace_all_strides(self, device, dtype, op):
        self._run_dispatch_meta_test(device, dtype, op, symbolic_meta=True, inplace=True, all_stride_variants=True)


    def test_empty_quantized(self):
        r = torch.empty(2 ** 52, device='meta', dtype=torch.qint8)
        self.assertEqual(r.device.type, 'meta')

    def test_huber_loss_backward(self):
        inps = [torch.rand(2**52, device='meta') for _ in range(3)]
        r = torch.ops.aten.huber_loss_backward(*inps, 0, 1.0)
        self.assertEqual(r.device.type, 'meta')
        self.assertEqual(r.shape, inps[0].shape)

    def test_fill__alias_relationship(self):
        inps = torch.rand(2**52, device='meta')
        r = torch.ops.aten.fill_(inps, 1.0)
        # aten.fill_ returns an aliase
        self.assertEqual(id(inps), id(r))

        # aten.fill returns a new tensor
        r2 = torch.ops.aten.fill(inps, 1.0)
        self.assertNotEqual(id(inps), id(r2))

    def test_meta__fused_moving_avg_obs_fq_helper(self, device):
        from torch.ao.quantization import FusedMovingAvgObsFakeQuantize
        to_meta = MetaConverter()

        x = torch.randn(5, 5, device=device)
        running_min_op = torch.tensor(float("inf"), device=device)
        running_max_op = torch.tensor(float("-inf"), device=device)
        avg_const = 0.01
        scale = torch.tensor([1.0], device=device)
        zero_point = torch.tensor([0], dtype=torch.int, device=device)

        mod = FusedMovingAvgObsFakeQuantize()
        torch.ao.quantization.enable_fake_quant(mod)
        torch.ao.quantization.enable_observer(mod)
        mod.to(device)

        meta_x = to_meta(x)

        args = [
            x,
            mod.observer_enabled,
            mod.fake_quant_enabled,
            running_min_op,
            running_max_op,
            scale,
            zero_point,
            avg_const,
            0,
            255,
            0,
        ]

        meta_args = args.copy()
        meta_args[0] = meta_x

        kwargss = [
            {},
            {"per_row_fake_quant": False, "symmetric_quant": False},
            {"per_row_fake_quant": False, "symmetric_quant": True},
        ]

        for kwargs in kwargss:
            ref_out = aten._fused_moving_avg_obs_fq_helper.default(*args, **kwargs)
            meta_out = aten._fused_moving_avg_obs_fq_helper.default(*meta_args, **kwargs)

            self.assertEqual(ref_out[0].size(), meta_out[0].size())
            self.assertEqual(ref_out[0].stride(), meta_out[0].stride())
            self.assertEqual(ref_out[1].size(), meta_out[1].size())
            self.assertEqual(ref_out[1].stride(), meta_out[1].stride())

    # opinfo test is using aten.fill_, it's not testing aten.fill
    @onlyCUDA
    def test_fill_stride(self):
        to_meta = MetaConverter()
        sample_args = [torch.rand(2, 2, 2, 2), 1.0]

        for args in get_strided_args(sample_args):
            meta_args = to_meta(args)
            ref_out = torch.ops.aten.fill(*args)
            meta_out = torch.ops.aten.fill(*meta_args)
            self.assertEqual(ref_out.size(), meta_out.size())
            self.assertEqual(ref_out.stride(), meta_out.stride())


    def test_map_location_deserialize(self):
        import io

        t = torch.rand(10)
        b = io.BytesIO()

        torch.save(t, b)
        b.seek(0)
        r = torch.load(b, map_location=torch.device("meta"))
        self.assertEqual(r.device.type, 'meta')
        self.assertEqual(r.shape, t.shape)
        self.assertEqual(r.dtype, t.dtype)
        self.assertEqual(r.storage().data_ptr(), 0)

instantiate_device_type_tests(TestMeta, globals())

def print_op_str_if_not_supported(op_str):
    op = OperatorName.parse(op_str)
    packet = getattr(torch.ops.aten, str(op.name))
    overload = getattr(packet, op.overload_name if op.overload_name else "default")
    if any(overload in d for d in [meta_dispatch_skips, meta_dispatch_device_skips['cuda']]):
        print(f"{overload}  # SKIP")
    if any(overload in d for d in [meta_dispatch_expected_failures, meta_dispatch_device_expected_failures['cuda']]):
        print(overload)


if __name__ == "__main__":
    COMPARE_XLA = os.getenv('PYTORCH_COMPARE_XLA', None)
    if COMPARE_XLA is not None:
        with open(COMPARE_XLA, "r") as f:
            d = yaml.load(f, Loader=YamlLoader)
            ops = d.get("full_codegen", []) + d.get("supported", []) + d.get("autograd", [])
            for op_str in ops:
                print_op_str_if_not_supported(op_str)
        sys.exit(0)

    COMPARE_TEXT = os.getenv('PYTORCH_COMPARE_TEXT', None)
    if COMPARE_TEXT is not None:
        with open(COMPARE_TEXT, "r") as f:
            for op_str in f:
                print_op_str_if_not_supported(op_str.strip())
        sys.exit(0)

    run_tests()
