import contextlib
import warnings
import weakref
from typing import ContextManager, List, Optional

import torch
from torch._guards import Source
from torch.fx.experimental.symbolic_shapes import DimConstraint, DimDynamic
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils.weak import WeakIdRef

DimList = List


def safe_is_leaf(t):
    try:
        return t.is_leaf
    except RuntimeError:
        # inference mode can trigger this
        return False


def safe_grad(t):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "The .grad attribute of a Tensor")
        return t.grad


def assert_eq(a, b):
    assert a == b, f"{a} != {b}"


def assert_metadata_eq(assert_eq, m1, m2, *, skip_symbolic=False):
    def go(m1, m2):
        assert_eq(m1.dtype, m2.dtype)
        if not skip_symbolic:
            assert_eq(m1.shape, m2.shape)
        assert_eq(m1.requires_grad, m2.requires_grad)
        assert_eq(m1.is_leaf, m2.is_leaf)
        assert_eq(m1.grad_fn is None, m2.grad_fn is None)
        assert_eq(m1.is_sparse, m2.is_sparse)
        assert_eq(m1.is_inference(), m2.is_inference())
        assert_eq(m1.is_conj(), m2.is_conj())
        assert_eq(m1.is_neg(), m2.is_neg())
        assert_eq(safe_grad(m1) is not None, safe_grad(m2) is not None)
        if safe_grad(m1) is not None:
            go(safe_grad(m1), safe_grad(m2))
        if m1.is_sparse:
            assert_eq(m1.dense_dim(), m2.dense_dim())
            assert_eq(m1.sparse_dim(), m2.sparse_dim())
            assert_eq(m1.is_coalesced(), m2.is_coalesced())
        else:
            if not skip_symbolic:
                assert_eq(m1.stride(), m2.stride())
                assert_eq(m1.storage_offset(), m2.storage_offset())
            assert_eq(m1._is_view(), m2._is_view())
            if m1._is_view():
                go(m1._base, m2._base)
        # TODO: test if is resizable (no direct query for this atm)
        # TODO: audit AutogradMeta to see if it matches
        # TODO: test forward AD

    return go(m1, m2)


# This is a class for converting multiple tensors into meta tensors which
# share the same view/storage structure.  The operation model is you allocate
# one of these, and then call it repeatedly on all the tensors you want to
# convert.  It's important to use the same object for tensors you want to
# share storage because this is how we correlate shared storages to the same
# meta storages. This class will hold weak references to cached tenosrs
# and tensor storages.
class MetaConverter:
    def __init__(self):
        self.storage_memo = {}
        self.tensor_memo: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        self.maybe_storages_to_delete = []
        self.check_expired_frequency = 128
        self.check_expired_count = 0
        self.hit = 0
        self.miss = 0
        self.del_hook = None
        self.arg_cnt = 0

    def successful(self):
        return self.hit > 0 and self.miss == 0

    def check_for_expired_weak_storages(self):
        new_li = []
        stor_to_delete = []
        for obj in self.maybe_storages_to_delete:
            if not obj.expired():
                new_li.append(obj)
            else:
                stor_to_delete.append(obj)
        for obj in stor_to_delete:
            self.storage_memo.pop(obj, None)
        self.maybe_storages_to_delete = new_li

        # if for some reason we have aquired many storages which have not expired
        # even though a tensor with their storage has expired (aliasing or otherwise)
        # check for expired storages less often so as to bound the amount of work we
        # do checking for expired storages
        self.check_expired_frequency = max(
            self.check_expired_frequency, len(self.maybe_storages_to_delete)
        )

    def get_tensor_memo(self, t):
        return self.tensor_memo.get(WeakIdRef(t), None)

    def set_tensor_memo(self, t, v):
        # hold a weak ref to self, otherwise it will be kept alive
        # by the del_ten closure
        self_weak_ref = weakref.ref(self)
        if t.is_sparse or t.is_mkldnn:
            weak_st = None
        else:
            weak_st = StorageWeakRef(t._typed_storage())
        tensor_ref_key = WeakIdRef(t)

        def del_ten():
            # tensor outlives the converter
            self_ref = self_weak_ref()
            if self_ref is None:
                return
            # on shutdown, tensor_ref_key may not be in memo
            self_ref.tensor_memo.pop(tensor_ref_key, None)
            if weak_st and weak_st.expired():
                self_ref.storage_memo.pop(weak_st, None)
            elif weak_st is not None:
                # [expired-storages]
                # NB: even though the tensor has died,
                # the deallocation of its storage can take longer,
                # even when the storage has no other uses/views.
                # In this case, the StorageWeakRef object will be kept alive
                # longer than it needs to be, however the storage itself
                # will be deallocated. We retain the possibly dead storages
                # and periodically check if any of them are expired and
                # can be freed.
                self_ref.maybe_storages_to_delete.append(weak_st)

        weakref.finalize(t, del_ten)
        self.tensor_memo[tensor_ref_key] = v

    # NB: doesn't actually return a storage, because meta storage is
    # not supported
    def meta_storage(self, s, callback):
        # NB: TypedStorage is freshly allocated and cannot be used as hash
        # key index.

        # Use a Weak Ref to s in order to not leak memory
        swr = StorageWeakRef(s)
        if swr not in self.storage_memo:
            self.storage_memo[swr] = callback(
                lambda: torch.empty(s.size(), dtype=torch.uint8, device="meta")
            ).untyped_storage()
        return self.storage_memo[swr]

    # This function assumes that it's possible to do the conversion
    # NB: name here is used in a conventional way by Dynamo; it corresponds
    # precisely to the Source.name() of the tensor we're fakeifying and
    # corresponds to a valid Python expression.  When we construct sub-names
    # as part of this process, we will maintain this invariant!  (Even though
    # other users of this may not need it this property to be upheld.)
    def meta_tensor(
        self,
        t,
        shape_env=None,
        callback=lambda t: t(),
        source: Optional[Source] = None,
        dynamic_dims: Optional[DimList[DimDynamic]] = None,
        constraint_dims: Optional[DimList[DimConstraint]] = None,
    ):
        if source is None:
            from torch._dynamo.source import ConstantSource

            # TODO: make a dedicated UnknownSource for this?
            source = ConstantSource(
                f"__meta_utils_unknown_tensor{len(self.tensor_memo)}"
            )

        # This indicates you set no_dispatch() before calling into this
        # function.  This is an error: we may be creating fake tensors and
        # will perform operations on them which need fake tensor mode to
        # be active.  You will segfault if you are in a no_dispatch() block.
        assert not torch._C._dispatch_tls_local_exclude_set().has(
            torch._C.DispatchKey.Python
        )
        arg_cnt = self.arg_cnt
        self.arg_cnt += 1

        # When we make as_strided calls, we end up generating a guard
        # that the new as_strided tensor is in bounds for the old storage
        # for the base (since as_strided calls can "bust" out of their
        # bounding box.)  This guard is unnecessary: if a user is able
        # to provide us a tensor with the view base setup this way, we
        # don't need to produce a guard, because the fact that they
        # were able to produce the view base means its in bounds.
        #
        # Now, ordinarily, this guard would be harmless.  However, the
        # generated guard refers to variables bound on the base variable.
        # At the moment, Dynamo doesn't actually guard on x._base, because
        # according to Voz this results in a lot of spurious invalidations,
        # and also if the user doesn't directly make use of _base, its
        # pointless anyway (because programs should be parametric over
        # whether or not the input tensor is a view or not--unless you're
        # mutating the input, but that's a whole 'nother ballgame).  So
        # for expediency, we suppress these guards so we don't have to
        # deal with this (yet, anyway.)
        #
        # NB: An old version of this code suppressed guards for ALL operations
        # happening during meta conversion, not just as_strided calls.
        # This is too aggressive: we do duck sizing and 0/1 simplification
        # as we allocate variables, and we do need to register guards for
        # these cases.
        maybe_suppress = contextlib.nullcontext
        if shape_env is not None:
            maybe_suppress = shape_env.suppress_guards

        def sym_sizes_strides_storage_offset(t):
            if shape_env is not None:
                return shape_env.create_symbolic_sizes_strides_storage_offset(
                    t,
                    source,
                    dynamic_dims=dynamic_dims,
                    constraint_dims=constraint_dims,
                )
            else:
                assert dynamic_dims is None
                assert constraint_dims is None
            return (t.size(), t.stride(), t.storage_offset())

        # see expired-storages
        self.check_expired_count += 1
        if self.check_expired_count >= self.check_expired_frequency:
            self.check_for_expired_weak_storages()
            self.check_expired_count = 0

        if self.get_tensor_memo(t) is None:
            with torch.inference_mode(t.is_inference()):
                if t.is_sparse:
                    # TODO: Delete this assert, and just attempt making the
                    # sparse tensor anyway; even if there is a shape_env, this
                    # tensor might be all static
                    assert shape_env is None, "symbolic on sparse NYI"
                    is_leaf = safe_is_leaf(t)
                    r = callback(
                        lambda: torch.ops.aten._sparse_coo_tensor_with_dims(
                            t.sparse_dim(),
                            t.dense_dim(),
                            t.shape,
                            dtype=t.dtype,
                            layout=torch.sparse_coo,
                            device="meta",
                        )
                    )
                    assert safe_is_leaf(r), "the callback you passed in doesn't detach"
                    # Note [is_coalesced is dispatched]
                    # Strangely enough, is_coalesced() is a dispatched operator,
                    # which means that it will get caught by fake tensor mode.
                    # Ordinarily this would error, but there's some logic in
                    # fake tensor ensure this doesn't happen.
                    r._coalesced_(t.is_coalesced())
                    if t.requires_grad:
                        r.requires_grad = True
                    if t.requires_grad and not is_leaf:
                        with torch.enable_grad():
                            r = r.clone()
                            r._coalesced_(t.is_coalesced())
                elif t.is_mkldnn:
                    is_leaf = safe_is_leaf(t)
                    sizes, strides, _storage_offset = sym_sizes_strides_storage_offset(
                        t
                    )
                    r = callback(
                        lambda: torch.empty_strided(
                            sizes, strides, dtype=t.dtype, device="meta"
                        )
                    )
                    assert safe_is_leaf(r), "the callback you passed in doesn't detach"
                    if t.requires_grad:
                        r.requires_grad = True
                    if t.requires_grad and not is_leaf:
                        with torch.enable_grad():
                            r = r.clone()
                elif t._is_view():
                    # Construct views in two steps: recursively meta-fy their
                    # base, and then create view(s) off that.  NB: doing it
                    # directly from storage is WRONG because this won't cause
                    # version counters to get shared.
                    assert t._is_view()

                    from torch._dynamo.source import AttrSource

                    if shape_env:
                        base_dynamic_dims = [DimDynamic.STATIC] * t._base.dim()
                    else:
                        base_dynamic_dims = None
                    base = self.meta_tensor(
                        t._base,
                        shape_env,
                        callback,
                        source=AttrSource(source, "_base"),
                        dynamic_dims=base_dynamic_dims,
                    )

                    def is_c_of_r(complex_dtype, real_dtype):
                        return (
                            utils.is_complex_dtype(complex_dtype)
                            and utils.corresponding_real_dtype(complex_dtype)
                            == real_dtype
                        )

                    # In some situations, MetaConverter may be called in a
                    # context where autograd is disabled.  For the _is_view
                    # assert to pass, we have to setup the autograd view
                    # metadata anyway.  Do this by reenabling the
                    # ADInplaceOrView key.  This is kind of a hack.
                    old_exclude = torch._C._dispatch_tls_is_dispatch_key_excluded(
                        torch._C.DispatchKey.ADInplaceOrView
                    )
                    torch._C._dispatch_tls_set_dispatch_key_excluded(
                        torch._C.DispatchKey.ADInplaceOrView, False
                    )
                    try:
                        if base.dtype == t.dtype:
                            pass
                        elif is_c_of_r(base.dtype, t.dtype):
                            base = torch.view_as_real(base)
                        elif is_c_of_r(t.dtype, base.dtype):
                            base = torch.view_as_complex(base)
                        else:
                            # This is not guaranteed to succeed.  If it fails, it
                            # means there is another dtype-converting view function
                            # that hasn't been handled here
                            base = base.view(t.dtype)

                        # This is very tricky.  Naively, you might expect this
                        # to hold:
                        #
                        #   if t.requires_grad and not safe_is_leaf(t)
                        #       assert t._base.requires_grad
                        #
                        # But it's not true!  As you can see in the following
                        # program:
                        #
                        #   x = torch.zeros(4)
                        #   y = x.view(1, 4)
                        #   y.requires_grad = True
                        #   z = y.view(1, 1, 4)
                        #   assert z._base is x
                        #
                        # So we may have to do *two* views out of the base to
                        # recreate this situation.

                        (
                            sizes,
                            strides,
                            storage_offset,
                        ) = sym_sizes_strides_storage_offset(t)

                        if safe_is_leaf(t):
                            # Leaf views that track view metadata are created by
                            # creating a view inside a no_grad block
                            with torch.no_grad(), maybe_suppress():
                                r = base.as_strided(sizes, strides, storage_offset)
                            # As it's a leaf, we can directly assign requires_grad
                            r.requires_grad = t.requires_grad
                        else:
                            if t._base.requires_grad == t.requires_grad:
                                # Easy case, just run the view op
                                with torch.enable_grad(), maybe_suppress():
                                    r = base.as_strided(sizes, strides, storage_offset)
                            else:
                                # Obscure case.  Create a leaf view and give it the
                                # correct requires_grad, then do the final view.
                                # NB: Can't have a non-leaf without requiring grad!
                                assert t.requires_grad
                                with torch.no_grad():
                                    mid = base.view(base.shape)
                                mid.requires_grad = t.requires_grad
                                with torch.enable_grad(), maybe_suppress():
                                    r = mid.as_strided(sizes, strides, storage_offset)
                    finally:
                        torch._C._dispatch_tls_set_dispatch_key_excluded(
                            torch._C.DispatchKey.ADInplaceOrView, old_exclude
                        )

                else:
                    is_leaf = safe_is_leaf(t)
                    sizes, strides, storage_offset = sym_sizes_strides_storage_offset(t)
                    r = callback(
                        lambda: torch.empty_strided(
                            sizes, strides, dtype=t.dtype, device="meta"
                        )
                    )
                    assert safe_is_leaf(r), "the callback you passed in doesn't detach"
                    if t.requires_grad:
                        r.requires_grad = t.requires_grad
                        if not is_leaf:
                            # Fake up some autograd history.
                            with torch.enable_grad():
                                # preserve_format is the default, but we want to
                                # emphasize how important it is to preserve
                                # format here
                                r = r.clone(memory_format=torch.preserve_format)

                    s = t.untyped_storage()
                    swr = StorageWeakRef(s)
                    if (
                        swr not in self.storage_memo
                        and r.stride() == strides
                        and r.storage_offset() == storage_offset
                    ):
                        # You're normal and happy, install the fresh storage into the memo
                        self.storage_memo[swr] = r.untyped_storage()
                    else:
                        # You're in crazy town; somehow you gave us a tensor
                        # that wasn't a view, but had nonzero storage offset,
                        # nontrivial strides (such that clone() couldn't
                        # preserve them), or already aliases with another
                        # tensor's storage.  The most typical way to end
                        # up here is with set_.  So use set_ to bludgeon this
                        # in.
                        r_s = self.meta_storage(s, callback=callback)
                        # NB: In principle, this should always work, but there
                        # is some subtle difference in the autograd metadata
                        # that means we will backprop the set_ call, even if
                        # r is declared as an input to grad.
                        # See https://github.com/pytorch/pytorch/issues/87956
                        # for the reproducer.
                        # NB: The in_kernel_invocation_manager here is necessary
                        # for fake tensor.  If we run the set_ call with fake
                        # tensor on, r will improperly report that it is NOT a
                        # meta tensor but a cpu tensor, and then the set_ call
                        # will fail due to device mismatch.  no_dispatch() is
                        # not enough, because the fake tensor will still claim
                        # to be a CPU tensor and you'll end up in the CPU
                        # kernel.  Arguably this is a hack; a cleaner way to
                        # solve this is to have a FakeStorage concept which
                        # would report it's CPU device--no problem now!  But
                        # this is difficult to do because we don't have storage
                        # subclasses.  Relevant test is
                        # DynamicShapesFunctionTests::test_add_dynamic_shapes in
                        # test/dynamo/test_dynamic_shapes.py
                        maybe_fake_mgr: ContextManager[None] = contextlib.nullcontext()
                        from torch._subclasses.fake_tensor import (
                            FakeTensor,
                            in_kernel_invocation_manager,
                        )

                        if isinstance(r, FakeTensor):
                            maybe_fake_mgr = in_kernel_invocation_manager(r.fake_mode)
                        with maybe_fake_mgr, torch.no_grad():
                            r.set_(r_s, storage_offset, sizes, strides)

                if safe_grad(t) is not None:
                    from torch._dynamo.source import AttrSource

                    r.grad = self.meta_tensor(
                        safe_grad(t),
                        shape_env,
                        callback,
                        source=AttrSource(source, "grad"),
                        dynamic_dims=dynamic_dims,
                        constraint_dims=constraint_dims,
                    )
                torch._C._set_conj(r, t.is_conj())
                torch._C._set_neg(r, t.is_neg())
            # This can be skipped if necessary for performance reasons
            assert_metadata_eq(assert_eq, t, r, skip_symbolic=True)
            self.set_tensor_memo(t, r)

        return self.get_tensor_memo(t)

    def __call__(
        self,
        t,
        shape_env=None,
        *,
        callback=lambda t: t(),
        ignore_subclass=False,
        source=None,
        dynamic_dims=None,
        constraint_dims=None,
    ):
        # TODO: zero tensors?  We appear to have eliminated them by
        # excluding complex for now
        from torch._subclasses.fake_tensor import FakeTensor

        if (
            type(t) is torch.Tensor
            or type(t) is torch.nn.Parameter
            or (ignore_subclass and isinstance(t, torch.Tensor))
            or isinstance(t, FakeTensor)
        ):
            if t.device.type != "xla" and any(
                [
                    t.is_sparse_csr,
                    t.layout in [torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc],
                    t.is_quantized,
                    t.is_nested,
                    t._is_view() and t._base is not None and t._base.is_sparse,
                    torch._is_functional_tensor(t),
                    t.device.type in ("lazy"),
                    # We need a way to test if a tensor is batched but there
                    # is no official APi to do it
                    # torch._C._is_batched(t),
                ]
            ):
                # TODO: sparse should support meta
                # NB technically to('meta') does work but our logging
                # instrumentation will see the meta conversions and the
                # tests all break so we just exclude this.  In any case
                # the to conversion isn't really right anyhow.
                self.miss += 1
                return NotImplemented
            else:
                self.hit += 1
                # When ignoring subclasses, we treat the input tensor "as if" it
                # were a normal tensor and create a non-subclassed fake tensor
                # that, modulo type and attributes, resembles the original tensor.
                # This can be helpful if you're planning to simulate the subclassness
                # by hand, e.g., as is done in Dynamo
                ctx = contextlib.nullcontext()
                if ignore_subclass:
                    ctx = torch._C.DisableTorchFunctionSubclass()
                with ctx:
                    r = self.meta_tensor(
                        t,
                        shape_env=shape_env,
                        callback=callback,
                        source=source,
                        dynamic_dims=dynamic_dims,
                        constraint_dims=constraint_dims,
                    )
                if type(t) is torch.nn.Parameter:
                    # NB: Cannot directly use Parameter constructor
                    # because that would force a detach, not desirable
                    r._is_param = True
                return r
        elif torch.overrides.is_tensor_like(t):
            # Blindly converting tensor subclasses to meta can cause
            # unpredictable problems; e.g., FX tests will trace meta
            # tensors into their trace / some subclasses don't correctly
            # support meta.  Trying to YOLO this is more trouble than it's
            # worth.
            self.miss += 1
            return NotImplemented
        else:
            # non-Tensor types don't count as hit or miss
            return t


import torch._prims_common as utils
