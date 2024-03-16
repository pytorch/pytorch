import contextlib
import warnings
import weakref
from typing import ContextManager, Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
from torch._C._functorch import (
    _add_batch_dim,
    _unwrap_functional_tensor,
    _wrap_functional_tensor,
    current_level,
    get_unwrapped,
    is_batchedtensor,
    is_functorch_wrapped_tensor,
    is_gradtrackingtensor,
    maybe_get_bdim,
    maybe_get_level,
    peek_interpreter_stack,
    TransformType,
)
from torch._guards import Source

from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import (
    is_traceable_wrapper_subclass,
    transform_subclass,
)
from torch.utils.weak import WeakIdRef

if TYPE_CHECKING:
    # Import the following modules during type checking to enable code intelligence features,
    # Do not import unconditionally, as they import sympy and importing sympy is very slow
    from torch.fx.experimental.symbolic_shapes import SymbolicContext

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


def is_sparse_coo(t):
    return isinstance(t, torch.Tensor) and t.layout is torch.sparse_coo


def is_sparse_compressed(t):
    return isinstance(t, torch.Tensor) and t.layout in {
        torch.sparse_csr,
        torch.sparse_csc,
        torch.sparse_bsr,
        torch.sparse_bsc,
    }


def is_sparse_any(t):
    return is_sparse_coo(t) or is_sparse_compressed(t)


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
        if is_sparse_any(t) or t.is_mkldnn or is_functorch_wrapped_tensor(t):
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
        symbolic_context: Optional["SymbolicContext"] = None,
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

        def sym_sizes_strides_storage_offset(
            t, src, symbolic_context=symbolic_context
        ) -> Tuple[Tuple[int, ...], Tuple[int, ...], int]:
            if shape_env is not None:
                fake_mode = torch._subclasses.fake_tensor.maybe_get_fake_mode(t)
                if fake_mode is not None and fake_mode.shape_env is shape_env:
                    # Don't reallocate the sizes; the shape envs are the same,
                    # so reuse the old sizes/strides/etc
                    return (t.size(), t.stride(), t.storage_offset())
                else:
                    return shape_env.create_symbolic_sizes_strides_storage_offset(
                        t,
                        src,
                        symbolic_context=symbolic_context,
                    )
            else:
                assert symbolic_context is None
            return (t.size(), t.stride(), t.storage_offset())

        def empty_create(inner_t, inner_src, symbolic_context=symbolic_context):
            (
                inner_sizes,
                inner_strides,
                inner_storage_offset,
            ) = sym_sizes_strides_storage_offset(inner_t, inner_src, symbolic_context)
            return torch.empty_strided(
                inner_sizes,
                inner_strides,
                dtype=inner_t.dtype,
                device="meta",
            )

        # Creates a subclass instance with empty inner tensors according to the specified
        # symbolic context.
        def empty_create_subclass(
            t,
            outer_size,
            outer_stride,
            symbolic_context=symbolic_context,
            callback=callback,
            source=source,
        ):
            from torch._dynamo.source import AttrSource
            from torch.fx.experimental.symbolic_shapes import SubclassSymbolicContext

            assert symbolic_context is None or isinstance(
                symbolic_context, SubclassSymbolicContext
            )

            # Note: transform_subclass will use __tensor_unflatten__ to generate
            # a fresh subclass wrapper with outer sizes / strides according to the
            # outer symbolic context (passed in to this function). Inner size / stride
            # / storage offset symbols are allocated according to the appropriate inner
            # symbolic contexts, after which the checks in transform_subclass() will
            # relate them to the outer metadata as possible.
            return transform_subclass(
                t,
                lambda attr, inner_t: callback(
                    lambda: empty_create(
                        inner_t,
                        AttrSource(source, attr),
                        symbolic_context=(
                            None
                            if symbolic_context is None
                            else symbolic_context.inner_contexts[attr]
                        ),
                    )
                ),
                outer_size=outer_size,
                outer_stride=outer_stride,
            )

        # Returns an all-dynamic symbolic context used for metafying the given tensor with
        # fully dynamic dims. This is useful when fake-ifying intermediate tensors in
        # closed-over ViewFunc state, as we don't have symbolic contexts for them, but we
        # don't want to over-specialize during view replay.
        def all_dynamic_symbolic_context(t, source, shape_env, callback):
            from torch._dynamo.source import AttrSource
            from torch.fx.experimental.symbolic_shapes import (
                DimDynamic,
                StatelessSymbolicContext,
                SubclassSymbolicContext,
                SymbolicContext,
            )

            view_base_context: Optional[SymbolicContext] = None
            if t._is_view():
                view_base_context = all_dynamic_symbolic_context(
                    t._base, AttrSource(source, "_base"), shape_env, callback
                )

            t_symbolic_context: SymbolicContext
            t_dynamic_sizes = [DimDynamic.DYNAMIC] * t.dim()
            if is_traceable_wrapper_subclass(t):
                inner_contexts: Dict[str, SymbolicContext] = {}
                attrs, _ = t.__tensor_flatten__()
                for attr in attrs:
                    assert isinstance(attr, str)
                    inner = getattr(t, attr)
                    inner_contexts[attr] = all_dynamic_symbolic_context(
                        inner, AttrSource(source, attr), shape_env, callback
                    )
                t_symbolic_context = SubclassSymbolicContext(
                    dynamic_sizes=t_dynamic_sizes,
                    constraint_sizes=[None] * t.dim(),
                    inner_contexts=inner_contexts,
                    tensor_source=source,
                    view_base_context=view_base_context,
                )
            else:
                t_symbolic_context = StatelessSymbolicContext(
                    dynamic_sizes=t_dynamic_sizes,
                    constraint_sizes=[None] * t.dim(),
                    view_base_context=view_base_context,
                )

            return t_symbolic_context

        # Returns a fake-ified version of an input view tensor t, given an already fake-ified
        # base. At a high level, we want two things:
        #   1. fake_t should have the same view relationship to the given fake base as the
        #      input t has to its _base.
        #   2. fake_t should have symbolic sizes / strides / storage offset according to the
        #      appropriate symbolic context (i.e. from the automatic dynamic algorithm).
        #
        # We currently take different strategies across view types:
        #   * For dense -> dense views, accomplish both (1) and (2) simultaneously via an
        #     as_strided() call on the fake-ified base, passing symbolic metadata.
        #   * For views involving subclasses, perform view replay using view funcs to
        #     achieve (1). It's necessary for (2) to swap out any closed-over state in
        #     the view funcs with symbolicized SymInts and fake-ified tensors. Doing this
        #     avoids specialization (and thus over-eager simplification of symbols) that
        #     could occur during view replay on the fake-ified base.
        #
        # Examples:
        #   * t.unsqueeze(-1) with dense t is a dense -> dense view. It can be modeled
        #     with an as_strided() call on the fake base passing symbolic metadata.
        #   * sub.select(dim=0, index=3) is a subclass -> subclass view. The index arg
        #     is made symbolic to avoid invalid specialization and view replay is then
        #     done to reconstruct the view.
        #   * _nested_from_jagged(values, offsets) is a dense -> subclass view
        #     that returns a subclass instance from a dense values tensor. The offsets
        #     tensor is closed over in the view func, as it can be considered view metadata.
        #     First, the offsets tensor is fake-ified according to the inner symbolic
        #     context and with the correct relationship to the outer size / stride metadata.
        #     Then view replay is done, swapping in the fake offsets so the view replay output
        #     is fully fake with no invalid specialization.
        def view_from_base(base, t, source=source, shape_env=shape_env):
            # fake-ify t's metadata according to the outer symbolic context
            (sizes, strides, storage_offset) = sym_sizes_strides_storage_offset(
                t, source
            )
            if not is_traceable_wrapper_subclass(
                t
            ) and not is_traceable_wrapper_subclass(base):
                # Dense -> Dense view case uses as_strided() to construct view relationship.
                # TODO: Change this logic to use view replay for consistency?
                # It's likely there is no view func available.
                return base.as_strided(sizes, strides, storage_offset)

            from torch._dynamo.source import EphemeralSource
            from torch.fx.experimental.symbolic_shapes import sym_eq

            def symint_visitor_fn(s):
                if shape_env is None:
                    return s

                # NB: The symbol here is expected to be simplified out because we a priori
                # allocate inner and outer symbols according to the appropriate symbolic
                # contexts and prefer those over this symbol during symbol simplification
                # (via usage of EphemeralSource below). This -shouldn't- happen, but if
                # this symbol somehow leaks out beyond the view tensor's shape metadata, our
                # assumption of it being simplified out will fail and it may be guarded on,
                # which will hard error.
                sym_source = EphemeralSource("symint_visitor_fn")
                symbol = shape_env.create_symbol(s, sym_source)
                return shape_env.create_symintnode(symbol, hint=s, source=sym_source)

            real_to_fake_mapping = {}
            if is_traceable_wrapper_subclass(t):
                # Fake-ify t naively here; this is only done so we can get fake-ified inner
                # tensors with the correct relationships to the outer sizes / strides for use
                # in view replay. It's done beforehand here because it's not easy to do when
                # visiting tensors one-by-one during view replay.
                #
                # Example:
                #   Consider a Dense -> NJT view. NJT has (values, offsets) components and we
                #   want a view of values with the offsets closed over. As the offsets component
                #   is needed to describe the output view, it's important that it's fakeified
                #   correctly.
                fake_t = empty_create_subclass(
                    t, outer_size=sizes, outer_stride=strides
                )
                attrs, _ = fake_t.__tensor_flatten__()
                for attr in attrs:
                    real_to_fake_mapping[getattr(t, attr)] = getattr(fake_t, attr)

            def tensor_visitor_fn(
                visited_t, shape_env=shape_env, callback=callback, source=source
            ):
                # It's possible to close over an undefined tensor (e.g. NJT's lengths).
                if visited_t is None:
                    return None

                # Fake inner tensors of view subclasses will come from the mapping built above.
                fake_visited_t = real_to_fake_mapping.get(visited_t, None)
                if fake_visited_t is not None:
                    return fake_visited_t

                # For other closed-over tensor state, fake-ify it as all dynamic with an
                # ephemeral source. This avoids invalid specialization during view replay.
                # If we find that in practice the usage of ephemeral sources isn't enough
                # to guarantee that we don't have guards on these symbols, we may need to
                # explicitly suppress guards (as is done for _base in the dense -> dense
                # view case).
                temp_source = EphemeralSource("tensor_visitor_fn")
                return self.meta_tensor(
                    visited_t,
                    shape_env,
                    callback,
                    source=temp_source,
                    symbolic_context=all_dynamic_symbolic_context(
                        visited_t, temp_source, shape_env, callback
                    ),
                )

            # Replay the view, swapping out any non-symbolic SymInts or real tensors
            # for symbolic SymInts or fake tensors.
            fake_t = t._view_func_unsafe(base, symint_visitor_fn, tensor_visitor_fn)

            # Ensure the output has symbolic shapes according to the outer symbolic context.
            # These checks should simplify out any symbols created for closed-over view func
            # SymInts.
            torch._check(sym_eq(fake_t.size(), sizes))
            torch._check(sym_eq(fake_t.stride(), strides))
            torch._check(sym_eq(fake_t.storage_offset(), storage_offset))
            return fake_t

        # see expired-storages
        self.check_expired_count += 1
        if self.check_expired_count >= self.check_expired_frequency:
            self.check_for_expired_weak_storages()
            self.check_expired_count = 0

        if self.get_tensor_memo(t) is None:
            with torch.inference_mode(t.is_inference()):
                if t.is_sparse:
                    is_leaf = safe_is_leaf(t)

                    # The lambda function below is similar to
                    # `t.to(device='meta')` except the latter
                    # preserves nnz value
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
                elif is_sparse_compressed(t):
                    is_leaf = safe_is_leaf(t)

                    def mk_meta():
                        nnz = 0
                        batch_dim = t.ndim - t.sparse_dim() - t.dense_dim()
                        batch_size = t.shape[:batch_dim]
                        if t.layout in {torch.sparse_csr, torch.sparse_bsr}:
                            index_dtype = t.crow_indices().dtype
                            compressed_indices = torch.empty(
                                t.crow_indices().shape, device="meta", dtype=index_dtype
                            )
                            plain_indices = torch.empty(
                                (*t.col_indices().shape[:-1], nnz),
                                device="meta",
                                dtype=index_dtype,
                            )
                        else:
                            index_dtype = t.ccol_indices().dtype
                            compressed_indices = torch.empty(
                                t.ccol_indices().shape, device="meta", dtype=index_dtype
                            )
                            plain_indices = torch.empty(
                                (*t.row_indices().shape[:-1], nnz),
                                device="meta",
                                dtype=index_dtype,
                            )
                        values_shape = t.values().shape
                        values = torch.empty(
                            (
                                *values_shape[:batch_dim],
                                nnz,
                                *values_shape[batch_dim + 1 :],
                            ),
                            dtype=t.dtype,
                            device="meta",
                        )
                        return torch.ops.aten.sparse_compressed_tensor(
                            compressed_indices,
                            plain_indices,
                            values,
                            t.shape,
                            layout=t.layout,
                            dtype=t.dtype,
                            device="meta",
                        )

                    # `mk_meta()` is similar to `t.to(device='meta'))`
                    # except `to('meta')` preserves nnz value while
                    # `mk_meta` result has nnz == 0.
                    r = callback(mk_meta)

                    assert safe_is_leaf(r), "the callback you passed in doesn't detach"
                    if t.requires_grad:
                        r.requires_grad = True
                    if t.requires_grad and not is_leaf:
                        with torch.enable_grad():
                            r = r.clone()
                elif t.is_nested and not is_traceable_wrapper_subclass(t):
                    # TODO: Handle this better in Dynamo?
                    # There are checks there now, but this can still be triggered by a dense
                    # tensor graph input that is a view of a strided NT.
                    from torch._dynamo.exc import unimplemented

                    unimplemented(
                        "strided nested tensors are not supported by meta conversion"
                    )
                elif t.is_mkldnn:
                    is_leaf = safe_is_leaf(t)
                    sizes, strides, _storage_offset = sym_sizes_strides_storage_offset(
                        t, source
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
                elif is_functorch_wrapped_tensor(t):
                    if t._is_view():
                        from torch._dynamo.exc import unimplemented

                        unimplemented(
                            "view functorch tensors are not supported by meta conversion"
                        )

                    # Wraps a functorch tensor class (BatchedTensor, GradTrackingTensor)
                    # in a FakeTensor
                    def _to_fake_tensor(t):
                        if is_batchedtensor(t):
                            ft = _to_fake_tensor(get_unwrapped(t))
                            lvl = maybe_get_level(t)
                            bdim = maybe_get_bdim(t)
                            r = _add_batch_dim(ft, bdim, lvl)
                        elif is_gradtrackingtensor(t):
                            disable_functorch = torch._C._DisableFuncTorch
                            with disable_functorch():
                                ft = _to_fake_tensor(get_unwrapped(t))
                            lvl = torch._C._functorch.maybe_get_level(t)
                            r = torch._C._functorch._wrap_for_grad(ft, lvl)

                            is_leaf = safe_is_leaf(t)
                            if t.requires_grad and safe_is_leaf(r):
                                r.requires_grad = True
                            elif t.requires_grad and not is_leaf:
                                with torch.enable_grad():
                                    r = r.clone()
                        else:
                            sizes = t.size()
                            strides = t.stride()
                            r = callback(
                                lambda: torch.empty_strided(
                                    sizes,
                                    strides,
                                    dtype=t.dtype,
                                    device="meta",
                                )
                            )
                        return r

                    r = _to_fake_tensor(t)

                elif t._is_view():
                    # Construct views in two steps: recursively meta-fy their
                    # base, and then create view(s) off that.  NB: doing it
                    # directly from storage is WRONG because this won't cause
                    # version counters to get shared.
                    assert t._is_view()

                    base_symbolic_context = None
                    if shape_env and symbolic_context is not None:
                        from torch.fx.experimental.symbolic_shapes import (
                            StatelessSymbolicContext,
                        )

                        assert isinstance(symbolic_context, StatelessSymbolicContext)
                        # NB: This should generally be set when the input is a view,
                        # but the exception right now is for fake-ifying grads, which is
                        # a work in progress.
                        if symbolic_context.view_base_context is not None:
                            base_symbolic_context = symbolic_context.view_base_context

                    base = self.meta_tensor(
                        t._base,
                        shape_env,
                        callback,
                        source=torch._dynamo.source.AttrSource(source, "_base"),
                        symbolic_context=base_symbolic_context,
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
                        if safe_is_leaf(t):
                            # Leaf views that track view metadata are created by
                            # creating a view inside a no_grad block
                            with torch.no_grad(), maybe_suppress():
                                r = view_from_base(base, t)
                            # As it's a leaf, we can directly assign requires_grad
                            r.requires_grad = t.requires_grad
                        else:
                            if t._base.requires_grad == t.requires_grad:
                                # Easy case, just run the view op
                                with torch.enable_grad(), maybe_suppress():
                                    r = view_from_base(base, t)

                                # NB: We don't actaully faithfully replicate
                                # autograd connectivity, but that doesn't matter
                                # today. See following for more info:
                                # https://gist.github.com/soulitzer/e03f015b314c3f5fcf80888c69390913
                            else:
                                # Obscure case.  Create a leaf view and give it the
                                # correct requires_grad, then do the final view.
                                # NB: Can't have a non-leaf without requiring grad!
                                assert t.requires_grad
                                with torch.no_grad():
                                    mid = base.view(base.shape)
                                mid.requires_grad = t.requires_grad
                                with torch.enable_grad(), maybe_suppress():
                                    r = view_from_base(mid, t)
                        # The CreationMeta influences whether or not inplace
                        # mutation is an error or not.  So we need to make
                        # sure we properly propagate this as well.
                        torch._C._autograd._set_creation_meta(
                            r, torch._C._autograd._get_creation_meta(t)
                        )
                    finally:
                        torch._C._dispatch_tls_set_dispatch_key_excluded(
                            torch._C.DispatchKey.ADInplaceOrView, old_exclude
                        )

                else:
                    is_leaf = safe_is_leaf(t)

                    (
                        sizes,
                        strides,
                        storage_offset,
                    ) = sym_sizes_strides_storage_offset(t, source, symbolic_context)

                    # If we have a subclass that desugars into dense tensors,
                    # perform our callback on each inner tensor.
                    if is_traceable_wrapper_subclass(t):
                        r = empty_create_subclass(
                            t, outer_size=sizes, outer_stride=strides
                        )
                    else:
                        r = callback(
                            lambda: torch.empty_strided(
                                sizes,
                                strides,
                                dtype=t.dtype,
                                device="meta",
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

                    # Graph-Break for wrapped tensors
                    if not (
                        is_batchedtensor(t) or is_gradtrackingtensor(t)
                    ) and torch._C._functorch.is_functorch_wrapped_tensor(t):
                        return NotImplemented

                    s = t.untyped_storage()
                    swr = StorageWeakRef(s)
                    if swr not in self.storage_memo and (
                        r.is_nested
                        or (
                            r.stride() == strides
                            and r.storage_offset() == storage_offset
                        )
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
                            in_kernel_invocation_manager,
                            maybe_get_fake_mode,
                        )

                        mb_fake_mode = maybe_get_fake_mode(r)
                        if mb_fake_mode is not None:
                            maybe_fake_mgr = in_kernel_invocation_manager(mb_fake_mode)
                        with maybe_fake_mgr, torch.no_grad():
                            r.set_(r_s, storage_offset, sizes, strides)

                if safe_grad(t) is not None:
                    from torch._dynamo.source import AttrSource

                    # TODO: Use a valid grad-specific symbolic context instead of recycling
                    # the one from t. This isn't correct if e.g. t._is_view() != t.grad._is_view().
                    r.grad = self.meta_tensor(
                        safe_grad(t),
                        shape_env,
                        callback,
                        source=AttrSource(source, "grad"),
                        symbolic_context=symbolic_context,
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
        source=None,
        symbolic_context=None,
    ):
        # TODO: zero tensors?  We appear to have eliminated them by
        # excluding complex for now

        if isinstance(t, torch.Tensor) or is_traceable_wrapper_subclass(t):
            if t.device.type != "xla" and any(
                [
                    t.is_quantized,
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

                if torch._is_functional_tensor(t) and t.device.type != "lazy":
                    if t._is_view():
                        raise RuntimeError(
                            "Cannot safely fakify a view because this process drops the view information right now."
                        )

                    st = peek_interpreter_stack()
                    assert (
                        st is None or st.key() == TransformType.Functionalize
                    ), "Expect st to be either None or have Functionalize transform key."
                    if st is None:
                        # the case of AOTAutograd
                        torch._sync(t)
                        unwrap_t = torch._from_functional_tensor(t)
                        with torch._dispatch.python.suspend_functionalization():
                            fake_t = self.meta_tensor(
                                unwrap_t,
                                shape_env=shape_env,
                                callback=callback,
                                source=source,
                                symbolic_context=symbolic_context,
                            )
                        out = torch._to_functional_tensor(fake_t)
                        torch._mirror_autograd_meta_to(fake_t, out)
                        return out
                    else:
                        # torch.func.functionalize
                        reapply_views = torch._C._functionalization_reapply_views_tls()
                        unwrap_t = _unwrap_functional_tensor(t, reapply_views)
                        pop_st_ctx = (
                            torch._functorch.pyfunctorch.temporarily_pop_interpreter_stack()
                        )
                        with pop_st_ctx:
                            fake_t = self.meta_tensor(
                                unwrap_t,
                                shape_env=shape_env,
                                callback=callback,
                                source=source,
                                symbolic_context=symbolic_context,
                            )
                        return _wrap_functional_tensor(fake_t, current_level())
                self.miss += 1
                return NotImplemented
            else:
                self.hit += 1

                disable_functorch = torch._C._DisableFuncTorch
                with disable_functorch():
                    r = self.meta_tensor(
                        t,
                        shape_env=shape_env,
                        callback=callback,
                        source=source,
                        symbolic_context=symbolic_context,
                    )
                if type(t) is torch.nn.Parameter:
                    # NB: Cannot directly use Parameter constructor
                    # because that would force a detach, not desirable
                    r._is_param = True
                return r
        elif torch.overrides.is_tensor_like(t):
            self.miss += 1
            return NotImplemented
        else:
            # non-Tensor types don't count as hit or miss
            return t


import torch._prims_common as utils
