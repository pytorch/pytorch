import weakref

import torch
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._mode_utils import no_dispatch


def safe_is_leaf(t):
    try:
        return t.is_leaf
    except RuntimeError:
        # inference mode can trigger this
        return False


# torch.Tensors cannot be used as a key in a dictionary
# because they define a custom __eq__ function which when used
# to resolve hash collisions will throw when comparing tensors:
# "RuntimeError: bool value of Tensor with more than one value is ambiguous."
# To avoid that, we use an object which will hold a Tensor and use
# its id for both hashing and equality.
# In order to use this as a weak key reference, we cannot
# simply use weakref.WeakKeyDictionary because the newly constructed
# WeakTensorRefKey only use would be a dictionary so it would have no strong
# references.
# To get around this issue, we can use it as a normal key, and then set
# `weakref.finalize` to delete the key when its contained tensor dies.


class WeakTensorRefKey(object):
    def __init__(self, ten):
        self.ten = weakref.ref(ten)
        # store id since as soon as ten is deallocated
        # the old id will no longer be recoverable, and
        # we need to be able to remove the WeakTensorRefKey
        # from the dictionary by hashing it to the same
        # value it had when ten was alive
        self.id = id(self.ten())

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        if id(self) == id(other):
            return True
        return self.id == other.id


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
        self.tensor_memo = {}
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
        return self.tensor_memo.get(WeakTensorRefKey(t), None)

    def set_tensor_memo(self, t, v):
        # hold a weak ref to self, otherwise it will be kept alive
        # by the del_ten closure
        self_weak_ref = weakref.ref(self)
        if t.is_sparse:
            weak_st = None
        else:
            weak_st = StorageWeakRef(t.storage())
        tensor_ref_key = WeakTensorRefKey(t)

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
    def meta_storage(self, s):
        # NB: TypedStorage is freshly allocated and cannot be used as hash
        # key index.

        # Use a Weak Ref to s in order to not leak memory
        swr = StorageWeakRef(s)
        if swr not in self.storage_memo:
            self.storage_memo[swr] = torch.empty(s.size(), dtype=s.dtype, device="meta")
        return self.storage_memo[swr]

    # This function assumes that it's possible to do the conversion
    def meta_tensor(self, t, shape_env=None):
        arg_cnt = self.arg_cnt
        self.arg_cnt += 1

        # Don't make parameters have symbolic shapes; they are assumed to stay
        # constant size across training runs
        make_symbolic = shape_env is not None and not isinstance(t, torch.nn.Parameter)

        def sym(name, x):
            if make_symbolic:
                return shape_env.create_symint(f"t{arg_cnt}.{name}()", x)
            else:
                return x

        def sym_list(name, xs):
            if make_symbolic:
                return [
                    shape_env.create_symint(f"t{arg_cnt}.{name}({i})", x)
                    for i, x in enumerate(xs)
                ]
            else:
                return xs

        def sym_size(t):
            return sym_list("size", t.size())

        def sym_stride(t):
            return sym_list("stride", t.stride())

        # NB: Although sym_stride variables initially have no correlation
        # with size, we will immediately introduce guards based on contiguity.
        # Thus, if the input tensor is contiguous, the stride variables
        # will typically immediately get reexpressed in terms of the size
        # variables.

        def sym_storage_offset(t):
            return sym("storage_offset", t.storage_offset())

        # see expired-storages
        self.check_expired_count += 1
        if self.check_expired_count >= self.check_expired_frequency:
            self.check_for_expired_weak_storages()
            self.check_expired_count = 0

        if self.get_tensor_memo(t) is None:
            with torch.inference_mode(t.is_inference()):
                if t.is_sparse:
                    assert shape_env is None, "symbolic on sparse NYI"
                    is_leaf = safe_is_leaf(t)
                    r = torch.ops.aten._sparse_coo_tensor_with_dims(
                        t.sparse_dim(),
                        t.dense_dim(),
                        t.shape,
                        dtype=t.dtype,
                        layout=torch.sparse_coo,
                        device="meta",
                    )
                    r._coalesced_(t.is_coalesced())
                    if t.requires_grad:
                        r.requires_grad = True
                    if t.requires_grad and not is_leaf:
                        with torch.enable_grad():
                            r = r.clone()
                            r._coalesced_(t.is_coalesced())

                elif t._is_view():
                    # Construct views in two steps: recursively meta-fy their
                    # base, and then create the view off that.  NB: doing it
                    # directly from storage is WRONG because this won't cause
                    # version counters to get shared.
                    assert t._is_view()
                    base = self.meta_tensor(t._base)

                    def is_c_of_r(complex_dtype, real_dtype):
                        return (
                            utils.is_complex_dtype(complex_dtype)
                            and utils.corresponding_real_dtype(complex_dtype)
                            == real_dtype
                        )

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

                    with torch.enable_grad():
                        r = base.as_strided(
                            sym_size(t), sym_stride(t), sym_storage_offset(t)
                        )
                else:
                    is_leaf = safe_is_leaf(t)
                    # Fake up some autograd history.
                    if t.requires_grad:
                        r = torch.empty(
                            (0,), dtype=t.dtype, device="meta", requires_grad=True
                        )
                        if not is_leaf:
                            with torch.enable_grad():
                                # The backward function here will be wrong, but
                                # that's OK; our goal is just to get the metadata
                                # looking as close as possible; we're not going to
                                # actually try to backward() on these produced
                                # metas.  TODO: would be safer to install some
                                # sort of unsupported grad_fn here
                                r = r.clone()
                    else:
                        r = torch.empty((0,), dtype=t.dtype, device="meta")
                    # As long as meta storage is not supported, need to prevent
                    # redispatching on set_(Storage, ...) which will choke with
                    # meta storage
                    s = self.meta_storage(t.storage())
                    with no_dispatch():
                        with torch.no_grad():
                            r.set_(s, sym_storage_offset(t), sym_size(t), sym_stride(t))

                torch._C._set_conj(r, t.is_conj())
                torch._C._set_neg(r, t.is_neg())
            self.set_tensor_memo(t, r)

        return self.get_tensor_memo(t)

    def __call__(self, t, shape_env=None):
        # TODO: zero tensors?  We appear to have eliminated them by
        # excluding complex for now
        from torch._subclasses.fake_tensor import FakeTensor

        if (
            type(t) is torch.Tensor
            or type(t) is torch.nn.Parameter
            or isinstance(t, FakeTensor)
        ):
            if any(
                [
                    t.is_sparse_csr,
                    t.layout in [torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc],
                    t.is_mkldnn,
                    t.is_quantized,
                    t.is_nested,
                    t._is_view() and t._base is not None and t._base.is_sparse,
                    torch._is_functional_tensor(t),
                    # these are supported in meta conversion but the fallbacks
                    # don't work
                    t.is_neg(),
                    t.is_conj(),
                    t.device.type in ("lazy", "meta"),
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
                return t
            else:
                self.hit += 1
                r = self.meta_tensor(t, shape_env=shape_env)
                if type(t) is torch.nn.Parameter:
                    r = torch.nn.Parameter(r, requires_grad=r.requires_grad)
                return r
        elif torch.overrides.is_tensor_like(t):
            # Blindly converting tensor subclasses to meta can cause
            # unpredictable problems; e.g., FX tests will trace meta
            # tensors into their trace / some subclasses don't correctly
            # support meta.  Trying to YOLO this is more trouble than it's
            # worth.
            self.miss += 1
            return t
        else:
            # non-Tensor types don't count as hit or miss
            return t


import torch._prims_common as utils
