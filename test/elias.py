import torch
import weakref
import gc

from torch.testing._internal.logging_tensor import no_dispatch

def safe_is_leaf(t):
    try:
        return t.is_leaf
    except RuntimeError:
        # inference mode can trigger this
        return False


class MetaConverter:
    def __init__(self):
        self.storage_memo = {}
        # Getting error when using weakref.WeakKeyDictionary directly
        self.tensor_memo = {}
        self.hit = 0
        self.miss = 0
        self.collected = 0

    def successful(self):
        return self.hit > 0 and self.miss == 0

    # NB: doesn't actually return a storage, because meta storage is
    # not supported
    def meta_storage(self, s):
        # NB: TypedStorage is freshly allocated and cannot be used as hash
        # key index.
        if s._cdata not in self.storage_memo:
            self.storage_memo[s._cdata] = torch.empty(s.size(), dtype=s.dtype, device='meta')
        return self.storage_memo[s._cdata]

    def remove_dead_weakref_key(self, key):
        self.collected += 1
        print("IN FINALIZE")
        print(self.tensor_memo)
        del self.tensor_memo[key]
        print("IN FINALIZE2")
        print(self.tensor_memo)


    # This function assumes that it's possible to do the conversion
    def meta_tensor(self, t):
        if weakref.ref(t) not in self.tensor_memo:
            with torch.inference_mode(t.is_inference()):
                if t._is_view():
                    # Construct views in two steps: recursively meta-fy their
                    # base, and then create the view off that.  NB: doing it
                    # directly from storage is WRONG because this won't cause
                    # version counters to get shared.
                    assert t._is_view()
                    base = self.meta_tensor(t._base)

                    def is_c_of_r(complex_dtype, real_dtype):
                        return is_complex_dtype(complex_dtype) and \
                            corresponding_real_dtype(complex_dtype) == real_dtype

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
                        r = base.as_strided(t.size(), t.stride(), t.storage_offset())
                else:
                    is_leaf = safe_is_leaf(t)
                    # Fake up some autograd history.
                    if t.requires_grad:
                        r = torch.empty((0,), dtype=t.dtype, device='meta', requires_grad=True)
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
                        r = torch.empty((0,), dtype=t.dtype, device='meta')
                    # As long as meta storage is not supported, need to prevent
                    # redispatching on set_(Storage, ...) which will choke with
                    # meta storage
                    s = self.meta_storage(t.storage())
                    with no_dispatch():
                        with torch.no_grad():
                            r.set_(s, t.storage_offset(), t.size(), t.stride())

                torch._C._set_conj(r, t.is_conj())
                torch._C._set_neg(r, t.is_neg())

            ref = weakref.ref(t)
            self.tensor_memo[ref] = r
            weakref.finalize(r, MetaConverter.remove_dead_weakref_key, self, ref)

        return self.tensor_memo[weakref.ref(t)]

    def __call__(self, t):
        # TODO: zero tensors?  We appear to have eliminated them by
        # excluding complex for now
        if type(t) is torch.Tensor or type(t) is torch.nn.Parameter:
            if any([
                t.is_sparse_csr, t.is_sparse, t.is_mkldnn, t.is_quantized,
                t.is_nested, torch._is_functional_tensor(t),
                # these are supported in meta conversion but the fallbacks
                # don't work
                t.is_neg(), t.is_conj(),
                # conjugate fallback does not support meta tensors
                t.dtype in (torch.complex128, torch.complex64),
            ]):
                # TODO: sparse should support meta
                # NB technically to('meta') does work but our logging
                # instrumentation will see the meta conversions and the
                # tests all break so we just exclude this.  In any case
                # the to conversion isn't really right anyhow.
                self.miss += 1
                return t
            elif any([
                t.device.type in ("lazy", "meta"), t.is_complex(),
                # We need a way to test if a tensor is batched but there
                # is no official APi to do it
                # torch._C._is_batched(t),
            ]):
                # TODO: this stuff should support storage
                # (well, maybe not batched)
                self.hit += 1
                return t.to("meta")
            else:
                self.hit += 1
                r = self.meta_tensor(t)
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

x = torch.rand([20, 20])
# y = weakref.ref(x)
# out = {}
# out[y] =  2
# weakref.finalize(x, print, y)
# del x
# print("here")
# import pdb; pdb.set_trace()
# out = weakref.ref(x)
# import pdb; pdb.set_trace()
conv = MetaConverter()
out = conv(x)
out2 = conv(x)
# # y = x[0]
del x
gc.collect()
print("hi")
print(len(conv.tensor_memo))
print(len(conv.tensor_memo))
print(len(conv.tensor_memo))
gc.collect()

gc.collect()
print(len(conv.tensor_memo))
print(len(conv.tensor_memo))
print(len(conv.tensor_memo))

while conv.collected == 0:
    pass
# assert out is out2
# # outy = conv(y)
# import pdb; pdb.set_trace()
# print(conv)
# # assert len(conv.tensor_memo) == 0


