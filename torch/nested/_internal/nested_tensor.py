from typing import Tuple

import torch
from torch._C import DispatchKey, DispatchKeySet
from torch._prims_common import is_expandable_to
from torch.utils.weak import WeakTensorKeyDictionary
from typing import *  # noqa: F403
import weakref

def _get_nested_int(equiv_set, vec):
    return torch._C._get_nested_int(equiv_set, coeff=1, vec=vec)

class NestedIntRegistry():
    # Class to manage the association between 1d tensors (vec) and nested ints.
    # Use this class to (1) obtain nested int from vec (2) read/write metadata
    # associated with vecs/nested ints
    #
    # Pass a vec to maybe_create to create a new nested int. Multiple nested int
    # can be associated with a vec. vec are then grouped into equivalence sets
    # so that comparisons between the vec's nested ints can be made by comparing
    # their equiv set. Morally two vec are in the same equiv set if and only if
    # they have the same data, e.g., vec.cpu() and vec.cuda() belong to the same
    # equiv set even though they are on different devices because they have the
    # same data.
    #
    # The first vec and nested int associated with a equiv set becomes the
    # "canonical" vec and nested int for that equiv set respectively. This is
    # purely an implementation detail, e.g. whether a given NT's offsets is
    # canonical or not has no bearing on behavior. In fact, one should consider
    # all nested ints (given that coeff=1) and vec that are associated with the
    # same equiv set to be fungible when dealing with the registry.
    #
    # One consequence of having canonical vec is that vec pointed to by
    # nt.shape[ragged_dim].node.nested_int_vec() is not necessarily the same
    # Tensor as NT's ragged_source. This shouldn't matter though as per the
    # above.
    #
    # Example:
    # --------
    # In the diagram below, arrows represent the direction of ownership. j0_0
    # and j0_1 are in the same equiv set, but are different instances. vec0 and
    # j0_0 are the canonical vec and nested int for equiv_set0.
    #
    #  symint       vec         equiv_set       cached metadata
    # --------------------------------------------------------------------------
    #  2*j0 -----\                        /---- weak vec0, weak j0_0
    #             v                      | /--> id: int = 0
    #  j0_0 ------> vec0 -----> equiv_set0 ---> sum_vec: Optional[SymInt]
    #  j0_1 ------^         /-^            \--> vec_{cuda,cpu}: Optional[Tensor]
    #                      /
    #               vec1--/               /---- weak vec2, weak j1_0
    #                                    | /--> id: int = 1
    #  j1_0 ------> vec2 -----> equiv_set1 ---> ...
    #
    # Details on canonical vec and nested int:
    # ----------------------------------------
    # Throughout the lifetime of a equiv set, the canonical vec may change.
    # If we detect that the canonical vec is no longer alive, the next time
    # someone asks us for a nested int for a vec in that equiv set we will make
    # that vec the canonical vec. The canonical vec may be on cpu or cuda.
    # nt.shape[ragged_dim].vec() is not necessarily the same Tensor
    # Tensor as NT's ragged_source.
    #
    # If canonical vec and nested int are alive, nested int's vec is the
    # canonical vec. This implies that if canonical is not alive, then the
    # nested int is also not alive. It is possible for canonical to be alive,
    # but for nested int to be dead. This happens when the nested int is only
    # alive in cpp, but its corresponding python object has died.
    #
    def __init__(self):
        self._equiv_set_counter = 0
        self._equiv_sets = WeakTensorKeyDictionary()
        self._version_counters = WeakTensorKeyDictionary()

    def maybe_create(self, vec, ctor_fn=None, equiv_set_from=None):
        # Given vec, return an associated nested int
        mb_equiv_set = self._equiv_sets.get(vec)

        if mb_equiv_set is not None:
            # How can we create a nicer message for this?
            assert self._version_counters[vec] == vec._version, (
                "Detected that vec has been mutated. This is not allowed. "
            )
            ret = mb_equiv_set["weak_nested_int"]()
            mb_vec = mb_equiv_set["weak_vec"]()
            if ret is None:
                # PyObject preservation does not exist for SymInts, so if
                # shape is saved for backward, the SymInt in Python will be
                # collected unless kept alive elsewhere.
                assert ctor_fn is None
                assert not mb_equiv_set["with_ctor_fn"]
                assert equiv_set_from is None
                if mb_vec is None:
                    # Promote vec to canonical vec
                    mb_equiv_set["weak_vec"] = weakref.ref(vec)
                ret = _get_nested_int(mb_equiv_set["id"], mb_equiv_set["weak_vec"]())
                mb_equiv_set["weak_nested_int"] = weakref.ref(ret)
            else:
                assert equiv_set_from is None, (
                    "Expected equiv_set_from to be None if vec already has equiv_set"
                )
                assert mb_vec is not None, (
                    "Expected vec to be alive if nested int is alive"
                )
        else:
            if equiv_set_from is None:
                equiv_set_id = self._equiv_set_counter
                self._equiv_set_counter += 1
                _ctor_fn = ctor_fn if ctor_fn is not None else _get_nested_int
                ret = _ctor_fn(equiv_set_id, vec)
                self._equiv_sets[vec] = {
                    "id": equiv_set_id,
                    "weak_nested_int": weakref.ref(ret),
                    "weak_vec": weakref.ref(vec),
                    "with_ctor_fn": ctor_fn is not None,
                }
                self._version_counters[vec] = vec._version
            else:
                assert ctor_fn is None
                equiv_set = self._equiv_sets[equiv_set_from.node.nested_int_vec()]
                ret = equiv_set["weak_nested_int"]()
                assert ret is not None
                self._equiv_sets[vec] = equiv_set
                self._version_counters[vec] = vec._version
        return ret

    def maybe_set_metadata(self, vec, key, value):
        equiv_set = self._equiv_sets.get(vec)
        assert equiv_set is not None
        assert self._version_counters[vec] == vec._version, (
            "Detected that vec has been mutated. This is not allowed. "
        )
        if key not in equiv_set:
            equiv_set[key] = value

    def get_metadata(self, vec, key):
        # Check version counter?
        assert self._version_counters[vec] == vec._version, (
            "Detected that vec has been mutated. This is not allowed. "
        )
        equiv_set = self._equiv_sets.get(vec)
        assert equiv_set is not None
        return equiv_set[key]

_nested_int_registry: Optional[NestedIntRegistry] = None

def get_nested_int_registry() -> NestedIntRegistry:
    global _nested_int_registry
    if _nested_int_registry is None:
        _nested_int_registry = NestedIntRegistry()
    return _nested_int_registry

# SDPA metadata; max / min seqlens are needed for e.g. flash
def _get_sdpa_extreme_seqlen(func, tensor):
    return int(func(tensor).item())


class NestedTensor(torch.Tensor):
    _values: torch.Tensor  # type: ignore[assignment]
    _offsets: torch.Tensor
    _lengths: Optional[torch.Tensor]
    # NOTE [ Nested ints for ragged sizes and strides ]
    #
    # Jagged layout tensors are tensors that represent a n-dim tensor with a
    # ragged dimension, but are backed by an (n-1)-dim tensor underneath, e.g.,
    # a jagged tensor with outer shape [B, x, D] is represented internally by a
    # tensor with shape [sum(x), D] where we introduce what we call a nested int
    # denoted as "x" here (but sometimes denoted with "*" to
    # represent the ragged dimension, and sum(x) represents the dim of the inner
    # tensor or equivalently the sum of all the sizes of the constituent
    # tensors' varying lengths.
    #
    # We also use nested ints to represent the strides of this tensor.
    # For example, a jagged tensor with shape [B, x, D] can be strided in two
    # ways: [xD, D, 1] and [x, 1, sum(x)], where xD represents x multiplied by D
    _size: Tuple[int, ...]
    _stride: Tuple[int, ...]
    # Indicates that the nth dimension is ragged
    _ragged_idx: int
    _metadata_cache: Dict[str, Any]

    @staticmethod
    def __new__(
        cls,
        values,
        offsets,
        *,
        lengths=None,
        nested_int=None,
        **kwargs,
    ):
        ks = DispatchKeySet(DispatchKey.NestedTensor)
        ks = ks.add(DispatchKey.AutogradNestedTensor)
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            (0,),
            (0,),
            0,
            torch.contiguous_format,
            values.dtype,
            torch.jagged,
            values.device,
            False,
            kwargs.get("requires_grad", False),
            "sizes",
            False,
            True,  # dispatch_layout
            ks,
        )
        return r

    def __init__(self, values, offsets, *, lengths=None, _nested_int=None, **kwargs):
        super().__init__()
        # Only support jagged for now.
        assert offsets is not None
        assert offsets.ndim == 1
        assert not isinstance(values, NestedTensor)

        # Query cache for the symint associated with offsets or lengths
        # (create a new one if needed).
        ragged_source = offsets if lengths is None else lengths
        registry = get_nested_int_registry()
        if _nested_int is None:
            ragged_size = registry.maybe_create(ragged_source)
        else:
            ragged_size = _nested_int
        registry.maybe_set_metadata(ragged_source, "sum_vec", values.shape[0])
        self._ragged_idx = kwargs.get("_ragged_idx", 1)
        B = offsets.shape[0] - 1
        Ds = values.shape[: self._ragged_idx - 1] + values.shape[self._ragged_idx :]

        nested_size = [B]
        nested_size.extend(Ds[: self._ragged_idx - 1])
        nested_size.append(ragged_size)
        nested_size.extend(Ds[self._ragged_idx - 1 :])
        self._size = tuple(nested_size)

        stride = values.stride()
        self._strides = (ragged_size * stride[self._ragged_idx - 1], *stride)

        if values.requires_grad:
            raise ValueError(
                "NestedTensor values cannot require grad, please "
                "detach before passing to NestedTensor constructor"
            )
        self._values = values
        self._offsets = offsets
        self._lengths = lengths

        # holds properties that are computed lazily
        self._metadata_cache = kwargs.get("_metadata_cache") or {}

        # collapsed ragged dim must always be dynamic
        torch._dynamo.mark_dynamic(self, self._ragged_idx)
        torch._dynamo.mark_dynamic(self._values, self._ragged_idx - 1)

    def values(self):
        return self._values

    def offsets(self):
        return self._offsets

    def lengths(self):
        return self._lengths

    @property
    def _max_seqlen(self):
        if "max_seqlen" not in self._metadata_cache:
            # compute & cache
            self._metadata_cache["max_seqlen"] = _get_sdpa_extreme_seqlen(
                torch.max,
                self._offsets.diff() if self._lengths is None else self._lengths,
            )
        return self._metadata_cache["max_seqlen"]

    @property
    def _min_seqlen(self):
        if "min_seqlen" not in self._metadata_cache:
            # compute & cache
            self._metadata_cache["min_seqlen"] = _get_sdpa_extreme_seqlen(
                torch.min,
                self._offsets.diff() if self._lengths is None else self._lengths,
            )
        return self._metadata_cache["min_seqlen"]

    def __repr__(self):
        # We should implement this in torch/_tensor_str.py instead
        grad_fn_str = (
            f", requires_grad={self.requires_grad}" if self.requires_grad else ""
        )
        if self.grad_fn:
            grad_fn_str = f", grad_fn={self.grad_fn}"
        return f"NestedTensor(size={self._size}, offsets={self._offsets}{grad_fn_str}, contiguous={self._lengths is None})"

    def __reduce_ex__(self, proto):
        state = torch._utils._get_obj_state(self)

        # SymNodes are not serializable
        assert "_size" in state and "_strides" in state
        state = dict(state)
        del state["_size"]
        del state["_strides"]

        func = NestedTensor
        args = (self._values, self._offsets)
        return (torch._tensor._rebuild_from_type_v2, (func, type(self), args, state))

    def __tensor_flatten__(self):
        ctx = {
            "requires_grad": self.requires_grad,
            # TODO: Don't guard on this!
            "metadata_cache": self._metadata_cache,
            "ragged_idx": self._ragged_idx,
        }
        inner_tensors = ["_values", "_offsets"]
        if self._lengths is not None:
            inner_tensors.append("_lengths")
        return inner_tensors, ctx

    @staticmethod
    def __tensor_unflatten__(inner_tensors: Dict, meta, outer_size, outer_stride):
        assert len(inner_tensors) >= 2 and len(inner_tensors) <= 3
        values = inner_tensors["_values"]
        offsets = inner_tensors["_offsets"]
        lengths = inner_tensors.get("_lengths", None)
        ragged_idx = meta["ragged_idx"]

        vec = offsets if lengths is None else lengths
        nested_int = outer_size[ragged_idx]
        old_vec = nested_int.node.nested_int_vec()

        # During compilation, new symbolic nested int must be created via
        # operations on existing ones, so that guard sources are propagated
        # and the new symint is properly tracked by proxies. We reach here in
        # three cases, each is handled slightly differently:
        #
        # (1) During fakification, normal tensors are being swapped for fake
        #     tensors. maybe_create is a no-op because we have already
        #     registered the vec.
        # (2) If AOTAutograd is constructing grad_outputs by viewing the outputs
        #     create a new nested int and make sure they have the same equiv set
        # (3) If AOTAutograd swapping out FakeTensor for FunctionalTensor during
        #     functionalization. Create a new nested int with a new equiv set.
        #
        def ctor_fn(i, v):
            return torch.SymInt(nested_int.node.clone_nested_int_with_new_vec(i, v))
        kwargs = {
            "ctor_fn": ctor_fn,
            "equiv_set_from": nested_int if type(vec) == type(old_vec) else None,
        }
        nested_int = get_nested_int_registry().maybe_create(vec, **kwargs)

        return NestedTensor(
            values,
            offsets=offsets,
            lengths=lengths,
            requires_grad=meta["requires_grad"],
            _ragged_idx=ragged_idx,
            _metadata_cache=meta["metadata_cache"],
            # Pass in explicitly instead of relying on getting it from the
            # registry later because we are responsible for keeping alive what
            # maybe_create returns.
            _nested_int=nested_int,
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        # Lazy import to avoid circular dependency
        from .ops import lookup_jagged

        fn = lookup_jagged(func, *args, **kwargs)
        if fn is not None:
            return fn(*args, **kwargs)

        raise NotImplementedError(func)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        from .ops import jagged_torch_function

        try:
            return jagged_torch_function(func, *args, **kwargs)
        except NotImplementedError:
            pass
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)


# Not actually a view!
class ViewBufferFromNested(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: NestedTensor):  # type: ignore[override]
        ctx.save_for_backward(x.offsets())
        ctx.metadata_cache = x._metadata_cache
        ctx.ragged_idx = x._ragged_idx
        return x.values()

    @staticmethod
    def backward(ctx, gO: torch.Tensor):  # type: ignore[override]
        (offsets,) = ctx.saved_tensors
        return NestedTensor(
            gO,
            offsets=offsets,
            _metadata_cache=ctx.metadata_cache,
            _ragged_idx=ctx.ragged_idx,
        )


# Not actually a view!
class ViewNestedFromBuffer(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        values: torch.Tensor,
        offsets: torch.Tensor,
        metadata_cache: Optional[Dict[str, Any]] = None,
    ):  # type: ignore[override]
        return NestedTensor(
            values.detach(),
            offsets=offsets,
            _metadata_cache=metadata_cache,
        )

    @staticmethod
    def backward(ctx, gO: NestedTensor):  # type: ignore[override]
        return gO.values(), None, None


# Not actually a view!
# NOTE: @jbschlosser is working on making it a view
class ViewNonContiguousNestedFromBuffer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, offsets: torch.Tensor, lengths: torch.Tensor):  # type: ignore[override]
        return NestedTensor(
            values.detach(),
            offsets=offsets,
            lengths=lengths,
        )

    @staticmethod
    def backward(ctx, gO: NestedTensor):  # type: ignore[override]
        return gO.values(), None, None


# Need to make it obvious that users should be passing in offsets
def jagged_from_list(
    tensors: List[torch.Tensor],
    offsets: Optional[torch.Tensor],
    dtype=None,
    device=None,
) -> Tuple[NestedTensor, torch.Tensor]:
    """Constructs a NestedTensor backed by jagged layout from a list of tensors"""

    if not len(set(t.dtype for t in tensors)) == 1:  # noqa: C401
        raise RuntimeError(
            "When constructing a nested tensor, all tensors in list must have the same dtype"
        )
    if not len(set(t.device for t in tensors)) == 1:  # noqa: C401
        raise RuntimeError(
            "When constructing a nested tensor, all tensors in list must be on the same device"
        )

    # Check that the NT is representable by the jagged layout.
    # Jagged layout represents (B, *, D_0, D_1, ..., D_N), where the only
    # raggedness allowed is for the single dim immediately adjacent to the batch dim.
    sizes = [t.shape for t in tensors]
    non_first_sizes = [s[1:] for s in sizes]
    at_most_first_ragged = all(s == non_first_sizes[0] for s in non_first_sizes)
    if not at_most_first_ragged:
        raise RuntimeError(
            "Cannot represent given tensor list as a nested tensor with the jagged layout. "
            "Note that the jagged layout only represents shapes of the form "
            "(B, *, D_0, D_1, ..., D_N), with only * allowed to be ragged."
        )

    # Set properties appropriately.
    values = torch.cat(tensors, dim=0)
    to_kwargs = {}
    if device is not None:
        to_kwargs["device"] = device
    if dtype is not None:
        to_kwargs["dtype"] = dtype
    values = values.to(**to_kwargs)

    # Calculate jagged offsets if not provided.
    if offsets is None:
        # Jagged layout specifies that offsets are stored as int64 on the same device as values.
        offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.int64, device=values.device),
                torch.tensor([s[0] for s in sizes], device=values.device).cumsum(dim=0),
            ]
        )

    ret_nt = ViewNestedFromBuffer.apply(values, offsets)
    ret_nt._metadata_cache = {
        # compute this now since it's easy
        "max_seqlen": max([t.shape[0] for t in tensors]),
        "min_seqlen": min([t.shape[0] for t in tensors]),
    }
    return (ret_nt, offsets)  # type: ignore[return-value]


def jagged_from_tensor_and_lengths(
    tensor: torch.Tensor, starts: torch.Tensor, lengths: torch.Tensor
) -> Tuple[NestedTensor, torch.Tensor, Optional[torch.Tensor]]:
    """Constructs a NestedTensor backed by jagged layout from a tensor, starts of sequences, and sequence lengths"""
    batch_size = tensor.shape[0]
    if is_expandable_to(starts.shape, (batch_size,)) and is_expandable_to(
        lengths.shape, (batch_size,)
    ):
        start_list = starts.expand(batch_size)
        length_list = lengths.expand(batch_size)
    else:
        raise RuntimeError(
            "When constructing a jagged nested tensor using narrow(), "
            "your start and length must be Tensors that broadcast to input.shape[0]"
        )

    # Calculate jagged offsets
    assert (
        len(tensor.shape) >= 2
    ), "tensor must at least be 2D for the nested narrow op to work"
    max_seq_len = tensor.shape[1]
    offset_lengths = max_seq_len * torch.arange(
        0, batch_size, dtype=torch.int64, device=tensor.device
    )
    # Jagged layout specifies that offsets are stored as int64 on the same device as values.
    offsets = torch.cat(
        [
            start_list + offset_lengths,
            (start_list[-1] + offset_lengths[-1] + length_list[-1]).unsqueeze(0),
        ]
    )

    # Reshape buffer to flatten the 1st and 2nd dimension (view used to enforce non-copy)
    if len(tensor.shape) > 2:
        values = tensor.view(-1, *tensor.shape[2:])
    else:
        values = tensor.view(-1)

    # Check if offsets and lengths make it possibly contiguous and return a regular NT
    is_contiguous = True
    orig_dim = tensor.shape[1]
    if torch.any(length_list[1:-1].ne(orig_dim)):
        is_contiguous = False
    if torch.any(offsets[1:-2].diff().ne(orig_dim)):
        is_contiguous = False
    if offsets[0] + length_list[0] != orig_dim:
        is_contiguous = False

    actual_max_seqlen = int(torch.max(lengths).item())
    min_seqlen = int(torch.min(lengths).item())

    if is_contiguous:
        ret_nt = ViewNestedFromBuffer.apply(
            values[offsets[0] : offsets[-1]],
            offsets - offsets[0],
        )
    else:
        ret_nt = ViewNonContiguousNestedFromBuffer.apply(values, offsets, length_list)

    # populate metadata cache with computed seqlen extremes
    ret_nt._metadata_cache = {
        "max_seqlen": actual_max_seqlen,
        "min_seqlen": min_seqlen,
    }

    return (ret_nt, offsets, None if is_contiguous else length_list)


def buffer_from_jagged(jagged):
    return ViewBufferFromNested.apply(jagged)
