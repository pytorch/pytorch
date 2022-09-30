from torch.autograd.grad_mode import F
from typing import List, Union
import torch
import warnings
from torch._prims_common.wrappers import out_wrapper
import torch._prims as prims
from torch.overrides import has_torch_function_unary, handle_torch_function

# potentially primitives that can be used to implement indexing

prim_gather = prims.gather

# for each dimension specified apply the respective slice to that dimension
# if the slice is an integer, this means selecting a single value from that dimension, resulting
# in a dimension of size 1
# always returns a view
def prim_slice(tensor: torch.Tensor, dims: List[int], slices: List[Union[slice, int]]):
    # print(f"prim_slice {dims} {slices}")
    args = [slice(None)] * tensor.ndim
    for d, s in zip(dims, slices):
        if isinstance(s, slice):
            args[d] = s
        else:
            if s < 0:
                s += tensor.shape[d]
            if s < 0 or s >= tensor.shape[d]:
                raise IndexError(
                    f"index {s} is out of bounds for dimension {d} with size {tensor.shape[d]}"
                )
            args[d] = slice(
                s, s + 1
            )  # by taking integer at the primitive level, we avoid having to do
            # math on the numbers coming into the primtiive
    # return torch._C._TensorBase.__getitem__(tensor, tuple(args))
    r = tensor
    for i, a in enumerate(args):
        r = torch.ops.aten.slice.Tensor(r, i, a.start, a.stop, a.step if a.step is not None else 1)
    return r


# reorders a dimension similar to permute, with the following extensions:
# * if a dimension does not appear, it must have size 1 and it is dropped (similar to squeeze)
# * a None creates a new dimension of size 1


def prim_reorder(tensor, order: List[Union[int, None]]):
    # print(f"prim_reorder {order}")
    sz, st = tensor.shape, tensor.stride()
    for i in range(tensor.ndim):
        assert i in order or sz[i] == 1

    nsz = []
    nst = []
    for o in order:
        if o is not None:
            nsz.append(sz[o])
            nst.append(st[o])
        else:
            nsz.append(1)
            nst.append(0)
    return torch.as_strided(tensor, nsz, nst, tensor.storage_offset())


# as close as we can get...
def pysequence_check(obj):
    return not isinstance(obj, dict) and hasattr(type(obj), "__getitem__")


def treat_sequence_as_tuple(index):
    if isinstance(index, tuple):
        return True
    if isinstance(index, torch.Tensor):
        return False
    if not pysequence_check(index):
        return False

    # This uses a heuristics from NumPy for determining whether to treat
    # non-tuple sequences as if they were a tuple. From the NumPy code comments:
    try:
        n = len(index)
    except:
        return False
    if n >= 32:
        return False  # yes, really...

    for x in index:
        if (
            x is None
            or x is ...
            or isinstance(x, (torch.Tensor, slice))
            or pysequence_check(x)
        ):
            return True
    return False


# XXX: real logic is wrapTuple in c++, and is much more invovled...
def wrap_tuple(index):
    if treat_sequence_as_tuple(index):
        return tuple(index)
    else:
        return (index,)


empty_slice = slice(None, None, None)


def n_specified(idx):
    if idx is None or isinstance(idx, bool) or idx is ...:
        return 0
    if isinstance(idx, torch.Tensor):
        if idx.dtype is torch.uint8:
            warnings.warn(
                "indexing with dtype torch.uint8 is now deprecated, please use a dtype torch.bool instead."
            )
        if idx.dtype in (torch.bool, torch.uint8):
            return idx.ndim
        else:
            return 1
    return 1


def wrap_sequences(idx):
    if (
        not isinstance(idx, torch.Tensor)
        and pysequence_check(idx)
        or isinstance(idx, bool)
    ):
        idx = torch.tensor(idx)
        if idx.numel() == 0:
            idx = idx.to(torch.long)
    return idx


def __getitem__(self_, index_):
    # TODO: also check for torch function in index
    if has_torch_function_unary(self_):
        return handle_torch_function(__getitem__, (self_,), self_, index_)

    self = self_

    index = tuple(wrap_sequences(idx) for idx in wrap_tuple(index_))

    indices_specified = sum(n_specified(x) for x in index)

    to_pad = self.ndim - indices_specified
    if to_pad < 0:
        raise IndexError(
            f"invalid index: expected at most {self.ndim} dimensions but found {indices_specified}"
        )

    padding = tuple(empty_slice for _ in range(to_pad))
    idx = index.index(...) if ... in index else len(index)
    index = (*index[0:idx], *padding, *index[idx + 1 :])

    new_index = []

    # turn masking tensors in integer indexing tensors
    initial_reorder = list(range(self.ndim))
    i = 0  # which dim are we currently indexing
    for idx in index:
        if isinstance(idx, torch.Tensor) and idx.dtype in (torch.bool, torch.uint8):
            if idx.ndim == 0:
                initial_reorder.insert(i + (len(initial_reorder) - self.ndim), None)
                idx = prim_reorder(idx, [None])
            else:
                # check the sizes match...
                if tuple(idx.shape) != tuple(self.shape[i : i + idx.ndim]):
                    raise IndexError(
                        f"mask size {idx.shape} does not match tensor dimensions {self.shape[i:i+idx.ndim]}"
                    )
                i += idx.ndim
            new_index.extend(torch.nonzero(idx).unbind(dim=1))
        else:
            new_index.append(idx)
            if idx is not None:
                i += 1

    index = new_index

    if len(initial_reorder) > indices_specified:
        self = prim_reorder(self, initial_reorder)

    slice_dims = []
    slices = []

    gather_dims = []
    gather_tensors = []

    permute = []
    has_none = False

    offset = 0
    seen_output_dims = 0
    gather_insert_point = None
    for i, idx in enumerate(index):
        if idx is None:
            permute.append(None)
            has_none = True
            seen_output_dims += 1
            continue
        if isinstance(idx, (int, slice)):
            if idx != empty_slice:
                slice_dims.append(offset)
                slices.append(idx)
            if isinstance(idx, slice):
                permute.append(offset)
                seen_output_dims += 1
        elif isinstance(idx, torch.Tensor):
            gather_dims.append(offset)
            gather_tensors.append(idx)
            if gather_insert_point is None:
                gather_insert_point = seen_output_dims
            elif gather_insert_point != seen_output_dims:
                # XXX: not 100% sure this is the right logic
                gather_insert_point = 0
        else:
            msg = f"only integers, slices (`:`), ellipsis (`...`), None and long or byte Variables are valid indices (got {type(idx).__name__})"
            raise IndexError(msg)

        offset += 1

    if slice_dims:
        self = prim_slice(self, slice_dims, slices)

    gather_indices = []
    if gather_dims:
        try:
            gather_tensors = torch.broadcast_tensors(*gather_tensors)
        except RuntimeError as e:
            raise IndexError(f"shape mismatch: {e}")
        self = prim_gather(self, gather_dims, gather_tensors)
        ndim = gather_tensors[0].ndim
        for i in range(len(permute)):
            # TODO: ezyang: I'm not sure if this is right
            if permute[i] is not None:
                permute[i] += ndim
        gather_indices = list(range(ndim))
        permute[gather_insert_point:gather_insert_point] = gather_indices

    # this condition can be made cheaper
    needs_reorder = self.ndim != len(permute) or list(range(self.ndim)) != permute
    if needs_reorder:
        self = prim_reorder(self, permute)

    if self is self_:
        self = torch.ops.aten.alias(self)
    return self


def __setitem__(self_, index_, val):
    sliced = self_[index_]
    if (sliced.storage().data_ptr() != self_.storage().data_ptr()):
        raise RuntimeError("__getitem__ returned a copy!")
    sliced.copy_(val)

torch.Tensor.__getitem__ = __getitem__
torch.Tensor.__setitem__ = __setitem__


if __name__ == '__main__':
    t = torch.rand(3, 4, 5)
    t2 = prim_slice(t, [0, 2], [slice(1, None), 3])
    assert list(t2.shape) == [2, 4, 1]

    i = torch.arange(4)[:, None].expand(4, 5)
    j = torch.arange(5)[None, :].expand(4, 5)

    assert list(prim_gather(t, [1], [i]).shape) == [4, 5, 3, 1, 5]
    assert list(prim_gather(t, [1, 2], [i, j]).shape) == [4, 5, 3, 1, 1]

    t = torch.rand(2, 1, 5, 3)
    assert list(prim_reorder(t, [0, 3, 2, None]).shape) == [2, 3, 5, 1]
