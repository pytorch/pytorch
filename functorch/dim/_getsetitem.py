from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING, Union

import torch
from ._dim_entry import _match_levels, DimEntry
from ._tensor_info import TensorInfo


if TYPE_CHECKING:
    from . import Dim


def _safe_index(lst: list, item: Any) -> Optional[int]:
    """
    Helper function to find index of item in list.

    For DimEntry objects, uses __eq__ comparison which properly handles
    both positional and Dim entries.

    Returns the index if found, None if not found.
    """
    for i, list_item in enumerate(lst):
        # Use == for DimEntry objects as they have proper __eq__ implementation
        if isinstance(item, DimEntry) and isinstance(list_item, DimEntry):
            if list_item == item:
                return i
        elif list_item is item:
            return i
    return None


@dataclass
class IndexingInfo:
    can_call_original: bool = False
    advanced_indexing: bool = False
    self_tensor: Optional[torch.Tensor] = None
    flat_inputs: list[Any] = field(default_factory=list)
    result_levels: list[DimEntry] = field(default_factory=list)
    has_device: bool = False


def has_dims(obj: Any) -> bool:
    """
    Check if an object has first-class dimensions.

    This function checks if the object is either a Dim or a functorch Tensor
    that has first-class dimensions, using the proper check_exact methods.
    """
    from . import Dim, Tensor

    return Dim.check_exact(obj) or Tensor.check_exact(obj)


def _bind_dims_to_size(sz: int, sd: int, dims: list, nsz: list, nsd: list) -> None:
    """
    Bind dimensions to size and calculate proper strides for dim packs.
    """
    from . import DimensionBindError

    rhs_prod = 1
    for i, dim in enumerate(dims):
        if not dim.is_bound:
            # Check for multiple unbound dimensions
            for j in range(i + 1, len(dims)):
                if not dims[j].is_bound:
                    raise DimensionBindError(
                        f"cannot infer the sizes of two dimensions at once {dim!r} and {dims[j]!r}"
                    )
                rhs_prod *= dims[j].size

            # Calculate the size for this unbound dimension
            if sz % rhs_prod != 0:
                tup = tuple(dim.size if dim.is_bound else "?" for dim in dims)
                raise DimensionBindError(
                    f"inferred dimension does not evenly fit into larger dimension: {sz} vs {tup}"
                )

            inferred_size = sz // rhs_prod
            dim.size = inferred_size
            rhs_prod = sz
            break
        else:
            rhs_prod *= dim.size

    # Final validation that dimensions match
    if rhs_prod != sz:
        tup = tuple(dims)
        raise DimensionBindError(
            f"Dimension sizes to do not match ({sz} != {rhs_prod}) when matching dimension pack {tup}"
        )

    # Calculate new sizes and strides for each dimension in the pack
    # First calculate all strides by iterating in reverse
    new_strides = [0] * len(dims)
    current_stride = sd
    for i in reversed(range(len(dims))):
        new_strides[i] = current_stride
        current_stride *= dims[i].size

    # Then append sizes and strides in forward order
    for i in range(len(dims)):
        nsz.append(dims[i].size)
        nsd.append(new_strides[i])


def slice_to_tuple(flat_inputs: list) -> tuple:
    return tuple(flat_inputs)


def extractIndices(index: Any, indices: list) -> bool:
    if isinstance(index, tuple):  # mpy::tuple_view::check
        indices.extend(index)
        return True
    elif isinstance(index, torch.Tensor):  # THPVariable_Check
        indices.append(index)
        return False
    elif not hasattr(index, "__iter__") or isinstance(
        index, (str, bytes)
    ):  # !mpy::is_sequence
        indices.append(index)
        return False

    # Handle sequence case (list)
    if isinstance(index, list):
        if len(index) >= 32:
            indices.extend(index)
            return True

        # Check each item in the sequence
        for item in index:
            if (
                isinstance(item, (torch.Tensor, slice))
                or hasattr(item, "__iter__")
                or item is ...
                or item is None
                or has_dims(item)
            ):
                indices.extend(index)
                return True

        # If we got here, treat as single index
        indices.append(index)
        return False

    # Default case
    indices.append(index)
    return False


def getitem(cls: Any, func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    self = args[0]
    index = args[1]

    iinfo = getsetitem(self, index, has_dims(self))
    if iinfo.can_call_original:
        # Call original tensor __getitem__ directly, bypassing __torch_function__
        return torch.Tensor.__getitem__(self, index)

    return invoke_getitem(iinfo)


def setitem(self: Any, index: Any, rhs: Any) -> None:
    """Set values in tensor using first-class dimensions."""
    from . import DimensionBindError, TensorInfo

    iinfo = getsetitem(self, index, has_dims(self) or has_dims(rhs))

    if iinfo.can_call_original:
        # Call original tensor __setitem__ directly, bypassing __torch_function__
        torch._C.TensorBase.__setitem__(self, index, rhs)
        return

    # Handle RHS tensor with dimensions
    rhs_info = TensorInfo.create(rhs, False, False)

    if rhs_info:
        # Check that rhs dimensions are compatible with result dimensions
        for l in rhs_info.levels:
            if not l.is_positional():
                # Find this dimension in result levels
                found = False
                for result_level in iinfo.result_levels:
                    if (
                        not result_level.is_positional()
                        and result_level.dim() is l.dim()
                    ):
                        found = True
                        break

                if not found:
                    # Create tuple representation of result levels for error message
                    result_dims: list[Union[int, Dim]] = []
                    for rl in iinfo.result_levels:
                        if rl.is_positional():
                            result_dims.append(rl.position())
                        else:
                            result_dims.append(rl.dim())

                    raise DimensionBindError(
                        f"rhs of setitem contains dimension {l.dim()!r} which is not in the dimension on the left "
                        f"({tuple(result_dims)!r})"
                    )

        # Match RHS tensor to result levels
        assert rhs_info.tensor is not None, "Cannot match levels on None tensor"
        matched_rhs = _match_levels(
            rhs_info.tensor, rhs_info.levels, iinfo.result_levels
        )
    else:
        matched_rhs = rhs

    # For advanced indexing with dimensions, we need special handling
    if iinfo.advanced_indexing:
        # Use advanced indexing - the flat_inputs already contain matched tensors
        tup = slice_to_tuple(iinfo.flat_inputs)
        if iinfo.self_tensor is None:
            raise RuntimeError("Cannot setitem on None tensor")
        torch._C.TensorBase.__setitem__(iinfo.self_tensor, tup, matched_rhs)
    else:
        # Simple copy operation
        if iinfo.self_tensor is None:
            raise RuntimeError("Cannot copy to None tensor")
        iinfo.self_tensor.copy_(matched_rhs)


def invoke_getitem(iinfo: IndexingInfo) -> Any:
    if iinfo.advanced_indexing:
        self_tensor = iinfo.self_tensor
        tup = slice_to_tuple(iinfo.flat_inputs)
        if self_tensor is None:
            raise RuntimeError("Cannot getitem on None tensor")
        rtensor = self_tensor[tup]
    else:
        rtensor = iinfo.self_tensor  # type: ignore[assignment]
        if rtensor is None:
            raise RuntimeError("Cannot getitem on None tensor")
        # rtensor is now guaranteed to be not None

    # Create a Tensor with the proper dimensions using the class method
    from . import Tensor

    return Tensor.from_positional(rtensor, iinfo.result_levels, iinfo.has_device)


def getsetitem(self: Any, index: Any, tensors_have_dims: bool) -> IndexingInfo:
    from . import DimList  # Import DimList for type checking

    can_call_original_getitem = not tensors_have_dims

    input_list = []
    if has_dims(index):
        input_list.append(index)
    else:
        is_sequence = extractIndices(index, input_list)
        # nothing about first class dims here, fallback to getitem
        if can_call_original_getitem and not is_sequence:
            return IndexingInfo(can_call_original=True)

    # Calculate how many dimensions have been indexed in order to compute the
    # size of ... or expand a potentially unbound dimension list.
    dims_indexed = 0
    expanding_object = -1
    unbound_dim_list = None
    dimlists = []  # Track DimList positions for later processing

    def check_expanding(i: int) -> None:
        nonlocal expanding_object
        if expanding_object != -1:
            from . import DimensionBindError

            raise DimensionBindError(
                f"at most one ... or unbound dimension list can exist in indexing list but found 2 at offsets "
                f"{expanding_object} and {i}"
            )
        expanding_object = i

    def is_dimpack(s: Any) -> bool:
        from . import Dim

        return (
            isinstance(s, (tuple, list))
            and len(s) > 0
            and all(Dim.check_exact(item) for item in s)
        )

    has_dimpacks_or_none = False
    for i, s in enumerate(input_list):
        if has_dims(s):
            can_call_original_getitem = False
            dims_indexed += 1
        elif s is ...:
            check_expanding(i)
        elif isinstance(s, DimList):
            can_call_original_getitem = False
            if not s.is_bound:
                check_expanding(i)
                unbound_dim_list = s
            else:
                dims_indexed += len(s._dims)
            dimlists.append(i)
        elif s is None:
            has_dimpacks_or_none = True
        elif is_dimpack(s):
            can_call_original_getitem = False
            has_dimpacks_or_none = True
            dims_indexed += 1
        else:
            dims_indexed += 1

    # Early return if we can use original getitem
    if can_call_original_getitem:
        return IndexingInfo(can_call_original=True)

    self_info = TensorInfo.create(self, False, True)
    total_dims = len(self_info.levels)  # Total dimensions (positional + named)
    if dims_indexed > total_dims:
        raise ValueError(
            f"at least {dims_indexed} indices were supplied but the tensor only has {total_dims} dimensions"
        )

    # Expand any unbound dimension list, or expand ... into individual : slices.
    expanding_dims = total_dims - dims_indexed
    if expanding_object != -1:
        if unbound_dim_list is not None:
            # Bind unbound dimension list to the expanding dimensions
            unbound_dim_list.bind_len(expanding_dims)
        else:
            # Expand ... into slice(None) objects
            no_slices = [slice(None)] * expanding_dims
            input_list = (
                input_list[:expanding_object]
                + no_slices
                + input_list[expanding_object + 1 :]
            )

    # Flatten out any dimensions stored in dimlist elements directly into the inputs
    # Process in reverse order to maintain indices
    for i in range(len(dimlists) - 1, -1, -1):
        idx = dimlists[i]

        # We added more elements to input because of ...
        # so we need to also adjust the index to get back to where the
        # dimlist existed
        if (
            unbound_dim_list is None
            and expanding_object != -1
            and idx > expanding_object
        ):
            idx += expanding_dims

        dl = input_list[idx]

        # PRIVATE here naughty
        input_list = input_list[:idx] + dl._dims + input_list[idx + 1 :]

    return getsetitem_flat(self_info, input_list, [], [], has_dimpacks_or_none)


def getsetitem_flat(
    self_info: TensorInfo,
    input_list: list,
    keys: list[DimEntry],
    values: list,
    has_dimpacks_or_none: bool,
) -> IndexingInfo:
    from . import Dim

    # Track dimension usage
    seen_dims: list[Any] = []
    seen_dims_nuses: list[int] = []

    def add_dim(dim: Any) -> None:
        # Use safe indexing to avoid triggering __torch_function__ on Dim objects
        idx = _safe_index(seen_dims, dim)
        if idx is not None:
            seen_dims_nuses[idx] += 1
        else:
            seen_dims.append(dim)
            seen_dims_nuses.append(1)

    flat_inputs = []
    tensor_inputs: list[Any] = []
    device_holding_tensor = None

    def append_flat_handle(handle: Any) -> None:
        flat_inputs.append(handle)
        tensor_inputs.append(None)

    def append_tensor_input(ti: TensorInfo) -> None:
        flat_inputs.append(None)
        tensor_inputs.append(ti)
        nonlocal device_holding_tensor
        if ti.has_device and device_holding_tensor is None:
            device_holding_tensor = ti.tensor

    nsz = []
    nsd = []
    if self_info.tensor is None:
        raise RuntimeError("Cannot get size/stride on None tensor")
    sz = self_info.tensor.size()
    sd = self_info.tensor.stride()

    def append_size(i: int) -> None:
        if has_dimpacks_or_none:
            nsz.append(sz[i])
            nsd.append(sd[i])

    input_it = input_list[:]

    def parse_nones() -> None:
        nonlocal input_it
        while input_it and input_it[0] is None:
            append_flat_handle(slice(None))
            nsz.append(1)
            nsd.append(0)
            input_it = input_it[1:]

    def append_item(i: int, arg: Any) -> None:
        if Dim.check_exact(arg):
            d = arg
            if d._size == -1:
                d.size = sz[i]
            add_dim(d)
            append_size(i)
            append_flat_handle(arg)
            return

        info = TensorInfo.create(arg, False, False)
        if info:
            append_size(i)
            append_tensor_input(info)
            for level in info.levels:
                if not level.is_positional():
                    add_dim(level.dim())
            return

        if has_dimpacks_or_none:
            if isinstance(arg, (tuple, list)) and all(Dim.check_exact(d) for d in arg):
                # dim pack
                dim_pack = list(arg)
                for d in dim_pack:
                    add_dim(d)
                    append_flat_handle(d)
                _bind_dims_to_size(sz[i], sd[i], dim_pack, nsz, nsd)
                return

        append_size(i)
        append_flat_handle(arg)

    # Match indexing expressions with tensor dimensions
    for i, level in enumerate(self_info.levels):
        # Use safe indexing to avoid triggering __torch_function__ on DimEntry comparisons
        idx = _safe_index(keys, level)
        if idx is not None:
            append_item(i, values[idx])
        else:
            if level.is_positional():
                parse_nones()
                if not input_it:
                    append_flat_handle(slice(None))
                    append_size(i)
                else:
                    arg = input_it[0]
                    input_it = input_it[1:]
                    append_item(i, arg)
            else:
                add_dim(level.dim())
                append_flat_handle(level.dim())
                append_size(i)

    parse_nones()

    # Restride tensor if needed
    if has_dimpacks_or_none and nsz:
        if self_info.tensor is None:
            raise RuntimeError("Cannot restride None tensor")
        self_tensor = self_info.tensor.as_strided(
            nsz, nsd, self_info.tensor.storage_offset()
        )
    else:
        self_tensor = self_info.tensor

    # Determine result shape and indexing requirements
    result_levels: list[Any] = []
    index_levels = []
    tensor_insert_point = -1
    requires_getindex = False

    def mark_tensor_index() -> None:
        nonlocal tensor_insert_point
        if tensor_insert_point == -1:
            tensor_insert_point = len(result_levels)
        elif tensor_insert_point != len(result_levels):
            tensor_insert_point = 0

    for i, inp in enumerate(flat_inputs):
        if tensor_inputs[i] is not None:
            requires_getindex = True
            mark_tensor_index()
            for level in tensor_inputs[i].levels:
                if level not in index_levels:
                    index_levels.append(level)
        elif Dim.check_exact(inp):
            d = inp
            # Use safe indexing to avoid triggering __torch_function__
            dim_idx = _safe_index(seen_dims, d)
            assert dim_idx is not None, f"Dim {d} not found in seen_dims"
            if seen_dims_nuses[dim_idx] == 1:
                flat_inputs[i] = slice(None)
                result_levels.append(DimEntry(d))
            else:
                requires_getindex = True
                flat_inputs[i] = None
                tensor_inputs[i] = TensorInfo(
                    d._get_range(), [DimEntry(d)], False, None
                )
                if DimEntry(d) not in index_levels:
                    index_levels.append(DimEntry(d))
                mark_tensor_index()
        else:
            if inp != slice(None):
                requires_getindex = True
            if not isinstance(inp, int):
                result_levels.append(DimEntry(-1))

    # Insert indexing dimensions at first tensor use point
    if tensor_insert_point != -1:
        for level in reversed(index_levels):
            result_levels.insert(tensor_insert_point, level)

    # Match tensors to indexing shape
    if requires_getindex:
        for i in range(len(flat_inputs)):
            if tensor_inputs[i] is not None:
                t = tensor_inputs[i].tensor
                assert t is not None, "TensorInfo should have valid tensor data"
                if (
                    not tensor_inputs[i].has_device
                    and device_holding_tensor is not None
                ):
                    t = t.to(device_holding_tensor.device)
                flat_inputs[i] = _match_levels(t, tensor_inputs[i].levels, index_levels)

    # Number positional dimensions correctly
    seen_positionals = 0
    for i in reversed(range(len(result_levels))):
        if result_levels[i].is_positional():
            seen_positionals += 1
            result_levels[i] = DimEntry(-seen_positionals)

    return IndexingInfo(
        can_call_original=False,
        advanced_indexing=requires_getindex,
        self_tensor=self_tensor,
        flat_inputs=flat_inputs,
        result_levels=result_levels,
        has_device=self_info.has_device,
    )
