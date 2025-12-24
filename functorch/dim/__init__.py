from __future__ import annotations

import dis
import inspect
import sys
from typing import Any, Optional, TYPE_CHECKING, Union


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

import torch
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from ._dim_entry import _match_levels, DimEntry, ndim_of_levels
from ._enable_all_layers import EnableAllLayers
from ._py_inst_decoder import _PyInstDecoder
from ._tensor_info import TensorInfo


POINTWISE_OPTIMIZE = True
DOT_OPTIMIZED = True

# Global dimension level counter
_n_dims_created = 0


def _relevant_op(opcode: Optional[str]) -> bool:
    """Check if opcode is relevant for variable assignment."""
    return bool(opcode and opcode.startswith("STORE_"))


def handle_from_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Handle tensor conversion for torch function integration."""
    return tensor


def _create_dim(name: str, size: Optional[int] = None) -> Dim:
    """Create a new Dim object."""
    return Dim(name, size if size is not None else -1)


def dims(
    n: Optional[int] = None, sizes: Optional[list[Optional[int]]] = None
) -> Union[Dim, tuple[Dim, ...]]:
    """
    Create and return one or more Dim objects.

    Uses bytecode inspection to determine variable names when possible.

    Args:
        n (int, optional): The number of dimensions to create. Can be omitted if sizes is specified.
        sizes (List[Optional[int]], optional): A list the same size as the number of dimensions to be
          created, specifying each dimensions size, or None to leave the size unset.

    Returns:
        Union[Dim, Tuple[Dim, ...]]: Single Dim if n=1, tuple of Dims otherwise.

    Examples:
        >>> batch, channel, width, height = dims(4)
        >>> batch, channel, width, height = dims(sizes=[None, 3, 224, 224])
        >>> single_dim = dims(1)
    """
    specified_ndims = -1
    found_ndims = 0

    # Parse arguments
    if sizes is not None:
        specified_ndims = len(sizes)
    if n is not None:
        specified_ndims = n

    # Use bytecode inspection
    frame = inspect.currentframe()
    if frame is None:
        raise RuntimeError("Unable to get current frame")
    frame = frame.f_back
    try:
        if frame is None:
            raise RuntimeError("Unable to get caller frame")
        code = frame.f_code
        lasti = frame.f_lasti

        decoder = _PyInstDecoder(code, lasti)

        if sys.version_info >= (3, 11):
            if decoder.opcode() == "PRECALL":
                decoder.next()

        # Move to next instruction after the call
        decoder.next()

        # Determine number of dimensions from bytecode
        if _relevant_op(decoder.opcode()):
            found_ndims = 1
        elif decoder.opcode() == "UNPACK_SEQUENCE":
            found_ndims = decoder.oparg()
            decoder.next()  # Move past UNPACK_SEQUENCE

        if specified_ndims == -1:
            if found_ndims == 0:
                raise SyntaxError(
                    "dims() must be assigned to a sequence of variable names or have argument n specified"
                )
            specified_ndims = found_ndims

        if found_ndims != specified_ndims:
            found_ndims = 0

        def genobject(i: int) -> Dim:
            nonlocal found_ndims
            name = None
            if i < found_ndims:
                name = decoder.name()

            if not name:
                name = f"d{i}"
                found_ndims = 0
            else:
                decoder.next()  # Move to next STORE instruction

            size = sizes[i] if sizes is not None else None
            return _create_dim(name, size)

        # Validate sizes parameter
        if sizes is not None and len(sizes) != specified_ndims:
            raise ValueError(f"expected {specified_ndims} sizes but found {len(sizes)}")

        if specified_ndims == 1:
            return genobject(0)

        result = []
        for i in range(specified_ndims):
            result.append(genobject(i))

        return tuple(result)

    finally:
        del frame


class DimList:
    """
    A list of first-class dimensions that can be bound to tensor dimensions.

    A DimList can be in one of two states:
    1. Unbound: Created with just a name, no specific dimensions yet
    2. Bound: Either created with specific dimensions/sizes, or bound later via bind() or bind_len()
    """

    _name: Optional[str]
    _dims: list[Dim]
    _bound: bool

    def __init__(
        self,
        len_or_dims: Optional[Union[int, Sequence]] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize a new DimList object.

        Args:
            len_or_dims: Optional length (int) or sequence of dimensions/sizes
            name: Optional name for the dimension list
        """
        # Initialize attributes
        self._name = name
        self._dims: list = []
        self._bound = False

        if isinstance(len_or_dims, int):
            self.bind_len(len_or_dims)
        elif len_or_dims is not None:
            dims = []
            for i, item in enumerate(len_or_dims):
                if isinstance(item, int):
                    dim_name = f"{self._name}{i}" if self._name else f"dim{i}"
                    dims.append(Dim(dim_name, item))
                else:
                    dims.append(Dim(item))
            self._set_dims(dims)

    def _set_dims(self, dims: list) -> None:
        """Set the dimensions and mark as bound."""
        self._bound = True
        self._dims = dims

    def bind_len(self, size: int) -> None:
        """
        Bind this DimList to a specific length.

        Args:
            size: Number of dimensions to bind to

        Raises:
            DimensionBindError: If already bound to a different size
        """
        if self._bound:
            if len(self._dims) != size:
                raise DimensionBindError(
                    f"Dimlist has size {len(self._dims)} but it is being bound to size {size}"
                )
        else:
            self._bound = True
            self._dims = []
            for i in range(size):
                dim_name = f"{self._name}{i}" if self._name else f"dim{i}"
                self._dims.append(Dim(dim_name))

    def bind(self, sizes: Sequence[int]) -> None:
        """
        Bind this DimList to specific sizes.

        Args:
            sizes: Sequence of sizes for each dimension

        Raises:
            ValueError: If sizes is not a sequence
        """
        if not hasattr(sizes, "__len__") or not hasattr(sizes, "__getitem__"):
            raise ValueError("expected a sequence")

        size = len(sizes)
        self.bind_len(size)

        for i, dim_size in enumerate(sizes):
            self._dims[i].size = int(dim_size)

    def _size(self) -> int:
        if not self._bound:
            raise DimensionBindError("DimList not bound")
        return len(self._dims)

    def size(self) -> int:
        """Return the size (number of dimensions) of this DimList."""
        return self._size()

    def _set_bound(self, b: bool) -> None:
        """Set the bound status (for internal use)."""
        self._bound = b

    @property
    def is_bound(self) -> bool:
        """Property to check if DimList is bound."""
        return self._bound

    def __len__(self) -> int:
        """Return the length of the DimList."""
        return self.size()

    def __getitem__(self, key: Union[int, slice]) -> Union[Dim, tuple[Dim, ...]]:
        if not self._bound:
            raise DimensionBindError("DimList not bound")

        if isinstance(key, int):
            if key < 0 or key >= len(self._dims):
                raise IndexError("index out of bounds")
            return self._dims[key]
        elif isinstance(key, slice):
            start, stop, step = key.indices(len(self._dims))
            result = []
            for i in range(start, stop, step):
                result.append(self._dims[i])
            return tuple(result)
        else:
            raise ValueError("expected an int or a slice")

    def __repr__(self) -> str:
        """Return string representation of the DimList."""
        if self._bound:
            # Show as tuple representation
            return f"({', '.join(repr(dim) for dim in self._dims)})"
        elif self._name is not None:
            # Show as *name for unbound with name
            return f"*{self._name}"
        else:
            # Show as <unbound_dimlist> for unbound without name
            return "<unbound_dimlist>"

    def __str__(self) -> str:
        """Return string representation of the DimList."""
        return self.__repr__()

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: tuple,
        args: tuple = (),
        kwargs: Optional[dict] = None,
    ) -> Any:
        return _Tensor.__torch_function__(func, types, args, kwargs)


def _create_dimlist(
    name: str, size: Optional[Union[int, list[Optional[int]]]] = None
) -> DimList:
    """Create a DimList object with the given name and optional size."""
    dimlist = DimList(name=name)
    if size is not None:
        if isinstance(size, int):
            dimlist.bind_len(size)
        else:
            # size is a list of optional ints
            dimlist.bind_len(len(size))
            for i, s in enumerate(size):
                if s is not None:
                    dimlist._dims[i].size = s
    return dimlist


def dimlists(
    n: Optional[int] = None, sizes: Optional[list[Optional[int]]] = None
) -> Union[DimList, tuple[DimList, ...]]:
    """
    Create and return one or more DimList objects.

    Similar to dims() but creates DimList objects instead.
    """
    specified_ndims = -1
    found_ndims = 0

    # Parse arguments
    if sizes is not None:
        specified_ndims = len(sizes)
    if n is not None:
        specified_ndims = n

    frame = inspect.currentframe()
    if frame is None:
        raise RuntimeError("Unable to get current frame")
    frame = frame.f_back
    try:
        if frame is None:
            raise RuntimeError("Unable to get caller frame")
        code = frame.f_code
        lasti = frame.f_lasti

        decoder = _PyInstDecoder(code, lasti)

        if sys.version_info >= (3, 11):
            if decoder.opcode() == "PRECALL":
                decoder.next()

        # Move to next instruction after the call
        decoder.next()

        # Determine number of dimensions from bytecode
        if _relevant_op(decoder.opcode()):
            found_ndims = 1
        elif decoder.opcode() == "UNPACK_SEQUENCE":
            found_ndims = decoder.oparg()
            decoder.next()  # Move past UNPACK_SEQUENCE

        if specified_ndims == -1:
            if found_ndims == 0:
                raise SyntaxError(
                    "dimlists() must be assigned to a sequence of variable names or have argument n specified"
                )
            specified_ndims = found_ndims

        if found_ndims != specified_ndims:
            found_ndims = 0

        # Generator function for dimlist names
        def genobject(i: int) -> str:
            nonlocal found_ndims
            name = None
            if i < found_ndims:
                name = decoder.name()

            if not name:
                name = f"d{i}"
                found_ndims = 0
            else:
                decoder.next()  # Move to next STORE instruction

            return name

        # Validate sizes
        if sizes is not None and len(sizes) != specified_ndims:
            raise ValueError(f"expected {specified_ndims} sizes but found {len(sizes)}")

        # Create dimlists
        if specified_ndims == 1:
            name = genobject(0)
            return _create_dimlist(name, sizes[0] if sizes is not None else None)

        result = []
        for i in range(specified_ndims):
            name = genobject(i)
            size = sizes[i] if sizes is not None else None
            result.append(_create_dimlist(name, size))

        return tuple(result)

    finally:
        del frame


class DimensionMismatchError(Exception):
    pass


class DimensionBindError(Exception):
    pass


from . import op_properties


def _safe_print(*args: Any, **kwargs: Any) -> None:
    """Safe print that avoids recursive torch function dispatches."""
    import sys

    # Convert any torch objects to basic representations
    safe_args = []
    for arg in args:
        if hasattr(arg, "__class__") and "torch" in str(type(arg)):
            safe_args.append(f"<{type(arg).__name__}>")
        else:
            safe_args.append(str(arg))

    print(*safe_args, **kwargs, file=sys.stderr)


class _Tensor:
    def _get_levels(self) -> list[Any]:
        raise NotImplementedError("_get_levels must be implemented by subclass")

    def _get_tensor(self) -> Optional[torch.Tensor]:
        raise NotImplementedError("_get_tensor must be implemented by subclass")

    @property
    def ndim(self) -> int:
        raise NotImplementedError("ndim must be implemented by subclass")

    @property
    def dims(self) -> tuple[Any, ...]:
        return tuple(l.dim() for l in self._get_levels() if not l.is_positional())

    def dim(self) -> int:
        return self.ndim

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: tuple,
        args: tuple = (),
        kwargs: Optional[dict] = None,
    ) -> Any:
        if kwargs is None:
            kwargs = {}

        if DOT_OPTIMIZED and func is torch.Tensor.__mul__:
            # Check conditions: 2 args, both are tensor-like, both 0-dimensional
            if (
                len(args) == 2
                and not kwargs
                and isinstance(args[0], (_Tensor, torch.Tensor))
                and isinstance(args[1], (_Tensor, torch.Tensor))
            ):
                # Get tensor info for both operands
                lhs_info = TensorInfo.create(
                    args[0], ensure_batched=False, ensure_present=False
                )
                rhs_info = TensorInfo.create(
                    args[1], ensure_batched=False, ensure_present=False
                )

                if (
                    lhs_info
                    and rhs_info
                    and lhs_info.tensor is not None
                    and rhs_info.tensor is not None
                    and lhs_info.tensor.dim() == 0
                    and rhs_info.tensor.dim() == 0
                ):
                    if (
                        lhs_info.tensor.is_floating_point()
                        and rhs_info.tensor.is_floating_point()
                    ):
                        # Collect all unique levels and has_device
                        has_device = lhs_info.has_device or rhs_info.has_device
                        levels = []

                        for level in lhs_info.levels:
                            if level not in levels:
                                levels.append(level)
                        for level in rhs_info.levels:
                            if level not in levels:
                                levels.append(level)

                        # Debug print
                        # print(f"DEBUG: Creating delayed mul, levels: {levels}, has_device: {has_device}")

                        # Create delayed tensor
                        return Tensor.create_delayed(func, args, levels, has_device)

        if func is torch.Tensor.__getitem__:
            from functorch.dim._getsetitem import getitem

            return getitem(cls, func, types, args, kwargs)

        if func is torch.Tensor.__setitem__:
            from functorch.dim._getsetitem import setitem

            # args should be (tensor, index, value)
            if len(args) == 3:
                setitem(args[0], args[1], args[2])
                return None
            else:
                raise ValueError(f"Expected 3 args for __setitem__, got {len(args)}")

        # Fast-path for len; mostly to avoid infinite loop in TestMinFunctorchOnly.test_softmax_split
        if func is torch.Tensor.__len__:
            return args[0].size(0)

        # Special handling for torch.softmax - use the pre-wrapped version
        if func is torch.softmax:
            return softmax(*args, **kwargs)

        # Special handling for torch.stack - use the custom stack function
        if func is torch.stack:
            return stack(*args, **kwargs)

        if (
            func is torch.Tensor.split
            or func is torch._VF.split  # type: ignore[attr-defined]
            or func is torch._VF.split_with_sizes  # type: ignore[attr-defined]
            or func is torch.split
        ):
            return split(*args, **kwargs)

        return _Tensor._torch_function_fallback(func, types, args, kwargs)

    @staticmethod
    def _torch_function_fallback(
        func: Callable, types: tuple, args: tuple, kwargs: dict
    ) -> Any:
        """Fallback torch function implementation for non-special-cased functions."""
        is_pointwise = POINTWISE_OPTIMIZE and func in op_properties.pointwise
        # TODO: optimize pytree here
        flat_args, spec = tree_flatten((args, kwargs))
        device_holding_tensor = None

        infos: list[TensorInfo] = []
        result_levels: list[DimEntry] = []

        for f in flat_args:
            info = TensorInfo.create(f, not is_pointwise, False)
            infos.append(info)
            if info:
                assert is_pointwise or info.batchedtensor is not None
                if device_holding_tensor is None and info.has_device:
                    device_holding_tensor = info.tensor
                # Collect all unique levels
                for level in info.levels:
                    assert isinstance(level, DimEntry)
                    if level not in result_levels:
                        result_levels.append(level)

        if is_pointwise:
            # Pointwise operation: match all tensors to common levels
            for i, info in enumerate(infos):
                if info and info.tensor is not None:
                    tensor = info.tensor
                    if device_holding_tensor is not None and not info.has_device:
                        tensor = tensor.to(device_holding_tensor.device)
                    ml = _match_levels(tensor, info.levels, result_levels)
                    flat_args[i] = handle_from_tensor(ml)

            unflat_args, unflat_kwargs = tree_unflatten(flat_args, spec)
            result = func(*unflat_args, **unflat_kwargs)

            # Wrap tensor results
            def wrap_tensor(obj: Any) -> Any:
                if isinstance(obj, torch.Tensor):
                    return Tensor.from_positional(
                        obj, result_levels, device_holding_tensor is not None
                    )
                return obj

            # Small fastpath
            if isinstance(result, torch.Tensor):
                return wrap_tensor(result)
            else:
                return tree_map(wrap_tensor, result)

        # Non-pointwise operation: use functorch vmap layers
        with EnableAllLayers(result_levels) as guard:
            # Update arguments with batched tensors
            for i, info in enumerate(infos):
                if info and info.batchedtensor is not None:
                    batched = info.batchedtensor
                    if device_holding_tensor is not None and not info.has_device:
                        batched = batched.to(device_holding_tensor.device)
                    guard.inplace_update_layers(batched, info.levels)
                    flat_args[i] = handle_from_tensor(batched)

            unflat_args, unflat_kwargs = tree_unflatten(flat_args, spec)
            result = func(*unflat_args, **unflat_kwargs)

            # Unwrap results from functorch layers
            def unwrap_tensor(obj: Any) -> Any:
                if isinstance(obj, torch.Tensor):
                    return guard.from_batched(obj, device_holding_tensor is not None)
                return obj

            if isinstance(result, torch.Tensor):
                return unwrap_tensor(result)
            else:
                return tree_map(unwrap_tensor, result)

    def __setitem__(self, index: Any, value: Any) -> None:
        """Set values in tensor using first-class dimensions."""
        from functorch.dim._getsetitem import setitem

        return setitem(self, index, value)

    # expand and index are OK to be methods because they don't have torch.*
    # versions, but if they did they need the stack/cat treatment

    def expand(self, *args: Dim) -> _Tensor:
        """
        Expand tensor by adding new dimensions or expanding existing dimensions.

        If all arguments are Dim objects, adds new named dimensions.
        Otherwise, falls back to regular tensor expansion behavior.

        Args:
            args: Either Dim objects for new dimensions or sizes for regular expansion

        Returns:
            New tensor with expanded dimensions

        Example:
            >>> i, j = dims()
            >>> t = torch.randn(3, 4)
            >>> expanded = t[i].expand(j, k)  # Add j, k dimensions
            >>> expanded2 = t[i].expand(2, 4)  # Regular expand with sizes
        """
        info = TensorInfo.create(self, ensure_batched=False, ensure_present=False)

        for arg in args:
            if not isinstance(arg, Dim):
                # Not all args are Dims, fallback to regular expand
                if isinstance(self, torch.Tensor) and not isinstance(self, _Tensor):
                    return torch.Tensor.expand(self, *args)
                else:
                    return self.__torch_function__(
                        torch.Tensor.expand, (type(self),), (self,) + args
                    )

        # All args are Dim objects - proceed with first-class dimension expansion
        if not info:
            # No tensor info available, fallback
            return self.__torch_function__(
                torch.Tensor.expand, (type(self),), (self,) + args
            )

        # First-class dimension expansion - all args are Dim objects
        data = info.tensor
        if data is None:
            # No tensor data available, fallback
            return self.__torch_function__(
                torch.Tensor.expand, (type(self),), (self,) + args
            )

        levels = info.levels

        new_levels: list[DimEntry] = []
        new_sizes = []
        new_strides = []

        for d in args:
            # Check if dimension already exists in current levels or new_levels
            for level in levels:
                if not level.is_positional() and level.dim() is d:
                    raise DimensionBindError(
                        f"expanding dimension {d} already exists in tensor with dims"
                    )
            for new_level in new_levels:
                if not new_level.is_positional() and new_level.dim() is d:
                    raise DimensionBindError(
                        f"expanding dimension {d} already exists in tensor with dims"
                    )

            new_levels.append(DimEntry(d))
            new_sizes.append(d.size)
            new_strides.append(0)

        # Add existing levels
        new_levels.extend(levels)

        # Add existing sizes and strides
        orig_sizes = list(data.size())
        orig_strides = list(data.stride())
        new_sizes.extend(orig_sizes)
        new_strides.extend(orig_strides)

        # Create expanded tensor using as_strided
        expanded_data = data.as_strided(new_sizes, new_strides, data.storage_offset())

        # Return new tensor with expanded dimensions
        result = Tensor.from_positional(expanded_data, new_levels, info.has_device)
        return result  # type: ignore[return-value]  # Tensor and torch.Tensor are interchangeable

    def index(
        self,
        dims: Union[int, Dim, tuple[Union[int, Dim], ...], list[Union[int, Dim]]],
        indices: Union[
            int,
            slice,
            torch.Tensor,
            tuple[Union[int, slice, torch.Tensor], ...],
            list[Union[int, slice, torch.Tensor]],
        ],
    ) -> _Tensor:
        """
        Index tensor using first-class dimensions.
        """
        from ._dim_entry import _match_levels
        from ._getsetitem import getsetitem_flat, invoke_getitem
        from ._wrap import _wrap_dim

        # Helper to check if obj is a dimpack (tuple/list) and extract items
        def maybe_dimpack(obj: Any, check_first: bool = False) -> tuple[Any, bool]:
            if isinstance(obj, (tuple, list)):
                return list(obj), True
            return None, False

        def parse_dim_entry(s: Any) -> Any:
            d = _wrap_dim(s, self.ndim, False)
            if d.is_none():
                raise TypeError(f"expected a dimension specifyer but found {repr(s)}")
            return d

        # Helper for dimension not present errors
        def dim_not_present(d: Any) -> None:
            if d.is_positional():
                raise TypeError(
                    f"dimension {d.position() + self.ndim} not in tensor of {self.ndim} dimensions"
                )
            else:
                raise TypeError(f"dimension {repr(d.dim())} not in tensor")

        dims_list: list[Union[int, Dim]] = []
        indices_list: list[Union[int, slice, torch.Tensor]] = []

        lhs_list = isinstance(dims, (tuple, list))
        rhs_list = isinstance(indices, (tuple, list))

        if lhs_list and rhs_list:
            # Type narrowing: we know dims and indices are sequences here
            dims_seq = dims  # type: ignore[assignment]
            indices_seq = indices  # type: ignore[assignment]
            if len(dims_seq) != len(indices_seq):  # type: ignore[arg-type]
                raise TypeError(
                    f"dims ({len(dims_seq)}) and indices ({len(indices_seq)}) must have the same length"  # type: ignore[arg-type]
                )
            dims_list.extend(dims_seq)  # type: ignore[arg-type]
            indices_list.extend(indices_seq)  # type: ignore[arg-type]
        else:
            dims_list.append(dims)  # type: ignore[arg-type]
            indices_list.append(indices)  # type: ignore[arg-type]

        # Create tensor info
        self_info = TensorInfo.create(self, False, False)

        new_levels: list[Any] = []
        to_flatten: list[Any] = []
        dims_list_flat = []

        # Process each dim specification
        for i in range(len(dims_list)):
            m, is_dimpack = maybe_dimpack(dims_list[i], check_first=False)
            if is_dimpack:
                if len(m) == 0:
                    dims_list_flat.append(DimEntry())  # Empty dimpack
                    continue

                first = parse_dim_entry(m[0])
                dims_list_flat.append(first)

                if len(m) == 1:
                    continue

                # Multi-element dimpack requires flattening
                if len(to_flatten) == 0:
                    new_levels.extend(self_info.levels)

                rest = []
                for j in range(1, len(m)):
                    d = parse_dim_entry(m[j])
                    removed = False
                    for k in range(len(new_levels)):
                        if new_levels[k] == d:
                            new_levels.pop(k)
                            removed = True
                            break
                    if not removed:
                        dim_not_present(d)
                    rest.append(d)

                # Find first in new_levels
                first_idx = None
                for k in range(len(new_levels)):
                    if new_levels[k] == first:
                        first_idx = k
                        break
                if first_idx is None:
                    dim_not_present(first)
                    continue  # Skip this iteration if dimension not found

                for j, r in enumerate(rest):
                    new_levels.insert(first_idx + 1 + j, r)
                to_flatten.extend(rest)
            else:
                dims_list_flat.append(parse_dim_entry(dims_list[i]))

        # Handle dimension flattening if needed
        if len(to_flatten) > 0:
            assert self_info.tensor is not None, (
                "Cannot perform dimension flattening on None tensor"
            )
            rearranged = _match_levels(self_info.tensor, self_info.levels, new_levels)
            sizes = rearranged.size()
            new_sizes: list[Any] = []
            reshape_levels = []

            for i in range(len(new_levels)):
                if new_levels[i] in to_flatten:
                    if len(new_sizes) == 0:
                        new_sizes.append(sizes[i])
                    else:
                        new_sizes[-1] *= sizes[i]
                else:
                    new_sizes.append(sizes[i])
                    reshape_levels.append(new_levels[i])

            self_info.tensor = rearranged.reshape(new_sizes)
            self_info.levels = reshape_levels

        # Check for dimpacks in indices
        has_dimpacks = False
        for idx in indices_list:
            if isinstance(idx, (tuple, list)):
                has_dimpacks = True
                break

        # Call getsetitem_flat with correct parameters
        info = getsetitem_flat(
            self_info,
            [],  # empty input_list
            dims_list_flat,  # keys
            indices_list,  # values
            has_dimpacks,
        )

        return invoke_getitem(info)

    def __repr__(self) -> str:
        tensor, levels, ndim = self._get_tensor(), self._get_levels(), self.ndim
        dims_repr = []
        for l in levels:
            if hasattr(l, "is_positional") and l.is_positional():
                # Convert negative positional to positive: -1 -> ndim-1, -2 -> ndim-2, etc.
                dims_repr.append(l.position() + ndim)
            elif hasattr(l, "dim"):
                dims_repr.append(l.dim())
            elif hasattr(l, "data"):
                dims_repr.append(l.data)
            else:
                dims_repr.append(l)
        return f"{tensor}\nwith dims={tuple(dims_repr)} sizes={tuple(tensor.size())}"  # type: ignore[union-attr]


TensorLike = (_Tensor, torch.Tensor)


class Dim(_Tensor):
    _level: int
    _name: str
    _size: int
    _range: Optional[torch.Tensor]
    _batchtensor: Optional[torch.Tensor]

    def __init__(self, name: str, s: int = -1) -> None:
        global _n_dims_created
        self._name = name
        self._size = s
        self._level = _n_dims_created
        _n_dims_created += 1
        self._range = None
        self._batchtensor = None

    @property
    def ndim(self) -> int:
        return 1

    @classmethod
    def check_exact(cls, obj: Any) -> bool:
        return type(obj) is cls

    @property
    def size(self) -> int:
        if self._size == -1:
            raise ValueError(f"dimension {self._name} is unbound")
        return self._size

    @size.setter
    def size(self, v: int) -> None:
        if self._size == -1:
            self._size = v
        elif self._size != v:
            raise DimensionBindError(
                f"Dim '{repr(self)}' previously bound to a dimension of size {self._size} "
                f"cannot bind to a dimension of size {v}"
            )

    @property
    def is_bound(self) -> bool:
        """Return True if this dimension is bound to a size."""
        return self._size != -1

    def _get_range(self) -> torch.Tensor:
        """
        Get a tensor representing the range [0, size) for this dimension.

        Returns:
            A 1D tensor with values [0, 1, 2, ..., size-1]
        """
        if self._range is None:
            self._range = torch.arange(self.size)
        return self._range

    def _get_batchtensor(self) -> torch.Tensor:
        """
        Get a batched tensor representation of this dimension.

        Returns:
            A batched tensor created from the range tensor
        """
        if self._batchtensor is None:
            self._batchtensor = torch._C._functorch._add_batch_dim(
                self._get_range(), 0, self._level
            )
        return self._batchtensor

    def __repr__(self) -> str:
        """String representation of a Dim object."""
        return self._name

    # note that Dim comes before tensor because we want the Dim API for things like size to take precedence.
    # Tensor defines format, but we want to print Dims with special formatting
    __format__ = object.__format__


# Somewhat confusingly, an FCD tensor is also called Tensor.  This confusion
# is somewhat intentional, as FCD tensors are intended to be substitutable
# with regular Tensor (just with some positional dims hidden).
class Tensor(_Tensor):
    _tensor: Optional[torch.Tensor]
    _batchtensor: Optional[torch.Tensor]
    _levels: list[DimEntry]
    _has_device: bool
    _delayed: Optional[Callable[[], torch.Tensor]]
    _delayed_orig: Optional[Callable]
    _delayed_args: Optional[tuple]

    @property
    def ndim(self) -> int:
        return sum(1 if l.is_positional() else 0 for l in self._levels)

    @classmethod
    def check_exact(cls, other: Any) -> bool:
        return type(other) is cls

    @classmethod
    def from_positional(
        cls, tensor: torch.Tensor, levels: list[DimEntry], has_device: bool
    ) -> Union[_Tensor, torch.Tensor]:
        """
        Create a functorch Tensor from a regular PyTorch tensor with specified dimension levels.

        This is the primary way to create Tensor objects with first-class dimensions.

        Args:
            tensor: The underlying PyTorch tensor
            levels: List of DimEntry objects specifying the dimension structure
            has_device: Whether the tensor is on a device (not CPU)

        Returns:
            A new Tensor instance with the specified dimensions, or a regular torch.Tensor
            if there are no named dimensions
        """
        seen_dims = 0
        last = 0

        for l in levels:
            if l.is_positional():
                # Validate consecutive positional dimensions
                assert last == 0 or last + 1 == l.position(), (
                    f"Positional dimensions must be consecutive, got {last} then {l.position()}"
                )
                last = l.position()
            else:
                # This is a named dimension
                seen_dims += 1

        # Validate final positional dimension
        assert last == 0 or last == -1, (
            f"Final positional dimension must be 0 or -1, got {last}"
        )

        if not seen_dims:
            return tensor

        # Create Tensor object with proper level management
        result = cls()
        result._tensor = tensor
        result._levels = levels
        result._has_device = has_device
        result._batchtensor = None  # Will be created lazily if needed
        result._delayed = None
        result._delayed_orig = None
        result._delayed_args = None

        # Validate tensor dimensionality matches levels
        assert tensor.dim() == len(levels), (
            f"Tensor has {tensor.dim()} dimensions but {len(levels)} levels provided"
        )

        return result

    @classmethod
    def create_delayed(
        cls, orig: Callable, args: tuple, levels: list[DimEntry], has_device: bool
    ) -> _Tensor:
        """
        Create a delayed tensor that defers the operation until later.
        """
        result = cls()
        result._tensor = None  # Will be computed when needed
        result._levels = levels
        result._has_device = has_device
        result._batchtensor = None
        result._delayed_orig = orig
        result._delayed_args = args

        # Create delayed evaluation function that unwraps Tensor objects
        def evaluate_delayed() -> torch.Tensor:
            unwrapped_args = []
            for arg in args:
                if hasattr(arg, "_get_tensor"):
                    unwrapped_args.append(arg._get_tensor())
                else:
                    unwrapped_args.append(arg)
            return orig(*unwrapped_args)

        result._delayed = evaluate_delayed

        return result

    def _get_tensor(self) -> Optional[torch.Tensor]:
        """Get the underlying tensor, handling delayed operations if needed."""
        if (
            hasattr(self, "_delayed")
            and self._delayed is not None
            and self._tensor is None
        ):
            # Execute the delayed operation
            self._tensor = self._delayed()
            # Clear delayed operation to avoid re-execution
            self._delayed = None
            self._delayed_orig = None
            self._delayed_args = None
        return self._tensor

    def _get_levels(self) -> list[Any]:
        """Get the dimension levels."""
        return self._levels

    def _get_has_device(self) -> bool:
        """Get whether this tensor has device information."""
        return self._has_device

    def _get_batchtensor(self) -> Optional[torch.Tensor]:
        """Get the batched tensor representation, creating it lazily if needed."""
        if self._batchtensor is None:
            self._batchtensor = self._add_batch_dims(
                self._get_tensor(), self._get_levels()
            )
        return self._batchtensor

    def _add_batch_dims(
        self, t: Optional[torch.Tensor], levels_: list[Any]
    ) -> Optional[torch.Tensor]:
        levels = list(levels_)

        while True:
            min_real_index = -1
            min_index = -1
            min_value = float("inf")  # INT_MAX equivalent
            i = 0
            r = 0

            for r, l in enumerate(levels):
                if not l.is_none():
                    if not l.is_positional() and l.dim()._level < min_value:
                        min_value = l.dim()._level
                        min_index = i
                        min_real_index = r
                    i += 1

            if min_index == -1:
                return t

            assert t is not None
            t = torch._C._functorch._add_batch_dim(t, min_index, int(min_value))

            levels[min_real_index] = DimEntry()
        return None

    def order(self, *dims: Any) -> _Tensor:
        """Reorder the dimensions of this tensor."""
        from ._order import order

        result = order(self, *dims)
        return result  # type: ignore[return-value]  # Tensor and torch.Tensor are interchangeable


def stack(tensors: Any, new_dim: Any, dim: int = 0) -> _Tensor:
    """
    Stack tensors along a new dimension.

    Args:
        tensors: Sequence of tensors to stack
        new_dim: The new Dim to create for stacking
        dim: The dimension position to insert the new dimension (default: 0)

    Returns:
        Stacked tensor with the new dimension
    """
    if not tensors:
        raise ValueError("stack expects a non-empty sequence of tensors")

    # Check if new_dim is a Dim object
    if not isinstance(new_dim, Dim):
        # Fall back to regular torch.stack
        result = torch.stack(tensors, dim=dim)
        return result  # type: ignore[return-value]

    # Collect all result_levels from input tensors
    result_levels = []
    infos = []

    for t in tensors:
        info = TensorInfo.create(t, ensure_batched=False, ensure_present=False)
        infos.append(info)
        for level in info.levels:
            if level not in result_levels:
                result_levels.append(level)

    # Set the new_dim size to match number of tensors
    new_dim.size = len(tensors)

    # Match all tensors to the common level structure using _match_levels
    inputs = []
    for info in infos:
        assert info.tensor is not None, "Cannot stack tensors with None tensor data"
        matched_tensor = _match_levels(info.tensor, info.levels, result_levels)
        inputs.append(matched_tensor)

    # Calculate ndim and resolve the dim parameter
    ndim = ndim_of_levels(result_levels)
    rawdim = 0
    if dim is not None and not (isinstance(dim, int) and dim == 0):
        from ._wrap import _wrap_dim

        d = _wrap_dim(dim, ndim, False)
        try:
            idx = result_levels.index(d)
        except ValueError:
            raise TypeError(f"Dimension {dim} does not exist in inputs") from None
        rawdim = idx

    # Stack tensors at the resolved dimension
    result = torch.stack(inputs, rawdim)

    # Insert new dimension entry at the correct position
    result_levels.insert(rawdim, DimEntry(new_dim))

    # Return as a first-class tensor
    tensor_result = Tensor.from_positional(
        result, result_levels, infos[0].has_device if infos else True
    )
    return tensor_result  # type: ignore[return-value]


def split(tensor: Any, split_size_or_sections: Any, dim: Any = None) -> tuple:
    """
    Split tensor along a dimension.

    Can handle both regular integer sizes and Dim objects for split sizes.
    When Dim objects are used, they get bound to the resulting tensor dimensions.
    """
    from ._wrap import _wrap_dim

    # Check if dim is a Dim object
    dim_is_object = isinstance(dim, Dim)

    # Parse split_size_or_sections
    if isinstance(split_size_or_sections, int):
        # Single integer - use regular split
        if dim_is_object:
            raise TypeError(
                "when dim is specified as a Dim object, split sizes must also be dimensions."
            )
        return _Tensor._torch_function_fallback(
            torch.Tensor.split,
            (type(tensor),),
            (tensor, split_size_or_sections),
            {"dim": dim},
        )

    # Check if it's a sequence
    sizes = []
    all_dims = True
    all_ints = True

    for item in split_size_or_sections:
        sizes.append(item)
        if isinstance(item, Dim):
            all_ints = False
        else:
            all_dims = False

    if all_ints:
        # All integers - use regular split
        if dim_is_object:
            raise TypeError(
                "when dim is specified as a Dim object, split sizes must also be dimensions."
            )
        return _Tensor._torch_function_fallback(
            torch.Tensor.split,
            (type(tensor),),
            (tensor, split_size_or_sections),
            {"dim": dim},
        )

    if not all_dims:
        raise TypeError("split list must be ints or dims but got a mix")

    # All are Dim objects - handle first-class dimension split
    self_info = TensorInfo.create(tensor, ensure_batched=False, ensure_present=False)
    ndim = self_info.ndim()

    if not dim_is_object and ndim == 0:
        raise TypeError("split expects at least a 1-dimension tensor")

    # Wrap the dimension
    dim_l = _wrap_dim(dim, ndim, False) if dim is not None else DimEntry(-ndim)

    # Find the index of the dimension in levels
    idx = None
    for i, level in enumerate(self_info.levels):
        if level == dim_l:
            idx = i
            break

    if idx is None:
        if dim is None:
            dim = 0
        raise TypeError(f"tensor does not contain dimension {dim}")

    # Calculate split indices
    indices = []
    total_size = 0
    unbound = []

    for i, size_dim in enumerate(sizes):
        if size_dim.is_bound:
            indices.append(size_dim.size)
            total_size += indices[-1]
        else:
            indices.append(0)
            unbound.append(i)

    assert self_info.tensor is not None, "Cannot get tensor size on None tensor"
    tensor_size = self_info.tensor.size(idx)

    # Handle unbound dimensions
    if unbound:
        if total_size > tensor_size:
            raise TypeError(
                f"sizes of target dimensions add up to more ({total_size}) than source dim ({tensor_size})"
            )
        remaining_size = tensor_size - total_size
        chunk_size = (remaining_size + len(unbound) - 1) // len(unbound)
        for u in unbound:
            sz = min(chunk_size, remaining_size)
            sizes[u].size = sz
            indices[u] = sz
            remaining_size -= sz
    elif tensor_size != total_size:
        raise TypeError(
            f"sum of sizes of target dimensions ({total_size}) do not match the source dim ({tensor_size})"
        )

    # Perform the split
    result_tensors = self_info.tensor.split_with_sizes(indices, idx)

    # Create result with new levels
    result = []
    new_levels = list(self_info.levels)

    for i, (result_tensor, size_dim) in enumerate(zip(result_tensors, sizes)):
        new_levels[idx] = DimEntry(size_dim)
        result.append(
            Tensor.from_positional(
                result_tensor, list(new_levels), self_info.has_device
            )
        )

    return tuple(result)


def cat(tensors: Any, dim: Any, new_dim: Any) -> _Tensor:
    n = dims(1)  # Get single Dim instead of tuple
    return stack(tensors, n, dim).index([n, dim], new_dim)  # type: ignore[list-item]


class DotPart:
    """
    Helper class for organizing dimensions in dot products.
    """

    def __init__(self) -> None:
        self.dims: list[DimEntry] = []
        self.total_size = 1

    def append(self, dim_entry: Any) -> None:
        """Add a dimension entry to this part."""
        self.dims.append(dim_entry)
        if not dim_entry.is_positional():
            self.total_size *= dim_entry.dim().size


def dot_prepare(parts: list[DotPart], tensor_info: TensorInfo) -> torch.Tensor:
    """
    Prepare tensor for dot product by matching levels and reshaping.
    """
    new_levels = []
    needs_reshape = False

    for part in parts:
        if len(part.dims) != 1:
            needs_reshape = True
        new_levels.extend(part.dims)

    if tensor_info.tensor is None:
        raise RuntimeError("Cannot perform dot product on None tensor")
    result = _match_levels(tensor_info.tensor, tensor_info.levels, new_levels)

    if not needs_reshape:
        return result

    # Reshape for matrix operations
    view = [part.total_size for part in parts]
    return result.reshape(view)


def dot_finish(parts: list[DotPart], result_tensor: torch.Tensor) -> Tensor:
    """
    Finish dot product by reshaping result and creating Tensor.
    """
    result_levels = []
    needs_reshape = False

    for part in parts:
        if len(part.dims) != 1:
            needs_reshape = True
        result_levels.extend(part.dims)

    if needs_reshape:
        new_size = []
        for level in result_levels:
            new_size.append(level.dim().size)
        result_tensor = result_tensor.reshape(new_size)

    tensor_result = Tensor.from_positional(result_tensor, result_levels, True)
    return tensor_result  # type: ignore[return-value]


def dot(lhs: Any, rhs: Any, sum_dims: Any) -> Union[_Tensor, torch.Tensor]:
    """
    Perform dot product between two tensors along specified dimensions.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side tensor
        sum_dims: Dimensions to sum over (contract)

    Returns:
        Result of dot product
    """
    # Get tensor info
    lhs_info = TensorInfo.create(lhs, ensure_batched=False, ensure_present=False)
    rhs_info = TensorInfo.create(rhs, ensure_batched=False, ensure_present=False)

    if not (lhs_info and rhs_info):
        # Fall back to regular operations
        return torch.matmul(lhs, rhs)

    assert lhs_info.tensor is not None and rhs_info.tensor is not None, (
        "Cannot perform dot product on None tensors"
    )

    lhs_strides = lhs_info.tensor.stride()
    rhs_strides = rhs_info.tensor.stride()

    # Create dot parts for different dimension categories
    lro_dims = DotPart()  # Left-right-output (batch dims)
    lo_dims = DotPart()  # Left-output only
    ro_dims = DotPart()  # Right-output only
    lr_dims = DotPart()  # Left-right (contracted dims)

    def insert_dim(d: Any, lhs_idx: Any, rhs_idx: Any) -> None:
        """Insert dimension into appropriate part based on stride pattern."""
        reduced = d in sum_dims
        lhs_stride = lhs_strides[lhs_idx] if lhs_idx is not None else 0
        rhs_stride = rhs_strides[rhs_idx] if rhs_idx is not None else 0

        if reduced:
            lr_dims.append(d)
        else:
            if (lhs_stride == 0) == (rhs_stride == 0):
                lro_dims.append(d)  # Both have or both lack this dim
            elif lhs_stride != 0:
                lo_dims.append(d)  # Only lhs has this dim
            else:
                ro_dims.append(d)  # Only rhs has this dim

    # Track which rhs dimensions we've seen
    rhs_seen = [False] * len(rhs_info.levels)

    # Process lhs dimensions
    for i, lhs_level in enumerate(lhs_info.levels):
        rhs_idx = None
        for j, rhs_level in enumerate(rhs_info.levels):
            if lhs_level == rhs_level:
                rhs_idx = j
                rhs_seen[j] = True
                break

        insert_dim(lhs_level, i, rhs_idx)

    # Process remaining rhs dimensions
    for i, rhs_level in enumerate(rhs_info.levels):
        if not rhs_seen[i]:
            insert_dim(rhs_level, None, i)

    # Validate sum dimensions exist
    if len(lr_dims.dims) != len(sum_dims):
        for d in sum_dims:
            if d not in lhs_info.levels and d not in rhs_info.levels:
                raise ValueError(f"summing over non-existent dimension {d}")

    # Prepare tensors and perform matrix multiplication
    if len(lro_dims.dims) != 0:
        # Batched matrix multiply
        lhs_tensor = dot_prepare([lro_dims, lo_dims, lr_dims], lhs_info)
        rhs_tensor = dot_prepare([lro_dims, lr_dims, ro_dims], rhs_info)
        result = torch.bmm(lhs_tensor, rhs_tensor)
        return dot_finish([lro_dims, lo_dims, ro_dims], result)
    else:
        # Regular matrix multiply
        lhs_tensor = dot_prepare([lo_dims, lr_dims], lhs_info)
        rhs_tensor = dot_prepare([lr_dims, ro_dims], rhs_info)
        result = torch.mm(lhs_tensor, rhs_tensor)
        return dot_finish([lo_dims, ro_dims], result)


from functorch.dim._wrap import _wrap
from functorch.dim.wrap_type import wrap_type


wrap_type(_Tensor, torch.Tensor, _Tensor.__torch_function__)
del _Tensor.ndim


def index(self: Any, positions: Any, dims: Any) -> _Tensor:
    """
    Index a regular tensor by binding specified positions to dims.

    This converts a regular tensor to a first-class tensor by binding
    the specified positional dimensions to Dim objects.

    Args:
        positions: Tuple of dimension positions to bind
        dims: Dim objects or tuple of Dim objects to bind to

    Returns:
        First-class tensor with specified dimensions bound
    """
    # If this is already a first-class tensor (_Tensor), call its index method directly
    if isinstance(self, _Tensor):
        return _Tensor.index(self, positions, dims)

    # Convert regular tensor to first-class tensor
    info = TensorInfo.create(self, ensure_batched=False, ensure_present=False)

    # Create the first-class tensor
    assert info.tensor is not None, "Cannot index None tensor"
    result = Tensor.from_positional(info.tensor, info.levels, info.has_device)

    # Now call the index method on the first-class tensor
    # Cast result to _Tensor for the method call
    return _Tensor.index(result, positions, dims)  # type: ignore[arg-type]


def _def(name: str, *args: Any, **kwargs: Any) -> None:
    orig = getattr(torch.Tensor, name)
    setattr(_Tensor, name, _wrap(orig, *args, **kwargs))


_def("mean")
_def("sum")
_def("all")
_def("amax")
_def("amin")
_def("aminmax")
_def("any")
_def("count_nonzero")
_def("logsumexp")
_def("nanmean")
_def("nansum")
_def("prod")
_def("std", keepdim_offset=2)
_def("var", keepdim_offset=2)
_def("max", single_dim=True)
_def("min", single_dim=True)
_def("argmax", single_dim=True)
_def("argmin", single_dim=True)
_def("kthvalue", single_dim=True)
_def("median", single_dim=True)
_def("nanmedian", single_dim=True)
_def("mode", single_dim=True)
_def("sort", reduce=False)
_def("argsort", reduce=False)
_def("unbind", single_dim=True)
_def("chunk", dim_offset=1, reduce=False)
_def("cummax", single_dim=True, reduce=False)
_def("cummin", single_dim=True, reduce=False)
_def("cumprod", single_dim=True, reduce=False)
_def("cumprod_", single_dim=True, reduce=False)
_def("cumsum", single_dim=True, reduce=False)
_def("cumsum_", single_dim=True, reduce=False)
_def("logcumsumexp", single_dim=True, reduce=False)
_def("renorm", dim_offset=1, single_dim=True, reduce=False)
_def("softmax", single_dim=True, reduce=False)
softmax = _wrap(torch.nn.functional.softmax, single_dim=True, reduce=False)

# stuff to handle in the future, because they require special
# binding logic for dims
# cross
# diag_embed
# diagonal
# diagonal_scatter
# diff
# nanquantile
# quantile
# roll
# rot90
# topk (new dimes on output)
# should these all be subsumed by inplace indexing?
# index_add_
# index_add
# index_copy
# index_copy_
# index_fill
# index_fill_
# index_select
# scatter
# scatter_
# scatter_add
# scatter_add_
# scatter_reduce
