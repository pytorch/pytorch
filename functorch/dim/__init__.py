from __future__ import annotations

import dis
import inspect
import sys
from collections.abc import Sequence
from typing import List, Optional, Union

import torch
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from ._dim_entry import _match_levels, DimEntry, ndim_of_levels
from ._enable_all_layers import EnableAllLayers
from ._py_inst_decoder import _PyInstDecoder
from ._tensor_info import TensorInfo


POINTWISE_OPTIMIZE = True

# Global dimension level counter (similar to C++ n_dims_created)
_n_dims_created = 0


def _relevant_op(opcode):
    """Check if opcode is relevant for variable assignment."""
    return opcode and opcode.startswith("STORE_")


def handle_from_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Handle tensor conversion for torch function integration."""
    return tensor


def _create_dim(name: str, size: Optional[int] = None):
    """Create a new Dim object."""
    return Dim(name, size if size is not None else -1)


def dims(
    n: Optional[int] = None, sizes: Optional[list[Optional[int]]] = None
) -> Union[Dim, tuple[Dim, ...]]:
    """
    Create and return one or more Dim objects.

    Uses bytecode inspection to determine variable names when possible,
    following the algorithm from functorch/csrc/dim/dim_creation.cpp

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

    # Parse arguments (equivalent to C++ argument parsing)
    if sizes is not None:
        specified_ndims = len(sizes)
    if n is not None:
        specified_ndims = n

    # Use bytecode inspection following C++ PyInstDecoder logic
    frame = inspect.currentframe().f_back
    try:
        code = frame.f_code
        lasti = frame.f_lasti

        # Create decoder following C++ pattern
        decoder = _PyInstDecoder(code, lasti)

        # Handle Python 3.11+ PRECALL instruction (like C++)
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

        # Determine final ndims (following C++ logic exactly)
        if specified_ndims == -1:
            if found_ndims == 0:
                raise SyntaxError(
                    "dims() must be assigned to a sequence of variable names or have argument n specified"
                )
            specified_ndims = found_ndims

        if found_ndims != specified_ndims:
            found_ndims = 0  # avoid taking the wrong names for dimensions

        # Generator function following C++ genobject lambda
        def genobject(i: int) -> Dim:
            nonlocal found_ndims
            name = None
            if i < found_ndims:
                name = decoder.name()

            if not name:
                name = f"d{i}"
                found_ndims = (
                    0  # once we fail at finding a name, we can't find any more
                )
            else:
                decoder.next()  # Move to next STORE instruction

            size = sizes[i] if sizes is not None else None
            return _create_dim(name, size)

        # Validate sizes parameter
        if sizes is not None and len(sizes) != specified_ndims:
            raise ValueError(f"expected {specified_ndims} sizes but found {len(sizes)}")

        # Create dimensions following C++ pattern
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

    This is the Python port of the C++ DimList class from functorch/csrc/dim/dimlist_class.cpp.

    A DimList can be in one of two states:
    1. Unbound: Created with just a name, no specific dimensions yet
    2. Bound: Either created with specific dimensions/sizes, or bound later via bind() or bind_len()
    """

    _name: str
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
        self._dims: List = []
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

    def _set_dims(self, dims: List) -> None:
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

    def __getitem__(self, key: Union[int, slice]):
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
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        if func is torch.Tensor.__getitem__:
            from functorch.dim._getsetitem import getitem

            return getitem(cls, func, types, args, kwargs)
        """
        return NotImplemented


def _create_dimlist(
    name: str, size: Optional[Union[int, List[Optional[int]]]] = None
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
    n: Optional[int] = None, sizes: Optional[List[Optional[int]]] = None
) -> Union[DimList, Tuple[DimList, ...]]:
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

    # Use bytecode inspection following dims() pattern
    frame = inspect.currentframe().f_back
    try:
        code = frame.f_code
        lasti = frame.f_lasti

        # Create decoder following C++ pattern
        decoder = _PyInstDecoder(code, lasti)

        # Handle Python 3.11+ PRECALL instruction (like C++)
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

        # Determine final ndims (following C++ logic exactly)
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
                found_ndims = (
                    0  # once we fail at finding a name, we can't find any more
                )
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


"""
def _levels_to_tuple(levels):
    return tuple(l.position() if l.is_positional() else l.dim() for l in levels)
"""


class _Tensor:
    # fast path around slow wrapping/unwrapping logic for simply queries used
    # by the implementation...

    @property
    def dims(self):
        return tuple(l.dim() for l in self._get_levels() if not l.is_positional())

    def dim(self):
        return self.ndim

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func is torch.Tensor.__getitem__:
            from functorch.dim._getsetitem import getitem

            return getitem(cls, func, types, args, kwargs)

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
                    if level not in result_levels:
                        result_levels.append(level)

        if is_pointwise:
            # Pointwise operation: match all tensors to common levels
            for i, info in enumerate(infos):
                if info:
                    tensor = info.tensor
                    if device_holding_tensor is not None and not info.has_device:
                        tensor = tensor.to(device_holding_tensor.device)
                    ml = _match_levels(tensor, info.levels, result_levels)
                    flat_args[i] = handle_from_tensor(ml)

            unflat_args, unflat_kwargs = tree_unflatten(flat_args, spec)
            result = func(*unflat_args, **unflat_kwargs)

            # Wrap tensor results
            def wrap_tensor(obj):
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
                if info:
                    batched = info.batchedtensor
                    if device_holding_tensor is not None and not info.has_device:
                        batched = batched.to(device_holding_tensor.device)
                    guard.inplace_update_layers(batched, info.levels)
                    flat_args[i] = handle_from_tensor(batched)

            breakpoint()

            unflat_args, unflat_kwargs = tree_unflatten(flat_args, spec)
            result = func(*unflat_args, **unflat_kwargs)

            # Unwrap results from functorch layers
            def unwrap_tensor(obj):
                if isinstance(obj, torch.Tensor):
                    return guard.from_batched(obj, device_holding_tensor is not None)
                return obj

            if isinstance(result, torch.Tensor):
                return unwrap_tensor(result)
            else:
                return tree_map(unwrap_tensor, result)

    def split():
        pass

    def expand():
        pass

    def index():
        pass

    def __repr__(self):
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
        return f"{tensor}\nwith dims={tuple(dims_repr)} sizes={tuple(tensor.size())}"


TensorLike = (_Tensor, torch.Tensor)


class Dim(_Tensor):
    _level: int
    _name: str
    _size: int
    _range: Optional[torch.Tensor]
    _batchtensor: Optional[torch.Tensor]

    def __init__(self, name, s: int = -1):
        global _n_dims_created
        self._name = name
        self._size = s
        self._level = _n_dims_created
        _n_dims_created += 1
        self._range = None
        self._batchtensor = None

    @classmethod
    def check_exact(cls, obj):
        return type(obj) is cls

    @property
    def size(self):
        if self._size == -1:
            raise ValueError(f"dimension {self._name} is unbound")
        return self._size

    @size.setter
    def size(self, v):
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

    def __repr__(self):
        """String representation of a Dim object."""
        return self._name

    # note that _C.Dim comes before tensor because we want the Dim API for things like size to take precedence.
    # Tensor defines format, but we want to print Dims with special formatting
    __format__ = object.__format__


class Tensor(_Tensor):
    _tensor: torch.Tensor
    _batchtensor: torch.Tensor
    _levels: list[DimEntry]
    _has_device: bool
    _delayed: callable[[], torch.Tensor]

    # NB: capture_levels is just assign to _levels

    @classmethod
    def check_exact(cls, other):
        return type(other) is cls

    @classmethod
    def from_positional(
        cls, tensor: torch.Tensor, levels: list[DimEntry], has_device: bool
    ):
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

        # Validate levels and count named dimensions (following C++ logic)
        for i, l in enumerate(levels):
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

        # If no named dimensions, return regular PyTorch tensor (optimization from C++)
        if not seen_dims:
            return tensor

        # Create Tensor object with proper level management
        result = cls()
        result._tensor = tensor
        result._levels = levels
        result._has_device = has_device
        result._batchtensor = None  # Will be created lazily if needed
        result._delayed = None

        # Validate tensor dimensionality matches levels
        assert tensor.dim() == len(levels), (
            f"Tensor has {tensor.dim()} dimensions but {len(levels)} levels provided"
        )

        # Add the ndim property that __repr__ expects
        result.ndim = ndim_of_levels(levels)

        return result

    def _get_tensor(self):
        """Get the underlying tensor, handling delayed operations if needed."""
        if hasattr(self, "_delayed") and self._delayed is not None:
            # Handle delayed operations (simplified version of C++ logic)
            # In a full implementation, this would execute the delayed operation
            pass
        return self._tensor

    def _get_levels(self):
        """Get the dimension levels."""
        return self._levels

    def _get_has_device(self):
        """Get whether this tensor has device information."""
        return self._has_device

    def _get_batchtensor(self):
        """Get the batched tensor representation, creating it lazily if needed."""
        if self._batchtensor is None:
            self._batchtensor = self._add_batch_dims(
                self._get_tensor(), self._get_levels()
            )
        return self._batchtensor

    def _add_batch_dims(self, t, levels_):
        levels = list(levels_)

        while True:
            min_real_index = -1
            min_index = -1
            min_value = float("inf")  # INT_MAX equivalent
            i = 0
            r = 0

            # Direct port of the C++ for loop
            for l in levels:
                if not l.is_none():
                    if not l.is_positional() and l.dim()._level < min_value:
                        min_value = l.dim()._level
                        min_index = i
                        min_real_index = r
                    i += 1
                r += 1

            if min_index == -1:
                return t

            # at::functorch::addBatchDim(std::move(t), min_index, min_value)
            t = torch._C._functorch._add_batch_dim(t, min_index, min_value)

            # levels[min_real_index] = DimEntry() (set to None as equivalent)
            levels[min_real_index] = DimEntry()
        return None

    def sum():
        pass

    def order(self, *dims):
        """Reorder the dimensions of this tensor."""
        from ._order import order

        return order(self, *dims)


def stack(tensors, new_dim, dim):
    """Stack tensors along a new dimension."""


def cat(tensors, dim, new_dim):
    n = dims()
    return stack(tensors, n, dim).index([n, dim], new_dim)


from functorch.dim._wrap import _wrap
from functorch.dim.wrap_type import wrap_type


wrap_type(_Tensor, torch.Tensor, _Tensor.__torch_function__)
del _Tensor.ndim


def _def(name, *args, **kwargs):
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
