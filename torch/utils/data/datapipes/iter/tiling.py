# mypy: allow-untyped-defs
from collections import OrderedDict
from collections.abc import Iterator, Sized
from enum import Enum
from typing import Any, NamedTuple, TypeVar

import torch
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe


__all__ = [
    "BoundaryMode",
    "TileInfo",
    "TileOrder",
    "TiledImageIterDataPipe",
]


_T_co = TypeVar("_T_co", covariant=True)


class TileInfo(NamedTuple):
    """Information about an extracted tile.

    Attributes:
        tile: The extracted tile tensor
        row: Row index of the tile in the grid
        col: Column index of the tile in the grid
        y_start: Starting y coordinate in the original image
        x_start: Starting x coordinate in the original image
        source_index: Index of the source image in the input datapipe
    """

    tile: torch.Tensor
    row: int
    col: int
    y_start: int
    x_start: int
    source_index: int


class TileOrder(Enum):
    """Tile ordering strategies for iteration."""

    ROW_MAJOR = "row_major"
    COLUMN_MAJOR = "column_major"
    HILBERT = "hilbert"


class BoundaryMode(Enum):
    """Boundary handling modes for edge tiles."""

    PAD = "pad"  # Pad incomplete tiles to full size
    CROP = "crop"  # Return smaller tiles at boundaries
    SKIP = "skip"  # Skip incomplete boundary tiles


def _hilbert_d2xy(n: int, d: int) -> tuple[int, int]:
    """Convert Hilbert curve index d to (x, y) coordinates.

    Args:
        n: Size of the grid (must be power of 2)
        d: Index along the Hilbert curve

    Returns:
        Tuple of (x, y) coordinates
    """
    x = y = 0
    s = 1
    while s < n:
        rx = 1 & (d // 2)
        ry = 1 & (d ^ rx)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        d //= 4
        s *= 2
    return x, y


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


def _generate_hilbert_indices(rows: int, cols: int) -> list[tuple[int, int]]:
    """Generate tile indices in Hilbert curve order.

    Args:
        rows: Number of tile rows
        cols: Number of tile columns

    Returns:
        List of (row, col) tuples in Hilbert curve order
    """
    # Use the larger dimension to determine grid size
    n = _next_power_of_2(max(rows, cols))
    indices = []
    seen = set()

    # Generate all Hilbert curve indices and filter valid ones
    for d in range(n * n):
        x, y = _hilbert_d2xy(n, d)
        if y < rows and x < cols and (y, x) not in seen:
            indices.append((y, x))
            seen.add((y, x))

    return indices


class _TileLRUCache:
    """LRU cache for tiles with configurable maximum size."""

    def __init__(self, max_size: int = 128) -> None:
        self.max_size = max_size
        self._cache: OrderedDict[tuple[int, int, int], torch.Tensor] = OrderedDict()

    def get(self, key: tuple[int, int, int]) -> torch.Tensor | None:
        """Get a tile from cache, updating access order."""
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: tuple[int, int, int], tile: torch.Tensor) -> None:
        """Store a tile in cache, evicting oldest if necessary."""
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            self._cache[key] = tile

    def clear(self) -> None:
        """Clear all cached tiles."""
        self._cache.clear()


@functional_datapipe("tile")
class TiledImageIterDataPipe(IterDataPipe[TileInfo]):
    r"""
    Extracts tiles from images in a DataPipe for memory-efficient processing of large images.

    (functional name: ``tile``).

    This DataPipe divides input images into a grid of tiles and yields them one at a time,
    enabling processing of gigapixel images with minimal memory usage. Images are expected
    to be tensors with shape (C, H, W) or (H, W).

    Args:
        datapipe: Source DataPipe yielding image tensors
        tile_size: Size of each tile as (height, width) tuple or single int for square tiles
        stride: Stride between tiles as (y_stride, x_stride) tuple or single int.
            If None, defaults to tile_size (non-overlapping tiles)
        tile_order: Order in which to yield tiles. Options:
            - TileOrder.ROW_MAJOR: Left-to-right, top-to-bottom (default)
            - TileOrder.COLUMN_MAJOR: Top-to-bottom, left-to-right
            - TileOrder.HILBERT: Hilbert curve order for better spatial locality
        boundary_mode: How to handle tiles at image boundaries. Options:
            - BoundaryMode.PAD: Pad incomplete tiles with pad_value
            - BoundaryMode.CROP: Return smaller tiles at boundaries
            - BoundaryMode.SKIP: Skip incomplete boundary tiles (default)
        pad_value: Value to use for padding when boundary_mode is PAD (default: 0)
        cache_size: Maximum number of tiles to cache (default: 0, no caching).
            Useful when tiles are accessed multiple times.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torch.utils.data.datapipes.iter import IterableWrapper, TiledImageIterDataPipe
        >>> import torch
        >>> # Create a sample image tensor (C, H, W)
        >>> image = torch.randn(3, 256, 256)
        >>> dp = IterableWrapper([image])
        >>> tiled_dp = dp.tile(tile_size=64, stride=64)
        >>> for tile_info in tiled_dp:
        ...     print(f"Tile at ({tile_info.row}, {tile_info.col}): {tile_info.tile.shape}")
        Tile at (0, 0): torch.Size([3, 64, 64])
        Tile at (0, 1): torch.Size([3, 64, 64])
        ...
    """

    datapipe: IterDataPipe
    tile_size: tuple[int, int]
    stride: tuple[int, int]
    tile_order: TileOrder
    boundary_mode: BoundaryMode
    pad_value: float
    cache_size: int

    def __init__(
        self,
        datapipe: IterDataPipe,
        tile_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        tile_order: TileOrder = TileOrder.ROW_MAJOR,
        boundary_mode: BoundaryMode = BoundaryMode.SKIP,
        pad_value: float = 0.0,
        cache_size: int = 0,
    ) -> None:
        super().__init__()
        self.datapipe = datapipe

        # Normalize tile_size to tuple
        if isinstance(tile_size, int):
            self.tile_size = (tile_size, tile_size)
        else:
            self.tile_size = tile_size

        if self.tile_size[0] <= 0 or self.tile_size[1] <= 0:
            raise ValueError("tile_size must be positive")

        # Normalize stride to tuple, default to tile_size
        if stride is None:
            self.stride = self.tile_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if self.stride[0] <= 0 or self.stride[1] <= 0:
            raise ValueError("stride must be positive")

        self.tile_order = tile_order
        self.boundary_mode = boundary_mode
        self.pad_value = pad_value
        self.cache_size = cache_size

        self._cache: _TileLRUCache | None = None
        if cache_size > 0:
            self._cache = _TileLRUCache(cache_size)

    def _get_tile_indices(
        self, num_rows: int, num_cols: int
    ) -> list[tuple[int, int]]:
        """Generate tile indices in the specified order."""
        if self.tile_order == TileOrder.ROW_MAJOR:
            return [(r, c) for r in range(num_rows) for c in range(num_cols)]
        elif self.tile_order == TileOrder.COLUMN_MAJOR:
            return [(r, c) for c in range(num_cols) for r in range(num_rows)]
        elif self.tile_order == TileOrder.HILBERT:
            return _generate_hilbert_indices(num_rows, num_cols)
        else:
            raise ValueError(f"Unknown tile order: {self.tile_order}")

    def _extract_tile(
        self,
        image: torch.Tensor,
        y_start: int,
        x_start: int,
    ) -> torch.Tensor | None:
        """Extract a tile from the image at the given position.

        Args:
            image: Source image tensor (C, H, W) or (H, W)
            y_start: Starting y coordinate
            x_start: Starting x coordinate

        Returns:
            Extracted tile tensor, or None if the tile should be skipped
        """
        tile_h, tile_w = self.tile_size

        # Handle both (C, H, W) and (H, W) formats
        if image.ndim == 3:
            _, img_h, img_w = image.shape
        elif image.ndim == 2:
            img_h, img_w = image.shape
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {image.ndim}D")

        y_end = min(y_start + tile_h, img_h)
        x_end = min(x_start + tile_w, img_w)

        actual_h = y_end - y_start
        actual_w = x_end - x_start

        is_complete = actual_h == tile_h and actual_w == tile_w

        if not is_complete:
            if self.boundary_mode == BoundaryMode.SKIP:
                return None
            elif self.boundary_mode == BoundaryMode.CROP:
                # Return the smaller tile
                if image.ndim == 3:
                    return image[:, y_start:y_end, x_start:x_end]
                else:
                    return image[y_start:y_end, x_start:x_end]
            elif self.boundary_mode == BoundaryMode.PAD:
                # Pad to full tile size
                if image.ndim == 3:
                    tile = torch.full(
                        (image.shape[0], tile_h, tile_w),
                        self.pad_value,
                        dtype=image.dtype,
                        device=image.device,
                    )
                    tile[:, :actual_h, :actual_w] = image[
                        :, y_start:y_end, x_start:x_end
                    ]
                else:
                    tile = torch.full(
                        (tile_h, tile_w),
                        self.pad_value,
                        dtype=image.dtype,
                        device=image.device,
                    )
                    tile[:actual_h, :actual_w] = image[y_start:y_end, x_start:x_end]
                return tile
        else:
            # Complete tile
            if image.ndim == 3:
                return image[:, y_start:y_end, x_start:x_end]
            else:
                return image[y_start:y_end, x_start:x_end]

    def _compute_grid_size(self, img_h: int, img_w: int) -> tuple[int, int]:
        """Compute the number of tile rows and columns for an image."""
        tile_h, tile_w = self.tile_size
        stride_y, stride_x = self.stride

        if self.boundary_mode == BoundaryMode.SKIP:
            # Only count complete tiles
            num_rows = max(0, (img_h - tile_h) // stride_y + 1)
            num_cols = max(0, (img_w - tile_w) // stride_x + 1)
        else:
            # Count all tiles including partial ones
            num_rows = max(0, (img_h - 1) // stride_y + 1) if img_h > 0 else 0
            num_cols = max(0, (img_w - 1) // stride_x + 1) if img_w > 0 else 0

        return num_rows, num_cols

    def __iter__(self) -> Iterator[TileInfo]:
        source_index = 0
        for image in self.datapipe:
            if not isinstance(image, torch.Tensor):
                raise TypeError(
                    f"Expected torch.Tensor, got {type(image).__name__}"
                )

            # Get image dimensions
            if image.ndim == 3:
                _, img_h, img_w = image.shape
            elif image.ndim == 2:
                img_h, img_w = image.shape
            else:
                raise ValueError(f"Expected 2D or 3D tensor, got {image.ndim}D")

            num_rows, num_cols = self._compute_grid_size(img_h, img_w)
            tile_indices = self._get_tile_indices(num_rows, num_cols)

            stride_y, stride_x = self.stride

            for row, col in tile_indices:
                y_start = row * stride_y
                x_start = col * stride_x

                cache_key = (source_index, row, col)

                # Check cache first
                tile = None
                if self._cache is not None:
                    tile = self._cache.get(cache_key)

                if tile is None:
                    tile = self._extract_tile(image, y_start, x_start)
                    if tile is None:
                        continue  # Skip this tile

                    # Store in cache
                    if self._cache is not None:
                        self._cache.put(cache_key, tile)

                yield TileInfo(
                    tile=tile,
                    row=row,
                    col=col,
                    y_start=y_start,
                    x_start=x_start,
                    source_index=source_index,
                )

            source_index += 1

    def __len__(self) -> int:
        """Return total number of tiles across all images.

        Note: This requires iterating through the source datapipe to get image
        dimensions, which may be expensive. Only available if the source
        datapipe has a known length.
        """
        if not isinstance(self.datapipe, Sized):
            raise TypeError(
                f"{type(self).__name__} instance doesn't have valid length"
            )

        # We cannot easily compute the length without knowing image sizes
        # This would require iterating through all images
        raise TypeError(
            f"{type(self).__name__} length cannot be determined without "
            "iterating through images to get their dimensions"
        )

    def reset(self) -> None:
        """Reset the DataPipe state."""
        if self._cache is not None:
            self._cache.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = {
            "datapipe": self.datapipe,
            "tile_size": self.tile_size,
            "stride": self.stride,
            "tile_order": self.tile_order,
            "boundary_mode": self.boundary_mode,
            "pad_value": self.pad_value,
            "cache_size": self.cache_size,
            "_valid_iterator_id": self._valid_iterator_id,
            "_number_of_samples_yielded": self._number_of_samples_yielded,
        }
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.datapipe = state["datapipe"]
        self.tile_size = state["tile_size"]
        self.stride = state["stride"]
        self.tile_order = state["tile_order"]
        self.boundary_mode = state["boundary_mode"]
        self.pad_value = state["pad_value"]
        self.cache_size = state["cache_size"]
        self._valid_iterator_id = state["_valid_iterator_id"]
        self._number_of_samples_yielded = state["_number_of_samples_yielded"]

        # Recreate cache
        self._cache = None
        if self.cache_size > 0:
            self._cache = _TileLRUCache(self.cache_size)
