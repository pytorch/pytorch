"""Coordinate-based sampling for PyTorch DataLoaders.

This module provides samplers that yield floating-point coordinates instead
of integer indices. This is useful for applications where data is naturally
indexed by continuous coordinates, such as:

- Geographic Information Systems (GIS)
- Medical imaging (sub-pixel locations)
- Scientific simulations
- Satellite imagery
- Computer graphics

The coordinate samplers follow the same interface as standard PyTorch samplers
but yield coordinate tuples (float, float) or (float, float, float) instead
of integers.

Example:
    Basic usage with a coordinate-aware dataset::

        >>> from torch.utils.data import DataLoader
        >>> from torch.utils.data import CoordinateSampler
        >>>
        >>> # Define coordinates
        >>> coords = [(lat, lon) for lat, lon in zip(latitudes, longitudes)]
        >>>
        >>> # Create sampler
        >>> sampler = CoordinateSampler(coords, shuffle=True)
        >>>
        >>> # Use with DataLoader
        >>> loader = DataLoader(
        ...     dataset,  # Dataset.__getitem__ accepts (float, float)
        ...     sampler=sampler,
        ...     batch_size=32,
        ...     collate_fn=custom_collate  # Handle coordinate batches
        ... )

    Grid-based sampling::

        >>> # Sample on a regular grid
        >>> sampler = GridCoordinateSampler(
        ...     bounds=(0.0, 0.0, 100.0, 100.0),
        ...     grid_size=(50, 50),
        ...     mode='regular'
        ... )

    Weighted sampling::

        >>> # Sample coordinates based on importance weights
        >>> sampler = WeightedCoordinateSampler(
        ...     coordinates=coords,
        ...     weights=importance_scores,
        ...     num_samples=1000,
        ...     replacement=True
        ... )

    Distributed training::

        >>> # Split coordinates across GPUs
        >>> sampler = DistributedCoordinateSampler(
        ...     coords,
        ...     num_replicas=world_size,
        ...     rank=rank
        ... )
        >>> # Remember to set epoch for proper shuffling
        >>> sampler.set_epoch(epoch)

Note:
    When using coordinate samplers, your Dataset class must accept coordinate
    tuples in its __getitem__ method instead of integer indices.
"""

import math
from typing import Iterator, Tuple, List, Optional, TypeVar, Generic, Sequence, Union
import torch
from torch.utils.data import Sampler

# Define coordinate type
Coordinate2D = Tuple[float, float]
Coordinate3D = Tuple[float, float, float]

# Make covariant type for coordinates
_CoordType = TypeVar("_CoordType", Coordinate2D, Coordinate3D, covariant=True)


class CoordinateSampler(Sampler[Coordinate2D]):
    r"""Samples floating-point coordinates.

    This sampler yields 2D coordinates as (x, y) tuples of floats.
    Useful for applications where data is indexed by continuous coordinates
    rather than discrete indices.

    Args:
        coordinates (List[Tuple[float, float]]): List of (x, y) coordinate tuples
        shuffle (bool): If ``True``, shuffle coordinates each epoch. Default: ``True``
        seed (Optional[int]): Random seed for shuffling. Default: ``None``
        generator (Optional[torch.Generator]): Generator for random sampling

    Example:
        >>> coords = [(1.5, 2.5), (3.0, 4.0), (5.5, 6.5)]
        >>> sampler = CoordinateSampler(coords, shuffle=True, seed=42)
        >>> for coord in sampler:
        ...     print(coord)  # Yields (float, float) tuples

        >>> # Use with DataLoader
        >>> loader = DataLoader(dataset, sampler=sampler, batch_size=2)
    """

    coordinates: List[Coordinate2D]

    def __init__(
        self,
        coordinates: List[Coordinate2D],
        shuffle: bool = True,
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None
    ) -> None:
        self.coordinates = coordinates
        self.shuffle = shuffle
        self.seed = seed
        self.generator = generator

        # Validate coordinates
        if not coordinates:
            raise ValueError("coordinates list cannot be empty")

        # Check all coordinates are valid tuples
        for coord in coordinates:
            if not isinstance(coord, tuple) or len(coord) != 2:
                raise ValueError(f"Invalid coordinate: {coord}. Expected (float, float) tuple.")

    def __iter__(self) -> Iterator[Coordinate2D]:
        """Iterate over coordinates."""
        n = len(self.coordinates)

        if self.shuffle:
            # Create generator for reproducibility
            if self.generator is None:
                generator = torch.Generator()
                if self.seed is not None:
                    generator.manual_seed(self.seed)
            else:
                generator = self.generator

            # Generate random permutation
            indices = torch.randperm(n, generator=generator).tolist()

            # Yield coordinates in shuffled order
            for i in indices:
                yield self.coordinates[i]
        else:
            # Yield coordinates in original order
            for coord in self.coordinates:
                yield coord

    def __len__(self) -> int:
        """Return the number of coordinates."""
        return len(self.coordinates)


class CoordinateSampler3D(Sampler[Coordinate3D]):
    r"""Samples 3D floating-point coordinates.

    Similar to CoordinateSampler but for 3D coordinates (x, y, z).

    Args:
        coordinates (List[Tuple[float, float, float]]): List of (x, y, z) coordinate tuples
        shuffle (bool): If ``True``, shuffle coordinates each epoch. Default: ``True``
        seed (Optional[int]): Random seed for shuffling. Default: ``None``
        generator (Optional[torch.Generator]): Generator for random sampling

    Example:
        >>> coords = [(1.5, 2.5, 3.5), (4.0, 5.0, 6.0)]
        >>> sampler = CoordinateSampler3D(coords, shuffle=False)
    """

    coordinates: List[Coordinate3D]

    def __init__(
        self,
        coordinates: List[Coordinate3D],
        shuffle: bool = True,
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None
    ) -> None:
        self.coordinates = coordinates
        self.shuffle = shuffle
        self.seed = seed
        self.generator = generator

        # Validate coordinates
        if not coordinates:
            raise ValueError("coordinates list cannot be empty")

        for coord in coordinates:
            if not isinstance(coord, tuple) or len(coord) != 3:
                raise ValueError(f"Invalid coordinate: {coord}. Expected (float, float, float) tuple.")

    def __iter__(self) -> Iterator[Coordinate3D]:
        """Iterate over 3D coordinates."""
        n = len(self.coordinates)

        if self.shuffle:
            if self.generator is None:
                generator = torch.Generator()
                if self.seed is not None:
                    generator.manual_seed(self.seed)
            else:
                generator = self.generator

            indices = torch.randperm(n, generator=generator).tolist()

            for i in indices:
                yield self.coordinates[i]
        else:
            for coord in self.coordinates:
                yield coord

    def __len__(self) -> int:
        """Return the number of coordinates."""
        return len(self.coordinates)


class GridCoordinateSampler(Sampler[Coordinate2D]):
    r"""Generates coordinates on a regular or random grid.

    Creates a grid of coordinates within specified bounds. Useful for
    systematic sampling of continuous spaces.

    Args:
        bounds (Tuple[float, float, float, float]): Bounding box as (x_min, y_min, x_max, y_max)
        grid_size (Tuple[int, int]): Number of points as (nx, ny)
        mode (str): Grid mode - 'regular' or 'random'. Default: 'regular'
        shuffle (bool): If ``True``, shuffle grid points. Default: ``False``
        include_edges (bool): Include boundary points in regular grid. Default: ``True``
        generator (Optional[torch.Generator]): Generator for random sampling

    Example:
        >>> # Regular 10x10 grid in [0, 100] x [0, 100]
        >>> sampler = GridCoordinateSampler(
        ...     bounds=(0.0, 0.0, 100.0, 100.0),
        ...     grid_size=(10, 10),
        ...     mode='regular'
        ... )
        >>> coords = list(sampler)  # 100 evenly spaced points

        >>> # Random points in same bounds
        >>> sampler = GridCoordinateSampler(
        ...     bounds=(0.0, 0.0, 100.0, 100.0),
        ...     grid_size=(10, 10),
        ...     mode='random'
        ... )
    """

    def __init__(
        self,
        bounds: Tuple[float, float, float, float],
        grid_size: Tuple[int, int],
        mode: str = 'regular',
        shuffle: bool = False,
        include_edges: bool = True,
        generator: Optional[torch.Generator] = None
    ) -> None:
        self.bounds = bounds
        self.grid_size = grid_size
        self.mode = mode
        self.shuffle = shuffle
        self.include_edges = include_edges
        self.generator = generator

        # Validate inputs
        x_min, y_min, x_max, y_max = bounds
        if x_min >= x_max or y_min >= y_max:
            raise ValueError(f"Invalid bounds: {bounds}")

        nx, ny = grid_size
        if nx <= 0 or ny <= 0:
            raise ValueError(f"Grid size must be positive: {grid_size}")

        if mode not in ['regular', 'random']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'regular' or 'random'")

        # Generate coordinates
        self.coordinates = self._generate_grid()

    def _generate_grid(self) -> List[Coordinate2D]:
        """Generate grid coordinates based on mode."""
        x_min, y_min, x_max, y_max = self.bounds
        nx, ny = self.grid_size

        if self.mode == 'regular':
            # Create regular grid
            if self.include_edges:
                x_coords = torch.linspace(x_min, x_max, nx)
                y_coords = torch.linspace(y_min, y_max, ny)
            else:
                # Exclude edges
                dx = (x_max - x_min) / (nx + 1)
                dy = (y_max - y_min) / (ny + 1)
                x_coords = torch.linspace(x_min + dx, x_max - dx, nx)
                y_coords = torch.linspace(y_min + dy, y_max - dy, ny)

            # Create all coordinate pairs
            coordinates = []
            for y in y_coords.tolist():
                for x in x_coords.tolist():
                    coordinates.append((x, y))

        else:  # random mode
            generator = self.generator or torch.Generator()

            # Generate random coordinates
            x_coords = torch.rand(nx * ny, generator=generator) * (x_max - x_min) + x_min
            y_coords = torch.rand(nx * ny, generator=generator) * (y_max - y_min) + y_min

            coordinates = list(zip(x_coords.tolist(), y_coords.tolist()))

        return coordinates

    def __iter__(self) -> Iterator[Coordinate2D]:
        """Iterate over grid coordinates."""
        if self.shuffle:
            generator = self.generator or torch.Generator()
            indices = torch.randperm(len(self.coordinates), generator=generator)
            for i in indices:
                yield self.coordinates[i]
        else:
            for coord in self.coordinates:
                yield coord

    def __len__(self) -> int:
        """Return total number of grid points."""
        return len(self.coordinates)


class WeightedCoordinateSampler(Sampler[Coordinate2D]):
    r"""Samples coordinates according to weights.

    Similar to WeightedRandomSampler but for coordinates.

    Args:
        coordinates (List[Tuple[float, float]]): List of coordinate tuples
        weights (Sequence[float]): Weight for each coordinate
        num_samples (int): Number of samples to draw
        replacement (bool): Sample with replacement. Default: ``True``
        generator (Optional[torch.Generator]): Generator for random sampling

    Example:
        >>> coords = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
        >>> weights = [0.1, 0.3, 0.6]  # Prefer later coordinates
        >>> sampler = WeightedCoordinateSampler(
        ...     coords, weights, num_samples=100, replacement=True
        ... )
    """

    def __init__(
        self,
        coordinates: List[Coordinate2D],
        weights: Sequence[float],
        num_samples: int,
        replacement: bool = True,
        generator: Optional[torch.Generator] = None
    ) -> None:
        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError(f"num_samples should be a positive integer, got {num_samples}")

        if not isinstance(replacement, bool):
            raise ValueError(f"replacement should be boolean, got {replacement}")

        if len(coordinates) != len(weights):
            raise ValueError("coordinates and weights must have same length")

        self.coordinates = coordinates
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator

        if not self.replacement and self.num_samples > len(self.coordinates):
            raise ValueError("Cannot sample more than len(coordinates) without replacement")

    def __iter__(self) -> Iterator[Coordinate2D]:
        """Sample coordinates according to weights."""
        indices = torch.multinomial(
            self.weights,
            self.num_samples,
            self.replacement,
            generator=self.generator
        ).tolist()

        for idx in indices:
            yield self.coordinates[idx]

    def __len__(self) -> int:
        """Return number of samples."""
        return self.num_samples


class CoordinateBatchSampler(Sampler[List[Coordinate2D]]):
    r"""Wraps a CoordinateSampler to yield batches of coordinates.

    Similar to BatchSampler but specifically for coordinate samplers.
    Yields lists of coordinates instead of single coordinates.

    Args:
        sampler (Sampler): Base sampler yielding coordinates
        batch_size (int): Size of each batch
        drop_last (bool): Drop incomplete last batch. Default: ``False``
        spatial_sorting (bool): Sort coordinates spatially within batch. Default: ``False``

    Example:
        >>> coords = [(i * 0.1, j * 0.1) for i in range(10) for j in range(10)]
        >>> coord_sampler = CoordinateSampler(coords, shuffle=True)
        >>> batch_sampler = CoordinateBatchSampler(coord_sampler, batch_size=16)
        >>> for batch in batch_sampler:
        ...     print(f"Batch of {len(batch)} coordinates")
    """

    def __init__(
        self,
        sampler: Sampler,
        batch_size: int,
        drop_last: bool = False,
        spatial_sorting: bool = False
    ) -> None:
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size must be positive integer, got {batch_size}")

        if not isinstance(drop_last, bool):
            raise ValueError(f"drop_last must be boolean, got {drop_last}")

        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.spatial_sorting = spatial_sorting

    def __iter__(self) -> Iterator[List[Coordinate2D]]:
        """Yield batches of coordinates."""
        batch = []
        for coord in self.sampler:
            batch.append(coord)
            if len(batch) == self.batch_size:
                if self.spatial_sorting:
                    batch = self._sort_spatially(batch)
                yield batch
                batch = []

        # Handle last incomplete batch
        if batch and not self.drop_last:
            if self.spatial_sorting:
                batch = self._sort_spatially(batch)
            yield batch

    def _sort_spatially(self, batch: List[Coordinate2D]) -> List[Coordinate2D]:
        """Sort coordinates by spatial proximity (simple distance from origin)."""
        # Simple sorting by distance from origin
        # More sophisticated methods could use Hilbert curves or KD-trees
        return sorted(batch, key=lambda c: math.sqrt(c[0]**2 + c[1]**2))

    def __len__(self) -> int:
        """Return number of batches."""
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class SpatialBatchSampler(Sampler[List[Coordinate2D]]):
    r"""Groups spatially nearby coordinates into batches.

    Creates batches where coordinates within each batch are spatially close.
    Useful for improving cache locality and reducing data loading overhead.

    Args:
        coordinates (List[Tuple[float, float]]): All coordinates
        batch_size (int): Target size of each batch
        shuffle_batches (bool): Shuffle batch order. Default: ``True``
        generator (Optional[torch.Generator]): Random generator

    Note:
        This sampler first clusters coordinates spatially, then yields
        batches from each cluster. This is different from random batching
        and can significantly improve performance for spatially-coherent data.

    Example:
        >>> coords = [(i * 0.1, j * 0.1) for i in range(100) for j in range(100)]
        >>> sampler = SpatialBatchSampler(coords, batch_size=32)
        >>> # Each batch contains spatially nearby coordinates
    """

    def __init__(
        self,
        coordinates: List[Coordinate2D],
        batch_size: int,
        shuffle_batches: bool = True,
        generator: Optional[torch.Generator] = None
    ) -> None:
        self.coordinates = coordinates
        self.batch_size = batch_size
        self.shuffle_batches = shuffle_batches
        self.generator = generator

        # Create spatial batches
        self.batches = self._create_spatial_batches()

    def _create_spatial_batches(self) -> List[List[int]]:
        """Create batches of spatially nearby coordinates."""
        n = len(self.coordinates)

        # Simple grid-based clustering
        # Find bounds
        xs = [c[0] for c in self.coordinates]
        ys = [c[1] for c in self.coordinates]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # Create grid cells
        n_cells = max(1, n // self.batch_size)
        grid_size = int(math.sqrt(n_cells))

        # Assign coordinates to grid cells
        cells = {}
        for idx, (x, y) in enumerate(self.coordinates):
            # Normalize to [0, 1]
            nx = (x - x_min) / (x_max - x_min + 1e-10)
            ny = (y - y_min) / (y_max - y_min + 1e-10)

            # Get cell indices
            cell_x = min(int(nx * grid_size), grid_size - 1)
            cell_y = min(int(ny * grid_size), grid_size - 1)
            cell_key = (cell_x, cell_y)

            if cell_key not in cells:
                cells[cell_key] = []
            cells[cell_key].append(idx)

        # Create batches from cells
        batches = []
        current_batch = []

        for cell_indices in cells.values():
            for idx in cell_indices:
                current_batch.append(idx)
                if len(current_batch) >= self.batch_size:
                    batches.append(current_batch)
                    current_batch = []

        # Add remaining indices
        if current_batch:
            batches.append(current_batch)

        return batches

    def __iter__(self) -> Iterator[List[Coordinate2D]]:
        """Yield batches of coordinates."""
        if self.shuffle_batches:
            generator = self.generator or torch.Generator()
            indices = torch.randperm(len(self.batches), generator=generator)
            batch_order = [self.batches[i] for i in indices]
        else:
            batch_order = self.batches

        for batch_indices in batch_order:
            batch_coords = [self.coordinates[i] for i in batch_indices]
            yield batch_coords

    def __len__(self) -> int:
        """Return number of batches."""
        return len(self.batches)


class DistributedCoordinateSampler(Sampler[Coordinate2D]):
    r"""Distributed sampler for coordinates.

    Splits coordinates across distributed processes. Each process gets
    a subset of coordinates, ensuring no overlap.

    Args:
        coordinates (List[Tuple[float, float]]): List of all coordinates
        num_replicas (Optional[int]): Number of processes. Default: world_size
        rank (Optional[int]): Rank of current process. Default: current rank
        shuffle (bool): Shuffle coordinates each epoch. Default: ``True``
        seed (int): Random seed for shuffling. Default: 0
        drop_last (bool): Drop tail to make even split. Default: ``False``

    Example:
        >>> # In distributed training
        >>> sampler = DistributedCoordinateSampler(
        ...     coordinates,
        ...     num_replicas=world_size,
        ...     rank=rank
        ... )
        >>> loader = DataLoader(dataset, sampler=sampler)
    """

    def __init__(
        self,
        coordinates: List[Coordinate2D],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False
    ) -> None:
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package")
            if not torch.distributed.is_initialized():
                raise RuntimeError("Default process group not initialized")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package")
            if not torch.distributed.is_initialized():
                raise RuntimeError("Default process group not initialized")
            rank = torch.distributed.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, should be in [0, {num_replicas})")

        self.coordinates = coordinates
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed

        # Calculate samples per replica
        if self.drop_last and len(self.coordinates) % self.num_replicas != 0:
            # Drop tail to make even split
            self.num_samples = math.ceil(
                (len(self.coordinates) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.coordinates) / self.num_replicas)

        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[Coordinate2D]:
        """Iterate over coordinates for this rank."""
        if self.shuffle:
            # Deterministic shuffling based on epoch
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.coordinates), generator=g).tolist()
            coords = [self.coordinates[i] for i in indices]
        else:
            coords = self.coordinates[:]

        # Add extra samples to make evenly divisible
        if not self.drop_last:
            padding_size = self.total_size - len(coords)
            if padding_size > 0:
                coords += coords[:padding_size]
        else:
            coords = coords[:self.total_size]

        # Subsample for this rank
        coords = coords[self.rank:self.total_size:self.num_replicas]

        return iter(coords)

    def __len__(self) -> int:
        """Return number of samples for this rank."""
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""Set epoch for shuffling.

        Args:
            epoch (int): Current epoch number

        This should be called at the beginning of each epoch in distributed
        training to ensure proper shuffling across epochs.
        """
        self.epoch = epoch


def create_coordinate_sampler(
    coordinates: Union[List[Coordinate2D], torch.Tensor],
    sampler_type: str = 'random',
    **kwargs
) -> Sampler:
    """Factory function to create coordinate samplers.

    Args:
        coordinates: List of coordinates or tensor of shape (N, 2)
        sampler_type: Type of sampler ('random', 'sequential', 'grid', 'weighted')
        **kwargs: Additional arguments for specific sampler

    Returns:
        Appropriate coordinate sampler instance

    Example:
        >>> coords = torch.rand(1000, 2) * 100  # Random coords in [0, 100]^2
        >>> sampler = create_coordinate_sampler(
        ...     coords, 'random', shuffle=True, seed=42
        ... )
    """
    # Convert tensor to list of tuples if needed
    if isinstance(coordinates, torch.Tensor):
        if coordinates.dim() != 2 or coordinates.size(1) not in [2, 3]:
            raise ValueError(f"Expected tensor of shape (N, 2) or (N, 3), got {coordinates.shape}")
        coordinates = [tuple(coord.tolist()) for coord in coordinates]

    if sampler_type == 'random':
        return CoordinateSampler(coordinates, shuffle=True, **kwargs)
    elif sampler_type == 'sequential':
        return CoordinateSampler(coordinates, shuffle=False, **kwargs)
    elif sampler_type == 'grid':
        if 'bounds' not in kwargs or 'grid_size' not in kwargs:
            raise ValueError("Grid sampler requires 'bounds' and 'grid_size'")
        return GridCoordinateSampler(**kwargs)
    elif sampler_type == 'weighted':
        if 'weights' not in kwargs:
            raise ValueError("Weighted sampler requires 'weights'")
        return WeightedCoordinateSampler(coordinates, **kwargs)
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")


def coordinates_to_tensor(coords: List[Union[Coordinate2D, Coordinate3D]]) -> torch.Tensor:
    """Convert list of coordinate tuples to tensor.

    Args:
        coords: List of coordinate tuples

    Returns:
        Tensor of shape (N, 2) or (N, 3)

    Example:
        >>> coords = [(1.0, 2.0), (3.0, 4.0)]
        >>> tensor = coordinates_to_tensor(coords)
        >>> print(tensor.shape)  # torch.Size([2, 2])
    """
    return torch.tensor(coords, dtype=torch.float32)


def tensor_to_coordinates(
    tensor: torch.Tensor
) -> Union[List[Coordinate2D], List[Coordinate3D]]:
    """Convert tensor to list of coordinate tuples.

    Args:
        tensor: Tensor of shape (N, 2) or (N, 3)

    Returns:
        List of coordinate tuples

    Example:
        >>> tensor = torch.rand(10, 2)
        >>> coords = tensor_to_coordinates(tensor)
        >>> print(len(coords))  # 10
    """
    if tensor.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {tensor.dim()}D")

    if tensor.size(1) == 2:
        return [(float(row[0]), float(row[1])) for row in tensor]
    elif tensor.size(1) == 3:
        return [(float(row[0]), float(row[1]), float(row[2])) for row in tensor]
    else:
        raise ValueError(f"Expected tensor with 2 or 3 columns, got {tensor.size(1)}")