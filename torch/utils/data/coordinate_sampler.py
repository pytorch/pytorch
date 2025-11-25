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