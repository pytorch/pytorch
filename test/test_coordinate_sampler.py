import unittest
import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import (
    CoordinateSampler,
    CoordinateSampler3D,
    GridCoordinateSampler,
    WeightedCoordinateSampler,
    CoordinateBatchSampler,
    SpatialBatchSampler,
    DistributedCoordinateSampler,
    create_coordinate_sampler,
    coordinates_to_tensor,
    tensor_to_coordinates,
)


class CoordinateDataset(Dataset):
    """Test dataset that accepts coordinate indices."""

    def __init__(self, size: int = 100):
        self.size = size
        # Create fake data grid
        self.data = torch.randn(size, size, 3)

    def __getitem__(self, index):
        """Accept (float, float) coordinate and return interpolated value."""
        if isinstance(index, tuple) and len(index) == 2:
            x, y = index
            # Simple nearest neighbor for testing
            x_idx = int(min(max(x, 0), self.size - 1))
            y_idx = int(min(max(y, 0), self.size - 1))
            return self.data[y_idx, x_idx]
        else:
            raise ValueError(f"Expected (x, y) tuple, got {index}")

    def __len__(self):
        return self.size * self.size


class TestCoordinateSampler(TestCase):
    """Test basic coordinate sampler."""

    def setUp(self):
        """Create test coordinates."""
        self.coords_2d = [(i * 0.5, j * 0.5) for i in range(10) for j in range(10)]
        self.coords_3d = [(i * 0.5, j * 0.5, k * 0.5)
                          for i in range(5) for j in range(5) for k in range(5)]

    def test_basic_iteration(self):
        """Test basic iteration over coordinates."""
        sampler = CoordinateSampler(self.coords_2d, shuffle=False)
        coords = list(sampler)

        self.assertEqual(len(coords), 100)
        self.assertEqual(coords[0], (0.0, 0.0))
        self.assertEqual(coords[-1], (4.5, 4.5))

        # Check all are tuples
        for coord in coords:
            self.assertIsInstance(coord, tuple)
            self.assertEqual(len(coord), 2)

    def test_shuffling(self):
        """Test coordinate shuffling."""
        sampler1 = CoordinateSampler(self.coords_2d, shuffle=True, seed=42)
        sampler2 = CoordinateSampler(self.coords_2d, shuffle=True, seed=42)
        sampler3 = CoordinateSampler(self.coords_2d, shuffle=True, seed=24)

        coords1 = list(sampler1)
        coords2 = list(sampler2)
        coords3 = list(sampler3)

        # Same seed should give same order
        self.assertEqual(coords1, coords2)
        # Different seed should give different order
        self.assertNotEqual(coords1, coords3)
        # But same elements
        self.assertEqual(set(coords1), set(coords3))

    def test_3d_coordinates(self):
        """Test 3D coordinate sampler."""
        sampler = CoordinateSampler3D(self.coords_3d, shuffle=False)
        coords = list(sampler)

        self.assertEqual(len(coords), 125)
        # Check all are 3D tuples
        for coord in coords:
            self.assertIsInstance(coord, tuple)
            self.assertEqual(len(coord), 3)

    def test_with_dataloader(self):
        """Test integration with DataLoader."""
        dataset = CoordinateDataset(100)
        sampler = CoordinateSampler(self.coords_2d, shuffle=True)

        loader = DataLoader(dataset, sampler=sampler, batch_size=None)

        # Should be able to iterate
        count = 0
        for data in loader:
            self.assertEqual(data.shape, (3,))  # Single sample
            count += 1

        self.assertEqual(count, len(self.coords_2d))

    def test_precision(self):
        """Test that float precision is maintained."""
        # Create coordinates with many decimal places
        precise_coords = [(0.123456789, 0.987654321), (1.111111111, 2.222222222)]
        sampler = CoordinateSampler(precise_coords, shuffle=False)

        coords = list(sampler)
        # Check precision is maintained
        self.assertEqual(coords[0][0], 0.123456789)
        self.assertEqual(coords[0][1], 0.987654321)

    def test_empty_coordinates_raises(self):
        """Test that empty coordinates raise ValueError."""
        with self.assertRaises(ValueError):
            CoordinateSampler([])

    def test_invalid_coordinates_raises(self):
        """Test that invalid coordinates raise ValueError."""
        with self.assertRaises(ValueError):
            CoordinateSampler([(1.0, 2.0), (3.0,)])  # Wrong tuple size

        with self.assertRaises(ValueError):
            CoordinateSampler([(1.0, 2.0), [3.0, 4.0]])  # List instead of tuple


class TestGridCoordinateSampler(TestCase):
    """Test grid-based coordinate sampler."""

    def test_regular_grid(self):
        """Test regular grid generation."""
        sampler = GridCoordinateSampler(
            bounds=(0.0, 0.0, 10.0, 10.0),
            grid_size=(5, 5),
            mode='regular',
            include_edges=True
        )

        coords = list(sampler)
        self.assertEqual(len(coords), 25)

        # Check corners
        self.assertIn((0.0, 0.0), coords)
        self.assertIn((10.0, 10.0), coords)

        # Check spacing
        x_coords = sorted(set(c[0] for c in coords))
        self.assertEqual(len(x_coords), 5)
        self.assertAlmostEqual(x_coords[1] - x_coords[0], 2.5, places=5)

    def test_random_grid(self):
        """Test random grid generation."""
        sampler = GridCoordinateSampler(
            bounds=(0.0, 0.0, 10.0, 10.0),
            grid_size=(10, 10),
            mode='random'
        )

        coords = list(sampler)
        self.assertEqual(len(coords), 100)

        # All should be within bounds
        for x, y in coords:
            self.assertGreaterEqual(x, 0.0)
            self.assertLessEqual(x, 10.0)
            self.assertGreaterEqual(y, 0.0)
            self.assertLessEqual(y, 10.0)

    def test_exclude_edges(self):
        """Test grid with excluded edges."""
        sampler = GridCoordinateSampler(
            bounds=(0.0, 0.0, 10.0, 10.0),
            grid_size=(5, 5),
            mode='regular',
            include_edges=False
        )

        coords = list(sampler)
        self.assertEqual(len(coords), 25)

        # Corners should NOT be present
        self.assertNotIn((0.0, 0.0), coords)
        self.assertNotIn((10.0, 10.0), coords)

    def test_invalid_bounds_raises(self):
        """Test that invalid bounds raise ValueError."""
        with self.assertRaises(ValueError):
            GridCoordinateSampler(
                bounds=(10.0, 0.0, 0.0, 10.0),  # x_min > x_max
                grid_size=(5, 5)
            )

    def test_invalid_mode_raises(self):
        """Test that invalid mode raises ValueError."""
        with self.assertRaises(ValueError):
            GridCoordinateSampler(
                bounds=(0.0, 0.0, 10.0, 10.0),
                grid_size=(5, 5),
                mode='invalid'
            )


class TestWeightedCoordinateSampler(TestCase):
    """Test weighted coordinate sampling."""

    def test_weighted_sampling(self):
        """Test that weights affect sampling distribution."""
        coords = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
        weights = [0.1, 0.1, 0.8]  # Heavily favor last coordinate

        sampler = WeightedCoordinateSampler(
            coords, weights, num_samples=1000, replacement=True
        )

        samples = list(sampler)
        self.assertEqual(len(samples), 1000)

        # Count occurrences
        from collections import Counter
        counts = Counter(samples)

        # Last coordinate should appear most frequently
        self.assertGreater(counts[(2.0, 2.0)], counts[(0.0, 0.0)])
        self.assertGreater(counts[(2.0, 2.0)], counts[(1.0, 1.0)])

        # Should be roughly 80% of samples (with some variance)
        self.assertGreater(counts[(2.0, 2.0)], 700)
        self.assertLess(counts[(2.0, 2.0)], 900)

    def test_without_replacement(self):
        """Test sampling without replacement."""
        coords = [(i, i) for i in range(10)]
        weights = [1.0] * 10

        sampler = WeightedCoordinateSampler(
            coords, weights, num_samples=10, replacement=False
        )

        samples = list(sampler)
        self.assertEqual(len(samples), 10)
        # Should have all unique samples
        self.assertEqual(len(set(samples)), 10)

    def test_invalid_num_samples_raises(self):
        """Test that invalid num_samples raises ValueError."""
        coords = [(0.0, 0.0), (1.0, 1.0)]
        weights = [0.5, 0.5]

        with self.assertRaises(ValueError):
            WeightedCoordinateSampler(coords, weights, num_samples=0)

        with self.assertRaises(ValueError):
            WeightedCoordinateSampler(coords, weights, num_samples=-1)

    def test_mismatched_lengths_raises(self):
        """Test that mismatched coordinates/weights lengths raise ValueError."""
        coords = [(0.0, 0.0), (1.0, 1.0)]
        weights = [0.5, 0.3, 0.2]  # Wrong length

        with self.assertRaises(ValueError):
            WeightedCoordinateSampler(coords, weights, num_samples=10)


class TestBatchSamplers(TestCase):
    """Test batch samplers for coordinates."""

    def test_coordinate_batch_sampler(self):
        """Test basic batch sampler."""
        coords = [(i, j) for i in range(10) for j in range(10)]
        coord_sampler = CoordinateSampler(coords, shuffle=False)
        batch_sampler = CoordinateBatchSampler(coord_sampler, batch_size=16)

        batches = list(batch_sampler)
        self.assertEqual(len(batches), 7)  # 100 coords / 16 = 6.25 -> 7 batches

        # First 6 batches should be full
        for i in range(6):
            self.assertEqual(len(batches[i]), 16)

        # Last batch should have 4 coords
        self.assertEqual(len(batches[6]), 4)

    def test_drop_last(self):
        """Test dropping incomplete last batch."""
        coords = [(i, j) for i in range(10) for j in range(10)]
        coord_sampler = CoordinateSampler(coords, shuffle=False)
        batch_sampler = CoordinateBatchSampler(
            coord_sampler, batch_size=16, drop_last=True
        )

        batches = list(batch_sampler)
        self.assertEqual(len(batches), 6)  # Dropped last incomplete batch

        # All batches should be full
        for batch in batches:
            self.assertEqual(len(batch), 16)

    def test_spatial_sorting(self):
        """Test spatial sorting within batches."""
        coords = [(5.0, 5.0), (1.0, 1.0), (3.0, 3.0), (0.0, 0.0)]
        coord_sampler = CoordinateSampler(coords, shuffle=False)
        batch_sampler = CoordinateBatchSampler(
            coord_sampler, batch_size=4, spatial_sorting=True
        )

        batches = list(batch_sampler)
        self.assertEqual(len(batches), 1)

        # Should be sorted by distance from origin
        batch = batches[0]
        self.assertEqual(batch[0], (0.0, 0.0))
        self.assertEqual(batch[-1], (5.0, 5.0))

    def test_spatial_batch_sampler(self):
        """Test spatial clustering batch sampler."""
        # Create coords in distinct spatial clusters
        coords = []
        # Cluster 1: around (0, 0)
        coords.extend([(i * 0.1, j * 0.1) for i in range(10) for j in range(10)])
        # Cluster 2: around (10, 10)
        coords.extend([(10 + i * 0.1, 10 + j * 0.1) for i in range(10) for j in range(10)])

        sampler = SpatialBatchSampler(coords, batch_size=32, shuffle_batches=False)

        batches = list(sampler)

        # Check that batches contain spatially nearby points
        for batch in batches:
            xs = [c[0] for c in batch]
            ys = [c[1] for c in batch]

            # Calculate spread
            x_spread = max(xs) - min(xs)
            y_spread = max(ys) - min(ys)

            # Spatial batches should have limited spread
            # (not mixing points from far clusters)
            self.assertLess(x_spread, 12)  # Should not span both clusters
            self.assertLess(y_spread, 12)

    def test_invalid_batch_size_raises(self):
        """Test that invalid batch_size raises ValueError."""
        coords = [(i, i) for i in range(10)]
        coord_sampler = CoordinateSampler(coords, shuffle=False)

        with self.assertRaises(ValueError):
            CoordinateBatchSampler(coord_sampler, batch_size=0)

        with self.assertRaises(ValueError):
            CoordinateBatchSampler(coord_sampler, batch_size=-1)


class TestDistributedCoordinateSampler(TestCase):
    """Test distributed coordinate sampler."""

    def test_even_split(self):
        """Test that coordinates are evenly split across ranks."""
        coords = [(i, j) for i in range(10) for j in range(10)]

        # Simulate 4 processes
        samplers = []
        for rank in range(4):
            sampler = DistributedCoordinateSampler(
                coords,
                num_replicas=4,
                rank=rank,
                shuffle=False,
                drop_last=False
            )
            samplers.append(sampler)

        # Each should have 25 samples (100 / 4)
        for sampler in samplers:
            self.assertEqual(len(sampler), 25)

        # Collect all samples
        all_samples = []
        for sampler in samplers:
            all_samples.extend(list(sampler))

        # Should have all 100 coords (no loss, no duplication)
        self.assertEqual(len(all_samples), 100)
        self.assertEqual(len(set(all_samples)), 100)

    def test_set_epoch(self):
        """Test that set_epoch changes shuffle order."""
        coords = [(i, j) for i in range(10) for j in range(10)]

        sampler = DistributedCoordinateSampler(
            coords,
            num_replicas=2,
            rank=0,
            shuffle=True,
            seed=42
        )

        # Epoch 0
        sampler.set_epoch(0)
        coords1 = list(sampler)

        # Epoch 1 - should be different
        sampler.set_epoch(1)
        coords2 = list(sampler)

        self.assertNotEqual(coords1, coords2)

        # Same epoch should give same order
        sampler.set_epoch(0)
        coords3 = list(sampler)
        self.assertEqual(coords1, coords3)

    def test_drop_last(self):
        """Test drop_last functionality in distributed sampler."""
        # 97 coordinates, 4 replicas - not evenly divisible
        coords = [(i, 0) for i in range(97)]

        samplers_no_drop = []
        samplers_drop = []
        for rank in range(4):
            samplers_no_drop.append(DistributedCoordinateSampler(
                coords, num_replicas=4, rank=rank, shuffle=False, drop_last=False
            ))
            samplers_drop.append(DistributedCoordinateSampler(
                coords, num_replicas=4, rank=rank, shuffle=False, drop_last=True
            ))

        # Without drop_last: each replica gets ceil(97/4) = 25 samples
        # With padding to make even
        for sampler in samplers_no_drop:
            self.assertEqual(len(sampler), 25)

        # With drop_last: each gets floor((97-4)/4) = 23 or similar
        total_drop = sum(len(sampler) for sampler in samplers_drop)
        self.assertLess(total_drop, 97)  # Some samples dropped

    def test_invalid_rank_raises(self):
        """Test that invalid rank raises ValueError."""
        coords = [(i, i) for i in range(10)]

        with self.assertRaises(ValueError):
            DistributedCoordinateSampler(coords, num_replicas=4, rank=5)

        with self.assertRaises(ValueError):
            DistributedCoordinateSampler(coords, num_replicas=4, rank=-1)


class TestHelperFunctions(TestCase):
    """Test helper functions."""

    def test_create_coordinate_sampler_random(self):
        """Test factory function for random sampler."""
        coords = [(i, i) for i in range(10)]
        sampler = create_coordinate_sampler(coords, 'random', seed=42)

        self.assertIsInstance(sampler, CoordinateSampler)
        self.assertEqual(len(sampler), 10)

    def test_create_coordinate_sampler_sequential(self):
        """Test factory function for sequential sampler."""
        coords = [(i, i) for i in range(10)]
        sampler = create_coordinate_sampler(coords, 'sequential')

        samples = list(sampler)
        self.assertEqual(samples, coords)  # Should be in order

    def test_create_coordinate_sampler_from_tensor(self):
        """Test factory function with tensor input."""
        tensor = torch.rand(100, 2)
        sampler = create_coordinate_sampler(tensor, 'random')

        self.assertEqual(len(sampler), 100)

    def test_create_coordinate_sampler_invalid_type_raises(self):
        """Test factory function with invalid sampler type."""
        coords = [(i, i) for i in range(10)]

        with self.assertRaises(ValueError):
            create_coordinate_sampler(coords, 'invalid_type')

    def test_coordinates_to_tensor(self):
        """Test coordinate list to tensor conversion."""
        coords = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
        tensor = coordinates_to_tensor(coords)

        self.assertEqual(tensor.shape, (3, 2))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tensor[0, 0].item(), 1.0)
        self.assertEqual(tensor[0, 1].item(), 2.0)

    def test_coordinates_to_tensor_3d(self):
        """Test 3D coordinate list to tensor conversion."""
        coords = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
        tensor = coordinates_to_tensor(coords)

        self.assertEqual(tensor.shape, (2, 3))

    def test_tensor_to_coordinates_2d(self):
        """Test tensor to 2D coordinate list conversion."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        coords = tensor_to_coordinates(tensor)

        self.assertEqual(len(coords), 2)
        self.assertEqual(coords[0], (1.0, 2.0))
        self.assertEqual(coords[1], (3.0, 4.0))

    def test_tensor_to_coordinates_3d(self):
        """Test tensor to 3D coordinate list conversion."""
        tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        coords = tensor_to_coordinates(tensor)

        self.assertEqual(len(coords), 2)
        self.assertEqual(coords[0], (1.0, 2.0, 3.0))

    def test_tensor_to_coordinates_invalid_dims_raises(self):
        """Test tensor_to_coordinates with invalid dimensions."""
        tensor_1d = torch.tensor([1.0, 2.0, 3.0])
        tensor_3d = torch.randn(2, 3, 4)

        with self.assertRaises(ValueError):
            tensor_to_coordinates(tensor_1d)

        with self.assertRaises(ValueError):
            tensor_to_coordinates(tensor_3d)

    def test_tensor_to_coordinates_invalid_cols_raises(self):
        """Test tensor_to_coordinates with invalid number of columns."""
        tensor = torch.randn(10, 4)  # 4 columns, not 2 or 3

        with self.assertRaises(ValueError):
            tensor_to_coordinates(tensor)

    def test_roundtrip_conversion(self):
        """Test roundtrip conversion between coords and tensor."""
        original_coords = [(1.5, 2.5), (3.5, 4.5), (5.5, 6.5)]

        tensor = coordinates_to_tensor(original_coords)
        recovered_coords = tensor_to_coordinates(tensor)

        self.assertEqual(len(recovered_coords), len(original_coords))
        for orig, recov in zip(original_coords, recovered_coords):
            self.assertAlmostEqual(orig[0], recov[0], places=5)
            self.assertAlmostEqual(orig[1], recov[1], places=5)


if __name__ == '__main__':
    run_tests()