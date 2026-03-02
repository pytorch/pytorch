# mypy: ignore-errors

# Owner(s): ["module: dataloader"]

import pickle

import torch
import torch.utils.data.datapipes as dp
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils.data.datapipes.iter.tiling import (
    BoundaryMode,
    TiledImageIterDataPipe,
    TileInfo,
    TileOrder,
)


class TestTiledImageIterDataPipe(TestCase):
    """Test cases for TiledImageIterDataPipe."""

    def test_basic_tiling_3d(self):
        """Test basic tiling of a 3D image (C, H, W)."""
        # Create a 3-channel 8x8 image
        image = torch.arange(192).reshape(3, 8, 8).float()
        input_dp = dp.iter.IterableWrapper([image])
        tiled_dp = TiledImageIterDataPipe(input_dp, tile_size=4, stride=4)

        tiles = list(tiled_dp)

        # Should produce 4 tiles (2x2 grid)
        self.assertEqual(len(tiles), 4)

        # Check tile shapes
        for tile_info in tiles:
            self.assertEqual(tile_info.tile.shape, torch.Size([3, 4, 4]))

        # Check tile positions
        positions = [(t.row, t.col) for t in tiles]
        self.assertEqual(positions, [(0, 0), (0, 1), (1, 0), (1, 1)])

    def test_basic_tiling_2d(self):
        """Test basic tiling of a 2D image (H, W)."""
        # Create an 8x8 grayscale image
        image = torch.arange(64).reshape(8, 8).float()
        input_dp = dp.iter.IterableWrapper([image])
        tiled_dp = TiledImageIterDataPipe(input_dp, tile_size=4, stride=4)

        tiles = list(tiled_dp)

        # Should produce 4 tiles (2x2 grid)
        self.assertEqual(len(tiles), 4)

        # Check tile shapes
        for tile_info in tiles:
            self.assertEqual(tile_info.tile.shape, torch.Size([4, 4]))

    def test_tile_size_tuple(self):
        """Test non-square tile sizes."""
        image = torch.randn(3, 12, 8)
        input_dp = dp.iter.IterableWrapper([image])
        tiled_dp = TiledImageIterDataPipe(
            input_dp, tile_size=(4, 2), stride=(4, 2)
        )

        tiles = list(tiled_dp)

        # 12/4 = 3 rows, 8/2 = 4 cols = 12 tiles
        self.assertEqual(len(tiles), 12)

        for tile_info in tiles:
            self.assertEqual(tile_info.tile.shape, torch.Size([3, 4, 2]))

    def test_overlapping_tiles(self):
        """Test overlapping tiles with stride < tile_size."""
        image = torch.randn(3, 8, 8)
        input_dp = dp.iter.IterableWrapper([image])
        tiled_dp = TiledImageIterDataPipe(input_dp, tile_size=4, stride=2)

        tiles = list(tiled_dp)

        # (8-4)/2 + 1 = 3 per dimension = 9 tiles
        self.assertEqual(len(tiles), 9)

    def test_boundary_mode_skip(self):
        """Test BoundaryMode.SKIP - skips incomplete tiles."""
        # 7x7 image with 4x4 tiles, stride 4
        # Only one complete tile (0,0)
        image = torch.randn(3, 7, 7)
        input_dp = dp.iter.IterableWrapper([image])
        tiled_dp = TiledImageIterDataPipe(
            input_dp, tile_size=4, stride=4, boundary_mode=BoundaryMode.SKIP
        )

        tiles = list(tiled_dp)

        self.assertEqual(len(tiles), 1)
        self.assertEqual(tiles[0].row, 0)
        self.assertEqual(tiles[0].col, 0)

    def test_boundary_mode_pad(self):
        """Test BoundaryMode.PAD - pads incomplete tiles."""
        # 6x6 image with 4x4 tiles, stride 4
        image = torch.zeros(3, 6, 6)
        input_dp = dp.iter.IterableWrapper([image])
        tiled_dp = TiledImageIterDataPipe(
            input_dp,
            tile_size=4,
            stride=4,
            boundary_mode=BoundaryMode.PAD,
            pad_value=99.0,
        )

        tiles = list(tiled_dp)

        # Should produce 4 tiles (2x2 grid including padded ones)
        self.assertEqual(len(tiles), 4)

        # All tiles should be 4x4
        for tile_info in tiles:
            self.assertEqual(tile_info.tile.shape, torch.Size([3, 4, 4]))

        # Check that boundary tiles are padded
        # Tile (0,1) should have padding on the right
        tile_01 = [t for t in tiles if t.row == 0 and t.col == 1][0]
        # The rightmost 2 columns should be padded with 99
        self.assertTrue(torch.all(tile_01.tile[:, :, 2:] == 99.0))

    def test_boundary_mode_crop(self):
        """Test BoundaryMode.CROP - returns smaller tiles at boundaries."""
        # 6x6 image with 4x4 tiles, stride 4
        image = torch.randn(3, 6, 6)
        input_dp = dp.iter.IterableWrapper([image])
        tiled_dp = TiledImageIterDataPipe(
            input_dp, tile_size=4, stride=4, boundary_mode=BoundaryMode.CROP
        )

        tiles = list(tiled_dp)

        # Should produce 4 tiles
        self.assertEqual(len(tiles), 4)

        # Tile (0,0) should be 4x4 (complete)
        tile_00 = [t for t in tiles if t.row == 0 and t.col == 0][0]
        self.assertEqual(tile_00.tile.shape, torch.Size([3, 4, 4]))

        # Tile (0,1) should be 4x2 (cropped on right)
        tile_01 = [t for t in tiles if t.row == 0 and t.col == 1][0]
        self.assertEqual(tile_01.tile.shape, torch.Size([3, 4, 2]))

        # Tile (1,1) should be 2x2 (cropped on both sides)
        tile_11 = [t for t in tiles if t.row == 1 and t.col == 1][0]
        self.assertEqual(tile_11.tile.shape, torch.Size([3, 2, 2]))

    def test_tile_order_row_major(self):
        """Test row-major tile ordering (default)."""
        image = torch.randn(3, 8, 8)
        input_dp = dp.iter.IterableWrapper([image])
        tiled_dp = TiledImageIterDataPipe(
            input_dp, tile_size=4, stride=4, tile_order=TileOrder.ROW_MAJOR
        )

        tiles = list(tiled_dp)

        positions = [(t.row, t.col) for t in tiles]
        expected = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.assertEqual(positions, expected)

    def test_tile_order_column_major(self):
        """Test column-major tile ordering."""
        image = torch.randn(3, 8, 8)
        input_dp = dp.iter.IterableWrapper([image])
        tiled_dp = TiledImageIterDataPipe(
            input_dp, tile_size=4, stride=4, tile_order=TileOrder.COLUMN_MAJOR
        )

        tiles = list(tiled_dp)

        positions = [(t.row, t.col) for t in tiles]
        expected = [(0, 0), (1, 0), (0, 1), (1, 1)]
        self.assertEqual(positions, expected)

    def test_tile_order_hilbert(self):
        """Test Hilbert curve tile ordering."""
        image = torch.randn(3, 16, 16)
        input_dp = dp.iter.IterableWrapper([image])
        tiled_dp = TiledImageIterDataPipe(
            input_dp, tile_size=4, stride=4, tile_order=TileOrder.HILBERT
        )

        tiles = list(tiled_dp)

        # Should produce 16 tiles (4x4 grid)
        self.assertEqual(len(tiles), 16)

        # All tiles should be unique
        positions = [(t.row, t.col) for t in tiles]
        self.assertEqual(len(set(positions)), 16)

    def test_multiple_images(self):
        """Test tiling multiple images."""
        images = [torch.randn(3, 8, 8) for _ in range(3)]
        input_dp = dp.iter.IterableWrapper(images)
        tiled_dp = TiledImageIterDataPipe(input_dp, tile_size=4, stride=4)

        tiles = list(tiled_dp)

        # 4 tiles per image * 3 images = 12 tiles
        self.assertEqual(len(tiles), 12)

        # Check source_index
        source_indices = [t.source_index for t in tiles]
        self.assertEqual(source_indices, [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

    def test_tile_info_contents(self):
        """Test that TileInfo contains correct coordinates."""
        image = torch.randn(3, 8, 8)
        input_dp = dp.iter.IterableWrapper([image])
        tiled_dp = TiledImageIterDataPipe(input_dp, tile_size=4, stride=4)

        tiles = list(tiled_dp)

        # Check tile at (1, 1)
        tile_11 = [t for t in tiles if t.row == 1 and t.col == 1][0]
        self.assertEqual(tile_11.y_start, 4)
        self.assertEqual(tile_11.x_start, 4)
        self.assertEqual(tile_11.source_index, 0)

    def test_tile_content_correctness(self):
        """Test that tile content matches the original image region."""
        # Create an image with known values
        image = torch.arange(64).reshape(1, 8, 8).float()
        input_dp = dp.iter.IterableWrapper([image])
        tiled_dp = TiledImageIterDataPipe(input_dp, tile_size=4, stride=4)

        tiles = list(tiled_dp)

        for tile_info in tiles:
            y_start = tile_info.y_start
            x_start = tile_info.x_start
            expected = image[:, y_start : y_start + 4, x_start : x_start + 4]
            self.assertTrue(torch.equal(tile_info.tile, expected))

    def test_functional_api(self):
        """Test the .tile() functional API."""
        image = torch.randn(3, 8, 8)
        input_dp = dp.iter.IterableWrapper([image])

        # Using functional API
        tiled_dp = input_dp.tile(tile_size=4, stride=4)

        tiles = list(tiled_dp)
        self.assertEqual(len(tiles), 4)

    def test_lru_cache(self):
        """Test LRU caching for tiles."""
        image = torch.randn(3, 8, 8)
        input_dp = dp.iter.IterableWrapper([image])
        tiled_dp = TiledImageIterDataPipe(
            input_dp, tile_size=4, stride=4, cache_size=4
        )

        # Iterate through all tiles
        tiles1 = list(tiled_dp)

        # Cache should now contain all 4 tiles
        self.assertIsNotNone(tiled_dp._cache)
        self.assertEqual(len(tiled_dp._cache._cache), 4)

    def test_invalid_tile_size(self):
        """Test that invalid tile sizes raise ValueError."""
        image = torch.randn(3, 8, 8)
        input_dp = dp.iter.IterableWrapper([image])

        with self.assertRaises(ValueError):
            TiledImageIterDataPipe(input_dp, tile_size=0)

        with self.assertRaises(ValueError):
            TiledImageIterDataPipe(input_dp, tile_size=-1)

    def test_invalid_stride(self):
        """Test that invalid strides raise ValueError."""
        image = torch.randn(3, 8, 8)
        input_dp = dp.iter.IterableWrapper([image])

        with self.assertRaises(ValueError):
            TiledImageIterDataPipe(input_dp, tile_size=4, stride=0)

    def test_non_tensor_input_raises_error(self):
        """Test that non-tensor input raises TypeError."""
        input_dp = dp.iter.IterableWrapper([[1, 2, 3]])
        tiled_dp = TiledImageIterDataPipe(input_dp, tile_size=2)

        with self.assertRaises(TypeError):
            list(tiled_dp)

    def test_wrong_dimension_raises_error(self):
        """Test that 4D or 1D tensors raise ValueError."""
        # 4D tensor
        image_4d = torch.randn(2, 3, 8, 8)
        input_dp = dp.iter.IterableWrapper([image_4d])
        tiled_dp = TiledImageIterDataPipe(input_dp, tile_size=4)

        with self.assertRaises(ValueError):
            list(tiled_dp)

        # 1D tensor
        image_1d = torch.randn(64)
        input_dp = dp.iter.IterableWrapper([image_1d])
        tiled_dp = TiledImageIterDataPipe(input_dp, tile_size=4)

        with self.assertRaises(ValueError):
            list(tiled_dp)

    def test_serialization(self):
        """Test that TiledImageIterDataPipe is serializable."""
        image = torch.randn(3, 8, 8)
        input_dp = dp.iter.IterableWrapper([image])
        tiled_dp = TiledImageIterDataPipe(
            input_dp,
            tile_size=4,
            stride=4,
            tile_order=TileOrder.ROW_MAJOR,
            boundary_mode=BoundaryMode.SKIP,
        )

        # Serialize and deserialize
        serialized = pickle.dumps(tiled_dp)
        deserialized = pickle.loads(serialized)

        # Compare results
        original_tiles = list(tiled_dp)
        deserialized_tiles = list(deserialized)

        self.assertEqual(len(original_tiles), len(deserialized_tiles))
        for orig, deser in zip(original_tiles, deserialized_tiles):
            self.assertEqual(orig.row, deser.row)
            self.assertEqual(orig.col, deser.col)
            self.assertTrue(torch.equal(orig.tile, deser.tile))

    def test_reset(self):
        """Test reset clears the cache."""
        image = torch.randn(3, 8, 8)
        input_dp = dp.iter.IterableWrapper([image])
        tiled_dp = TiledImageIterDataPipe(
            input_dp, tile_size=4, stride=4, cache_size=4
        )

        # Iterate to populate cache
        _ = list(tiled_dp)
        self.assertEqual(len(tiled_dp._cache._cache), 4)

        # Reset should clear cache
        tiled_dp.reset()
        self.assertEqual(len(tiled_dp._cache._cache), 0)

    def test_empty_image(self):
        """Test handling of empty images."""
        # Image with zero height
        image = torch.randn(3, 0, 8)
        input_dp = dp.iter.IterableWrapper([image])
        tiled_dp = TiledImageIterDataPipe(input_dp, tile_size=4, stride=4)

        tiles = list(tiled_dp)
        self.assertEqual(len(tiles), 0)

    def test_image_smaller_than_tile(self):
        """Test handling when image is smaller than tile size."""
        image = torch.randn(3, 2, 2)
        input_dp = dp.iter.IterableWrapper([image])

        # With SKIP mode, no tiles produced
        tiled_dp = TiledImageIterDataPipe(
            input_dp, tile_size=4, boundary_mode=BoundaryMode.SKIP
        )
        tiles = list(tiled_dp)
        self.assertEqual(len(tiles), 0)

        # With PAD mode, one padded tile
        input_dp = dp.iter.IterableWrapper([image])
        tiled_dp = TiledImageIterDataPipe(
            input_dp, tile_size=4, boundary_mode=BoundaryMode.PAD
        )
        tiles = list(tiled_dp)
        self.assertEqual(len(tiles), 1)
        self.assertEqual(tiles[0].tile.shape, torch.Size([3, 4, 4]))

        # With CROP mode, one small tile
        input_dp = dp.iter.IterableWrapper([image])
        tiled_dp = TiledImageIterDataPipe(
            input_dp, tile_size=4, boundary_mode=BoundaryMode.CROP
        )
        tiles = list(tiled_dp)
        self.assertEqual(len(tiles), 1)
        self.assertEqual(tiles[0].tile.shape, torch.Size([3, 2, 2]))

    def test_large_stride(self):
        """Test stride larger than tile size (sparse sampling)."""
        image = torch.randn(3, 16, 16)
        input_dp = dp.iter.IterableWrapper([image])
        tiled_dp = TiledImageIterDataPipe(input_dp, tile_size=4, stride=8)

        tiles = list(tiled_dp)

        # 16/8 = 2 per dimension = 4 tiles (with some skipping)
        self.assertEqual(len(tiles), 4)

        positions = [(t.row, t.col) for t in tiles]
        expected_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.assertEqual(positions, expected_positions)


class TestTileLRUCache(TestCase):
    """Test cases for the internal LRU cache."""

    def test_cache_basic(self):
        """Test basic cache operations."""
        from torch.utils.data.datapipes.iter.tiling import _TileLRUCache

        cache = _TileLRUCache(max_size=3)

        # Put items
        cache.put((0, 0, 0), torch.tensor([1.0]))
        cache.put((0, 0, 1), torch.tensor([2.0]))
        cache.put((0, 1, 0), torch.tensor([3.0]))

        # Get items
        self.assertTrue(torch.equal(cache.get((0, 0, 0)), torch.tensor([1.0])))
        self.assertTrue(torch.equal(cache.get((0, 0, 1)), torch.tensor([2.0])))

        # Non-existent item
        self.assertIsNone(cache.get((9, 9, 9)))

    def test_cache_eviction(self):
        """Test LRU eviction policy."""
        from torch.utils.data.datapipes.iter.tiling import _TileLRUCache

        cache = _TileLRUCache(max_size=2)

        cache.put((0, 0, 0), torch.tensor([1.0]))
        cache.put((0, 0, 1), torch.tensor([2.0]))

        # Access first item to make it recently used
        cache.get((0, 0, 0))

        # Add new item, should evict (0, 0, 1)
        cache.put((0, 1, 0), torch.tensor([3.0]))

        self.assertIsNotNone(cache.get((0, 0, 0)))  # Still present
        self.assertIsNone(cache.get((0, 0, 1)))  # Evicted
        self.assertIsNotNone(cache.get((0, 1, 0)))  # Present

    def test_cache_clear(self):
        """Test cache clear operation."""
        from torch.utils.data.datapipes.iter.tiling import _TileLRUCache

        cache = _TileLRUCache(max_size=3)

        cache.put((0, 0, 0), torch.tensor([1.0]))
        cache.put((0, 0, 1), torch.tensor([2.0]))

        cache.clear()

        self.assertIsNone(cache.get((0, 0, 0)))
        self.assertIsNone(cache.get((0, 0, 1)))


class TestHilbertCurve(TestCase):
    """Test cases for Hilbert curve utilities."""

    def test_hilbert_d2xy(self):
        """Test Hilbert curve coordinate generation."""
        from torch.utils.data.datapipes.iter.tiling import _hilbert_d2xy

        # For n=2, the Hilbert curve visits: (0,0), (0,1), (1,1), (1,0)
        coords = [_hilbert_d2xy(2, d) for d in range(4)]
        expected = [(0, 0), (0, 1), (1, 1), (1, 0)]
        self.assertEqual(coords, expected)

    def test_generate_hilbert_indices(self):
        """Test Hilbert index generation for grid."""
        from torch.utils.data.datapipes.iter.tiling import _generate_hilbert_indices

        # 2x2 grid
        indices = _generate_hilbert_indices(2, 2)
        self.assertEqual(len(indices), 4)
        self.assertEqual(len(set(indices)), 4)  # All unique

        # 3x3 grid (non-power-of-2)
        indices = _generate_hilbert_indices(3, 3)
        self.assertEqual(len(indices), 9)
        self.assertEqual(len(set(indices)), 9)

    def test_next_power_of_2(self):
        """Test next power of 2 calculation."""
        from torch.utils.data.datapipes.iter.tiling import _next_power_of_2

        self.assertEqual(_next_power_of_2(1), 1)
        self.assertEqual(_next_power_of_2(2), 2)
        self.assertEqual(_next_power_of_2(3), 4)
        self.assertEqual(_next_power_of_2(5), 8)
        self.assertEqual(_next_power_of_2(8), 8)
        self.assertEqual(_next_power_of_2(9), 16)


if __name__ == "__main__":
    run_tests()
