"""
Tests for the kernel lookup table functionality.
"""

import csv
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch._inductor import config
from torch._inductor.kernel.kernel_lut import (
    ConfigEntry,
    LookupTable,
    ProblemSize,
    get_lookup_table,
    parse_csv_lut,
    parse_json_lut,
    parse_lut_file,
)
from torch._inductor.template_heuristics import GemmConfig


class TestKernelLUT(unittest.TestCase):
    """Tests for the kernel lookup table functionality."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create sample configs for testing
        self.configs = [
            GemmConfig(block_m=128, block_n=128, block_k=32, num_stages=3, num_warps=4),
            GemmConfig(block_m=64, block_n=64, block_k=32, num_stages=2, num_warps=4),
            GemmConfig(block_m=32, block_n=32, block_k=16, num_stages=1, num_warps=2),
        ]
        
        # Create a sample lookup table
        self.lut = LookupTable()
        self.lut.add_entry(
            ConfigEntry(
                problem_size=ProblemSize(m=1024, n=1024, k=1024),
                config_params={
                    "BLOCK_M": 128,
                    "BLOCK_N": 128,
                    "BLOCK_K": 32,
                    "num_stages": 3,
                    "num_warps": 4,
                    "GROUP_M": 8,
                }
            )
        )
        self.lut.add_entry(
            ConfigEntry(
                problem_size=ProblemSize(m=512, n=512, k=512),
                config_params={
                    "BLOCK_M": 64,
                    "BLOCK_N": 64,
                    "BLOCK_K": 32,
                    "num_stages": 2,
                    "num_warps": 4,
                    "GROUP_M": 8,
                }
            )
        )
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_problem_size_matches(self):
        """Test the ProblemSize.matches method."""
        problem_size = ProblemSize(m=1000, n=1000, k=1000)
        
        # Test exact match
        self.assertTrue(problem_size.matches(1000, 1000, 1000))
        
        # Test within tolerance (default 10%)
        self.assertTrue(problem_size.matches(950, 950, 950))
        self.assertTrue(problem_size.matches(1050, 1050, 1050))
        
        # Test outside tolerance
        self.assertFalse(problem_size.matches(800, 1000, 1000))
        self.assertFalse(problem_size.matches(1000, 800, 1000))
        self.assertFalse(problem_size.matches(1000, 1000, 800))
        
        # Test with custom tolerance
        self.assertTrue(problem_size.matches(800, 800, 800, tolerance=0.2))
        self.assertFalse(problem_size.matches(790, 790, 790, tolerance=0.2))
    
    def test_config_entry_to_gemm_config(self):
        """Test the ConfigEntry.to_gemm_config method."""
        entry = ConfigEntry(
            problem_size=ProblemSize(m=1024, n=1024, k=1024),
            config_params={
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "num_stages": 3,
                "num_warps": 4,
                "GROUP_M": 8,
            }
        )
        
        config = entry.to_gemm_config()
        
        self.assertEqual(config.block_m, 128)
        self.assertEqual(config.block_n, 128)
        self.assertEqual(config.block_k, 32)
        self.assertEqual(config.num_stages, 3)
        self.assertEqual(config.num_warps, 4)
        self.assertEqual(config.group_m, 8)
    
    def test_lookup_table_find_config(self):
        """Test the LookupTable.find_config method."""
        # Test exact match
        config = self.lut.find_config(1024, 1024, 1024)
        self.assertIsNotNone(config)
        self.assertEqual(config.block_m, 128)
        self.assertEqual(config.block_n, 128)
        self.assertEqual(config.block_k, 32)
        
        # Test within tolerance
        config = self.lut.find_config(950, 950, 950)
        self.assertIsNotNone(config)
        self.assertEqual(config.block_m, 128)
        self.assertEqual(config.block_n, 128)
        self.assertEqual(config.block_k, 32)
        
        # Test no match
        config = self.lut.find_config(2048, 2048, 2048)
        self.assertIsNone(config)
    
    def test_lookup_table_filter_configs(self):
        """Test the LookupTable.filter_configs method."""
        # Test with matching config
        filtered_configs = self.lut.filter_configs(self.configs, 1024, 1024, 1024)
        self.assertEqual(len(filtered_configs), 1)
        self.assertEqual(filtered_configs[0].block_m, 128)
        self.assertEqual(filtered_configs[0].block_n, 128)
        self.assertEqual(filtered_configs[0].block_k, 32)
        
        # Test with no matching config
        filtered_configs = self.lut.filter_configs(self.configs, 2048, 2048, 2048)
        self.assertEqual(len(filtered_configs), 3)  # Returns all configs
    
    def test_parse_csv_lut(self):
        """Test parsing a CSV lookup table file."""
        # Create a sample CSV file
        csv_path = os.path.join(self.temp_dir.name, "test_lut.csv")
        with open(csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["M", "N", "K", "BLOCK_M", "BLOCK_N", "BLOCK_K", "num_stages", "num_warps", "GROUP_M"])
            writer.writerow([1024, 1024, 1024, 128, 128, 32, 3, 4, 8])
            writer.writerow([512, 512, 512, 64, 64, 32, 2, 4, 8])
        
        # Parse the CSV file
        lut = parse_csv_lut(csv_path)
        
        # Check the parsed lookup table
        self.assertEqual(len(lut.entries), 2)
        
        # Check the first entry
        entry = lut.entries[0]
        self.assertEqual(entry.problem_size.m, 1024)
        self.assertEqual(entry.problem_size.n, 1024)
        self.assertEqual(entry.problem_size.k, 1024)
        self.assertEqual(entry.config_params["BLOCK_M"], 128)
        self.assertEqual(entry.config_params["BLOCK_N"], 128)
        self.assertEqual(entry.config_params["BLOCK_K"], 32)
        self.assertEqual(entry.config_params["num_stages"], 3)
        self.assertEqual(entry.config_params["num_warps"], 4)
        self.assertEqual(entry.config_params["GROUP_M"], 8)
        
        # Check the second entry
        entry = lut.entries[1]
        self.assertEqual(entry.problem_size.m, 512)
        self.assertEqual(entry.problem_size.n, 512)
        self.assertEqual(entry.problem_size.k, 512)
        self.assertEqual(entry.config_params["BLOCK_M"], 64)
        self.assertEqual(entry.config_params["BLOCK_N"], 64)
        self.assertEqual(entry.config_params["BLOCK_K"], 32)
        self.assertEqual(entry.config_params["num_stages"], 2)
        self.assertEqual(entry.config_params["num_warps"], 4)
        self.assertEqual(entry.config_params["GROUP_M"], 8)
    
    def test_parse_json_lut(self):
        """Test parsing a JSON lookup table file."""
        # Create a sample JSON file
        json_path = os.path.join(self.temp_dir.name, "test_lut.json")
        data = [
            {
                "problem_size": {"m": 1024, "n": 1024, "k": 1024},
                "config": {
                    "BLOCK_M": 128,
                    "BLOCK_N": 128,
                    "BLOCK_K": 32,
                    "num_stages": 3,
                    "num_warps": 4,
                    "GROUP_M": 8
                }
            },
            {
                "problem_size": {"m": 512, "n": 512, "k": 512},
                "config": {
                    "BLOCK_M": 64,
                    "BLOCK_N": 64,
                    "BLOCK_K": 32,
                    "num_stages": 2,
                    "num_warps": 4,
                    "GROUP_M": 8
                }
            }
        ]
        with open(json_path, "w") as f:
            json.dump(data, f)
        
        # Parse the JSON file
        lut = parse_json_lut(json_path)
        
        # Check the parsed lookup table
        self.assertEqual(len(lut.entries), 2)
        
        # Check the first entry
        entry = lut.entries[0]
        self.assertEqual(entry.problem_size.m, 1024)
        self.assertEqual(entry.problem_size.n, 1024)
        self.assertEqual(entry.problem_size.k, 1024)
        self.assertEqual(entry.config_params["BLOCK_M"], 128)
        self.assertEqual(entry.config_params["BLOCK_N"], 128)
        self.assertEqual(entry.config_params["BLOCK_K"], 32)
        self.assertEqual(entry.config_params["num_stages"], 3)
        self.assertEqual(entry.config_params["num_warps"], 4)
        self.assertEqual(entry.config_params["GROUP_M"], 8)
        
        # Check the second entry
        entry = lut.entries[1]
        self.assertEqual(entry.problem_size.m, 512)
        self.assertEqual(entry.problem_size.n, 512)
        self.assertEqual(entry.problem_size.k, 512)
        self.assertEqual(entry.config_params["BLOCK_M"], 64)
        self.assertEqual(entry.config_params["BLOCK_N"], 64)
        self.assertEqual(entry.config_params["BLOCK_K"], 32)
        self.assertEqual(entry.config_params["num_stages"], 2)
        self.assertEqual(entry.config_params["num_warps"], 4)
        self.assertEqual(entry.config_params["GROUP_M"], 8)
    
    def test_parse_lut_file(self):
        """Test parsing a lookup table file based on extension."""
        # Create a sample CSV file
        csv_path = os.path.join(self.temp_dir.name, "test_lut.csv")
        with open(csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["M", "N", "K", "BLOCK_M", "BLOCK_N", "BLOCK_K", "num_stages", "num_warps", "GROUP_M"])
            writer.writerow([1024, 1024, 1024, 128, 128, 32, 3, 4, 8])
        
        # Create a sample JSON file
        json_path = os.path.join(self.temp_dir.name, "test_lut.json")
        data = [
            {
                "problem_size": {"m": 1024, "n": 1024, "k": 1024},
                "config": {
                    "BLOCK_M": 128,
                    "BLOCK_N": 128,
                    "BLOCK_K": 32,
                    "num_stages": 3,
                    "num_warps": 4,
                    "GROUP_M": 8
                }
            }
        ]
        with open(json_path, "w") as f:
            json.dump(data, f)
        
        # Create an unsupported file
        txt_path = os.path.join(self.temp_dir.name, "test_lut.txt")
        with open(txt_path, "w") as f:
            f.write("This is not a supported format")
        
        # Test parsing CSV file
        lut = parse_lut_file(csv_path)
        self.assertIsNotNone(lut)
        self.assertEqual(len(lut.entries), 1)
        
        # Test parsing JSON file
        lut = parse_lut_file(json_path)
        self.assertIsNotNone(lut)
        self.assertEqual(len(lut.entries), 1)
        
        # Test parsing unsupported file
        lut = parse_lut_file(txt_path)
        self.assertIsNone(lut)
        
        # Test parsing non-existent file
        lut = parse_lut_file("non_existent_file.csv")
        self.assertIsNone(lut)
    
    def test_get_lookup_table(self):
        """Test getting the global lookup table instance."""
        # Create a sample CSV file
        csv_path = os.path.join(self.temp_dir.name, "test_lut.csv")
        with open(csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["M", "N", "K", "BLOCK_M", "BLOCK_N", "BLOCK_K", "num_stages", "num_warps", "GROUP_M"])
            writer.writerow([1024, 1024, 1024, 128, 128, 32, 3, 4, 8])
        
        # Test with no config
        with patch.object(config, "kernel_lut_path", None):
            from torch._inductor.kernel.kernel_lut import _global_lut
            _global_lut = None  # Reset the global instance
            lut = get_lookup_table()
            self.assertIsNone(lut)
        
        # Test with config
        with patch.object(config, "kernel_lut_path", csv_path):
            from torch._inductor.kernel.kernel_lut import _global_lut
            _global_lut = None  # Reset the global instance
            lut = get_lookup_table()
            self.assertIsNotNone(lut)
            self.assertEqual(len(lut.entries), 1)


if __name__ == "__main__":
    unittest.main()
