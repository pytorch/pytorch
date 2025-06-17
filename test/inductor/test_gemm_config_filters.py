# Owner(s): ["module: inductor"]

import unittest
from typing import List

import torch
from torch._inductor.template_heuristics import BaseConfig, GemmConfig, BaseConfigHeuristic
from torch._inductor.kernel.kernel_filters import gemm_config_registry, set_scale_params


class TestGemmConfigFilters(unittest.TestCase):
    def setUp(self):
        # Save the original filters to restore after the test
        self.original_filters = gemm_config_registry.filters.copy()
        self.original_filter_names = gemm_config_registry.filter_names.copy()
        
        # Clear the registry for testing
        gemm_config_registry.clear()
        
        # Create some test configs
        self.test_configs = [
            GemmConfig(16, 16, 16, 1, 2),
            GemmConfig(32, 32, 32, 2, 4),
            GemmConfig(64, 64, 64, 3, 8),
            GemmConfig(128, 128, 128, 4, 8),
            GemmConfig(256, 256, 256, 5, 8),
        ]

    def tearDown(self):
        # Restore the original filters
        gemm_config_registry.filters = self.original_filters
        gemm_config_registry.filter_names = self.original_filter_names

    def test_register_filter(self):
        # Register a simple filter
        @gemm_config_registry.register
        def filter_small_blocks(configs: List[BaseConfig]) -> List[BaseConfig]:
            return [c for c in configs if c.block_m >= 32]
        
        # Apply the filter
        filtered_configs = gemm_config_registry.apply_filters(self.test_configs)
        
        # Check that the filter was applied correctly
        self.assertEqual(len(filtered_configs), 4)  # Should have removed the 16x16x16 config
        self.assertEqual(filtered_configs[0].block_m, 32)
        self.assertEqual(filtered_configs[0].block_n, 32)
        
        # Check that the filter was registered correctly
        self.assertEqual(len(gemm_config_registry.filters), 1)
        self.assertEqual(gemm_config_registry.filter_names[0], "filter_small_blocks")

    def test_multiple_filters(self):
        # Register multiple filters
        @gemm_config_registry.register
        def filter_small_blocks(configs: List[BaseConfig]) -> List[BaseConfig]:
            return [c for c in configs if c.block_m >= 32]
        
        @gemm_config_registry.register(name="filter_large_blocks")
        def filter_large(configs: List[BaseConfig]) -> List[BaseConfig]:
            return [c for c in configs if c.block_m <= 128]
        
        # Apply the filters
        filtered_configs = gemm_config_registry.apply_filters(self.test_configs)
        
        # Check that both filters were applied correctly
        self.assertEqual(len(filtered_configs), 3)  # Should have removed 16x16x16 and 256x256x256
        self.assertEqual(filtered_configs[0].block_m, 32)
        self.assertEqual(filtered_configs[-1].block_m, 128)
        
        # Check that the filters were registered correctly
        self.assertEqual(len(gemm_config_registry.filters), 2)
        self.assertEqual(gemm_config_registry.filter_names, ["filter_small_blocks", "filter_large_blocks"])

    def test_scale_params(self):
        # Register a filter that uses scale_params
        @gemm_config_registry.register
        def scale_configs(configs: List[BaseConfig]) -> List[BaseConfig]:
            from torch._inductor.kernel.kernel_filters import get_filter_context
            ctx = get_filter_context()
            scale = ctx.scale_params["scale"]
            
            scaled_configs = []
            for c in configs:
                # Simple scaling for testing - just double the block_m if scale is 2.0
                import dataclasses
                scaled_config = dataclasses.replace(
                    c,
                    block_m=int(c.block_m * scale)
                )
                scaled_configs.append(scaled_config)
            return scaled_configs
        
        # Set scale params
        set_scale_params(m=128, n=128, k=128, scale=2.0)
        
        # Apply the filter
        filtered_configs = gemm_config_registry.apply_filters(self.test_configs)
        
        # Check that the scaling was applied correctly
        self.assertEqual(filtered_configs[0].block_m, 32)  # 16 * 2 = 32
        self.assertEqual(filtered_configs[1].block_m, 64)  # 32 * 2 = 64
        self.assertEqual(filtered_configs[2].block_m, 128)  # 64 * 2 = 128
        
    def test_default_filters(self):
        """Test that the default filters are registered and working correctly."""
        # Clear the registry
        gemm_config_registry.clear()
        
        # Import the default filters
        from torch._inductor.kernel.kernel_filters import import_default_filters
        import_default_filters()
        
        # Check that the default filters were registered
        self.assertEqual(len(gemm_config_registry.filters), 2)
        self.assertEqual(set(gemm_config_registry.filter_names), {"finalize_mm_configs", "scale_mm_configs"})
        
        # Set scale params for testing
        set_scale_params(m=128, n=128, k=128, scale=1.0)
        
        # Apply the filters
        filtered_configs = gemm_config_registry.apply_filters(self.test_configs)
        
        # Check that we got some results (basic sanity check)
        self.assertTrue(len(filtered_configs) > 0)
        
    def test_integration_with_heuristics(self):
        """Test integration with the template_heuristics system."""
        # Create a heuristic instance
        heuristic = BaseConfigHeuristic()
        
        # Get some configs
        configs = heuristic.mm_configs
        
        # Set scale params
        set_scale_params(m=128, n=128, k=128, scale=1.0)
        
        # Apply the filters directly
        filtered_configs = gemm_config_registry.apply_filters(configs)
        
        # Check that we got some results
        self.assertTrue(len(filtered_configs) > 0)
        
        # Now use the preprocess_mm_configs method
        # This should internally use our middleware system
        triton_configs = list(heuristic.preprocess_mm_configs(
            m=128, n=128, k=128, configs=configs
        ))
        
        # Check that we got some results
        self.assertTrue(len(triton_configs) > 0)


if __name__ == "__main__":
    unittest.main()
