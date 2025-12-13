#!/usr/bin/env python3
"""
Test case for EmbeddingBag consistency fix.

This test verifies that all backends (CPU, CUDA, MPS) now consistently
enforce the offsets[0] == 0 requirement.

Related issue: https://github.com/pytorch/pytorch/issues/170370
"""

import unittest
import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase, run_tests


class TestEmbeddingBagConsistency(TestCase):
    """Test EmbeddingBag consistency across different backends."""
    
    def setUp(self):
        """Set up test data."""
        self.emb = nn.EmbeddingBag(num_embeddings=100, embedding_dim=16, mode="mean")
        self.indices = torch.randint(0, 100, (10,))
        self.invalid_offsets = torch.tensor([5, 8])  # offsets[0] != 0
        self.valid_offsets = torch.tensor([0, 5, 8])  # offsets[0] == 0
    
    def test_cpu_rejects_invalid_offsets(self):
        """Test that CPU backend rejects offsets[0] != 0."""
        with self.assertRaises(RuntimeError) as context:
            self.emb(self.indices, self.invalid_offsets)
        
        error_msg = str(context.exception)
        self.assertIn("offsets[0] has to be 0", error_msg)
        self.assertIn("However, got 5", error_msg)
    
    def test_cpu_accepts_valid_offsets(self):
        """Test that CPU backend accepts offsets[0] == 0."""
        try:
            output = self.emb(self.indices, self.valid_offsets)
            self.assertEqual(output.shape, (3, 16))  # 3 bags, 16 features
        except Exception as e:
            self.fail(f"CPU should accept valid offsets, but got error: {e}")
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_rejects_invalid_offsets(self):
        """Test that CUDA backend rejects offsets[0] != 0 after fix."""
        emb_cuda = self.emb.to("cuda")
        indices_cuda = self.indices.to("cuda")
        offsets_cuda = self.invalid_offsets.to("cuda")
        
        with self.assertRaises(RuntimeError) as context:
            emb_cuda(indices_cuda, offsets_cuda)
        
        error_msg = str(context.exception)
        self.assertIn("offsets[0] has to be 0", error_msg)
        self.assertIn("However, got 5", error_msg)
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_accepts_valid_offsets(self):
        """Test that CUDA backend accepts offsets[0] == 0 after fix."""
        emb_cuda = self.emb.to("cuda")
        indices_cuda = self.indices.to("cuda")
        offsets_cuda = self.valid_offsets.to("cuda")
        
        try:
            output = emb_cuda(indices_cuda, offsets_cuda)
            self.assertEqual(output.shape, (3, 16))  # 3 bags, 16 features
        except Exception as e:
            self.fail(f"CUDA should accept valid offsets, but got error: {e}")
    
    @unittest.skipIf(not torch.backends.mps.is_available(), "MPS not available")
    def test_mps_rejects_invalid_offsets(self):
        """Test that MPS backend rejects offsets[0] != 0 after fix."""
        emb_mps = self.emb.to("mps")
        indices_mps = self.indices.to("mps")
        offsets_mps = self.invalid_offsets.to("mps")
        
        with self.assertRaises(RuntimeError) as context:
            emb_mps(indices_mps, offsets_mps)
        
        error_msg = str(context.exception)
        self.assertIn("offsets[0] has to be 0", error_msg)
        self.assertIn("However, got 5", error_msg)
    
    @unittest.skipIf(not torch.backends.mps.is_available(), "MPS not available")
    def test_mps_accepts_valid_offsets(self):
        """Test that MPS backend accepts offsets[0] == 0 after fix."""
        emb_mps = self.emb.to("mps")
        indices_mps = self.indices.to("mps")
        offsets_mps = self.valid_offsets.to("mps")
        
        try:
            output = emb_mps(indices_mps, offsets_mps)
            self.assertEqual(output.shape, (3, 16))  # 3 bags, 16 features
        except Exception as e:
            self.fail(f"MPS should accept valid offsets, but got error: {e}")
    
    def test_consistency_across_backends(self):
        """Test that all available backends have consistent behavior."""
        available_backends = ['cpu']
        if torch.cuda.is_available():
            available_backends.append('cuda')
        if torch.backends.mps.is_available():
            available_backends.append('mps')
        
        # Test invalid offsets - all should fail
        for backend in available_backends:
            with self.subTest(backend=backend):
                emb_device = self.emb.to(backend)
                indices_device = self.indices.to(backend)
                offsets_device = self.invalid_offsets.to(backend)
                
                with self.assertRaises(RuntimeError) as context:
                    emb_device(indices_device, offsets_device)
                
                error_msg = str(context.exception)
                self.assertIn("offsets[0] has to be 0", error_msg)
        
        # Test valid offsets - all should succeed
        for backend in available_backends:
            with self.subTest(backend=backend):
                emb_device = self.emb.to(backend)
                indices_device = self.indices.to(backend)
                offsets_device = self.valid_offsets.to(backend)
                
                try:
                    output = emb_device(indices_device, offsets_device)
                    self.assertEqual(output.shape, (3, 16))
                except Exception as e:
                    self.fail(f"{backend} should accept valid offsets, but got error: {e}")
    
    def test_edge_cases(self):
        """Test edge cases for offsets validation."""
        # Empty offsets
        empty_offsets = torch.tensor([], dtype=torch.long)
        empty_indices = torch.tensor([], dtype=torch.long)
        
        try:
            # This should work (empty case)
            output = self.emb(empty_indices, empty_offsets)
            self.assertEqual(output.shape, (0, 16))
        except Exception as e:
            self.fail(f"Empty offsets should be handled gracefully, but got error: {e}")
        
        # Single offset with value 0 (valid)
        single_valid_offset = torch.tensor([0])
        single_indices = torch.tensor([1, 2, 3])
        
        try:
            output = self.emb(single_indices, single_valid_offset)
            self.assertEqual(output.shape, (1, 16))
        except Exception as e:
            self.fail(f"Single valid offset should work, but got error: {e}")
        
        # Single offset with value != 0 (invalid)
        single_invalid_offset = torch.tensor([3])
        
        with self.assertRaises(RuntimeError) as context:
            self.emb(single_indices, single_invalid_offset)
        
        error_msg = str(context.exception)
        self.assertIn("offsets[0] has to be 0", error_msg)


if __name__ == "__main__":
    run_tests()
