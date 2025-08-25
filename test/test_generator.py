#!/usr/bin/env python3
# Owner(s): ["module: tensor"]

import sys
import pickle
import unittest
from typing import List, Optional
from unittest import mock

import torch
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
    TEST_CUDA,
    TEST_WITH_ROCM,
    skipIfNoLapack,
    skipIfTorchDynamo,
)
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyCPU,
    onlyCUDA,
    dtypesIfCUDA,
    deviceCountAtLeast,
    skipMeta,
)


# Test for torch.Generator
class TestGenerator(TestCase):
    def test_generator_cpu(self):
        # Test creating a CPU generator
        g = torch.Generator()
        self.assertEqual(g.device.type, 'cpu')
        
        # Test manual seed and get_state/set_state
        g.manual_seed(12345)
        state = g.get_state()
        
        # Generate random numbers
        t1 = torch.rand(5, 5, generator=g)
        
        # Reset the state and generate again
        g.set_state(state)
        t2 = torch.rand(5, 5, generator=g)
        
        # Should be equal because we reset the state
        self.assertEqual(t1, t2)
        
        # Check initial seed
        self.assertEqual(g.initial_seed(), 12345)

    def test_generator_explicit_device(self):
        # Test explicit CPU device specification
        g = torch.Generator(device='cpu')
        self.assertEqual(g.device.type, 'cpu')
        
        # Test with a device object
        g = torch.Generator(device=torch.device('cpu'))
        self.assertEqual(g.device.type, 'cpu')

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_generator_cuda(self):
        # Test creating a CUDA generator
        g = torch.Generator(device='cuda')
        self.assertEqual(g.device.type, 'cuda')
        
        # Test with a device object
        g = torch.Generator(device=torch.device('cuda'))
        self.assertEqual(g.device.type, 'cuda')
        
        # Test manual seed and get_state/set_state
        g.manual_seed(12345)
        state = g.get_state()
        
        # Generate random numbers
        t1 = torch.rand(5, 5, generator=g, device='cuda')
        
        # Reset the state and generate again
        g.set_state(state)
        t2 = torch.rand(5, 5, generator=g, device='cuda')
        
        # Should be equal because we reset the state
        self.assertEqual(t1, t2)
        
        # Check initial seed
        self.assertEqual(g.initial_seed(), 12345)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @deviceCountAtLeast(2)
    def test_generator_cuda_multi_device(self):
        # Test creating generators for specific CUDA devices
        g0 = torch.Generator(device='cuda:0')
        g1 = torch.Generator(device='cuda:1')
        
        self.assertEqual(g0.device.type, 'cuda')
        self.assertEqual(g0.device.index, 0)
        self.assertEqual(g1.device.type, 'cuda')
        self.assertEqual(g1.device.index, 1)
        
        # Test they generate different random numbers
        g0.manual_seed(12345)
        g1.manual_seed(12345)
        
        t0 = torch.rand(5, 5, generator=g0, device='cuda:0')
        t1 = torch.rand(5, 5, generator=g1, device='cuda:1')
        
        # Move to same device to compare
        self.assertNotEqual(t0.cpu(), t1.cpu())

    def test_generator_methods(self):
        g = torch.Generator()
        
        # Test seed() - should set a random seed
        original_state = g.get_state()
        g.seed()
        new_state = g.get_state()
        self.assertNotEqual(original_state, new_state)

    def test_generator_deterministic(self):
        # Test deterministic behavior with same seed
        g1 = torch.Generator()
        g2 = torch.Generator()
        
        g1.manual_seed(12345)
        g2.manual_seed(12345)
        
        t1 = torch.rand(10, generator=g1)
        t2 = torch.rand(10, generator=g2)
        
        self.assertEqual(t1, t2)
        
        # Different seeds should produce different results
        g1.manual_seed(12345)
        g2.manual_seed(54321)
        
        t1 = torch.rand(10, generator=g1)
        t2 = torch.rand(10, generator=g2)
        
        self.assertNotEqual(t1, t2)

    def test_generator_with_different_random_ops(self):
        # Test generator with different random operations
        g = torch.Generator()
        g.manual_seed(12345)
        
        # Test with different random distributions
        t1 = torch.rand(5, 5, generator=g)
        t2 = torch.randn(5, 5, generator=g)
        t3 = torch.randint(0, 10, (5, 5), generator=g)
        
        # Reset and regenerate
        g.manual_seed(12345)
        t1_copy = torch.rand(5, 5, generator=g)
        t2_copy = torch.randn(5, 5, generator=g)
        t3_copy = torch.randint(0, 10, (5, 5), generator=g)
        
        # Should match the originals
        self.assertEqual(t1, t1_copy)
        self.assertEqual(t2, t2_copy)
        self.assertEqual(t3, t3_copy)

    def test_generator_inplace_ops(self):
        # Test in-place random operations
        g = torch.Generator()
        g.manual_seed(12345)
        
        t1 = torch.empty(5, 5)
        t1.random_(generator=g)
        
        g.manual_seed(12345)
        t2 = torch.empty(5, 5)
        t2.random_(generator=g)
        
        self.assertEqual(t1, t2)
        
        # Test other in-place distributions
        g.manual_seed(12345)
        t1 = torch.empty(5, 5)
        t1.uniform_(0, 1, generator=g)
        
        g.manual_seed(12345)
        t2 = torch.empty(5, 5)
        t2.uniform_(0, 1, generator=g)
        
        self.assertEqual(t1, t2)
        
        g.manual_seed(12345)
        t1 = torch.empty(5, 5)
        t1.normal_(mean=0, std=1, generator=g)
        
        g.manual_seed(12345)
        t2 = torch.empty(5, 5)
        t2.normal_(mean=0, std=1, generator=g)
        
        self.assertEqual(t1, t2)

    def test_generator_device_mismatch(self):
        # Test behavior when generator device doesn't match tensor device
        g_cpu = torch.Generator(device='cpu')
        g_cpu.manual_seed(12345)
        
        # CPU generator with CPU tensor - should work
        t_cpu = torch.empty(5, 5, device='cpu')
        t_cpu.uniform_(0, 1, generator=g_cpu)  # This should work fine
        
        if torch.cuda.is_available():
            g_cuda = torch.Generator(device='cuda')
            g_cuda.manual_seed(12345)
            
            # CUDA generator with CUDA tensor - should work
            t_cuda = torch.empty(5, 5, device='cuda')
            t_cuda.uniform_(0, 1, generator=g_cuda)  # This should work fine
            
            # Test mismatched devices
            with self.assertRaises(RuntimeError):
                # CPU generator with CUDA tensor - should fail
                t_cuda = torch.empty(5, 5, device='cuda')
                t_cuda.uniform_(0, 1, generator=g_cpu)
            
            with self.assertRaises(RuntimeError):
                # CUDA generator with CPU tensor - should fail
                t_cpu = torch.empty(5, 5, device='cpu')
                t_cpu.uniform_(0, 1, generator=g_cuda)

    def test_pickle_generator(self):
        # Test pickling and unpickling generator
        devices = ['cpu']
        if torch.cuda.is_available():
            devices.append('cuda')
        if torch.backends.mps.is_available():
            devices.append('mps')
            
        for device in devices:
            with self.subTest(device=device):
                g = torch.Generator(device=device)
                g.manual_seed(12345)
                
                # Use the generator to advance its state
                torch.rand(10, 10, generator=g, device=device)
                
                # Pickle and unpickle
                serialized = pickle.dumps(g)
                g2 = pickle.loads(serialized)
                
                # Check device and initial seed are preserved
                self.assertEqual(g.device, g2.device)
                self.assertEqual(g.initial_seed(), g2.initial_seed())
                
                # Check the state is preserved
                self.assertEqual(g.get_state(), g2.get_state())
                
                # Generate tensors and check they're the same
                t1 = torch.rand(5, 5, generator=g, device=device)
                t2 = torch.rand(5, 5, generator=g2, device=device)
                self.assertEqual(t1, t2)

    def test_invalid_device(self):
        # Test invalid device type - expect RuntimeError since XLA is not registered
        with self.assertRaises(RuntimeError):
            torch.Generator(device='xla')


if __name__ == '__main__':
    run_tests()
