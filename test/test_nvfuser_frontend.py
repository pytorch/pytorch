# Owner(s): ["module: nvfuser"]

import unittest

import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import run_tests, TEST_WITH_ROCM, TestCase
from torch.testing._internal.jit_utils import RUN_CUDA
from torch._C._nvfuser import FusionManager, FusionDefinition, DataType
import torch._refs as refs
import torch._prims as prims

RUN_NVFUSER = RUN_CUDA and not TEST_WITH_ROCM

def is_pre_volta():
    if not RUN_NVFUSER:
        return False
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return prop.major < 7

@unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
@unittest.skipIf(is_pre_volta(), "Only supported on Volta and newer devices.")
class TestNVFuserFrontend(TestCase):
    def test_basic(self) :
        fm = FusionManager.get()
        with FusionDefinition(fm) as fd :
            t0 = fd.define_tensor(3)
            t1 = fd.define_tensor(3)
            c0 = fd.define_constant(3.0)
  
            t2 = fd.ops.add(t0, t1)
            t3 = fd.ops.mul(t2, c0)
            t4 = fd.ops.sum(t3, [-1], False, DataType.Float)
  
            fd.add_output(t4)

        input1 = torch.ones(2, 4, 8, device='cuda')
        input2 = torch.ones(2, 4, 8, device='cuda')

        # Expected Output is a tensor of 48's
        nvf_out1 = fm.execute([input1, input2])[0]
      
        # Run the same definition to check caching
        with FusionDefinition(fm) as fd :
            t0 = fd.define_tensor(3)
            t1 = fd.define_tensor(3)
            c0 = fd.define_constant(3.0)
  
            t2 = fd.ops.add(t0, t1)
            t3 = fd.ops.mul(t2, c0)
            t4 = fd.ops.sum(t3, [-1], False, DataType.Float)
  
            fd.add_output(t4)
       
        nvf_out2 = fm.execute([input1, input2])[0]
        eager_out = torch.sum((input1 + input2) * 3.0, dim=-1)
        self.assertEqual(eager_out, nvf_out1)
        self.assertEqual(eager_out, nvf_out2)
    
    def test_basic_fp16(self) :
        fm = FusionManager.get()
        with FusionDefinition(fm) as fd :
            t0 = fd.define_tensor(3, DataType.Half)
            t1 = fd.define_tensor(3, DataType.Half)
            c0 = fd.define_constant(3.0)
  
            t2 = fd.ops.add(t0, t1)
            t3 = fd.ops.mul(t2, c0)
            t4 = fd.ops.sum(t3, [-1], False, DataType.Float)

            t5 = fd.ops.cast(t4, DataType.Half)
            fd.add_output(t5)

        input1 = torch.ones(2, 4, 8, device='cuda', dtype=torch.float16)
        input2 = torch.ones(2, 4, 8, device='cuda', dtype=torch.float16)

        # Expected Output is a tensor of 48's
        nvf_out = fm.execute([input1, input2])[0]
        eager_out = torch.sum((input1 + input2) * 3.0, dim=-1)
        self.assertEqual(eager_out, nvf_out)
    
    def test_cast_double_to_half(self) :
        fm = FusionManager.get()
        with FusionDefinition(fm) as fd :
            t0 = fd.define_tensor(2, DataType.Double)
            t1 = fd.define_tensor(2, DataType.Double)
  
            t0h = fd.ops.cast(t0, DataType.Half)
            t1h = fd.ops.cast(t1, DataType.Half)
            t2 = fd.ops.add(t0h, t1h)
            t3 = fd.ops.relu(t2)
            t4 = fd.ops.cast(t3, DataType.Half)
  
            fd.add_output(t4)

        input1 = torch.randn(2, 4, device='cuda', dtype=torch.float64)
        input2 = torch.randn(2, 4, device='cuda', dtype=torch.float64)
    
        nvf_out = fm.execute([input1, input2])[0]
        eager_out = torch.relu(input1.to(torch.half) + input2.to(torch.half))
        self.assertEqual(eager_out, nvf_out)
    
    def test_promote_to_double(self) :
        fm = FusionManager.get()

        with FusionDefinition(fm) as fd :
            t0 = fd.define_tensor(2, DataType.Half)
            t1 = fd.define_tensor(2, DataType.Double)
  
            t2 = fd.ops.add(t0, t1)
            t5 = fd.ops.relu(t2)
  
            fd.add_output(t5)

        input1 = torch.randn(2, 4, device='cuda', dtype=torch.float16)
        input2 = torch.randn(2, 4, device='cuda', dtype=torch.float64)

        nvf_out = fm.execute([input1, input2])[0]
        eager_out = torch.relu(input1 + input2)
        self.assertEqual(eager_out, nvf_out)
    
    def test_implicit_broadcast_input(self) :
        fm = FusionManager.get()
        with FusionDefinition(fm) as fd :
            t0 = fd.define_tensor(1)
            t1 = fd.define_tensor(3)
  
            t0_b = fd.ops.broadcast_in_dim(t0, [2, 3, 4], [1])
            t2 = fd.ops.add(t0_b, t1)
  
            fd.add_output(t2)

        input1 = torch.randn(3, device='cuda')
        input2 = torch.randn(2, 3, 4, device='cuda')
        
        nvf_out = fm.execute([input1, input2])[0]
        eager_out = refs.add(prims.broadcast_in_dim(input1, [2, 3, 4], [1]), input2)
        self.assertEqual(eager_out, nvf_out)

    def test_explicit_broadcast_input(self) :
        input1 = torch.randn(1, 1, 4, device='cuda')
        input2 = torch.randn(2, 3, 4, device='cuda')

        fm = FusionManager.get()
        with FusionDefinition(fm) as fd :
            t0 = fd.define_tensor(sizes=input1.size(), strides=input1.stride())
            t1 = fd.define_tensor(sizes=input2.size(), strides=input2.stride())
  
            t0_b = fd.ops.broadcast_in_dim(t0, [2, 3, 4], [0, 1, 2])
            t2 = fd.ops.add(t0_b, t1)
  
            fd.add_output(t2)

        nvf_out = fm.execute([input1, input2])[0]
        eager_out = refs.add(prims.broadcast_in_dim(input1, [2, 3, 4], [0, 1, 2]), input2)
        self.assertEqual(eager_out, nvf_out)

    def test_broadcast_mixing(self) :
        fm = FusionManager.get()
        with FusionDefinition(fm) as fd :
            t0 = fd.define_tensor([3, 1], [1, 1])
            t1 = fd.define_tensor(1)

            t1_b = fd.ops.broadcast_in_dim(t1, [3, 3], [0]) 
            t2 = fd.ops.add(t0, t1_b)
  
            fd.add_output(t2)

        input1 = torch.randn(3, 1, device='cuda')
        input2 = torch.randn(3, device='cuda')

        nvf_out = fm.execute([input1, input2])[0]
        eager_out = refs.add(input1, prims.broadcast_in_dim(input2, [3, 3], [0]))
        self.assertEqual(eager_out, nvf_out)

if __name__ == '__main__':
    run_tests()
