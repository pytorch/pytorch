# Owner(s): ["module: nvfuser"]

import unittest

import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import run_tests, TEST_WITH_ROCM, TestCase
from torch.testing._internal.jit_utils import RUN_CUDA
from torch._C._nvfuser import FusionManager, FusionDefinition, DataType

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
  
            t2 = fd.ops.add(t0, t1)
  
            c0 = fd.define_constant(3.0)
  
            t3 = fd.ops.mul(t2, c0)
            t4 = fd.ops.sum(t3, [-1], False, DataType.Float)
  
            fd.add_output(t4)

        input1 = torch.ones(2, 4, 8, device='cuda')
        input2 = torch.ones(2, 4, 8, device='cuda')

        # Expected Output is a tensor of 48's
        nvf_out1 = fm.execute([input1, input2])
      
        # Run the same definition to check caching
        with FusionDefinition(fm) as fd :
            t0 = fd.define_tensor(3)
            t1 = fd.define_tensor(3)
  
            t2 = fd.ops.add(t0, t1)
  
            c0 = fd.define_constant(3.0)
  
            t3 = fd.ops.mul(t2, c0)
            t4 = fd.ops.sum(t3, [-1], False, DataType.Float)
  
            fd.add_output(t4)
       
        nvf_out2 = fm.execute([input1, input2])
        
        eager_out = torch.sum((input1 + input2) * 3.0, dim=-1)
        self.assertEqual(eager_out, nvf_out1[0])
        self.assertEqual(eager_out, nvf_out2[0])
    
    def test_basic_fp16(self) :
        fm = FusionManager.get()
        with FusionDefinition(fm) as fd :
            t0 = fd.define_tensor(3, DataType.Half)
            t1 = fd.define_tensor(3, DataType.Half)
  
            t2 = fd.ops.add(t0, t1)
  
            c0 = fd.define_constant(3.0)
  
            t3 = fd.ops.mul(t2, c0)
            t4 = fd.ops.sum(t3, [-1], False)

            t5 = fd.ops.cast(t4, DataType.Half)
            fd.add_output(t5)

        input1 = torch.ones(2, 4, 8, device='cuda', dtype=torch.float16)
        input2 = torch.ones(2, 4, 8, device='cuda', dtype=torch.float16)

        # Expected Output is a tensor of 48's
        nvf_out = fm.execute([input1, input2])
      
        eager_out = torch.sum((input1 + input2) * 3.0, dim=-1)
        self.assertEqual(eager_out, nvf_out[0])

if __name__ == '__main__':
    run_tests()
