




import unittest

import torch
from caffe2.python import core, workspace

# This is a standalone test that doesn't use test_util as we're testing
# initialization and thus we should be the ones calling GlobalInit
@unittest.skipIf(not workspace.has_cuda_support,
                 "THC pool testing is obscure and doesn't work on HIP yet")
class TestGPUInit(unittest.TestCase):
    def testTHCAllocator(self):
        cuda_or_hip = 'hip' if workspace.has_hip_support else 'cuda'
        flag = '--caffe2_{}_memory_pool=thc'.format(cuda_or_hip)
        core.GlobalInit(['caffe2', flag])
        # just run one operator
        # it's importantant to not call anything here from Torch API
        # even torch.cuda.memory_allocated would initialize CUDA context
        workspace.RunOperatorOnce(core.CreateOperator(
            'ConstantFill', [], ["x"], shape=[5, 5], value=1.0,
            device_option=core.DeviceOption(workspace.GpuDeviceType)
        ))
        # make sure we actually used THC allocator
        self.assertGreater(torch.cuda.memory_allocated(), 0)

if __name__ == '__main__':
    unittest.main()
