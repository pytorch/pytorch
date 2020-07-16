import torch
import unittest
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.common_cuda import TEST_CUDNN
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_device_type import instantiate_device_type_tests

class TestAAA(NNTestCase):
    def test_aaa(self, device):
        if not TEST_CUDNN:
            print('cuDNN is not available')
        else:
            print('cuDNN version ' + str(torch.backends.cudnn.version()))

        x: torch.Tensor = torch.randn(1, 2, 3, 3, 3, dtype=torch.float, device=device).requires_grad_()
        conv = torch.nn.Conv3d(2, 4, kernel_size=3, groups=2).float().to(device=device)

        out = conv(x)
        go = torch.randn_like(out)

        gi = torch.autograd.grad(out, x, go, create_graph=True)[0]

        with torch.autograd.profiler.profile(use_cuda=(device == 'cuda'), record_shapes=True) as prof:
            gi.sum().backward()

        print(prof.table())

        if device == 'cuda':
            torch.cuda.synchronize()
 
        self.fail('stop running all other tests')

instantiate_device_type_tests(TestAAA, globals())

if __name__ == '__main__':
    run_tests()