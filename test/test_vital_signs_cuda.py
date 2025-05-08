# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
# Owner(s): ["module: tests"]

import torch
import os
import contextlib
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.common_device_type import onlyCUDA

@contextlib.contextmanager
def torch_vital_set(value):
    stash = None
    if 'TORCH_VITAL' in os.environ:
        stash = os.environ['TORCH_VITAL']
    os.environ['TORCH_VITAL'] = value
    try:
        yield
    finally:
        if stash:
            os.environ['TORCH_VITAL'] = stash
        else:
            del os.environ['TORCH_VITAL']

# FIXME: document or deprecate whatever this is
class TestVitalSignsCuda(TestCase):
    @onlyCUDA
    def test_cuda_vitals_gpu_only(self, device):
        with torch_vital_set('ON'):
            self.assertIn('CUDA.used\t\t true', torch.read_vitals())

if __name__ == '__main__':
    from torch.testing._internal.common_utils import run_tests
    run_tests()
