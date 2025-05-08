# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
# Owner(s): ["module: tests"]

import torch
import torch.utils.data
import os
import contextlib
from torch.testing._internal.common_utils import TestCase

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

# Tests Vital Signs for Torch
# FIXME: document or deprecate whatever this is
class TestBasicVitalSigns(TestCase):
    def test_basic_vitals(self):
        with torch_vital_set(''):
            self.assertFalse(torch.vitals_enabled())
        with torch_vital_set('ON'):
            self.assertTrue(torch.vitals_enabled())

    def test_basic_vitals_read_write(self):
        with torch_vital_set('ON'):
            self.assertTrue(torch.vitals_enabled())
            # This tests the code path of setting a vital
            self.assertTrue(torch.set_vital('Dataloader', 'basic_unit_test', 'TEST_VALUE_STRING'))
            self.assertIn('TEST_VALUE_STRING', torch.read_vitals())
            self.assertIn('CUDA.used', torch.read_vitals())

    def test_dataloader_vitals(self):
        with torch_vital_set('ON'):
            inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
            tgts = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
            dataset = torch.utils.data.TensorDataset(inps, tgts)
            torch.utils.data.DataLoader(dataset, batch_size=2)
            self.assertIn('Dataloader.enabled\t\t True', torch.read_vitals())

if __name__ == '__main__':
    from torch.testing._internal.common_utils import run_tests
    run_tests()
