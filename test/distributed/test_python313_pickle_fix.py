"""
Test for Python 3.13 pickle compatibility fix (Issue #174669)
"""
import unittest
import sys
import traceback as tb
import torch
from torch.distributed.distributed_c10d import _object_to_tensor


class TestPython313PickleFix(unittest.TestCase):
    
    def test_traceback_pickle_compatibility(self):
        """Test that traceback objects can be processed safely"""
        
        # Create a traceback object
        try:
            raise RuntimeError("Test error for traceback")
        except Exception:
            trace = tb.extract_tb(sys.exc_info()[2])
        
        # This should not raise TypeError even in Python 3.13
        device = torch.device('cpu')
        try:
            result = _object_to_tensor(trace, device, None)
            self.assertIsNotNone(result)
        except TypeError as e:
            if "cannot pickle code objects" in str(e):
                self.fail("Python 3.13 pickle bug not fixed")
            else:
                raise
    
    def test_normal_objects_still_work(self):
        """Test that normal objects still pickle correctly"""
        
        test_obj = {"test": "data", "number": 42}
        device = torch.device('cpu')
        
        # This should work normally
        result = _object_to_tensor(test_obj, device, None)
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()
