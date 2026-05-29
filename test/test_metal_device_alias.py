import unittest
import torch
import pickle
import copy
import os

class TestMetalDeviceAlias(unittest.TestCase):
    def test_metal_parsing(self):
        # String parsing to mps device type (torch.device("metal") -> mps)
        d = torch.device("metal")
        self.assertEqual(d.type, "mps")
        self.assertEqual(d.index, None)

        d0 = torch.device("metal:0")
        self.assertEqual(d0.type, "mps")
        self.assertEqual(d0.index, 0)

    def test_metal_keyword_arguments(self):
        """Verify keyword-based instantiation routes correctly."""
        d1 = torch.device(type="metal")
        self.assertEqual(d1.type, "mps")
        
        d2 = torch.device(type="metal", index=1)
        self.assertEqual(d2.type, "mps")
        self.assertEqual(d2.index, 1)

    def test_metal_deepcopy_correctly(self):
        """Verify copy.deepcopy preserves alias routing to canonical mps."""
        original = torch.device("metal")
        copied = copy.deepcopy(original)
        self.assertEqual(copied, original)
        self.assertEqual(copied.type, "mps")

    def test_semantic_equality(self):
        # Semantic equality checking (torch.device("metal") == torch.device("mps"))
        self.assertEqual(torch.device("metal"), torch.device("mps"))
        self.assertEqual(torch.device("metal:0"), torch.device("mps:0"))
        self.assertNotEqual(torch.device("metal:0"), torch.device("mps:1"))

    def test_serialization(self):
        # Pickling/unpickling
        d = torch.device("metal:0")
        pickled = pickle.dumps(d)
        unpickled = pickle.loads(pickled)
        # Note: Canonical name is still "mps", so it should unpickle to "mps"
        self.assertEqual(unpickled.type, "mps")
        self.assertEqual(unpickled.index, 0)

    def test_hashing_and_dict(self):
        # Hashing and dict-key access
        d_metal = torch.device("metal:0")
        d_mps = torch.device("mps:0")
        self.assertEqual(hash(d_metal), hash(d_mps))
        
        d = {d_metal: "value"}
        self.assertEqual(d[d_mps], "value")

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS hardware not available")
    def test_allocation(self):
        # Hardware-gated execution blocks
        t = torch.randn(5, device="metal")
        self.assertEqual(t.device.type, "mps")
        
        t0 = torch.randn(5, device="metal:0")
        self.assertEqual(t0.device.type, "mps")
        self.assertEqual(t0.device.index, 0)

if __name__ == "__main__":
    unittest.main()
