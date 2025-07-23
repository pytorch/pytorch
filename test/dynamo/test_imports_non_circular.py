# Owner(s): ["module: dynamo"]
"""
This file is aimed at providing a simple testcase to reproduce
https://github.com/pytorch/pytorch/issues/158120

This means that we cannot rely on torch.dynamo before importing 
torch.export, so we can't add this to a file that is a dynamo testcase
"""
import unittest
import torch


class TestImports(unittest.TestCase):

    def test_circular_import_with_export_meta(self):
        from torch import nn
        from torch.export import export
        try:
            conv = torch.nn.Conv2d(3, 64, 3, padding=1)
            with torch.device("meta"):
                _ = export(conv, (torch.zeros(64, 3, 1, 1),))
        except AttributeError as e:
            # Fail due to circular import
            raise e

if __name__ == "__main__":
    unittest.main()

  
