import os.path
import tempfile
import unittest

import torch
from torch import ops

from model import Model


class TestCustomOperators(unittest.TestCase):

    def test_calling_custom_op_inside_script_module(self):
        #model = Model()
        #output = model.forward(["python input"])
        #self.assertTrue(output == "hi")
        pass
        #print(f"test_jit_hooks.py output:{output}")

    def test_saving_and_loading_script_module_with_custom_op(self):
        model = Model()
        # Ideally we would like to not have to manually delete the file, but NamedTemporaryFile
        # opens the file, and it cannot be opened multiple times in Windows. To support Windows,
        # close the file after creation and try to remove it manually.
        file = tempfile.NamedTemporaryFile(delete=False)
        try:
            file.close()
            model.save(file.name)
            loaded = torch.jit.load(file.name)
        finally:
            os.unlink(file.name)

        output = loaded.forward(torch.ones(5))
        self.assertTrue(output == "hi")


if __name__ == "__main__":
    unittest.main()
