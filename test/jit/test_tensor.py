import torch
from torch.testing._internal.jit_utils import JitTestCase


class TestTensor(JitTestCase):
    def test_numel(self):
        @torch.jit.script
        def get_numel_script(x):
            return x.numel()

        x = torch.rand(3, 4)
        numel = get_numel_script(x)
        self.assertEqual(numel, 3 * 4)

    def test_element_size(self):
        @torch.jit.script
        def get_element_size_script(x):
            return x.element_size()

        x = torch.rand(3, 4)
        element_size = get_element_size_script(x)
        self.assertEqual(element_size, 4)

if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )
