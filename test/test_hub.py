

import torch
import tempfile
from torch.testing._internal.common_utils import TestCase


class TestHub(TestCase):
    def test_hub_dir(self):
        with tempfile.TemporaryDirectory('hub_dir') as dirname:
            torch.hub.set_dir(dirname)
            self.assertEqual(torch.hub._get_torch_home(), dirname)
