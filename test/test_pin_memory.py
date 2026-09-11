from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils.data._utils.pin_memory import pin_memory


class TestCustomPinMemory(TestCase):
    def test_device_argument(self):
        class Batch:
            def __init__(self):
                self.device = None

            def pin_memory(self, device):
                self.device = device
                return self

        batch = Batch()
        result = pin_memory(batch, "xpu")

        self.assertIs(result, batch)
        self.assertEqual(batch.device, "xpu")

    def test_legacy_no_argument(self):
        class Batch:
            def __init__(self):
                self.called = False

            def pin_memory(self):
                self.called = True
                return self

        batch = Batch()
        result = pin_memory(batch, "xpu")

        self.assertIs(result, batch)
        self.assertTrue(batch.called)

    def test_legacy_optional_argument(self):
        class Batch:
            def __init__(self):
                self.copy = False

            def pin_memory(self, copy=False):
                self.copy = copy
                return self

        batch = Batch()
        pin_memory(batch, "xpu")

        self.assertFalse(batch.copy)


if __name__ == "__main__":
    run_tests()
