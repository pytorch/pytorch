# Owner(s): ["module: PrivateUse1"]

import torch
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


class DummyPrivateUse1Module:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def is_autocast_enabled():
        return True

    @staticmethod
    def get_autocast_dtype():
        return torch.float16

    @staticmethod
    def set_autocast_enabled(enable):
        pass

    @staticmethod
    def set_autocast_dtype(dtype):
        pass

    @staticmethod
    def get_amp_supported_dtype():
        return [torch.float16]


class TestRenamePrivateuseoneToExistingBackend(TestCase):
    @skipIfTorchDynamo(
        "TorchDynamo exposes https://github.com/pytorch/pytorch/issues/166696"
    )
    def test_external_module_register_with_existing_backend(self):
        torch.utils.rename_privateuse1_backend("maia")
        with self.assertRaisesRegex(RuntimeError, "has already been set"):
            torch.utils.rename_privateuse1_backend("dummmy")

        custom_backend_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(custom_backend_name, "maia")

        with self.assertRaises(AttributeError):
            torch.maia.is_available()

        with self.assertRaisesRegex(AssertionError, "Tried to use AMP with the"):
            with torch.autocast(device_type=custom_backend_name):
                pass
        torch._register_device_module("maia", DummyPrivateUse1Module)

        torch.maia.is_available()  # type: ignore[attr-defined]
        with torch.autocast(device_type=custom_backend_name):
            pass

        self.assertEqual(torch._utils._get_device_index("maia:1"), 1)
        self.assertEqual(torch._utils._get_device_index(torch.device("maia:2")), 2)


if __name__ == "__main__":
    run_tests()
