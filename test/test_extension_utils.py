# Owner(s): ["module: PrivateUse1"]
import os

from torch.testing._internal.common_utils import run_tests, TestCase

class TestExtensionUtils(TestCase):
    def _get_external_module_register_with_renamed_backend_template(self, backend_name):
        return f"""\
import sys
import torch
from torch.testing._internal.common_utils import run_tests, TestCase

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

class TestRenamedBackend(TestCase):
    def test_external_module_register(self):
        # Built-in module
        with self.assertRaisesRegex(RuntimeError, "The runtime module of"):
            torch._register_device_module("cuda", torch.cuda)

        # Wrong device type
        with self.assertRaisesRegex(RuntimeError, "Expected one of cpu"):
            torch._register_device_module("dummmy", DummyPrivateUse1Module)

        with self.assertRaises(AttributeError):
            torch.privateuseone.is_available()  # type: ignore[attr-defined]

        torch._register_device_module("privateuseone", DummyPrivateUse1Module)

        torch.privateuseone.is_available()  # type: ignore[attr-defined]

        # No supporting for override
        with self.assertRaisesRegex(RuntimeError, "The runtime module of"):
            torch._register_device_module("privateuseone", DummyPrivateUse1Module)

    def test_external_module_register_with_renamed_backend(self):
        torch.utils.rename_privateuse1_backend("{backend_name}")
        with self.assertRaisesRegex(RuntimeError, "has already been set"):
            torch.utils.rename_privateuse1_backend("dummmy")

        custom_backend_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(custom_backend_name, "{backend_name}")

        with self.assertRaises(AttributeError):
            getattr(torch, "{backend_name}").is_available()  # type: ignore[attr-defined]

        with self.assertRaisesRegex(AssertionError, "Tried to use AMP with the"):
            with torch.autocast(device_type=custom_backend_name):
                pass
        torch._register_device_module("{backend_name}", DummyPrivateUse1Module)

        getattr(torch, "{backend_name}").is_available()  # type: ignore[attr-defined]
        with torch.autocast(device_type=custom_backend_name):
            pass

        self.assertEqual(torch._utils._get_device_index("{backend_name}:1"), 1)
        self.assertEqual(torch._utils._get_device_index(torch.device("{backend_name}:2")), 2)

if __name__ == "__main__":
    run_tests()
"""

    def test_external_module_register_with_renamed_backend(self):
        env = dict(os.environ)
        # a backend name is not a c10:DeviceType
        _, stderr = self.run_process_no_exception(
            self._get_external_module_register_with_renamed_backend_template("foo"),
            env=env,
        )
        self.assertIn("Ran 2 test", stderr.decode("utf-8"))
        # a backend name is a c10:DeviceType
        _, stderr = self.run_process_no_exception(
            self._get_external_module_register_with_renamed_backend_template("maia"),
            env=env,
        )
        self.assertIn("Ran 2 test", stderr.decode("utf-8"))

if __name__ == "__main__":
    run_tests()
