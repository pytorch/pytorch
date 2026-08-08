# Owner(s): ["module: PrivateUse1"]
import sys

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


class TestExtensionUtils(TestCase):
    def tearDown(self):
        # Clean up
        backend_name = torch._C._get_privateuse1_backend_name()
        for name in {backend_name, "privateuseone", "bar", "baz"}:
            if hasattr(torch, name):
                delattr(torch, name)
            if f"torch.{name}" in sys.modules:
                del sys.modules[f"torch.{name}"]

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

    @skipIfTorchDynamo(
        "accelerator doesn't compose with privateuse1 : https://github.com/pytorch/pytorch/issues/166696"
    )
    def test_external_module_register_with_renamed_backend(self):
        torch.utils.rename_privateuse1_backend("foo")
        with self.assertRaisesRegex(RuntimeError, "has already been set"):
            torch.utils.rename_privateuse1_backend("dummmy")

        custom_backend_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(custom_backend_name, "foo")
        self.assertEqual(torch.accelerator.current_accelerator(), torch.device("foo"))
        torch.accelerator.set_current_accelerator("foo")
        self.assertEqual(torch.accelerator.current_accelerator(), torch.device("foo"))

        bar_device = torch.utils.register_backend("bar", DummyPrivateUse1Module)
        self.assertEqual(bar_device, torch.device("bar"))
        self.assertEqual(torch.utils.register_backend("bar"), bar_device)
        self.assertIs(torch.get_device_module("bar"), DummyPrivateUse1Module)
        torch.bar.is_available()  # type: ignore[attr-defined]
        self.assertEqual(torch.accelerator.current_accelerator(), torch.device("foo"))
        torch.accelerator.set_current_accelerator("bar")
        self.assertEqual(torch.accelerator.current_accelerator(), torch.device("bar"))
        self.assertEqual(
            torch._C._dispatch_key_parse("BAR"), torch._C.DispatchKey.PrivateUse2
        )
        self.assertEqual(
            torch._C._dispatch_key_parse("SparseBAR"),
            torch._C.DispatchKey.SparsePrivateUse2,
        )
        self.assertEqual(
            torch._C._dispatch_key_parse("QuantizedBAR"),
            torch._C.DispatchKey.QuantizedPrivateUse2,
        )
        self.assertEqual(
            torch._C._dispatch_key_parse("AutogradBAR"),
            torch._C.DispatchKey.AutogradPrivateUse2,
        )

        baz_device = torch.utils.register_backend("baz")
        self.assertEqual(baz_device, torch.device("baz"))
        self.assertEqual(
            torch._C._dispatch_key_parse("BAZ"), torch._C.DispatchKey.PrivateUse3
        )
        self.assertEqual(
            torch._C._dispatch_key_parse("SparseBAZ"),
            torch._C.DispatchKey.SparsePrivateUse3,
        )
        with self.assertRaisesRegex(RuntimeError, "No private-use backend slots"):
            torch.utils.register_backend("qux")

        with self.assertRaises(AttributeError):
            torch.foo.is_available()  # type: ignore[attr-defined]

        with self.assertRaisesRegex(AssertionError, "Tried to use AMP with the"):
            with torch.autocast(device_type=custom_backend_name):
                pass
        torch._register_device_module("foo", DummyPrivateUse1Module)

        torch.foo.is_available()  # type: ignore[attr-defined]
        with torch.autocast(device_type=custom_backend_name):
            pass

        self.assertEqual(torch._utils._get_device_index("foo:1"), 1)
        self.assertEqual(torch._utils._get_device_index(torch.device("foo:2")), 2)


if __name__ == "__main__":
    run_tests()
