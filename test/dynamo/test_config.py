# Owner(s): ["module: dynamo"]

from collections import namedtuple
from enum import Enum
from types import ModuleType

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.utils import disable_cache_limit

# NB: do NOT include this test class in test_dynamic_shapes.py


class MyModule(ModuleType):
    pass


class ConfigTests(torch._dynamo.test_case.TestCase):
    def test_allowed_config_types(self):
        my_module = MyModule("my_module")

        my_module.config_1 = ["a", 1, 1.0, True]
        my_module.config_2 = namedtuple("Name", "x y")(1, 2), None
        my_module.config_3 = {"k": 1, 1: True, 2.0: "a"}, b"bytestring".hex()

        torch._dynamo.config_utils.install_config_module(my_module)
        assert all(f"config_{i+1}" in my_module._config for i in range(3))

    def test_allowed_but_not_officially_supported(self):
        class IntEnum(int, Enum):
            RED = 1
            GREEN = 2

        class FloatEnum(float, Enum):
            RED = 1  # serialized to 1.0
            GREEN = 2

        class StringEnum(str, Enum):
            RED = 1  # serialized to "1"
            GREEN = "GREEN"

        class ListEnum(list, Enum):
            RED = [1]
            GREEN = "GREEN"  # serialized to ["G", "R", "E", "E", "N"]

        my_module = MyModule("my_module")
        my_module.config_1 = [
            IntEnum.RED,
            FloatEnum.GREEN,
            StringEnum.RED,
            ListEnum.GREEN,
        ]
        torch._dynamo.config_utils.install_config_module(my_module)

    def test_disallowed_configs(self):
        def fn(x):
            return x + 1

        class MyClass:
            pass

        for config in [
            # Nest disallowed types in containers
            [{"a"}],
            {fn: 1},
            {"b": b"bytestring"},
            [MyClass()],
            [MyClass],
            {"m": MyModule},
            {"m": MyModule("a_module")},
        ]:
            my_module = MyModule("my_module")
            my_module.config = config
            with self.assertRaisesRegex(
                ValueError, "Config needs to be deterministically serializable"
            ):
                torch._dynamo.config_utils.install_config_module(my_module)

    def test_able_to_import_constant_functions(self):
        for fn_name in torch._dynamo.config.constant_functions:
            fn = torch._dynamo.variables.torch.get_function_from_string(fn_name)

    @disable_cache_limit()
    def test_no_automatic_dynamic(self):
        def fn(a, b):
            return a - b * 10

        torch._dynamo.reset()
        cnt_static = torch._dynamo.testing.CompileCounter()
        with torch._dynamo.config.patch(
            automatic_dynamic_shapes=False, assume_static_by_default=True
        ):
            opt_fn = torch._dynamo.optimize(cnt_static)(fn)
            for i in range(2, 12):
                opt_fn(torch.randn(i), torch.randn(i))
        self.assertEqual(cnt_static.frame_count, 10)

    @disable_cache_limit()
    def test_automatic_dynamic(self):
        def fn(a, b):
            return a - b * 10

        torch._dynamo.reset()
        cnt_dynamic = torch._dynamo.testing.CompileCounter()
        with torch._dynamo.config.patch(
            automatic_dynamic_shapes=True, assume_static_by_default=True
        ):
            opt_fn = torch._dynamo.optimize(cnt_dynamic)(fn)
            # NB: must not do 0, 1 as they specialized
            for i in range(2, 12):
                opt_fn(torch.randn(i), torch.randn(i))
        # two graphs now rather than 10
        self.assertEqual(cnt_dynamic.frame_count, 2)

    @disable_cache_limit()
    def test_no_assume_static_by_default(self):
        def fn(a, b):
            return a - b * 10

        torch._dynamo.reset()
        cnt_dynamic = torch._dynamo.testing.CompileCounter()
        with torch._dynamo.config.patch(
            automatic_dynamic_shapes=True, assume_static_by_default=False
        ):
            opt_fn = torch._dynamo.optimize(cnt_dynamic)(fn)
            # NB: must not do 0, 1 as they specialized
            for i in range(2, 12):
                opt_fn(torch.randn(i), torch.randn(i))
        # one graph now, as we didn't wait for recompile
        self.assertEqual(cnt_dynamic.frame_count, 1)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
