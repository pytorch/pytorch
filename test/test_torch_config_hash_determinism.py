# Owner(s): ["module: unknown"]

import getpass
import re
import socket

from torch._inductor import config as inductor_config
from torch.testing._internal.common_utils import run_tests, TestCase


PATH_PATTERN = re.compile(
    r"(?:\/[a-zA-Z0-9_\-\.]+\/[a-zA-Z0-9_\-\.\/]+)|"  # Unix
    r"(?:[a-zA-Z]:\\[a-zA-Z0-9_\-\.\\]+)"  # Windows
)


USERNAME = getpass.getuser()
HOSTNAME = socket.gethostname()


class TestConfigModule(TestCase):
    def check_deterministic(self, key: str, value: object):
        if isinstance(value, (int, float, bool)) or value is None:
            return
        elif isinstance(value, str):
            self.assertFalse(
                PATH_PATTERN.match(value),
                f"Detected path in config value '{value}', key='{key}', "
                "this may cause non-deterministic behavior in compile caching.",
            )
            if USERNAME:
                self.assertNotIn(
                    USERNAME,
                    value,
                    f"Detected username in config value '{value}', key='{key}', "
                    "this may cause non-deterministic behavior in compile caching.",
                )
            if HOSTNAME:
                self.assertNotIn(
                    HOSTNAME,
                    value,
                    f"Detected hostname in config value '{value}', key='{key}', "
                    "this may cause non-deterministic behavior in compile caching.",
                )
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                self.check_deterministic(f"{key}[{i}]", item)
        elif isinstance(value, dict):
            for k, v in value.items():
                self.check_deterministic(f"{key}[{k}]", v)
        else:
            self.fail(f"Unexpected type: {type(value)}")

    def test_basic_detection(self):
        self.check_deterministic("test", "test")
        self.check_deterministic("test", 1)
        self.check_deterministic("test", 1.0)
        self.check_deterministic("test", True)
        self.check_deterministic("test", False)
        self.check_deterministic("test", [1, 2, 3])
        self.check_deterministic("test", (1, 2, 3))
        self.check_deterministic("test", {"a": 1, "b": 2, "c": 3})
        with self.assertRaisesRegex(
            AssertionError, "Detected path in config value '.*', key='test'"
        ):
            self.check_deterministic("test", "/tmp/test")
        if USERNAME:
            with self.assertRaisesRegex(
                AssertionError, "Detected username in config value '.*', key='test'"
            ):
                self.check_deterministic("test", f"123_{USERNAME}")
        if HOSTNAME:
            with self.assertRaisesRegex(
                AssertionError, "Detected hostname in config value '.*', key='test'"
            ):
                self.check_deterministic("test", f"456-{HOSTNAME}")

    def test_inductor_config_hash_portable_deterministic(self):
        torch_config = inductor_config.save_config_portable()

        for key, value in torch_config.items():
            self.check_deterministic(key, value)

    def test_inductor_config_hash_portable_without_ignore(self):
        idx = inductor_config._cache_config_ignore_prefix.index("cuda.cutlass_dir")
        inductor_config._cache_config_ignore_prefix.remove("cuda.cutlass_dir")
        try:
            changed_torch_config = inductor_config.save_config_portable()
            with self.assertRaisesRegex(
                AssertionError,
                "Detected path in config value '.*', key='cuda.cutlass_dir'",
            ):
                for key, value in changed_torch_config.items():
                    self.check_deterministic(key, value)
        finally:
            inductor_config._cache_config_ignore_prefix.insert(idx, "cuda.cutlass_dir")


if __name__ == "__main__":
    run_tests()
