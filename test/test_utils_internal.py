# Owner(s): ["module: unknown"]

import os

from torch._utils_internal import justknobs_feature, JustKnobsConfig
from torch.testing._internal.common_utils import (  # type: ignore[attr-defined]
    load_tests,
)


# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

from torch.testing._internal.common_utils import run_tests, TestCase


class TestJustKnob(TestCase):
    def test_justknob_config(self):
        with self.subTest("Returns True"):
            a = JustKnobsConfig()
            self.assertTrue(a.get())
        with self.subTest("Returns False"):
            a = JustKnobsConfig(name="fake_name", default=False)
            self.assertFalse(a.get())
        with self.subTest("Returns True via config"):
            a = JustKnobsConfig(name="fake_name", default=False)
            a.set(True)
            self.assertTrue(a.get())
        with self.subTest("Returns True via env"):
            os.environ["FAKE_FEATURE"] = "1"
            a = JustKnobsConfig(
                name="fake_name", env_name="FAKE_FEATURE", default=False
            )
            self.assertTrue(a.get())
        with self.subTest("Returns same value consistently"):
            a = JustKnobsConfig(name="fake_name", default=False)
            a.set(True)
            self.assertTrue(a.get())
            a.set(False)
            self.assertTrue(a.get())
        with self.subTest("Checks __bool__"):
            a = JustKnobsConfig(name="fake_name", default=False)
            if a:
                raise RuntimeError("Should not be true")
            self.assertFalse(a)

    def test_justknob_feature(self):
        with self.subTest("OSS is True"):
            self.assertTrue(justknobs_feature("testname"))
        with self.subTest("OSS default=True"):
            self.assertTrue(justknobs_feature("testname", default=True))
        with self.subTest("OSS default=False"):
            self.assertFalse(justknobs_feature("testname", default=False))
        with self.subTest("OSS config=True, default=False"):
            self.assertTrue(
                justknobs_feature("testname", config_value=True, default=False)
            )
        with self.subTest("OSS config=None, default=False"):
            self.assertFalse(
                justknobs_feature("testname", config_value=None, default=False)
            )
        with self.subTest("OSS config=False, default=True"):
            self.assertFalse(
                justknobs_feature("testname", config_value=False, default=True)
            )
        with self.subTest("OSS env is missing, config=False, default=True"):
            self.assertFalse(
                justknobs_feature(
                    "testname", config_value=False, env_name="NOTDEFINED", default=False
                )
            )
        with self.subTest("OSS env is missing, default=False"):
            self.assertFalse(
                justknobs_feature("testname", env_name="NOTDEFINED", default=False)
            )
        with self.subTest(
            "OSS config overrides env, config=True, env=False, default=False"
        ):
            os.environ["FEATURE_ENV"] = "0"
            self.assertTrue(
                justknobs_feature(
                    "testname",
                    config_value=True,
                    env_name="FEATURE_ENV",
                    default=False,
                )
            )
        with self.subTest("OSS env overrides default, , default=False"):
            os.environ["FEATURE_ENV"] = "1"
            self.assertTrue(
                justknobs_feature("testname", env_name="FEATURE_ENV", default=False)
            )
        with self.subTest("OSS env truthy, config=False, default=False"):
            os.environ["FEATURE_ENV"] = "1"
            self.assertTrue(
                justknobs_feature(
                    "testname",
                    env_name="FEATURE_ENV",
                    default=False,
                )
            )
            os.environ["FEATURE_ENV"] = "true"
            self.assertTrue(
                justknobs_feature(
                    "testname",
                    env_name="FEATURE_ENV",
                    default=False,
                )
            )
            os.environ["FEATURE_ENV"] = "TRUE"
            self.assertTrue(
                justknobs_feature(
                    "testname",
                    env_name="FEATURE_ENV",
                    default=False,
                )
            )
            os.environ["FEATURE_ENV"] = "very weird true"
            self.assertTrue(
                justknobs_feature(
                    "testname",
                    env_name="FEATURE_ENV",
                    default=False,
                )
            )
        with self.subTest("OSS env false, default=True"):
            os.environ["FEATURE_ENV"] = "0"
            self.assertFalse(
                justknobs_feature("testname", env_name="FEATURE_ENV", default=True)
            )
            os.environ["FEATURE_ENV"] = "false"
            self.assertFalse(
                justknobs_feature("testname", env_name="FEATURE_ENV", default=True)
            )
            os.environ["FEATURE_ENV"] = "FALSE"
            self.assertFalse(
                justknobs_feature("testname", env_name="FEATURE_ENV", default=True)
            )


if __name__ == "__main__":
    run_tests()
