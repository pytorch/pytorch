# Owner(s): ["module: inductor"]
# Unit test for TORCHINDUCTOR_LITE_MODE, the env switch that re-runs an existing
# test list under all-fallback mode without editing the tests. The variable is read
# when torch._inductor.config is imported, so it cannot be toggled in-process: this
# asserts both polarities instead, and is meaningful whether or not it is set.
import os
import sys
import unittest

import torch
from torch._inductor import config, lite_mode_options
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_cpp_code
from torch.testing import FileCheck
from torch.testing._internal.common_utils import IS_CI, IS_WINDOWS


if IS_WINDOWS and IS_CI:
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_torchinductor yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/functorch/filelock")

try:
    try:
        from .test_aot_inductor import AOTIRunnerUtil
    except ImportError:
        from test_aot_inductor import AOTIRunnerUtil  # @manual
except (unittest.SkipTest, ImportError):
    if __name__ == "__main__":
        sys.exit(0)
    raise


LITE_MODE = os.environ.get("TORCHINDUCTOR_LITE_MODE") == "1"


class LiteModeEnvTest(TestCase):
    def test_env_var_installs_the_whole_bundle(self):
        # Every knob torch.compile(mode="lite") sets must already be in effect
        # from the environment alone, before any test-side config.patch. The
        # dict lookup also fails loudly if a lite_mode_options key is renamed.
        current = {k: config.get_config_copy()[k] for k in lite_mode_options}
        if LITE_MODE:
            # Catches drift too: a knob added to lite_mode_options but not wired
            # to lite_mode_default shows up here as a mismatch.
            self.assertEqual(current, lite_mode_options)
        else:
            self.assertNotEqual(current, lite_mode_options)

    def test_helper_is_not_a_config_entry(self):
        # A module-level bool would become a settable config knob that does
        # nothing once the defaults have been computed at import.
        self.assertNotIn("_lite_mode", config.get_config_copy())

    def test_ops_actually_fall_back(self):
        # Ops Inductor would normally fuse into one generated kernel go to ATen
        # instead, so no fused kernel is emitted at all.
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.cos(torch.sin(x) + 1)

        example_inputs = (torch.randn(8, 8),)
        _, code = run_and_get_cpp_code(AOTIRunnerUtil.compile, Model(), example_inputs)
        if LITE_MODE:
            self.assertNotIn("cpp_fused_", code)
        else:
            FileCheck().check("cpp_fused_").run(code)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests(needs="filelock")
