# Owner(s): ["module: inductor"]
"""Tests for ``torch._inductor.config.rocm.mfma_nonkdim``.

The config reads the env var ``TORCHINDUCTOR_MFMA_NONKDIM`` at module
import time, so we test the helper functions in
``torch._inductor.template_heuristics.triton`` by patching the config
value directly (`torch._inductor.config.patch`) rather than reloading
the module. We also exercise the env path with a subprocess for the
import-time read.
"""

import os
import subprocess
import sys
import unittest

import torch._inductor.config as inductor_config
from torch._inductor.template_heuristics.triton import (
    _amd_mm_nonkdim_autotune_choices,
    _amd_mm_nonkdim_default,
)
from torch.testing._internal.common_utils import run_tests, TEST_WITH_ROCM, TestCase


@unittest.skipIf(not TEST_WITH_ROCM, "matrix_instr_nonkdim is ROCm/HIP-only")
class TestAmdMfmaNonkdimConfig(TestCase):
    def test_unset_matches_upstream(self):
        # Patch to None so we mimic the "env unset" case regardless of the
        # current process environment.
        with inductor_config.patch({"rocm.mfma_nonkdim": None}):
            self.assertEqual(_amd_mm_nonkdim_default(), 16)
            self.assertEqual(_amd_mm_nonkdim_autotune_choices(), [0, 16])

    def test_force_16_matches_upstream(self):
        with inductor_config.patch({"rocm.mfma_nonkdim": 16}):
            self.assertEqual(_amd_mm_nonkdim_default(), 16)
            self.assertEqual(_amd_mm_nonkdim_autotune_choices(), [16])

    def test_force_32(self):
        with inductor_config.patch({"rocm.mfma_nonkdim": 32}):
            self.assertEqual(_amd_mm_nonkdim_default(), 32)
            self.assertEqual(_amd_mm_nonkdim_autotune_choices(), [32])

    def test_auto_extends_list(self):
        with inductor_config.patch({"rocm.mfma_nonkdim": "auto"}):
            # default stays 16 in "auto" mode (matches upstream ROCmGemmConfig default)
            self.assertEqual(_amd_mm_nonkdim_default(), 16)
            self.assertEqual(_amd_mm_nonkdim_autotune_choices(), [0, 16, 32])

    def test_force_zero(self):
        with inductor_config.patch({"rocm.mfma_nonkdim": 0}):
            self.assertEqual(_amd_mm_nonkdim_default(), 0)
            self.assertEqual(_amd_mm_nonkdim_autotune_choices(), [0])

    def _spawn_env_probe(self, env_value):
        """Spawn a fresh python process with TORCHINDUCTOR_MFMA_NONKDIM set
        to ``env_value`` and read back the parsed config value.

        This validates the import-time env parsing in config.py (which the
        in-process patch tests can't exercise because the value was set at
        import).
        """
        env = os.environ.copy()
        env.pop("TORCHINDUCTOR_MFMA_NONKDIM", None)
        if env_value is not None:
            env["TORCHINDUCTOR_MFMA_NONKDIM"] = env_value
        code = (
            "import torch, json; "
            "import torch._inductor.config as c; "
            "v = c.rocm.mfma_nonkdim; "
            "print(json.dumps({'type': type(v).__name__, 'value': v}))"
        )
        out = (
            subprocess.check_output([sys.executable, "-c", code], env=env)
            .decode()
            .strip()
        )
        import json

        return json.loads(out)

    def test_env_parsing_in_fresh_process(self):
        for env_value, expected in [
            (None, None),
            ("16", 16),
            ("32", 32),
            ("0", 0),
            ("auto", "auto"),
            ("AUTO", "auto"),
            ("notanint", None),
            ("", None),
        ]:
            r = self._spawn_env_probe(env_value)
            self.assertEqual(
                r["value"],
                expected,
                f"env=`{env_value}` got {r['value']} expected {expected}",
            )


if __name__ == "__main__":
    run_tests()
