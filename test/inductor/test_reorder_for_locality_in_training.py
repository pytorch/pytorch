# Owner(s): ["module: inductor"]
"""Tests for ``torch._inductor.config.reorder_for_locality_in_training``.

The flag reads ``TORCHINDUCTOR_REORDER_LOCALITY_TRAINING`` at module
import time, so we test the runtime gating in
``torch._inductor.fx_passes.post_grad`` by patching the flag directly
(``torch._inductor.config.patch``). We also exercise the env path with
a subprocess.
"""

import os
import subprocess
import sys
from unittest.mock import patch as mock_patch

import torch
import torch._inductor.config as inductor_config
import torch.fx as fx
from torch.testing._internal.common_utils import run_tests, TestCase


class TestReorderForLocalityInTraining(TestCase):
    def test_default_off(self):
        # The default-off statement is what we ship: importing config with
        # the env unset should leave the flag False.
        env = os.environ.copy()
        env.pop("TORCHINDUCTOR_REORDER_LOCALITY_TRAINING", None)
        out = (
            subprocess.check_output(
                [
                    sys.executable,
                    "-c",
                    "import torch._inductor.config as c; print(int(c.reorder_for_locality_in_training))",
                ],
                env=env,
            )
            .decode()
            .strip()
        )
        self.assertEqual(out, "0")

    def test_env_one_turns_on(self):
        env = os.environ.copy()
        env["TORCHINDUCTOR_REORDER_LOCALITY_TRAINING"] = "1"
        out = (
            subprocess.check_output(
                [
                    sys.executable,
                    "-c",
                    "import torch._inductor.config as c; print(int(c.reorder_for_locality_in_training))",
                ],
                env=env,
            )
            .decode()
            .strip()
        )
        self.assertEqual(out, "1")

    def test_env_zero_keeps_off(self):
        env = os.environ.copy()
        env["TORCHINDUCTOR_REORDER_LOCALITY_TRAINING"] = "0"
        out = (
            subprocess.check_output(
                [
                    sys.executable,
                    "-c",
                    "import torch._inductor.config as c; print(int(c.reorder_for_locality_in_training))",
                ],
                env=env,
            )
            .decode()
            .strip()
        )
        self.assertEqual(out, "0")

    def _make_tiny_gm(self):
        g = fx.Graph()
        x = g.placeholder("x")
        y = g.call_function(torch.relu, args=(x,))
        g.output(y)
        return fx.GraphModule(torch.nn.Module(), g)

    def test_pass_does_not_run_on_training_when_flag_off(self):
        from torch._inductor.fx_passes import post_grad

        called = {"reorder": False}
        orig = post_grad.reorder_for_locality

        def _spy(graph):
            called["reorder"] = True
            return orig(graph)

        with inductor_config.patch({"reorder_for_locality_in_training": False}):
            with mock_patch.object(post_grad, "reorder_for_locality", _spy):
                post_grad.post_grad_passes(self._make_tiny_gm(), is_inference=False)
        self.assertFalse(called["reorder"])

    def test_pass_runs_on_training_when_flag_on(self):
        from torch._inductor.fx_passes import post_grad

        called = {"reorder": False}
        orig = post_grad.reorder_for_locality

        def _spy(graph):
            called["reorder"] = True
            return orig(graph)

        with inductor_config.patch({"reorder_for_locality_in_training": True}):
            with mock_patch.object(post_grad, "reorder_for_locality", _spy):
                post_grad.post_grad_passes(self._make_tiny_gm(), is_inference=False)
        self.assertTrue(called["reorder"])

    def test_inference_path_unchanged(self):
        # The pass should still run on inference graphs regardless of the
        # new flag's value (preserves the existing default-on inference
        # behaviour).
        from torch._inductor.fx_passes import post_grad

        called = {"reorder": False}
        orig = post_grad.reorder_for_locality

        def _spy(graph):
            called["reorder"] = True
            return orig(graph)

        with inductor_config.patch({"reorder_for_locality_in_training": False}):
            with mock_patch.object(post_grad, "reorder_for_locality", _spy):
                post_grad.post_grad_passes(self._make_tiny_gm(), is_inference=True)
        self.assertTrue(called["reorder"])


if __name__ == "__main__":
    run_tests()
