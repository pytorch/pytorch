# Owner(s): ["module: inductor"]
import os
import unittest
from unittest.mock import patch

import torch
import torch._inductor.config as inductor_config
from torch._dynamo.device_interface import get_interface_for_device
from torch._inductor.autoheuristic.autoheuristic import AutoHeuristic, LocalFeedback
from torch._inductor.autoheuristic.autoheuristic_utils import AHContext
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import get_gpu_shared_memory
from torch.testing._internal.common_utils import skipIfXpu
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU, IS_A100, IS_H100


class AutoHeuristicTest(TestCase):
    def count_lines_in_file(self, file_path):
        with open(file_path) as file:
            line_count = sum(1 for line in file)
        return line_count

    def run_mm(self):
        def f(a, b):
            return torch.mm(a, b)

        cf = torch.compile(f)
        a = torch.randn(2047, 2048, device=GPU_TYPE, dtype=torch.float16)
        b = torch.randn(2048, 2048, device=GPU_TYPE, dtype=torch.float16)
        cf(a, b)

    def get_path_to_autoheuristic_log(self, name):
        device_name = AutoHeuristic.get_device_identifier()
        path = cache_dir() + "/autoheuristic/" + device_name + "/" + name + ".txt"
        return path

    @skipIfXpu(msg="AutoHeuristic doesn't currently work on the XPU stack")
    def test_autoheuristic_pad_mm_default(self):
        # this test ensures that data is not collected for pad_mm when autoheuristic config is set to its default value
        self.run_mm()
        self.assertFalse(os.path.exists(self.get_path_to_autoheuristic_log("pad_mm")))

    @skipIfXpu(msg="AutoHeuristic doesn't currently work on the XPU stack")
    @inductor_config.patch("autoheuristic_collect.pad_mm", False)
    def test_autoheuristic_pad_mm_off(self):
        # this test ensures that data is not collected for pad_mm when autoheuristic_collect does not contain "pad_mm"
        self.run_mm()
        self.assertFalse(os.path.exists(self.get_path_to_autoheuristic_log("pad_mm")))

    def assert_autoheuristic_collected_data(self):
        self.run_mm()
        AutoHeuristic.get_device_identifier()
        path = self.get_path_to_autoheuristic_log("pad_mm")
        self.assertTrue(os.path.exists(path))
        num_lines = self.count_lines_in_file(path)

        # 1 line for metadata, 1 line for header, 1 line per choice (orig, padded)
        self.assertEqual(num_lines, 4)

    @skipIfXpu(msg="AutoHeuristic doesn't currently work on the XPU stack")
    @inductor_config.patch("autoheuristic_collect.pad_mm", True)
    def test_autoheuristic_pad_mm_collect_data(self):
        # this test ensures that data is collected for pad_mm when autoheuristic_collect.pad_mm=True
        self.assert_autoheuristic_collected_data()

    @skipIfXpu(msg="AutoHeuristic doesn't currently work on the XPU stack")
    @inductor_config.patch("autoheuristic_collect.pad_mm", True)
    def test_autoheuristic_pad_mm_collect_data2(self):
        # this test ensures that data is collected for pad_mm when autoheuristic_collect.pad_mm=True
        self.assert_autoheuristic_collected_data()

    @skipIfXpu(msg="AutoHeuristic doesn't currently work on the XPU stack")
    @patch.dict(os.environ, {"TORCHINDUCTOR_AUTOHEURISTIC_COLLECT": "test"})
    def test_autoheuristic(self):
        # test basic functionality of autoheuristic
        def fallback():
            return "fallback"

        choices = ["a", "b", "c"]

        def feedback_fn(choice):
            if choice == "a":
                return 1
            elif choice == "b":
                return 2
            elif choice == "c":
                return 3
            else:
                raise RuntimeError("unexpected choice")

        feedback = LocalFeedback(feedback_fn)
        context = AHContext()
        context.add_feature("fa", 5)
        name = "test"
        autoheuristic = AutoHeuristic(fallback, choices, feedback, context, name)

        # when autoheuristic is configured to only collect data, we always return fallback
        self.assertEqual(autoheuristic.get_choice(), "fallback")
        self.assertEqual(autoheuristic.get_collected_feedback("a"), 1)
        self.assertEqual(autoheuristic.get_collected_feedback("b"), 2)
        self.assertEqual(autoheuristic.get_collected_feedback("c"), 3)

        path = self.get_path_to_autoheuristic_log(name)
        self.assertTrue(os.path.exists(path))
        num_lines = self.count_lines_in_file(path)
        self.assertEqual(num_lines, 5)

        shared_memory = get_gpu_shared_memory()
        compute_cap = get_interface_for_device(GPU_TYPE).get_compute_capability()
        # Convert single int compute capability (e.g., 90) to tuple (e.g., (9, 0))
        fst, snd = compute_cap // 10, compute_cap % 10

        with open(path) as file:
            lines = file.readlines()
            self.assertTrue('"numerical_features": ["fa"]' in lines[0])
            self.assertTrue('"categorical_features": []' in lines[0])
            self.assertTrue(f'"shared_memory": {shared_memory}' in lines[0])
            self.assertTrue(f'"device_capa": [{fst}, {snd}]' in lines[0])
            self.assertTrue('"name": "test"' in lines[0])
            self.assertEqual("fa,choice,feedback", lines[1].rstrip())
            self.assertEqual("5,a,1", lines[2].rstrip())
            self.assertEqual("5,b,2", lines[3].rstrip())
            self.assertEqual("5,c,3", lines[4].rstrip())

    @skipIfXpu(msg="AutoHeuristic doesn't currently work on the XPU stack")
    @unittest.skipIf(not IS_A100, "heuristic only run on A100")
    @inductor_config.patch("autoheuristic_use.pad_mm", True)
    def test_autoheuristic_a100(self):
        # Make sure heuristic does not break anything
        # TODO (AlnisM): Find a way to check whether heuristic is used
        self.run_mm()

    @skipIfXpu(msg="AutoHeuristic doesn't currently work on the XPU stack")
    @unittest.skipIf(not IS_H100, "heuristic only run on H100")
    @inductor_config.patch("autoheuristic_use.pad_mm", True)
    def test_autoheuristic_h100(self):
        # Make sure heuristic does not break anything
        # TODO (AlnisM): Find a way to check whether heuristic is used
        self.run_mm()

    def run_mixed_mm(self):
        def fn(a, b):
            return torch.mm(a, b.to(a.dtype))

        a = torch.randn(8, 1024, device=GPU_TYPE, dtype=torch.float16)
        b = torch.randint(
            -128, 127, (1024, 1024), dtype=torch.int8, device=GPU_TYPE
        ).t()
        torch.compile(fn, mode="max-autotune-no-cudagraphs")(a, b)

    # have to set autoheuristic_use="" because if autoheuristic_use="mixed_mm",
    # autoheuristic creates a precompile key, puts it into the registry, and then
    # a choice made by the heuristic might be added to the list of choices
    # and if select_algorithm now creates a new precompile key, it will be
    # different from the precompile key created by autoheuristic
    @skipIfXpu(msg="AutoHeuristic doesn't currently work on the XPU stack")
    @unittest.skip(
        "mixed_mm autoheuristic collection is broken after mixed_mm special casing deletion (PR #147151)"
    )
    @inductor_config.patch("autoheuristic_collect.mixed_mm", True)
    @inductor_config.patch("autoheuristic_use.mixed_mm", False)
    @inductor_config.patch(fx_graph_cache=False)
    @inductor_config.patch(fx_graph_remote_cache=False)
    def test_global_feedback(self):
        self.run_mixed_mm()
        path = self.get_path_to_autoheuristic_log("mixed_mm")
        self.assertTrue(os.path.exists(path))
        num_lines = self.count_lines_in_file(path)

        # 1 line for metadata, 1 line for header
        # 1 line for fallback + at least 1 config
        self.assertTrue(num_lines > 4)

    @skipIfXpu(msg="AutoHeuristic doesn't currently work on the XPU stack")
    @inductor_config.patch("autoheuristic_use.mixed_mm", True)
    @unittest.skipIf(not IS_A100, "heuristic only run on A100")
    def test_mixed_mm_a100(self):
        self.run_mixed_mm()
        # TODO (AlnisM): Find a way to check whether heuristic is used

    @skipIfXpu(msg="AutoHeuristic doesn't currently work on the XPU stack")
    @unittest.skipIf(not IS_H100 and not IS_A100, "heuristic only run on H100")
    @inductor_config.patch(deterministic=True)
    @inductor_config.patch("autoheuristic_use.pad_mm", True)
    def test_pad_mm_autoheuristic_deterministic_mode(self):
        """Test that pad_mm AutoHeuristics works in deterministic mode."""
        from torch._dynamo.utils import counters

        counters.clear()

        def f(a, b):
            return torch.mm(a, b)

        cf = torch.compile(f)
        # Use shapes that would normally trigger padding but aren't well-aligned
        a = torch.randn(2047, 2048, device=GPU_TYPE, dtype=torch.float16)
        b = torch.randn(2048, 2049, device=GPU_TYPE, dtype=torch.float16)

        # Run the compiled function
        result = cf(a, b)

        # Verify correctness with tolerance for potential padding differences
        expected = torch.mm(a, b)
        torch.testing.assert_close(result, expected, atol=0.1, rtol=0.05)

        # In deterministic mode with AutoHeuristics enabled,
        # we should not do any benchmarking (pad_mm_bench should stay 0)
        # because AutoHeuristics makes the decision without benchmarking
        self.assertEqual(counters["inductor"]["pad_mm_bench"], 0)


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
