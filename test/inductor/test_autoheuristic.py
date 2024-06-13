# Owner(s): ["module: inductor"]
import os

import torch

import torch._inductor.config as inductor_config

from torch._inductor.autoheuristic import AHContext, AutoHeuristic, LocalFeedback
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.inductor_utils import HAS_CUDA


class AutoHeuristicTest(TestCase):
    def count_lines_in_file(self, file_path):
        with open(file_path) as file:
            line_count = sum(1 for line in file)
        return line_count

    def run_mm(self):
        def f(a, b):
            return torch.mm(a, b)

        cf = torch.compile(f)
        a = torch.randn(2047, 2048, device="cuda", dtype=torch.float16)
        b = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
        cf(a, b)

    def get_path_to_autoheuristic_log(self, name):
        device_name = AutoHeuristic.get_device_identifier()
        path = cache_dir() + "/autoheuristic/" + device_name + "/" + name + ".txt"
        return path

    def test_autoheuristic_pad_mm_default(self):
        # this test ensure that data is not collected for pad_mm when autoheuristic_mode is set to its default value ("OFF")
        self.run_mm()
        self.assertFalse(os.path.exists(self.get_path_to_autoheuristic_log("pad_mm")))

    @inductor_config.patch(autoheuristic_mode="OFF")
    def test_autoheuristic_pad_mm_off(self):
        # this test ensure that data is not collected for pad_mm when autoheuristic_mode="OFF"
        self.run_mm()
        self.assertFalse(os.path.exists(self.get_path_to_autoheuristic_log("pad_mm")))

    @inductor_config.patch(autoheuristic_mode="COLLECT_DATA")
    def test_autoheuristic_pad_mm_collect_data(self):
        # this test ensure that data is collected for pad_mm when autoheuristic_mode="COLLECT_DATA"
        self.run_mm()
        device_name = AutoHeuristic.get_device_identifier()
        path = self.get_path_to_autoheuristic_log("pad_mm")
        self.assertTrue(os.path.exists(path))
        num_lines = self.count_lines_in_file(path)

        # 1 line for metadata, 1 line for header, 1 line per choice (orig, padded)
        self.assertEqual(num_lines, 4)

    @inductor_config.patch(autoheuristic_mode="COLLECT_DATA")
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

        # when autoheuristic_mode is COLLECT_DATA, we always return fallback
        self.assertEqual(autoheuristic.get_choice(), "fallback")
        self.assertEqual(autoheuristic.get_collected_feedback("a"), 1)
        self.assertEqual(autoheuristic.get_collected_feedback("b"), 2)
        self.assertEqual(autoheuristic.get_collected_feedback("c"), 3)

        path = self.get_path_to_autoheuristic_log(name)
        self.assertTrue(os.path.exists(path))
        num_lines = self.count_lines_in_file(path)
        self.assertEqual(num_lines, 5)

        with open(path) as file:
            lines = file.readlines()
            self.assertTrue('"numerical_features": ["fa"]' in lines[0])
            self.assertTrue('"categorical_features": []' in lines[0])
            self.assertTrue('"choices": ["a", "b", "c"]' in lines[0])
            self.assertEqual("fa,choice,feedback", lines[1].rstrip())
            self.assertEqual("5,a,1", lines[2].rstrip())
            self.assertEqual("5,b,2", lines[3].rstrip())
            self.assertEqual("5,c,3", lines[4].rstrip())


if __name__ == "__main__":
    if HAS_CUDA:
        run_tests()
