# Owner(s): ["module: inductor"]
import json
import unittest

import torch
import torch._inductor.test_case
import torch._inductor.utils

from torch._inductor import config
from torch.profiler import ProfilerActivity

from torch.testing._internal.common_utils import TemporaryFileName
from torch.testing._internal.inductor_utils import HAS_CUDA

from torch.utils._triton import has_triton

HAS_TRITON = has_triton()


class DynamoProfilerTests(torch._inductor.test_case.TestCase):
    @unittest.skipIf(not HAS_TRITON, "requires cuda & triton")
    def test_inductor_profiling_triton_launch(self):
        # Verify that we get some sort of CPU-side indication of triton kernel launches
        # in the profile traces. Currently, those appear as `cuLaunchKernel`. If this
        # detail changes, the test can be updated or removed.
        @torch.compile
        def fn(x, y):
            return (x + y).sin().cos()

        x, y = (torch.rand((4, 4), device="cuda") for _ in range(2))

        with torch.profiler.profile() as prof:
            fn(x, y)

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                trace_json = json.load(f)

        self.assertTrue("traceEvents" in trace_json)
        events = trace_json["traceEvents"]

        kernel_name = "hipModuleLaunchKernel" if torch.version.hip else "cuLaunchKernel"

        def nameMatchesLaunchKernel(event_name):
            return kernel_name in event_name

        self.assertTrue(
            any(("name" in event and kernel_name == event["name"]) for event in events)
        )

    def _test_profiling_kernel_names(self, fn, args, kernel_name_str: str):
        """
        We expect a record_function event to be added on the CPU side, surrounding
        the launch of each triton kernel.
        """
        fn_opt = torch.compile(fn)

        for _ in range(2):
            fn_opt(*args)

        with torch.profiler.profile(
            activities=[ProfilerActivity.CPU], record_shapes=True
        ) as prof:
            fn_opt(*args)

        # The name of the kernel is expected to match the name of the kernel in debug
        # files etc. The name could change in the future, but it seems reasonable that
        # the name should always contain "triton" and "kernel_name_str" - e.g. if the
        # kernel contains a sin op, it should probably contain "str" in the name.
        # If this changes in the future, feel free to change the assertion here.
        # Debugging tips: you can add prof.export_chrome_trace("test.json") inline in
        # this test, and then view test.json in chrome://tracing to see the trace.
        self.assertTrue(
            any(
                (
                    hasattr(event, "name")
                    and kernel_name_str in event.name
                    and "triton" in event.name
                )
                for event in prof.events()
            )
        )
        return prof.events()

    @unittest.skipIf(not HAS_TRITON, "requires cuda & triton")
    def test_inductor_profiling_kernel_names_pointwise(self):
        def fn(x, y):
            return (x + y).sin().cos()

        args = [torch.rand((4, 4), device="cuda") for _ in range(2)]

        events = self._test_profiling_kernel_names(fn, args, "sin")
        event_found = False
        for event in events:
            if event.name == "triton_poi_fused_add_cos_sin_0":
                event_found = True
                self.assertTrue(event.input_shapes == [[4, 4], [4, 4], [4, 4], []])
        self.assertTrue(event_found)

    @unittest.skipIf(not HAS_TRITON, "requires cuda & triton")
    def test_inductor_profiling_kernel_names_template(self):
        with config.patch(
            {"max_autotune": True, "max_autotune_gemm_backends": "TRITON"}
        ):

            def fn(x, y):
                return x @ y

            args = [torch.rand((4, 4), device="cuda") for _ in range(2)]

            events = self._test_profiling_kernel_names(fn, args, "mm")
            event_found = False
            for event in events:
                if event.name == "triton_tem_fused_mm_0":
                    event_found = True
                    self.assertTrue(event.input_shapes == [[4, 4], [4, 4], [4, 4]])
            self.assertTrue(event_found)

    @unittest.skipIf(not HAS_TRITON, "requires cuda & triton")
    def test_inductor_profiling_kernel_names_foreach(self):
        with config.patch(
            {"max_autotune": True, "max_autotune_gemm_backends": "TRITON"}
        ):

            def fn(x, y):
                return torch._foreach_add(x, y)

            x = [torch.rand((4, 4), device="cuda") for _ in range(3)]
            y = [torch.rand((4, 4), device="cuda") for _ in range(3)]

            args = (x, y)

            events = self._test_profiling_kernel_names(fn, args, "_for_")
            event_found = False
            for event in events:
                if event.name == "triton_for_fused_0":
                    event_found = True
                    self.assertTrue(
                        event.input_shapes
                        == [
                            [4, 4],
                            [4, 4],
                            [4, 4],
                            [4, 4],
                            [4, 4],
                            [4, 4],
                            [4, 4],
                            [4, 4],
                            [4, 4],
                        ]
                    )
            self.assertTrue(event_found)

    @unittest.skipIf(not HAS_TRITON, "requires cuda & triton")
    def test_inductor_profiling_triton_hooks(self):
        from triton.compiler import CompiledKernel

        hooks_called = {"enter": False, "exit": False}

        def launch_enter_hook(lazy_dict):
            hooks_called["enter"] = True

        def launch_exit_hook(lazy_dict):
            hooks_called["exit"] = True

        CompiledKernel.launch_enter_hook = launch_enter_hook
        CompiledKernel.launch_exit_hook = launch_exit_hook

        def fn(x, y):
            return torch._foreach_add(x, y)

        x = [torch.rand((4, 4), device="cuda") for _ in range(3)]
        y = [torch.rand((4, 4), device="cuda") for _ in range(3)]

        args = (x, y)
        fn_opt = torch.compile(fn)
        fn_opt(*args)

        self.assertTrue(hooks_called["enter"])
        self.assertTrue(hooks_called["exit"])


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CUDA:
        run_tests()
