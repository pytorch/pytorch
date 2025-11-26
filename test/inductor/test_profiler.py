# Owner(s): ["module: inductor"]
import json
import os
import tempfile
import unittest
from collections.abc import Callable
from typing import Optional

import torch
import torch._inductor.test_case
import torch._inductor.utils
from torch import _dynamo as torchdynamo
from torch._inductor import config
from torch.profiler import ProfilerActivity
from torch.testing._internal.common_utils import TemporaryFileName
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON, IS_BIG_GPU
from torch.torch_version import TorchVersion
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

        valid_names = {
            "hipModuleLaunchKernel",
            "cuLaunchKernel",
            "triton_poi_fused_add_cos_sin_0",
        }
        self.assertTrue(any((event.get("name") in valid_names) for event in events))

    def _test_profiling_kernel_names(
        self, fn, args, kernel_name_str: str, check_fn: Optional[Callable] = None
    ):
        """
        We expect a record_function event to be added on the CPU side, surrounding
        the launch of each triton kernel.
        """
        fn_opt = torch.compile(fn)

        for _ in range(2):
            fn_opt(*args)

        if check_fn is not None:
            check_fn()

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
                # Note: depending on the triton version, we might get 4 or 5 args
                # (including / not including the constexpr args). The last two are
                # both empty args, so we just truncate the event.input_shapes to the
                # first 4.
                self.assertEqual(event.input_shapes[:4], [[4, 4], [4, 4], [4, 4], []])
        self.assertTrue(event_found)

    @unittest.skipIf(
        not IS_BIG_GPU, "Skipping triton backend only since not big GPU (not enough SM)"
    )
    def test_inductor_profiling_kernel_names_template(self):
        with config.patch(
            {"max_autotune": True, "max_autotune_gemm_backends": "TRITON"}
        ):

            def fn(x, y):
                return x @ y

            args = [torch.rand((4, 4), device="cuda") for _ in range(2)]

            def check_fn():
                # test_profiling_kernel_names will check this before asserting mm is in the trace.
                # reason: sometimes testing runs on machines with not enough SMs, and autotuning is skipped.
                if (
                    torch._dynamo.utils.counters["inductor"][
                        "select_algorithm_autotune"
                    ]
                    == 0
                ):
                    raise unittest.SkipTest(
                        "select_algorithm didn't run, we probably won't get profiling data. GPU might not have enough SMs."
                    )

            events = self._test_profiling_kernel_names(fn, args, "mm", check_fn)

            event_found = False
            for event in events:
                if event.name == "triton_tem_fused_mm_0":
                    event_found = True
                    self.assertEqual(event.input_shapes[:3], [[4, 4], [4, 4], [4, 4]])
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
    @config.patch(
        "compile_threads", 1
    )  # This test monkey patches global variables, which workers don't see
    def test_inductor_profiling_triton_hooks(self):
        from triton.compiler import CompiledKernel  # @manual

        from torch._inductor.runtime.triton_compat import knobs

        hooks_called = {"enter": False, "exit": False}

        def launch_enter_hook(lazy_dict):
            hooks_called["enter"] = True

        def launch_exit_hook(lazy_dict):
            hooks_called["exit"] = True

        if knobs:
            knobs.runtime.launch_enter_hook = launch_enter_hook
            knobs.runtime.launch_exit_hook = launch_exit_hook
        else:
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

    @unittest.skipIf(not HAS_TRITON, "requires cuda & triton")
    def test_pt2_triton_attributes(self):
        from torch._inductor.codecache import code_hash

        device = "cuda"
        debug = False  # set to True to get output file

        @torchdynamo.optimize("inductor")
        def fn(a, b, c):
            x = torch.nn.functional.linear(a, b)
            x = x + c
            return x.cos()

        a, b, c = (torch.randn(4, 4, requires_grad=True).to(device) for _ in range(3))

        inputs = [a, b, c]
        with config.patch(compile_threads=1):
            fn(*inputs)

        fp = tempfile.NamedTemporaryFile("w+t", suffix=".json", delete=not debug)
        fp.close()

        with torch.profiler.profile(
            activities=torch.profiler.supported_activities(),
            record_shapes=True,
            schedule=torch.profiler.schedule(
                skip_first=3, wait=1, warmup=1, active=2, repeat=1
            ),
        ) as prof:
            for _ in range(10):
                fn(*inputs)
                prof.step()

        prof.export_chrome_trace(fp.name)
        print(f"Trace written to {fp.name}, set debug=True to retain file.")

        triton_events = []
        with open(fp.name) as f:
            trace_json = json.load(f)
            triton_events = [
                event
                for event in trace_json["traceEvents"]
                if "kernel_backend" in event.get("args", {}).keys()
            ]

        print(triton_events)
        self.assertEqual(len(triton_events), 2)

        def get_hash(kernel_file: str) -> str:
            with open(kernel_file) as f:
                kernel_src = f.read()
            return code_hash(kernel_src.strip())

        def check_triton_event(e) -> None:
            args = e.get("args", {})
            self.assertNotEqual(args, {}, msg=f"event = {e}")

            self.assertEqual(args["kernel_backend"], "triton", msg=f"event = {e}")

            self.assertTrue("stream" in args, msg=f"event = {e}")
            self.assertTrue("kernel_file" in args, msg=f"event = {e}")
            kernel_file = args["kernel_file"]
            self.assertTrue(os.path.isfile(kernel_file), msg=f"event = {e}")

            self.assertTrue("kernel_hash" in args, msg=f"event = {e}")
            self.assertEqual(
                args["kernel_hash"], get_hash(kernel_file), msg=f"event = {e}"
            )

            self.assertTrue("kernel_kwargs" in args, msg=f"event = {e}")
            self.assertTrue(
                args["kernel_kwargs"].startswith("XBLOCK="), msg=f"event = {e}"
            )

        for e in triton_events:
            check_triton_event(e)

    @unittest.skipIf(not HAS_TRITON, "requires cuda & triton")
    def test_cupti_lazy_reinit(self):
        x, y = (torch.randn(4, 4, device="cuda") for _ in range(2))

        def fn(x, y):
            return (x + y).sin()

        fn_c = torch.compile(fn, mode="reduce-overhead")

        with torch.profiler.profile():
            fn_c(x, y)

        if TorchVersion(torch.version.cuda) >= "12.6":
            self.assertEqual("0", os.environ.get("DISABLE_CUPTI_LAZY_REINIT", "0"))
        else:
            self.assertEqual("1", os.environ.get("DISABLE_CUPTI_LAZY_REINIT", "0"))


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CUDA_AND_TRITON:
        run_tests()
