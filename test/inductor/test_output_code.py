# Owner(s): ["module: inductor"]

import re
from unittest import mock

from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.cudagraph_utils import CudagraphCachedInfo
from torch._inductor.output_code import (
    CompiledFxGraph,
    cudagraph_partition_post_compile,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import BoxedBool
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


class TestCudagraphPartitionPostCompile(TestCase):
    @parametrize("skip_reason", ("failed", "missing", "empty"))
    @parametrize("cudagraph_or_error", (False, True))
    @parametrize("device_type", ("cpu", "cuda"))
    def test_skip(self, skip_reason, cudagraph_or_error, device_type):
        compiled_graph = mock.Mock(spec=CompiledFxGraph)
        fail_reasons = ["non-Tensor inputs"] if skip_reason == "failed" else []
        compiled_graph.cudagraph_info = CudagraphCachedInfo((), [], (), fail_reasons)
        compiled_graph.partition_maps = None if skip_reason == "missing" else []
        compiled_graph.fx_kwargs = {"is_backward": False}
        compiled_graph.current_callable = lambda args: args
        compiled_graph.device_types = {device_type}
        message = {
            "failed": "skipping cudagraphs due to ['non-Tensor inputs']",
            "missing": "skipping cudagraphs as compiled_graph.partition_maps is None",
            "empty": "skipping cudagraphs as len(compiled_graph.partition_maps) == 0",
        }[skip_reason]
        cudagraphs = BoxedBool(True)
        skips_before = counters["inductor"]["cudagraph_skips"]
        with (
            config.patch("triton.cudagraph_or_error", cudagraph_or_error),
            mock.patch(
                "torch._inductor.compiler_bisector.CompilerBisector.disable_subsystem",
                return_value=False,
            ),
        ):
            if cudagraph_or_error and device_type == "cuda":
                with self.assertRaisesRegex(RuntimeError, re.escape(message)):
                    cudagraph_partition_post_compile(
                        [], compiled_graph, cudagraphs, {}, None
                    )
            else:
                cudagraph_partition_post_compile(
                    [], compiled_graph, cudagraphs, {}, None
                )

        self.assertFalse(cudagraphs)
        self.assertEqual(
            counters["inductor"]["cudagraph_skips"],
            skips_before + (device_type == "cuda"),
        )


instantiate_parametrized_tests(TestCudagraphPartitionPostCompile)

if __name__ == "__main__":
    run_tests()
