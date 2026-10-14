import contextlib
import inspect
import os
from pathlib import Path

from torch._dynamo.test_case import (
    run_tests as dynamo_run_tests,
    TestCase as DynamoTestCase,
)
from torch._functorch import config as functorch_config
from torch._inductor import config
from torch._inductor.utils import fresh_cache


# Test files that run with config.strict_flexible_layout_strides on. Files are
# added as the code they exercise is fixed, until it can be on everywhere.
STRICT_FLEXIBLE_LAYOUT_STRIDES_FILES = {
    "test_flex_attention",
    "test_flex_decoding",
    "test_fused_attention",
    "test_layout_freezing",
    "test_lookup_table",
    "test_max_autotune",
    "test_padding",
    "test_select_algorithm",
    "test_subgraph_choice",
    "test_unbacked_symints",
}


def run_tests(needs: str | tuple[str, ...] = ()) -> None:
    dynamo_run_tests(needs)


class TestCase(DynamoTestCase):
    """
    A base TestCase for inductor tests. Enables FX graph caching and isolates
    the cache directory for each test.
    """

    def setUp(self) -> None:
        super().setUp()
        self._inductor_test_stack = contextlib.ExitStack()
        self._inductor_test_stack.enter_context(
            functorch_config.patch(
                {
                    "enable_autograd_cache": True,
                }
            )
        )

        if (
            "TORCHINDUCTOR_FX_GRAPH_CACHE" not in os.environ
            and "TORCHINDUCTOR_FX_GRAPH_CACHE_DEFAULT" not in os.environ
        ):
            self._inductor_test_stack.enter_context(
                config.patch({"fx_graph_cache": True})
            )

        test_file = Path(inspect.getfile(type(self))).stem
        if test_file in STRICT_FLEXIBLE_LAYOUT_STRIDES_FILES:
            self._inductor_test_stack.enter_context(
                config.patch({"strict_flexible_layout_strides": True})
            )

        if (
            os.environ.get("INDUCTOR_TEST_DISABLE_FRESH_CACHE") != "1"
            and os.environ.get("TORCH_COMPILE_DEBUG") != "1"
        ):
            self._inductor_test_stack.enter_context(fresh_cache())

    def tearDown(self) -> None:
        super().tearDown()
        self._inductor_test_stack.close()
