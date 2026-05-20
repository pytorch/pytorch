import contextlib
import os

from torch._dynamo.test_case import (
    run_tests as dynamo_run_tests,
    TestCase as DynamoTestCase,
)
from torch._functorch import config as functorch_config
from torch._inductor import config
from torch._inductor.utils import fresh_cache


def run_tests(needs: str | tuple[str, ...] = ()) -> None:
    dynamo_run_tests(needs)


class TestCase(DynamoTestCase):
    """
    A base TestCase for inductor tests. Enables FX graph caching and isolates
    the cache directory for each test.
    """

    # Share Triton compilation cache across tests by default. Triton
    # compilation is a pure function of the kernel source so reusing
    # compiled kernels is safe and avoids redundant compilation.
    # Subclasses that explicitly delete or inspect the Triton cache
    # directory should set this to False.
    _share_triton_cache = True

    # Cap the number of autotune configs in tests to speed up compilation.
    # Set to None in subclasses that need the full config space.
    _max_mm_configs: int | None = 2
    _max_flex_configs: int | None = 2

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

        if (
            config.test_configs.max_mm_configs is None
            and self._max_mm_configs is not None
        ):
            self._inductor_test_stack.enter_context(
                config.patch({"test_configs.max_mm_configs": self._max_mm_configs})
            )
        if (
            config.test_configs.max_flex_configs is None
            and self._max_flex_configs is not None
        ):
            self._inductor_test_stack.enter_context(
                config.patch({"test_configs.max_flex_configs": self._max_flex_configs})
            )

        if (
            os.environ.get("INDUCTOR_TEST_DISABLE_FRESH_CACHE") != "1"
            and os.environ.get("TORCH_COMPILE_DEBUG") != "1"
        ):
            self._inductor_test_stack.enter_context(
                fresh_cache(share_triton=self._share_triton_cache)
            )

    def tearDown(self) -> None:
        super().tearDown()
        self._inductor_test_stack.close()
