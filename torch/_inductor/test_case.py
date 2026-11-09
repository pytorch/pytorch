import contextlib
import os
from collections.abc import Iterator

from torch._dynamo.test_case import (
    run_tests as dynamo_run_tests,
    TestCase as DynamoTestCase,
)
from torch._functorch import config as functorch_config
from torch._inductor import config
from torch._inductor.utils import fresh_cache


def run_tests(needs: str | tuple[str, ...] = ()) -> None:
    dynamo_run_tests(needs)


@contextlib.contextmanager
def maybe_fresh_cache() -> Iterator[None]:
    """fresh_cache(), unless INDUCTOR_TEST_DISABLE_FRESH_CACHE=1 or TORCH_COMPILE_DEBUG=1."""
    if (
        os.environ.get("INDUCTOR_TEST_DISABLE_FRESH_CACHE") != "1"
        and os.environ.get("TORCH_COMPILE_DEBUG") != "1"
    ):
        with fresh_cache():
            yield
    else:
        yield


class TestCase(DynamoTestCase):
    """
    A base TestCase for inductor tests. Enables FX graph caching and isolates
    the cache directory for each test.
    """

    def setUp(self) -> None:
        super().setUp()
        self._inductor_test_stack = contextlib.ExitStack()
        # A cleanup rather than tearDown, which does not run when setUp raises
        # (e.g. skipTest). A leaked fresh_cache() otherwise exits whenever the
        # garbage collector finalizes it.
        self.addCleanup(self._inductor_test_stack.close)
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

        self._inductor_test_stack.enter_context(maybe_fresh_cache())
