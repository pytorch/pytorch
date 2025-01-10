import contextlib
import importlib
import logging
from typing import Tuple, Union

import torch
import torch.testing
from torch._logging._internal import trace_log
from torch.testing._internal.common_utils import (  # type: ignore[attr-defined]
    IS_WINDOWS,
    TEST_WITH_CROSSREF,
    TEST_WITH_TORCHDYNAMO,
    TestCase as TorchTestCase,
)

from . import config, reset, utils


log = logging.getLogger(__name__)


def run_tests(needs: Union[str, Tuple[str, ...]] = ()) -> None:
    from torch.testing._internal.common_utils import run_tests

    if TEST_WITH_TORCHDYNAMO or IS_WINDOWS or TEST_WITH_CROSSREF:
        return  # skip testing

    if isinstance(needs, str):
        needs = (needs,)
    for need in needs:
        if need == "cuda":
            if not torch.cuda.is_available():
                return
        else:
            try:
                importlib.import_module(need)
            except ImportError:
                return
    run_tests()


class CompileCounterInt(int):
    def __add__(self, other):
        return CompileCounterInt(super().__add__(other))


class TestCase(TorchTestCase):
    _exit_stack: contextlib.ExitStack

    @classmethod
    def tearDownClass(cls) -> None:
        cls._exit_stack.close()
        super().tearDownClass()

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._exit_stack = contextlib.ExitStack()  # type: ignore[attr-defined]
        cls._exit_stack.enter_context(  # type: ignore[attr-defined]
            config.patch(
                raise_on_ctx_manager_usage=True,
                suppress_errors=False,
                log_compilation_metrics=False,
            ),
        )

    def setUp(self) -> None:
        self._prior_is_grad_enabled = torch.is_grad_enabled()
        super().setUp()
        reset()
        utils.counters.clear()
        self.handler = logging.NullHandler()
        trace_log.addHandler(self.handler)

    def tearDown(self) -> None:
        trace_log.removeHandler(self.handler)
        for k, v in utils.counters.items():
            print(k, v.most_common())
        reset()
        utils.counters.clear()
        super().tearDown()
        if self._prior_is_grad_enabled is not torch.is_grad_enabled():
            log.warning("Running test changed grad mode")
            torch.set_grad_enabled(self._prior_is_grad_enabled)

    def assertEqual(self, x, y, *args, **kwargs):
        if (
            config.debug_disable_compile_counter
            and isinstance(x, CompileCounterInt)
            or isinstance(y, CompileCounterInt)
        ):
            return
        return super().assertEqual(x, y, *args, **kwargs)

    # assertExpectedInline might also need to be disabled for wrapped nested
    # graph break tests
