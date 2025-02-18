"""Testing utilities for Dynamo, providing a specialized TestCase class and test running functionality.

This module extends PyTorch's testing framework with Dynamo-specific testing capabilities.
It includes:
- A custom TestCase class that handles Dynamo-specific setup/teardown
- Test running utilities with dependency checking
- Automatic reset of Dynamo state between tests
- Proper handling of gradient mode state
"""

import contextlib
import importlib
import logging
from typing import Union

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


def run_tests(needs: Union[str, tuple[str, ...]] = ()) -> None:
    from torch.testing._internal.common_utils import run_tests

    if TEST_WITH_TORCHDYNAMO or TEST_WITH_CROSSREF:
        return  # skip testing

    if not torch.xpu.is_available() and IS_WINDOWS:
        return

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
