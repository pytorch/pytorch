import contextlib
import importlib
import sys

import torch
import torch.testing
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    TEST_WITH_CROSSREF,
    TEST_WITH_TORCHDYNAMO,
    TestCase as TorchTestCase,
)

from . import config, reset, utils


def run_tests(needs=()):
    from torch.testing._internal.common_utils import run_tests

    if (
        TEST_WITH_TORCHDYNAMO
        or IS_WINDOWS
        or TEST_WITH_CROSSREF
        or sys.version_info >= (3, 12)
    ):
        return  # skip testing

    if isinstance(needs, str):
        needs = (needs,)
    for need in needs:
        if need == "cuda" and not torch.cuda.is_available():
            return
        else:
            try:
                importlib.import_module(need)
            except ImportError:
                return
    run_tests()


class TestCase(TorchTestCase):
    @classmethod
    def tearDownClass(cls):
        cls._exit_stack.close()
        super().tearDownClass()

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._exit_stack = contextlib.ExitStack()
        cls._exit_stack.enter_context(
            config.patch(raise_on_ctx_manager_usage=True, suppress_errors=False),
        )

    def setUp(self):
        super().setUp()
        reset()
        utils.counters.clear()

    def tearDown(self):
        for k, v in utils.counters.items():
            print(k, v.most_common())
        reset()
        utils.counters.clear()
        super().tearDown()
