import contextlib
import importlib
import sys
from unittest.mock import patch

import torch
import torch.testing

from . import config, reset, utils

from torch.testing._internal.common_utils import TestCase as TorchTestCase
from torch.testing._internal.common_utils import IS_WINDOWS
from torch.testing._internal.common_utils import TEST_WITH_CROSSREF
from torch.testing._internal.common_utils import TEST_WITH_TORCHDYNAMO
from torch.testing._internal.common_utils import run_tests


def run_tests(needs=()):

    if (
        TEST_WITH_TORCHDYNAMO
        or IS_WINDOWS
        or TEST_WITH_CROSSREF
        or sys.version_info >= (3, 11)
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
            patch.object(config, "raise_on_backend_error", True)
        )
        cls._exit_stack.enter_context(
            patch.object(config, "raise_on_ctx_manager_usage", True)
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
