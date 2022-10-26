# Owner(s): ["oncall: distributed"]

import logging
import unittest
from unittest.mock import patch

from torch.distributed.c10d_error_logger import _get_or_create_logger
from torch.testing._internal.common_utils import run_tests

class C10dErrorLoggerTest(unittest.TestCase):

    @patch("torch.distributed.c10d_error_logger._get_logging_handler")
    def test_get_or_create_logger(self, logging_handler_mock):
        logging_handler_mock.return_value = logging.NullHandler(), "NullHandler"
        logger = _get_or_create_logger()
        self.assertIsNotNone(logger)
        self.assertEqual(1, len(logger.handlers))
        self.assertIsInstance(logger.handlers[0], logging.NullHandler)


if __name__ == "__main__":
    run_tests()
