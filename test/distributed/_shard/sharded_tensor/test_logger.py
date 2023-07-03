# Owner(s): ["oncall: distributed"]

import logging

from torch.distributed._shard.sharded_tensor.logger import _get_or_create_logger
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)


class ShardingSpecLoggerTest(TestCase):
    def test_get_or_create_logger(self):
        logger = _get_or_create_logger()
        self.assertIsNotNone(logger)
        self.assertEqual(1, len(logger.handlers))
        self.assertIsInstance(logger.handlers[0], logging.NullHandler)


if __name__ == "__main__":
    run_tests()
