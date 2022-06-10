# -*- coding: utf-8 -*-
# Owner(s): ["module: unknown"]

import logging

from torch.testing._internal.common_utils import TestCase

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


class TestBaseDataSparsifier(TestCase):
    def test_constructor(self):
        pass  # Nothing to test so far
