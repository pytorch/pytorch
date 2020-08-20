from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import math
import os
import random
import signal
import sys
import tempfile
import threading
import time
import traceback
import unittest
from datetime import timedelta
from sys import platform
from contextlib import contextmanager

from itertools import groupby, product
from functools import reduce
import operator

import torch
import torch.testing._internal.common_utils as common
from torch import nn
import torch.nn.functional as F
import torch.distributed as c10d
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from torch.testing._internal.common_distributed import MultiProcessTestCase, \
    requires_gloo, requires_nccl, requires_nccl_version, \
    skip_if_not_multigpu, skip_if_lt_x_gpu, get_timeout, skip_if_rocm, \
    simple_sparse_reduce_tests

from torch.testing._internal.common_utils import TestCase, load_tests, run_tests, \
    retry_on_connect_failures, ADDRESS_IN_USE, CONNECT_TIMEOUT, TEST_WITH_TSAN


if not c10d.is_available():
    print('c10d not available, skipping tests', file=sys.stderr)
    sys.exit(0)


if platform == 'darwin':
    LOOPBACK = 'lo0'
else:
    LOOPBACK = 'lo'


@requires_gloo()
@unittest.skipIf(TEST_WITH_TSAN, "TSAN is not fork-safe since we're forking in a multi-threaded environment")
class ProcessGroupGlooJitTest(MultiProcessTestCase):
    def setUp(self):
        super(ProcessGroupGlooJitTest, self).setUp()
        self._fork_processes()

    def opts(self, threads=2):
        opts = c10d.ProcessGroupGloo.Options()
        opts.devices = [c10d.ProcessGroupGloo.create_device(interface=LOOPBACK)]
        opts.timeout = 5.0
        opts.threads = threads
        return opts

    def test_jit_process_group(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        @torch.jit.script
        def test_process_group_script(pg: torch.classes.dist_c10d.ProcessGroup) -> torch.classes.dist_c10d.ProcessGroup:
            return pg

        print(test_process_group_script.graph)
        try:
            test_process_group_script(pg)
        except Exception as e:
            traceback.print_exc()

