# Owner(s): ["oncall: distributed"]

import copy
import json
import os
import pickle
import random
import re
import signal
import sys
import tempfile
import threading
import time
import warnings
from contextlib import contextmanager
from datetime import datetime, timedelta
from enum import auto, Enum
from itertools import chain, product
from unittest import mock, SkipTest

import torch
import torch.distributed as c10d


from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    retry_on_connect_failures,
    run_tests,
    TestCase,
)
from tools.flight_recorder.fr_trace import match_one_event, MatchState


def create_one_event(collectcive_name, pg_info, input_sizes, output_sizes, state='scheduled', collective_seq_id=0, p2p_seq_id=0):
    return {
        "profiling_name": f"nccl:{collectcive_name}",
        "state": state,
        "process_group": pg_info,
        "input_sizes": input_sizes,
        "output_sizes": output_sizes,
        "collective_seq_id": str(collective_seq_id),
        "p2p_seq_id": str(p2p_seq_id),
    }

class FlightRecorderEventTest(TestCase):
    def test_match_one_event(self):
        e1 = create_one_event('all_reduce', ('0', 'default'), [4, 4], [4, 4], 'scheduled', 1)
        e2 = create_one_event('all_reduce', ('0', 'default'), [4, 4], [4, 4], 'scheduled', 1)
        membership = {'0': {"0", "1"}}
        self.assertEqual(match_one_event(e1, e2, membership), MatchState.FULLY_MATCHED)
