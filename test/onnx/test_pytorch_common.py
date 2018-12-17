from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools
import os
import unittest
import sys
import torch
import torch.autograd.function as function

pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(-1, pytorch_test_dir)

from common_utils import *

torch.set_default_tensor_type('torch.FloatTensor')


def _skipper(condition, reason):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if condition():
                raise unittest.SkipTest(reason)
            return f(*args, **kwargs)
        return wrapper
    return decorator


skipIfNoCuda = _skipper(lambda: not torch.cuda.is_available(),
                        'CUDA is not available')

skipIfTravis = _skipper(lambda: os.getenv('TRAVIS'),
                        'Skip In Travis')

skipIfCI = _skipper(lambda: os.getenv('CI') or os.getenv('TRAVIS') or
                    os.getenv('JENKINS_URL') or os.getenv('INTEGRATED'),
                    'Skip In CI')


def flatten(x):
    return tuple(function._iter_filter(lambda o: isinstance(o, torch.Tensor))(x))
