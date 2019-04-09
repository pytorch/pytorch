from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools  # noqa
import os
import unittest  # noqa
import sys  # noqa
import torch  # noqa
import torch.autograd.function as function  # noqa

pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(-1, pytorch_test_dir)

from common_utils import *  # noqa
from common_nn import module_tests, new_module_tests, criterion_tests  # noqa
from common_methods_invocations import method_tests as autograd_method_tests  # noqa
from common_methods_invocations import create_input, unpack_variables  # noqa
from common_methods_invocations import exclude_tensor_method, non_differentiable  # noqa
from common_methods_invocations import EXCLUDE_GRADCHECK, EXCLUDE_FUNCTIONAL  # noqa
from copy import deepcopy  # noqa
import random  # noqa
from typing import List, Dict, Optional, Tuple  # noqa
