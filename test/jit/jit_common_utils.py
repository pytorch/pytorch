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
from common_nn import module_tests, new_module_tests, criterion_tests
from common_methods_invocations import method_tests as autograd_method_tests
from common_methods_invocations import create_input, unpack_variables, \
    exclude_tensor_method, non_differentiable, EXCLUDE_GRADCHECK, EXCLUDE_FUNCTIONAL
from copy import deepcopy
import random
from typing import List, Dict, Optional, Tuple

from test_jit import JitTestCase, enable_cpu_fuser, RUN_CUDA, RUN_CUDA_HALF, RUN_CUDA_MULTI_GPU, \
    backward_graph
