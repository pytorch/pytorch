# mypy: ignore-errors

import sys
from io import StringIO

import torch
import torch.cuda

from torch.testing._internal.common_utils import TestCase


RUN_CUDA = torch.cuda.is_available()


class JitTestCase(TestCase):
    _do_cuda_memory_leak_check = True

    class capture_stderr(list):
        """
        Replace sys.stderr with a temporary StringIO
        """
        def __enter__(self):
            self.sys_stderr = sys.stderr
            self.stringio = StringIO()
            sys.stderr = self.stringio
            return self

        def __exit__(self, *args):
            self.append(str(self.stringio.getvalue()))
            del self.stringio
            sys.stderr = self.sys_stderr


def get_forward(c):
    return c._get_method('forward')

def get_forward_graph(c):
    return c._get_method('forward').graph
