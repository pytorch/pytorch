## @package test_util
# Module caffe2.python.test_util




import numpy as np
from caffe2.python import core, workspace

import os
import pathlib
import shutil
import tempfile
import unittest
from typing import Any, Callable, Tuple, Type
from types import TracebackType


def rand_array(*dims):
    # np.random.rand() returns float instead of 0-dim array, that's why need to
    # do some tricks
    return np.array(np.random.rand(*dims) - 0.5).astype(np.float32)


def randBlob(name, type, *dims, **kwargs):
    offset = kwargs['offset'] if 'offset' in kwargs else 0.0
    workspace.FeedBlob(name, np.random.rand(*dims).astype(type) + offset)


def randBlobFloat32(name, *dims, **kwargs):
    randBlob(name, np.float32, *dims, **kwargs)


def randBlobsFloat32(names, *dims, **kwargs):
    for name in names:
        randBlobFloat32(name, *dims, **kwargs)


def numOps(net):
    return len(net.Proto().op)


def str_compare(a, b, encoding="utf8"):
    if isinstance(a, bytes):
        a = a.decode(encoding)
    if isinstance(b, bytes):
        b = b.decode(encoding)
    return a == b


def get_default_test_flags():
    return [
        'caffe2',
        '--caffe2_log_level=0',
        '--caffe2_cpu_allocator_do_zero_fill=0',
        '--caffe2_cpu_allocator_do_junk_fill=1',
    ]


class TestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        workspace.GlobalInit(get_default_test_flags())
        # clear the default engines settings to separate out its
        # affect from the ops tests
        core.SetEnginePref({}, {})

    def setUp(self):
        self.ws = workspace.C.Workspace()
        workspace.ResetWorkspace()

    def tearDown(self):
        workspace.ResetWorkspace()

    def make_tempdir(self) -> pathlib.Path:
        tmp_folder = pathlib.Path(tempfile.mkdtemp(prefix="caffe2_test."))
        self.addCleanup(self._remove_tempdir, tmp_folder)
        return tmp_folder

    def _remove_tempdir(self, path: pathlib.Path) -> None:
        def _onerror(
            fn: Callable[..., Any],
            path: str,
            exc_info: Tuple[Type[BaseException], BaseException, TracebackType],
        ) -> None:
            # Ignore FileNotFoundError, but re-raise anything else
            if not isinstance(exc_info[1], FileNotFoundError):
                raise exc_info[1].with_traceback(exc_info[2])

        shutil.rmtree(str(path), onerror=_onerror)
