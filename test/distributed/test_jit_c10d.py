from __future__ import absolute_import, division, print_function, unicode_literals


import unittest
import traceback
from sys import platform
import torch
import torch.distributed as c10d

from torch.testing._internal.common_distributed import MultiProcessTestCase, \
    requires_gloo
from torch.testing._internal.common_utils import TEST_WITH_TSAN

from torch.testing._internal.jit_utils import JitTestCase, _inline_everything


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
        @torch.jit.script
        def test_create_store(filename: str, world_size: int) -> torch.classes.dist_c10d.FileStore:
            store = torch.classes.dist_c10d.FileStore(filename, world_size)
            return store

        store = test_create_store(self.file_name, self.world_size)


        @torch.jit.script
        def test_store_script_pass(store: torch.classes.dist_c10d.Store) -> torch.classes.dist_c10d.Store:
            return store

        print(test_store_script_pass.graph)
        test_store_script_pass(store)

        # pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        # try:
        #     test_process_group_script(pg)
        # except Exception as e:
        #     traceback.print_exc()



class SingleJitTest(JitTestCase):
    def test_filestore(self):
        @torch.jit.script
        def test_process_group_script() -> torch.classes.dist_c10d.FileStore:
            store = torch.classes.dist_c10d.FileStore("test", 2)
            return store

        print(test_process_group_script.graph)

        test_process_group_script()
