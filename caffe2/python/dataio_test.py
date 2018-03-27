from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python.dataio import Reader, ReaderWithLimit, ReaderWithTimeLimit
from caffe2.python.dataset import Dataset
from caffe2.python.pipeline import pipe
from caffe2.python.schema import Struct, NewRecord, FeedRecord
from caffe2.python.session import LocalSession
from caffe2.python.task import TaskGroup, final_output, WorkspaceType
from caffe2.python.test_util import TestCase
from caffe2.python.cached_reader import CachedReader
from caffe2.python import core, workspace
from caffe2.python.net_builder import ops

import numpy as np
import os
import shutil
import tempfile
import time


def init_dataset(ws, size=100):
    src_init = core.Net('src_init')
    with core.NameScope('src'):
        src_values = Struct(('label', np.array(range(size))))
        src_blobs = NewRecord(src_init, src_values)
        src_ds = Dataset(src_blobs)
        FeedRecord(src_blobs, src_values, ws)
    ws.run(src_init)
    return src_ds


def read_all_data(ws, reader, session):
    dst_init = core.Net('dst_init')
    with core.NameScope('dst'):
        dst_ds = Dataset(reader.schema().clone_schema())
        dst_ds.init_empty(dst_init)
    session.run(dst_init)

    with TaskGroup(workspace_type=WorkspaceType.GLOBAL) as tg:
        pipe(reader, dst_ds.writer(), num_runtime_threads=8)
    session.run(tg)

    return ws.blobs[str(dst_ds.content().label())].fetch()


class ReaderWithDelay(Reader):
    """Test reader class that inserts a delay between reading batches."""
    def __init__(self, reader, delay):
        Reader.__init__(self, schema=reader._schema)
        self.reader = reader
        self.delay = delay

    def setup_ex(self, global_init_net, global_finish_net):
        self.reader.setup_ex(global_init_net, global_finish_net)

    def read_ex(self, local_init_net, local_finish_net):
        read_net = core.Net('reader_body')

        def sleep_op(*args, **argd):
            time.sleep(self.delay)

        read_net.Python(sleep_op)([], [])
        return ([read_net], ) + self.reader.read(read_net)


class TestReaderWithLimit(TestCase):
    def test_runtime_threads(self):
        ws = workspace.C.Workspace()
        session = LocalSession(ws)
        src_ds = init_dataset(ws)
        totals = [None] * 3

        def proc(rec):
            # executed once
            with ops.task_init():
                counter1 = ops.CreateCounter([], ['global_counter'])
                counter2 = ops.CreateCounter([], ['global_counter2'])
                counter3 = ops.CreateCounter([], ['global_counter3'])
            # executed once per thread
            with ops.task_instance_init():
                task_counter = ops.CreateCounter([], ['task_counter'])
            # executed on each iteration
            ops.CountUp(counter1)
            ops.CountUp(task_counter)
            # executed once per thread
            with ops.task_instance_exit():
                with ops.loop(ops.RetrieveCount(task_counter)):
                    ops.CountUp(counter2)
                ops.CountUp(counter3)
            # executed once
            with ops.task_exit():
                totals[0] = final_output(ops.RetrieveCount(counter1))
                totals[1] = final_output(ops.RetrieveCount(counter2))
                totals[2] = final_output(ops.RetrieveCount(counter3))
            return rec

        # Read full data set from original reader
        with TaskGroup() as tg:
            pipe(src_ds.reader(), num_runtime_threads=8, processor=proc)
        session.run(tg)
        self.assertEqual(totals[0].fetch(), 100)
        self.assertEqual(totals[1].fetch(), 100)
        self.assertEqual(totals[2].fetch(), 8)

        # Read with a count-limited reader
        with TaskGroup() as tg:
            q1 = pipe(src_ds.reader(), num_runtime_threads=2)
            q2 = pipe(
                ReaderWithLimit(q1.reader(), num_iter=25),
                num_runtime_threads=3)
            pipe(q2, processor=proc, num_runtime_threads=6)
        session.run(tg)
        self.assertEqual(totals[0].fetch(), 25)
        self.assertEqual(totals[1].fetch(), 25)
        self.assertEqual(totals[2].fetch(), 6)

    def _test_limit_reader_init_shared(self, size):
        ws = workspace.C.Workspace()
        session = LocalSession(ws)

        # Build test dataset
        src_ds = init_dataset(ws, size=size)

        # Create an identically sized empty destnation dataset
        dst_init = core.Net('dst_init')
        with core.NameScope('dst'):
            dst_ds = Dataset(src_ds.content().clone_schema())
            dst_ds.init_empty(dst_init)
        ws.run(dst_init)

        return ws, session, src_ds, dst_init, dst_ds

    def _test_limit_reader_shared(self, reader_class, size, expected_read_len,
                                  expected_finish, num_threads, read_delay,
                                  **limiter_args):
        ws, session, src_ds, dst_init, dst_ds = \
            self._test_limit_reader_init_shared(size)

        # Read without limiter
        # WorkspaceType.GLOBAL is required because we are fetching
        # reader.data_finished() after the TaskGroup finishes.
        with TaskGroup(workspace_type=WorkspaceType.GLOBAL) as tg:
            if read_delay > 0:
                reader = reader_class(ReaderWithDelay(src_ds.reader(),
                                                      read_delay),
                                      **limiter_args)
            else:
                reader = reader_class(src_ds.reader(), **limiter_args)
            pipe(reader, dst_ds.writer(), num_runtime_threads=num_threads)
        session.run(tg)
        read_len = len(sorted(ws.blobs[str(dst_ds.content().label())].fetch()))
        self.assertEqual(read_len, expected_read_len)
        self.assertEqual(
            sorted(ws.blobs[str(dst_ds.content().label())].fetch()),
            list(range(expected_read_len))
        )
        self.assertEqual(ws.blobs[str(reader.data_finished())].fetch(),
                         expected_finish)

    def test_count_limit_reader_without_limit(self):
        # No iter count specified, should read all records.
        self._test_limit_reader_shared(ReaderWithLimit,
                                       size=100,
                                       expected_read_len=100,
                                       expected_finish=True,
                                       num_threads=8,
                                       read_delay=0,
                                       num_iter=None)

    def test_count_limit_reader_with_zero_limit(self):
        # Zero iter count specified, should read 0 records.
        self._test_limit_reader_shared(ReaderWithLimit,
                                       size=100,
                                       expected_read_len=0,
                                       expected_finish=False,
                                       num_threads=8,
                                       read_delay=0,
                                       num_iter=0)

    def test_count_limit_reader_with_low_limit(self):
        # Read with limit smaller than size of dataset
        self._test_limit_reader_shared(ReaderWithLimit,
                                       size=100,
                                       expected_read_len=10,
                                       expected_finish=False,
                                       num_threads=8,
                                       read_delay=0,
                                       num_iter=10)

    def test_count_limit_reader_with_high_limit(self):
        # Read with limit larger than size of dataset
        self._test_limit_reader_shared(ReaderWithLimit,
                                       size=100,
                                       expected_read_len=100,
                                       expected_finish=True,
                                       num_threads=8,
                                       read_delay=0,
                                       num_iter=110)

    def test_time_limit_reader_without_limit(self):
        # No duration specified, should read all records.
        self._test_limit_reader_shared(ReaderWithTimeLimit,
                                       size=100,
                                       expected_read_len=100,
                                       expected_finish=True,
                                       num_threads=8,
                                       read_delay=0.1,
                                       duration=0)

    def test_time_limit_reader_with_short_limit(self):
        # Read with insufficient time limit
        size = 50
        num_threads = 4
        sleep_duration = 0.25
        duration = 1
        expected_read_len = int(round(num_threads * duration / sleep_duration))
        # Because the time limit check happens before the delay + read op,
        # subtract a little bit of time to ensure we don't get in an extra read
        duration = duration - 0.25 * sleep_duration
        self._test_limit_reader_shared(ReaderWithTimeLimit,
                                       size=size,
                                       expected_read_len=expected_read_len,
                                       expected_finish=False,
                                       num_threads=num_threads,
                                       read_delay=sleep_duration,
                                       duration=duration)

    def test_time_limit_reader_with_long_limit(self):
        # Read with ample time limit
        self._test_limit_reader_shared(ReaderWithTimeLimit,
                                       size=50,
                                       expected_read_len=50,
                                       expected_finish=True,
                                       num_threads=4,
                                       read_delay=0.25,
                                       duration=6)

    def test_cached_reader(self):
        ws = workspace.C.Workspace()
        session = LocalSession(ws)

        def build_source_reader(size):
            src_ds = init_dataset(ws, size)
            return src_ds.reader()

        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name
            f.close()
            os.remove(path)

            # Read data for the first time.
            cached_reader1 = CachedReader(build_source_reader(100))
            init_step = cached_reader1.build_cache(path)
            session.run(init_step)

            data = read_all_data(ws, cached_reader1, session)
            self.assertEqual(sorted(data), list(range(100)))

            # Read data from cache.
            workspace.ResetWorkspace()
            cached_reader2 = CachedReader(build_source_reader(200))
            init_step = cached_reader2.build_cache(path)
            session.run(init_step)

            data = read_all_data(ws, cached_reader2, session)
            self.assertEqual(sorted(data), list(range(100)))

            shutil.rmtree(path)

            # We removed cache so we expect to receive data from original reader
            workspace.ResetWorkspace()
            cached_reader3 = CachedReader(build_source_reader(300))
            init_step = cached_reader3.build_cache(path)
            session.run(init_step)

            data = read_all_data(ws, cached_reader3, session)
            self.assertEqual(sorted(data), list(range(300)))

            shutil.rmtree(path)
