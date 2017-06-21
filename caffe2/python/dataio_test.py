from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python.dataio import ReaderWithLimit
from caffe2.python.dataset import Dataset
from caffe2.python.pipeline import pipe
from caffe2.python.schema import Struct, NewRecord, FeedRecord
from caffe2.python.session import LocalSession
from caffe2.python.task import TaskGroup, final_output, WorkspaceType
from caffe2.python.test_util import TestCase
from caffe2.python import core, workspace
from caffe2.python.net_builder import ops
import numpy as np


def init_dataset(ws):
    src_init = core.Net('src_init')
    with core.NameScope('src'):
        src_values = Struct(('label', np.array(range(100))))
        src_blobs = NewRecord(src_init, src_values)
        src_ds = Dataset(src_blobs)
        FeedRecord(src_blobs, src_values, ws)
    ws.run(src_init)
    return src_ds


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

        """ 1. Feed full dataset """
        with TaskGroup() as tg:
            pipe(src_ds.reader(), num_runtime_threads=8, processor=proc)
        session.run(tg)
        self.assertEquals(totals[0].fetch(), 100)
        self.assertEquals(totals[1].fetch(), 100)
        self.assertEquals(totals[2].fetch(), 8)

        """ 2. Add a few steps in between """
        with TaskGroup() as tg:
            q1 = pipe(src_ds.reader(), num_runtime_threads=2)
            q2 = pipe(
                ReaderWithLimit(q1.reader(), num_iter=25),
                num_runtime_threads=3)
            pipe(q2, processor=proc, num_runtime_threads=6)
        session.run(tg)
        self.assertEquals(totals[0].fetch(), 25)
        self.assertEquals(totals[1].fetch(), 25)
        self.assertEquals(totals[2].fetch(), 6)


    def test_reader_with_limit(self):
        ws = workspace.C.Workspace()
        session = LocalSession(ws)

        """ 1. feed full dataset """
        src_ds = init_dataset(ws)

        """ 2. Read with limit smaller than size of dataset """
        dst_init = core.Net('dst_init')
        with core.NameScope('dst'):
            dst_ds = Dataset(src_ds.content().clone_schema())
            dst_ds.init_empty(dst_init)
        ws.run(dst_init)

        # WorkspaceType.GLOBAL is required because we are fetching
        # reader.data_finished() after the TaskGroup finishes.
        with TaskGroup(workspace_type=WorkspaceType.GLOBAL) as tg:
            reader = ReaderWithLimit(src_ds.reader(), num_iter=10)
            pipe(reader, dst_ds.writer(), num_threads=8)
        session.run(tg)

        self.assertFalse(ws.blobs[str(reader.data_finished())].fetch())
        self.assertEquals(
            sorted(ws.blobs[str(dst_ds.content().label())].fetch()),
            list(range(10))
        )

        """ 3. Read with limit larger than size of dataset """
        ws.run(dst_init)
        with TaskGroup(workspace_type=WorkspaceType.GLOBAL) as tg:
            reader = ReaderWithLimit(src_ds.reader(), num_iter=110)
            pipe(reader, dst_ds.writer(), num_runtime_threads=8)
        session.run(tg)
        self.assertEquals(
            sorted(ws.blobs[str(dst_ds.content().label())].fetch()),
            list(range(100))
        )
        self.assertTrue(ws.blobs[str(reader.data_finished())].fetch())

        """ 4. Read without counter """
        ws.run(dst_init)
        with TaskGroup(workspace_type=WorkspaceType.GLOBAL) as tg:
            reader = ReaderWithLimit(src_ds.reader(), num_iter=None)
            pipe(reader, dst_ds.writer(), num_threads=8)
        session.run(tg)
        self.assertEquals(
            sorted(ws.blobs[str(dst_ds.content().label())].fetch()),
            list(range(100))
        )
        self.assertTrue(ws.blobs[str(reader.data_finished())].fetch())

        """ 5. Read using the same reader without resetting workspace """
        session.run(tg)
        self.assertEquals(
            sorted(ws.blobs[str(dst_ds.content().label())].fetch()),
            sorted(list(range(100)) * 2)
        )
