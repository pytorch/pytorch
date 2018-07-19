from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python.schema import (
    Struct, FetchRecord, NewRecord, FeedRecord, InitEmptyRecord)
from caffe2.python import core, workspace
from caffe2.python.session import LocalSession
from caffe2.python.dataset import Dataset
from caffe2.python.pipeline import pipe
from caffe2.python.queue_util import Queue
from caffe2.python.task import TaskGroup
from caffe2.python.test_util import TestCase
from caffe2.python.net_builder import ops
import numpy as np
import math


class TestPipeline(TestCase):
    def test_dequeue_many(self):
        init_net = core.Net('init')
        N = 17
        NUM_DEQUEUE_RECORDS = 3
        src_values = Struct(
            ('uid', np.array(range(N))),
            ('value', 0.1 * np.array(range(N))))
        expected_dst = Struct(
            ('uid', 2 * np.array(range(N))),
            ('value', np.array(N * [0.0])))

        with core.NameScope('init'):
            src_blobs = NewRecord(init_net, src_values)
            dst_blobs = InitEmptyRecord(init_net, src_values.clone_schema())
            counter = init_net.Const(0)
            ONE = init_net.Const(1)

        def proc1(rec):
            with core.NameScope('proc1'):
                out = NewRecord(ops, rec)
            ops.Add([rec.uid(), rec.uid()], [out.uid()])
            out.value.set(blob=rec.value(), unsafe=True)
            return out

        def proc2(rec):
            with core.NameScope('proc2'):
                out = NewRecord(ops, rec)
            out.uid.set(blob=rec.uid(), unsafe=True)
            ops.Sub([rec.value(), rec.value()], [out.value()])
            ops.Add([counter, ONE], [counter])
            return out

        src_ds = Dataset(src_blobs)
        dst_ds = Dataset(dst_blobs)

        with TaskGroup() as tg:
            out1 = pipe(
                src_ds.reader(),
                output=Queue(
                    capacity=11, num_dequeue_records=NUM_DEQUEUE_RECORDS),
                processor=proc1)
            out2 = pipe(out1, processor=proc2)
            pipe(out2, dst_ds.writer())

        ws = workspace.C.Workspace()
        FeedRecord(src_blobs, src_values, ws)
        session = LocalSession(ws)
        session.run(init_net)
        session.run(tg)
        output = FetchRecord(dst_blobs, ws=ws)
        num_dequeues = ws.blobs[str(counter)].fetch()

        self.assertEquals(
            num_dequeues, int(math.ceil(float(N) / NUM_DEQUEUE_RECORDS)))

        for a, b in zip(output.field_blobs(), expected_dst.field_blobs()):
            np.testing.assert_array_equal(a, b)
