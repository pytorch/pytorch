




from caffe2.python.schema import (
    Struct, FetchRecord, NewRecord, FeedRecord, InitEmptyRecord)
from caffe2.python import core, workspace
from caffe2.python.session import LocalSession
from caffe2.python.dataset import Dataset
from caffe2.python.pipeline import pipe
from caffe2.python.task import TaskGroup
from caffe2.python.test_util import TestCase
import numpy as np


class TestLocalSession(TestCase):
    def test_local_session(self):
        init_net = core.Net('init')
        src_values = Struct(
            ('uid', np.array([1, 2, 6])),
            ('value', np.array([1.4, 1.6, 1.7])))
        expected_dst = Struct(
            ('uid', np.array([2, 4, 12])),
            ('value', np.array([0.0, 0.0, 0.0])))

        with core.NameScope('init'):
            src_blobs = NewRecord(init_net, src_values)
            dst_blobs = InitEmptyRecord(init_net, src_values.clone_schema())

        def proc1(rec):
            net = core.Net('proc1')
            with core.NameScope('proc1'):
                out = NewRecord(net, rec)
            net.Add([rec.uid(), rec.uid()], [out.uid()])
            out.value.set(blob=rec.value(), unsafe=True)
            return [net], out

        def proc2(rec):
            net = core.Net('proc2')
            with core.NameScope('proc2'):
                out = NewRecord(net, rec)
            out.uid.set(blob=rec.uid(), unsafe=True)
            net.Sub([rec.value(), rec.value()], [out.value()])
            return [net], out

        src_ds = Dataset(src_blobs)
        dst_ds = Dataset(dst_blobs)

        with TaskGroup() as tg:
            out1 = pipe(src_ds.reader(), processor=proc1)
            out2 = pipe(out1, processor=proc2)
            pipe(out2, dst_ds.writer())

        ws = workspace.C.Workspace()
        FeedRecord(src_blobs, src_values, ws)
        session = LocalSession(ws)
        session.run(init_net)
        session.run(tg)
        output = FetchRecord(dst_blobs, ws=ws)

        for a, b in zip(output.field_blobs(), expected_dst.field_blobs()):
            np.testing.assert_array_equal(a, b)
