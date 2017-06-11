from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python.dataio import ReaderWithLimit
from caffe2.python.dataset import Dataset
from caffe2.python.pipeline import pipe
from caffe2.python.schema import Struct, NewRecord, FeedRecord
from caffe2.python.session import LocalSession
from caffe2.python.task import TaskGroup
from caffe2.python.test_util import TestCase
from caffe2.python import core, workspace
import numpy as np


class TestReaderWithLimit(TestCase):
    def test_reader_with_limit(self):
        ws = workspace.C.Workspace()
        session = LocalSession(ws)

        """ 1. feed full dataset """
        src_init = core.Net('src_init')
        with core.NameScope('src'):
            src_values = Struct(('label', np.array(list(range(100)))))
            src_blobs = NewRecord(src_init, src_values)
            src_ds = Dataset(src_blobs)
            FeedRecord(src_blobs, src_values, ws)
        ws.run(src_init)

        """ 2. Read with limit smaller than size of dataset """
        dst_init = core.Net('dst_init')
        with core.NameScope('dst'):
            dst_ds = Dataset(src_values.clone_schema())
            dst_ds.init_empty(dst_init)
        ws.run(dst_init)

        with TaskGroup() as tg:
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
        with TaskGroup() as tg:
            reader = ReaderWithLimit(src_ds.reader(), num_iter=110)
            pipe(reader, dst_ds.writer(), num_threads=8)
        session.run(tg)
        self.assertEquals(
            sorted(ws.blobs[str(dst_ds.content().label())].fetch()),
            list(range(100))
        )
        self.assertTrue(ws.blobs[str(reader.data_finished())].fetch())

        """ 4. Read without counter """
        ws.run(dst_init)
        with TaskGroup() as tg:
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
