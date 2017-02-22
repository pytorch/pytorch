from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python.schema import Struct, ConstRecord
from caffe2.python import core, workspace
from caffe2.python.session import LocalSession
from caffe2.python.dataset import Dataset
from caffe2.python.pipeline import pipe
from caffe2.python.checkpoint import (
    CheckpointManager, MultiNodeCheckpointManager, Job, JobRunner)
from caffe2.python.task import Task, Node
from caffe2.python.test_util import TestCase
from caffe2.python.dataio import ReaderWithLimit
import tempfile
import numpy as np
import shutil


def build_job():
    with Node('reader'):
        with Job() as job:
            with job.init_group:
                init_net = core.Net('init_net')
                data_arr = Struct(('val', np.array(range(10))))
                data = ConstRecord(init_net, data_arr)
                ds = Dataset(data)
                full_reader = ds.reader(init_net)
                total = init_net.Const([100])
                Task(step=init_net)

            def inc_total(rec):
                net = core.Net('inc_total')
                net.Add([total, rec.val()], [total])
                return [net]

            epoch_reader = ReaderWithLimit(full_reader, num_iter=3)
            pipe(epoch_reader, processor=inc_total)
            job.add_stop_signal(epoch_reader.data_finished())

        total_fetcher = Task(step=core.Net('empty'), outputs=[total])
    return job, total_fetcher


EXPECTED_TOTALS = [103, 115, 136, 145]


class TestCheckpoint(TestCase):
    def run_with(self, builder):
        job, output_fetcher = build_job()

        def fetch_total(session):
            session.run(output_fetcher)
            return output_fetcher.outputs()[0].fetch()

        session, checkpoint = builder()
        compiled_job = job.compile(LocalSession)
        num_epochs = JobRunner(compiled_job, checkpoint)(session)
        self.assertEquals(num_epochs, len(EXPECTED_TOTALS))
        self.assertEquals(fetch_total(session), EXPECTED_TOTALS[-1])

        for initial_epoch in range(1, num_epochs + 1):
            session, checkpoint = builder()
            JobRunner(
                compiled_job,
                checkpoint, resume_from_epoch=initial_epoch)(session)
            self.assertEquals(fetch_total(session), EXPECTED_TOTALS[-1])

        for epoch in range(1, num_epochs + 1):
            session.run(checkpoint.load(epoch))
            self.assertEquals(fetch_total(session), EXPECTED_TOTALS[epoch - 1])

    def test_single_checkpoint(self):
        # test single node
        with tempfile.NamedTemporaryFile() as tmp:

            def builder():
                ws = workspace.C.Workspace()
                session = LocalSession(ws)
                checkpoint = CheckpointManager(tmp.name, 'minidb')
                return session, checkpoint

            self.run_with(builder)

        # test multi-node
        try:
            tmpdir = tempfile.mkdtemp()

            def builder():
                ws = workspace.C.Workspace()
                session = LocalSession(ws)
                checkpoint = MultiNodeCheckpointManager(tmpdir, 'minidb')
                return session, checkpoint

            self.run_with(builder)
        finally:
            shutil.rmtree(tmpdir)
