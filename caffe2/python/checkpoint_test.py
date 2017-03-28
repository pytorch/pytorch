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


def build_job(node_id):
    all_outputs = []
    with Job() as job:
        with Node('reader' + str(node_id)):
            with job.init_group:
                init_net = core.Net('init_net' + str(node_id))
                data_arr = Struct(('val', np.array(range(10))))
                data = ConstRecord(init_net, data_arr)
                ds = Dataset(data, name='dataset' + str(node_id))
                full_reader = ds.reader(init_net)
                total = init_net.Const([100])
                Task(step=init_net)

            def inc_total(rec):
                net = core.Net('inc_total' + str(node_id))
                net.Add([total, rec.val()], [total])
                return [net]

            epoch_reader = ReaderWithLimit(full_reader, num_iter=3)
            pipe(epoch_reader, processor=inc_total)
            job.add_stop_signal(epoch_reader.data_finished())
            all_outputs.append(total)

    total_fetcher = Task(step=core.Net('empty'), outputs=all_outputs)
    return job, total_fetcher

EXPECTED_TOTALS = [103, 115, 136, 145]


class TestCheckpoint(TestCase):
    def run_with(self, builder):
        job, output_fetcher = build_job(node_id=0)

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

    def test_load_model_from_checkpoints(self):
        try:
            tmpdir = tempfile.mkdtemp()
            ws = workspace.C.Workspace()
            session = LocalSession(ws)
            checkpoint = MultiNodeCheckpointManager(tmpdir, 'minidb')

            for node_id in range(3):
                job, output_fetcher = build_job(node_id)
                compiled_job = job.compile(LocalSession)
                job_runner = JobRunner(compiled_job, checkpoint)
                num_epochs = job_runner(session)
                self.assertEquals(num_epochs, len(EXPECTED_TOTALS))

            # There are 44 blobs after finishing up the job runner.
            self.assertEquals(len(ws.blobs), 44)

            ws = workspace.C.Workspace()
            session = LocalSession(ws)
            self.assertEquals(len(ws.blobs), 0)
            model_blob_names = ['init_net0/GivenTensorInt64Fill:0',
                                'init_net1/GivenTensorInt64Fill:0']
            job_runner.load_blobs_from_checkpoints(blob_names=model_blob_names,
                                                   epoch=1, session=session)
            # In addition to the two model blobs, we also have 3 output blobs
            # and one runnable blob. So there are 6 blobs in total.
            self.assertEquals(len(ws.blobs), 6)
            # Check that all the model blobs are loaded.
            for blob_name in model_blob_names:
                self.assertTrue(ws.has_blob(blob_name))
                self.assertEquals(ws.fetch_blob(blob_name), np.array([103]))

        finally:
            shutil.rmtree(tmpdir)
