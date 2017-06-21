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
from caffe2.python.net_builder import ops
from caffe2.python.task import Task, Node
from caffe2.python.test_util import TestCase
from caffe2.python.dataio import ReaderWithLimit
import tempfile
import numpy as np
import shutil


def build_pipeline(node_id):
    with Node('reader:%d' % node_id):
        with Job.current().init_group, Task():
            data_arr = Struct(('val', np.array(list(range(10)))))
            data = ConstRecord(ops, data_arr)
            ds = Dataset(data, name='dataset:%d' % node_id)
            full_reader = ds.reader(ops)
            total = ops.Const([100])

        def inc_total(rec):
            ops.Add([total, rec.val()], [total])

        epoch_reader = ReaderWithLimit(full_reader, num_iter=3)
        pipe(epoch_reader, processor=inc_total)
        Job.current().add_stop_signal(epoch_reader.data_finished())
    return [total]


EXPECTED_TOTALS = [103, 115, 136, 145]


class TestCheckpoint(TestCase):
    def run_with(self, builder):
        with Job() as job:
            outputs = build_pipeline(node_id=0)
        output_fetcher = Task(step=core.Net('empty'), outputs=outputs)

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

            for node_id in range(3):
                ws = workspace.C.Workspace()
                session = LocalSession(ws)
                checkpoint = MultiNodeCheckpointManager(tmpdir, 'minidb')
                with Job() as job:
                    build_pipeline(node_id)
                compiled_job = job.compile(LocalSession)
                job_runner = JobRunner(compiled_job, checkpoint)
                num_epochs = job_runner(session)
                self.assertEquals(num_epochs, len(EXPECTED_TOTALS))

                # There are 12 global blobs after finishing up the job runner.
                # (only blobs on init_group are checkpointed)
                self.assertEquals(len(ws.blobs), 12)

            ws = workspace.C.Workspace()
            session = LocalSession(ws)
            self.assertEquals(len(ws.blobs), 0)
            model_blob_names = ['reader:1/task/GivenTensorInt64Fill:0',
                                'reader:2/task/GivenTensorInt64Fill:0']
            checkpoint = MultiNodeCheckpointManager(tmpdir, 'minidb')
            with Job() as job:
                for node_id in range(3):
                    build_pipeline(node_id)
            compiled_job = job.compile(LocalSession)
            job_runner = JobRunner(compiled_job, checkpoint)
            job_runner.load_blobs_from_checkpoints(blob_names=model_blob_names,
                                                   epoch=1, session=session)

            # Check that we can successfully load from checkpoints of epochs
            # 1 to 4, but not epoch 5.
            for epoch in range(1, 5):
                self.assertTrue(
                    job_runner.load_blobs_from_checkpoints(
                        blob_names=model_blob_names, epoch=epoch,
                        session=session))
                # Check that all the model blobs are loaded.
                for blob_name in model_blob_names:
                    self.assertTrue(ws.has_blob(blob_name))
                    self.assertEquals(ws.fetch_blob(blob_name),
                                      np.array([EXPECTED_TOTALS[epoch - 1]]))
            self.assertFalse(
                job_runner.load_blobs_from_checkpoints(
                    blob_names=model_blob_names, epoch=5, session=session))

        finally:
            shutil.rmtree(tmpdir)
