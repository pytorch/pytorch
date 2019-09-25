from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core, workspace
from caffe2.python.dataset import Dataset
from caffe2.python.schema import (
    Struct, Map, Scalar, from_blob_list, NewRecord, FeedRecord)
from caffe2.python.record_queue import RecordQueue
from caffe2.python.test_util import TestCase
import numpy as np


class TestRecordQueue(TestCase):
    def test_record_queue(self):
        num_prod = 8
        num_consume = 3
        schema = Struct(
            ('floats', Map(
                Scalar(np.int32),
                Scalar(np.float32))),
        )
        contents_raw = [
            [1, 2, 3],  # len
            [11, 21, 22, 31, 32, 33],  # key
            [1.1, 2.1, 2.2, 3.1, 3.2, 3.3],  # value
        ]
        contents = from_blob_list(schema, contents_raw)
        ds = Dataset(schema)
        net = core.Net('init')
        ds.init_empty(net)

        content_blobs = NewRecord(net, contents)
        FeedRecord(content_blobs, contents)
        writer = ds.writer(init_net=net)
        writer.write_record(net, content_blobs)
        reader = ds.reader(init_net=net)

        # prepare receiving dataset
        rec_dataset = Dataset(contents, name='rec')
        rec_dataset.init_empty(init_net=net)
        rec_dataset_writer = rec_dataset.writer(init_net=net)

        workspace.RunNetOnce(net)

        queue = RecordQueue(contents, num_threads=num_prod)

        def process(net, fields):
            new_fields = []
            for f in fields.field_blobs():
                new_f = net.Copy(f)
                new_fields.append(new_f)
            new_fields = from_blob_list(fields, new_fields)
            return new_fields

        q_reader, q_step, q_exit, fields = queue.build(reader, process)
        producer_step = core.execution_step('producer', [q_step, q_exit])

        consumer_steps = []
        for i in range(num_consume):
            name = 'queue_reader_' + str(i)
            net_consume = core.Net(name)
            should_stop, fields = q_reader.read_record(net_consume)
            step_consume = core.execution_step(name, net_consume)

            name = 'dataset_writer_' + str(i)
            net_dataset = core.Net(name)
            rec_dataset_writer.write(net_dataset, fields.field_blobs())
            step_dataset = core.execution_step(name, net_dataset)

            step = core.execution_step(
                'consumer_' + str(i),
                [step_consume, step_dataset],
                should_stop_blob=should_stop)
            consumer_steps.append(step)
        consumer_step = core.execution_step(
            'consumers', consumer_steps, concurrent_substeps=True)

        work_steps = core.execution_step(
            'work', [producer_step, consumer_step], concurrent_substeps=True)

        plan = core.Plan('test')
        plan.AddStep(work_steps)
        core.workspace.RunPlan(plan)
        data = workspace.FetchBlobs(rec_dataset.get_blobs())
        self.assertEqual(6, sum(data[0]))
        self.assertEqual(150, sum(data[1]))
        self.assertAlmostEqual(15, sum(data[2]), places=5)

if __name__ == "__main__":
    import unittest
    unittest.main()
