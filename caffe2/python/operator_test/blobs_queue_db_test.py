from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np

import caffe2.proto.caffe2_pb2 as caffe2_pb2
from caffe2.python import core, workspace, timeout_guard


class BlobsQueueDBTest(unittest.TestCase):
    def test_create_blobs_queue_db_string(self):
        def add_blobs(queue, num_samples):
            blob = core.BlobReference("blob")
            status = core.BlobReference("blob_status")
            for i in range(num_samples):
                self._add_blob_to_queue(
                    queue, self._create_test_tensor_protos(i), blob, status
                )
        self._test_create_blobs_queue_db(add_blobs)

    def test_create_blobs_queue_db_tensor(self):
        def add_blobs(queue, num_samples):
            blob = core.BlobReference("blob")
            status = core.BlobReference("blob_status")
            for i in range(num_samples):
                data = self._create_test_tensor_protos(i)
                data = np.array([data], dtype=str)
                self._add_blob_to_queue(
                    queue, data, blob, status
                )
        self._test_create_blobs_queue_db(add_blobs)

    def _test_create_blobs_queue_db(self, add_blobs_fun):
        num_samples = 10000
        batch_size = 10
        init_net = core.Net('init_net')
        net = core.Net('test_create_blobs_queue_db')
        queue = init_net.CreateBlobsQueue([], 'queue', capacity=num_samples)
        reader = init_net.CreateBlobsQueueDB(
            [queue],
            'blobs_queue_db_reader',
            value_blob_index=0,
            timeout_secs=0.1,
        )
        workspace.RunNetOnce(init_net)

        add_blobs_fun(queue, num_samples)

        net.TensorProtosDBInput(
            [reader], ['image', 'label'], batch_size=batch_size)
        workspace.CreateNet(net)

        close_net = core.Net('close_net')
        close_net.CloseBlobsQueue([queue], [])

        for i in range(int(num_samples / batch_size)):
            print("Running net, iteration {}".format(i))
            with timeout_guard.CompleteInTimeOrDie(2.0):
                workspace.RunNet(net)

            images = workspace.FetchBlob('image')
            labels = workspace.FetchBlob('label')
            self.assertEqual(batch_size, len(images))
            self.assertEqual(batch_size, len(labels))
            for idx, item in enumerate(images):
                self.assertEqual(
                    "foo{}".format(i * batch_size + idx).encode('utf-8'), item
                )
            for item in labels:
                self.assertEqual(1, item)
        workspace.RunNetOnce(close_net)

    def _add_blob_to_queue(self, queue, data, blob, status):
        workspace.FeedBlob(blob, data)
        op = core.CreateOperator(
            "SafeEnqueueBlobs",
            [queue, blob],
            [blob, status],
        )
        workspace.RunOperatorOnce(op)

    def _create_test_tensor_protos(self, idx):
        item = caffe2_pb2.TensorProtos()
        data = item.protos.add()
        data.data_type = core.DataType.STRING
        data.string_data.append("foo{}".format(idx).encode('utf-8'))
        label = item.protos.add()
        label.data_type = core.DataType.INT32
        label.int32_data.append(1)

        return item.SerializeToString()
