from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, dataio


class QueueReader(dataio.Reader):
    def __init__(self, queue, num_blobs=None, schema=None):
        dataio.Reader.__init__(self, schema)
        assert schema is not None or num_blobs is not None, (
            'Either schema or num_blobs must be provided.')

        self.queue = queue
        self.num_blobs = num_blobs

        if schema is not None:
            schema_num_blobs = len(schema.field_names())
            assert num_blobs is None or num_blobs == schema_num_blobs
            self.num_blobs = schema_num_blobs

    def setup_ex(self, init_net, exit_net):
        exit_net.CloseBlobsQueue([self.queue], 0)

    def read_ex(self, local_init_net, local_finish_net):
        dequeue_net = core.Net('dequeue_net')
        fields, status_blob = dequeue(dequeue_net, self.queue, self.num_blobs)
        return [dequeue_net], status_blob, fields


class QueueWriter(dataio.Writer):
    def __init__(self, queue):
        self.queue = queue

    def setup_ex(self, init_net, exit_net):
        exit_net.CloseBlobsQueue([self.queue], 0)

    def write_ex(self, fields, local_init_net, local_finish_net, status):
        enqueue_net = core.Net('enqueue_net')
        enqueue(enqueue_net, self.queue, fields, status)
        return [enqueue_net]


class QueueWrapper(object):
    def __init__(self, init_net, capacity, schema):
        self._queue = init_net.CreateBlobsQueue(
            [],
            capacity=capacity,
            num_blobs=len(schema.field_names()))
        self._schema = schema

    def reader(self):
        return QueueReader(self._queue, schema=self._schema)

    def writer(self):
        return QueueWriter(self._queue)

    def queue(self):
        return self._queue

    def schema(self):
        return self._schema


def enqueue(net, queue, data_blobs, status=None):
    if status is None:
        status = net.NextName("%s_enqueue_status" % str(queue))
    results = net.SafeEnqueueBlobs([queue] + data_blobs, data_blobs + [status])
    return results[-1]


def dequeue(net, queue, num_blobs, status=None):
    data_names = [net.NextName("%s_dequeue_data", i) for i in range(num_blobs)]
    if status is None:
        status = net.NextName("%s_dequeue_status")
    results = net.SafeDequeueBlobs(queue, data_names + [status])
    results = list(results)
    status_blob = results.pop(-1)
    return results, status_blob


def close_queue(step, *queues):
    close_net = core.Net("close_queue_net")
    for queue in queues:
        close_net.CloseBlobsQueue([queue], 0)
    close_step = core.execution_step("%s_step" % str(close_net), close_net)
    return core.execution_step(
        "%s_wraper_step" % str(close_net),
        [step, close_step])
