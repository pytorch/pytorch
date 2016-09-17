from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core


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
