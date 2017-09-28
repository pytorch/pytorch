# Copyright (c) 2016-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

## @package queue_util
# Module caffe2.python.queue_util
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, dataio
from caffe2.python.task import TaskGroup


class _QueueReader(dataio.Reader):
    def __init__(self, wrapper, num_dequeue_records=1):
        assert wrapper.schema is not None, (
            'Queue needs a schema in order to be read from.')
        dataio.Reader.__init__(self, wrapper.schema())
        self._wrapper = wrapper
        self._num_dequeue_records = num_dequeue_records

    def setup_ex(self, init_net, exit_net):
        exit_net.CloseBlobsQueue([self._wrapper.queue()], 0)

    def read_ex(self, local_init_net, local_finish_net):
        self._wrapper._new_reader(local_init_net)
        dequeue_net = core.Net('dequeue')
        fields, status_blob = dequeue(
            dequeue_net,
            self._wrapper.queue(),
            len(self.schema().field_names()),
            field_names=self.schema().field_names(),
            num_records=self._num_dequeue_records)
        return [dequeue_net], status_blob, fields

    def read(self, net):
        net, _, fields = self.read_ex(net, None)
        return net, fields


class _QueueWriter(dataio.Writer):
    def __init__(self, wrapper):
        self._wrapper = wrapper

    def setup_ex(self, init_net, exit_net):
        exit_net.CloseBlobsQueue([self._wrapper.queue()], 0)

    def write_ex(self, fields, local_init_net, local_finish_net, status):
        self._wrapper._new_writer(self.schema(), local_init_net)
        enqueue_net = core.Net('enqueue')
        enqueue(enqueue_net, self._wrapper.queue(), fields, status)
        return [enqueue_net]


class QueueWrapper(dataio.Pipe):
    def __init__(self, handler, schema=None, num_dequeue_records=1):
        dataio.Pipe.__init__(self, schema, TaskGroup.LOCAL_SETUP)
        self._queue = handler
        self._num_dequeue_records = num_dequeue_records

    def reader(self):
        return _QueueReader(
            self, num_dequeue_records=self._num_dequeue_records)

    def writer(self):
        return _QueueWriter(self)

    def queue(self):
        return self._queue


class Queue(QueueWrapper):
    def __init__(self, capacity, schema=None, name='queue',
                 num_dequeue_records=1):
        # find a unique blob name for the queue
        net = core.Net(name)
        queue_blob = net.AddExternalInput(net.NextName('handler'))
        QueueWrapper.__init__(
            self, queue_blob, schema, num_dequeue_records=num_dequeue_records)
        self.capacity = capacity
        self._setup_done = False

    def setup(self, global_init_net):
        assert self._schema, 'This queue does not have a schema.'
        self._setup_done = True
        global_init_net.CreateBlobsQueue(
            [],
            [self._queue],
            capacity=self.capacity,
            num_blobs=len(self._schema.field_names()),
            field_names=self._schema.field_names())


def enqueue(net, queue, data_blobs, status=None):
    if status is None:
        status = net.NextName('status')
    results = net.SafeEnqueueBlobs([queue] + data_blobs, data_blobs + [status])
    return results[-1]


def dequeue(net, queue, num_blobs, status=None, field_names=None,
            num_records=1):
    if field_names is not None:
        assert len(field_names) == num_blobs
        data_names = [net.NextName(name) for name in field_names]
    else:
        data_names = [net.NextName('data', i) for i in range(num_blobs)]
    if status is None:
        status = net.NextName('status')
    results = net.SafeDequeueBlobs(
        queue, data_names + [status], num_records=num_records)
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
