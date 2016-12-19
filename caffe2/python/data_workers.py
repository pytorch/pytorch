from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


'''
This module provides a python-land multithreaded data input mechanism
for Caffe2 nets.

Basic usage is as follows:
   coordinator = data_workers.init_data_input_workers(
      net,
      ["data", "label"],
      my_fetch_fun,
      32,
      "train"
   )
   ...
   coordinator.start()

First argument is the Caffe2 net (or model helper), and second argument
is list of input blobs that are to be fed.

Last argument is used to distinguish different sources of data, such as train
or test data. This is to ensure the data does not get mixed up, although the
nets would share blobs.

To do the actual data loading, one defines a "fetcher function"
that has call signature
   my_fetch_fun(worker_id, batch_size)

This function returns a list of numpy arrays corresponding to the different
input blobs. In the example above, it would return two arrays, one for the
data blob and another for the labels. These arrays can have arbitrary number
of elements (i.e they do not need to match the batch size). The batch size
is provided for the function as a hint only.

For example, fetcher function could download images from a remote service or
load random images from a directory on a file system.

For a dummy example, see the data_workers_test unit test.

Note that for data_parallel_models, init_data_input_workers will be called
for each GPU. Note that the 'coordinator' returned by the function is same
each time.
'''

import Queue
import logging
import threading
import atexit
import numpy as np

from caffe2.python import workspace, core, scope
from caffe2.proto import caffe2_pb2

log = logging.getLogger("data_workers")


def init_data_input_workers(
    net,
    input_blob_names,
    fetch_fun,
    batch_size,
    num_worker_threads=2,
    input_source_name="train",
):
    global global_coordinator
    device_option = scope.CurrentDeviceScope()
    if (device_option is None):
        device_option = caffe2_pb2.DeviceOption(device_type=caffe2_pb2.CPU)

    # Create coordinator object
    coordinator = DataInputCoordinator(
        net,
        input_blob_names,
        batch_size,
        device_option,
        scope.CurrentNameScope(),
        input_source_name,
    )

    # Launch fetch worker threads
    workers = [
        threading.Thread(
            target=fetcher,
            args=[coordinator, global_coordinator._fetcher_id_seq + i,
                  fetch_fun, batch_size, input_blob_names],
        ) for i in range(num_worker_threads)
    ]
    global_coordinator._fetcher_id_seq += num_worker_threads

    workers.append(threading.Thread(
        target=enqueuer,
        args=[coordinator]))
    coordinator._workers = workers
    global_coordinator.add(coordinator)

    return global_coordinator


class DataInputCoordinator(object):
    def __init__(self, net, input_blob_names, batch_size,
                 device_option, namescope, input_source_name):
        self._net = net
        self._input_blob_names = input_blob_names
        self._batch_size = batch_size
        self._internal_queue = Queue.Queue(maxsize=500)
        self._queues = []
        self._device_option = device_option
        self._namescope = namescope
        self._active = True
        self._started = False
        self._workers = []
        self._input_source_name = input_source_name
        self._create_caffe2_queues_and_ops()

    def is_active(self):
        return self._active

    def _start(self):
        if self._started:
            return
        self._active = True
        self._started = True

        for w in self._workers:
            w.daemon = True
            w.start()

    def _stop(self, reason=None):
        self._active = False
        if reason is not None:
            log.error("Data input failed due to an error: {}".format(reason))
        self._started = False

    def _wait_finish(self):
        log.info("Wait for workers to die")
        for w in self._workers:
            if w != threading.current_thread():
                w.join(1.0)  # don't wait forever, thread may be blocked in i/o
        log.info("...finished")

    def _get(self):
        while self.is_active():
            try:
                return self._internal_queue.get(block=True, timeout=0.5)
            except Queue.Empty:
                continue
        return None

    def put(self, chunk):
        while self.is_active():
            try:
                self._internal_queue.put(chunk, block=True, timeout=0.5)
                return
            except Queue.Full:
                log.debug("Queue full: stalling fetchers...")
                continue

    def _enqueue_batch(self):
        '''
        This pulls data from the python-side queue and collects them
        into batch-sized pieces.
        '''
        cur_batch = [np.array([]) for d in self._input_blob_names]

        # Collect data until we have a full batch size
        while cur_batch[0].shape[0] < self._batch_size and self.is_active():
            chunk = self._get()
            if chunk is None:
                continue

            for j, chunk_elem in enumerate(chunk):
                if cur_batch[j].shape[0] == 0:
                    cur_batch[j] = chunk_elem.copy()
                else:
                    cur_batch[j] = np.append(cur_batch[j], chunk_elem, axis=0)

        # Return data over the batch size back to queue
        if cur_batch[0].shape[0] > self._batch_size:
            leftover = [c[self._batch_size:] for c in cur_batch]
            cur_batch = [c[:self._batch_size] for c in cur_batch]
            try:
                self._internal_queue.put(leftover, block=False)
            except Queue.Full:
                pass

            assert cur_batch[0].shape[0] == self._batch_size

        if self.is_active():
            for b, q, c in zip(self._input_blob_names, self._queues, cur_batch):
                self._enqueue(b, q, c)

    def _enqueue(self, blob_name, queue, data_arr):
        '''
        Enqueue the correctly sized batch arrays to Caffe2's queue.
        '''
        scratch_name = self._namescope + blob_name + \
            "_scratch_" + self._input_source_name
        blob = core.BlobReference(scratch_name)
        workspace.FeedBlob(
            blob,
            data_arr,
            device_option=self._device_option
        )

        op = core.CreateOperator(
            "EnqueueBlobs",
            [queue, blob],
            [blob],
            device_option=self._device_option
        )
        workspace.RunOperatorOnce(op)

    def _create_caffe2_queues_and_ops(self):
        '''
        Creates queues on caffe2 side, and respective operators
        to pull (dequeue) blobs from the queues.
        '''
        def create_queue(queue_name, num_blobs, capacity):
            workspace.RunOperatorOnce(
                core.CreateOperator(
                    "CreateBlobsQueue",
                    [], [queue_name],
                    num_blobs=1,
                    capacity=capacity))
            return core.ScopedBlobReference(queue_name)

        for blob_name in self._input_blob_names:
            qname = blob_name + "_c2queue" + "_" + self._input_source_name
            q = create_queue(qname, num_blobs=1, capacity=4)
            self._queues.append(q)
            log.info("Created queue: {}".format(q))

            # Add operator to the Caffe2 network to dequeue
            self._net.DequeueBlobs(q, blob_name)


class GlobalCoordinator(object):
    def __init__(self):
        self._coordinators = []
        self._fetcher_id_seq = 0
        self.register_shutdown_handler()

    def add(self, coordinator):
        self._coordinators.append(coordinator)

    def start(self):
        for c in self._coordinators:
            c._start()

    def stop(self):
        for c in self._coordinators:
            c._stop()
        for c in self._coordinators:
            c._wait_finish()
        self._coordinators = []

    def register_shutdown_handler(self):
        def cleanup():
            self.stop()

        atexit.register(cleanup)


global_coordinator = GlobalCoordinator()


def fetcher(coordinator, worker_id, fetch_fun, batch_size, input_blob_names):
    while coordinator.is_active():
        try:
            input_data = fetch_fun(worker_id, batch_size)
            if input_data is None:
                log.warn("Fetcher function returned None")
                continue

            assert len(input_data) == len(input_blob_names), \
                "Expecting data blob for each input"
            for d in input_data:
                assert isinstance(d, np.ndarray), \
                    "Fetcher function must return a numpy array"
            for d in input_data[1:]:
                assert d.shape[0] == input_data[0].shape[0], \
                    "Each returned input must have equal number of samples"

            coordinator.put(input_data)
        except Exception as e:
            log.error(e)
            coordinator._stop("Exception in fetcher {}: {}".format(
                worker_id, e
            ))


def enqueuer(coordinator):
    while coordinator.is_active():
        coordinator._enqueue_batch()
