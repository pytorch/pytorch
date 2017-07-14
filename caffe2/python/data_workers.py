## @package data_workers
# Module caffe2.python.data_workers
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
      batch_size=32,
      input_source_name="train",
      dont_rebatch=False
   )
   ...
   coordinator.start()

First argument is the Caffe2 net (or model helper), and second argument
is list of input blobs that are to be fed.

Argument 'input_source_name' is used to distinguish different sources of data,
such as train or test data. This is to ensure the data does not get mixed up,
although two nets would share blobs.

To do the actual data loading, one defines a "fetcher function"
that has call signature
   my_fetch_fun(worker_id, batch_size)

Optionally, one can define a "init function" that is called once before
threads start, and has call signature:
   my_init_fun(data_coordinator, global_coordinator)

If dont_rebatch is set to True, the data input is not batched into equal sized
chunks but data directly provided by fetchers is used.

'batch_columns' can be used to specify which dimension is the batch dimension,
for each of the inputs. Default is 0 for all iputs.

'timeout' is the timeout in seconds after which if no data is available, the
net will fail (default 600s = 10 mins).

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

try:
    import Queue
except ImportError:
    # Py3
    import queue as Queue
from itertools import chain
import logging
import threading
import atexit
import numpy as np
import time
import collections

from caffe2.python import workspace, core, scope, utils
from caffe2.proto import caffe2_pb2

log = logging.getLogger("data_workers")
log.setLevel(logging.INFO)
LOG_INT_SECS = 60


def get_worker_ids(num_workers):
    return list(range(0, num_workers))


def init_data_input_workers(
    net,
    input_blob_names,
    fetch_fun,
    batch_size,
    num_worker_threads=2,
    input_source_name="train",
    max_buffered_batches=800,
    init_fun=None,
    external_loggers=None,
    dont_rebatch=False,
    batch_columns=None,
    timeout=600
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
        global_coordinator.get_queue(input_source_name, max_buffered_batches),
        init_fun=init_fun,
        external_loggers=external_loggers,
        dont_rebatch=dont_rebatch,
        batch_columns=batch_columns,
    )

    # Launch fetch worker threads
    worker_ids = [
        global_coordinator.get_new_worker_id()
        for i in range(num_worker_threads)
    ]
    workers = [
        threading.Thread(
            target=fetcher,
            name="data_workers fetcher id {}".format(worker_id),
            args=[coordinator, worker_id, fetch_fun, batch_size, input_blob_names],
        ) for worker_id in worker_ids
    ]

    workers.append(threading.Thread(
        target=enqueuer,
        name="Enqueuer {} {}".format(input_source_name, scope.CurrentNameScope()),
        args=[coordinator]))
    coordinator._workers = workers
    global_coordinator.add(coordinator)

    return global_coordinator


class DataInputCoordinator(object):
    def __init__(self, net, input_blob_names, batch_size,
                 device_option, namescope, input_source_name, queue,
                 init_fun=None, external_loggers=None, dont_rebatch=False,
                 batch_columns=None, timeout=600):
        self._counter = 0
        self._input_blob_names = input_blob_names
        self._batch_size = batch_size
        self._internal_queue = queue
        self._queues = []
        self._device_option = device_option
        self._namescope = namescope
        self._active = True
        self._started = False
        self._timeout = timeout
        self._workers = []
        self._input_source_name = input_source_name
        self._c2_queue_capacity = 4
        self._create_caffe2_queues(net)
        self._create_caffe2_ops(net)
        self._inputs = 0
        self._prev_seconds = 0
        self._last_warning = time.time()
        self._init_fun = init_fun
        self._metrics = collections.defaultdict(lambda: 0)
        self._external_loggers = external_loggers
        self._dont_rebatch = dont_rebatch
        self._init_scratch()

        if batch_columns is None:
            batch_columns = [0 for _ in input_blob_names]
        self._batch_columns = batch_columns

    def is_active(self):
        return self._active

    def init(self, global_coordinator):
        if self._init_fun and not self._started:
            self._init_fun(self, global_coordinator)

    def _start(self):
        if self._started:
            return
        self._active = True
        self._started = True
        self._inputs = 0
        self._prev_seconds = time.time()

        for w in self._workers:
            w.daemon = True
            w.start()

    def _stop(self, reason=None):
        try:
            self._active = False
            if reason is not None:
                log.error("Data input failed due to an error: {}".format(reason))

            for q in self._queues:
                workspace.RunOperatorOnce(
                    core.CreateOperator("CloseBlobsQueue", [q], [])
                )
            self._started = False
        finally:
            self._log_inputs_per_interval(0, force=True)

    def _wait_finish(self):
        print("Wait for workers to die: {}".format(self._input_source_name))
        for w in self._workers:
            if w != threading.current_thread():
                w.join(5.0)  # don't wait forever, thread may be blocked in i/o
        success = True
        for w in self._workers:
            if w.isAlive():
                print("Worker {} failed to close while waiting".format(w))
                success = False

        # Release memory for the scratch blobs
        if success:
            utils.ResetBlobs(self._scratch_blob.values())
            utils.ResetBlobs(self._scratch_status.values())

        print("All workers terminated: {}".format(success))
        return success

    def _get(self):
        while self.is_active():
            try:
                return self._internal_queue.get(block=True, timeout=0.5)
            except Queue.Empty:
                continue
        return None

    def put(self, chunk):
        if len(chunk) == 0:
            print("Worker provided zero length input")
            return
        while self.is_active():
            try:
                qsize = self._internal_queue.qsize()
                if qsize < 2 and (time.time() - self._last_warning) > LOG_INT_SECS:
                    print("Warning, data loading lagging behind: " +
                             "name={}".format(qsize, self._input_source_name))
                    self._last_warning = time.time()
                self._counter += 1
                self._internal_queue.put(chunk, block=True, timeout=0.5)
                self._log_inputs_per_interval(chunk[0].shape[0])
                return
            except Queue.Full:
                log.debug("Queue full: stalling fetchers...")
                continue

    def _enqueue_batch_direct(self):
        data = self._get()
        if data is None:
            return
        if self.is_active():
            for b, q, c in zip(self._input_blob_names, self._queues, data):
                self._enqueue(b, q, c)

    def _enqueue_batch(self):
        '''
        This pulls data from the python-side queue and collects them
        into batch-sized pieces, unless dont_rebatch is set to true.
        '''
        if self._dont_rebatch:
            self._enqueue_batch_direct()
            return

        cur_batch = [np.array([]) for d in self._input_blob_names]
        first_batch_col = self._batch_columns[0]

        # Collect data until we have a full batch size
        while (
            cur_batch[0].shape[0] == 0 or
            cur_batch[0].shape[first_batch_col] < self._batch_size
        ) and self.is_active():
            chunk = self._get()
            if chunk is None:
                continue

            for j, chunk_elem in enumerate(chunk):
                if cur_batch[j].shape[0] == 0:
                    cur_batch[j] = chunk_elem.copy()
                else:
                    cur_batch[j] = np.append(
                        cur_batch[j], chunk_elem, axis=self._batch_columns[j]
                    )

        start_time = time.time()
        try:
            # Return data over the batch size back to queue
            if cur_batch[0].shape[0] > 0 and cur_batch[0].shape[
                first_batch_col
            ] > self._batch_size:
                leftover = []
                trimmed_batch = []
                for j, b in enumerate(cur_batch):
                    [c, l] = np.split(
                        b, [self._batch_size], axis=self._batch_columns[j]
                    )
                    leftover.append(l)
                    trimmed_batch.append(c)
                cur_batch = trimmed_batch
                try:
                    self._internal_queue.put(leftover, block=False)
                except Queue.Full:
                    pass

                assert cur_batch[0].shape[first_batch_col] == self._batch_size

            if self.is_active():
                for b, q, c in zip(
                    self._input_blob_names, self._queues, cur_batch
                ):
                    self._enqueue(b, q, c)
        finally:
            self.put_metric('enqueue_time', time.time() - start_time)

    def _init_scratch(self):
        self._scratch_blob = {}
        self._scratch_status = {}
        for blob_name in self._input_blob_names:
            scratch_name = self._namescope + blob_name + \
                "_scratch_" + self._input_source_name
            self._scratch_blob[blob_name] = core.BlobReference(scratch_name)
            self._scratch_status[blob_name] = core.BlobReference(
                scratch_name + "_status"
            )

        # Feed empty arrays to the scratch blobs here, so that there won't be
        # race conditions when calling FeedBlob (which calls wworkspace
        # CreateBlob()) from enqueue threads
        for b in chain(
            self._scratch_blob.values(), self._scratch_status.values()
        ):
            workspace.FeedBlob(
                b,
                np.array([]).astype(np.float32),
                device_option=self._device_option,
            )

    def _enqueue(self, blob_name, queue, data_arr):
        '''
        Enqueue the correctly sized batch arrays to Caffe2's queue.
        '''
        workspace.FeedBlob(
            self._scratch_blob[blob_name],
            data_arr,
            device_option=self._device_option
        )

        op = core.CreateOperator(
            "SafeEnqueueBlobs",
            [queue, self._scratch_blob[blob_name]],
            [self._scratch_blob[blob_name], self._scratch_status[blob_name]],
            device_option=self._device_option
        )
        workspace.RunOperatorOnce(op)

    def _create_caffe2_queues(self, net):
        '''
        Creates queues on caffe2 side
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
            q = create_queue(
                qname, num_blobs=1, capacity=self._c2_queue_capacity
            )
            self._queues.append(q)

    def _create_caffe2_ops(self, net):
        '''
        Creates dequeue-ops on caffe2 side
        '''
        for q, blob_name in zip(self._queues, self._input_blob_names):
            # Add operator to the Caffe2 network to dequeue
            net.DequeueBlobs(q, blob_name, timeout_secs=float(self._timeout))

    def _log_inputs_per_interval(self, inputs, force=False):
        self._inputs += inputs
        current_seconds = time.time()
        delta_seconds = current_seconds - self._prev_seconds
        if delta_seconds >= LOG_INT_SECS or force:
            inputs_per_sec = int(self._inputs / delta_seconds)
            qsize = self._internal_queue.qsize()
            print("{}/{}: {} inputs/sec".format(
                self._input_source_name,
                self._namescope,
                inputs_per_sec,
            ))
            print("-- queue: {} batches".format(qsize))
            # log and reset perf metrics
            self.put_metric('inputs_per_sec', inputs_per_sec, False)
            self.put_metric('queue_size', qsize, False)
            self.put_metric('time_elapsed', delta_seconds, False)
            self._log(self._metrics)
            self._reset_metrics()
            self._inputs = 0
            self._prev_seconds = current_seconds

    def _log(self, metrics):
        if not self._external_loggers:
            return
        for logger in self._external_loggers:
            try:
                logger.log(metrics)
            except Exception as e:
                print("Failed to call ExternalLogger: {}".format(e))

    def put_metric(self, key, value, count=True):
        self._metrics[key] += value
        if count:
            count_key = '{}_count'.format(key)
            self._metrics[count_key] += 1

    def _reset_metrics(self):
        self._metrics = collections.defaultdict(lambda: 0)


class GlobalCoordinator(object):
    def __init__(self):
        self._coordinators = []
        self._fetcher_id_seq = 0
        self._worker_ids = []
        self._queues = {}
        self.register_shutdown_handler()

    def add(self, coordinator):
        self._coordinators.append(coordinator)

    def get_new_worker_id(self):
        worker_id = self._fetcher_id_seq
        self._worker_ids.append(worker_id)
        self._fetcher_id_seq += 1
        return worker_id

    def get_worker_ids(self):
        return self._worker_ids

    def get_queue(self, queue_name, max_buffered_batches):
        assert isinstance(max_buffered_batches, int)
        if queue_name not in self._queues:
            self._queues[queue_name] = Queue.Queue(maxsize=max_buffered_batches)
        return self._queues[queue_name]

    def start(self):
        for c in self._coordinators:
            c.init(self)
            c._start()

    def reset_data_input(self, namescope, name, net, batch_size):
        log.info("Reset data input {}, batch size {}: ".format(name, batch_size))
        for c in self._coordinators:
            if c._input_source_name == name and c._namescope == namescope:
                c._batch_size = batch_size
                c._create_caffe2_ops(net)

    def stop(self):
        all_success = True
        for c in self._coordinators:
            c._stop()
        for c in self._coordinators:
            success = c._wait_finish()
            all_success = all_success and success
        self._coordinators = []
        return all_success

    def stop_coordinator(self, input_source_name):
        '''
        Stop a specific coordinator
        '''
        for c in self._coordinators:
            if c._input_source_name == input_source_name:
                c._stop()
                c._wait_finish()
        self._coordinators = [
            c for c in self._coordinators
            if c._input_source_name != input_source_name
        ]

    def register_shutdown_handler(self):
        def cleanup():
            self.stop()

        atexit.register(cleanup)


global_coordinator = GlobalCoordinator()


def fetcher(coordinator, worker_id, fetch_fun, batch_size, input_blob_names):
    while coordinator.is_active():
        start_time = time.time()
        try:
            input_data = fetch_fun(worker_id, batch_size)
            if input_data is None:
                print("Fetcher function returned None")
                continue

            assert len(input_data) == len(input_blob_names), \
                "Expecting data blob for each input"
            for d in input_data:
                assert isinstance(d, np.ndarray), \
                    "Fetcher function must return a numpy array"
            if not coordinator._dont_rebatch:
                j = 1
                for d in input_data[1:]:
                    assert d.shape[coordinator._batch_columns[j]] == input_data[0].shape[coordinator._batch_columns[0]], \
                        "Each returned input must have equal number of samples"
                    j += 1

            coordinator.put(input_data)
        except Exception as e:
            print(e)
            logging.exception("Exception in fetcher", e)
            coordinator._stop("Exception in fetcher {}: {}".format(
                worker_id, e
            ))
        finally:
            coordinator.put_metric('fetcher_time', time.time() - start_time)


def enqueuer(coordinator):
    while coordinator.is_active():
        coordinator._enqueue_batch()
