## @package store_ops_test_util
# Module caffe2.distributed.store_ops_test_util
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from multiprocessing import Process, Queue

import numpy as np

from caffe2.python import core, workspace


class StoreOpsTests(object):
    @classmethod
    def _test_set_get(cls, queue, create_store_handler_fn, index, num_procs):
        store_handler = create_store_handler_fn()
        blob = "blob"
        value = np.full(1, 1, np.float32)

        # Use last process to set blob to make sure other processes
        # are waiting for the blob before it is set.
        if index == (num_procs - 1):
            workspace.FeedBlob(blob, value)
            workspace.RunOperatorOnce(
                core.CreateOperator(
                    "StoreSet",
                    [store_handler, blob],
                    [],
                    blob_name=blob))

        output_blob = "output_blob"
        workspace.RunOperatorOnce(
            core.CreateOperator(
                "StoreGet",
                [store_handler],
                [output_blob],
                blob_name=blob))

        try:
            np.testing.assert_array_equal(workspace.FetchBlob(output_blob), 1)
        except AssertionError as err:
            queue.put(err)

        workspace.ResetWorkspace()

    @classmethod
    def test_set_get(cls, create_store_handler_fn):
        # Queue for assertion errors on subprocesses
        queue = Queue()

        # Start N processes in the background
        num_procs = 4
        procs = []
        for index in range(num_procs):
            proc = Process(
                target=cls._test_set_get,
                args=(queue, create_store_handler_fn, index, num_procs, ))
            proc.start()
            procs.append(proc)

        # Test complete, join background processes
        for proc in procs:
            proc.join()

        # Raise first error we find, if any
        if not queue.empty():
            raise queue.get()

    @classmethod
    def test_get_timeout(cls, create_store_handler_fn):
        store_handler = create_store_handler_fn()
        net = core.Net('get_missing_blob')
        net.StoreGet([store_handler], 1, blob_name='blob')
        workspace.RunNetOnce(net)
