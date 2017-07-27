#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from hypothesis import given
import hypothesis.strategies as st
from multiprocessing import Process, Queue

import numpy as np
import os
import pickle
import tempfile
import shutil

from caffe2.python import core, workspace, dyndep
import caffe2.python.hypothesis_test_util as hu

dyndep.InitOpsLibrary("@/caffe2/caffe2/distributed:file_store_handler_ops")
dyndep.InitOpsLibrary("@/caffe2/caffe2/distributed:redis_store_handler_ops")
dyndep.InitOpsLibrary("@/caffe2/caffe2/distributed:store_ops")
dyndep.InitOpsLibrary("@/caffe2/caffe2/contrib/gloo:gloo_ops")
dyndep.InitOpsLibrary("@/caffe2/caffe2/contrib/gloo:gloo_ops_gpu")

op_engine = 'GLOO'


class TemporaryDirectory:
    def __enter__(self):
        self.tmpdir = tempfile.mkdtemp()
        return self.tmpdir

    def __exit__(self, type, value, traceback):
        shutil.rmtree(self.tmpdir)


class TestCase(hu.HypothesisTestCase):
    test_counter = 0
    sync_counter = 0

    def run_test_locally(self, fn, device_option=None, **kwargs):
        # Queue for assertion errors on subprocesses
        queue = Queue()

        # Capture any exception thrown by the subprocess
        def run_fn(*args, **kwargs):
            try:
                with core.DeviceScope(device_option):
                    fn(*args, **kwargs)
                    workspace.ResetWorkspace()
                    queue.put(True)
            except Exception as ex:
                queue.put(ex)

        # Start N processes in the background
        procs = []
        for i in range(kwargs['comm_size']):
            kwargs['comm_rank'] = i
            proc = Process(
                target=run_fn,
                kwargs=kwargs)
            proc.start()
            procs.append(proc)

        # Test complete, join background processes
        while len(procs) > 0:
            proc = procs.pop(0)
            while proc.is_alive():
                proc.join(10)

                # Raise exception if we find any. Otherwise each worker
                # should put a True into the queue
                # Note that the following is executed ALSO after
                # the last process was joined, so if ANY exception
                # was raised, it will be re-raised here.
                self.assertFalse(queue.empty(), "Job failed without a result")
                o = queue.get()
                if isinstance(o, Exception):
                    raise o
                else:
                    self.assertTrue(o)

    def run_test_distributed(self, fn, device_option=None, **kwargs):
        comm_rank = os.getenv('COMM_RANK')
        self.assertIsNotNone(comm_rank)
        comm_size = os.getenv('COMM_SIZE')
        self.assertIsNotNone(comm_size)
        kwargs['comm_rank'] = int(comm_rank)
        kwargs['comm_size'] = int(comm_size)
        with core.DeviceScope(device_option):
            fn(**kwargs)
            workspace.ResetWorkspace()

    def create_common_world(self, comm_rank, comm_size, tmpdir=None, existing_cw=None):
        store_handler = "store_handler"

        # If REDIS_HOST is set, use RedisStoreHandler for rendezvous.
        if existing_cw is None:
            redis_host = os.getenv("REDIS_HOST")
            redis_port = int(os.getenv("REDIS_PORT", 6379))
            if redis_host is not None:
                workspace.RunOperatorOnce(
                    core.CreateOperator(
                        "RedisStoreHandlerCreate",
                        [],
                        [store_handler],
                        prefix=str(TestCase.test_counter) + "/",
                        host=redis_host,
                        port=redis_port))
            else:
                workspace.RunOperatorOnce(
                    core.CreateOperator(
                        "FileStoreHandlerCreate",
                        [],
                        [store_handler],
                        path=tmpdir))
            common_world = "common_world"
        else:
            common_world = str(existing_cw) + ".forked"

        inputs = [store_handler]
        if existing_cw is not None:
            inputs.append(existing_cw)
        workspace.RunOperatorOnce(
            core.CreateOperator(
                "CreateCommonWorld",
                inputs,
                [common_world],
                size=comm_size,
                rank=comm_rank,
                sync=True,
                engine=op_engine))
        return (store_handler, common_world)

    def synchronize(self, store_handler, value, comm_rank=None):
        TestCase.sync_counter += 1
        blob = "sync_{}".format(TestCase.sync_counter)
        if comm_rank == 0:
            workspace.FeedBlob(blob, pickle.dumps(value))
            workspace.RunOperatorOnce(
                core.CreateOperator(
                    "StoreSet",
                    [store_handler, blob],
                    []))
        else:
            workspace.RunOperatorOnce(
                core.CreateOperator(
                    "StoreGet",
                    [store_handler],
                    [blob]))
        return pickle.loads(workspace.FetchBlob(blob))

    def _test_broadcast(self,
                        comm_rank=None,
                        comm_size=None,
                        blob_size=None,
                        num_blobs=None,
                        tmpdir=None,
                        use_float16=False,
                        ):
        store_handler, common_world = self.create_common_world(
            comm_rank=comm_rank,
            comm_size=comm_size,
            tmpdir=tmpdir)

        blob_size = self.synchronize(
            store_handler,
            blob_size,
            comm_rank=comm_rank)

        num_blobs = self.synchronize(
            store_handler,
            num_blobs,
            comm_rank=comm_rank)

        for i in range(comm_size):
            blobs = []
            for j in range(num_blobs):
                blob = "blob_{}".format(j)
                offset = (comm_rank * num_blobs) + j
                value = np.full(blob_size, offset,
                                np.float16 if use_float16 else np.float32)
                workspace.FeedBlob(blob, value)
                blobs.append(blob)

            net = core.Net("broadcast")
            net.Broadcast(
                [common_world] + blobs,
                blobs,
                root=i,
                engine=op_engine)

            workspace.CreateNet(net)
            workspace.RunNet(net.Name())

            for j in range(num_blobs):
                np.testing.assert_array_equal(
                    workspace.FetchBlob(blobs[j]),
                    i * num_blobs)

            # Run the net a few more times to check the operator
            # works not just the first time it's called
            for _tmp in range(4):
                workspace.RunNet(net.Name())

    @given(comm_size=st.integers(min_value=2, max_value=8),
           blob_size=st.integers(min_value=1e3, max_value=1e6),
           num_blobs=st.integers(min_value=1, max_value=4),
           device_option=st.sampled_from([hu.cpu_do]),
           use_float16=st.booleans())
    def test_broadcast(self, comm_size, blob_size, num_blobs, device_option,
                       use_float16):
        TestCase.test_counter += 1
        if os.getenv('COMM_RANK') is not None:
            self.run_test_distributed(
                self._test_broadcast,
                blob_size=blob_size,
                num_blobs=num_blobs,
                use_float16=use_float16,
                device_option=device_option)
        else:
            with TemporaryDirectory() as tmpdir:
                self.run_test_locally(
                    self._test_broadcast,
                    comm_size=comm_size,
                    blob_size=blob_size,
                    num_blobs=num_blobs,
                    device_option=device_option,
                    tmpdir=tmpdir,
                    use_float16=use_float16)

    def _test_allreduce(self,
                        comm_rank=None,
                        comm_size=None,
                        blob_size=None,
                        num_blobs=None,
                        tmpdir=None,
                        use_float16=False
                        ):
        store_handler, common_world = self.create_common_world(
            comm_rank=comm_rank,
            comm_size=comm_size,
            tmpdir=tmpdir)

        blob_size = self.synchronize(
            store_handler,
            blob_size,
            comm_rank=comm_rank)

        num_blobs = self.synchronize(
            store_handler,
            num_blobs,
            comm_rank=comm_rank)

        blobs = []
        for i in range(num_blobs):
            blob = "blob_{}".format(i)
            value = np.full(blob_size, (comm_rank * num_blobs) + i,
                            np.float16 if use_float16 else np.float32)
            workspace.FeedBlob(blob, value)
            blobs.append(blob)

        net = core.Net("allreduce")
        net.Allreduce(
            [common_world] + blobs,
            blobs,
            engine=op_engine)

        workspace.CreateNet(net)
        workspace.RunNet(net.Name())

        for i in range(num_blobs):
            np.testing.assert_array_equal(
                workspace.FetchBlob(blobs[i]),
                (num_blobs * comm_size) * (num_blobs * comm_size - 1) / 2)

        # Run the net a few more times to check the operator
        # works not just the first time it's called
        for _tmp in range(4):
            workspace.RunNet(net.Name())

    def _test_allreduce_multicw(self,
                                comm_rank=None,
                                comm_size=None,
                                tmpdir=None
                                ):
        _store_handler, common_world = self.create_common_world(
            comm_rank=comm_rank,
            comm_size=comm_size,
            tmpdir=tmpdir)

        _, common_world2 = self.create_common_world(
            comm_rank=comm_rank,
            comm_size=comm_size,
            tmpdir=tmpdir,
            existing_cw=common_world)

        blob_size = 1e4
        num_blobs = 4

        for cw in [common_world, common_world2]:
            blobs = []
            for i in range(num_blobs):
                blob = "blob_{}".format(i)
                value = np.full(blob_size, (comm_rank * num_blobs) + i, np.float32)
                workspace.FeedBlob(blob, value)
                blobs.append(blob)

            net = core.Net("allreduce_multicw")
            net.Allreduce(
                [cw] + blobs,
                blobs,
                engine=op_engine)

            workspace.RunNetOnce(net)
            for i in range(num_blobs):
                np.testing.assert_array_equal(
                    workspace.FetchBlob(blobs[i]),
                    (num_blobs * comm_size) * (num_blobs * comm_size - 1) / 2)

    @given(comm_size=st.integers(min_value=2, max_value=8),
           blob_size=st.integers(min_value=1e3, max_value=1e6),
           num_blobs=st.integers(min_value=1, max_value=4),
           device_option=st.sampled_from([hu.cpu_do]),
           use_float16=st.booleans())
    def test_allreduce(self, comm_size, blob_size, num_blobs, device_option,
                       use_float16):
        TestCase.test_counter += 1
        if os.getenv('COMM_RANK') is not None:
            self.run_test_distributed(
                self._test_allreduce,
                blob_size=blob_size,
                num_blobs=num_blobs,
                use_float16=use_float16,
                device_option=device_option)
        else:
            with TemporaryDirectory() as tmpdir:
                self.run_test_locally(
                    self._test_allreduce,
                    comm_size=comm_size,
                    blob_size=blob_size,
                    num_blobs=num_blobs,
                    device_option=device_option,
                    tmpdir=tmpdir,
                    use_float16=use_float16)

    @given(device_option=st.sampled_from([hu.cpu_do]))
    def test_forked_cw(self, device_option):
        TestCase.test_counter += 1
        if os.getenv('COMM_RANK') is not None:
            self.run_test_distributed(
                self._test_allreduce_multicw,
                device_option=device_option)
        else:
            with TemporaryDirectory() as tmpdir:
                self.run_test_locally(
                    self._test_allreduce_multicw,
                    comm_size=8,
                    device_option=device_option,
                    tmpdir=tmpdir)

    def _test_barrier(
        self,
        comm_rank=None,
        comm_size=None,
        tmpdir=None,
    ):
        store_handler, common_world = self.create_common_world(
            comm_rank=comm_rank, comm_size=comm_size, tmpdir=tmpdir
        )

        net = core.Net("barrier")
        net.Barrier(
            [common_world],
            [],
            engine=op_engine)

        workspace.CreateNet(net)
        workspace.RunNet(net.Name())

        # Run the net a few more times to check the operator
        # works not just the first time it's called
        for _tmp in range(4):
            workspace.RunNet(net.Name())

    @given(comm_size=st.integers(min_value=2, max_value=8),
           device_option=st.sampled_from([hu.cpu_do]))
    def test_barrier(self, comm_size, device_option):
        TestCase.test_counter += 1
        if os.getenv('COMM_RANK') is not None:
            self.run_test_distributed(
                self._test_broadcast,
                device_option=device_option)
        else:
            with TemporaryDirectory() as tmpdir:
                self.run_test_locally(
                    self._test_barrier,
                    comm_size=comm_size,
                    device_option=device_option,
                    tmpdir=tmpdir)

if __name__ == "__main__":
    import unittest
    unittest.main()
