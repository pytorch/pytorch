#!/usr/bin/env python






from hypothesis import given, settings
import hypothesis.strategies as st
from multiprocessing import Process, Queue

import numpy as np
import os
import pickle
import tempfile
import shutil

from caffe2.python import core, workspace, dyndep
import caffe2.python.hypothesis_test_util as hu
from gloo.python import IoError

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

        if existing_cw is not None:
            workspace.RunOperatorOnce(
                core.CreateOperator(
                    "CloneCommonWorld",
                    [existing_cw],
                    [common_world],
                    sync=True,
                    engine=op_engine))
        else:
            workspace.RunOperatorOnce(
                core.CreateOperator(
                    "CreateCommonWorld",
                    [store_handler],
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
           blob_size=st.integers(min_value=int(1e3), max_value=int(1e6)),
           num_blobs=st.integers(min_value=1, max_value=4),
           device_option=st.sampled_from([hu.cpu_do]),
           use_float16=st.booleans())
    @settings(deadline=10000)
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

        blob_size = int(1e4)
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
           blob_size=st.integers(min_value=int(1e3), max_value=int(1e6)),
           num_blobs=st.integers(min_value=1, max_value=4),
           device_option=st.sampled_from([hu.cpu_do]),
           use_float16=st.booleans())
    @settings(deadline=10000)
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

    def _test_reduce_scatter(self,
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

        # Specify distribution among ranks i.e. number of elements
        # scattered/distributed to each process.
        recv_counts = np.zeros(comm_size, dtype=np.int32)
        remaining = blob_size
        chunk_size = (blob_size + comm_size - 1) / comm_size
        for i in range(comm_size):
            recv_counts[i] = min(chunk_size, remaining)
            remaining = remaining - chunk_size if remaining > chunk_size else 0
        recv_counts_blob = "recvCounts"
        workspace.FeedBlob(recv_counts_blob, recv_counts)
        blobs.append(recv_counts_blob)

        net = core.Net("reduce_scatter")
        net.ReduceScatter(
            [common_world] + blobs,
            blobs,
            engine=op_engine)

        workspace.CreateNet(net)
        workspace.RunNet(net.Name())

        for i in range(num_blobs):
            np.testing.assert_array_equal(
                np.resize(workspace.FetchBlob(blobs[i]), recv_counts[comm_rank]),
                (num_blobs * comm_size) * (num_blobs * comm_size - 1) / 2)

        # Run the net a few more times to check the operator
        # works not just the first time it's called
        for _tmp in range(4):
            workspace.RunNet(net.Name())

    @given(comm_size=st.integers(min_value=2, max_value=8),
           blob_size=st.integers(min_value=int(1e3), max_value=int(1e6)),
           num_blobs=st.integers(min_value=1, max_value=4),
           device_option=st.sampled_from([hu.cpu_do]),
           use_float16=st.booleans())
    @settings(deadline=10000)
    def test_reduce_scatter(self, comm_size, blob_size, num_blobs,
                            device_option, use_float16):
        TestCase.test_counter += 1
        if os.getenv('COMM_RANK') is not None:
            self.run_test_distributed(
                self._test_reduce_scatter,
                blob_size=blob_size,
                num_blobs=num_blobs,
                use_float16=use_float16,
                device_option=device_option)
        else:
            with TemporaryDirectory() as tmpdir:
                self.run_test_locally(
                    self._test_reduce_scatter,
                    comm_size=comm_size,
                    blob_size=blob_size,
                    num_blobs=num_blobs,
                    device_option=device_option,
                    tmpdir=tmpdir,
                    use_float16=use_float16)

    def _test_allgather(self,
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

        net = core.Net("allgather")
        net.Allgather(
            [common_world] + blobs,
            ["Gathered"],
            engine=op_engine)

        workspace.CreateNet(net)
        workspace.RunNet(net.Name())
        # create expected output
        expected_output = np.array([])
        for i in range(comm_size):
            for j in range(num_blobs):
                value = np.full(blob_size, (i * num_blobs) + j,
                                np.float16 if use_float16 else np.float32)
                expected_output = np.concatenate((expected_output, value))
        np.testing.assert_array_equal(
            workspace.FetchBlob("Gathered"), expected_output)

        # Run the net a few more times to check the operator
        # works not just the first time it's called
        for _tmp in range(4):
            workspace.RunNet(net.Name())

    @given(comm_size=st.integers(min_value=2, max_value=8),
           blob_size=st.integers(min_value=int(1e3), max_value=int(1e6)),
           num_blobs=st.integers(min_value=1, max_value=4),
           device_option=st.sampled_from([hu.cpu_do]),
           use_float16=st.booleans())
    @settings(max_examples=10, deadline=None)
    def test_allgather(self, comm_size, blob_size, num_blobs, device_option,
                       use_float16):
        TestCase.test_counter += 1
        if os.getenv('COMM_RANK') is not None:
            self.run_test_distributed(
                self._test_allgather,
                blob_size=blob_size,
                num_blobs=num_blobs,
                use_float16=use_float16,
                device_option=device_option)
        else:
            with TemporaryDirectory() as tmpdir:
                self.run_test_locally(
                    self._test_allgather,
                    comm_size=comm_size,
                    blob_size=blob_size,
                    num_blobs=num_blobs,
                    device_option=device_option,
                    tmpdir=tmpdir,
                    use_float16=use_float16)

    @given(device_option=st.sampled_from([hu.cpu_do]))
    @settings(deadline=10000)
    def test_forked_cw(self, device_option):
        TestCase.test_counter += 1
        if os.getenv('COMM_RANK') is not None:
            self.run_test_distributed(
                self._test_allreduce_multicw,
                device_option=device_option)
        else:
            # Note: this test exercises the path where we fork a common world.
            # We therefore don't need a comm size larger than 2. It used to be
            # run with comm_size=8, which causes flaky results in a stress run.
            # The flakiness was caused by too many listening sockets being
            # created by Gloo context initialization (8 processes times
            # 7 sockets times 20-way concurrency, plus TIME_WAIT).
            with TemporaryDirectory() as tmpdir:
                self.run_test_locally(
                    self._test_allreduce_multicw,
                    comm_size=2,
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
    @settings(deadline=10000)
    def test_barrier(self, comm_size, device_option):
        TestCase.test_counter += 1
        if os.getenv('COMM_RANK') is not None:
            self.run_test_distributed(
                self._test_barrier,
                device_option=device_option)
        else:
            with TemporaryDirectory() as tmpdir:
                self.run_test_locally(
                    self._test_barrier,
                    comm_size=comm_size,
                    device_option=device_option,
                    tmpdir=tmpdir)

    def _test_close_connection(
        self,
        comm_rank=None,
        comm_size=None,
        tmpdir=None,
    ):
        '''
        One node calls close connection, others wait it on barrier.
        Test will check that all will exit eventually.
        '''
        # Caffe's for closers only:
        # https://www.youtube.com/watch?v=QMFwFgG9NE8
        closer = comm_rank == comm_size // 2,

        store_handler, common_world = self.create_common_world(
            comm_rank=comm_rank, comm_size=comm_size, tmpdir=tmpdir
        )

        net = core.Net("barrier_or_close")
        if not closer:
            net.Barrier(
                [common_world],
                [],
                engine=op_engine)
        else:
            net.DestroyCommonWorld(
                [common_world], [common_world], engine=op_engine)
            # Sleep a bit to ensure others start the barrier
            import time
            time.sleep(0.1)

        workspace.CreateNet(net)
        workspace.RunNet(net.Name())

    @given(comm_size=st.integers(min_value=2, max_value=8),
           device_option=st.sampled_from([hu.cpu_do]))
    @settings(deadline=10000)
    def test_close_connection(self, comm_size, device_option):
        import time
        start_time = time.time()
        TestCase.test_counter += 1
        if os.getenv('COMM_RANK') is not None:
            self.run_test_distributed(
                self._test_close_connection,
                device_option=device_option)
        else:
            with TemporaryDirectory() as tmpdir:
                self.run_test_locally(
                    self._test_close_connection,
                    comm_size=comm_size,
                    device_option=device_option,
                    tmpdir=tmpdir)
        # Check that test finishes quickly because connections get closed.
        # This assert used to check that the end to end runtime was less
        # than 2 seconds, but this may not always be the case if there
        # is significant overhead in starting processes. Ideally, this
        # assert is replaced by one that doesn't depend on time but rather
        # checks the success/failure status of the barrier that is run.
        self.assertLess(time.time() - start_time, 20.0)

    def _test_io_error(
        self,
        comm_rank=None,
        comm_size=None,
        tmpdir=None,
    ):
        '''
        Only one node will participate in allreduce, resulting in an IoError
        '''
        store_handler, common_world = self.create_common_world(
            comm_rank=comm_rank,
            comm_size=comm_size,
            tmpdir=tmpdir)

        if comm_rank == 0:
            blob_size = 1000
            num_blobs = 1

            blobs = []
            for i in range(num_blobs):
                blob = "blob_{}".format(i)
                value = np.full(
                    blob_size, (comm_rank * num_blobs) + i, np.float32
                )
                workspace.FeedBlob(blob, value)
                blobs.append(blob)

            net = core.Net("allreduce")
            net.Allreduce(
                [common_world] + blobs,
                blobs,
                engine=op_engine)

            workspace.CreateNet(net)
            workspace.RunNet(net.Name())

    @given(comm_size=st.integers(min_value=2, max_value=8),
           device_option=st.sampled_from([hu.cpu_do]))
    @settings(deadline=10000)
    def test_io_error(self, comm_size, device_option):
        TestCase.test_counter += 1
        with self.assertRaises(IoError):
            if os.getenv('COMM_RANK') is not None:
                self.run_test_distributed(
                    self._test_io_error,
                    device_option=device_option)
            else:
                with TemporaryDirectory() as tmpdir:
                    self.run_test_locally(
                        self._test_io_error,
                        comm_size=comm_size,
                        device_option=device_option,
                        tmpdir=tmpdir)

if __name__ == "__main__":
    import unittest
    unittest.main()
