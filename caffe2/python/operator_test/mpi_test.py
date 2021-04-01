




from hypothesis import given
import hypothesis.strategies as st

import numpy as np
import unittest

from caffe2.python import core, workspace, dyndep
import caffe2.python.hypothesis_test_util as hu

dyndep.InitOpsLibrary("@/caffe2/caffe2/mpi:mpi_ops")

_has_mpi =False
COMM = None
RANK = 0
SIZE = 0

def SetupMPI():
    try:
        # pyre-fixme[21]: undefined import
        from mpi4py import MPI
        global _has_mpi, COMM, RANK, SIZE
        _has_mpi = core.IsOperatorWithEngine("CreateCommonWorld", "MPI")
        COMM = MPI.COMM_WORLD
        RANK = COMM.Get_rank()
        SIZE = COMM.Get_size()
    except ImportError:
        _has_mpi = False


@unittest.skipIf(not _has_mpi,
                 "MPI is not available. Skipping.")
class TestMPI(hu.HypothesisTestCase):
    @given(X=hu.tensor(),
           root=st.integers(min_value=0, max_value=SIZE - 1),
           device_option=st.sampled_from(hu.device_options),
           **hu.gcs)
    def test_broadcast(self, X, root, device_option, gc, dc):
        # Use mpi4py's broadcast to make sure that all nodes inherit the
        # same hypothesis test.
        X = COMM.bcast(X)
        root = COMM.bcast(root)
        device_option = COMM.bcast(device_option)
        X[:] = RANK
        self.assertTrue(
            workspace.RunOperatorOnce(
                core.CreateOperator(
                    "CreateCommonWorld", [], "comm", engine="MPI",
                    device_option=device_option)))
        self.assertTrue(workspace.FeedBlob("X", X, device_option))
        mpi_op = core.CreateOperator(
            "Broadcast", ["comm", "X"], "X", engine="MPI", root=root,
            device_option=device_option)
        self.assertTrue(workspace.RunOperatorOnce(mpi_op))
        new_X = workspace.FetchBlob("X")
        np.testing.assert_array_equal(new_X, root)
        workspace.ResetWorkspace()

    @given(X=hu.tensor(),
           root=st.integers(min_value=0, max_value=SIZE - 1),
           device_option=st.sampled_from(hu.device_options),
           **hu.gcs)
    def test_reduce(self, X, root, device_option, gc, dc):
        # Use mpi4py's broadcast to make sure that all nodes inherit the
        # same hypothesis test.
        X = COMM.bcast(X)
        root = COMM.bcast(root)
        device_option = COMM.bcast(device_option)
        X[:] = RANK
        self.assertTrue(
            workspace.RunOperatorOnce(
                core.CreateOperator(
                    "CreateCommonWorld", [], "comm", engine="MPI",
                    device_option=device_option)))
        self.assertTrue(workspace.FeedBlob("X", X, device_option))
        mpi_op = core.CreateOperator(
            "Reduce", ["comm", "X"], "X_reduced", engine="MPI", root=root,
            device_option=device_option)
        self.assertTrue(workspace.RunOperatorOnce(mpi_op))
        if (RANK == root):
            new_X = workspace.FetchBlob("X")
            np.testing.assert_array_equal(new_X, root)
        workspace.ResetWorkspace()

    @given(X=hu.tensor(),
           root=st.integers(min_value=0, max_value=SIZE - 1),
           device_option=st.sampled_from(hu.device_options),
           inplace=st.booleans(),
           **hu.gcs)
    def test_allreduce(self, X, root, device_option, inplace, gc, dc):
        # Use mpi4py's broadcast to make sure that all nodes inherit the
        # same hypothesis test.
        X = COMM.bcast(X)
        root = COMM.bcast(root)
        device_option = COMM.bcast(device_option)
        inplace = COMM.bcast(inplace)
        X[:] = RANK
        self.assertTrue(
            workspace.RunOperatorOnce(
                core.CreateOperator(
                    "CreateCommonWorld", [], "comm", engine="MPI",
                    device_option=device_option)))
        # Use mpi4py's broadcast to make sure that all copies have the same
        # tensor size.
        X = COMM.bcast(X)
        X[:] = RANK
        self.assertTrue(workspace.FeedBlob("X", X, device_option))
        mpi_op = core.CreateOperator(
            "Allreduce", ["comm", "X"],
            "X" if inplace else "X_reduced",
            engine="MPI", root=root,
            device_option=device_option)
        self.assertTrue(workspace.RunOperatorOnce(mpi_op))
        new_X = workspace.FetchBlob("X" if inplace else "X_reduced")
        np.testing.assert_array_equal(new_X, SIZE * (SIZE - 1) / 2)
        workspace.ResetWorkspace()

    @given(X=hu.tensor(),
           device_option=st.sampled_from(hu.device_options),
           specify_send_blob=st.booleans(),
           specify_recv_blob=st.booleans(),
           **hu.gcs)
    def test_sendrecv(
            self, X, device_option, specify_send_blob, specify_recv_blob,
            gc, dc):
        # Use mpi4py's broadcast to make sure that all nodes inherit the
        # same hypothesis test.
        X = COMM.bcast(X)
        device_option = COMM.bcast(device_option)
        specify_send_blob = COMM.bcast(specify_send_blob)
        specify_recv_blob = COMM.bcast(specify_recv_blob)
        X[:] = RANK

        self.assertTrue(
            workspace.RunOperatorOnce(
                core.CreateOperator(
                    "CreateCommonWorld", [], "comm", engine="MPI",
                    device_option=device_option)))
        self.assertTrue(workspace.FeedBlob("X", X, device_option))
        for src in range(SIZE):
            for dst in range(SIZE):
                tag = src * SIZE + dst
                if src == dst:
                    continue
                elif RANK == src:
                    X[:] = RANK
                    self.assertTrue(workspace.FeedBlob("X", X, device_option))
                    if specify_send_blob:
                        self.assertTrue(workspace.FeedBlob(
                            "dst", np.array(dst, dtype=np.int32)))
                        self.assertTrue(workspace.FeedBlob(
                            "tag", np.array(tag, dtype=np.int32)))
                        mpi_op = core.CreateOperator(
                            "SendTensor", ["comm", "X", "dst", "tag"], [],
                            engine="MPI", raw_buffer=True,
                            device_option=device_option)
                    else:
                        mpi_op = core.CreateOperator(
                            "SendTensor", ["comm", "X"], [], engine="MPI",
                            dst=dst, tag=tag, raw_buffer=True,
                            device_option=device_option)
                    self.assertTrue(workspace.RunOperatorOnce(mpi_op))
                elif RANK == dst:
                    if specify_recv_blob:
                        self.assertTrue(workspace.FeedBlob(
                            "src", np.array(src, dtype=np.int32)))
                        self.assertTrue(workspace.FeedBlob(
                            "tag", np.array(tag, dtype=np.int32)))
                        mpi_op = core.CreateOperator(
                            "ReceiveTensor", ["comm", "X", "src", "tag"],
                            ["X", "src", "tag"],
                            engine="MPI",
                            src=src, tag=tag, raw_buffer=True,
                            device_option=device_option)
                    else:
                        mpi_op = core.CreateOperator(
                            "ReceiveTensor", ["comm", "X"], ["X", "src", "tag"],
                            engine="MPI",
                            src=src, tag=tag, raw_buffer=True,
                            device_option=device_option)
                    self.assertTrue(workspace.RunOperatorOnce(mpi_op))
                    received = workspace.FetchBlob("X")
                    np.testing.assert_array_equal(received, src)
                    src_blob = workspace.FetchBlob("src")
                    np.testing.assert_array_equal(src_blob, src)
                    tag_blob = workspace.FetchBlob("tag")
                    np.testing.assert_array_equal(tag_blob, tag)
                # simply wait for the guys to finish
                COMM.barrier()
        workspace.ResetWorkspace()

if __name__ == "__main__":
    SetupMPI()
    import unittest
    unittest.main()
