#!/usr/bin/env python2

from multiprocessing import Process
import numpy as np
import socket
import unittest
import zmq

from caffe2.python import core, workspace, utils, test_util
from caffe2.proto import caffe2_pb2

def PickUnusedPort():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('localhost', 0))
    addr, port = s.getsockname()
    s.close()
    # There is a slight chance that the socket get acquired in between this
    # close and the next open, but this seems to be reliable enough for
    # testing.
    return port

def Produce(max_item, port):
    context = zmq.Context()
    zmq_socket = context.socket(zmq.PUSH)
    zmq_socket.bind("tcp://127.0.0.1:{}".format(port))
    for idx in range(max_item):
        zmq_socket.send("key_{}".format(idx))
        protos = caffe2_pb2.TensorProtos()
        protos.protos.extend(
            [utils.NumpyArrayToCaffe2Tensor(np.ones((1, 10)) * idx, "data")])
        zmq_socket.send(protos.SerializeToString())
    zmq_socket.close()

class ZmqSocketTest(test_util.TestCase):

    def testProduceFunction(self):
        """Test to make sure the Python harness runs correctly."""
        port = PickUnusedPort()
        proc = Process(target=Produce, args=(10, port))
        proc.start()
        context = zmq.Context()
        zmq_socket = context.socket(zmq.PULL)
        zmq_socket.connect("tcp://127.0.0.1:{}".format(port))
        for idx in range(10):
            key = zmq_socket.recv()
            zmq_socket.recv()  # Receiving value.
            self.assertEqual(key, "key_{}".format(idx))
        proc.join()

    def testZmqDB(self):
        """Tests ZmqDB by feeding from Python and loading from C++."""
        port = PickUnusedPort()
        proc = Process(target=Produce, args=(10, port))
        proc.start()
        zmq_net = core.Net("zmq")
        reader = zmq_net.DBReader([], "reader",
                                  db="tcp://127.0.0.1:{}".format(port),
                                  db_type="zmqdb")
        zmq_net.TensorProtosDBInput(
            reader, "data", batch_size=1,
            db="tcp://127.0.0.1:{}".format(port),
            db_type="zmqdb")
        workspace.CreateNet(zmq_net)
        for i in range(5):
            workspace.RunNet("zmq")
            arr = workspace.FetchBlob("data")
            self.assertEqual(arr.shape, (1, 10))
            np.testing.assert_array_almost_equal(arr, i)
        # Wrap up the testing environment.
        proc.join(timeout=1)


if __name__ == "__main__":
    unittest.main()
