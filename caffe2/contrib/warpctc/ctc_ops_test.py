from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from caffe2.proto import caffe2_pb2

from caffe2.python import core, workspace, dyndep, test_util

dyndep.InitOpsLibrary('@/caffe2/caffe2/contrib/warpctc:ctc_ops')
workspace.GlobalInit(["python"])


def softmax(w):
    maxes = np.amax(w, axis=-1, keepdims=True)
    e = np.exp(w - maxes)
    dist = e / np.sum(e, axis=-1, keepdims=True)
    return dist


class CTCOpsTest(test_util.TestCase):
    def verify_cost(self, device_option, is_test):
        alphabet_size = 5
        N = 1
        T = 2

        inputs = np.asarray(
            [
                [[0.1, 0.6, 0.1, 0.1, 0.1]],
                [[0.1, 0.1, 0.6, 0.1, 0.1]],
            ]
        ).reshape(T, N, alphabet_size).astype(np.float32)

        labels = np.asarray([1, 2]).astype(np.int32).reshape(T)
        label_lengths = np.asarray([2]).astype(np.int32).reshape(N)
        input_lengths = np.asarray([T]).astype(np.int32)

        net = core.Net("test-net")
        output_blobs = ["costs", "workspace"] if is_test \
                else ["inputs_grad_to_be_copied", "costs", "workspace"]
        net.CTC(["inputs", "labels", "label_lengths", "input_lengths"],
                output_blobs,
                is_test=is_test,
                device_option=device_option)
        if not is_test:
            net.AddGradientOperators(["costs"])
        self.ws.create_blob("inputs").feed(inputs, device_option=device_option)
        self.ws.create_blob("labels").feed(labels)
        self.ws.create_blob("label_lengths").feed(label_lengths)
        self.ws.create_blob("input_lengths").feed(input_lengths)
        self.ws.run(net)
        probs = softmax(inputs)
        expected = probs[0, 0, 1] * probs[1, 0, 2]
        self.assertEqual(self.ws.blobs["costs"].fetch().shape, (N,))
        self.assertEqual(self.ws.blobs["costs"].fetch().dtype, np.float32)
        cost = self.ws.blobs["costs"].fetch()[0]
        print(cost)
        self.assertAlmostEqual(np.exp(-cost), expected)
        if not is_test:
            # Make sure inputs_grad was added by AddGradientOperators and
            # it is equal to the inputs_grad_to_be_copied blob returned by CTCop
            assert np.array_equal(
                self.ws.blobs["inputs_grad"].fetch(),
                self.ws.blobs["inputs_grad_to_be_copied"].fetch()
            )

    def test_ctc_cost_cpu(self):
        self.verify_cost(
            caffe2_pb2.DeviceOption(device_type=caffe2_pb2.CPU),
            is_test=False)

    def test_ctc_cost_gpu(self):
        self.verify_cost(
            caffe2_pb2.DeviceOption(device_type=caffe2_pb2.CUDA,
                                    cuda_gpu_id=0),
            is_test=False)

    def test_ctc_forward_only_cpu(self):
        self.verify_cost(
            caffe2_pb2.DeviceOption(device_type=caffe2_pb2.CPU),
            is_test=True)

    def test_ctc_forward_only_gpu(self):
        self.verify_cost(
            caffe2_pb2.DeviceOption(device_type=caffe2_pb2.CUDA,
                                    cuda_gpu_id=0),
            is_test=True)
