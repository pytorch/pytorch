# This a large test that goes through the translation of the bvlc caffenet
# model, runs an example through the whole model, and verifies numerically
# that all the results look right. In default, it is disabled unless you
# explicitly want to run it.

from google.protobuf import text_format
import numpy as np
import os
import sys

CAFFE_FOUND = False
try:
    from caffe.proto import caffe_pb2
    from caffe2.python import caffe_translator
    CAFFE_FOUND = True
except Exception as e:
    # Safeguard so that we only catch the caffe module not found exception.
    if ("'caffe'" in str(e)):
        print(
            "PyTorch/Caffe2 now requires a separate installation of caffe. "
            "Right now, this is not found, so we will skip the caffe "
            "translator test.")

from caffe2.python import utils, workspace, test_util
import unittest


@unittest.skipIf(not CAFFE_FOUND,
                 'No Caffe installation found.')
@unittest.skipIf(not os.path.exists('data/testdata/caffe_translator'),
                 'No testdata existing for the caffe translator test. Exiting.')
class TestNumericalEquivalence(test_util.TestCase):
    @classmethod
    def setUpClass(cls):
        # We will do all the computation stuff in the global space.
        caffenet = caffe_pb2.NetParameter()
        caffenet_pretrained = caffe_pb2.NetParameter()
        with open('data/testdata/caffe_translator/deploy.prototxt') as f:
            text_format.Merge(f.read(), caffenet)
        with open('data/testdata/caffe_translator/'
                  'bvlc_reference_caffenet.caffemodel') as f:
            caffenet_pretrained.ParseFromString(f.read())
        for remove_legacy_pad in [True, False]:
            net, pretrained_params = caffe_translator.TranslateModel(
                caffenet, caffenet_pretrained, is_test=True,
                remove_legacy_pad=remove_legacy_pad
            )
            with open('data/testdata/caffe_translator/'
                      'bvlc_reference_caffenet.translatedmodel',
                      'w') as fid:
                fid.write(str(net))
            for param in pretrained_params.protos:
                workspace.FeedBlob(param.name, utils.Caffe2TensorToNumpyArray(param))
            # Let's also feed in the data from the Caffe test code.
            data = np.load('data/testdata/caffe_translator/data_dump.npy').astype(
                np.float32)
            workspace.FeedBlob('data', data)
            # Actually running the test.
            workspace.RunNetOnce(net.SerializeToString())

    def testBlobs(self):
        names = [
            "conv1", "pool1", "norm1", "conv2", "pool2", "norm2", "conv3",
            "conv4", "conv5", "pool5", "fc6", "fc7", "fc8", "prob"
        ]
        for name in names:
            print('Verifying {}'.format(name))
            caffe2_result = workspace.FetchBlob(name)
            reference = np.load(
                'data/testdata/caffe_translator/' + name + '_dump.npy'
            )
            self.assertEqual(caffe2_result.shape, reference.shape)
            scale = np.max(caffe2_result)
            np.testing.assert_almost_equal(
                caffe2_result / scale,
                reference / scale,
                decimal=5
            )


@unittest.skipIf(not CAFFE_FOUND, 'No Caffe installation found.')
class TestTranslateReluLayer(test_util.TestCase):
    def test(self):
        # If the relu layer has negative_slope param, it will become a caffe2
        # LeakyRelu op.
        leakyrelu_net_str = """
        name: 'test_leakyrelu'
        input: 'data'
        input_shape {
        dim: 1 dim: 3 dim: 256 dim: 256
        }
        layer {
        name: 'leakyrelu'
        type: 'ReLU'
        bottom: 'data'
        top: 'top'
        relu_param {
        negative_slope: 0.5
        }
        }
        """
        relu_net_str = """
        name: 'test_relu'
        input: 'data'
        input_shape {
        dim: 1 dim: 3 dim: 256 dim: 256
        }
        layer {
        name: 'relu'
        type: 'ReLU'
        bottom: 'data'
        top: 'top'
        }
        """

        # (key, value) = (caffe2_op name, caffe_net_def)
        net_test_case = {'LeakyRelu': leakyrelu_net_str, 'Relu': relu_net_str}

        for caffe2_op_name in net_test_case:
            caffe_net_def = caffe_pb2.NetParameter()
            text_format.Merge(net_test_case[caffe2_op_name], caffe_net_def)
            caffe_pretrained = caffe_pb2.NetParameter()
            caffe2_net, _ = caffe_translator.TranslateModel(
                caffe_net_def, caffe_pretrained, is_test=True)

            self.assertEqual(caffe2_net.op[0].type, caffe2_op_name)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print(
            'If you do not explicitly ask to run this test, I will not run it. '
            'Pass in any argument to have the test run for you.'
        )
        sys.exit(0)
    unittest.main()
