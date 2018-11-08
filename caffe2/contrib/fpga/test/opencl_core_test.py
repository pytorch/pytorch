from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.proto import caffe2_pb2
from caffe2.python import core, test_util

import logging
logging.basicConfig()
logger = logging.getLogger("OpenCL")


class TestOpenCLDevice(test_util.TestCase):
    def setUp(self):
        device_option = caffe2_pb2.DeviceOption()
        device_option.device_type = caffe2_pb2.OPENCL
        device_option.device_id = 1
        device_option.extra_info.append('FPGA')
        self.ocl_option = device_option
        self.cpu_option = caffe2_pb2.DeviceOption()

    def _test_op(
        self,
        op_name,
        in_option,
        out_option,
        op_option=None,
        inputs=None,
        outputs=None
    ):
        op_option = self.ocl_option if not op_option else op_option
        inputs = ["blob_1"] if not inputs else inputs
        outputs = ["blob_2"] if not outputs else outputs
        with core.DeviceScope(op_option):
            op = core.CreateOperator(op_name, inputs, outputs)
        input_dev, output_dev = core.InferOpBlobDevices(op)
        logger.info('my op', op)
        logger.info('input=', input_dev, 'output=', output_dev)

        if isinstance(in_option, list):
            assert len(in_option) == len(input_dev), \
                'Length of input device option should match' \
                '{} vs. {}'.format(in_option, input_dev)
            for in_dev, in_opt in zip(input_dev, in_option):
                self.assertEqual(in_dev, in_opt)
        else:
            for in_dev in input_dev:
                self.assertEqual(in_dev, in_option)
        if isinstance(out_option, list):
            assert len(out_option) == len(output_dev), \
                'Length of output device option should match' \
                '{} vs. {}'.format(out_option, output_dev)
            for out_dev, out_opt in zip(output_dev, out_option):
                self.assertEqual(out_dev, out_opt)
        else:
            for out_dev in output_dev:
                print('ff', out_dev, 'out---', out_option)
                self.assertEqual(out_dev, out_option)

    def test_infer_device_cross_device(self):
        self._test_op("CopyFromOpenCL", self.ocl_option, self.cpu_option)
        self._test_op("CopyToOpenCL", self.cpu_option, self.ocl_option)
