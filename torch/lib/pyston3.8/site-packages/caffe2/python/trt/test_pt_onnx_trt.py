###################################################################################################
# ATTENTION! This test will most probably fail if you install TensorRT 6.0.1 only.
# That's because it's shipped with older version of ONNX parser not supporting some
# required features. To make it work please use new version: https://github.com/onnx/onnx-tensorrt
# Just clone it and do something like this:
#
# ~/pt/third_party/onnx-tensorrt$ mkdir build/
# ~/pt/third_party/onnx-tensorrt$ cd build/
# ~/pt/third_party/onnx-tensorrt/build$ cmake ..
# ~/pt/third_party/onnx-tensorrt/build$ make
# ~/pt/third_party/onnx-tensorrt/build$ sudo cp libnvonnxparser.so.6.0.1 /usr/lib/x86_64-linux-gnu
#
# This note is valid for 6.0.1 release only. September 18th, 2019.
###################################################################################################

import os
import unittest

from PIL import Image
import numpy as np
import torch
import torchvision.models as models

import pycuda.driver as cuda
# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit

import tensorrt as trt
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def allocate_buffers(engine):
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)),
                                    dtype=trt.nptype(trt.float32))
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)),
                                     dtype=trt.nptype(trt.float32))
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()
    return h_input, d_input, h_output, d_output, stream

def load_normalized_test_case(input_shape, test_image, pagelocked_buffer, normalization_hint):
    def normalize_image(image):
        c, h, w = input_shape
        image_arr = np.asarray(image.resize((w, h), Image.ANTIALIAS)).transpose([2, 0, 1])\
            .astype(trt.nptype(trt.float32)).ravel()
        if (normalization_hint == 0):
            return (image_arr / 255.0 - 0.45) / 0.225
        elif (normalization_hint == 1):
            return (image_arr / 256.0 - 0.5)
    np.copyto(pagelocked_buffer, normalize_image(Image.open(test_image)))
    return test_image

class Test_PT_ONNX_TRT(unittest.TestCase):
    def __enter__(self):
        return self

    def setUp(self):
        data_path = os.path.join(os.path.dirname(__file__), 'data')
        self.image_files=["binoculars.jpeg", "reflex_camera.jpeg", "tabby_tiger_cat.jpg"]
        for index, f in enumerate(self.image_files):
            self.image_files[index] = os.path.abspath(os.path.join(data_path, f))
            if not os.path.exists(self.image_files[index]):
                raise FileNotFoundError(self.image_files[index] + " does not exist.")
        self.labels = open(os.path.abspath(os.path.join(data_path, "class_labels.txt")), 'r').read().split('\n')

    def build_engine_onnx(self, model_file):
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(flags = 1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 33
            with open(model_file, 'rb') as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        self.fail("ERROR: {}".format(parser.get_error(error)))
            return builder.build_cuda_engine(network)

    def _test_model(self, model_name, input_shape = (3, 224, 224), normalization_hint = 0):

        model = getattr(models, model_name)(pretrained=True)

        shape = (1,) + input_shape
        dummy_input  = (torch.randn(shape),)
        onnx_name = model_name + ".onnx"

        torch.onnx.export(model,
                          dummy_input,
                          onnx_name,
                          input_names = [],
                          output_names = [],
                          verbose=False,
                          export_params=True,
                          opset_version=9)

        with self.build_engine_onnx(onnx_name) as engine:
            h_input, d_input, h_output, d_output, stream = allocate_buffers(engine)
            with engine.create_execution_context() as context:
                err_count = 0
                for index, f in enumerate(self.image_files):
                    test_case = load_normalized_test_case(input_shape, f,\
                        h_input, normalization_hint)
                    cuda.memcpy_htod_async(d_input, h_input, stream)

                    context.execute_async_v2(bindings=[d_input, d_output],
                                             stream_handle=stream.handle)
                    cuda.memcpy_dtoh_async(h_output, d_output, stream)
                    stream.synchronize()

                    amax = np.argmax(h_output)
                    pred = self.labels[amax]
                    if "_".join(pred.split()) not in\
                            os.path.splitext(os.path.basename(test_case))[0]:
                        err_count = err_count + 1
                self.assertLessEqual(err_count, 1, "Too many recognition errors")

    def test_alexnet(self):
        self._test_model("alexnet", (3, 227, 227))

    def test_resnet18(self):
        self._test_model("resnet18")
    def test_resnet34(self):
        self._test_model("resnet34")
    def test_resnet50(self):
        self._test_model("resnet50")
    def test_resnet101(self):
        self._test_model("resnet101")
    @unittest.skip("Takes 2m")
    def test_resnet152(self):
        self._test_model("resnet152")

    def test_resnet50_2(self):
        self._test_model("wide_resnet50_2")
    @unittest.skip("Takes 2m")
    def test_resnet101_2(self):
        self._test_model("wide_resnet101_2")

    def test_squeezenet1_0(self):
        self._test_model("squeezenet1_0")
    def test_squeezenet1_1(self):
        self._test_model("squeezenet1_1")

    def test_googlenet(self):
        self._test_model("googlenet")
    def test_inception_v3(self):
        self._test_model("inception_v3")

    def test_mnasnet0_5(self):
        self._test_model("mnasnet0_5", normalization_hint = 1)
    def test_mnasnet1_0(self):
        self._test_model("mnasnet1_0", normalization_hint = 1)

    def test_mobilenet_v2(self):
        self._test_model("mobilenet_v2", normalization_hint = 1)

    def test_shufflenet_v2_x0_5(self):
        self._test_model("shufflenet_v2_x0_5")
    def test_shufflenet_v2_x1_0(self):
        self._test_model("shufflenet_v2_x1_0")

    def test_vgg11(self):
        self._test_model("vgg11")
    def test_vgg11_bn(self):
        self._test_model("vgg11_bn")
    def test_vgg13(self):
        self._test_model("vgg13")
    def test_vgg13_bn(self):
        self._test_model("vgg13_bn")
    def test_vgg16(self):
        self._test_model("vgg16")
    def test_vgg16_bn(self):
        self._test_model("vgg16_bn")
    def test_vgg19(self):
        self._test_model("vgg19")
    def test_vgg19_bn(self):
        self._test_model("vgg19_bn")

    @unittest.skip("Takes 13m")
    def test_densenet121(self):
        self._test_model("densenet121")
    @unittest.skip("Takes 25m")
    def test_densenet161(self):
        self._test_model("densenet161")
    @unittest.skip("Takes 27m")
    def test_densenet169(self):
        self._test_model("densenet169")
    @unittest.skip("Takes 44m")
    def test_densenet201(self):
        self._test_model("densenet201")

if __name__ == '__main__':
    unittest.main()
