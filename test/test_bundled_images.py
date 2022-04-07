#!/usr/bin/env python3
# Owner(s): ["oncall: mobile"]

import numpy
import torch
import torch.utils.bundled_inputs
import io
import cv2
from torch.testing._internal.common_utils import TestCase

BAD_IMG_PATH = "caffe2/test/test_img/bad_image.gif"

torch.ops.load_library("//caffe2/torch/fb/operators:decode_bundled_image")

def model_size(sm):
    buffer = io.BytesIO()
    torch.jit.save(sm, buffer)
    return len(buffer.getvalue())

def save_and_load(sm):
    buffer = io.BytesIO()
    torch.jit.save(sm, buffer)
    buffer.seek(0)
    return torch.jit.load(buffer)

"""Return an InflatableArg that contains a tensor of the compressed image and the way to decode it

    keyword arguments:
    img_tensor -- the raw image tensor in HWC or NCHW with pixel value of type unsigned int
                  if in NCHW format, N should be 1
    quality -- the quality needed to compress the image
"""
def bundle_jpeg_image(img_tensor, quality):
    # turn NCHW to HWC
    if img_tensor.dim() == 4:
        assert(img_tensor.size(0) == 1)
        img_tensor = img_tensor[0].permute(1, 2, 0)
    pixels = img_tensor.numpy()
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc_img = cv2.imencode(".JPEG", pixels, encode_param)
    enc_img_tensor = torch.from_numpy(enc_img)
    enc_img_tensor = torch.flatten(enc_img_tensor).byte()
    obj = torch.utils.bundled_inputs.InflatableArg(enc_img_tensor, "torch.ops.fb.decode_bundled_image({})")
    return obj

def get_tensor_from_raw_BGR(im) -> torch.Tensor:
    raw_data = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    raw_data = torch.from_numpy(raw_data).float()
    raw_data = raw_data.permute(2, 0, 1)
    raw_data = torch.div(raw_data, 255).unsqueeze(0)
    return raw_data


class TestBundledImages(TestCase):
    def test_single_tensors(self):
        class SingleTensorModel(torch.nn.Module):
            def forward(self, arg):
                return arg
        im = cv2.imread("caffe2/test/test_img/p1.jpg")
        tensor = torch.from_numpy(im)
        inflatable_arg = bundle_jpeg_image(tensor, 90)
        input = [(inflatable_arg,)]
        sm = torch.jit.script(SingleTensorModel())
        torch.utils.bundled_inputs.augment_model_with_bundled_inputs(sm, input)
        loaded = save_and_load(sm)
        inflated = loaded.get_all_bundled_inputs()
        decoded_data = inflated[0][0]

        # raw image
        raw_data = get_tensor_from_raw_BGR(im)

        self.assertEqual(len(inflated), 1)
        self.assertEqual(len(inflated[0]), 1)
        self.assertEqual(raw_data.shape, decoded_data.shape)
        self.assertEqual(raw_data, decoded_data, atol=0.1, rtol=1e-01)

        # Check if fb::image_decode_to_NCHW works as expected
        with open("caffe2/test/test_img/p1.jpg", "rb") as fp:
            weight = torch.full((3,), 1.0 / 255.0).diag()
            bias = torch.zeros(3)
            byte_tensor = torch.tensor(list(fp.read())).byte()
            im2_tensor = torch.ops.fb.image_decode_to_NCHW(byte_tensor, weight, bias)
            self.assertEqual(raw_data.shape, im2_tensor.shape)
            self.assertEqual(raw_data, im2_tensor, atol=0.1, rtol=1e-01)

    def test_decode_bundled_bad_image_or_empty(self):
        """If the operator fails to decode an image, it returns a
        0-color-channel image tensor with size 0x0.
        """

        # Use in TorchScript
        @torch.jit.script
        def decode_image(image: torch.Tensor) -> torch.Tensor:
            return torch.ops.fb.decode_bundled_image_or_empty(image)

        with open(BAD_IMG_PATH, "rb") as f:
            bad_img_data = f.read()
            np_data = numpy.frombuffer(bad_img_data, dtype=numpy.uint8)
            image = decode_image(torch.from_numpy(np_data))
            # decoding returns a 1x0x0x0 NCHW tensor
            self.assertTrue(4 == len(image.shape))
            self.assertTrue(0 == image.numel())
            n, c, w, h = image.size()
            self.assertTrue(1 == n)
            self.assertTrue(0 == c)
            self.assertTrue(0 == w)
            self.assertTrue(0 == h)

    def test_decode_bundled_bad_image(self):
        """If the operator fails to decode an image, it throws an exception."""

        # Use in TorchScript
        @torch.jit.script
        def decode_image(image: torch.Tensor) -> torch.Tensor:
            return torch.ops.fb.decode_bundled_image(image)

        with open(BAD_IMG_PATH, "rb") as f:
            bad_img_data = f.read()
            np_data = numpy.frombuffer(bad_img_data, dtype=numpy.uint8)
            with self.assertRaises(RuntimeError):
                image = decode_image(torch.from_numpy(np_data))
