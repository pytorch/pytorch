from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
try:
    import cv2
except ImportError:
    pass  # Handled below

from PIL import Image
import numpy as np
import lmdb
import shutil
try:
    import StringIO
except ImportError:
    from io import StringIO
import sys
import tempfile

# TODO: This test does not test scaling because
# the algorithms used by OpenCV in the C and Python
# version seem to differ slightly. It does test
# most other features

from hypothesis import given, settings, Verbosity
import hypothesis.strategies as st

from caffe2.proto import caffe2_pb2
import caffe2.python.hypothesis_test_util as hu

from caffe2.python import workspace, core


# Verification routines (applies transformations to image to
# verify if the operator produces same result)
def verify_apply_bounding_box(img, box):
    import skimage.util
    if any(type(box[f]) is not int or np.isnan(box[f] or box[f] < 0)
           for f in range(0, 4)):
        return img
    # Box is ymin, xmin, bound_height, bound_width
    y_bounds = (box[0], img.shape[0] - box[0] - box[2])
    x_bounds = (box[1], img.shape[1] - box[1] - box[3])
    c_bounds = (0, 0)

    if any(el < 0 for el in list(y_bounds) + list(x_bounds) + list(c_bounds)):
        return img

    bboxed = skimage.util.crop(img, (y_bounds, x_bounds, c_bounds))
    return bboxed


# This function is called but not used. It will trip on assert False if
# the arguments are wrong (improper example)
def verify_rescale(img, minsize):
    # Here we use OpenCV transformation to match the C code
    scale_amount = float(minsize) / min(img.shape[0], img.shape[1])
    if scale_amount <= 1.0:
        return img

    print("Scale amount is %f -- should be < 1.0; got shape %s" %
          (scale_amount, str(img.shape)))
    assert False
    img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    output_shape = (int(np.ceil(scale_amount * img_cv.shape[0])),
                    int(np.ceil(scale_amount * img_cv.shape[1])))
    resized = cv2.resize(img_cv,
                         dsize=output_shape,
                         interpolation=cv2.INTER_AREA)

    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    assert resized.shape[0] >= minsize
    assert resized.shape[1] >= minsize
    return resized


def verify_crop(img, crop):
    import skimage.util
    assert img.shape[0] >= crop
    assert img.shape[1] >= crop
    y_offset = 0
    if img.shape[0] > crop:
        y_offset = (img.shape[0] - crop) // 2

    x_offset = 0
    if img.shape[1] > crop:
        x_offset = (img.shape[1] - crop) // 2

    y_bounds = (y_offset, img.shape[0] - crop - y_offset)
    x_bounds = (x_offset, img.shape[1] - crop - x_offset)
    c_bounds = (0, 0)
    cropped = skimage.util.crop(img, (y_bounds, x_bounds, c_bounds))
    assert cropped.shape[0] == crop
    assert cropped.shape[1] == crop
    return cropped


def verify_color_normalize(img, means, stds):
    # Note the RGB/BGR inversion
    # Operate on integers like the C version
    img = img * 255.0
    img[:, :, 0] = (img[:, :, 0] - means[2]) / stds[2]
    img[:, :, 1] = (img[:, :, 1] - means[1]) / stds[1]
    img[:, :, 2] = (img[:, :, 2] - means[0]) / stds[0]
    return img * (1.0 / 255.0)


# Printing function (for debugging)
def caffe2_img(img):
    # Convert RGB to BGR
    img = img[:, :, (2, 1, 0)]
    # Convert HWC to CHW
    img = img.swapaxes(1, 2).swapaxes(0, 1)
    img = img * 255.0
    return img.astype(np.int32)


# Bounding box is ymin, xmin, height, width
def create_test(output_dir, width, height, default_bound,
                minsize, crop, means, stds, count):
    print("Creating a temporary lmdb database of %d pictures..." % (count))

    if default_bound is None:
        default_bound = [-1] * 4

    LMDB_MAP_SIZE = 1 << 40
    env = lmdb.open(output_dir, map_size=LMDB_MAP_SIZE, subdir=True)
    index = 0
    # Create images and the expected results
    expected_results = []
    with env.begin(write=True) as txn:
        while index < count:
            img_array = np.random.random_integers(
                0, 255, [height, width, 3]).astype(np.uint8)
            img_obj = Image.fromarray(img_array)
            img_str = StringIO.StringIO()
            img_obj.save(img_str, 'PNG')

            # Create a random bounding box for every other image
            # ymin, xmin, bound_height, bound_width
            # TODO: To ensure that we never need to scale, we
            # ensure that the bounding-box is larger than the
            # minsize parameter
            bounding_box = list(default_bound)
            do_default_bound = True
            if index % 2 == 0:
                if height > minsize and width > minsize:
                    do_default_bound = False
                    bounding_box[0:2] = [np.random.randint(a) for a in
                                         (height - minsize, width - minsize)]
                    bounding_box[2:4] = [np.random.randint(a) + minsize for a in
                                         (height - bounding_box[0] - minsize + 1,
                                          width - bounding_box[1] - minsize + 1)]
                    # print("Bounding box is %s" % (str(bounding_box)))
            # Create expected result
            img_expected = img_array.astype(np.float32) * (1.0 / 255.0)
            # print("Orig image: %s" % (str(caffe2_img(img_expected))))
            img_expected = verify_apply_bounding_box(
                img_expected,
                bounding_box)
            # print("Bounded image: %s" % (str(caffe2_img(img_expected))))

            img_expected = verify_rescale(img_expected, minsize)

            img_expected = verify_crop(img_expected, crop)
            # print("Crop image: %s" % (str(caffe2_img(img_expected))))

            img_expected = verify_color_normalize(img_expected, means, stds)
            # print("Color image: %s" % (str(caffe2_img(img_expected))))

            expected_results.append(caffe2_img(img_expected))

            tensor_protos = caffe2_pb2.TensorProtos()
            image_tensor = tensor_protos.protos.add()
            image_tensor.data_type = 4  # string data
            image_tensor.string_data.append(img_str.getvalue())
            img_str.close()

            label_tensor = tensor_protos.protos.add()
            label_tensor.data_type = 2  # int32 data
            label_tensor.int32_data.append(index)

            if not do_default_bound:
                bounding_tensor = tensor_protos.protos.add()
                bounding_tensor.data_type = 2  # int32 data
                bounding_tensor.int32_data.extend(bounding_box)

            txn.put(
                '{}'.format(index).encode('ascii'),
                tensor_protos.SerializeToString()
            )
            index = index + 1
        # End while
    # End with
    return expected_results


@unittest.skipIf('cv2' not in sys.modules, 'python-opencv is not installed')
class TestImport(hu.HypothesisTestCase):
    @given(size_tuple=st.tuples(
        st.integers(min_value=8, max_value=4096),
        st.integers(min_value=8, max_value=4096)).flatmap(lambda t: st.tuples(
            st.just(t[0]), st.just(t[1]),
            st.just(min(t[0] - 6, t[1] - 4)),
            st.integers(min_value=1, max_value=min(t[0] - 6, t[1] - 4)))),
        means=st.tuples(st.integers(min_value=0, max_value=255),
                        st.integers(min_value=0, max_value=255),
                        st.integers(min_value=0, max_value=255)),
        stds=st.tuples(st.floats(min_value=1, max_value=10),
                       st.floats(min_value=1, max_value=10),
                       st.floats(min_value=1, max_value=10)),
        **hu.gcs)
    @settings(verbosity=Verbosity.verbose)
    def test_imageinput(self, size_tuple, means, stds, gc, dc):
        # TODO: Does not test on GPU and does not test use_gpu_transform
        # WARNING: Using ModelHelper automatically does NHWC to NCHW
        # transformation if needed.
        width, height, minsize, crop = size_tuple
        means = [float(m) for m in means]
        stds = [float(s) for s in stds]
        out_dir = tempfile.mkdtemp()
        count_images = 2  # One with bounding box and one without
        expected_images = create_test(
            out_dir,
            width=width,
            height=height,
            default_bound=(3, 5, height - 3, width - 5),
            minsize=minsize,
            crop=crop,
            means=means,
            stds=stds,
            count=count_images
        )
        for device_option in dc:
            with hu.temp_workspace():
                reader_net = core.Net('reader')
                reader_net.CreateDB(
                    [],
                    'DB',
                    db=out_dir,
                    db_type="lmdb"
                )
                workspace.RunNetOnce(reader_net)
                imageop = core.CreateOperator(
                    'ImageInput',
                    ['DB'],
                    ["data", "label"],
                    batch_size=count_images,
                    color=3,
                    minsize=minsize,
                    crop=crop,
                    is_test=True,
                    bounding_ymin=3,
                    bounding_xmin=5,
                    bounding_height=height - 3,
                    bounding_width=width - 5,
                    mean_per_channel=means,
                    std_per_channel=stds,
                    use_gpu_transform=(device_option.device_type == 1)
                )

                imageop.device_option.CopyFrom(device_option)
                main_net = core.Net('main')
                main_net.Proto().op.extend([imageop])
                workspace.RunNetOnce(main_net)
                l = workspace.FetchBlob('label')
                result = workspace.FetchBlob('data').astype(np.int32)
                # If we don't use_gpu_transform, the output is in NHWC
                # Our reference output is CHW so we swap
                if device_option.device_type != 1:
                    expected = [img.swapaxes(0, 1).swapaxes(1, 2) for
                                img in expected_images]
                else:
                    expected = expected_images
                for i in range(count_images):
                    self.assertEqual(l[i], i)
                    self.assertEqual((expected[i] - result[i] > 1).sum(), 0)
                # End for
            # End with
        # End for
        shutil.rmtree(out_dir)
    # End test_imageinput


if __name__ == '__main__':
    import unittest
    unittest.main()
