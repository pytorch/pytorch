# Owner(s): ["module: unknown"]

import io
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import expecttest
import numpy as np


TEST_TENSORBOARD = True
try:
    import tensorboard.summary.writer.event_file_writer  # noqa: F401
    from tensorboard.compat.proto.summary_pb2 import Summary
except ImportError:
    TEST_TENSORBOARD = False

HAS_TORCHVISION = True
try:
    import torchvision
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

TEST_MATPLOTLIB = True
try:
    import matplotlib
    if os.environ.get('DISPLAY', '') == '':
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    TEST_MATPLOTLIB = False
skipIfNoMatplotlib = unittest.skipIf(not TEST_MATPLOTLIB, "no matplotlib")

import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_MACOS,
    IS_S390X,
    IS_WINDOWS,
    parametrize,
    run_tests,
    TEST_WITH_CROSSREF,
    TestCase,
    skipIfTorchDynamo,
)


def tensor_N(shape, dtype=float):
    numel = np.prod(shape)
    x = (np.arange(numel, dtype=dtype)).reshape(shape)
    return x

class BaseTestCase(TestCase):
    """ Base class used for all TensorBoard tests """
    def setUp(self):
        super().setUp()
        if not TEST_TENSORBOARD:
            return self.skipTest("Skip the test since TensorBoard is not installed")
        if TEST_WITH_CROSSREF:
            return self.skipTest("Don't run TensorBoard tests with crossref")
        self.temp_dirs = []

    def createSummaryWriter(self):
        # Just to get the name of the directory in a writable place. tearDown()
        # is responsible for clean-ups.
        temp_dir = tempfile.TemporaryDirectory(prefix="test_tensorboard").name
        self.temp_dirs.append(temp_dir)
        return SummaryWriter(temp_dir)

    def tearDown(self):
        super().tearDown()
        # Remove directories created by SummaryWriter
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


if TEST_TENSORBOARD:
    from google.protobuf import text_format
    from PIL import Image
    from tensorboard.compat.proto.graph_pb2 import GraphDef
    from tensorboard.compat.proto.types_pb2 import DataType

    from torch.utils.tensorboard import summary, SummaryWriter
    from torch.utils.tensorboard._convert_np import make_np
    from torch.utils.tensorboard._pytorch_graph import graph
    from torch.utils.tensorboard._utils import _prepare_video, convert_to_HWC
    from torch.utils.tensorboard.summary import int_to_half, tensor_proto

class TestTensorBoardPyTorchNumpy(BaseTestCase):
    def test_pytorch_np(self):
        tensors = [torch.rand(3, 10, 10), torch.rand(1), torch.rand(1, 2, 3, 4, 5)]
        for tensor in tensors:
            # regular tensor
            self.assertIsInstance(make_np(tensor), np.ndarray)

            # CUDA tensor
            if torch.cuda.is_available():
                self.assertIsInstance(make_np(tensor.cuda()), np.ndarray)

            # regular variable
            self.assertIsInstance(make_np(torch.autograd.Variable(tensor)), np.ndarray)

            # CUDA variable
            if torch.cuda.is_available():
                self.assertIsInstance(make_np(torch.autograd.Variable(tensor).cuda()), np.ndarray)

        # python primitive type
        self.assertIsInstance(make_np(0), np.ndarray)
        self.assertIsInstance(make_np(0.1), np.ndarray)

    def test_pytorch_autograd_np(self):
        x = torch.autograd.Variable(torch.empty(1))
        self.assertIsInstance(make_np(x), np.ndarray)

    def test_pytorch_write(self):
        with self.createSummaryWriter() as w:
            w.add_scalar('scalar', torch.autograd.Variable(torch.rand(1)), 0)

    def test_pytorch_histogram(self):
        with self.createSummaryWriter() as w:
            w.add_histogram('float histogram', torch.rand((50,)))
            w.add_histogram('int histogram', torch.randint(0, 100, (50,)))
            w.add_histogram('bfloat16 histogram', torch.rand(50, dtype=torch.bfloat16))

    def test_pytorch_histogram_raw(self):
        with self.createSummaryWriter() as w:
            num = 50
            floats = make_np(torch.rand((num,)))
            bins = [0.0, 0.25, 0.5, 0.75, 1.0]
            counts, limits = np.histogram(floats, bins)
            sum_sq = floats.dot(floats).item()
            w.add_histogram_raw('float histogram raw',
                                min=floats.min().item(),
                                max=floats.max().item(),
                                num=num,
                                sum=floats.sum().item(),
                                sum_squares=sum_sq,
                                bucket_limits=limits[1:].tolist(),
                                bucket_counts=counts.tolist())

            ints = make_np(torch.randint(0, 100, (num,)))
            bins = [0, 25, 50, 75, 100]
            counts, limits = np.histogram(ints, bins)
            sum_sq = ints.dot(ints).item()
            w.add_histogram_raw('int histogram raw',
                                min=ints.min().item(),
                                max=ints.max().item(),
                                num=num,
                                sum=ints.sum().item(),
                                sum_squares=sum_sq,
                                bucket_limits=limits[1:].tolist(),
                                bucket_counts=counts.tolist())

            ints = torch.tensor(range(0, 100)).float()
            nbins = 100
            counts = torch.histc(ints, bins=nbins, min=0, max=99)
            limits = torch.tensor(range(nbins))
            sum_sq = ints.dot(ints).item()
            w.add_histogram_raw('int histogram raw',
                                min=ints.min().item(),
                                max=ints.max().item(),
                                num=num,
                                sum=ints.sum().item(),
                                sum_squares=sum_sq,
                                bucket_limits=limits.tolist(),
                                bucket_counts=counts.tolist())

class TestTensorBoardUtils(BaseTestCase):
    def test_to_HWC(self):
        test_image = np.random.randint(0, 256, size=(3, 32, 32), dtype=np.uint8)
        converted = convert_to_HWC(test_image, 'chw')
        self.assertEqual(converted.shape, (32, 32, 3))
        test_image = np.random.randint(0, 256, size=(16, 3, 32, 32), dtype=np.uint8)
        converted = convert_to_HWC(test_image, 'nchw')
        self.assertEqual(converted.shape, (64, 256, 3))
        test_image = np.random.randint(0, 256, size=(32, 32), dtype=np.uint8)
        converted = convert_to_HWC(test_image, 'hw')
        self.assertEqual(converted.shape, (32, 32, 3))

    def test_convert_to_HWC_dtype_remains_same(self):
        # test to ensure convert_to_HWC restores the dtype of input np array and
        # thus the scale_factor calculated for the image is 1
        test_image = torch.tensor([[[[1, 2, 3], [4, 5, 6]]]], dtype=torch.uint8)
        tensor = make_np(test_image)
        tensor = convert_to_HWC(tensor, 'NCHW')
        scale_factor = summary._calc_scale_factor(tensor)
        self.assertEqual(scale_factor, 1, msg='Values are already in [0, 255], scale factor should be 1')


    def test_prepare_video(self):
        # At each timeframe, the sum over all other
        # dimensions of the video should be the same.
        shapes = [
            (16, 30, 3, 28, 28),
            (36, 30, 3, 28, 28),
            (19, 29, 3, 23, 19),
            (3, 3, 3, 3, 3)
        ]
        for s in shapes:
            V_input = np.random.random(s)
            V_after = _prepare_video(np.copy(V_input))
            total_frame = s[1]
            V_input = np.swapaxes(V_input, 0, 1)
            for f in range(total_frame):
                x = np.reshape(V_input[f], newshape=(-1))
                y = np.reshape(V_after[f], newshape=(-1))
                np.testing.assert_array_almost_equal(np.sum(x), np.sum(y))

    def test_numpy_vid_uint8(self):
        V_input = np.random.randint(0, 256, (16, 30, 3, 28, 28)).astype(np.uint8)
        V_after = _prepare_video(np.copy(V_input)) * 255
        total_frame = V_input.shape[1]
        V_input = np.swapaxes(V_input, 0, 1)
        for f in range(total_frame):
            x = np.reshape(V_input[f], newshape=(-1))
            y = np.reshape(V_after[f], newshape=(-1))
            np.testing.assert_array_almost_equal(np.sum(x), np.sum(y))

freqs = [262, 294, 330, 349, 392, 440, 440, 440, 440, 440, 440]

true_positive_counts = [75, 64, 21, 5, 0]
false_positive_counts = [150, 105, 18, 0, 0]
true_negative_counts = [0, 45, 132, 150, 150]
false_negative_counts = [0, 11, 54, 70, 75]
precision = [0.3333333, 0.3786982, 0.5384616, 1.0, 0.0]
recall = [1.0, 0.8533334, 0.28, 0.0666667, 0.0]

class TestTensorBoardWriter(BaseTestCase):
    def test_writer(self):
        with self.createSummaryWriter() as writer:
            sample_rate = 44100

            n_iter = 0
            writer.add_hparams(
                {'lr': 0.1, 'bsize': 1},
                {'hparam/accuracy': 10, 'hparam/loss': 10}
            )
            writer.add_scalar('data/scalar_systemtime', 0.1, n_iter)
            writer.add_scalar('data/scalar_customtime', 0.2, n_iter, walltime=n_iter)
            writer.add_scalar('data/new_style', 0.2, n_iter, new_style=True)
            writer.add_scalars('data/scalar_group', {
                "xsinx": n_iter * np.sin(n_iter),
                "xcosx": n_iter * np.cos(n_iter),
                "arctanx": np.arctan(n_iter)
            }, n_iter)
            x = np.zeros((32, 3, 64, 64))  # output from network
            writer.add_images('Image', x, n_iter)  # Tensor
            writer.add_image_with_boxes('imagebox',
                                        np.zeros((3, 64, 64)),
                                        np.array([[10, 10, 40, 40], [40, 40, 60, 60]]),
                                        n_iter)
            x = np.zeros(sample_rate * 2)

            writer.add_audio('myAudio', x, n_iter)
            writer.add_video('myVideo', np.random.rand(16, 48, 1, 28, 28).astype(np.float32), n_iter)
            writer.add_text('Text', 'text logged at step:' + str(n_iter), n_iter)
            writer.add_text('markdown Text', '''a|b\n-|-\nc|d''', n_iter)
            writer.add_histogram('hist', np.random.rand(100, 100), n_iter)
            writer.add_pr_curve('xoxo', np.random.randint(2, size=100), np.random.rand(
                100), n_iter)  # needs tensorboard 0.4RC or later
            writer.add_pr_curve_raw('prcurve with raw data', true_positive_counts,
                                    false_positive_counts,
                                    true_negative_counts,
                                    false_negative_counts,
                                    precision,
                                    recall, n_iter)

            v = np.array([[[1, 1, 1], [-1, -1, 1], [1, -1, -1], [-1, 1, -1]]], dtype=float)
            c = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 255]]], dtype=int)
            f = np.array([[[0, 2, 3], [0, 3, 1], [0, 1, 2], [1, 3, 2]]], dtype=int)
            writer.add_mesh('my_mesh', vertices=v, colors=c, faces=f)

class TestTensorBoardSummaryWriter(BaseTestCase):
    def test_summary_writer_ctx(self):
        # after using a SummaryWriter as a ctx it should be closed
        with self.createSummaryWriter() as writer:
            writer.add_scalar('test', 1)
        self.assertIs(writer.file_writer, None)

    def test_summary_writer_close(self):
        # Opening and closing SummaryWriter a lot should not run into
        # OSError: [Errno 24] Too many open files
        passed = True
        try:
            writer = self.createSummaryWriter()
            writer.close()
        except OSError:
            passed = False

        self.assertTrue(passed)

    def test_pathlib(self):
        with tempfile.TemporaryDirectory(prefix="test_tensorboard_pathlib") as d:
            p = Path(d)
            with SummaryWriter(p) as writer:
                writer.add_scalar('test', 1)

class TestTensorBoardEmbedding(BaseTestCase):
    def test_embedding(self):
        w = self.createSummaryWriter()
        all_features = torch.tensor([[1., 2., 3.], [5., 4., 1.], [3., 7., 7.]])
        all_labels = torch.tensor([33., 44., 55.])
        all_images = torch.zeros(3, 3, 5, 5)

        w.add_embedding(all_features,
                        metadata=all_labels,
                        label_img=all_images,
                        global_step=2)

        dataset_label = ['test'] * 2 + ['train'] * 2
        all_labels = list(zip(all_labels, dataset_label))
        w.add_embedding(all_features,
                        metadata=all_labels,
                        label_img=all_images,
                        metadata_header=['digit', 'dataset'],
                        global_step=2)
        # assert...

    def test_embedding_64(self):
        w = self.createSummaryWriter()
        all_features = torch.tensor([[1., 2., 3.], [5., 4., 1.], [3., 7., 7.]])
        all_labels = torch.tensor([33., 44., 55.])
        all_images = torch.zeros((3, 3, 5, 5), dtype=torch.float64)

        w.add_embedding(all_features,
                        metadata=all_labels,
                        label_img=all_images,
                        global_step=2)

        dataset_label = ['test'] * 2 + ['train'] * 2
        all_labels = list(zip(all_labels, dataset_label))
        w.add_embedding(all_features,
                        metadata=all_labels,
                        label_img=all_images,
                        metadata_header=['digit', 'dataset'],
                        global_step=2)

class TestTensorBoardSummary(BaseTestCase):
    def test_uint8_image(self):
        '''
        Tests that uint8 image (pixel values in [0, 255]) is not changed
        '''
        test_image = np.random.randint(0, 256, size=(3, 32, 32), dtype=np.uint8)
        scale_factor = summary._calc_scale_factor(test_image)
        self.assertEqual(scale_factor, 1, msg='Values are already in [0, 255], scale factor should be 1')

    def test_float32_image(self):
        '''
        Tests that float32 image (pixel values in [0, 1]) are scaled correctly
        to [0, 255]
        '''
        test_image = np.random.rand(3, 32, 32).astype(np.float32)
        scale_factor = summary._calc_scale_factor(test_image)
        self.assertEqual(scale_factor, 255, msg='Values are in [0, 1], scale factor should be 255')

    def test_list_input(self):
        with self.assertRaises(Exception) as e_info:
            summary.histogram('dummy', [1, 3, 4, 5, 6], 'tensorflow')

    def test_empty_input(self):
        with self.assertRaises(Exception) as e_info:
            summary.histogram('dummy', np.ndarray(0), 'tensorflow')

    def test_image_with_boxes(self):
        self.assertTrue(compare_image_proto(summary.image_boxes('dummy',
                                            tensor_N(shape=(3, 32, 32)),
                                            np.array([[10, 10, 40, 40]])),
                                            self))

    def test_image_with_one_channel(self):
        self.assertTrue(compare_image_proto(
            summary.image('dummy',
                          tensor_N(shape=(1, 8, 8)),
                          dataformats='CHW'),
                          self))  # noqa: E131

    def test_image_with_one_channel_batched(self):
        self.assertTrue(compare_image_proto(
            summary.image('dummy',
                          tensor_N(shape=(2, 1, 8, 8)),
                          dataformats='NCHW'),
                          self))  # noqa: E131

    def test_image_with_3_channel_batched(self):
        self.assertTrue(compare_image_proto(
            summary.image('dummy',
                          tensor_N(shape=(2, 3, 8, 8)),
                          dataformats='NCHW'),
                          self))  # noqa: E131

    def test_image_without_channel(self):
        self.assertTrue(compare_image_proto(
            summary.image('dummy',
                          tensor_N(shape=(8, 8)),
                          dataformats='HW'),
                          self))  # noqa: E131

    def test_video(self):
        try:
            import moviepy  # noqa: F401
        except ImportError:
            return
        self.assertTrue(compare_proto(summary.video('dummy', tensor_N(shape=(4, 3, 1, 8, 8))), self))
        summary.video('dummy', np.random.rand(16, 48, 1, 28, 28))
        summary.video('dummy', np.random.rand(20, 7, 1, 8, 8))

    @unittest.skipIf(IS_MACOS, "Skipping on mac, see https://github.com/pytorch/pytorch/pull/109349 ")
    @unittest.skipIf(IS_S390X, "Fails on s390x")
    def test_audio(self):
        self.assertTrue(compare_proto(summary.audio('dummy', tensor_N(shape=(42,))), self))

    @unittest.skipIf(IS_MACOS, "Skipping on mac, see https://github.com/pytorch/pytorch/pull/109349 ")
    def test_text(self):
        self.assertTrue(compare_proto(summary.text('dummy', 'text 123'), self))

    @unittest.skipIf(IS_MACOS, "Skipping on mac, see https://github.com/pytorch/pytorch/pull/109349 ")
    def test_histogram_auto(self):
        self.assertTrue(compare_proto(summary.histogram('dummy', tensor_N(shape=(1024,)), bins='auto', max_bins=5), self))

    @unittest.skipIf(IS_MACOS, "Skipping on mac, see https://github.com/pytorch/pytorch/pull/109349 ")
    def test_histogram_fd(self):
        self.assertTrue(compare_proto(summary.histogram('dummy', tensor_N(shape=(1024,)), bins='fd', max_bins=5), self))

    @unittest.skipIf(IS_MACOS, "Skipping on mac, see https://github.com/pytorch/pytorch/pull/109349 ")
    def test_histogram_doane(self):
        self.assertTrue(compare_proto(summary.histogram('dummy', tensor_N(shape=(1024,)), bins='doane', max_bins=5), self))

    def test_custom_scalars(self):
        layout = {
            'Taiwan': {
                'twse': ['Multiline', ['twse/0050', 'twse/2330']]
            },
            'USA': {
                'dow': ['Margin', ['dow/aaa', 'dow/bbb', 'dow/ccc']],
                'nasdaq': ['Margin', ['nasdaq/aaa', 'nasdaq/bbb', 'nasdaq/ccc']]
            }
        }
        summary.custom_scalars(layout)  # only smoke test. Because protobuf in python2/3 serialize dictionary differently.


    @unittest.skipIf(IS_MACOS, "Skipping on mac, see https://github.com/pytorch/pytorch/pull/109349 ")
    def test_mesh(self):
        v = np.array([[[1, 1, 1], [-1, -1, 1], [1, -1, -1], [-1, 1, -1]]], dtype=float)
        c = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 255]]], dtype=int)
        f = np.array([[[0, 2, 3], [0, 3, 1], [0, 1, 2], [1, 3, 2]]], dtype=int)
        mesh = summary.mesh('my_mesh', vertices=v, colors=c, faces=f, config_dict=None)
        self.assertTrue(compare_proto(mesh, self))

    @unittest.skipIf(IS_MACOS, "Skipping on mac, see https://github.com/pytorch/pytorch/pull/109349 ")
    def test_scalar_new_style(self):
        scalar = summary.scalar('test_scalar', 1.0, new_style=True)
        self.assertTrue(compare_proto(scalar, self))
        with self.assertRaises(AssertionError):
            summary.scalar('test_scalar2', torch.Tensor([1, 2, 3]), new_style=True)


def remove_whitespace(string):
    return string.replace(' ', '').replace('\t', '').replace('\n', '')

def get_expected_file(function_ptr):
    module_id = function_ptr.__class__.__module__
    test_file = sys.modules[module_id].__file__
    # Look for the .py file (since __file__ could be pyc).
    test_file = ".".join(test_file.split('.')[:-1]) + '.py'

    # Use realpath to follow symlinks appropriately.
    test_dir = os.path.dirname(os.path.realpath(test_file))
    functionName = function_ptr.id().split('.')[-1]
    return os.path.join(test_dir,
                        "expect",
                        'TestTensorBoard.' + functionName + ".expect")

def read_expected_content(function_ptr):
    expected_file = get_expected_file(function_ptr)
    assert os.path.exists(expected_file), expected_file
    with open(expected_file) as f:
        return f.read()

def compare_image_proto(actual_proto, function_ptr):
    if expecttest.ACCEPT:
        expected_file = get_expected_file(function_ptr)
        with open(expected_file, 'w') as f:
            f.write(text_format.MessageToString(actual_proto))
        return True
    expected_str = read_expected_content(function_ptr)
    expected_proto = Summary()
    text_format.Parse(expected_str, expected_proto)

    [actual, expected] = [actual_proto.value[0], expected_proto.value[0]]
    actual_img = Image.open(io.BytesIO(actual.image.encoded_image_string))
    expected_img = Image.open(io.BytesIO(expected.image.encoded_image_string))

    return (
        actual.tag == expected.tag and
        actual.image.height == expected.image.height and
        actual.image.width == expected.image.width and
        actual.image.colorspace == expected.image.colorspace and
        actual_img == expected_img
    )

def compare_proto(str_to_compare, function_ptr):
    if expecttest.ACCEPT:
        write_proto(str_to_compare, function_ptr)
        return True
    expected = read_expected_content(function_ptr)
    str_to_compare = str(str_to_compare)
    return remove_whitespace(str_to_compare) == remove_whitespace(expected)

def write_proto(str_to_compare, function_ptr):
    expected_file = get_expected_file(function_ptr)
    with open(expected_file, 'w') as f:
        f.write(str(str_to_compare))

class TestTensorBoardPytorchGraph(BaseTestCase):
    def test_pytorch_graph(self):
        dummy_input = (torch.zeros(1, 3),)

        class myLinear(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.l = torch.nn.Linear(3, 5)

            def forward(self, x):
                return self.l(x)

        with self.createSummaryWriter() as w:
            w.add_graph(myLinear(), dummy_input)

        actual_proto, _ = graph(myLinear(), dummy_input)

        expected_str = read_expected_content(self)
        expected_proto = GraphDef()
        text_format.Parse(expected_str, expected_proto)

        self.assertEqual(len(expected_proto.node), len(actual_proto.node))
        for i in range(len(expected_proto.node)):
            expected_node = expected_proto.node[i]
            actual_node = actual_proto.node[i]
            self.assertEqual(expected_node.name, actual_node.name)
            self.assertEqual(expected_node.op, actual_node.op)
            self.assertEqual(expected_node.input, actual_node.input)
            self.assertEqual(expected_node.device, actual_node.device)
            self.assertEqual(
                sorted(expected_node.attr.keys()), sorted(actual_node.attr.keys()))

    def test_nested_nn_squential(self):

        dummy_input = torch.randn(2, 3)

        class InnerNNSquential(torch.nn.Module):
            def __init__(self, dim1, dim2):
                super().__init__()
                self.inner_nn_squential = torch.nn.Sequential(
                    torch.nn.Linear(dim1, dim2),
                    torch.nn.Linear(dim2, dim1),
                )

            def forward(self, x):
                x = self.inner_nn_squential(x)
                return x

        class OuterNNSquential(torch.nn.Module):
            def __init__(self, dim1=3, dim2=4, depth=2):
                super().__init__()
                layers = []
                for _ in range(depth):
                    layers.append(InnerNNSquential(dim1, dim2))
                self.outer_nn_squential = torch.nn.Sequential(*layers)

            def forward(self, x):
                x = self.outer_nn_squential(x)
                return x

        with self.createSummaryWriter() as w:
            w.add_graph(OuterNNSquential(), dummy_input)

        actual_proto, _ = graph(OuterNNSquential(), dummy_input)

        expected_str = read_expected_content(self)
        expected_proto = GraphDef()
        text_format.Parse(expected_str, expected_proto)

        self.assertEqual(len(expected_proto.node), len(actual_proto.node))
        for i in range(len(expected_proto.node)):
            expected_node = expected_proto.node[i]
            actual_node = actual_proto.node[i]
            self.assertEqual(expected_node.name, actual_node.name)
            self.assertEqual(expected_node.op, actual_node.op)
            self.assertEqual(expected_node.input, actual_node.input)
            self.assertEqual(expected_node.device, actual_node.device)
            self.assertEqual(
                sorted(expected_node.attr.keys()), sorted(actual_node.attr.keys()))

    def test_pytorch_graph_dict_input(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.l = torch.nn.Linear(3, 5)

            def forward(self, x):
                return self.l(x)

        class ModelDict(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.l = torch.nn.Linear(3, 5)

            def forward(self, x):
                return {"out": self.l(x)}


        dummy_input = torch.zeros(1, 3)

        with self.createSummaryWriter() as w:
            w.add_graph(Model(), dummy_input)

        with self.createSummaryWriter() as w:
            w.add_graph(Model(), dummy_input, use_strict_trace=True)

        # expect error: Encountering a dict at the output of the tracer...
        with self.assertRaises(RuntimeError):
            with self.createSummaryWriter() as w:
                w.add_graph(ModelDict(), dummy_input, use_strict_trace=True)

        with self.createSummaryWriter() as w:
            w.add_graph(ModelDict(), dummy_input, use_strict_trace=False)


    def test_mlp_graph(self):
        dummy_input = (torch.zeros(2, 1, 28, 28),)

        # This MLP class with the above input is expected
        # to fail JIT optimizations as seen at
        # https://github.com/pytorch/pytorch/issues/18903
        #
        # However, it should not raise an error during
        # the add_graph call and still continue.
        class myMLP(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_len = 1 * 28 * 28
                self.fc1 = torch.nn.Linear(self.input_len, 1200)
                self.fc2 = torch.nn.Linear(1200, 1200)
                self.fc3 = torch.nn.Linear(1200, 10)

            def forward(self, x, update_batch_stats=True):
                h = torch.nn.functional.relu(
                    self.fc1(x.view(-1, self.input_len)))
                h = self.fc2(h)
                h = torch.nn.functional.relu(h)
                h = self.fc3(h)
                return h

        with self.createSummaryWriter() as w:
            w.add_graph(myMLP(), dummy_input)

    def test_wrong_input_size(self):
        with self.assertRaises(RuntimeError) as e_info:
            dummy_input = torch.rand(1, 9)
            model = torch.nn.Linear(3, 5)
            with self.createSummaryWriter() as w:
                w.add_graph(model, dummy_input)  # error

    @skipIfNoTorchVision
    def test_torchvision_smoke(self):
        model_input_shapes = {
            'alexnet': (2, 3, 224, 224),
            'resnet34': (2, 3, 224, 224),
            'resnet152': (2, 3, 224, 224),
            'densenet121': (2, 3, 224, 224),
            'vgg16': (2, 3, 224, 224),
            'vgg19': (2, 3, 224, 224),
            'vgg16_bn': (2, 3, 224, 224),
            'vgg19_bn': (2, 3, 224, 224),
            'mobilenet_v2': (2, 3, 224, 224),
        }
        for model_name, input_shape in model_input_shapes.items():
            with self.createSummaryWriter() as w:
                model = getattr(torchvision.models, model_name)()
                w.add_graph(model, torch.zeros(input_shape))

class TestTensorBoardFigure(BaseTestCase):
    @skipIfNoMatplotlib
    def test_figure(self):
        writer = self.createSummaryWriter()

        figure, axes = plt.figure(), plt.gca()
        circle1 = plt.Circle((0.2, 0.5), 0.2, color='r')
        circle2 = plt.Circle((0.8, 0.5), 0.2, color='g')
        axes.add_patch(circle1)
        axes.add_patch(circle2)
        plt.axis('scaled')
        plt.tight_layout()

        writer.add_figure("add_figure/figure", figure, 0, close=False)
        self.assertTrue(plt.fignum_exists(figure.number))

        writer.add_figure("add_figure/figure", figure, 1)
        if matplotlib.__version__ != '3.3.0':
            self.assertFalse(plt.fignum_exists(figure.number))
        else:
            print("Skipping fignum_exists, see https://github.com/matplotlib/matplotlib/issues/18163")

        writer.close()

    @skipIfNoMatplotlib
    def test_figure_list(self):
        writer = self.createSummaryWriter()

        figures = []
        for i in range(5):
            figure = plt.figure()
            plt.plot([i * 1, i * 2, i * 3], label="Plot " + str(i))
            plt.xlabel("X")
            plt.xlabel("Y")
            plt.legend()
            plt.tight_layout()
            figures.append(figure)

        writer.add_figure("add_figure/figure_list", figures, 0, close=False)
        self.assertTrue(all(plt.fignum_exists(figure.number) is True for figure in figures))  # noqa: F812

        writer.add_figure("add_figure/figure_list", figures, 1)
        if matplotlib.__version__ != '3.3.0':
            self.assertTrue(all(plt.fignum_exists(figure.number) is False for figure in figures))  # noqa: F812
        else:
            print("Skipping fignum_exists, see https://github.com/matplotlib/matplotlib/issues/18163")

        writer.close()

class TestTensorBoardNumpy(BaseTestCase):
    @unittest.skipIf(IS_WINDOWS, "Skipping on windows, see https://github.com/pytorch/pytorch/pull/109349 ")
    @unittest.skipIf(IS_MACOS, "Skipping on mac, see https://github.com/pytorch/pytorch/pull/109349 ")
    def test_scalar(self):
        res = make_np(1.1)
        self.assertIsInstance(res, np.ndarray) and self.assertEqual(res.shape, (1,))
        res = make_np(1 << 64 - 1)  # uint64_max
        self.assertIsInstance(res, np.ndarray) and self.assertEqual(res.shape, (1,))
        res = make_np(np.float16(1.00000087))
        self.assertIsInstance(res, np.ndarray) and self.assertEqual(res.shape, (1,))
        res = make_np(np.float128(1.00008 + 9))
        self.assertIsInstance(res, np.ndarray) and self.assertEqual(res.shape, (1,))
        res = make_np(np.int64(100000000000))
        self.assertIsInstance(res, np.ndarray) and self.assertEqual(res.shape, (1,))

    def test_pytorch_np_expect_fail(self):
        with self.assertRaises(NotImplementedError):
            res = make_np({'pytorch': 1.0})



class TestTensorProtoSummary(BaseTestCase):
    @parametrize(
        "tensor_type,proto_type",
        [
            (torch.float16, DataType.DT_HALF),
            (torch.bfloat16, DataType.DT_BFLOAT16),
        ],
    )
    @skipIfTorchDynamo("Unsuitable test for Dynamo, behavior changes with version")
    def test_half_tensor_proto(self, tensor_type, proto_type):
        float_values = [1.0, 2.0, 3.0]
        actual_proto = tensor_proto(
            "dummy",
            torch.tensor(float_values, dtype=tensor_type),
        ).value[0].tensor
        self.assertSequenceEqual(
            [int_to_half(x) for x in actual_proto.half_val],
            float_values,
        )
        self.assertTrue(actual_proto.dtype == proto_type)

    def test_float_tensor_proto(self):
        float_values = [1.0, 2.0, 3.0]
        actual_proto = (
            tensor_proto("dummy", torch.tensor(float_values)).value[0].tensor
        )
        self.assertEqual(actual_proto.float_val, float_values)
        self.assertTrue(actual_proto.dtype == DataType.DT_FLOAT)

    def test_int_tensor_proto(self):
        int_values = [1, 2, 3]
        actual_proto = (
            tensor_proto("dummy", torch.tensor(int_values, dtype=torch.int32))
            .value[0]
            .tensor
        )
        self.assertEqual(actual_proto.int_val, int_values)
        self.assertTrue(actual_proto.dtype == DataType.DT_INT32)

    def test_scalar_tensor_proto(self):
        scalar_value = 0.1
        actual_proto = (
            tensor_proto("dummy", torch.tensor(scalar_value)).value[0].tensor
        )
        self.assertAlmostEqual(actual_proto.float_val[0], scalar_value)

    def test_complex_tensor_proto(self):
        real = torch.tensor([1.0, 2.0])
        imag = torch.tensor([3.0, 4.0])
        actual_proto = (
            tensor_proto("dummy", torch.complex(real, imag)).value[0].tensor
        )
        self.assertEqual(actual_proto.scomplex_val, [1.0, 3.0, 2.0, 4.0])

    def test_empty_tensor_proto(self):
        actual_proto = tensor_proto("dummy", torch.empty(0)).value[0].tensor
        self.assertEqual(actual_proto.float_val, [])

instantiate_parametrized_tests(TestTensorProtoSummary)

if __name__ == '__main__':
    run_tests()
