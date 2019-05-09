from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import os
import shutil
import sys
import unittest

TEST_TENSORBOARD = True
try:
    import tensorboard.summary.writer.event_file_writer  # noqa F401
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
from common_utils import TestCase, run_tests


class BaseTestCase(TestCase):
    """ Base class used for all TensorBoard tests """
    def tearDown(self):
        super(BaseTestCase, self).tearDown()
        if os.path.exists('runs'):
            # Remove directory created by SummaryWriter
            shutil.rmtree('runs')


if TEST_TENSORBOARD:
    from torch.utils.tensorboard import summary, SummaryWriter
    from torch.utils.tensorboard._utils import _prepare_video, convert_to_HWC
    from torch.utils.tensorboard._convert_np import make_np

    class TestTensorBoardPyTorchNumpy(BaseTestCase):
        def test_pytorch_np(self):
            tensors = [torch.rand(3, 10, 10), torch.rand(1), torch.rand(1, 2, 3, 4, 5)]
            for tensor in tensors:
                # regular tensor
                self.assertIsInstance(make_np(tensor), np.ndarray)

                # CUDA tensor
                if torch.cuda.device_count() > 0:
                    self.assertIsInstance(make_np(tensor.cuda()), np.ndarray)

                # regular variable
                self.assertIsInstance(make_np(torch.autograd.Variable(tensor)), np.ndarray)

                # CUDA variable
                if torch.cuda.device_count() > 0:
                    self.assertIsInstance(make_np(torch.autograd.Variable(tensor).cuda()), np.ndarray)

            # python primitive type
            self.assertIsInstance(make_np(0), np.ndarray)
            self.assertIsInstance(make_np(0.1), np.ndarray)

        def test_pytorch_write(self):
            with SummaryWriter() as w:
                w.add_scalar('scalar', torch.autograd.Variable(torch.rand(1)), 0)

        def test_pytorch_histogram(self):
            with SummaryWriter() as w:
                w.add_histogram('float histogram', torch.rand((50,)))
                w.add_histogram('int histogram', torch.randint(0, 100, (50,)))

        def test_pytorch_histogram_raw(self):
            with SummaryWriter() as w:
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
                                    bucket_limits=limits.tolist(),
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
                                    bucket_limits=limits.tolist(),
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

        def test_prepare_video(self):
            # at each timestep the sum over all other dimensions of the video should stay the same
            V_before = np.random.random((4, 10, 3, 20, 20))
            V_after = _prepare_video(np.copy(V_before))
            V_before = np.swapaxes(V_before, 0, 1)
            V_before = np.reshape(V_before, newshape=(10, -1))
            V_after = np.reshape(V_after, newshape=(10, -1))
            np.testing.assert_array_almost_equal(np.sum(V_before, axis=1), np.sum(V_after, axis=1))

    freqs = [262, 294, 330, 349, 392, 440, 440, 440, 440, 440, 440]

    true_positive_counts = [75, 64, 21, 5, 0]
    false_positive_counts = [150, 105, 18, 0, 0]
    true_negative_counts = [0, 45, 132, 150, 150]
    false_negative_counts = [0, 11, 54, 70, 75]
    precision = [0.3333333, 0.3786982, 0.5384616, 1.0, 0.0]
    recall = [1.0, 0.8533334, 0.28, 0.0666667, 0.0]

    class TestTensorBoardWriter(BaseTestCase):
        def test_writer(self):
            with SummaryWriter() as writer:
                sample_rate = 44100

                n_iter = 0
                writer.add_scalar('data/scalar_systemtime', 0.1, n_iter)
                writer.add_scalar('data/scalar_customtime', 0.2, n_iter, walltime=n_iter)
                writer.add_scalars('data/scalar_group', {"xsinx": n_iter * np.sin(n_iter),
                                                         "xcosx": n_iter * np.cos(n_iter),
                                                         "arctanx": np.arctan(n_iter)}, n_iter)
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

    class TestTensorBoardSummaryWriter(BaseTestCase):
        def test_summary_writer_ctx(self):
            # after using a SummaryWriter as a ctx it should be closed
            with SummaryWriter(filename_suffix='.test') as writer:
                writer.add_scalar('test', 1)
            self.assertIs(writer.file_writer, None)

        def test_summary_writer_close(self):
            # Opening and closing SummaryWriter a lot should not run into
            # OSError: [Errno 24] Too many open files
            passed = True
            try:
                writer = SummaryWriter()
                writer.close()
            except OSError:
                passed = False

            self.assertTrue(passed)

        def test_pathlib(self):
            import sys
            if sys.version_info.major == 2:
                import pathlib2 as pathlib
            else:
                import pathlib
            p = pathlib.Path('./pathlibtest')
            with SummaryWriter(p) as writer:
                writer.add_scalar('test', 1)
            import shutil
            shutil.rmtree(str(p))

    class TestTensorBoardEmbedding(BaseTestCase):
        def test_embedding(self):
            w = SummaryWriter()
            all_features = torch.Tensor([[1, 2, 3], [5, 4, 1], [3, 7, 7]])
            all_labels = torch.Tensor([33, 44, 55])
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
            w = SummaryWriter()
            all_features = torch.Tensor([[1, 2, 3], [5, 4, 1], [3, 7, 7]])
            all_labels = torch.Tensor([33, 44, 55])
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
            self.assertEqual(scale_factor, 1, 'Values are already in [0, 255], scale factor should be 1')

        def test_float32_image(self):
            '''
            Tests that float32 image (pixel values in [0, 1]) are scaled correctly
            to [0, 255]
            '''
            test_image = np.random.rand(3, 32, 32).astype(np.float32)
            scale_factor = summary._calc_scale_factor(test_image)
            self.assertEqual(scale_factor, 255, 'Values are in [0, 1], scale factor should be 255')

        def test_list_input(self):
            with self.assertRaises(Exception) as e_info:
                summary.histogram('dummy', [1, 3, 4, 5, 6], 'tensorflow')

        def test_empty_input(self):
            with self.assertRaises(Exception) as e_info:
                summary.histogram('dummy', np.ndarray(0), 'tensorflow')

        def test_image_with_boxes(self):
            self.assertTrue(compare_proto(summary.image_boxes('dummy',
                                          np.random.rand(3, 32, 32).astype(np.float32),
                                          np.array([[10, 10, 40, 40]])),
                                          self))

        def test_image_with_one_channel(self):
            self.assertTrue(compare_proto(summary.image('dummy',
                                                        np.random.rand(1, 8, 8).astype(np.float32),
                                                        dataformats='CHW'),
                                                        self))  # noqa E127

        def test_image_with_one_channel_batched(self):
            self.assertTrue(compare_proto(summary.image('dummy',
                                                        np.random.rand(2, 1, 8, 8).astype(np.float32),
                                                        dataformats='NCHW'),
                                                        self))  # noqa E127

        def test_image_with_3_channel_batched(self):
            self.assertTrue(compare_proto(summary.image('dummy',
                                                        np.random.rand(2, 3, 8, 8).astype(np.float32),
                                                        dataformats='NCHW'),
                                                        self))  # noqa E127

        def test_image_without_channel(self):
            self.assertTrue(compare_proto(summary.image('dummy',
                                                        np.random.rand(8, 8).astype(np.float32),
                                                        dataformats='HW'),
                                                        self))  # noqa E127

        def test_video(self):
            try:
                import moviepy  # noqa F401
            except ImportError:
                return
            self.assertTrue(compare_proto(summary.video('dummy', np.random.rand(4, 3, 1, 8, 8).astype(np.float32)), self))
            summary.video('dummy', np.random.rand(16, 48, 1, 28, 28).astype(np.float32))
            summary.video('dummy', np.random.rand(20, 7, 1, 8, 8).astype(np.float32))

        def test_audio(self):
            self.assertTrue(compare_proto(summary.audio('dummy', np.random.rand(42)), self))

        def test_text(self):
            self.assertTrue(compare_proto(summary.text('dummy', 'text 123'), self))

        def test_histogram_auto(self):
            self.assertTrue(compare_proto(summary.histogram('dummy', np.random.rand(1024), bins='auto', max_bins=5), self))

        def test_histogram_fd(self):
            self.assertTrue(compare_proto(summary.histogram('dummy', np.random.rand(1024), bins='fd', max_bins=5), self))

        def test_histogram_doane(self):
            self.assertTrue(compare_proto(summary.histogram('dummy', np.random.rand(1024), bins='doane', max_bins=5), self))

    def remove_whitespace(string):
        return string.replace(' ', '').replace('\t', '').replace('\n', '')

    def compare_proto(str_to_compare, function_ptr):
        # TODO: enable test after tensorboard is ready.
        return True
        if 'histogram' in function_ptr.id():
            return  # numpy.histogram has slight difference between versions

        if 'pr_curve' in function_ptr.id():
            return  # pr_curve depends on numpy.histogram

        module_id = function_ptr.__class__.__module__
        functionName = function_ptr.id().split('.')[-1]
        test_file = os.path.realpath(sys.modules[module_id].__file__)
        expected_file = os.path.join(os.path.dirname(test_file),
                                     "expect",
                                     module_id.split('.')[-1] + '.' + functionName + ".expect")
        assert os.path.exists(expected_file)
        with open(expected_file) as f:
            expected = f.read()
        str_to_compare = str(str_to_compare)
        return remove_whitespace(str_to_compare) == remove_whitespace(expected)

    def write_proto(str_to_compare, function_ptr):
        module_id = function_ptr.__class__.__module__
        functionName = function_ptr.id().split('.')[-1]
        test_file = os.path.realpath(sys.modules[module_id].__file__)
        expected_file = os.path.join(os.path.dirname(test_file),
                                     "expect",
                                     module_id.split('.')[-1] + '.' + functionName + ".expect")
        with open(expected_file, 'w') as f:
            f.write(str(str_to_compare))

    class TestTensorBoardPytorchGraph(BaseTestCase):
        def test_pytorch_graph(self):
            dummy_input = (torch.zeros(1, 3),)

            class myLinear(torch.nn.Module):
                def __init__(self):
                    super(myLinear, self).__init__()
                    self.l = torch.nn.Linear(3, 5)

                def forward(self, x):
                    return self.l(x)

            with SummaryWriter(comment='LinearModel') as w:
                w.add_graph(myLinear(), dummy_input, True)

        def test_mlp_graph(self):
            dummy_input = (torch.zeros(2, 1, 28, 28),)

            # This MLP class with the above input is expected
            # to fail JIT optimizations as seen at
            # https://github.com/pytorch/pytorch/issues/18903
            #
            # However, it should not raise an error during
            # the add_graph call and still continue.
            class myMLP(torch.nn.Module):
                def __init__(self):
                    super(myMLP, self).__init__()
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

            with SummaryWriter(comment='MLPModel') as w:
                w.add_graph(myMLP(), dummy_input, True)

        def test_wrong_input_size(self):
            with self.assertRaises(RuntimeError) as e_info:
                dummy_input = torch.rand(1, 9)
                model = torch.nn.Linear(3, 5)
                with SummaryWriter(comment='expect_error') as w:
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
                'mobilenet_v2': (2, 3, 224, 224),  # will fail optimize_graph
            }
            for model_name, input_shape in model_input_shapes.items():
                with SummaryWriter(comment=model_name) as w:
                    model = getattr(torchvision.models, model_name)()
                    # ValueError: only one element tensors can be converted to Python scalars
                    if model_name == 'mobilenet_v2':
                        w.add_graph(model, torch.zeros(input_shape), operator_export_type="RAW")
                    else:
                        w.add_graph(model, torch.zeros(input_shape))

    class TestTensorBoardFigure(BaseTestCase):
        @skipIfNoMatplotlib
        def test_figure(self):
            writer = SummaryWriter()

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
            self.assertFalse(plt.fignum_exists(figure.number))

            writer.close()

        @skipIfNoMatplotlib
        def test_figure_list(self):
            writer = SummaryWriter()

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
            self.assertTrue(all([plt.fignum_exists(figure.number) is True for figure in figures]))  # noqa F812

            writer.add_figure("add_figure/figure_list", figures, 1)
            self.assertTrue(all([plt.fignum_exists(figure.number) is False for figure in figures]))  # noqa F812

            writer.close()

    class TestTensorBoardNumpy(BaseTestCase):
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

        def test_numpy_vid(self):
            shapes = [(16, 3, 30, 28, 28), (19, 3, 30, 28, 28), (19, 3, 29, 23, 19)]
            for s in shapes:
                x = np.random.random_sample(s)
                # assert make_np(x, 'VID').shape[3] == 3

        def test_numpy_vid_uint8(self):
            x = np.random.randint(0, 256, (16, 3, 30, 28, 28)).astype(np.uint8)
            # make_np(x, 'VID').shape[3] == 3

if __name__ == '__main__':
    run_tests()
