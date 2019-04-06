from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from common_utils import TestCase, run_tests

from torch.utils.tensorboard import x2num, SummaryWriter
import torch
import numpy as np
import unittest
from torch.utils.tensorboard import summary
from torch.utils.tensorboard.utils import make_grid, _prepare_video, convert_to_HWC
import pytest
# from torch.utils.tensorboard import SummaryWriter
# from .expect_reader import compare_proto
import os
import sys
import matplotlib.pyplot as plt
# from torch.utils.tensorboard import x2num


class PyTorchNumpyTest(unittest.TestCase):
    def test_pytorch_np(self):
        tensors = [torch.rand(3, 10, 10), torch.rand(1), torch.rand(1, 2, 3, 4, 5)]
        for tensor in tensors:
            # regular tensor
            assert isinstance(x2num.make_np(tensor), np.ndarray)

            # CUDA tensor
            if torch.cuda.device_count() > 0:
                assert isinstance(x2num.make_np(tensor.cuda()), np.ndarray)

            # regular variable
            assert isinstance(x2num.make_np(torch.autograd.Variable(tensor)), np.ndarray)

            # CUDA variable
            if torch.cuda.device_count() > 0:
                assert isinstance(x2num.make_np(torch.autograd.Variable(tensor).cuda()), np.ndarray)

        # python primitive type
        assert(isinstance(x2num.make_np(0), np.ndarray))
        assert(isinstance(x2num.make_np(0.1), np.ndarray))

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
            floats = x2num.make_np(torch.rand((num,)))
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

            ints = x2num.make_np(torch.randint(0, 100, (num,)))
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


class UtilsTest(unittest.TestCase):
    def test_to_HWC(self):
        np.random.seed(1)
        test_image = np.random.randint(0, 256, size=(3, 32, 32), dtype=np.uint8)
        converted = convert_to_HWC(test_image, 'chw')
        assert converted.shape == (32, 32, 3)
        test_image = np.random.randint(0, 256, size=(16, 3, 32, 32), dtype=np.uint8)
        converted = convert_to_HWC(test_image, 'nchw')
        assert converted.shape == (64, 256, 3)
        test_image = np.random.randint(0, 256, size=(32, 32), dtype=np.uint8)
        converted = convert_to_HWC(test_image, 'hw')
        assert converted.shape == (32, 32, 3)

    def test_prepare_video(self):
        # at each timestep the sum over all other dimensions of the video should stay the same
        np.random.seed(1)
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


class WriterTest(unittest.TestCase):
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
            # writer.add_text('Text', 'text logged at step:' + str(n_iter), n_iter)
            # writer.add_text('markdown Text', '''a|b\n-|-\nc|d''', n_iter)
            writer.add_histogram('hist', np.random.rand(100, 100), n_iter)
            # writer.add_pr_curve('xoxo', np.random.randint(2, size=100), np.random.rand(
            #     100), n_iter)  # needs tensorboard 0.4RC or later
            # writer.add_pr_curve_raw('prcurve with raw data', true_positive_counts,
            #                         false_positive_counts,
            #                         true_negative_counts,
            #                         false_negative_counts,
            #                         precision,
            #                         recall, n_iter)
            # export scalar data to JSON for external processing
            writer.export_scalars_to_json("./all_scalars.json")


def test_linting():
    import subprocess
    subprocess.check_output(['flake8', 'tensorboardX'])


class SummaryWriterTest(unittest.TestCase):
    def test_summary_writer_ctx(self):
        # after using a SummaryWriter as a ctx it should be closed
        with SummaryWriter(filename_suffix='.test') as writer:
            writer.add_scalar('test', 1)
        assert writer.file_writer is None

    def test_summary_writer_close(self):
        # Opening and closing SummaryWriter a lot should not run into
        # OSError: [Errno 24] Too many open files
        passed = True
        try:
            writer = SummaryWriter()
            writer.close()
        except OSError:
            passed = False

        assert passed

    def test_windowsPath(self):
        dummyPath = "C:\\Downloads\\fjoweifj02utj43tj430"
        with SummaryWriter(dummyPath) as writer:
            writer.add_scalar('test', 1)
        import shutil
        shutil.rmtree(dummyPath)

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


class EmbeddingTest(unittest.TestCase):
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
np.random.seed(0)
# compare_proto = write_proto  # massive update expect


class SummaryTest(unittest.TestCase):
    def test_uint8_image(self):
        '''
        Tests that uint8 image (pixel values in [0, 255]) is not changed
        '''
        test_image = np.random.randint(0, 256, size=(3, 32, 32), dtype=np.uint8)
        scale_factor = summary._calc_scale_factor(test_image)
        assert scale_factor == 1, 'Values are already in [0, 255], scale factor should be 1'

    def test_float32_image(self):
        '''
        Tests that float32 image (pixel values in [0, 1]) are scaled correctly
        to [0, 255]
        '''
        test_image = np.random.rand(3, 32, 32).astype(np.float32)
        scale_factor = summary._calc_scale_factor(test_image)
        assert scale_factor == 255, 'Values are in [0, 1], scale factor should be 255'

    def test_list_input(self):
        with pytest.raises(Exception) as e_info:
            summary.histogram('dummy', [1, 3, 4, 5, 6], 'tensorflow')

    def test_empty_input(self):
        print('expect error here:')
        with pytest.raises(Exception) as e_info:
            summary.histogram('dummy', np.ndarray(0), 'tensorflow')

    def test_image_with_boxes(self):
        compare_proto(summary.image_boxes('dummy',
                                          np.random.rand(3, 32, 32).astype(np.float32),
                                          np.array([[10, 10, 40, 40]])),
                      self)

    def test_image_with_one_channel(self):
        np.random.seed(0)
        compare_proto(summary.image('dummy', np.random.rand(1, 8, 8).astype(np.float32), dataformats='CHW'), self)

    def test_image_with_one_channel_batched(self):
        np.random.seed(0)
        compare_proto(summary.image('dummy', np.random.rand(2, 1, 8, 8).astype(np.float32), dataformats='NCHW'), self)

    def test_image_with_3_channel_batched(self):
        np.random.seed(0)
        compare_proto(summary.image('dummy', np.random.rand(2, 3, 8, 8).astype(np.float32), dataformats='NCHW'), self)

    def test_image_without_channel(self):
        np.random.seed(0)
        compare_proto(summary.image('dummy', np.random.rand(8, 8).astype(np.float32), dataformats='HW'), self)

    def test_video(self):
        try:
            import moviepy
        except ImportError:
            return
        np.random.seed(0)
        compare_proto(summary.video('dummy', np.random.rand(4, 3, 1, 8, 8).astype(np.float32)), self)
        summary.video('dummy', np.random.rand(16, 48, 1, 28, 28).astype(np.float32))
        summary.video('dummy', np.random.rand(20, 7, 1, 8, 8).astype(np.float32))

    def test_audio(self):
        np.random.seed(0)
        compare_proto(summary.audio('dummy', np.random.rand(42)), self)

    # TODO: add back after new PR in tensorboard landed
    # def test_text(self):
    #     compare_proto(summary.text('dummy', 'text 123'), self)

    def test_histogram_auto(self):
        np.random.seed(0)
        compare_proto(summary.histogram('dummy', np.random.rand(1024), bins='auto', max_bins=5), self)

    def test_histogram_fd(self):
        np.random.seed(0)
        compare_proto(summary.histogram('dummy', np.random.rand(1024), bins='fd', max_bins=5), self)

    def test_histogram_doane(self):
        np.random.seed(0)
        compare_proto(summary.histogram('dummy', np.random.rand(1024), bins='doane', max_bins=5), self)


def removeWhiteChar(string):
    return string.replace(' ', '').replace('\t', '').replace('\n', '')


def compare_proto(str_to_compare, function_ptr):
    # TODO: reable test
    return
    if 'histogram' in function_ptr.id():
        return  # numpy.histogram has slightly different between different version

    if 'pr_curve' in function_ptr.id():
        return  # pr_curve depends on numpy.histogram

    module_id = function_ptr.__class__.__module__
    functionName = function_ptr.id().split('.')[-1]
    test_file = os.path.realpath(sys.modules[module_id].__file__)
    expected_file = os.path.join(os.path.dirname(test_file),
                                 "expect",
                                 module_id.split('.')[-1] + '.' + functionName + ".expect")
    print(expected_file)
    assert os.path.exists(expected_file)
    with open(expected_file) as f:
        expected = f.read()
    str_to_compare = str(str_to_compare)
    print(removeWhiteChar(str_to_compare))
    print(removeWhiteChar(expected))
    assert removeWhiteChar(str_to_compare) == removeWhiteChar(expected)


def write_proto(str_to_compare, function_ptr):
    module_id = function_ptr.__class__.__module__
    functionName = function_ptr.id().split('.')[-1]
    test_file = os.path.realpath(sys.modules[module_id].__file__)
    expected_file = os.path.join(os.path.dirname(test_file),
                                 "expect",
                                 module_id.split('.')[-1] + '.' + functionName + ".expect")
    print(expected_file)
    with open(expected_file, 'w') as f:
        f.write(str(str_to_compare))


class PytorchGraphTest(unittest.TestCase):
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

    def test_wrong_input_size(self):
        print('expect error here:')
        with self.assertRaises(RuntimeError) as e_info:
            dummy_input = torch.rand(1, 9)
            model = torch.nn.Linear(3, 5)
            with SummaryWriter(comment='expect_error') as w:
                w.add_graph(model, dummy_input)  # error


class FigureTest(unittest.TestCase):
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
        assert plt.fignum_exists(figure.number) is True

        writer.add_figure("add_figure/figure", figure, 1)
        assert plt.fignum_exists(figure.number) is False

        writer.close()

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
        assert all([plt.fignum_exists(figure.number) is True for figure in figures])

        writer.add_figure("add_figure/figure_list", figures, 1)
        assert all([plt.fignum_exists(figure.number) is False for figure in figures])

        writer.close()


class NumpyTest(unittest.TestCase):
    def test_scalar(self):
        res = x2num.make_np(1.1)
        assert isinstance(res, np.ndarray) and res.shape == (1,)
        res = x2num.make_np(1 << 64 - 1)  # uint64_max
        assert isinstance(res, np.ndarray) and res.shape == (1,)
        res = x2num.make_np(np.float16(1.00000087))
        assert isinstance(res, np.ndarray) and res.shape == (1,)
        res = x2num.make_np(np.float128(1.00008 + 9))
        assert isinstance(res, np.ndarray) and res.shape == (1,)
        res = x2num.make_np(np.int64(100000000000))
        assert isinstance(res, np.ndarray) and res.shape == (1,)

    def test_make_grid(self):
        pass

    def test_numpy_vid(self):
        shapes = [(16, 3, 30, 28, 28), (19, 3, 30, 28, 28), (19, 3, 29, 23, 19)]
        for s in shapes:
            x = np.random.random_sample(s)
            # assert x2num.make_np(x, 'VID').shape[3] == 3

    def test_numpy_vid_uint8(self):
        x = np.random.randint(0, 256, (16, 3, 30, 28, 28)).astype(np.uint8)
        # x2num.make_np(x, 'VID').shape[3] == 3

if __name__ == '__main__':
    run_tests()
