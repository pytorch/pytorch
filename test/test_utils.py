import sys
import os
import math
import shutil
import random
import tempfile
import unittest
import sys
import traceback
import torch
import torch.cuda
from torch.autograd import Variable
from torch.utils.trainer import Trainer
from torch.utils.trainer.plugins import *
from torch.utils.trainer.plugins.plugin import Plugin
from torch.utils.data import *

HAS_CUDA = torch.cuda.is_available()

from common import TestCase

try:
    import cffi
    from torch.utils.ffi import compile_extension
    HAS_CFFI = True
except ImportError:
    HAS_CFFI = False

class SimplePlugin(Plugin):
    def __init__(self, interval):
        super(SimplePlugin, self).__init__(interval)
        self.trainer = None
        self.num_iteration = 0
        self.num_epoch = 0
        self.num_batch = 0
        self.num_update = 0

    def register(self, trainer):
        self.trainer = trainer

    def iteration(self, *args):
        self.iteration_args = args
        self.num_iteration += 1

    def epoch(self, *args):
        self.epoch_args = args
        self.num_epoch += 1

    def batch(self, *args):
        self.batch_args = args
        self.num_batch += 1

    def update(self, *args):
        self.update_args = args
        self.num_update += 1


class ModelMock(object):
    def __init__(self):
        self.num_calls = 0
        self.output = Variable(torch.ones(1, 1))

    def __call__(self, i):
        self.num_calls += 1
        return self.output * 2


class CriterionMock(object):
    def __init__(self):
        self.num_calls = 0

    def __call__(self, out, target):
        self.num_calls += 1
        return out


class OptimizerMock(object):
    max_evals = 5
    min_evals = 1

    def __init__(self):
        self.num_steps = 0
        self.num_evals = 0

    def step(self, closure):
        for i in range(random.randint(1, self.max_evals)):
            loss = closure()
            self.num_evals += 1
        loss.backward()
        self.num_steps += 1


class DatasetMock(object):
    def __iter__(self):
        for i in range(10):
            yield torch.randn(2, 10), torch.randperm(10)[:2]

    def __len__(self):
        return 10


class TestTrainer(TestCase):

    intervals = [
        [(1, 'iteration')],
        [(1, 'epoch')],
        [(1, 'batch')],
        [(1, 'update')],
        [(5, 'iteration')],
        [(5, 'epoch')],
        [(5, 'batch')],
        [(5, 'update')],
        [(1, 'iteration'), (1, 'epoch')],
        [(5, 'update'), (1, 'iteration')],
        [(2, 'epoch'), (1, 'batch')],
    ]

    def setUp(self):
        self.trainer = Trainer(ModelMock(), CriterionMock(), OptimizerMock(),
                DatasetMock())
        self.num_epochs = 3
        self.dataset_size = len(self.trainer.dataset)
        self.num_iters = self.num_epochs * self.dataset_size

    def test_register_plugin(self):
        for interval in self.intervals:
            simple_plugin = SimplePlugin(interval)
            self.trainer.register_plugin(simple_plugin)
            self.assertEqual(simple_plugin.trainer, self.trainer)

    def test_optimizer_step(self):
        self.trainer.run(epochs=1)
        self.assertEqual(self.trainer.optimizer.num_steps, 10)

    def test_plugin_interval(self):
        for interval in self.intervals:
            self.setUp()
            simple_plugin = SimplePlugin(interval)
            self.trainer.register_plugin(simple_plugin)
            self.trainer.run(epochs=self.num_epochs)
            units = {
                ('iteration', self.num_iters),
                ('epoch', self.num_epochs),
                ('batch', self.num_iters),
                ('update', self.num_iters)
            }
            for unit, num_triggers in units:
                call_every = None
                for i, i_unit in interval:
                    if i_unit == unit:
                        call_every = i
                        break
                if call_every:
                    expected_num_calls = math.floor(num_triggers / call_every)
                else:
                    expected_num_calls = 0
                num_calls = getattr(simple_plugin, 'num_' + unit)
                self.assertEqual(num_calls, expected_num_calls, 0)

    def test_model_called(self):
        self.trainer.run(epochs=self.num_epochs)
        num_model_calls = self.trainer.model.num_calls
        num_crit_calls = self.trainer.criterion.num_calls
        self.assertEqual(num_model_calls, num_crit_calls)
        for num_calls in [num_model_calls, num_crit_calls]:
            lower_bound = OptimizerMock.min_evals * self.num_iters
            upper_bound = OptimizerMock.max_evals * self.num_iters
            self.assertEqual(num_calls, self.trainer.optimizer.num_evals)
            self.assertLessEqual(lower_bound, num_calls)
            self.assertLessEqual(num_calls, upper_bound)

    def test_model_gradient(self):
        self.trainer.run(epochs=self.num_epochs)
        output_var = self.trainer.model.output
        expected_grad = torch.ones(1, 1) * 2 * self.num_iters
        self.assertEqual(output_var.grad, expected_grad)


class TestTensorDataset(TestCase):

    def test_len(self):
        source = TensorDataset(torch.randn(15, 10, 2, 3, 4, 5), torch.randperm(15))
        self.assertEqual(len(source), 15)

    def test_getitem(self):
        t = torch.randn(15, 10, 2, 3, 4, 5)
        l = torch.randn(15, 10)
        source = TensorDataset(t, l)
        for i in range(15):
            self.assertEqual(t[i], source[i][0])
            self.assertEqual(l[i], source[i][1])

    def test_getitem_1d(self):
        t = torch.randn(15)
        l = torch.randn(15)
        source = TensorDataset(t, l)
        for i in range(15):
            self.assertEqual(t[i:i+1], source[i][0])
            self.assertEqual(l[i:i+1], source[i][1])


class ErrorDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

class TestDataLoader(TestCase):

    def setUp(self):
        self.data = torch.randn(100, 2, 3, 5)
        self.labels = torch.randperm(50).repeat(2)
        self.dataset = TensorDataset(self.data, self.labels)

    def _test_sequential(self, loader):
        batch_size = loader.batch_size
        for i, (sample, target) in enumerate(loader):
            idx = i * batch_size
            self.assertEqual(sample, self.data[idx:idx+batch_size])
            self.assertEqual(target, self.labels[idx:idx+batch_size].view(-1, 1))
        self.assertEqual(i, math.floor((len(self.dataset)-1) / batch_size))

    def _test_shuffle(self, loader):
        found_data = {i: 0 for i in range(self.data.size(0))}
        found_labels = {i: 0 for i in range(self.labels.size(0))}
        batch_size = loader.batch_size
        for i, (batch_samples, batch_targets) in enumerate(loader):
            for sample, target in zip(batch_samples, batch_targets):
                for data_point_idx, data_point in enumerate(self.data):
                    if data_point.eq(sample).all():
                        self.assertFalse(found_data[data_point_idx])
                        found_data[data_point_idx] += 1
                        break
                self.assertEqual(target, self.labels.narrow(0, data_point_idx, 1))
                found_labels[data_point_idx] += 1
            self.assertEqual(sum(found_data.values()), (i+1) * batch_size)
            self.assertEqual(sum(found_labels.values()), (i+1) * batch_size)
        self.assertEqual(i, math.floor((len(self.dataset)-1) / batch_size))

    def _test_error(self, loader):
        it = iter(loader)
        errors = 0
        while True:
            try:
                it.next()
            except NotImplementedError:
                msg = "".join(traceback.format_exception(*sys.exc_info()))
                self.assertTrue("_processBatch" in msg)
                errors += 1
            except StopIteration:
                self.assertEqual(errors,
                    math.ceil(float(len(loader.dataset))/loader.batch_size))
                return


    def test_sequential(self):
        self._test_sequential(DataLoader(self.dataset))

    def test_sequential_batch(self):
        self._test_sequential(DataLoader(self.dataset, batch_size=2))

    def test_shuffle(self):
        self._test_shuffle(DataLoader(self.dataset, shuffle=True))

    def test_shuffle_batch(self):
        self._test_shuffle(DataLoader(self.dataset, batch_size=2, shuffle=True))


    def test_sequential_workers(self):
        # still use test shuffle here because the workers may shuffle the order
        self._test_shuffle(DataLoader(self.dataset, num_workers=4))

    def test_seqential_batch_workers(self):
        # still use test shuffle here because the workers may shuffle the order
        self._test_shuffle(DataLoader(self.dataset, batch_size=2, num_workers=4))

    def test_shuffle_workers(self):
        self._test_shuffle(DataLoader(self.dataset, shuffle=True, num_workers=4))

    def test_shuffle_batch_workers(self):
        self._test_shuffle(DataLoader(self.dataset, batch_size=2, shuffle=True, num_workers=4))

    def test_error(self):
        self._test_error(DataLoader(ErrorDataset(100), batch_size=2, shuffle=True))

    def test_error_workers(self):
        self._test_error(DataLoader(ErrorDataset(41), batch_size=2, shuffle=True, num_workers=4))


test_dir = os.path.abspath(os.path.dirname(str(__file__)))

class TestFFI(TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.chdir(self.tmpdir)
        sys.path.append(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @unittest.skipIf(not HAS_CFFI, "ffi tests require cffi package")
    def test_cpu(self):
        compile_extension(
                name='test_extensions.cpulib',
                header=test_dir + '/ffi/src/cpu/lib.h',
                sources=[
                    test_dir + '/ffi/src/cpu/lib1.c',
                    test_dir + '/ffi/src/cpu/lib2.c',
                ],
                verbose=False,
        )
        from test_extensions import cpulib
        tensor = torch.ones(2, 2).float()

        cpulib.good_func(tensor, 2, 1.5)
        self.assertEqual(tensor, torch.ones(2, 2) * 2 + 1.5)

        new_tensor = cpulib.new_tensor(4)
        self.assertEqual(new_tensor, torch.ones(4, 4) * 4)

        f = cpulib.int_to_float(5)
        self.assertIs(type(f), float)

        self.assertRaises(TypeError,
                lambda: cpulib.good_func(tensor.double(), 2, 1.5))
        self.assertRaises(torch.FatalError,
                lambda: cpulib.bad_func(tensor, 2, 1.5))

    @unittest.skipIf(not HAS_CFFI or not HAS_CUDA, "ffi tests require cffi package")
    def test_gpu(self):
        compile_extension(
                name='gpulib',
                header=test_dir + '/ffi/src/cuda/cudalib.h',
                sources=[
                    test_dir + '/ffi/src/cuda/cudalib.c',
                ],
                with_cuda=True,
                verbose=False,
        )
        import gpulib
        tensor = torch.ones(2, 2).float()

        gpulib.good_func(tensor, 2, 1.5)
        self.assertEqual(tensor, torch.ones(2, 2) * 2 + 1.5)

        ctensor = tensor.cuda().fill_(1)
        gpulib.cuda_func(ctensor, 2, 1.5)
        self.assertEqual(ctensor, torch.ones(2, 2) * 2 + 1.5)

        self.assertRaises(TypeError,
                lambda: gpulib.cuda_func(tensor, 2, 1.5))
        self.assertRaises(TypeError,
                lambda: gpulib.cuda_func(ctensor.storage(), 2, 1.5))


if __name__ == '__main__':
    unittest.main()

