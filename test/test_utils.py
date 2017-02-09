from __future__ import print_function
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
import warnings
from torch.autograd import Variable
from torch.utils.trainer import Trainer
from torch.utils.trainer.plugins import *
from torch.utils.trainer.plugins.plugin import Plugin
from torch.utils.serialization import load_lua

HAS_CUDA = torch.cuda.is_available()

from common import TestCase, run_tests, download_file

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
        self.output = Variable(torch.ones(1, 1), requires_grad=True)

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
        for i in range(random.randint(self.min_evals, self.max_evals)):
            loss = closure()
            self.num_evals += 1
        self.num_steps += 1

    def zero_grad(self):
        pass


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
        self.optimizer = OptimizerMock()
        self.trainer = Trainer(ModelMock(), CriterionMock(),
                               self.optimizer, DatasetMock())
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
        expected_grad = torch.ones(1, 1) * 2 * self.optimizer.num_evals
        self.assertEqual(output_var.grad.data, expected_grad)


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


class TestLuaReader(TestCase):

    @staticmethod
    def _module_test(name, test):
        def do_test(self):
            module = test['module']
            input = test['input']
            grad_output = test['grad_output']
            if hasattr(self, '_transform_' + name):
                input = getattr(self, '_transform_' + name)(input)
            output = module.forward(input)
            module.zeroGradParameters()
            grad_input = module.backward(input, grad_output)
            self.assertEqual(output, test['output'])
            self.assertEqual(grad_input, test['grad_input'])
            if module.parameters() is not None:
                params, d_params = module.parameters()
                self.assertEqual(params, test['params'])
                self.assertEqual(d_params, test['d_params'])
            else:
                self.assertFalse('params' in test and test['params'])
                self.assertFalse('params' in test and test['d_params'])
        return do_test

    @staticmethod
    def _criterion_test(name, test):
        def do_test(self):
            module = test['module']
            input = test['input']
            if name == 'L1Cost':
                target = None
            else:
                target = test['target']
            if hasattr(self, '_transform_' + name):
                input, target = getattr(self, '_transform_' + name)(input, target)

            output = module.forward(input, target)
            grad_input = module.backward(input, target)
            self.assertEqual(output, test['loss'])
            self.assertEqual(grad_input, test['grad_input'])
        return do_test

    @classmethod
    def init(cls):
        DATA_URL = 'https://s3.amazonaws.com/pytorch/legacy_modules.t7'
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        test_file_path = os.path.join(data_dir, 'legacy_modules.t7')
        succ = download_file(DATA_URL, test_file_path)
        if not succ:
            warnings.warn(("Couldn't download the test file for TestLuaReader! "
                           "Tests will be incomplete!"), RuntimeWarning)
            return

        tests = load_lua(test_file_path)
        for name, test in tests['modules'].items():
            test_name = 'test_' + name.replace('nn.', '')
            setattr(cls, test_name, cls._module_test(name, test))
        for name, test in tests['criterions'].items():
            test_name = 'test_' + name.replace('nn.', '')
            setattr(cls, test_name, cls._criterion_test(name, test))

    def _transform_Index(self, input):
        return [input[0], input[1].sub(1)]

    def _transform_LookupTable(self, input):
        return input.sub(1)

    def _transform_MultiLabelMarginCriterion(self, input, target):
        return input, target.sub(1)

    def _transform_ClassNLLCriterion(self, input, target):
        return input, target.sub(1)

    def _transform_SpatialClassNLLCriterion(self, input, target):
        return input, target.sub(1)

    def _transform_ClassSimplexCriterion(self, input, target):
        return input, target.sub(1)

    def _transform_CrossEntropyCriterion(self, input, target):
        return input, target.sub(1)

    def _transform_ParallelCriterion(self, input, target):
        return input, [target[0].sub(1), target[1]]

    def _transform_MultiCriterion(self, input, target):
        return input, target.sub(1)

    def _transform_MultiMarginCriterion(self, input, target):
        return input, target.sub(1)


TestLuaReader.init()
if __name__ == '__main__':
    run_tests()
