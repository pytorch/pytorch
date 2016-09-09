import math
import random
import unittest
import torch
from torch.autograd import Variable
from torch.utils.trainer import Trainer
from torch.utils.trainer.plugins import *
from torch.utils.trainer.plugins.plugin import Plugin
from torch.utils.data import *

from common import TestCase

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


class TestTensorDataSource(TestCase):

    def test_len(self):
        source = TensorDataSource(torch.randn(15, 10, 2, 3, 4, 5), torch.randperm(15))
        self.assertEqual(len(source), 15)

    def test_getitem(self):
        t = torch.randn(15, 10, 2, 3, 4, 5)
        l = torch.randn(15, 10)
        source = TensorDataSource(t, l)
        for i in range(15):
            self.assertEqual(t[i], source[i][0])
            self.assertEqual(l[i], source[i][1])

    def test_getitem_1d(self):
        t = torch.randn(15)
        l = torch.randn(15)
        source = TensorDataSource(t, l)
        for i in range(15):
            self.assertEqual(t[i:i+1], source[i][0])
            self.assertEqual(l[i:i+1], source[i][1])


class TestDataset(TestCase):

    def setUp(self):
        self.data = torch.randn(10, 2, 3, 5)
        self.labels = torch.randperm(5).repeatTensor(2)
        self.datasource = TensorDataSource(self.data, self.labels)

    def _test_sequential(self, dataset):
        batch_size = dataset.batch_size
        for i, (sample, target) in enumerate(dataset):
            idx = i * batch_size
            self.assertEqual(sample, self.data[idx:idx+batch_size])
            self.assertEqual(target, self.labels[idx:idx+batch_size].view(-1, 1))
        self.assertEqual(i, math.floor((len(self.datasource)-1) / batch_size))

    def _test_shuffle(self, dataset):
        batch_size = dataset.batch_size
        found_data = {i: 0 for i in range(self.data.size(0))}
        found_labels = {i: 0 for i in range(self.labels.size(0))}
        for i, (batch_samples, batch_targets) in enumerate(dataset):
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
        self.assertEqual(i, math.floor((len(self.datasource)-1) / batch_size))

    def test_seqential(self):
        self._test_sequential(Dataset(self.datasource))

    def test_seqential_batch(self):
        self._test_sequential(Dataset(self.datasource, batch_size=2))

    def test_shuffle(self):
        self._test_shuffle(Dataset(self.datasource, shuffle=True))

    def test_shuffle_batch(self):
        self._test_shuffle(Dataset(self.datasource, batch_size=2, shuffle=True))

    def test_types(self):
        dataset = Dataset(self.datasource, batch_size=2)
        for samples, targets in dataset:
            self.assertIs(type(samples), torch.DoubleTensor)
            self.assertIs(type(targets), torch.DoubleTensor)
        dataset.input_type(torch.FloatTensor)
        for samples, targets in dataset:
            self.assertIs(type(samples), torch.FloatTensor)
            self.assertIs(type(targets), torch.DoubleTensor)
        dataset.target_type(torch.IntTensor)
        for samples, targets in dataset:
            self.assertIs(type(samples), torch.FloatTensor)
            self.assertIs(type(targets), torch.IntTensor)


if __name__ == '__main__':
    unittest.main()

