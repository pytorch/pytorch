# Owner(s): ["module: dynamo"]
import unittest
from copy import deepcopy

import torch

import torch._dynamo
import torch._dynamo.backends.ipex
import torch._dynamo.test_case
from torch._dynamo.backends.train_step import _compile_train_step
from torch._dynamo.testing import same


class Seq(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class TestCompileTrainStep(torch._dynamo.test_case.TestCase):
    """
    The Basic Idea
    1) dynamo stashes optimizer on graphmodule in special key
    2) optimizer.step() just sticks a call in graph without trying to arg-proxy anything
    3) inside train_step_compiler, we reparameterize the optimizer

    WIP/Issues
    - handle more than one optimizer (e.g. for different submodules)
    """

    def test_no_optimizer(self):
        def train_step(model, inputs):
            out = model(*inputs)
            loss = out.sum()
            loss.backward()
            return loss

        model = Seq()
        model.apply(init_weights)
        inputs = [torch.randn((128, 10))]

        correct_loss = train_step(model, inputs)

        opt_train_step = _compile_train_step(train_step, backend="train_step_eager")
        opt_loss = opt_train_step(model, inputs)

        self.assertTrue(same(correct_loss, opt_loss))

    def test_sgd_optimizer(self):
        model = Seq()
        model.apply(init_weights)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        def train_step(model, optimizer, inputs):
            out = model(*inputs)
            loss = out.sum()

            # dynamo special case lets this pass through as a call in the FX graph
            # it gets traced out in the train_step backend, before being functionalized
            loss.backward()

            # dynamo tracks the optimizer and smuggles it on the graphmodule as an attr
            # train_step backend can reparametrize it with the fake parameter tensors it already
            # uses for module tracing
            optimizer.step()

            optimizer.zero_grad()
            return loss

        # copy the model/optimizer up front so we don't have to reset them between eager/compile runs
        opt_model = deepcopy(model)
        opt_optimizer = deepcopy(optimizer)
        inputs = [torch.randn((128, 10))]

        correct_loss = train_step(model, optimizer, inputs)
        correct_params = {
            name: param.clone().detach() for name, param in model.named_parameters()
        }

        opt_train_step = _compile_train_step(train_step, backend="train_step_eager")
        opt_loss = opt_train_step(opt_model, opt_optimizer, inputs)
        opt_params = {
            name: param.clone().detach() for name, param in opt_model.named_parameters()
        }

        self.assertTrue(same(correct_loss, opt_loss))
        for name in correct_params:
            self.assertTrue(name in opt_params)
            self.assertTrue(same(correct_params[name], opt_params[name]))

            # Note: the train_step compiler never sets .grad on the original param objects due to how it traces,
            # so we insist that the user puts .zero_grad in the train_step so there is no discrepancy between running
            # under eager or under compile
            self.assertTrue(correct_params[name].grad is None)
            self.assertTrue(opt_params[name].grad is None)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_adam_optimizer(self):
        model = Seq().cuda()
        model.apply(init_weights)

        optimizer = torch.optim.Adam(model.parameters(), capturable=True)

        def train_step(model, optimizer, inputs):
            out = model(*inputs)
            loss = out.sum()

            # dynamo special case lets this pass through as a call in the FX graph
            # it gets traced out in the train_step backend, before being functionalized
            loss.backward()

            # dynamo tracks the optimizer and smuggles it on the graphmodule as an attr
            # train_step backend can reparametrize it with the fake parameter tensors it already
            # uses for module tracing
            optimizer.step()

            optimizer.zero_grad()
            return loss

        # copy the model/optimizer up front so we don't have to reset them between eager/compile runs
        opt_model = deepcopy(model)
        opt_optimizer = deepcopy(optimizer)
        inputs = [torch.randn((128, 10)).cuda()]

        opt_train_step = _compile_train_step(train_step, backend="train_step_eager")
        for step in range(10):
            correct_loss = train_step(model, optimizer, inputs)
            opt_loss = opt_train_step(opt_model, opt_optimizer, inputs)
            self.assertEqual(correct_loss, opt_loss)

        correct_params = {
            name: param.clone().detach() for name, param in model.named_parameters()
        }

        opt_params = {
            name: param.clone().detach() for name, param in opt_model.named_parameters()
        }

        self.assertTrue(same(correct_loss, opt_loss))
        for name in correct_params:
            self.assertTrue(name in opt_params)
            self.assertTrue(same(correct_params[name], opt_params[name]))

            # Note: the train_step compiler never sets .grad on the original param objects due to how it traces,
            # so we insist that the user puts .zero_grad in the train_step so there is no discrepancy between running
            # under eager or under compile
            self.assertTrue(correct_params[name].grad is None)
            self.assertTrue(opt_params[name].grad is None)

    def test_smoke(self):
        def train_step(model, optimizer, inputs):
            out = model(*inputs)
            loss = out.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            return loss

        opt_model = Seq()
        opt_model.apply(init_weights)
        opt_optimizer = torch.optim.SGD(opt_model.parameters(), lr=0.01, momentum=0.9)
        inputs = [torch.randn((128, 10))]
        opt_train_step = _compile_train_step(train_step, backend="train_step_eager")

        loss = []
        for step in range(10):
            opt_loss = opt_train_step(opt_model, opt_optimizer, inputs)
            loss.append(opt_loss)
            if step > 0:
                # in practice, this model loss goes 684, 458, 264, 125, ... so this check should not be too noisy
                self.assertTrue(loss[-2] > loss[-1])

    def test_dynamo_safety_checks(self):
        """Since dynamo train_step compile traces .backward() call, it's imperative that no .grad_fn exists
        for inputs to the train_step graph, otherwise their backwards will incorrectly be traced
        """

        def train_step(model, optimizer, inputs):
            out = model(*inputs)
            loss = out.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            return loss

        opt_model = Seq()
        opt_model.apply(init_weights)
        opt_optimizer = torch.optim.SGD(opt_model.parameters(), lr=0.01, momentum=0.9)

        # Cause the inputs to the model to have grad_fn
        pre_inputs = torch.randn((128, 10))
        pre_input_layer = torch.nn.Linear(10, 10)
        inputs = [
            pre_input_layer(pre_inputs),
        ]
        opt_train_step = _compile_train_step(train_step, backend="train_step_eager")

        with self.assertRaisesRegex(AssertionError, r"an input tensor has a grad_fn"):
            opt_train_step(opt_model, opt_optimizer, inputs)

        def train_step_multi_backward(model, optimizer, inputs):
            out = model(*inputs)
            loss = out.sum()
            loss.backward()
            # ok, not a real double backward, but either way would cause the same assert to fire
            loss.backward()
            return loss

        inputs = [
            torch.randn((128, 10)),
        ]
        opt_train_step = _compile_train_step(
            train_step_multi_backward, backend="train_step_eager"
        )
        with self.assertRaisesRegex(AssertionError, r"multiple \.backward\(\) calls"):
            opt_train_step(opt_model, opt_optimizer, inputs)

        def train_step_backward_args(model, optimizer, inputs):
            out = model(*inputs)
            loss = out.sum()
            loss.backward(loss)

        opt_train_step = _compile_train_step(
            train_step_backward_args, backend="train_step_eager"
        )

        with self.assertRaisesRegex(
            AssertionError, r"\.backward\(\) call with non-empty args"
        ):
            opt_train_step(opt_model, opt_optimizer, inputs)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
