# Owner(s): ["module: dynamo"]

import torch

import torch._dynamo
import torch._dynamo.backends.ipex
import torch._dynamo.test_case
from torch._dynamo.backends.train_step import (
    _compile_train_step,
    _train_step_compiler,
    _train_step_eager,
    _train_step_inductor,
)
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

        opt_train_step = _compile_train_step(train_step, backend=_train_step_eager)
        opt_loss = opt_train_step(model, inputs)

        self.assertTrue(same(correct_loss, opt_loss))

    def test_dynamo_safety_checks(self):
        """Since dynamo train_step compile traces .backward() call, it's imperative that no .grad_fn exists
        for inputs to the train_step graph, otherwise their backwards will incorrectly be traced
        """

        def train_step(model, inputs):
            out = model(*inputs)
            loss = out.sum()
            loss.backward()
            return loss

        opt_model = Seq()
        opt_model.apply(init_weights)

        # Cause the inputs to the model to have grad_fn
        pre_inputs = torch.randn((128, 10))
        pre_input_layer = torch.nn.Linear(10, 10)
        inputs = [
            pre_input_layer(pre_inputs),
        ]
        opt_train_step = _compile_train_step(train_step, backend=_train_step_eager)

        with self.assertRaisesRegex(AssertionError, r"an input tensor has a grad_fn"):
            opt_train_step(opt_model, inputs)

        def train_step_multi_backward(model, inputs):
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
            train_step_multi_backward, backend=_train_step_eager
        )
        with self.assertRaisesRegex(AssertionError, r"multiple \.backward\(\) calls"):
            opt_train_step(opt_model, inputs)

        def train_step_backward_args(model, inputs):
            out = model(*inputs)
            loss = out.sum()
            loss.backward(loss)

        opt_train_step = _compile_train_step(
            train_step_backward_args, backend=_train_step_eager
        )

        with self.assertRaisesRegex(
            AssertionError, r"\.backward\(\) call with non-empty args"
        ):
            opt_train_step(opt_model, inputs)

    def test_custom_backend(self):
        def train_step(model, inputs):
            out = model(*inputs)
            loss = out.sum()
            loss.backward()
            return loss

        opt_model = Seq()
        opt_model.apply(init_weights)
        inputs = [torch.randn((128, 10))]

        cnt = torch._dynamo.testing.CompileCounterWithBackend("eager")
        train_step_cnt = _train_step_compiler(cnt)
        opt_train_step = _compile_train_step(train_step, backend=train_step_cnt)

        loss = []
        for _ in range(10):
            opt_loss = opt_train_step(opt_model, inputs)
            loss.append(opt_loss)
            # no-op until optimizer is added
            # if step > 0:
            #     # in practice, this model loss goes 684, 458, 264, 125, ... so this check should not be too noisy
            #     self.assertTrue(loss[-2] > loss[-1])

        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 37)

    def test_inductor_backend(self):
        def train_step(model, inputs):
            out = model(*inputs)
            loss = out.sum()
            loss.backward()
            return loss

        opt_model = Seq()
        opt_model.apply(init_weights)
        inputs = [torch.randn((128, 10))]

        torch._dynamo.reset()
        ind_train_step = _compile_train_step(train_step, backend=_train_step_inductor)

        loss = []
        for _ in range(10):
            ind_loss = ind_train_step(opt_model, inputs)
            loss.append(ind_loss)
            # no-op until optimizer is added
            # if step > 0:
            #     self.assertTrue(loss[-2] > loss[-1])

        torch._dynamo.reset()
        with self.assertRaisesRegex(
            RuntimeError, r"_compile_train_step does not support inductor"
        ):
            ind_train_step = _compile_train_step(train_step, backend="inductor")


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
