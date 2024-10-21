# Owner(s): ["module: dynamo"]
"""
PYTEST_DONT_REWRITE (prevents pytest from rewriting assertions, which interferes
with test_adam in OptimizerTests)
"""
import functools

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._dynamo.testing
from torch.nn import Parameter
from torch.optim import (
    Adadelta,
    Adagrad,
    Adam,
    Adamax,
    AdamW,
    NAdam,
    RAdam,
    RMSprop,
    Rprop,
    SGD,
)
from torch.testing._internal.common_utils import gradcheck, load_tests, TestCase


class MyOptimizer(torch.optim.Optimizer):
    def __init__(self, params):
        super().__init__(params, {})

    def _init_group(self, params, group):
        any_complex = False
        for p in group["params"]:
            params.append(p)
            any_complex |= p.is_complex()
        return any_complex

    def step(self):
        for group in self.param_groups:
            params = []
            any_complex = self._init_group(params, group)
            if any_complex:
                params[0] -= 1
            else:
                params[0] += 1


class End2EndTests(torch._dynamo.test_case.TestCase):
    # https://github.com/pytorch/torchdynamo/issues/1604
    def test_optimizing_over_tensor_with_requires_grad(self):
        class Net(torch.nn.Module):
            def forward(self, x, y):
                z = torch.bmm(x, y)
                z = torch.flatten(z, 1)
                return z

        def training_iter_fn(batch, model, optimizer):
            optimizer.zero_grad()
            out = model(**batch)
            target = torch.tensor([0, 7])
            loss = torch.nn.CrossEntropyLoss()(out, target)
            loss.backward()
            optimizer.step()
            return loss

        net = Net()
        input1 = torch.randn(2, 1, 4)
        input2 = torch.randn(2, 4, 8, requires_grad=True)
        optimizer = torch.optim.Adam([input2], lr=0.1)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_training_iter_fn = torch._dynamo.optimize(cnts)(training_iter_fn)
        batch = {"x": input1, "y": input2}
        for _ in range(2):
            opt_training_iter_fn(batch, net, optimizer)
        self.assertEqual(cnts.frame_count, 2)

    def test_state_dict(self):
        @torch.compile(backend="eager")
        def _test_state_dict(weight, bias, input):
            def fn_base(optimizer, weight, bias):
                optimizer.zero_grad()
                i = input
                loss = (weight.mv(i) + bias).pow(2).sum()
                loss.backward()
                return loss

            optimizer = torch.optim.Adagrad([weight, bias])
            fn = functools.partial(fn_base, optimizer, weight, bias)
            return optimizer, fn

        optimizer, fn = _test_state_dict(
            Parameter(torch.randn(10, 5)),
            Parameter(torch.randn(10)),
            torch.randn(5, requires_grad=True),
        )
        optimizer.step(fn)

    def test_init_group(self):
        for dtype in [torch.float32, torch.cfloat]:
            tensor = torch.randn(5, 5, dtype=dtype)
            params = Parameter(tensor.detach().clone(), requires_grad=False)
            opt_params = Parameter(tensor.detach().clone(), requires_grad=False)

            optim = MyOptimizer([params])
            optim.step()

            opt_optim = MyOptimizer([opt_params])
            opt_step = torch.compile(backend="eager", fullgraph=True)(opt_optim.step)
            opt_step()

            self.assertEqual(params, opt_params)


# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests


def _diff_fn(p, grad, opt_differentiable_state, opt_class, kwargs, *ignored):
    # Ignored is the list of values in `opt_differentiable_state`, we do this
    # for `gradcheck` to correctly track the state tensors as function inputs
    # because otherwise it can't unpack the values in the `opt_differentiable_state`
    # dict
    p = p.clone()
    p.grad = grad
    opt_differentiable_state = {
        k: v.clone() if isinstance(v, torch.Tensor) else v
        for k, v in opt_differentiable_state.items()
    }
    opt = opt_class([p], **kwargs)
    opt.state[p].update(opt_differentiable_state)
    opt.step()
    return (p,) + tuple(
        v
        for v in opt.state[p].values()
        if isinstance(v, torch.Tensor) and v.requires_grad
    )


class TestDifferentiableOptimizer(TestCase):
    def test_sgd(self):
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        mbuff = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state = {"momentum_buffer": mbuff}
        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                SGD,
                {"lr": 0.9, "differentiable": True},
                *state.values(),
            ),
        )

    def test_adam(self):
        state = {}
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # `step` is not a continuous variable (even though we define it as a float)
        # and so it shouldn't require gradients.
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        state["exp_avg"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["exp_avg_sq"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["max_exp_avg_sq"] = torch.rand(
            10, requires_grad=True, dtype=torch.float64
        )

        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                Adam,
                {"lr": 0.9, "differentiable": True, "amsgrad": True},
                *state.values(),
            ),
        )

    def test_rmsprop(self):
        state = {}
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["step"] = torch.zeros((), dtype=torch.float64)
        state["square_avg"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["momentum_buffer"] = torch.rand(
            10, requires_grad=True, dtype=torch.float64
        )
        # This can cause issues with large values and nan due to sqrt ops
        state["grad_avg"] = 1e-2 * torch.rand(
            10, requires_grad=True, dtype=torch.float64
        )
        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                RMSprop,
                {
                    "lr": 0.9,
                    "maximize": True,
                    "momentum": 0.9,
                    "differentiable": True,
                    "centered": True,
                    "weight_decay": 0.1,
                },
                *state.values(),
            ),
        )

    def test_adadelta(self):
        state = {}
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # `step` is not a continuous variable (even though we define it as a float)
        # and so it shouldn't require gradients.
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        state["square_avg"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["acc_delta"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                Adadelta,
                {"lr": 0.9, "weight_decay": 0.1, "differentiable": True},
                *state.values(),
            ),
        )

    def test_adagrad(self):
        state = {}
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # `step` is not a continuous variable (even though we define it as a float)
        # and so it shouldn't require gradients.
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        state["sum"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                Adagrad,
                {"lr": 0.9, "weight_decay": 0.1, "differentiable": True},
                *state.values(),
            ),
        )

    def test_adamax(self):
        state = {}
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # `step` is not a continuous variable (even though we define it as a float)
        # and so it shouldn't require gradients.
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        state["exp_avg"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["exp_inf"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                Adamax,
                {"lr": 0.9, "weight_decay": 0.1, "differentiable": True},
                *state.values(),
            ),
        )

    def test_rprop(self):
        state = {}
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # `step` is not a continuous variable (even though we define it as a float)
        # and so it shouldn't require gradients.
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        state["prev"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["step_size"] = torch.rand(10, requires_grad=True, dtype=torch.float64)

        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                Rprop,
                {"lr": 0.9, "differentiable": True},
                *state.values(),
            ),
        )

    def test_adamw(self):
        state = {}
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # `step` is not a continuous variable (even though we define it as a float)
        # and so it shouldn't require gradients.
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        state["exp_avg"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["exp_avg_sq"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["max_exp_avg_sq"] = torch.rand(
            10, requires_grad=True, dtype=torch.float64
        )

        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                AdamW,
                {"lr": 0.9, "differentiable": True, "amsgrad": True},
                *state.values(),
            ),
        )

    def test_nadam(self):
        state = {}
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # `step` is not a continuous variable (even though we define it as a float)
        # and so it shouldn't require gradients.
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        state["exp_avg"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["exp_avg_sq"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["mu_product"] = torch.tensor(1.0, requires_grad=True, dtype=torch.float64)

        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                NAdam,
                {"lr": 0.9, "differentiable": True},
                *state.values(),
            ),
        )

        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                NAdam,
                {"lr": 0.9, "decoupled_weight_decay": True, "differentiable": True},
                *state.values(),
            ),
        )

    def test_radam(self):
        state = {}
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # `step` is not a continuous variable (even though we define it as a float)
        # and so it shouldn't require gradients.
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        state["exp_avg"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["exp_avg_sq"] = torch.rand(10, requires_grad=True, dtype=torch.float64)

        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                RAdam,
                {"lr": 0.9, "differentiable": True},
                *state.values(),
            ),
        )
        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                RAdam,
                {
                    "lr": 0.9,
                    "weight_decay": 0.1,
                    "decoupled_weight_decay": True,
                    "differentiable": True,
                },
                *state.values(),
            ),
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
