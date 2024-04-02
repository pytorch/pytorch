# Owner(s): ["module: optimizer"]

import functools

import torch
from torch.nn import Parameter
from torch.optim import (
    Adadelta, Adagrad, Adam, Adamax, AdamW, ASGD, NAdam, RAdam, RMSprop, Rprop, SGD, SparseAdam
)
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.testing._internal.common_utils import (
    TestCase,
    load_tests,
    gradcheck,
    skipIfTorchDynamo
)



# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests


def rosenbrock(tensor):
    assert tensor.size() == torch.Size([2]), f"Requires tensor with 2 scalars but got {tensor.size()}"
    x, y = tensor
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def drosenbrock(tensor):
    assert tensor.size() == torch.Size([2]), f"Requires tensor with 2 scalars but got {tensor.size()}"
    x, y = tensor
    return torch.tensor((-400 * x * (y - x**2) - 2 * (1 - x), 200 * (y - x**2)))

@skipIfTorchDynamo("This is a TEMPORARY stopgap, see https://github.com/pytorch/pytorch/issues/103322")
class TestOptim(TestCase):
    exact_dtype = True

    def _test_rosenbrock_sparse(
        self,
        constructor,
        scheduler_constructors=None,
        sparse_only=False,
        maximize=False,
        multi_tensor=False
    ):
        if scheduler_constructors is None:
            scheduler_constructors = []
        # For rosenbrock tests, it is mandated that the param is a tensor with 2 numbers
        if multi_tensor:
            params_t = [torch.tensor([1.5, 1.5]), torch.tensor([1.5, 1.5], dtype=torch.float64)]
        else:
            params_t = [torch.tensor([1.5, 1.5])]

        params = [Parameter(param_t) for param_t in params_t]
        optimizer = constructor(params)
        schedulers = []
        for scheduler_constructor in scheduler_constructors:
            schedulers.append(scheduler_constructor(optimizer))

        if not sparse_only:
            params_c = [Parameter(param_t.clone()) for param_t in params_t]
            optimizer_c = constructor(params_c)

        solution = torch.tensor([1, 1])
        with torch.no_grad():
            initial_dist = sum([param.dist(solution) for param in params])

        def get_grad(param, sparse_grad):
            grad = drosenbrock(param)
            # NB: We torture test the optimizer by returning an
            # uncoalesced sparse tensor

            # Depending on w, provide only the x or y gradient
            if sparse_grad:
                if w:
                    i = torch.LongTensor([[0, 0]])
                    x = grad[0]
                    v = torch.tensor([x / 4.0, x - x / 4.0])
                else:
                    i = torch.LongTensor([[1, 1]])
                    y = grad[1]
                    v = torch.tensor([y - y / 4.0, y / 4.0])
                grad_out = torch.sparse_coo_tensor(i, v, (2,), dtype=v.dtype)
            else:
                if w:
                    grad_out = torch.tensor([grad[0], 0], dtype=param.dtype)
                else:
                    grad_out = torch.tensor([0, grad[1]], dtype=param.dtype)
            return grad_out

        def eval(params, sparse_grad, w):
            optimizer.zero_grad()
            if multi_tensor:
                loss = sum(rosenbrock(param) for param in params)
            else:
                loss = rosenbrock(params[0])
            loss.backward()

            grads_out = [get_grad(param, sparse_grad) for param in params]
            with torch.no_grad():
                params[0].grad = grads_out[0]
                if multi_tensor:
                    params[1].grad = grads_out[1].to(dtype=torch.float64)
            return loss

        for i in range(2000):
            # Do cyclic coordinate descent
            w = i % 2
            optimizer.step(functools.partial(eval, params, True, w))
            for scheduler in schedulers:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(rosenbrock(params[0]))
                else:
                    scheduler.step()
            if not sparse_only:
                optimizer_c.step(functools.partial(eval, params_c, False, w))
                # Tolerance is increased due to floating point error from different
                # code path for dense case: x v.s. x - x / 4.0 + x / 4.0
                self.assertEqual(params, params_c, atol=5e-6, rtol=5e-6)

        if not maximize:
            self.assertLessEqual(
                sum([param.dist(solution) for param in params]),
                initial_dist
            )
        else:
            self.assertGreaterEqual(
                sum([rosenbrock(param) for param in params]),
                sum([rosenbrock(param_t) for param_t in params_t]),
            )


    def test_sgd_sparse(self):
        for foreach in (False, True):
            self._test_rosenbrock_sparse(
                lambda params: SGD(params, lr=4.8e-3, foreach=foreach),
                multi_tensor=foreach,
            )
            self._test_rosenbrock_sparse(
                lambda params: SGD(params, lr=0.0048, foreach=foreach),
                scheduler_constructors=[lambda opt: StepLR(opt, gamma=0.99999, step_size=300)],
                multi_tensor=foreach,
            )


    def test_sparse_adam(self):
        self._test_rosenbrock_sparse(
            lambda params: SparseAdam(params, lr=4e-2), [], True
        )
        self._test_rosenbrock_sparse(
            lambda params: SparseAdam(params, lr=4e-2, maximize=True),
            scheduler_constructors=[],
            sparse_only=True,
            maximize=True,
        )


    def test_adagrad_sparse(self):
        for foreach in (False, True):
            self._test_rosenbrock_sparse(
                lambda params: Adagrad(params, lr=1e-1, foreach=foreach),
                multi_tensor=foreach,
            )
            self._test_rosenbrock_sparse(
                lambda params: Adagrad(params, lr=0.1, foreach=foreach),
                scheduler_constructors=[
                    lambda opt: StepLR(opt, gamma=1 - 1e-5, step_size=500),
                    lambda opt: ReduceLROnPlateau(opt, threshold=1e-4),
                ],
                multi_tensor=foreach,
            )


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


@skipIfTorchDynamo("Differentiable optimizers not supported")
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


    @skipIfTorchDynamo("The inplace mu update fails with dynamo, "
                       "since this is only happening when differentiable is enabled, skipping for now")
    def test_asgd(self):
        state = {}
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # `step` `eta` & `mu` are not continuous variables (even though we define them as floats)
        # and so they shouldn't require gradients.
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        state["eta"] = torch.tensor(0.9, requires_grad=False, dtype=torch.float64)
        state["mu"] = torch.tensor(1.0, requires_grad=False, dtype=torch.float64)
        state["ax"] = torch.rand(10, requires_grad=True, dtype=torch.float64)

        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                ASGD,
                {"lr": 0.9, "differentiable": True},
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
                {"lr": 0.9, "weight_decay": 0.1, "decoupled_weight_decay": True, "differentiable": True},
                *state.values(),
            ),
        )


if __name__ == "__main__":
    print("These tests should be run through test/test_optim.py instead")
