# Owner(s): ["module: dynamo"]

import inspect
import unittest

import torch

import torch._dynamo
import torch._dynamo.test_case
import torch._dynamo.testing

input = torch.ones([10, 10])
model = torch.nn.Sequential(*[torch.nn.Linear(10, 10) for _ in range(2)])
model(input).sum().backward()


def make_test(optim_cls, exp_frame_cnt=1, closure=None, **kwargs):
    opt = optim_cls(model.parameters(), **kwargs)

    def test_fn(self):
        nonlocal opt

        counter = torch._dynamo.testing.CompileCounter()

        if closure is not None:

            def fn():
                opt.step(closure)

        else:
            fn = opt.step

        opt_fn = torch._dynamo.optimize(counter)(fn)
        opt_fn()

        self.assertEqual(counter.frame_count, exp_frame_cnt)

    return test_fn


class OptimizerTests(torch._dynamo.test_case.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # needed until pytorch assertion is changed to enable Adam
        # to be called with capturable=True
        cls._exit_stack.enter_context(
            unittest.mock.patch.object(
                torch._dynamo.config, "capture_scalar_outputs", True
            )
        )
        cls._exit_stack.enter_context(
            unittest.mock.patch.object(
                torch._dynamo.config, "fake_tensor_propagation", False
            )
        )
        cls._exit_stack.enter_context(
            unittest.mock.patch.object(
                torch._dynamo.config, "raise_on_assertion_error", True
            )
        )

    test_sgd = make_test(torch.optim.SGD, lr=0.01)
    # lgbfs has data-dependent control and internally iterates
    # calling the closure
    # TODO mlazos: re-enable once we have latest pytorch with FakeTensor fix #497
    # test_lbfgs = make_test(
    #    torch.optim.LBFGS, exp_frame_cnt=3, closure=lambda: model(input).sum()
    # )
    # RAdam has data-dependent control which breaks the graph
    test_radam = make_test(torch.optim.RAdam, exp_frame_cnt=1)

    # ASGD has a small optimization that avoids averaging
    # This will fully capture the graph once that optimization is removed
    # NB: in python versions < 3.8, we don't capture graphs when breaks
    # occur in a loop

    # Fails without fake tensor:
    # TypeError: clamp() received an invalid combination of arguments - got (float, min=int)
    # test_asgd = make_test(
    #     torch.optim.ASGD, exp_frame_cnt=(0 if sys.version_info < (3, 8) else 6)
    # )


# exclude SparseAdam because other areas of the stack don't support it yet
# the others are handled specially above
exclude = set(["SGD", "Optimizer", "SparseAdam", "LBFGS", "RAdam", "ASGD"])
optimizers = [
    opt
    for opt in torch.optim.__dict__.values()
    if inspect.isclass(opt)
    and issubclass(opt, torch.optim.Optimizer)
    and opt.__name__ not in exclude
]


for opt in optimizers:
    setattr(OptimizerTests, "test_" + opt.__name__.lower(), make_test(opt))

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
