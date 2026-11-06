# Owner(s): ["module: dynamo"]
"""
PYTEST_DONT_REWRITE (prevents pytest from rewriting assertions, which interferes
with test_adam in OptimizerTests)
"""

import functools
from copy import deepcopy

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.variables.optimizer import OptimizerVariable
from torch.nn import Parameter
from torch.testing._internal.common_utils import HardwareClassification


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
    hw_classification = HardwareClassification.GENERIC

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
        opt_training_iter_fn = torch.compile(training_iter_fn, backend=cnts)
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

    def test_stock_optimizer_init_group_after_disable_patch(self):
        # TorchPatcher wraps stock Optimizer._init_group with compiler.disable.
        # Compiling the inner step must still constant-fold _init_group via
        # OptimizerVariable.tp_methods, not inline the disable wrapper.
        torch._dynamo.eval_frame.TorchPatcher.patch()

        # Adadelta is the inductor compiled-optimizer shard that failed; Adamax
        # is the dynamo-wrapped test_optim case. Both wrap _init_group.
        for opt_cls in (torch.optim.Adadelta, torch.optim.Adamax):
            with self.subTest(opt_cls=opt_cls.__name__):
                p = Parameter(torch.ones(2, 2), requires_grad=True)
                opt = opt_cls([p])
                p.grad = torch.ones_like(p)

                step_fn = opt.step.__wrapped__.__wrapped__

                @torch.compile(backend="eager", fullgraph=True)
                def compiled_step():
                    step_fn(opt)

                before = p.detach().clone()
                with torch.no_grad():
                    compiled_step()
                self.assertFalse(torch.equal(p, before))
                torch._dynamo.reset()

    def test_capturable_optim_on_privateuse1(self):
        backend_name = torch._C._get_privateuse1_backend_name()
        if (
            backend_name == "privateuseone"
            or not getattr(torch, backend_name).is_available()
        ):
            self.skipTest("no PrivateUse1 accelerator available")

        dev = backend_name
        p = torch.randn(8, device=dev).requires_grad_(True)
        p.grad = torch.full_like(p, 1.0)
        opt = torch.optim.Adam([p], lr=1e-2)

        original_capturable = opt.param_groups[0]["capturable"]
        self.assertFalse(original_capturable)

        ov = OptimizerVariable(opt)
        self.assertTrue(ov._safe_to_set_capturable(opt.param_groups[0]))

        opt.step()
        self.assertEqual(opt.param_groups[0]["capturable"], original_capturable)

        sd = deepcopy(opt.state_dict())
        self.assertEqual(
            sd["param_groups"][0].get("capturable", False),
            original_capturable,
        )

    def test_capturable_predicate_covers_runtime_devices(self):
        from torch.optim.optimizer import _get_capturable_supported_devices

        class FakeParam:
            def __init__(self, device_type):
                self.device = type("Device", (), {"type": device_type})()

        opt = torch.optim.Adam([torch.randn(2)], lr=1e-2)
        ov = OptimizerVariable(opt)

        def fake_group(device_type):
            return {"params": [FakeParam(device_type)], "capturable": False}

        for device_type in _get_capturable_supported_devices(supports_xla=True):
            with self.subTest(device=device_type):
                self.assertTrue(
                    ov._safe_to_set_capturable(fake_group(device_type)),
                    f"{device_type} should be capturable",
                )

        for device_type in ("cpu", "mps"):
            with self.subTest(device=device_type):
                self.assertFalse(
                    ov._safe_to_set_capturable(fake_group(device_type)),
                    f"{device_type} should NOT be capturable",
                )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
