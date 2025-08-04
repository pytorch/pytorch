# Owner(s): ["module: optimizer"]

import copy
import math

import torch
from torch import nn
from torch.nn import Linear, MSELoss
from torch.optim import AdamW, Muon
from torch.testing._internal.common_utils import (
    load_tests,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)


# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests


class MoonshotReferenceMuon(torch.optim.Optimizer):
    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
    ):
        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )

        params = list(muon_params)
        super().__init__(params, defaults)
        for p in muon_params:
            assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True

    def zeropower_via_newtonschulz5(self, G, steps):
        assert len(G.shape) == 2
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G.bfloat16()
        if G.size(0) > G.size(1):
            X = X.T
        X = X / (X.norm() + 1e-7)
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X

        if G.size(0) > G.size(1):
            X = X.T
        return X

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                u = self.zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)
                p.data.mul_(1 - lr * wd)
                p.data.add_(u, alpha=-adjusted_lr)

        return loss


class TestMuon(TestCase):
    @skipIfTorchDynamo("MoonshotReferenceMuon compile issue.")
    def test_muon_fqn_set_empty_equals_to_adamw(self):
        model0 = nn.Linear(5, 5)
        model1 = copy.deepcopy(model0)

        lr = 1e-3
        wd = 0.1
        adamw_betas = (0.9, 0.95)
        adamw_eps = 1e-8

        adamw = AdamW(
            model0.parameters(),
            lr=lr,
            betas=adamw_betas,
            eps=adamw_eps,
            weight_decay=wd,
        )

        # Muon on model1, with an “empty” FQN so it should fall back to AdamW behavior
        muon_param_fqns = [""]
        muon = Muon(
            model1.named_parameters(),
            lr=lr,
            wd=wd,
            muon_param_fqns=muon_param_fqns,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        torch.manual_seed(0)
        for p0, p1 in zip(model0.parameters(), model1.parameters()):
            g = torch.randn_like(p0)
            p0.grad = g.clone()
            p1.grad = g.clone()

        adamw.step()
        muon.step()

        for p0, p1 in zip(model0.parameters(), model1.parameters()):
            self.assertTrue(
                torch.allclose(p0, p1, atol=1e-6),
                "Muon did not match AdamW for empty-FQN case",
            )

    @skipIfTorchDynamo("MoonshotReferenceMuon compile issue.")
    def test_muon_implementation_equivalency(self):
        torch.manual_seed(0)

        # simple model and data
        model0 = Linear(5, 5, bias=False)
        model1 = copy.deepcopy(model0)
        inputs = torch.randn(8, 5)
        targets = torch.randn(8, 5)
        loss = MSELoss()

        lr = 1e-3
        wd = 0.1
        momentum = 0.95
        nesterov = True
        ns_steps = 5

        moonshot_muon_params = [
            p for name, p in model0.named_parameters() if "weight" in name
        ]

        opt_moonshot_muon = MoonshotReferenceMuon(
            lr=lr,
            wd=wd,
            muon_params=moonshot_muon_params,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )

        muon_fqns = ["weight"]
        opt_pytorch_muon = Muon(
            params=model1.named_parameters(),
            lr=lr,
            wd=wd,
            muon_param_fqns=muon_fqns,
            momentum=momentum,
            nesterov=nesterov,
        )

        for _ in range(10):
            out_m = model0(inputs)
            loss_m = loss(out_m, targets)
            opt_moonshot_muon.zero_grad()
            loss_m.backward()
            opt_moonshot_muon.step()

            out_t = model1(inputs)
            loss_t = loss(out_t, targets)
            opt_pytorch_muon.zero_grad()
            loss_t.backward()
            opt_pytorch_muon.step()

        for idx, (p_m, p_t) in enumerate(zip(model0.parameters(), model1.parameters())):
            self.assertTrue(
                torch.allclose(p_m, p_t, atol=1e-6, rtol=1e-5),
                msg=f"Parameter #{idx} differs by more than tolerance",
            )


if __name__ == "__main__":
    run_tests()
