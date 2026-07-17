# Owner(s): ["module: inductor"]
"""Tests for the reorder_for_locality_in_training flag: env parsing, and that on
a real fwd+bwd training graph the flag reorders a node without changing
grads/params."""

import os
import subprocess
import sys
from unittest.mock import patch as mock_patch

import torch
import torch._dynamo
import torch._inductor.config as inductor_config
from torch._inductor.fx_passes import post_grad
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    run_tests,
    TestCase,
    xfailIfNoAcceleratorTriton,
)


def _flag_in_subprocess(env_value):
    # Parse-only: a fresh interpreter imports just the config module and prints
    # the resolved flag. env_value=None means the var is unset (ship default).
    env = os.environ.copy()
    if env_value is None:
        env.pop("TORCHINDUCTOR_REORDER_LOCALITY_TRAINING", None)
    else:
        env["TORCHINDUCTOR_REORDER_LOCALITY_TRAINING"] = env_value
    out = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import torch._inductor.config as c;"
            "print(int(c.reorder_for_locality_in_training))",
        ],
        env=env,
    )
    return out.decode().strip()


class _Recorder:
    """Wraps ``reorder_for_locality`` and records, per call, whether the pass
    changed the graph node order. ``orig`` is captured before patching so the
    real pass still runs."""

    def __init__(self):
        self.orig = post_grad.reorder_for_locality
        self.calls = 0
        self.moved = False

    def __call__(self, graph):
        before = [n.name for n in graph.nodes]
        self.orig(graph)
        after = [n.name for n in graph.nodes]
        self.calls += 1
        if before != after:
            self.moved = True


class _TwoBranch(torch.nn.Module):
    # Two independent matmul branches that only meet at the end. After
    # functionalization the branch producers land far from their sole
    # consumers, so reorder_for_locality has real moves to make.
    def __init__(self):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.randn(16, 16))
        self.w2 = torch.nn.Parameter(torch.randn(16, 16))

    def forward(self, x):
        a = torch.tanh(x @ self.w1)
        b = torch.sigmoid(x @ self.w2)
        return (a * b + torch.sin(a) * torch.cos(b)).sum(dim=1)


def _train_once(device, flag_on, reorder_on=True):
    rec = _Recorder()
    torch._dynamo.reset()
    torch.manual_seed(1234)
    model = _TwoBranch().to(device)
    x = torch.randn(8, 16, device=device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with inductor_config.patch(
        {
            "reorder_for_locality": reorder_on,
            "reorder_for_locality_in_training": flag_on,
            "fx_graph_cache": False,
            "force_disable_caches": True,
        }
    ):
        with mock_patch.object(post_grad, "reorder_for_locality", rec):
            compiled = torch.compile(model, backend="inductor", fullgraph=True)
            compiled(x).sum().backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
    opt.step()
    params = {n: p.detach().clone() for n, p in model.named_parameters()}
    return params, grads, rec


class TestReorderForLocalityInTrainingEnv(TestCase):
    def test_default_off(self):
        self.assertEqual(_flag_in_subprocess(None), "0")

    def test_env_one_turns_on(self):
        self.assertEqual(_flag_in_subprocess("1"), "1")

    def test_env_zero_keeps_off(self):
        self.assertEqual(_flag_in_subprocess("0"), "0")

    def test_inference_gate_unchanged(self):
        # Gate-only check (invocation, not a feature exercise): inference graphs
        # must still run the pass regardless of the training flag. Pure FX, no
        # device execution, so it stays out of the device-generic class below.
        rec = _Recorder()
        with inductor_config.patch({"reorder_for_locality_in_training": False}):
            with mock_patch.object(post_grad, "reorder_for_locality", rec):
                g = torch.fx.Graph()
                xn = g.placeholder("x")
                g.output(g.call_function(torch.relu, args=(xn,)))
                gm = torch.fx.GraphModule(torch.nn.Module(), g)
                post_grad.post_grad_passes(gm, is_inference=True)
        self.assertGreater(rec.calls, 0)


class TestReorderForLocalityInTraining(TestCase):
    @xfailIfNoAcceleratorTriton
    def test_training_flag_reorders_and_preserves_semantics(self, device):
        p_off, g_off, rec_off = _train_once(device, flag_on=False)
        p_on, g_on, rec_on = _train_once(device, flag_on=True)

        # (a) semantics preserved: same seed/init/input, so grads and post-step
        # params match between flag-off and flag-on. reorder_for_locality is a
        # mathematical (not bitwise) equivalence -- it can change fusion grouping
        # and thus low-bit FP accumulation -- so use default tolerances.
        self.assertEqual(set(g_off), set(g_on))
        for k in g_off:
            self.assertEqual(g_on[k], g_off[k])
            self.assertEqual(p_on[k], p_off[k])

        # (b) the flag actually exercised the pass: with it on the reorder ran
        # and moved at least one node on a training graph; with it off the pass
        # never ran on training. The moved check is what gives this teeth, a
        # no-op graph would leave moved False and fail here.
        self.assertEqual(rec_off.calls, 0)
        self.assertGreater(rec_on.calls, 0)
        self.assertTrue(
            rec_on.moved,
            "flag on must reorder at least one node on the training graph",
        )

    @xfailIfNoAcceleratorTriton
    def test_training_flag_noop_when_reorder_off(self, device):
        # The training flag is gated by reorder_for_locality: enabling it while
        # reorder_for_locality is off must not run the pass on a training graph.
        _, _, rec = _train_once(device, flag_on=True, reorder_on=False)
        self.assertEqual(rec.calls, 0)


instantiate_device_type_tests(TestReorderForLocalityInTraining, globals())


if __name__ == "__main__":
    run_tests()
