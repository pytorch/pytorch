# Owner(s): ["module: inductor"]
import contextlib
import copy
import os
import random

import torch
from torch import nn
from torch._dynamo.utils import same
from torch._inductor import config
from torch._inductor.graph import GraphLowering
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_cuda import tf32_off
from torch.testing._internal.common_utils import skipIfXpu
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU, requires_gpu


USE_DDP_WRAPPER = os.environ.get("USE_DDP_WRAPPER", "1") == "1"


class Model2Conv(nn.Module):
    def __init__(self, dim=512, manual_graph_break=False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, dim, kernel_size=3, stride=2, bias=False)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=2, bias=False)
        self.manual_graph_break = manual_graph_break

    def forward(self, x):
        x = self.conv1(x)
        if self.manual_graph_break:
            torch._dynamo.graph_break()
        x = self.conv2(x)
        return x

    def get_example_inputs(self):
        return (torch.rand(2, 3, 16, 16),)


@skipIfXpu(msg="ccl doesn't currently work on the XPU stack")
class TestLayoutOptim(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        import torch.distributed as dist

        # not use a fixed port for stress test
        tot_retry = 5
        for retry_no in range(tot_retry):
            try:
                port = random.randint(10000, 60000)
                if GPU_TYPE == "cuda":
                    backend = "nccl"
                elif GPU_TYPE == "xpu":
                    backend = "ccl"
                dist.init_process_group(
                    backend=backend,
                    init_method=f"tcp://localhost:{port}",
                    world_size=1,
                    rank=0,
                )
                break
            except RuntimeError:
                if retry_no == tot_retry - 1:
                    raise
                else:
                    continue

    def verify_accuracy(
        self, model_class, use_ddp_wrapper=USE_DDP_WRAPPER, is_train=False
    ):
        # there are 2 potential ways to introduce graph breaks
        # 1. manually
        # 2. using DDP
        # if we are not using DDP to introduce graph breaks, do that manually
        def wrap_mod(m):
            if is_train:

                def f(*inp):
                    x = m(*inp)
                    x.sum().backward()

                    grads = []
                    for _, param in m.named_parameters():
                        grad = param.grad
                        if param.grad is None:
                            grad = torch.zeros_like(param)
                        grads.append(grad)
                    return grads

                return f
            else:
                return m

        manual_graph_break = not use_ddp_wrapper
        mod = model_class(manual_graph_break=manual_graph_break).to(GPU_TYPE)
        inp = [t.to(GPU_TYPE) for t in mod.get_example_inputs()]
        expected_out = wrap_mod(mod)(*inp)

        fp64_mod = copy.deepcopy(mod).to(torch.float64)
        fp64_inp = [t.to(torch.float64) for t in copy.deepcopy(inp)]
        fp64_out = wrap_mod(fp64_mod)(*fp64_inp)

        if use_ddp_wrapper:
            from torch.nn.parallel import DistributedDataParallel as DDP

            ddp_wrapped_mod = DDP(mod)
            opt_mod = torch.compile(wrap_mod(ddp_wrapped_mod))
        else:
            opt_mod = torch.compile(wrap_mod(mod))
        actual_out = opt_mod(*inp)

        if is_train:
            self.assertTrue(same(expected_out, actual_out, fp64_ref=fp64_out))
        else:
            expected_sum = expected_out.sum()
            actual_sum = actual_out.sum()
            print(f"Expected sum {expected_sum}, actual sum {actual_sum}")
            self.assertTrue(same(expected_out, actual_out, fp64_ref=fp64_out))

    def verify_accuracy_for_infer(self, *args, **kwargs):
        self.verify_accuracy(*args, **kwargs, is_train=False)

    def verify_accuracy_for_train(self, *args, **kwargs):
        self.verify_accuracy(*args, **kwargs, is_train=True)

    def test_2conv_with_graph_break(self):
        """
        Make sure graph break does not cause any accuracy issue.
        """
        self.verify_accuracy_for_infer(Model2Conv)

    def test_3conv_with_graph_break(self):
        class Model(nn.Module):
            def __init__(
                self, dim=512, patch_size=7, kernel_size=7, manual_graph_break=False
            ):
                super().__init__()
                self.seq = nn.Sequential(
                    nn.Conv2d(
                        3, dim, kernel_size=patch_size, stride=patch_size, bias=False
                    ),
                    nn.Conv2d(
                        dim, dim, kernel_size, groups=dim, padding="same", bias=False
                    ),
                )
                self.conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
                self.manual_graph_break = manual_graph_break

            def forward(self, x):
                x = self.seq(x)
                if self.manual_graph_break:
                    torch._dynamo.graph_break()
                x = self.conv(x)
                return x

            def get_example_inputs(self):
                return (torch.randn(2, 3, 16, 16),)

        self.verify_accuracy_for_infer(Model)

    @torch.no_grad()
    def test_keep_output_layout_infer(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(
                    3, 128, kernel_size=3, padding=1, stride=1, bias=False
                )

            def forward(self, x):
                x = self.conv(x)
                return x

            def get_example_inputs(self):
                return (torch.randn(2, 3, 5, 5),)

        mod = Model().to(GPU_TYPE)
        inp = [t.to(GPU_TYPE) for t in mod.get_example_inputs()]
        out = mod(*inp)

        opt_mod = torch.compile(mod)
        opt_out = opt_mod(*inp)

        # We should be able to do view on eager output
        out.view(5, -1)

        # We should be able to do view on the output of the optimized module
        # Note that if the output is channels last, the view op will fail.
        opt_out.view(5, -1)

    def test_keep_output_layout_with_freezing(self):
        with config.patch(
            {
                "freezing": True,
            }
        ):
            self.test_keep_output_layout_infer()

    def test_training_acc(self):
        self.verify_accuracy_for_train(Model2Conv)

    def test_mutate_view(self):
        """
        The GraphModule passed to GraphLowering init method is like:
        https://gist.github.com/shunting314/07228313fd017e2267101ff32edc6d64

        It shows that we will call copy_ to update the argument in the end. This
        guarantees the correctnesss.
        """

        @torch.compile
        def f(x):
            y = x.view(3, 2)
            y.mul_(2)

        x = torch.ones(2, 3).to(GPU_TYPE)
        f(x)
        self.assertTrue(torch.equal(x, torch.ones(2, 3).to(GPU_TYPE) * 2))

    def test_mutate_base(self):
        """
        The GraphModule passed to GraphLowering init method is like:
        https://gist.github.com/shunting314/fd60fe11d1f844c6db76aba7b06811bc

        It shows that the output of the graph is the mul node which contains
        the update we applied to the base tensor.
        """

        @torch.compile
        def f(x):
            y = x.view(3, 2)
            x.mul_(2)
            return y

        x = torch.ones(2, 3).to(GPU_TYPE)
        y = f(x)
        self.assertTrue(torch.equal(y, torch.ones(3, 2).to(GPU_TYPE) * 2))

    @tf32_off()
    def test_mutate_base_for_conv_output(self):
        class Model(nn.Module):
            def __init__(self, manual_graph_break=False):
                super().__init__()
                self.conv = nn.Conv2d(3, 512, kernel_size=3, stride=2, bias=False)

            def forward(self, x):
                x = self.conv(x)
                y = x.view(-1)
                x.mul_(2)
                return y

            def get_example_inputs(self):
                return (torch.rand(2, 3, 16, 16),)

        self.verify_accuracy_for_infer(Model)

    @tf32_off()
    def test_mutate_view_for_conv_output(self):
        class Model(nn.Module):
            def __init__(self, manual_graph_break=False):
                super().__init__()
                self.conv = nn.Conv2d(3, 512, kernel_size=3, stride=2, bias=False)

            def forward(self, x):
                x = self.conv(x)
                y = x.view(-1)
                y.mul_(2)
                return x

            def get_example_inputs(self):
                return (torch.rand(2, 3, 16, 16),)

        self.verify_accuracy_for_infer(Model)

    def test_dynamic_shape_specialization(self):
        """
        Previously in aot_autograd.py we compare strides of FakeTensor
        with real tensor. That cause dynamic dimensions of the FakeTensor
        being specialized to static shapes. This test protects against that.
        """

        def f(a, b):
            x = a.sin()
            y = b.cos()
            z = x + y
            return z

        for size in [4, 8, 16]:
            a = torch.randn(2, size, requires_grad=True).to(GPU_TYPE)
            b = torch.randn(2, size).to(GPU_TYPE)
            actual = torch.compile(f, dynamic=True)(a, b)
            self.assertTrue(torch.allclose(f(a, b), actual))

            # Trigger the compiling of the backward graph
            actual.sum().backward()

    def test_nll_loss_backward(self):
        """
        Repro for issue https://github.com/pytorch/pytorch/issues/120759

        The CUDA implementation of aten.nll_loss2d_backward.default requires
        the self tensor (whose layout will be used to create grad_input)
        to be contiguous. Layout optimization may change the self tensor's layout
        and cause failure. We fix that by adding layout constraints to the
        fallback of aten.nll_loss2d_backward.default .
        """

        class MyModel(torch.nn.Module):
            def __init__(self, input_dim, num_classes):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, num_classes, 3, 1, padding="same")
                self.out = torch.nn.Linear(input_dim * num_classes, num_classes)

            def forward(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                x = self.conv(x)
                b, c, t, f = x.size()
                x = self.out(x.reshape(b, t, c * f))
                logits = x.reshape(x.size(0), x.size(2), x.size(1))
                loss = torch.nn.functional.cross_entropy(logits, targets)
                return loss

        device = GPU_TYPE
        batch_size = 48
        seq_len = 144
        input_dim = 39
        num_classes = 111

        model = MyModel(input_dim, num_classes)
        model.to(device)

        opt_model = torch.compile(model)  # noqa: F841

        x = torch.ones((batch_size, 1, seq_len, input_dim), device=device)
        targets = torch.randint(
            0, num_classes - 1, (batch_size, seq_len), device=device, dtype=torch.int64
        )

        loss = model(x, targets)
        loss.backward()

        ref = model(x, targets)
        self.assertTrue(torch.allclose(ref, loss))


@requires_gpu()
class TestLayoutOptimROCmMultipliers(TestCase):
    """Validate the per-arch ROCm channels-last cost-multiplier overrides
    (gfx942 / gfx950) in GraphLowering.decide_layout_opt (inference path).

    The NVIDIA-tuned defaults are
        GROUPED=1.358, DEFAULT=0.823, IN_OUT=0.725, SMALL=0.783
    On gfx942 (MI300) channels-last regresses small/grouped convs, so the
    override sets SMALL=1.25, GROUPED=1.05 (DEFAULT/IN_OUT unchanged). On gfx950
    (MI350) channels-last is favorable, so GROUPED=0.553, DEFAULT=0.813,
    IN_OUT=0.642, SMALL=0.795. decide_layout_opt enables channels-last iff
    weighted_flops <= total_flops, so the re-tuned weights flip the decision for
    the affected graphs while leaving default-dominated graphs unchanged.

    The arch decision is driven through the cached helper
    ``torch._inductor.graph._rocm_native_device_arch_name`` and through
    ``torch.version.hip`` so the gfx942, gfx950 and non-ROCm outcomes are checked
    regardless of the real card. Graph construction still needs FakeTensor conv
    meta on a cuda device, hence the GPU gate.
    """

    NON_GFX942_ARCH = "gfx90a:sramecc+:xnack-"
    GFX942_ARCH = "gfx942:sramecc+:xnack-"
    GFX950_ARCH = "gfx950:sramecc+:xnack-"

    @staticmethod
    def _build_aten_gm(mod, example_input):
        """Trace mod to an aten fx graph whose convolution.default nodes carry
        FakeTensor meta['val'] on a cuda device."""
        mod = mod.to(GPU_TYPE).eval()
        x = example_input.to(GPU_TYPE)
        from torch.fx.experimental.proxy_tensor import make_fx

        gm = make_fx(mod, tracing_mode="fake", _allow_non_fake_inputs=True)(x)
        convs = [
            n for n in gm.graph.nodes if n.target is torch.ops.aten.convolution.default
        ]
        assert len(convs) > 0, "expected aten.convolution.default nodes"
        for n in convs:
            w = n.args[1].meta["val"]
            assert isinstance(w, torch.Tensor)
            assert w.device.type == "cuda", w.device
        return gm

    @staticmethod
    def _flop_counts(gm):
        """Reproduce decide_layout_opt's flop classification so tests can assert
        non-vacuity (which bucket dominates) independently of the decision."""
        from collections import defaultdict

        from torch._inductor.fx_utils import count_flops_fx

        def is_grouped(n):
            mv = n.args[1].meta["val"]
            return n.args[-1] > 1 and mv.size(1) > 1

        def is_in_out(n):
            mv = n.args[1].meta["val"]
            return mv.size(0) * 2 <= mv.size(1) and mv.size(2) > 1

        def is_small(n):
            mv = n.args[1].meta["val"]
            return mv.size(0) <= 64 and mv.size(1) <= 64

        fc = defaultdict(float)
        for n in gm.graph.nodes:
            if n.target is not torch.ops.aten.convolution.default:
                continue
            f = count_flops_fx(n)
            if f is None:
                continue
            if is_grouped(n):
                t = "grouped"
            elif is_small(n):
                t = "small"
            elif is_in_out(n):
                t = "in_out"
            else:
                t = "default"
            fc[t] += f
        return fc

    def _decide(self, gm, arch_name, hip_version="7.0.0", props=None):
        """Run decide_layout_opt with layout_optimization forced on and the
        arch / hip version mocked to a deterministic value.

        The gfx942 helper is patched directly (it is functools.cache'd on the
        device, so patching the underlying torch.cuda.get_device_properties
        would otherwise leak between cases). When ``props`` is supplied it
        replaces torch.cuda.get_device_properties so we can assert the non-HIP
        path never touches the driver (it must not raise).
        """
        from unittest import mock

        import torch._inductor.graph as graph_mod

        ctxs = [
            mock.patch.object(config, "layout_optimization", True),
            mock.patch.object(torch.version, "hip", hip_version),
            mock.patch.object(
                graph_mod,
                "_rocm_native_device_arch_name",
                lambda device: arch_name,
            ),
        ]
        if props is not None:
            ctxs.append(mock.patch("torch.cuda.get_device_properties", props))
        with contextlib.ExitStack() as stack:
            for c in ctxs:
                stack.enter_context(c)
            return GraphLowering.decide_layout_opt(gm, is_inference=True)

    # --- small-channel dominated: gfx942 SKIPS, NVIDIA ENABLES ----------------
    def test_small_channel_graph_skips_on_gfx942(self):
        class SmallConvNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.c1 = nn.Conv2d(32, 32, 3)
                self.c2 = nn.Conv2d(32, 32, 3)
                self.c3 = nn.Conv2d(32, 32, 3)

            def forward(self, x):
                return self.c3(self.c2(self.c1(x)))

        gm = self._build_aten_gm(SmallConvNet(), torch.rand(2, 32, 28, 28))

        # Non-vacuity: the graph is small-channel dominated.
        fc = self._flop_counts(gm)
        total = sum(fc.values())
        self.assertGreater(total, 0.0)
        self.assertGreater(fc["small"] / total, 0.9)

        # gfx942: SMALL=1.25 => weighted_flops > total_flops => skip.
        self.assertFalse(
            self._decide(gm, self.GFX942_ARCH),
            "gfx942 weights should SKIP layout opt for a small-conv graph",
        )
        # non-gfx942 ROCm (NVIDIA defaults, SMALL=0.783) => enable.
        self.assertTrue(
            self._decide(gm, self.NON_GFX942_ARCH),
            "non-gfx942 weights should ENABLE layout opt for a small-conv graph",
        )
        # non-HIP build also takes the NVIDIA path => enable.
        self.assertTrue(
            self._decide(gm, self.GFX942_ARCH, hip_version=None),
            "non-HIP build should ignore the gfx942 override",
        )

    # --- M2: grouped portion that actually FLIPS the decision -----------------
    def test_grouped_graph_flips_decision_on_gfx942(self):
        # resnext-like: one dense (default) conv + four lightly-grouped convs,
        # sized so grouped flops are ~2/3 of total. At that fraction the decision
        # crosses the threshold only when GROUPED moves 1.358 -> 1.05:
        #   gfx942:  0.823*0.333 + 1.05*0.667  ~= 0.97 <= 1  => ENABLE
        #   nvidia:  0.823*0.333 + 1.358*0.667 ~= 1.18 >  1  => SKIP
        class GroupedFlipNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.d1 = nn.Conv2d(256, 256, 3, padding=1)  # dense / default
                self.g1 = nn.Conv2d(256, 256, 3, padding=1, groups=2)
                self.g2 = nn.Conv2d(256, 256, 3, padding=1, groups=2)
                self.g3 = nn.Conv2d(256, 256, 3, padding=1, groups=2)
                self.g4 = nn.Conv2d(256, 256, 3, padding=1, groups=2)

            def forward(self, x):
                a = self.d1(x)
                b = self.g4(self.g3(self.g2(self.g1(x))))
                return a + b

        gm = self._build_aten_gm(GroupedFlipNet(), torch.rand(2, 256, 32, 32))

        # Non-vacuity: grouped ~0.67, default ~0.33, nothing else.
        fc = self._flop_counts(gm)
        total = sum(fc.values())
        self.assertGreater(total, 0.0)
        self.assertEqual(set(fc), {"grouped", "default"})
        self.assertGreater(fc["grouped"] / total, 0.55)
        self.assertLess(fc["grouped"] / total, 0.78)

        # The decision genuinely flips with the gfx942 weights.
        self.assertTrue(
            self._decide(gm, self.GFX942_ARCH),
            "gfx942 GROUPED=1.05 should ENABLE this resnext-like graph",
        )
        self.assertFalse(
            self._decide(gm, self.NON_GFX942_ARCH),
            "NVIDIA GROUPED=1.358 should SKIP this resnext-like graph",
        )

    # --- default-channel dominated: ENABLE under BOTH weight sets -------------
    def test_default_channel_graph_enables_on_both(self):
        class DefaultConvNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.c1 = nn.Conv2d(256, 256, 3)
                self.c2 = nn.Conv2d(256, 256, 3)

            def forward(self, x):
                return self.c2(self.c1(x))

        gm = self._build_aten_gm(DefaultConvNet(), torch.rand(2, 256, 28, 28))

        fc = self._flop_counts(gm)
        total = sum(fc.values())
        self.assertGreater(fc["default"] / total, 0.9)

        # DEFAULT_MULTIPLIER is unchanged (0.823 < 1) so both enable.
        self.assertTrue(
            self._decide(gm, self.GFX942_ARCH),
            "default-conv graph should ENABLE under gfx942 weights",
        )
        self.assertTrue(
            self._decide(gm, self.NON_GFX942_ARCH),
            "default-conv graph should ENABLE under NVIDIA weights",
        )

    # --- M4: gfx950 has its own (favorable) weights; grouped graph flips -----
    def test_grouped_graph_flips_decision_on_gfx950(self):
        # Same resnext-like graph as the gfx942 grouped test (grouped ~2/3):
        #   gfx950:  0.823*0.333 + 0.553*0.667 ~= 0.64 <= 1  => ENABLE
        #   nvidia:  0.823*0.333 + 1.358*0.667 ~= 1.18 >  1  => SKIP
        class GroupedFlipNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.d1 = nn.Conv2d(256, 256, 3, padding=1)
                self.g1 = nn.Conv2d(256, 256, 3, padding=1, groups=2)
                self.g2 = nn.Conv2d(256, 256, 3, padding=1, groups=2)
                self.g3 = nn.Conv2d(256, 256, 3, padding=1, groups=2)
                self.g4 = nn.Conv2d(256, 256, 3, padding=1, groups=2)

            def forward(self, x):
                a = self.d1(x)
                b = self.g4(self.g3(self.g2(self.g1(x))))
                return a + b

        gm = self._build_aten_gm(GroupedFlipNet(), torch.rand(2, 256, 32, 32))

        fc = self._flop_counts(gm)
        total = sum(fc.values())
        self.assertGreater(total, 0.0)
        self.assertGreater(fc["grouped"] / total, 0.55)

        # gfx950 GROUPED=0.553 ENABLES; NVIDIA GROUPED=1.358 SKIPS.
        self.assertTrue(
            self._decide(gm, self.GFX950_ARCH),
            "gfx950 GROUPED=0.553 should ENABLE this resnext-like graph",
        )
        self.assertFalse(
            self._decide(gm, self.NON_GFX942_ARCH),
            "NVIDIA GROUPED=1.358 should SKIP this resnext-like graph",
        )
        # non-HIP build ignores the gfx950 override too.
        self.assertFalse(
            self._decide(gm, self.GFX950_ARCH, hip_version=None),
            "non-HIP build should ignore the gfx950 override",
        )

    # --- M3 / C1: NVIDIA (non-HIP) path is unchanged and never touches the ----
    # --- driver, even if get_device_properties would raise.              ------
    def test_non_hip_path_unchanged_and_driverless(self):
        class SmallConvNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.c1 = nn.Conv2d(32, 32, 3)
                self.c2 = nn.Conv2d(32, 32, 3)
                self.c3 = nn.Conv2d(32, 32, 3)

            def forward(self, x):
                return self.c3(self.c2(self.c1(x)))

        gm = self._build_aten_gm(SmallConvNet(), torch.rand(2, 32, 28, 28))

        def _boom(*a, **k):
            raise RuntimeError("driver must not be queried on the non-HIP path")

        # torch.version.hip=None: branch not entered. Even with
        # get_device_properties rigged to raise, no exception should escape and
        # the decision must equal the stock NVIDIA outcome (ENABLE for a
        # small-conv graph). This pins both M3 and the C1 hardening.
        decision = self._decide(gm, self.GFX942_ARCH, hip_version=None, props=_boom)
        self.assertTrue(
            decision,
            "non-HIP path should use stock constants and ENABLE this graph",
        )


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
