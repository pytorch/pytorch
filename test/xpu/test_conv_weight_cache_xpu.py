# Owner(s): ["module: intel"]
# XPU inference convolutions: forward_inference primitive + opt-in packed-weight cache.

import gc
import os
import subprocess
import sys

import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    tol,
    toleranceOverride,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


# The cache is opt-in and the variable is read at the first inference conv.
os.environ["TORCH_XPU_CONV_WEIGHT_CACHE"] = "1"

CL = torch.channels_last


def train_ref(x, w, b):
    # grad mode + requires_grad weight -> forward_training path, no cache
    with torch.enable_grad():
        return F.conv2d(x, w.detach().requires_grad_(), b, padding=1).detach()


def reorders_per_call(fn, calls=10):
    for _ in range(3):
        fn()
    torch.xpu.synchronize()
    with profile(activities=[ProfilerActivity.XPU]) as prof:
        for _ in range(calls):
            fn()
        torch.xpu.synchronize()
    events = prof.key_averages()
    return sum(e.count for e in events if "reorder" in e.key.lower()) / calls


class TestConvWeightCache(TestCase):
    hw_classification = HardwareClassification.XPU

    def _data(self, device, dtype, c=64):
        # Weights must be channels last like the input, otherwise _convolution_out
        # passes a per-call copy of the weight and the cache never sees the original.
        torch.manual_seed(0)
        x = torch.randn(1, c, 40, 40, device=device, dtype=dtype).contiguous(
            memory_format=CL
        )
        w = (torch.randn(c, c, 3, 3, device=device, dtype=dtype) * 0.05).contiguous(
            memory_format=CL
        )
        b = torch.randn(c, device=device, dtype=dtype) * 0.1
        return x, w, b

    @dtypes(torch.float32, torch.float16)
    @toleranceOverride({torch.float16: tol(atol=1e-2, rtol=1e-2)})
    def test_inference_matches_training_path(self, device, dtype):
        x, w, b = self._data(device, dtype)
        ref = train_ref(x, w, b)
        with torch.no_grad():
            for _ in range(3):  # first call packs, later calls hit the cache
                out = F.conv2d(x, w, b, padding=1)
        self.assertEqual(out, ref)

    @dtypes(torch.float16)
    def test_no_per_call_weight_reorder(self, device, dtype):
        x, w, b = self._data(device, dtype)
        with torch.no_grad():
            self.assertEqual(reorders_per_call(lambda: F.conv2d(x, w, b, padding=1)), 0)

    @dtypes(torch.float16)
    def test_module_parameters_are_cached(self, device, dtype):
        # nn.Parameter keeps requires_grad=True under no_grad; still inference.
        x, _, _ = self._data(device, dtype)
        m = torch.nn.Conv2d(64, 64, 3, padding=1).to(device, dtype).to(memory_format=CL)
        with torch.no_grad():
            self.assertEqual(reorders_per_call(lambda: m(x)), 0)

    @dtypes(torch.float16)
    def test_frozen_compiled_model_is_cached(self, device, dtype):
        import torch._inductor.config as inductor_config

        x, _, _ = self._data(device, dtype)
        m = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1),
        )
        m = m.eval().to(device, dtype).to(memory_format=CL)
        torch._dynamo.reset()
        with inductor_config.patch(freezing=True), torch.inference_mode():
            f = torch.compile(m)
            self.assertEqual(reorders_per_call(lambda: f(x)), 0)

    @dtypes(torch.float32)
    def test_inplace_weight_update_invalidates(self, device, dtype):
        x, w, b = self._data(device, dtype)
        with torch.no_grad():
            F.conv2d(x, w, b, padding=1)
            w.mul_(2.0)
            out = F.conv2d(x, w, b, padding=1)
        self.assertEqual(out, train_ref(x, w, b))

    @dtypes(torch.float32)
    def test_inference_tensor_inplace_update(self, device, dtype):
        with torch.inference_mode():
            x, w, b = self._data(device, dtype)
            F.conv2d(x, w, b, padding=1)
            w.mul_(2.0)
            out = F.conv2d(x, w, b, padding=1)
        self.assertEqual(out, train_ref(x.clone(), w.clone(), b.clone()))

    @dtypes(torch.float32)
    def test_aliases_of_same_storage(self, device, dtype):
        # Same data pointer, different shapes: must not share an entry.
        x, w, b = self._data(device, dtype)
        w_half = w[:32]
        self.assertEqual(w_half.data_ptr(), w.data_ptr())
        with torch.no_grad():
            out_w = F.conv2d(x, w, b, padding=1)
            out_half = F.conv2d(x, w_half, b[:32], padding=1)
            out_w_again = F.conv2d(x, w, b, padding=1)
        self.assertEqual(out_w, train_ref(x, w, b))
        self.assertEqual(out_half, train_ref(x, w_half, b[:32]))
        self.assertEqual(out_w_again, out_w)

    @dtypes(torch.float16)
    def test_freed_weights_are_released(self, device, dtype):
        x, w, b = self._data(device, dtype)
        with torch.no_grad():
            torch.xpu.synchronize()
            before = torch.xpu.memory_allocated()
            for _ in range(8):
                w = torch.randn_like(w)
                F.conv2d(x, w, b, padding=1)
            del w
            gc.collect()
            # the next miss drops the entries of the freed weights
            w = torch.randn(64, 64, 3, 3, device=device, dtype=dtype).contiguous(
                memory_format=CL
            )
            out = F.conv2d(x, w, b, padding=1)
            del out
            torch.xpu.synchronize()
        # only the live weight and its packed copy may remain
        self.assertLessEqual(torch.xpu.memory_allocated() - before, 2 * w.nbytes + 4096)

    def test_cache_is_off_by_default(self, device):
        # The env var is read once per process, so check in fresh ones. A cached
        # packed weight stays allocated after the conv; nothing does without it.
        code = (
            "import torch, torch.nn.functional as F\n"
            "cl = torch.channels_last\n"
            "x = torch.randn(1, 64, 40, 40, device='xpu').contiguous(memory_format=cl)\n"
            "w = torch.randn(64, 64, 3, 3, device='xpu').contiguous(memory_format=cl)\n"
            "torch.xpu.synchronize()\n"
            "before = torch.xpu.memory_allocated()\n"
            "with torch.no_grad():\n"
            "    F.conv2d(x, w, padding=1)\n"
            "torch.xpu.synchronize()\n"
            "print(torch.xpu.memory_allocated() - before, w.nbytes)\n"
        )

        def retained(**env_update):
            env = {
                k: v
                for k, v in os.environ.items()
                if k != "TORCH_XPU_CONV_WEIGHT_CACHE"
            }
            env.update(env_update)
            out = subprocess.check_output(
                [sys.executable, "-c", code], env=env, text=True
            )
            return [int(v) for v in out.strip().splitlines()[-1].split()]

        self.assertEqual(retained()[0], 0)
        packed, weight_bytes = retained(TORCH_XPU_CONV_WEIGHT_CACHE="1")
        self.assertGreaterEqual(packed, weight_bytes)


instantiate_device_type_tests(
    TestConvWeightCache, globals(), only_for="xpu", allow_xpu=True
)


if __name__ == "__main__":
    run_tests()
