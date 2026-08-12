# Owner(s): ["module: dsl-native-ops"]
#
# Tests for the CuTeDSL distribution (RNG) overrides: uniform_ and normal_ on CUDA.
#
# Unlike the other override families, these are tested for BIT-EXACTNESS against aten
# rather than routed to an OpInfo numeric suite. An RNG kernel has two observable outputs
# -- the values AND how far it advances the generator's offset -- and a statistical test
# would catch neither a counter-mapping error nor a wrong reservation. A wrong reservation
# is especially insidious: this call's values can be perfectly correct while every
# SUBSEQUENT random op in the process silently diverges.

import unittest

import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, skipIfNoCuteDSL, TestCase


def _disabled():
    return torch.backends.python_native.cutedsl.disabled()


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@skipIfNoCuteDSL
class TestCuTeDSLRng(TestCase):
    def setUp(self):
        super().setUp()
        torch.cuda.init()
        self.gen = torch.cuda.default_generators[torch.cuda.current_device()]

    def _served(self, fn):
        # Count real kernel launches to prove a call routed to us rather than aten.
        from torch._native.ops.rng import cutedsl_kernels as CK

        orig, n = CK.fill_random, [0]

        def counting(*a, **k):
            n[0] += 1
            return orig(*a, **k)

        CK.fill_random = counting
        try:
            out = fn()
        finally:
            CK.fill_random = orig
        return n[0], out

    def _both(self, fn, seed=7):
        # Run fn from an identical generator state with the override on, then off, and
        # return (ours, aten, ours_offset_delta, aten_offset_delta).
        self.gen.manual_seed(seed)
        before = self.gen.philox_state(0)[1].item()
        ours = fn()
        ours_delta = self.gen.philox_state(0)[1].item() - before
        self.gen.manual_seed(seed)
        before = self.gen.philox_state(0)[1].item()
        with _disabled():
            aten = fn()
        aten_delta = self.gen.philox_state(0)[1].item() - before
        return ours, aten, ours_delta, aten_delta

    def test_values_and_offset_are_bit_exact(self):
        # Both distributions, over sizes that exercise a single draw, several grid-stride
        # iterations, and a tail that is not a multiple of the unroll factor.
        for n in (4, 256, 1000, 4096, 100_000, 1 << 20, (1 << 20) + 7):
            for fn in (
                lambda n=n: torch.empty(n, device="cuda").uniform_(),
                lambda n=n: torch.empty(n, device="cuda").uniform_(-2.0, 2.0),
                lambda n=n: torch.empty(n, device="cuda").normal_(),
                lambda n=n: torch.empty(n, device="cuda").normal_(10.0, 2.5),
            ):
                ours, aten, od, ad = self._both(fn)
                self.assertEqual(ours, aten, atol=0, rtol=0, msg=f"n={n}")
                self.assertEqual(od, ad, msg=f"offset advance differs at n={n}")

    def test_constructors_ride_the_same_override(self):
        # rand / randn / rand_like / randn_like are CompositeExplicitAutograd wrappers
        # that allocate and then call uniform_ / normal_, so overriding those two serves
        # the whole family with no per-constructor code.
        for fn in (
            lambda: torch.rand(4096, device="cuda"),
            lambda: torch.randn(4096, device="cuda"),
            lambda: torch.rand_like(torch.empty(4096, device="cuda")),
            lambda: torch.randn_like(torch.empty(4096, device="cuda")),
        ):
            self.gen.manual_seed(11)
            n, ours = self._served(fn)
            self.assertEqual(n, 1, "constructor should route through our kernel")
            self.gen.manual_seed(11)
            with _disabled():
                self.assertEqual(ours, fn(), atol=0, rtol=0)

    def test_graph_capture_replays_fresh_values(self):
        # Under capture the kernel must LOAD seed/offset from the generator's extragraph
        # tensors (DevState) rather than bake them, so replay_prologue's refill takes
        # effect: each replay draws different values, and they match aten's replays
        # bit-for-bit.
        def capture_replays(disabled, k=3):
            ctx = _disabled() if disabled else None
            if ctx:
                ctx.__enter__()
            try:
                torch.cuda.manual_seed(99)
                x = torch.empty(4096, device="cuda")
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        x.uniform_()
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    x.uniform_()
                out = []
                for _ in range(k):
                    g.replay()
                    torch.cuda.synchronize()
                    out.append(x.clone())
                return out
            finally:
                if ctx:
                    ctx.__exit__(None, None, None)

        ours = capture_replays(False)
        self.assertNotEqual(ours[0], ours[1])  # the generator really advanced
        for a, b in zip(ours, capture_replays(True)):
            self.assertEqual(a, b, atol=0, rtol=0)

    def test_unsupported_inputs_decline(self):
        # Capability gates. fp64/fp16 need a different unroll factor and box-muller
        # variant (each its own exactness proof), an explicit generator= is not the
        # default engine we reserve from, a non-contiguous target has no flat layout, and
        # an empty one is a zero-element grid.
        gen = torch.Generator(device="cuda")
        gen.manual_seed(5)
        for fn in (
            lambda: torch.empty(64, device="cuda", dtype=torch.float64).uniform_(),
            lambda: torch.empty(64, device="cuda", dtype=torch.float16).normal_(),
            lambda: torch.empty(64, device="cuda").uniform_(0, 1, generator=gen),
            lambda: torch.empty(8, 8, device="cuda").t().uniform_(),
            lambda: torch.empty(0, device="cuda").uniform_(),
        ):
            n, _ = self._served(fn)
            self.assertEqual(n, 0, "unsupported input must fall back to aten")
        # ... and the supported cases fire, INCLUDING the degenerate from == to range,
        # which the kernel now handles via aten's own bound reversal.
        for fn in (
            lambda: torch.empty(64, device="cuda").uniform_(),
            lambda: torch.empty(64, device="cuda").uniform_(2.0, 2.0),
        ):
            n, _ = self._served(fn)
            self.assertEqual(n, 1)

    def test_one_kernel_serves_every_size(self):
        # num_iters is a RUNTIME argument, not a baked constant. When it was baked, the
        # grid-stride loop became a constexpr unroll and compile time grew LINEARLY with
        # numel (cold: 0.33s at 1 iteration, 3.2s at 56, 14.6s at 222 -- a 256M-element
        # randn paid ~15s of nvrtc), with a separate kernel per distinct iteration count.
        # Kernel count must stay O(kind x dtype x capture), independent of shape.
        from torch._native.ops.rng import cutedsl_kernels as CK

        before = len(CK._compile.cache) if hasattr(CK._compile, "cache") else None
        for n in (1 << 10, 1 << 16, 1 << 20, 1 << 24):
            torch.empty(n, device="cuda").uniform_()
        if before is not None:
            # every size shares the ONE uniform-eager kernel
            self.assertLessEqual(len(CK._compile.cache) - before, 1)


if __name__ == "__main__":
    run_tests()
