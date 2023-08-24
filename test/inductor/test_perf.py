# Owner(s): ["module: inductor"]
import contextlib
from unittest.mock import patch

import functorch

import torch
import torch._inductor.compile_fx
import torch._inductor.config as config
from torch._inductor import metrics
from torch.testing._internal.common_utils import IS_WINDOWS, TestCase as TorchTestCase
from torch.testing._internal.inductor_utils import HAS_CUDA

aten = torch.ops.aten


if not IS_WINDOWS:

    @torch._dynamo.optimize("count_bytes_inductor")
    def f(x):
        return torch.cat([x, x.cos()])

else:

    def f(x):
        return torch.cat([x, x.cos()])


def count_numel(f, *args):
    """
    Assumes all inputs are fp32
    """
    metrics.reset()
    torch._dynamo.optimize("count_bytes_inductor")(f)(*args)
    print(metrics.nodes_num_elem)
    return str(metrics.num_bytes_accessed // 4)


def count_numel_train(f, *args):
    """
    Assumes all inputs are fp32
    """
    metrics.reset()

    f = torch._dynamo.optimize("count_bytes_inductor")(f)
    out = f(*args)
    res = 0
    for o in out:
        res += o.mean()
    res.backward()
    print(metrics.nodes_num_elem)
    return str(metrics.num_bytes_accessed // 4)


DEVICE = "cuda"


def T(*size, dtype=torch.float32, device=DEVICE, grad=False):
    return torch.randn(size, dtype=dtype, device=device, requires_grad=grad)


def TI(*size, mx=10, dtype=torch.int32, device=DEVICE):
    return torch.randint(0, mx, size, dtype=dtype, device=device)


class TestCase(TorchTestCase):
    device = DEVICE
    pass


class NumBytesMetricTests(TestCase):
    """
    Primarily used for sanity testing that the num_bytes_accessed metrics is correct.
    """

    def test_pointwise(self):
        def f(x):
            return x.cos()

        inp = (T(10),)
        self.assertExpectedInline(count_numel(f, *inp), """20""")

        def f(x, y):
            return x + y

        inp = (T(10), T(10))
        self.assertExpectedInline(count_numel(f, *inp), """30""")

        def f(x, y):
            return x + y

        inp = (T(10, 10), T(10))
        self.assertExpectedInline(count_numel(f, *inp), """210""")

        def f(x):
            return x + x

        inp = (T(10),)
        self.assertExpectedInline(count_numel(f, *inp), """20""")

        def f(x):
            return x + x.t()

        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), """200""")

        def f(a, b, c):
            return a.cos(), b.sin() + c.sin()

        inp = (T(10), T(10), T(10))
        self.assertExpectedInline(count_numel(f, *inp), """50""")

    def test_reduction(self):
        def f(x):
            return x.sum(dim=1)

        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), """110""")

        def f(x):
            return x.sum(dim=0)

        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), """110""")

    def test_extern(self):
        def f(x):
            return torch.mm(x, x)

        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), """200""")

        def f(a, b):
            return torch.mm(a, b)

        inp = (T(10, 10), T(10, 10))
        self.assertExpectedInline(count_numel(f, *inp), """300""")

        def f(x):
            x = x.cos()
            x = torch.mm(x, x)
            x = x.cos()
            return x

        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), """600""")

        def f(x):
            a = x.cos()
            b = x.sin()
            x = torch.mm(a, b)
            return x

        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), """600""")

    def test_cat(self):
        def f(a, b):
            return torch.cat([a.sin(), b.sin()])

        inp = (T(10), T(10))
        self.assertExpectedInline(count_numel(f, *inp), """40""")

        def f(a, b):
            return torch.cat([a, b])

        inp = (T(10), T(10))
        self.assertExpectedInline(count_numel(f, *inp), """40""")

        def f(a, b):
            return torch.cat([a.cos(), b])

        inp = (T(10), T(10))
        self.assertExpectedInline(count_numel(f, *inp), """40""")

        def f(a):
            return torch.cat([a.cos(), a.sin()])

        inp = (T(10),)
        self.assertExpectedInline(count_numel(f, *inp), """30""")

    def test_index(self):
        def f(a, b):
            return a[b]

        inp = (T(10), TI(10, mx=10))
        self.assertExpectedInline(count_numel(f, *inp), """30""")


class FusionTests(TestCase):
    """
    Tests that things can be fused into a single kernel
    """

    def test_horizontal_reduction_pointwise(self):
        def f(a):
            b = a.sum(dim=1)
            c = a.cos()
            return b, c

        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), """210""")

    def test_horizontal_reduction_reduction(self):
        def f(a):
            b = a.sum(dim=1)
            c = a.amax(dim=1)
            return b, c

        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), """120""")

    def test_horizontal_reduction_pointwise2(self):
        def f(a, b):
            c = a.sum(dim=1)
            b = b.cos()
            return b + c

        inp = (T(10, 10), T(10))
        self.assertExpectedInline(count_numel(f, *inp), """120""")

    def test_horizontal_reduction_outer_pointwise(self):
        def f(a, b):
            c = a.sum(dim=0)
            b = b.cos()
            return b + c

        inp = (T(10, 10), T(10))
        self.assertExpectedInline(count_numel(f, *inp), """120""")

    def test_horizontal_sum_pw_broadcast(self):
        def f(a, b):
            a = a.sum(dim=1, keepdim=True)
            b = b.cos()
            return a * b

        inp = (T(10, 10), T(10))
        self.assertExpectedInline(count_numel(f, *inp), """210""")

    def test_vertical_sum_pw(self):
        def f(a):
            a = a.cos()
            a = a.sum(dim=1)
            return a.cos()

        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), """110""")

    def test_norm_chain(self):
        def f(a):
            b = a.sum(dim=1, keepdim=True)
            a = a * b
            b = a.sum(dim=1, keepdim=True)
            a = a * b
            b = a.sum(dim=1, keepdim=True)
            a = a * b
            return a

        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), """200""")

    def test_softmax_inner(self):
        def f(a):
            return torch.softmax(a, dim=1)

        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), """200""")

    def test_layer_norm(self):
        # TODO: Suboptimal! We shouldn't need to save normalization stats.
        mod = torch.nn.LayerNorm(10, device=self.device)

        def f(x):
            return mod(x)

        inp = (T(10, 10),)
        with torch.no_grad():
            self.assertExpectedInline(count_numel(f, *inp), """220""")

    def test_double_softmax(self):
        def f(x):
            x = torch.softmax(x, dim=1)
            x = torch.softmax(x, dim=1)
            return x

        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), """200""")

    def test_softmax_backward(self):
        def f(grad_out, out):
            return aten._softmax_backward_data(grad_out, out, 1, torch.float32)

        inp = (T(10, 10), T(10, 10))
        self.assertExpectedInline(count_numel(f, *inp), """300""")

    def test_neighbor(self):
        def f(a, b):
            return ((a - b) ** 2).sum(dim=-1).amax(dim=1)

        inp = (T(10, 1, 4), T(1, 10, 4))
        self.assertExpectedInline(count_numel(f, *inp), """90""")

    def test_factory_reduction(self):
        def f():
            a = torch.ones(10, device=self.device)
            b = torch.ones(10, 10, device=self.device)
            return (a + b).sum(dim=-1)

        inp = ()
        self.assertExpectedInline(count_numel(f, *inp), """10""")

    def test_index_pointwise(self):
        def f(a, b):
            return a[b].cos()

        inp = (T(10, 10), TI(20, mx=10))
        self.assertExpectedInline(count_numel(f, *inp), """320""")

    def test_index_reduction(self):
        def f(a, b):
            return a[b].cos().sum(dim=1)

        inp = (T(10, 10), TI(20, mx=10))
        self.assertExpectedInline(count_numel(f, *inp), """140""")

    def test_mutation_fusion(self):
        def f(a, b, c):
            a0 = a.add(c)
            b0 = b.add(a0)
            b.copy_(b0)
            a.copy_(a0)

        inp = (T(10, 10), T(10, 10), T(10, 10))
        self.assertExpectedInline(count_numel(f, *inp), """500""")


class SchedulerFusionTests(TestCase):
    """
    Testing the fusion group creation heuristic (i.e. cases where we can't fuse
    everything into a single kernel)
    Disables inductor rematerialization for easier reasoning of tests.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._stack = contextlib.ExitStack()
        cls._stack.enter_context(patch.object(config, "realize_bytes_threshold", 0))

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()
        super().tearDownClass()

    @patch.object(config, "pattern_matcher", False)
    def test_fusion_choice1(self):
        # Doesn't matter where we break fusion group here
        def f(a):
            c = a.cos()
            d = torch.mm(c, c)
            e = c.cos()
            return d + e

        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), """700""")

    @patch.object(config, "pattern_matcher", False)
    def test_fusion_choice2(self):
        # We should materialize e (it's smaller!)
        # [c, e]: 210, [f]: 210, [d]: 200
        def f(a):
            c = a.cos()
            d = torch.mm(c, c)
            e = c.sum(dim=1)
            f = d + e
            return f

        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), """620""")

    @patch.object(config, "pattern_matcher", False)
    def test_fusion_choice3(self):
        # We should materialize e.
        # [c, e]: 300, [f]: 300, [d]: 200
        def f(a):
            c = a.cos()
            d = torch.mm(c, c)
            e = c + a
            f = d + e
            return f, e

        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), """800""")


class TilingTests(TestCase):
    def test_tiling_simple(self):
        def f(a, b):
            return a + b.t()

        inp = (T(10, 10), T(10, 10))
        self.assertExpectedInline(count_numel(f, *inp), """300""")

        def f(a, b):
            return a.t() + b

        inp = (T(10, 10), T(10, 10))
        self.assertExpectedInline(count_numel(f, *inp), """300""")

    def test_tiling_three(self):
        def f(a, b, c):
            return a + b.permute(1, 2, 0) + c.permute(2, 0, 1)

        inp = (T(10, 10, 10), T(10, 10, 10), T(10, 10, 10))
        self.assertExpectedInline(count_numel(f, *inp), """4000""")


class MinCutPartitioningTests(TestCase):
    def test_partitioning_full_remat(self):
        def f(x):
            return x.cos().cos().cos()

        inp = (T(10, grad=True),)
        self.assertExpectedInline(count_numel_train(f, *inp), """50""")

    def test_partitioning_partial_remat(self):
        def f(a, b, c, d):
            x = a + b + c + d
            return x.cos().cos()

        inp = (T(10, grad=True), T(10, grad=True), T(10, grad=True), T(10, grad=True))
        self.assertExpectedInline(count_numel_train(f, *inp), """90""")

    def test_partitioning_dtype(self):
        def f(x):
            return (x < 0) * x

        inp = (T(100, grad=True),)
        self.assertExpectedInline(count_numel_train(f, *inp), """450""")

    @patch.object(functorch.compile.config, "max_dist_from_bw", 1000)
    def test_partitioning_unremat_bw(self):
        def f(x):
            return torch.mm(x, x.new_ones(x.shape)).tanh().tanh()

        inp = (T(10, 10, grad=True),)
        self.assertExpectedInline(count_numel_train(f, *inp), """1300""")

    @patch.object(config, "pattern_matcher", False)
    def test_partitioning_unremat_bw2(self):
        def f(a):
            a = torch.mm(a, a)
            a = a + 1
            b = a + 2
            c = torch.mm(a, b)
            return c

        inp = (T(10, 10, grad=True),)
        self.assertExpectedInline(count_numel_train(f, *inp), """2600""")

    def test_partitioning_keops(self):
        def f(a, b):
            return (a * b).cos().sum(dim=1)

        inp = (T(20, 1, grad=True), T(1, 20, grad=True))
        self.assertExpectedInline(count_numel_train(f, *inp), """220""")

unfusible = lambda x: aten.special_bessel_j0(x)
class NoopTests(TestCase):
    def test_noop_clones(self):
        def f(a):
            b = a.clone()
            b = unfusible(b)
            return b

        inp = T(10)
        self.assertExpectedInline(count_numel(f, inp), """20""")

        def f(a):
            b = a.clone()
            c = unfusible(b)
            return b, c
        self.assertExpectedInline(count_numel(f, inp), """40""")

    def test_noop_slice_scatter(self):
        def f(a):
            b = aten.slice_scatter(a, a)
            c = unfusible(b)
            return c
        inp = T(10)
        self.assertExpectedInline(count_numel(f, inp), """20""")


class InplacingTests(TestCase):
    def test_inplace_scatter(self):
        def f(a, b):
            a = a.cos()
            a[b] = 1
            return a

        inp = (T(10), TI(2, mx=5))
        self.assertExpectedInline(count_numel(f, *inp), """26""")

    def test_inplace_scatter_noop_view(self):
        def f(a, b):
            a[:, b] = 1
            return a

        inp = (T(10, 10), TI(2, mx=5))
        self.assertExpectedInline(count_numel(f, *inp), """42""")


# Test cases where we don't do the right thing yet.
class WouldBeNiceIfItWorked:
    def test_horizontal(self):
        def f(a):
            b = a.sum(dim=0)
            c = a.cos()
            return b, c

        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), """210""")

    # TODO: We aren't fusing outer dim softmaxes
    def test_softmax_outer(self):
        def f(a):
            return torch.softmax(a, dim=0)

        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), """200""")

    # TODO: The greedy fusion strategy results in suboptimal grouping
    @patch.object(config, "realize_bytes_threshold", 0)
    def test_fusion_choice4(self):
        def f(a, b, b2):
            c = a + b
            d = torch.mm(c, c)
            e = c + b + b2
            f = d + e + b2
            return f, e

        inp = (T(10, 10), T(10, 10, dtype=torch.float16), T(10, 10))
        self.assertExpectedInline(count_numel(f, *inp), """1000""")

    # TODO: We materialize the intermediate if we don't unroll the reduction
    def test_neighbor(self):
        def f(a, b):
            return ((a - b) ** 2).sum(dim=-1).amax(dim=1)

        inp = (T(10, 1, 8), T(1, 10, 8))
        self.assertExpectedInline(count_numel(f, *inp), """170""")


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if HAS_CUDA:
        run_tests(needs="filelock")
