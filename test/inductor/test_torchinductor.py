# Owner(s): ["module: inductor"]
import contextlib
import dataclasses
import functools
import importlib
import itertools
import os
import random
import sys
import typing
import unittest
import weakref
from unittest.mock import patch

import torch

import torch._dynamo
from torch._dynamo.debug_utils import same_two_models
from torch._dynamo.testing import rand_strided, same
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn import functional as F
from torch.testing._internal.common_utils import (
    IS_FBCODE,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
    TestCase as TorchTestCase,
)
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_unflatten

try:
    import sympy

    importlib.import_module("functorch")
    importlib.import_module("filelock")

    import torch._inductor.config
    from functorch.compile import config as functorch_config
    from torch._decomp import get_decompositions
    from torch._inductor import codecache, config, metrics
    from torch._inductor.compile_fx import compile_fx, complex_memory_overlap
    from torch._inductor.ir import IndexingDiv, ModularIndexing
    from torch._inductor.sizevars import SizeVarAllocator
    from torch._inductor.utils import has_torchvision_roi_align, has_triton, timed

    # This will only pass on pytorch builds newer than roughly 5/15/2022
    assert get_decompositions([torch.ops.aten.trace])
    # Requires functorch
    from torch._inductor.compile_fx import compile_fx_inner
except (ImportError, AssertionError) as e:
    sys.stderr.write(f"{type(e)}: {e}\n")
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/functorch/filelock")

HAS_CPU = False
try:
    from subprocess import CalledProcessError

    from torch._inductor.codecache import CppCodeCache

    CppCodeCache.load("")
    HAS_CPU = not IS_FBCODE
except (
    CalledProcessError,
    OSError,
    torch._inductor.exc.InvalidCxxCompiler,
    torch._inductor.exc.CppCompileError,
):
    pass

aten = torch.ops.aten

HAS_CUDA = has_triton()
requires_cuda = functools.partial(unittest.skipIf, not HAS_CUDA, "requires cuda")

torch._inductor.config.triton.autotune = False  # too slow


# For OneDNN bf16 path, OneDNN requires the cpu has intel avx512 with avx512bw,
# avx512vl, and avx512dq at least. So we will skip the test case if one processor
# is not meet the requirement.
@functools.lru_cache(maxsize=None)
def has_bf16_support():
    import sys

    if sys.platform != "linux":
        return False
    with open("/proc/cpuinfo", encoding="ascii") as f:
        lines = f.read()
    return all(word in lines for word in ["avx512bw", "avx512vl", "avx512dq"])


unary_list = [
    torch.nn.ReLU(),
    torch.nn.Sigmoid(),
    torch.nn.Tanh(),
    torch.nn.Hardswish(),
    torch.nn.LeakyReLU(0.1, inplace=False),
    torch.nn.Hardtanh(min_val=-0.5, max_val=4, inplace=False),
    torch.nn.GELU(approximate="none"),
    torch.nn.GELU(approximate="tanh"),
]


binary_list = [
    lambda x, y: torch.add(x, y),  # call_function
    lambda x, y: torch.add(y, x),  # call_function
    lambda x, y: x.add(y),  # call_method
    lambda x, y: x.add_(y),  # call_method
    lambda x, y: torch.sub(x, y),  # call_function
    lambda x, y: x.sub(y),  # call_method
    lambda x, y: x.sub_(y),  # call_method
]


def requires_decomp(fn):
    """Decorator to disable test if a decomp is missing"""

    def wrap_test(test):
        @functools.wraps(test)
        def maybe_test(*args, **kwargs):
            if len(get_decompositions([fn])) == 0:
                raise unittest.SkipTest(f"requires decomp for {fn.__name__}")
            return test(*args, **kwargs)

        return maybe_test

    return wrap_test


class TestCase(TorchTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._stack = contextlib.ExitStack()
        cls._stack.enter_context(patch.object(config, "debug", True))
        cls._stack.enter_context(patch.object(config.cpp, "min_chunk_size", 1))

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()
        super().tearDownClass()


class ToTuple(torch.nn.Module):
    def forward(self, x):
        return (x,)


@dataclasses.dataclass
class InputGen:
    n: int
    device: str

    def dense(self):
        return torch.randn((self.n, self.n), device=self.device)

    def transposed(self):
        return self.dense().transpose(0, 1)

    def strided(self):
        return torch.randn((self.n * 2, self.n * 3), device=self.device)[
            self.n :, self.n :: 2
        ]

    def broadcast1(self):
        return torch.randn((self.n,), device=self.device)

    def broadcast2(self):
        return torch.randn((1, self.n, 1), device=self.device)

    def broadcast3(self):
        return torch.randn((1,), device=self.device)

    def double(self):
        return torch.randn((self.n, self.n), device=self.device, dtype=torch.double)

    def int(self):
        return torch.arange(self.n, device=self.device, dtype=torch.int32)


def compute_grads(args, kwrags, results, grads):
    def gather_leaf_tensors(args, kwargs):
        args, _ = tree_flatten(args)
        kwargs, _ = tree_flatten(kwargs)
        args = args + kwargs
        leaf_tensors = [
            arg for arg in args if isinstance(arg, torch.Tensor) and arg.requires_grad
        ]
        return leaf_tensors

    flat_results, _ = tree_flatten(results)
    flat_diff_results = [r for r in flat_results if r.requires_grad]
    assert len(flat_diff_results) > 0

    leaf_tensors = gather_leaf_tensors(args, kwrags)
    assert len(leaf_tensors) > 0
    return torch.autograd.grad(
        flat_diff_results,
        leaf_tensors,
        grads,
        allow_unused=True,
        retain_graph=True,
    )


@patch.object(torch._inductor.config.triton, "cudagraphs", False)
def check_model(
    self: TestCase,
    model,
    example_inputs,
    kwargs=None,
    *,
    atol=None,
    rtol=None,
    check_lowp=True,
    exact_dtype=True,
    nopython=True,
    copy_to_cuda=True,
    reference_in_float=True,
    assert_equal=True,
    check_gradient=False,
):
    kwargs = kwargs or {}
    torch._dynamo.reset()

    ref_inputs = example_inputs
    ref_kwargs = kwargs
    has_lowp_args = False
    original_lowp_dtype = torch.half

    if reference_in_float:
        # check_lowp is ignored here, it's kept just to be able to call `common` with extra arg
        def upcast_fn(x):
            nonlocal has_lowp_args
            if isinstance(x, torch.Tensor) and (
                x.dtype == torch.float16 or x.dtype == torch.bfloat16
            ):
                has_lowp_args = True
                return x.float()
            else:
                return x

        def get_original_lowp_dtype(example_inputs):
            dtypes = [x.dtype for x in example_inputs if isinstance(x, torch.Tensor)]
            dtype_set = set(dtypes)
            return dtype_set.pop() if len(dtype_set) == 1 else torch.half

        ref_inputs = list(map(upcast_fn, example_inputs))
        ref_kwargs = {k: upcast_fn(v) for k, v in kwargs.items()}
        if has_lowp_args:
            original_lowp_dtype = get_original_lowp_dtype(example_inputs)
            if hasattr(model, "to"):
                model = model.to(torch.float)

    torch.manual_seed(0)

    correct = model(*ref_inputs, **ref_kwargs)
    # downcast the model back if needed
    if reference_in_float and has_lowp_args:
        if hasattr(model, "to"):
            model = model.to(original_lowp_dtype)

    torch._inductor.metrics.reset()

    called = False

    def compile_fx_wrapper(model_, example_inputs_):
        nonlocal called
        called = True
        return compile_fx(model_, example_inputs_)

    def run(*ex, **kwargs):
        return model(*ex, **kwargs)

    run = torch._dynamo.optimize(compile_fx_wrapper, nopython=nopython)(run)

    torch.manual_seed(0)
    actual = run(*example_inputs, **kwargs)
    # if not called:
    #     exp = torch._dynamo.explain(run, *example_inputs)
    #     print("Explain:", exp[0])
    #     for graph in exp[2]:
    #         print("Graph", graph)
    assert called, "Ran graph without calling compile_fx"
    assert type(actual) == type(correct)

    correct_flat, correct_spec = tree_flatten(correct)
    actual_flat, _ = tree_flatten(actual)
    if reference_in_float:
        correct_flat = tuple(
            y.to(x.dtype)
            if isinstance(y, torch.Tensor) and y.dtype.is_floating_point
            else y
            for x, y in zip(actual_flat, correct_flat)
        )
        correct = tree_unflatten(correct_flat, correct_spec)

    if assert_equal:
        self.assertEqual(
            actual,
            correct,
            atol=atol,
            rtol=rtol,
            equal_nan=True,
            exact_dtype=exact_dtype,
        )
    else:
        for correct_val, actual_val in zip(correct_flat, actual_flat):
            if isinstance(correct_val, torch.Tensor):
                assert correct_val.device == actual_val.device
                assert correct_val.size() == actual_val.size()
                assert correct_val.stride() == actual_val.stride()
                assert correct_val.layout == actual_val.layout
                if exact_dtype:
                    assert correct_val.dtype == actual_val.dtype

    if check_gradient:

        # generate random unit norm gradients
        grads = [
            torch.rand(r.shape, device=r.device, dtype=r.dtype)
            for r in correct_flat
            if r.requires_grad
        ]
        for g in grads:
            g /= g.norm()

        correct_grad = compute_grads(ref_inputs, ref_kwargs, correct, grads)
        actual_grad = compute_grads(example_inputs, kwargs, actual, grads)

        self.assertEqual(
            actual_grad,
            correct_grad,
            atol=atol,
            rtol=rtol,
            equal_nan=True,
            exact_dtype=exact_dtype,
        )

    torch._dynamo.reset()


@patch.object(torch._inductor.config.triton, "cudagraphs", False)
def check_model_cuda(
    self: TestCase,
    model,
    example_inputs,
    kwargs=None,
    *,
    atol=None,
    rtol=None,
    check_lowp=True,
    exact_dtype=True,
    nopython=True,
    copy_to_cuda=True,
    reference_in_float=True,
    assert_equal=True,
    check_gradient=False,
):
    kwargs = kwargs or {}
    if hasattr(model, "to"):
        model = model.to("cuda")

    def copy_fn(x):
        # preserve strides of the input on the device
        if not isinstance(x, torch.Tensor):
            return x
        return torch.empty_strided(
            x.size(), x.stride(), device="cuda", dtype=x.dtype
        ).copy_(x)

    if copy_to_cuda:
        example_inputs = tuple(copy_fn(x) for x in example_inputs)

    check_model(
        self,
        model,
        example_inputs,
        kwargs,
        atol=atol,
        rtol=rtol,
        exact_dtype=exact_dtype,
        nopython=nopython,
        reference_in_float=reference_in_float,
        assert_equal=assert_equal,
        check_gradient=check_gradient,
    )

    if check_lowp:

        def downcast_fn(x):
            if not isinstance(x, torch.Tensor) or not x.dtype == torch.float:
                return x
            return torch.empty_strided(
                x.size(), x.stride(), device="cuda", dtype=torch.half
            ).copy_(x)

        example_inputs = list(map(downcast_fn, example_inputs))
        if hasattr(model, "to"):
            model = model.to(torch.half)
        check_model(
            self,
            model,
            example_inputs,
            kwargs,
            atol=atol,
            rtol=rtol,
            exact_dtype=exact_dtype,
            nopython=nopython,
            reference_in_float=reference_in_float,
            assert_equal=assert_equal,
            check_gradient=check_gradient,
        )


class SweepInputs2:
    input_gen_types1 = [
        "dense",
        "transposed",
        "strided",
        "broadcast1",
        "broadcast2",
        "broadcast3",
        "double",
        "int",
    ]
    input_gen_types2 = input_gen_types1
    gen = None

    @staticmethod
    def kernel(a, b):
        return (a + b,)

    @classmethod
    def gen_template(cls, name1, name2):
        def test(self):
            check_model(
                self,
                cls.kernel,
                (
                    getattr(cls.gen, name1)(),
                    getattr(cls.gen, name2)(),
                ),
            )

        test.__name__ = f"test_{cls.gen.device}_{name1}_{name2}"
        setattr(cls, test.__name__, test)

    @classmethod
    def populate(cls):
        for name1 in cls.input_gen_types1:
            for name2 in cls.input_gen_types2:
                cls.gen_template(name1, name2)


class TestIndexingSimplification(TorchTestCase):
    def test_indexing_simplification(self):
        sizevars = SizeVarAllocator()
        i0 = sympy.Symbol("i0")
        i1 = sympy.Symbol("i1")
        i2 = sympy.Symbol("i2")
        r3 = sympy.Symbol("r3")

        var_ranges = {i0: 3136, i1: 64, i2: 32, r3: 3}
        expr = (
            128 * i2
            + ModularIndexing(i1, 1, 64)
            + 64 * ModularIndexing(i1 + 64 * r3, 64, 2)
        )
        # check that `i1//64` is removed when i1 is always less than 64,
        # and the next simplificaton doesn't happen
        self.assertEqual(
            sizevars.simplify_with_ranges(expr, var_ranges),
            i1 + 128 * i2 + 64 * ModularIndexing(r3, 1, 2),
        )
        # all the modular indexing should be removed when the body cant be larger than the modulus
        var_ranges[r3] = 2
        self.assertEqual(
            sizevars.simplify_with_ranges(expr, var_ranges), i1 + 128 * i2 + 64 * r3
        )

        # small terms should be kept if the rest is not guaranteed to be divisible
        self.assertEqual(
            sizevars.simplify_with_ranges(IndexingDiv(r3 + i2 + i1, 32), var_ranges),
            IndexingDiv(r3 + i2 + i1, 32),
        )

        expr = ModularIndexing(2 * i2 + r3, 1, 64)
        # modular indexing is removed if base is smaller than modulo
        self.assertEqual(sizevars.simplify_with_ranges(expr, var_ranges), 2 * i2 + r3)

        # check the same thing but with symbolic divisor
        self.assertEqual(IndexingDiv(r3 * i0, r3), i0)
        self.assertEqual(ModularIndexing(r3 * i0, r3, 10), ModularIndexing(i0, 1, 10))

        # (10*i) % 10 is always zero and should get optimized away
        self.assertEqual(
            ModularIndexing(i0 + i1 * 10, 1, 10), ModularIndexing(i0, 1, 10)
        )

        # ((20*i)//2) % 10 is always zero and should get optimized away
        self.assertEqual(
            ModularIndexing(i0 + i1 * 20, 2, 10), ModularIndexing(i0, 2, 10)
        )

        # the same things happens with symbolic divisor
        self.assertEqual(
            ModularIndexing(i0 + i1 * i2 * r3, i2, r3), ModularIndexing(i0, i2, r3)
        )

        # Constant fold from divisor into base
        self.assertEqual(ModularIndexing(i0 * 4, 2, 10), ModularIndexing(i0 * 2, 1, 10))
        self.assertEqual(IndexingDiv(i0 * 4, 2), i0 * 2)

        # Nested modular indexing is correctly simplified
        var_ranges = {"i1": 13, "i2": 121}
        expr = ModularIndexing(ModularIndexing(121 * i1 + i2, 1, 784), 1, 28)
        self.assertEqual(sizevars.simplify_with_ranges(expr, var_ranges), expr)
        expr = ModularIndexing(ModularIndexing(121 * i1 + i2, 1, 784) + 1, 1, 28)
        self.assertEqual(sizevars.simplify_with_ranges(expr, var_ranges), expr)
        var_ranges = {"i2": 784}
        expr = ModularIndexing(ModularIndexing(i2, 1, 28), 7, 4)
        expected = IndexingDiv(ModularIndexing(i2, 1, 28), 7)
        self.assertEqual(sizevars.simplify_with_ranges(expr, var_ranges), expected)
        expr = ModularIndexing(ModularIndexing(i2, 1, 28) + 1, 7, 4)
        self.assertEqual(sizevars.simplify_with_ranges(expr, var_ranges), expr)

    def test_indexing_join(self):
        sizevars = SizeVarAllocator()
        i0 = sympy.Symbol("i0")
        i1 = sympy.Symbol("i1")
        i2 = sympy.Symbol("i2")

        # join two ModularIndexing calls into one larger one when possible
        expr1 = ModularIndexing(i0, 1, 32) + 32 * ModularIndexing(i0, 32, 4)
        self.assertEqual(
            sizevars.simplify_with_ranges(expr1, {}), ModularIndexing(i0, 1, 128)
        )

        # it should also work with a scale
        self.assertEqual(
            sizevars.simplify_with_ranges(2 * expr1, {}),
            2 * ModularIndexing(i0, 1, 128),
        )

        # it should work when divisor is not 1
        expr2 = ModularIndexing(i0, 3, 32) + 32 * ModularIndexing(i0, 32 * 3, 4)
        simplified = sizevars.simplify_with_ranges(expr2, {})
        self.assertEqual(simplified, ModularIndexing(i0, 3, 128))
        self.assertEqual(expr2.subs({i0: 39485}), simplified.subs({i0: 39485}))

        # it should not happen in this case as the modulus is wrong
        expr3 = ModularIndexing(i0, 1, 30) + 32 * ModularIndexing(i0, 32, 4)
        self.assertEqual(sizevars.simplify_with_ranges(expr3, {}), expr3)

        # check that it also works with a modulus>1
        expr4 = ModularIndexing(i0, 10, i1) + i1 * ModularIndexing(i0, i1 * 10, i2)
        res0 = expr4.subs({i0: 24056, i1: 13, i2: 19})
        simplified = sizevars.simplify_with_ranges(expr4, {})
        res1 = simplified.subs({i0: 24056, i1: 13, i2: 19})
        self.assertEqual(res0, res1)
        self.assertEqual(simplified, ModularIndexing(i0, 10, i1 * i2))

        # and also works with an offset
        self.assertEqual(
            sizevars.simplify_with_ranges(expr4 + 10, {}),
            ModularIndexing(i0, 10, i1 * i2) + 10,
        )

        # works for ModularIndexing + IndexingDiv
        expr5 = 197 * IndexingDiv(i0, 197) + ModularIndexing(i0, 1, 197)
        simplified = sizevars.simplify_with_ranges(expr5, {})
        self.assertEqual(simplified, i0)
        self.assertEqual(expr5.subs({i0: 39485}), simplified.subs({i0: 39485}))

        # works with a scale
        self.assertEqual(
            sizevars.simplify_with_ranges(2 * expr5, {}),
            2 * i0,
        )

        # divisor != 1
        expr6 = 197 * IndexingDiv(i0, 197 * 3) + ModularIndexing(i0, 3, 197)
        simplified = sizevars.simplify_with_ranges(expr6, {})
        self.assertEqual(simplified, IndexingDiv(i0, 3))
        self.assertEqual(expr6.subs({i0: 39485}), simplified.subs({i0: 39485}))


class CommonTemplate:
    @classmethod
    def install(my_cls, other_cls, suffix):  # noqa: B902
        for name, value in my_cls.__dict__.items():
            if name.startswith("test_"):
                setattr(other_cls, f"{name}_{suffix}", value)

    def test_bool(self):
        def fn(a, b):
            return (
                a + b,
                a * b,
                a & b,
                a | b,
                a ^ b,
                torch.logical_and(a, b),
                torch.logical_or(a, b),
                torch.logical_not(a),
                torch.sign(b),
            )

        self.common(
            fn,
            (
                torch.tensor([True, False, True, False]),
                torch.tensor([False, False, True, True]),
            ),
        )

    def test_add_const_int(self):
        def fn(a):
            return (a + 1,)

        self.common(fn, (torch.randn(32),))

    def test_add_const_float(self):
        def fn(a):
            return (a + 1.5,)

        self.common(fn, (torch.randn(32),))

    def test_add_inplace_permuted(self):
        def fn(x, y):
            return x.add_(y)

        x = torch.ones([2, 12, 13, 17]).transpose(1, 2)
        y = torch.randn([2, 13, 1, 17])

        self.common(fn, (x, y))

    def test_abs(self):
        def fn(a):
            return (a / (torch.abs(a) + 1),)

        self.common(fn, (torch.randn(17),))

    def test_sgn(self):
        def fn(a):
            return torch.sgn(a), torch.sgn(a + 1) - 1

        self.common(fn, [torch.linspace(-10, 10, 41)])

    def test_max_min(self):
        def fn(a, b):
            return (torch.maximum(a, b), torch.minimum(a, b))

        self.common(fn, (torch.randn(8), torch.randn(8)))

    def test_horizonal_fusion1(self):
        def fn(a, b, c):
            return (a + b, a - c, b * c)

        self.common(
            fn, (torch.randn(8, 16, 16), torch.randn(8, 16, 16), torch.randn(1, 16, 1))
        )

    def test_horizonal_fusion2(self):
        def fn(a, b, c):
            return a + 1, b + 2, c + 3

        self.common(fn, (torch.randn(8, 16, 8), torch.randn(8, 16), torch.randn(16, 8)))

    def test_vertical_fusion1(self):
        def fn(sa, ct, p):
            # From torchbench.pyhpc_equation_of_state
            v17 = -3.087032500374211e-7
            v18 = -1.988366587925593e-8
            v19 = -1.061519070296458e-11
            v20 = 1.550932729220080e-10
            t15 = v19 * ct
            t19 = v17 + ct * (v18 + t15) + v20 * sa
            t20 = 1.0 / t19
            t128 = t19 * p
            return t20 + t128

        self.common(
            fn,
            (
                torch.randn(204, 204, 26),
                torch.randn(204, 204, 26),
                torch.randn(26),
            ),
        )
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    def test_sum1(self):
        def fn(a, b):
            return ((a + b).sum(-1),)

        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_sum2(self):
        def fn(a, b):
            return ((a + b).sum([1, 2]), (a + b).sum(-1))

        self.common(fn, (torch.randn(8, 9, 3, 21), torch.randn(8, 9, 3, 21)))

    def test_sum3(self):
        def fn(a, b):
            r1 = a + b
            r2 = r1.sum(-1)
            r3 = torch.squeeze(b) + 10
            return (r1, r2, r3)

        # Mismatched elements: 2 / 10 (20.0%)
        # Greatest absolute difference: 0.0029296875 at index (8,) (up to 1e-05 allowed)
        # Greatest relative difference: 0.0017482517482517483 at index (6,) (up to 0.001 allowed)
        self.common(fn, (torch.randn(10, 10), torch.randn(1, 10)), atol=1e-5, rtol=2e-3)

    def test_sum4(self):
        def fn(a):
            b = a + 1
            c = b.sum(-1)
            d = c + 3
            e = d.sum(-1)
            f = e + 5
            return (f, e, d, c, b)

        self.common(fn, (torch.randn(1, 16, 8, 8),))

    def test_sum5(self):
        def fn(a):
            b = a + 1
            c = b.sum(-1)
            d = c + 3
            e = d.sum(-1)
            f = e + 5
            return (f,)

        self.common(fn, (torch.randn(1, 17, 8, 9),))

    def test_reduction1(self):
        def fn(a):
            return (a.sum(), a.max(), a.min(), a.argmax(), a.argmin())

        self.common(fn, (torch.tensor([float("-inf"), 0.0, float("inf")]),))

    def test_reduction2(self):
        def fn(a):
            # FIXME: a.argmax
            return (a.sum(), a.max(), a.min(), a.argmin())

        self.common(fn, (torch.full((4,), float("inf")),))

    def test_reduction3(self):
        def fn(a):
            # FIXME: a.argmin
            return (a.sum(), a.max(), a.min(), a.argmax())

        self.common(fn, (torch.full((4,), float("-inf")),))

    def test_reduction4(self):
        if self.device == "cpu":
            raise unittest.SkipTest("Non-deterministic CPU results")

        def fn(a):
            return (a.argmax(-1), a.argmin(-1))

        inputs = (torch.ones(128), torch.ones(4, 4, 1))
        for i in inputs:
            self.common(fn, (i,))

    @patch.object(config, "dynamic_shapes", False)
    def test_unroll_small_reduction(self):
        def fn(x):
            val1, index1 = x.min(-1)
            val2, index2 = x.max(-1)
            return (
                val1,
                index1,
                val2,
                index2,
                x.sum(-1),
                (x > 1).any(-1),
                (x > 0).all(-1),
                x.argmin(-1),
                x.argmax(-1),
                x.amin(-1),
                x.amax(-1),
            )

        with patch.object(config, "unroll_reductions_threshold", 8):
            # small sized reductions will get unrolled
            self.common(fn, (torch.randn(8, 3),))
        torch._dynamo.reset()
        with patch.object(config, "unroll_reductions_threshold", 1):
            # make sure things also work if they aren't unrolled
            self.common(fn, (torch.randn(8, 3),))

    def test_multilayer_low_prec(self):
        # fp16 nyi for cpu
        if self.device == "cpu":
            raise unittest.SkipTest("requires CUDA")

        def fn(a):
            return torch.mean(a)

        self.common(fn, ((torch.rand((10, 3, 352, 352), dtype=torch.float16),)))

    def test_expanded_reduction(self):
        if self.device == "cpu":
            raise unittest.SkipTest(
                "https://github.com/pytorch/torchdynamo/issues/1697"
            )

        def fn(x, y):
            z = x * y
            return z.sum((0, 1))

        self.common(fn, (torch.randn(2, 197, 256), torch.randn(2, 1, 256)))

    def test_min_max_reduction(self):
        def fn(a, b):
            return ((a + b).max(), (a + b).min(), torch.amax(a + 1, keepdim=True))

        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_sum_int(self):
        def fn(x):
            return 2 * x.sum(-1) + x.sum()

        dtypes = torch.bool, torch.uint8, torch.int
        inps = [torch.randint(2, (64,), dtype=dtype) for dtype in dtypes]
        for i in inps:
            self.common(fn, (i,), check_lowp=False)

    def test_sum_dtype(self):
        def fn(x):
            return x * x.sum(-1, dtype=torch.double) + x.sum(dtype=torch.double)

        self.common(fn, (torch.ones(32, 32) * 70,))

    def test_clamp(self):
        def fn(a, b):
            return (a.clamp(-0.1, 0.1), b.clamp(0), torch.clamp(a + b, max=0))

        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_arange1(self):
        def fn(x):
            rng1 = torch.arange(8 * 8, dtype=torch.float32, device=x.device).view(8, 8)
            rng2 = torch.arange(10, 18, device=x.device)
            tmp = x * rng1
            return tmp, tmp + rng2

        self.common(fn, (torch.randn(8, 8),))

    def test_arange2(self):
        def fn(x):
            rng1 = torch.arange(8, device=x.device)
            return (x + rng1,)

        self.common(fn, (torch.randint(4, (8, 8)),), check_lowp=False)

    def test_arange3(self):
        def fn(x):
            return x + torch.ops.aten.arange.start_step(
                0, 53, 4, dtype=torch.int64, device=x.device
            )

        self.common(fn, (torch.randn(14),))

    def test_arange4(self):
        def fn(x):
            return x - torch.arange(512, -512, -1.0, device=x.device)

        self.common(fn, (torch.randn(1024),))

    def test_linspace(self):
        def fn(x):
            return torch.linspace(0.125, 0.875, 7, device=x.device) + x

        self.common(fn, (torch.randn(1, 7),))

    def test_tensor1(self):
        def fn(x):
            return torch.tensor([1], device=x.device) + x, torch.tensor(
                5, device=x.device
            )

        self.common(fn, (torch.randn(10),))

    def test_tensor2(self):
        def fn(x):
            return torch.tensor(list(range(2, 40, 2)), device=x.device) + x

        self.common(fn, (torch.randn(1),))

    def test_tensor3(self):
        def fn(x):
            return (
                torch.tensor([], device=x.device),
                torch.tensor([1, 2], device=x.device) + 1,
                torch.tensor([1, 2, 3], device=x.device) + 2,
                torch.tensor([1, 2, 3, 4], device=x.device) + x,
            )

        self.common(fn, [torch.randn(4)])

    def test_views1(self):
        def fn1(x, y):
            return (x.view(size2) + y,)

        def fn2(x, y):
            return ((x + 1).view(size2) + y,)

        views = [
            ([5 * 7], [5, 7]),
            ([2 * 3 * 4 * 5 * 6 * 7], [2, 3, 4, 5, 6, 7]),
            ([2 * 3, 4, 5, 6 * 7], [2, 3, 4, 5, 6, 7]),
            ([10 * 5, 20], [10, 5, 20]),
            ([1, 10, 1], [10]),
            ([10, 1, 10, 1, 10], [10, 100]),
            ([2, 2, 2, 2], [4, 4]),
        ]
        for size1, size2 in views:
            self.common(fn1, (torch.randn(size1), torch.randn(size2)))
            self.common(fn2, (torch.randn(size1), torch.randn(size2)))

        for size2, size1 in views:
            self.common(fn1, (torch.randn(size1), torch.randn(size2)))
            self.common(fn2, (torch.randn(size1), torch.randn(size2)))

    def test_views2(self):
        def fn1(x):
            return (x.view(size2) + 1,)

        def fn2(x):
            return ((x * 2).view(size2) + 1,)

        for size1, size2 in [
            ([2, 2, 2, 2], [4, -1]),
            ([10, 1, 10, 1, 10], [-1, 100]),
            ([10 * 5, 20], [10, -1, 20]),
        ]:
            self.common(fn1, (torch.randn(size1),))
            self.common(fn2, (torch.randn(size1),))

    def test_views3(self):
        # example taken from hf_BigBird
        def forward(arg1, arg2):
            index = torch.ops.aten.index(arg1, [arg2])
            view_1 = torch.ops.aten.view(index, [1, 2232, 64])
            view_2 = torch.ops.aten.view(view_1, [1, 12, 62, 192])
            return view_2

        self.common(
            forward,
            (
                rand_strided((64, 64), (64, 1), torch.float32),
                rand_strided((2232,), (1,), torch.int64),
            ),
        )

    def test_relu(self):
        def fn(a, b):
            return (torch.relu(a), torch.relu(a + b) / 10)

        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_exp(self):
        def fn(a, b):
            return (torch.exp(a), torch.exp(a + b))

        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_sigmoid(self):
        def fn(a, b):
            return (torch.sigmoid(a), torch.sigmoid(a + b))

        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_round(self):
        def fn(a, b):
            return torch.round(a), torch.round(b + 1), torch.round(a, decimals=2)

        # without manual_seed, there is some chance this test fails due to:
        # https://github.com/openai/triton/issues/530
        torch.manual_seed(0)

        # with *100 we are always getting a number exactly at .5 which we don't do right in half
        self.common(fn, (torch.randn(8, 8) * 100, torch.randn(8, 8) * 10))

    def test_round_correctness(self):
        if self.device == "cuda":
            raise unittest.SkipTest("need to debug tl.libdevice on A100/V100")

        def fn(a):
            return torch.round(a)

        self.common(
            fn,
            [torch.arange(-10, 10, 0.1, dtype=torch.float64)],
            check_lowp=False,
        )

    def test_silu(self):
        def fn(a):
            return (torch.nn.functional.silu(a),)

        self.common(fn, (torch.randn(8, 8),))

    # TODO(voz): Re-enable this test ASAP https://github.com/pytorch/pytorch/issues/82763
    @unittest.skip("Skipping due to op bugs")
    def test_nan_to_num(self):
        def fn(a):
            return (
                torch.nan_to_num(a),
                torch.nan_to_num(a, nan=3.0),
                torch.nan_to_num(a, nan=None),
                torch.nan_to_num(a, posinf=4.0),
                torch.nan_to_num(a, neginf=5.0),
                torch.nan_to_num(a, nan=3.0, posinf=4.0, neginf=5.0),
            )

        self.common(
            fn,
            (torch.tensor((float("nan"), float("inf"), float("-inf"), 1.0)),),
            check_lowp=False,  # a much more elaborate test is required to match finfo max's for float and half
        )

    def test_div1(self):
        def fn(a, b):
            return (
                aten.div(a, b, rounding_mode=None),
                aten.div(a, b, rounding_mode="floor"),
                aten.div(a, b, rounding_mode="trunc"),
                a / b,
                a // b,
            )

        self.common(fn, (torch.randn(8, 8) * 100, torch.randn(8, 8) * 100))

    def test_div2(self):
        def fn(a, b):
            return (
                aten.div(a, b, rounding_mode=None),
                aten.div(a, b, rounding_mode="floor"),
                aten.div(a, b, rounding_mode="trunc"),
                a / b,
                a // b,
            )

        self.common(fn, (torch.randint(-100, 100, [8, 8]), 100 * torch.randn(8, 8)))

    def test_div3(self):
        def fn(a, b):
            return (
                aten.div(a, b, rounding_mode=None),
                aten.div(a, b, rounding_mode="floor"),
                aten.div(a, b, rounding_mode="trunc"),
                a / b,
                a // b,
            )

        a = torch.randint(1, 100, [8, 8])
        self.common(fn, (a * 2, a))

    def test_div4(self):
        def fn(a, b):
            return (
                aten.div(a, b, rounding_mode=None),
                aten.div(a, b, rounding_mode="floor"),
                aten.div(a, b, rounding_mode="trunc"),
                a / b,
                a // b,
            )

        self.common(
            fn,
            (torch.randint(-100, 0, [8, 8]), torch.randint(1, 10, [8, 8])),
        )

    def test_div5(self):
        def fn(a, b):
            return (
                aten.div(a, b, rounding_mode=None),
                aten.div(a, b, rounding_mode="floor"),
                aten.div(a, b, rounding_mode="trunc"),
                a / b,
                a // b,
            )

        # divide a scalar
        self.common(fn, (torch.randint(-100, 0, [8, 8]), 16))

    def test_div6(self):
        def fn(a, b):
            return (
                aten.div(a, b, rounding_mode=None),
                aten.div(a, b, rounding_mode="floor"),
                aten.div(a, b, rounding_mode="trunc"),
                a / b,
                a // b,
            )

        # treat boolean as integer
        self.common(
            fn,
            (torch.ones([8, 8], dtype=torch.bool), torch.randint(-100, -1, [8, 8])),
        )

    def test_div7(self):
        def fn(a, b):
            return (
                aten.div(a, b, rounding_mode=None),
                aten.div(a, b, rounding_mode="floor"),
                aten.div(a, b, rounding_mode="trunc"),
                a / b,
                a // b,
            )

        self.common(
            fn,
            (
                torch.randint(2**32, 2**40, [100, 100]),
                torch.randint(-10, -1, [100, 100]),
            ),
        )

    def test_div8(self):
        def fn(a, b):
            return (
                aten.div(a, b, rounding_mode=None),
                aten.div(a, b, rounding_mode="floor"),
                aten.div(a, b, rounding_mode="trunc"),
                a / b,
                a // b,
            )

        self.common(fn, (1024, 100))

    def test_both_scalars(self):
        def fn(a, b):
            return (
                aten.add(a, b),
                aten.add(b, a),
                aten.sub(a, b),
                aten.sub(b, a),
                aten.mul(a, b),
                aten.mul(b, a),
            )

        self.common(fn, (4, 3.3), reference_in_float=False)

    def test_sum_keepdims(self):
        def fn(a, b):
            return (torch.sum(a + b, -1, keepdim=True),)

        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_softmax(self):
        def fn(a, b):
            return (torch.softmax(a + b, -1), torch.softmax(a, 0), torch.softmax(b, 1))

        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_log_softmax(self):
        def fn(a, b):
            return (F.log_softmax(a + b, -1), F.log_softmax(a, 0), F.log_softmax(b, 1))

        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_transpose(self):
        def fn(a, b):
            return (
                torch.t(a) + b,
                torch.transpose(b * 2, 0, 1) + 10,
            )

        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_permute(self):
        def fn(a):
            return (
                torch.permute(a + 1, [2, 1, 4, 0, 3]) + 2,
                torch.permute(a, [2, 1, 4, 0, 3]) + 2,
            )

        self.common(fn, (torch.randn(2, 2, 2, 2, 2),))

    def test_expand(self):
        def fn(a):
            return (
                (a + 1).expand(3, 4, 2, 3, 2) + 2,
                a.expand(2, 1, 2, 3, 2) + 2,
            ), a.expand(2, -1, 5, -1)

        self.common(fn, (torch.randn(2, 1, 2),))

    def test_squeeze1(self):
        def fn(a):
            return ((a + 1).squeeze() + 2, a.squeeze() + 2)

        self.common(fn, (torch.randn(1, 2, 1, 2, 2, 1, 1),))

    def test_squeeze2(self):
        def fn(a):
            return ((a + 1).squeeze(-1).squeeze(2) + 2, a.squeeze(0) + 2)

        self.common(fn, (torch.randn(1, 2, 1, 2, 2, 2, 1),))

    def test_simplify_loops(self):
        def fn(a, b):
            return a + b

        self.common(
            fn,
            (
                torch.randn(2, 3, 4, 5, 6),
                torch.randn(4, 2, 3, 5, 6).permute(1, 2, 0, 3, 4),
            ),
        )

    def test_unsqueeze(self):
        def fn(a):
            return (
                torch.unsqueeze(a + 1, -1) + 2,
                torch.unsqueeze(a, 2) + 2,
                torch.unsqueeze(a + 1, 0) + 2,
                torch.unsqueeze(a, -2) + 2,
            )

        self.common(
            fn,
            (
                torch.randn(
                    2,
                    2,
                    2,
                    2,
                ),
            ),
        )

    def test_unsqueeze_inplace(self):
        def fn(a):
            tmp1 = a + 1
            aten.unsqueeze_(tmp1, 2)
            tmp2 = aten.unsqueeze_(a + 1, 0) + 2
            return (tmp1, tmp2)

        self.common(
            fn,
            (
                torch.randn(
                    2,
                    2,
                    2,
                    2,
                ),
            ),
        )

    def test_addmm(self):
        def fn(a, b, c):
            return (torch.addmm(a + 1, b + 2, c + 3) + 4,)

        self.common(
            fn,
            (
                torch.randn(8, 8),
                torch.randn(8, 8),
                torch.randn(8, 8),
            ),
        )

    def test_linear1(self):
        mod = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.Sigmoid(),
            ToTuple(),
        )
        self.common(mod, (torch.randn(2, 8),))

    def test_linear2(self):
        mod = torch.nn.Sequential(
            torch.nn.Linear(8, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 8),
            torch.nn.ReLU(),
        )
        self.common(mod, (torch.randn(2, 8),))

    def test_bmm1(self):
        def fn(a, b):
            return (
                torch.bmm(a, b),
                torch.bmm(a + 1, b + 2) + 3,
            )

        self.common(
            fn,
            (
                torch.randn(2, 8, 8),
                torch.randn(2, 8, 8),
            ),
            check_lowp=False,
        )
        self.common(
            fn,
            (
                torch.randn(1, 16, 8),
                torch.randn(1, 8, 10),
            ),
            check_lowp=False,
        )

    def test_bmm2(self):
        def fn(a, b):
            return torch.bmm(a.permute(0, 2, 1), b)

        self.common(
            fn,
            (
                torch.randn(1, 8, 8),
                torch.randn(1, 8, 8),
            ),
            check_lowp=False,
        )

    # For gpu path, there has a accurcy issue,
    @unittest.skipIf(HAS_CUDA, "only support cpu conv  bn test")
    def test_conv_bn_fuse(self):
        input_shapes = {1: (112,), 2: (112, 112), 3: (55, 55, 55)}
        conv_modules = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}
        bn_modules = {
            1: torch.nn.BatchNorm1d,
            2: torch.nn.BatchNorm2d,
            3: torch.nn.BatchNorm3d,
        }
        options = itertools.product(
            [1, 2, 3],
            [True, False],
            [1, 3],
            [1, 2],
            [1, 4],
        )

        for (
            dim,
            bias,
            kernel_size,
            dilation,
            groups,
        ) in options:
            oC = 32 * groups
            iC = 3 * groups
            x_shape = (1, iC) + input_shapes[dim]
            mod = torch.nn.Sequential(
                conv_modules[dim](
                    iC,
                    oC,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                ),
                bn_modules[dim](oC),
            ).eval()
            test_memory_format = [torch.contiguous_format]
            # TODO: GPU path doesn't support channels_last now.
            if not HAS_CUDA and dim > 1:
                channels_last = (
                    torch.channels_last if dim == 2 else torch.channels_last_3d
                )
                test_memory_format.append(channels_last)
            for memory_format in test_memory_format:
                v = torch.randn(x_shape, dtype=torch.float32).to(
                    memory_format=memory_format
                )
                with torch.no_grad():
                    self.common(
                        mod,
                        (v,),
                    )

    # For gpu path, there has a accurcy issue,
    # see https://github.com/pytorch/pytorch/issues/87745.
    @unittest.skipIf(HAS_CUDA, "only support cpu conv2d unary test")
    def test_conv2d_unary(self):
        test_memory_format = [torch.contiguous_format, torch.channels_last]
        options = itertools.product(
            unary_list,
            [True, False],
            [1, 3],
            [1, 2],
            [1, 4],
            test_memory_format,
        )

        for (
            unary_fn,
            bias,
            kernel_size,
            dilation,
            groups,
            memory_format,
        ) in options:
            oC = 32 * groups
            iC = 3 * groups
            x_shape = (1, iC, 112, 112)
            mod = torch.nn.Sequential(
                torch.nn.Conv2d(
                    iC,
                    oC,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                ),
                unary_fn,
            ).eval()

            # TODO: add bf16 test for cpu path?
            v = torch.randn(x_shape, dtype=torch.float32).to(
                memory_format=memory_format
            )
            with torch.no_grad():
                self.common(
                    mod,
                    (v,),
                )

    # For gpu path, there has a accurcy issue,
    # see https://github.com/pytorch/pytorch/issues/87745.
    @unittest.skipIf(HAS_CUDA, "only support cpu conv2d binary test")
    def test_conv2d_binary(self):
        class M(torch.nn.Module):
            def __init__(
                self,
                binary_fn,
                in_channels,
                out_channels,
                dilation,
                groups,
                bias,
                has_relu,
                **kwargs,
            ):
                super(M, self).__init__()
                self.conv1 = torch.nn.Conv2d(
                    in_channels,
                    out_channels,
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                    **kwargs,
                )
                self.conv2 = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels,
                        out_channels,
                        dilation=dilation,
                        groups=groups,
                        bias=bias,
                        **kwargs,
                    )
                )
                self.binary_fn = binary_fn
                self.relu = torch.nn.ReLU() if has_relu else torch.nn.Identity()

            def forward(self, x):
                x1 = self.conv1(x)
                x2 = self.conv2(x)
                return self.relu(self.binary_fn(x1, x2))

        test_memory_format = [torch.contiguous_format, torch.channels_last]
        options = itertools.product(
            binary_list,
            [True, False],
            [True, False],
            [1, 3],
            [1, 2],
            [1, 4],
            test_memory_format,
        )

        for (
            binary_fn,
            has_relu,
            bias,
            kernel_size,
            dilation,
            groups,
            memory_format,
        ) in options:
            oC = 32 * groups
            iC = 3 * groups
            x_shape = (1, iC, 112, 112)
            mod = M(
                binary_fn,
                iC,
                oC,
                dilation,
                groups,
                bias,
                has_relu,
                kernel_size=kernel_size,
            ).eval()
            mod = mod.to(memory_format=memory_format)
            # TODO: add bf16 test
            v = torch.randn(x_shape, dtype=torch.float32).to(
                memory_format=memory_format
            )
            with torch.no_grad():
                self.common(
                    mod,
                    (v,),
                )

    def test_linear_unary(self):
        options = itertools.product(unary_list, [[2, 3, 10], [2, 10]], [True, False])
        dtype = torch.bfloat16
        if has_bf16_support():
            for eltwise_fn, input_shape, bias in options:
                mod = torch.nn.Sequential(
                    torch.nn.Linear(input_shape[-1], 30, bias=bias), eltwise_fn
                ).eval()

                # only fuse for linear when the dtype is bf16
                mod = mod.to(dtype)
                v = torch.randn(input_shape).to(dtype)
                with torch.no_grad():
                    self.common(
                        mod,
                        (v,),
                    )

    def test_linear_binary(self):
        class M(torch.nn.Module):
            def __init__(self, eltwise_fn, in_channels, out_channels, bias, **kwargs):
                super(M, self).__init__()
                self.linear = torch.nn.Linear(
                    in_channels, out_channels, bias=bias, **kwargs
                )
                self.eltwise = eltwise_fn

            def forward(self, x, y):
                x = self.linear(x)
                x = self.eltwise(x, y)
                return x

        options = itertools.product(binary_list, [[2, 3, 10], [2, 10]], [True, False])
        dtype = torch.bfloat16
        out_feature = 30
        if has_bf16_support():
            for binary_ops, input_shape, bias in options:
                mod = M(binary_ops, input_shape[-1], out_feature, bias).eval()

                # only fuse for linear when the dtype is bf16
                mod = mod.to(dtype)
                v = torch.randn(input_shape).to(dtype)
                other = torch.randn(input_shape[:-1] + [out_feature]).to(dtype)
                with torch.no_grad():
                    self.common(mod, (v, other), atol=2e-3, rtol=0.016)

    def test_gather1(self):
        def fn(a, b):
            return (
                torch.gather(a.expand([4, 5, 10, 6]), 3, b + 1),
                torch.gather(a.expand([4, 5, 10, 6]), -1, b + 1),
            )

        self.common(
            fn,
            (
                torch.randn([1, 1, 10, 6]),
                torch.randint(5, [4, 5, 10, 1], dtype=torch.int64),
            ),
        )

    def test_gather2(self):
        # 0d tensor
        def fn(a, b):
            return torch.gather(a, 0, b) + torch.gather(a, -1, b)

        x = torch.tensor(123)
        y = torch.tensor(0)
        self.assertEqual(fn(x, y), x + x)

    def test_slice1(self):
        def fn(a):
            return (
                a[:, :10, 0] + a[:, 10:, 0],
                (a + 1)[:, :10, 0] + (a + 1)[:, 10:, 0],
            )

        self.common(
            fn,
            (torch.randn([2, 20, 2]),),
        )

    def test_slice2(self):
        def fn(a):
            return (
                a[:-1, ::2, -1] + a[-1:, 1::2, -2],
                (a + 1)[:-1, ::2, -1] + (a + 2)[-1:, 1::2, -2],
            )

        self.common(
            fn,
            (torch.randn([2, 20, 2]),),
        )

    def test_split_with_sizes(self):
        def fn(a, sizes):
            return [t + 1.0 for t in torch.split(a * 2.0, sizes, -1)]

        self.common(fn, (torch.randn(2, 2, 10), [3, 3, 4]))
        self.common(fn, (torch.randn(2, 2, 10), [4, 3, 3]))
        self.common(fn, (torch.randn(2, 2, 10), [1, 2, 3, 4]))

    def test_split(self):
        def fn(a):
            t = torch.split(a, 3, -1)
            return (t[0], t[1], t[2], t[3])

        def fn2(a):
            return fn(a + 1)

        self.common(
            fn,
            (torch.randn([2, 2, 10]),),
        )

        self.common(
            fn2,
            (torch.randn([2, 2, 10]),),
        )

    def test_to_dtype(self):
        def fn(a, b):
            return (
                aten._to_copy(a, dtype=6),
                aten._to_copy(b + 1, dtype=6),
                aten.to(b, torch.float64),
                aten.to(b, torch.bool),
            )

        self.common(
            fn,
            (
                torch.randn([2, 2, 10]),
                torch.randn([2, 2, 10], dtype=torch.float64),
            ),
        )

    @requires_cuda()
    def test_to_device(self):
        def fn(a):
            if a.device.type == "cpu":
                return aten._to_copy(a, device=torch.device("cuda"), dtype=6, layout=0)
            else:
                return aten._to_copy(a, device=torch.device("cpu"), dtype=6, layout=0)

        self.common(
            fn,
            (torch.randn([2, 2, 10]),),
        )

    @requires_cuda()
    def test_to_device_constant(self):
        def fn(a):
            d1 = a.device.type
            if d1 == "cpu":
                d2 = "cuda"
            else:
                d2 = "cpu"

            const1 = torch.as_tensor(list(range(64)), device=d2)
            return (
                torch.arange(10, device=d2).to(d1) + a,
                const1.to(d1),
                (const1 + 1).to(d1),
            )

        self.common(
            fn,
            (torch.randn([10]),),
        )

    @requires_cuda()
    def test_multi_device(self):
        def fn(x):
            x = x + 1
            x = x + 2
            x = x.cuda()
            x = x + 3
            x = x + 4
            x = x.cpu()
            x = x + 5
            x = x + 6
            x = x.cuda()
            x = x + 7
            x = x + 8
            x = x.cpu()
            x = x + 9
            x = x + 10
            return x

        self.common(
            fn,
            (torch.randn([2, 2, 10]),),
            check_lowp=False,  # cpu doesn't understand fp16, and there are explicit .cpu() calls
        )

    def test_unbind(self):
        def fn(a):
            return torch.unbind(a), torch.unbind(a, -1)

        self.common(
            fn,
            (torch.randn([4, 4, 4]),),
        )

    def test_convolution1(self):
        m = torch.nn.Sequential(
            torch.nn.Conv2d(5, 6, [3, 3]),
            torch.nn.ReLU(),
            ToTuple(),
        )

        self.common(
            m,
            (torch.randn([2, 5, 16, 16]),),
            # Mismatched elements: 10 / 2352 (0.4%)
            # Greatest absolute difference: 5.7220458984375e-05 at index (0, 3, 12, 12) (up to 1e-05 allowed)
            # Greatest relative difference: 0.06512477175897748 at index (0, 4, 11, 9) (up to 0.001 allowed)
            atol=6e-5,
            rtol=0.001,
        )

    def test_convolution2(self):
        def fn(x, w, b):
            # transposed conv
            return (aten.convolution(x, w, b, [4], [0], [1], True, [0], 1),)

        self.common(
            fn,
            (
                torch.randn([2, 32, 90]),
                torch.randn([32, 16, 8]),
                torch.randn([16]),
            ),
            check_lowp=False,
        )

    @unittest.skipIf(HAS_CUDA, "only support cpu channels_last")
    def test_conv2d_channels_last(self):
        m = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, 1, 1),
            ToTuple(),
        )
        # only weight is channels_last
        self.common(
            m.to(memory_format=torch.channels_last),
            (torch.randn([2, 3, 16, 16]),),
            check_lowp=False,
        )
        # only activation is channels_last
        self.common(
            m,
            (torch.randn([2, 3, 16, 16]).to(memory_format=torch.channels_last),),
            check_lowp=False,
        )
        # activation and weight are all channels_last
        self.common(
            m.to(memory_format=torch.channels_last),
            (torch.randn([2, 3, 16, 16]).to(memory_format=torch.channels_last),),
            check_lowp=False,
        )

    def test_conv2d_backward_channels_last(self):
        def fn(grad_output, inp, weight):
            convolution_backward_8 = torch.ops.aten.convolution_backward.default(
                grad_output,
                inp,
                weight,
                [320],
                [1, 1],
                [0, 0],
                [1, 1],
                False,
                [0, 0],
                1,
                [True, True, True],
            )
            return convolution_backward_8

        # only weight is channels_last
        self.common(
            fn,
            (
                torch.randn([2, 320, 8, 8]),
                torch.randn([2, 2048, 8, 8]),
                torch.randn([320, 2048, 1, 1]).to(memory_format=torch.channels_last),
            ),
            check_lowp=False,
        )

    @unittest.skipIf(HAS_CUDA, "only support cpu channels_last")
    def test_conv3d_channels_last(self):
        m = torch.nn.Sequential(
            torch.nn.Conv3d(3, 3, 1, 1),
            ToTuple(),
        )
        # only weight is channels_last
        self.common(
            m.to(memory_format=torch.channels_last_3d),
            (torch.randn([2, 3, 16, 16, 16]),),
        )
        # only activation is channels_last
        self.common(
            m,
            (torch.randn([2, 3, 16, 16, 16]).to(memory_format=torch.channels_last_3d),),
        )
        # activation and weight are all channels_last
        self.common(
            m.to(memory_format=torch.channels_last_3d),
            (torch.randn([2, 3, 16, 16, 16]).to(memory_format=torch.channels_last_3d),),
        )

    def test_adaptive_avg_pool2d1(self):
        def fn(x):
            return aten._adaptive_avg_pool2d(x, (6, 6)), aten._adaptive_avg_pool2d(
                x + 1, (2, 5)
            )

        self.common(
            fn,
            (torch.randn(2, 4, 16, 16),),
        )

        # lowering to avg_pool2d case
        self.common(
            fn,
            (torch.randn(2, 4, 3, 3),),
        )

        # no-op case
        self.common(
            fn,
            (torch.randn(2, 4, 6, 6),),
        )

    def test_max_pool2d1(self):
        def fn(x):
            return aten.max_pool2d_with_indices(x, [3, 3], [2, 2])

        self.common(
            fn,
            (torch.randn(2, 4, 16, 16),),
        )

    def test_max_pool2d2(self):
        def fn(x):
            return aten.max_pool2d_with_indices(x, [3, 3], [2, 2])

        self.common(
            fn,
            (torch.randn([16, 64, 55, 55]),),
        )

    def test_max_pool2d3(self):
        def fn(x):
            # with padding
            return aten.max_pool2d_with_indices(x, [3, 3], [2, 2], [1, 1])

        self.common(
            fn,
            (-torch.arange(1 * 8 * 8, dtype=torch.float32).view(1, 1, 8, 8),),
        )

    def test_max_pool2d4(self):
        def fn(x):
            # with padding
            return aten.max_pool2d_with_indices(x, [3, 3], [2, 2], [0, 0], [1, 1], True)

        self.common(
            fn,
            (torch.randn([2, 8, 111, 111]),),
        )

    def test_max_pool2d5(self):
        def fn(x):
            return aten.max_pool2d_with_indices(x, [3, 3], [])

        self.common(
            fn,
            (torch.randn([16, 64, 55, 55]),),
        )

    def test_avg_pool2d1(self):
        def fn(x):
            return aten.avg_pool2d(x, [3, 3], [2, 2])

        self.common(
            fn,
            (torch.randn(2, 4, 16, 16),),
        )

    def test_avg_pool2d2(self):
        def fn(x):
            return aten.avg_pool2d(x, [3, 3], [2, 2])

        self.common(
            fn,
            (torch.randn([16, 64, 55, 55]),),
        )

    def test_avg_pool2d3(self):
        def fn(x):
            return aten.avg_pool2d(x, [3, 3], [2, 2], [1, 1])

        self.common(
            fn,
            (-torch.arange(1 * 8 * 8, dtype=torch.float32).view(1, 1, 8, 8),),
        )

    def test_avg_pool2d4(self):
        def fn(x):
            return aten.avg_pool2d(x, [3, 3], [2, 2], [0, 0], True)

        self.common(
            fn,
            (torch.randn([2, 8, 111, 111]),),
        )

    def test_avg_pool2d5(self):
        def fn(x):
            return aten.avg_pool2d(x, [3, 3], [2, 2], [1, 1], count_include_pad=False)

        self.common(
            fn,
            (-torch.arange(1 * 8 * 8, dtype=torch.float32).view(1, 1, 8, 8),),
        )

    def test_avg_pool2d6(self):
        def fn(x):
            return aten.avg_pool2d(x, [3, 3], [2, 2], [1, 1], divisor_override=3)

        self.common(
            fn,
            (-torch.arange(1 * 8 * 8, dtype=torch.float32).view(1, 1, 8, 8),),
        )

    def test_alexnet_prefix(self):
        def forward(arg6, arg7, arg16):
            convolution = torch.ops.aten.convolution(
                arg16, arg7, arg6, [4, 4], [2, 2], [1, 1], False, [0, 0], 1
            )
            relu = torch.ops.aten.relu(convolution)
            max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices(
                relu, [3, 3], [2, 2]
            )
            getitem = max_pool2d_with_indices[0]
            return (getitem,)

        self.common(
            forward,
            (
                rand_strided((64,), (1,), torch.float32, "cpu"),
                rand_strided((64, 3, 11, 11), (363, 121, 11, 1), torch.float32, "cpu"),
                rand_strided(
                    (16, 3, 224, 224), (150528, 50176, 224, 1), torch.float32, "cpu"
                ),
            ),
            # Mismatched elements: 127 / 746496 (0.0%)
            # Greatest absolute difference: 0.0009765625 at index (1, 62, 7, 16) (up to 1e-05 allowed)
            # Greatest relative difference: 0.05187467899332306 at index (14, 18, 11, 0) (up to 0.001 allowed)
            atol=1e-3,
            rtol=0.001,
        )

    def test_elu(self):
        def fn(x):
            return aten.elu(x, 1.6732632423543772, 1.0507009873554805) + 2, aten.elu(
                x + 1, 2, 3, 4
            )

        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    def test_tanh(self):
        def fn(x):
            return aten.tanh(x) + 2, aten.tanh(x + 1)

        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    def test_lgamma(self):
        def fn(x):
            return aten.lgamma(x) + 2, aten.cos(x + 1)

        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    def test_cos(self):
        def fn(x):
            return aten.cos(x) + 2, aten.cos(x + 1)

        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    def test_sin(self):
        def fn(x):
            return aten.sin(x) + 2, aten.sin(x + 1)

        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    def test_repeat(self):
        def fn(x):
            return (
                x.repeat(2, 2, 3, 1),
                x.repeat(8, 1, 1, 1),
                x.repeat(2, 1, 1, 1, 1, 1),
            )

        self.common(
            fn,
            (torch.randn([1, 2, 4, 8]),),
        )

    def test_embedding(self):
        m = torch.nn.Sequential(
            torch.nn.Embedding(10, 4, padding_idx=0),
            torch.nn.ReLU(),
            ToTuple(),
        )

        self.common(
            m,
            (torch.randint(10, [2, 8]),),
        )

    def test_mean(self):
        def fn(x):
            return (
                x.mean(),
                x.mean(-1),
                torch.mean(x, -2, keepdim=True),
                x.mean([0, 1]),
            )

        self.common(
            fn,
            (torch.randn([1, 2, 4, 8]),),
        )

    def test_var_mean(self):
        def fn(x):
            return (
                *torch.var_mean(x, -1),
                *torch.var_mean(x, [1, 3]),
            )

        self.common(
            fn,
            (torch.randn([1, 2, 4, 8]),),
        )

    @patch.object(config, "pick_loop_orders", True)
    def test_transposed_propagates(self):
        @torch._dynamo.optimize("inductor", nopython=True)
        def fn(x, y):
            return x + y

        a = torch.randn(1, 4, 4, 4, device=self.device).permute(0, 2, 3, 1)
        b = torch.randn(4, 4, 4, device=self.device).permute(1, 2, 0)
        c = fn(a, b)
        self.assertEqual(a.stride(), c.stride())
        self.assertEqual(c.stride()[2], 1)

    @requires_cuda()
    @patch.object(config.triton, "convolution", "triton")
    @patch.object(config.triton, "dense_indexing", "True")
    def test_triton_conv(self):
        @torch._dynamo.optimize("inductor", nopython=True)
        def triton_conv(
            x,
            w,
            bias,
            stride,
            padding,
            dilation,
            groups,
        ):
            y = torch.conv2d(x, w, bias, stride, padding, dilation, groups)
            return y

        stride, padding, dilation, groups = (1, 1), (0, 0), (1, 1), 1
        dtype = torch.float32
        x = torch.randn((32, 128, 32, 32), dtype=dtype, device=self.device)
        w = torch.randn((32, 128, 1, 1), dtype=dtype, device=self.device)
        bias = torch.randn((32), dtype=dtype, device=self.device)

        y = triton_conv(x, w, bias, stride, padding, dilation, groups)
        y_correct = torch.conv2d(x, w, bias, stride, padding, dilation, groups)
        self.assertTrue(same(y, y_correct, cos_similarity=True, tol=0.1))

    @requires_cuda()
    @patch.object(config.triton, "convolution", "autotune")
    @patch.object(config.triton, "dense_indexing", "True")
    def test_conv_autotune(self):
        @torch._dynamo.optimize("inductor", nopython=True)
        def triton_conv(
            x,
            w,
            bias,
            stride,
            padding,
            dilation,
            groups,
        ):
            y = torch.conv2d(x, w, bias, stride, padding, dilation, groups)
            return y

        stride, padding, dilation, groups = (1, 1), (0, 0), (1, 1), 1
        dtype = torch.float32
        x = torch.randn((32, 128, 32, 32), dtype=dtype, device=self.device)
        w = torch.randn((32, 128, 1, 1), dtype=dtype, device=self.device)
        bias = torch.randn((32), dtype=dtype, device=self.device)

        y = triton_conv(x, w, bias, stride, padding, dilation, groups)
        y_correct = torch.conv2d(x, w, bias, stride, padding, dilation, groups)
        self.assertTrue(same(y, y_correct, cos_similarity=True, tol=0.1))

    @patch.object(config.triton, "mm", "triton")
    def test_triton_mm2(self):
        @torch._dynamo.optimize("inductor", nopython=True)
        def fn(x, y):
            return torch.relu(torch.mm(x, y))

        N = 1024
        a = torch.randn([N, N], device=self.device, dtype=torch.float32)
        b = torch.randn([N, N], device=self.device, dtype=torch.float32)
        c1 = torch.relu(torch.mm(a, b))
        torch._inductor.metrics.reset()
        c = fn(a, b)
        assert torch.allclose(c1, c, atol=1e-3, rtol=1e-3)
        if self.device == "cuda":
            assert torch._inductor.metrics.generated_kernel_count == 1

    def test_std(self):
        def fn(x):
            return (
                torch.var(x, True),
                torch.var(x, False),
                torch.var(x, -1, True),
                torch.var(x, -1, False),
                torch.std(x, False),
                torch.std(x, [0, 1], True),
                torch.std(x, [0, 1], False),
                torch.std(x, -2, True, keepdim=True),
            )

        self.common(
            fn,
            (torch.randn([2, 4, 4, 8]),),
        )

    def test_embedding_bag(self):
        def fn(w, i, o):
            return aten._embedding_bag(w, i, o, False, 0, False, None)

        self.common(
            fn,
            (torch.randn([10, 4]), torch.randint(10, [8]), torch.tensor([0, 2, 6])),
        )

    def test_batch_norm_2d(self):
        m = torch.nn.Sequential(
            torch.nn.BatchNorm2d(10),
            torch.nn.ReLU(),
        )
        m.eval()
        self.common(m, (torch.randn([2, 10, 8, 8]),), check_lowp=False)
        self.common(
            m,
            (torch.randn([3, 10, 16, 16]),),
            check_lowp=False,  # too painful to match types of bn model
        )

    def test_layer_norm(self):
        m = torch.nn.Sequential(
            torch.nn.LayerNorm(32),
            torch.nn.ReLU(),
        )
        m.eval()
        self.common(m, (torch.randn([16, 32]),), check_lowp=False)
        if self.device != "cpu":
            self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    def test_transpose_add(self):
        def fn(a, b):
            return a.t() + b

        self.common(
            fn, (torch.randn([16, 32]), torch.randn([32, 16])), check_lowp=False
        )
        if self.device != "cpu":
            self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    def test_softmax_one_kernel(self):
        def fn(x):
            dim = 1
            x_max = torch.amax(x, dim, keepdim=True)
            unnormalized = torch.exp(x * x_max)
            result = unnormalized / torch.sum(unnormalized, dim, keepdim=True)
            return result

        self.common(fn, (torch.randn([16, 32]),), check_lowp=False)
        if self.device != "cpu":
            self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    def test_cauchy(self):
        def fn(x, y):
            return torch.sum(1 / (torch.unsqueeze(x, -1) - y))

        self.common(
            fn,
            (
                torch.randn(32),
                torch.randn(32),
            ),
            # Absolute difference: 0.0003662109375 (up to 0.0001 allowed)
            # Relative difference: 1.8804297408767818e-05 (up to 1e-05 allowed)
            atol=5 * 1e-4,
            rtol=5 * 1e-5,
            check_lowp=False,
        )
        if self.device != "cpu":
            self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    def test_gather_scatter(self):
        def fn(node_feat, edge_index):
            src_node_feat = node_feat[edge_index[0]]
            dst_node_feat = node_feat[edge_index[1]]
            edge_feat = src_node_feat - dst_node_feat + 1
            new_node_feat = torch.zeros_like(node_feat)
            new_node_feat.scatter_add_(
                0, edge_index[1].unsqueeze(-1).expand_as(edge_feat), edge_feat
            )
            return new_node_feat

        num_nodes = 16
        num_features = 32
        node_feat = torch.randn(num_nodes, num_features)
        edge_index = torch.randint(0, num_nodes, size=(2, num_nodes * 5))
        self.common(
            fn,
            (
                node_feat,
                edge_index,
            ),
            check_lowp=False,
        )
        if self.device != "cpu":
            self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

    @patch.object(torch._inductor.config, "max_fusion_size", 1)
    def test_no_mega_fusion_during_lowering(self):
        n = 50

        def fn(*args):
            x = args[0]
            for i in range(n):
                x = torch.add(x, args[i])
            return x

        self.common(
            fn,
            [torch.randn(64) for _ in range(n)],
            check_lowp=False,
        )
        print("-->", torch._inductor.metrics.generated_kernel_count)
        if self.device != "cpu":
            self.assertTrue(torch._inductor.metrics.generated_kernel_count > 1)

    def test_move_arange(self):
        def fn(x):
            return torch.arange(len(x), device="cpu").to(x.device) + x

        self.common(fn, (torch.randn([32]),), check_lowp=False)
        # if we have a copy there will be more than 1 kernel
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    def test_leaky_relu(self):
        def fn(x):
            return aten.leaky_relu(x, 0.2) + 2, aten.leaky_relu(x + 1)

        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    def test_gelu(self):
        def fn(x):
            return aten.gelu(x) + 2, aten.gelu(x + 1)

        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    def test_clone(self):
        def fn(x):
            return aten.clone(x) + 2, aten.clone(x + 1)

        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    def test_masked_fill(self):
        def fn(mask, value):
            return aten.masked_fill(value, mask, -10000.0) + 2, aten.masked_fill(
                value / 2.0, torch.logical_not(mask), 667
            )

        self.common(
            fn,
            (
                torch.randint(0, 1, [1, 16], dtype=torch.bool),
                torch.randn([16, 16]),
            ),
        )

    def test_masked_fill_promotion(self):
        def fn(mask, value):
            return aten.masked_fill(value, mask, torch.tensor(3.5))

        opt_fn = torch._dynamo.optimize("inductor")(fn)
        for inp in (
            torch.randn(
                [16, 16],
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device=self.device,
            ),
            torch.randint(16, (16, 16), device=self.device),
        ):

            inputs = (
                torch.randint(0, 1, [1, 16], dtype=torch.bool, device=self.device),
                inp,
            )
            self.assertEqual(fn(*inputs), opt_fn(*inputs))

    def test_fill1(self):
        def fn(x):
            tmp = torch.ones_like(x)
            return tmp, aten.fill.Scalar(tmp, 2)

        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    def test_fill2(self):
        def fn(x):
            tmp = torch.ones_like(x)
            return tmp, aten.fill.Tensor(tmp, torch.tensor(3.0))

        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    def test_pow1(self):
        def fn(x):
            return [aten.pow(x, e) for e in range(-8, 9)]

        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    def test_pow2(self):
        def fn(x):
            return aten.pow(1000, x), aten.pow(x, 1000)

        self.common(
            fn,
            (torch.randn([16, 16]),),
            # Mismatched elements: 9 / 256 (3.5%)
            # Greatest absolute difference: 2.491354329061828e+28 at index (6, 6) (up to 1e-05 allowed)
            # Greatest relative difference: 2.9793410720160818e-05 at index (4, 5) (up to 1.3e-06 allowed)
            atol=1e-5,
            rtol=3e-05,
        )

    def test_pow3(self):
        # power of 0.5 is special-cased, arbitrary power would still produce triton codegen error
        def fn(x):
            z = torch.tensor(0.123, device=self.device)
            w = z + x
            return torch.pow(w, 0.5)

        opt = torch._dynamo.optimize("inductor")(fn)
        input = torch.rand(())
        self.assertTrue(same(opt(input), fn(input)))

    def test_glu(self):
        def fn(x):
            return aten.glu(x, -1), aten.glu(x, 1), aten.glu(x, 2)

        self.common(
            fn,
            (torch.randn([8, 16, 8, 8]),),
        )

    def test_cat(self):
        def fn(a):
            tmp = a * 2
            return (
                torch.cat((a, a[:, :4] + 1, a + 2), -1),
                torch.cat((tmp, tmp), 0),
                torch.cat((tmp, tmp.double()), 0),
            )

        self.common(
            fn,
            (torch.randn([8, 16]),),
        )

    def test_cat_upcasting(self):
        def fn(arg4_1, slice_7):
            cat_1 = aten.cat.default([arg4_1, slice_7], 1)
            return (cat_1,)

        self.common(
            fn,
            (
                torch.randn([8, 16], dtype=torch.float32),
                torch.randn([8, 20], dtype=torch.float16),
            ),
        )

    def test_cat_extern_kernel(self):
        def fn(x1, x2, x3, x4):
            x = torch.mm(x2, x3)
            s = torch.narrow(x, 1, 0, 100)
            x = torch.mm(s, x4)
            c = torch.cat((x, x1), 1)
            return (c,)

        self.common(
            fn,
            (
                torch.randn(256, 256),
                torch.randn(256, 1024),
                torch.randn(1024, 1600),
                torch.randn(100, 256),
            ),
            check_lowp=False,  # accuracy issues with relatively large matmuls
        )

    def test_stack(self):
        def fn(a, b):
            return torch.stack(
                [
                    a.expand(12, 16),
                    b.expand(12, 16),
                ],
                2,
            )

        self.common(fn, (torch.randn([1, 16]), torch.randn([12, 1])))

    def test_hardtanh(self):
        def fn(x):
            return F.hardtanh(x), F.hardtanh(x + 1), F.hardtanh(x - 1)

        self.common(
            fn,
            (torch.randn([64]),),
        )

    def test_hardsigmoid(self):
        def fn(x):
            return F.hardsigmoid(x), F.hardsigmoid(x + 3), F.hardsigmoid(x - 3)

        self.common(
            fn,
            (torch.randn([64]),),
        )

    def test_hardswish(self):
        def fn(x):
            return F.hardswish(x), F.hardswish(x + 3), F.hardswish(x - 3)

        self.common(
            fn,
            (torch.randn([64]),),
        )

    def test_rsqrt(self):
        def fn(x):
            return torch.rsqrt(x), torch.rsqrt(x + 1) - 2

        self.common(
            fn,
            (torch.randn([64]),),
        )

    def test_flip(self):
        def fn(x):
            return torch.flip(x, (-1,)), torch.flip(x, (0, 2)) - 2

        self.common(
            fn,
            (torch.randn([1, 2, 6, 6]),),
        )

    def test_signbit(self):
        def fn(x):
            return torch.signbit(x), ~torch.signbit(-x) & 1

        self.common(
            fn,
            (torch.randn([1, 2, 6, 6]),),
        )

    def test_fmod(self):
        def fn(a, b):
            return torch.fmod(a, b), torch.fmod(3.0 * a, b) - 2.0

        shape = [1, 2, 6, 6]
        self.common(fn, (torch.randn(shape), torch.randn(shape)))

    def test_log2(self):
        def fn(x):
            return torch.log2(x), torch.log2(x + 1) - 2

        self.common(
            fn,
            (torch.randn([64]) + 10,),
        )

    def test_logsumexp(self):
        def fn(x):
            return torch.logsumexp(x, -1), torch.logsumexp(x, 0) - 2

        self.common(
            fn,
            (torch.randn([8, 8]) + 10,),
        )

    def test_log_fp64(self):
        def fn(x):
            return torch.log(x), torch.log2(x)

        self.common(
            fn,
            (torch.randn([1024], dtype=torch.float64) + 10,),
        )

    def test_bitwise(self):
        def fn(x, y):
            return (
                torch.bitwise_not(x),
                torch.bitwise_or(x, y),
                torch.bitwise_xor(x, y),
                torch.bitwise_and(x, y),
            )

        self.common(
            fn,
            (
                torch.randint(0, 2**30, [64], dtype=torch.int32),
                torch.randint(0, 2**30, [64], dtype=torch.int32),
            ),
        )

    def test_bitwise2(self):
        # again with bool types
        def fn(x, y):
            return (
                torch.bitwise_not(x),
                torch.bitwise_or(x, y),
                torch.bitwise_xor(x, y),
                torch.bitwise_and(x, y),
            )

        self.common(
            fn,
            (
                torch.randint(0, 2, (2, 20), dtype=torch.bool),
                torch.randint(0, 2, (2, 20), dtype=torch.bool),
            ),
        )

    def test_inf(self):
        def fn(a):
            return a + float("inf"), a + float("-inf"), a * -float("inf")

        self.common(fn, (torch.randn(8),))

    def test_remainder(self):
        def fn(a, b):
            return (
                torch.remainder(a, b),
                torch.remainder(a + 1, b - 1),
                torch.remainder(a - 1, b + 1),
            )

        self.common(fn, (torch.randn(64), torch.randn(64)))

    def test_zeros(self):
        def fn(a):
            return (
                a + 1,
                torch.zeros(
                    (1, 8, 64, 64),
                    dtype=torch.float32,
                    device=a.device,
                ),
                torch.zeros(
                    1,
                    8,
                    64,
                    64,
                    dtype=torch.float32,
                    device=a.device,
                ),
                torch.zeros(2, 3, names=None),
                a + torch.ones(8, device=a.device),
                torch.full((2, 3), 3.1416, device=a.device),
            )

        self.common(fn, (torch.randn(8),))

    def test_new_ones(self):
        def fn(a):
            return (
                aten.new_ones(
                    a, [], device=a.device, dtype=6, layout=0, pin_memory=False
                ),
                aten.new_zeros(
                    a, [], device=a.device, dtype=6, layout=0, pin_memory=False
                ),
            )

        self.common(fn, (torch.randn(8),))

    def test_full_like(self):
        def fn(a):
            return torch.full_like(a, 7.777) - 1

        self.common(fn, (torch.randn(8),))

    def test_index1(self):
        def fn(a, b, c):
            return aten.index(a, [b, c])

        self.common(
            fn,
            (
                torch.randn(8, 8, 12),
                torch.tensor([0, 0, 2, 2], dtype=torch.int64),
                torch.tensor([3, 4, 4, 3], dtype=torch.int64),
            ),
        )
        self.common(
            fn,
            (
                torch.randn(8, 8, 12),
                torch.tensor([[0, 0, 2, 2]], dtype=torch.int64),
                torch.tensor([[3], [4], [4], [3]], dtype=torch.int64),
            ),
        )

    def test_index2(self):
        def fn(a, b):
            return (
                aten.index(a, [b]),
                aten.index(a, [None, b]),
            )

        self.common(
            fn,
            (
                torch.randn(8, 8, 8),
                torch.tensor([[0, 0, 2, 2]], dtype=torch.int64),
            ),
        )

    def test_index_select(self):
        def fn(a, b):
            return (
                torch.index_select(a, 0, b),
                torch.index_select(a, 1, b),
                torch.index_select(torch.index_select(a, 2, b), 1, b),
            )

        for ind_dtype in (torch.int32, torch.int64):
            self.common(
                fn,
                (
                    torch.randn(8, 8, 8),
                    torch.tensor([0, 0, 2, 1], dtype=ind_dtype),
                ),
            )

    # https://github.com/pytorch/torchdynamo/issues/467
    @patch.object(torch._dynamo.config, "fake_tensor_propagation", False)
    def test_cudnn_rnn(self):
        if self.device == "cpu":
            raise unittest.SkipTest("requires CUDA")

        def fn(
            a0,
            b0,
            b1,
            b2,
            b3,
            b4,
            b5,
            b6,
            b7,
            b8,
            b9,
            b10,
            b11,
            b12,
            b13,
            b14,
            b15,
            a3,
            a4,
            a5,
        ):
            a1 = [
                b0,
                b1,
                b2,
                b3,
                b4,
                b5,
                b6,
                b7,
                b8,
                b9,
                b10,
                b11,
                b12,
                b13,
                b14,
                b15,
            ]
            return aten._cudnn_rnn(
                a0,
                a1,
                4,
                a3,
                a4,
                a5,
                2,
                2048,
                0,
                2,
                False,
                0.0,
                False,
                True,
                [],
                None,
            )

        self.common(
            fn,
            (
                torch.randn([92, 8, 2048]),
                torch.randn([8192, 2048]),
                torch.randn([8192, 2048]),
                torch.randn([8192]),
                torch.randn([8192]),
                torch.randn([8192, 2048]),
                torch.randn([8192, 2048]),
                torch.randn([8192]),
                torch.randn([8192]),
                torch.randn([8192, 4096]),
                torch.randn([8192, 2048]),
                torch.randn([8192]),
                torch.randn([8192]),
                torch.randn([8192, 4096]),
                torch.randn([8192, 2048]),
                torch.randn([8192]),
                torch.randn([8192]),
                torch.randn([167837696]),
                torch.randn([4, 8, 2048]),
                torch.randn([4, 8, 2048]),
            ),
            check_lowp=False,  # difference in rnn is too large between half and float inputs
        )

    def test_upsample_nearest1d(self):
        def fn(a):
            return (
                aten.upsample_nearest1d(a, [74], None),
                aten.upsample_nearest1d(a, [70], None),
                aten.upsample_nearest1d(a, [45], None),
                aten.upsample_nearest1d(a, [36], None),
                aten.upsample_nearest1d(a, None, [2.0]),
            )

        self.common(fn, (torch.randn([2, 4, 37]),))

    def test_upsample_nearest2d(self):
        def fn(a):
            return (
                aten.upsample_nearest2d(a, [74, 76]),
                aten.upsample_nearest2d(a, [70, 75]),
                aten.upsample_nearest2d(a, [45, 74]),
                aten.upsample_nearest2d(a, [36, 39]),
                aten.upsample_nearest2d(a, None, [2.0, 2.0]),
            )

        self.common(fn, (torch.randn([2, 4, 37, 38]),))

    def test_upsample_nearest3d(self):
        def fn(a):
            return (
                aten.upsample_nearest3d(a, [74, 76, 78], None),
                aten.upsample_nearest3d(a, [70, 75, 80], None),
                aten.upsample_nearest3d(a, [45, 74, 103], None),
                aten.upsample_nearest3d(a, [36, 39, 40], None),
                aten.upsample_nearest3d(a, None, [2.0, 2.0, 2.0]),
            )

        self.common(fn, (torch.randn([2, 4, 37, 38, 39]),))

    def test_upsample_nearest2d_backward(self):
        func = torch.ops.aten.upsample_nearest2d_backward

        def fn(a):
            return (
                func(a, output_size=[6, 12], input_size=[3, 3, 3, 6]),
                func(a, output_size=[6, 12], input_size=[3, 3, 4, 5]),
                func(a, output_size=[6, 12], input_size=[3, 3, 2, 8]),
                func(a, output_size=[6, 12], input_size=[3, 3, 2, 8]),
                func(a, output_size=[6, 12], input_size=[3, 3, 4, 7]),
            )

        self.common(fn, (torch.randn([3, 3, 6, 12]),))

    def test_upsample_bilinear2d_a(self):
        def fn(a):
            return (
                aten.upsample_bilinear2d(a, [45, 45], False, None),
                aten.upsample_bilinear2d(a, None, True, [2.0, 2.0]),
            )

        self.common(fn, (torch.randn([2, 4, 37, 38]),))

    def test_upsample_bilinear2d_b(self):
        def fn(a):
            return aten.upsample_bilinear2d(a, None, True, [2.0, 2.0])

        self.common(
            fn,
            [
                torch.randn([1, 2, 40, 59]),
            ],
        )

    def test_reflection_pad2d(self):
        def fn(a):
            return (
                aten.reflection_pad2d(a, [1, 1, 1, 1]),
                aten.reflection_pad2d(a, [1, 2, 3, 4]),
            )

        self.common(
            fn, (torch.randint(0, 999, size=[1, 1, 8, 8], dtype=torch.float32),)
        )

    def test_reflection_pad2d_backward(self):
        def template(size, padding):
            def fn(grad_output, x):
                return aten.reflection_pad2d_backward(grad_output, x, padding)

            x = torch.randint(0, 999, size=size, dtype=torch.float32)
            result = aten.reflection_pad2d(x, padding)
            grad_output = torch.randn_like(result)

            self.common(fn, (grad_output, x))

        template([1, 1, 8, 8], [0, 0, 0, 0])
        template([1, 1, 8, 8], [1, 1, 1, 1])
        template([1, 1, 8, 8], [1, 2, 3, 4])

    def test_grid_sampler_2d(self):
        def fn(a, b):
            return (
                aten.grid_sampler_2d(a, b, 0, 0, True),
                aten.grid_sampler_2d(a, b, 0, 1, False),
            )

        self.common(
            fn,
            (
                torch.randn([4, 3, 352, 352], dtype=torch.float32),
                torch.rand([4, 352, 352, 2], dtype=torch.float32) * 2 - 1,
            ),
            check_lowp=False,
            # Mismatched elements: 154697 / 1486848 (10.4%)
            # Greatest absolute difference: 0.0001976490020751953 at index (0, 0, 101, 243) (up to 1e-05 allowed)
            # Greatest relative difference: 7.332530120481928 at index (1, 1, 258, 301) (up to 1.3e-06 allowed)
            atol=0.0002,
            rtol=1.3e-06,
        )

    def test_upsample_bicubic2d(self):
        def fn(a):
            return (
                aten.upsample_bicubic2d(a, (128, 128), True),
                aten.upsample_bicubic2d(a, (128, 256), False),
            )

        # Mismatched elements: 10 / 196608 (0.0%)
        # Greatest absolute difference: 1.3869255781173706e-05 at index (2, 1, 88, 65) (up to 1e-05 allowed)
        # Greatest relative difference: 0.0033082996811011046 at index (3, 1, 88, 91) (up to 1.3e-06 allowed)
        self.common(
            fn,
            (torch.randn([4, 3, 64, 32], dtype=torch.float32),),
            atol=2e-5,
            rtol=1e-3,
        )

    def test_sort(self):
        def fn(a):
            return torch.sort(a)

        self.common(
            fn, (torch.randint(0, 999, size=[1, 1, 8, 8], dtype=torch.float32),)
        )

    def test_topk(self):
        def fn(a):
            return torch.topk(a, 2, -1)

        self.common(
            fn, (torch.randint(0, 999, size=[1, 1, 8, 8], dtype=torch.float32),)
        )

    def test_long_tensor(self):
        def fn(a):
            return (
                torch.LongTensor([294]).to(a.device) - a,
                torch.as_tensor([295]).to(a.device) + a,
            )

        self.common(fn, (torch.randint(0, 999, size=[8, 8]),))

    def test_constant_pad_1d(self):
        def fn(a):
            return (
                aten.constant_pad_nd(a, [0, 1], 6.0),
                aten.constant_pad_nd(a, [2, 3], 99.0),
            )

        self.common(fn, (torch.randint(0, 999, size=[2, 16, 31], dtype=torch.float32),))

    def test_constant_pad_2d(self):
        def fn(a):
            return (
                aten.constant_pad_nd(a, [1, 1, 1, 1], 6.0),
                aten.constant_pad_nd(a, [1, 2, 3, 4], 99.0),
            )

        self.common(
            fn, (torch.randint(0, 999, size=[1, 1, 8, 8], dtype=torch.float32),)
        )

    def test_constant_pad_3d(self):
        def fn(a):
            return (
                aten.constant_pad_nd(a, [1, 2, 3, 4, 5, 6], 6.0),
                aten.constant_pad_nd(a, [0, 0, 3, 4, 0, 0], 6.0),
            )

        self.common(
            fn, (torch.randint(0, 999, size=[2, 4, 4, 4], dtype=torch.float32),)
        )

    def test_l1_loss(self):
        def fn(a, b):
            return torch.nn.functional.l1_loss(a, b), torch.nn.functional.mse_loss(a, b)

        self.common(
            fn,
            (
                torch.randn([2, 3, 16, 16]),
                torch.randn([2, 3, 16, 16]),
            ),
            check_lowp=False,
        )

    def test_triu(self):
        def fn(a):
            return aten.triu(a, 1), aten.triu(a, 0), aten.triu(a, 2)

        self.common(fn, (torch.randn([2, 10, 10]),))

    def test_no_op_reduction(self):
        def fn(a):
            return a.sum(-1), torch.amax(a + 1, 1, keepdim=True)

        self.common(fn, (torch.randn([8, 1, 1]),))

    def test_inplace_add(self):
        @torch._dynamo.optimize("inductor")
        def fn(x, y):
            return x.add_(y)

        inputs = (
            rand_strided((4, 4), (4, 1), device=self.device),
            rand_strided((4, 4), (4, 1), device=self.device),
        )
        inp_clone = inputs[0].clone()
        out = fn(*inputs)
        self.assertTrue(same(out, inp_clone + inputs[1]))
        self.assertTrue(out is inputs[0])

    def test_inplace_mixed_dtype_ops(self):
        @torch._dynamo.optimize("inductor")
        def fn(x, y):
            z = x + y.float()
            w = z.add_(y)
            return w.mul_(y)

        inputs = (
            rand_strided((4, 4), (4, 1), device=self.device, dtype=torch.float),
            rand_strided((4, 4), (4, 1), device=self.device, dtype=torch.double),
        )
        out = fn(*inputs)
        out_eager = (inputs[0] + inputs[1].float()).add_(inputs[1]).mul_(inputs[1])
        self.assertTrue(same(out, out_eager))

    @patch.object(config.triton, "cudagraphs", True)
    def test_strided_inputs(self):
        @torch._dynamo.optimize("inductor")
        def fn(x, y):
            return x + y

        inputs = (
            rand_strided((8, 16), (32, 2), device=self.device),
            rand_strided((8, 16), (16, 1), device=self.device),
        )
        self.assertTrue(same(fn(*inputs), inputs[0] + inputs[1]))

    @patch.object(config.triton, "cudagraphs", True)
    @patch.object(functorch_config, "use_fake_tensor", True)
    def test_input_mutation1(self):
        def fn(a):
            b = a + 1
            a.copy_(b)
            c = a + 2
            return a * b / c

        arg1 = torch.randn(64, device=self.device)
        arg2 = arg1.clone()
        arg3 = torch.randn(64, device=self.device)
        arg4 = arg3.clone()
        correct1 = fn(arg1)
        correct2 = fn(arg3)
        opt_fn = torch._dynamo.optimize_assert(compile_fx)(fn)
        actual1 = opt_fn(arg2)
        actual2 = opt_fn(arg4)

        self.assertTrue(same(actual1, correct1))
        self.assertTrue(same(actual2, correct2))
        self.assertTrue(same(arg1, arg2))
        self.assertTrue(same(arg3, arg4))

    @patch.object(functorch_config, "use_fake_tensor", True)
    def test_input_mutation2(self):
        def fn(a):
            b = a + 1
            a.view(64).copy_(torch.tensor([66.0], device=a.device))
            c = a + 2
            return b, c

        arg1 = torch.randn([1, 64], device=self.device)
        arg2 = arg1.clone()
        correct1 = fn(arg1)
        opt_fn = torch._dynamo.optimize_assert(compile_fx)(fn)
        actual1 = opt_fn(arg2)

        self.assertTrue(same(actual1, correct1))
        self.assertTrue(same(arg1, arg2))

    @patch.object(functorch_config, "use_fake_tensor", True)
    def test_input_mutation3(self):
        def fn(a):
            a += 1
            a *= 2
            aten.sigmoid_(a)
            a = a.view(64)
            a += 3
            a *= 4
            aten.relu_(a)
            return a

        arg1 = torch.randn([1, 64], device=self.device)
        arg2 = arg1.clone()
        correct1 = fn(arg1)
        opt_fn = torch._dynamo.optimize_assert(compile_fx)(fn)
        actual1 = opt_fn(arg2)

        self.assertTrue(same(actual1, correct1))
        self.assertTrue(same(arg1, arg2))

    def test_input_mutation4(self):
        def fn(a):
            torch.relu_(a)
            return a

        arg1 = torch.randn([1, 64], device=self.device)
        arg2 = arg1.clone()
        correct1 = fn(arg1)
        opt_fn = torch._dynamo.optimize_assert(compile_fx)(fn)
        actual1 = opt_fn(arg2)

        self.assertTrue(same(actual1, correct1))
        self.assertTrue(same(arg1, arg2))

    @patch.object(functorch_config, "use_fake_tensor", True)
    def test_slice_mutation1(self):
        def fn(a):
            x = torch.zeros_like(a)
            b = x + 1
            x[:, 3] = 3.0
            c = torch.clone(x)
            x[4, :] = 4.0
            d = x + 1
            return x, b, c, d

        self.common(fn, (torch.randn([8, 8]),))

    @patch.object(functorch_config, "use_fake_tensor", True)
    def test_slice_mutation2(self):
        def fn(a):
            a[:, 20:40] = a[:, 20:40] + 1
            a[:, 2:11] = a[:, 1:10] + 2

        arg1 = torch.randn([1, 64], device=self.device)
        arg2 = arg1.clone()
        fn(arg1)
        opt_fn = torch._dynamo.optimize_assert(compile_fx)(fn)
        opt_fn(arg2)

        self.assertTrue(same(arg1, arg2))

    def test_indirect_load_broadcast(self):
        def fn(in_ptr0, in_ptr1, in_ptr2):
            return torch.gather(in_ptr1, 0, in_ptr2) + in_ptr0

        arg190 = rand_strided((32, 21), (1, 32), device=self.device, dtype=torch.int64)
        arg190.fill_(0)
        arg111 = rand_strided(
            (9521, 512), (512, 1), device=self.device, dtype=torch.float32
        )
        self.common(
            fn,
            (
                torch.randn(32, 1),
                arg111,
                arg190,
            ),
        )

    @unittest.skipIf(not has_torchvision_roi_align(), "requirs torchvision")
    def test_roi_align(self):
        def fn(a, b):
            return torch.ops.torchvision.roi_align(a, b, 0.25, 7, 7, 2, False)

        self.common(fn, (torch.zeros([4, 256, 296, 304]), torch.zeros([2292, 5])))

    @requires_decomp(aten.nll_loss_forward)
    def test_nll_loss_forward(self):
        def fn(a, b):
            return aten.nll_loss_forward(a, b, None, 1, -100)

        self.common(
            fn,
            (
                torch.randn([5, 5]),
                torch.zeros([5], dtype=torch.int64),
            ),
        )

    def test_isinf(self):
        def fn(x):
            return x.isinf(), x.isnan()

        self.common(
            fn, [torch.tensor([1, float("inf"), 2, float("-inf"), float("nan")])]
        )
        self.common(
            fn,
            [
                torch.tensor(
                    [1, float("inf"), 2, float("-inf"), float("nan")],
                    dtype=torch.float64,
                )
            ],
        )

    def test_any(self):
        def fn(x):
            return (
                x.any(-1),
                x.isinf().any(),
                torch.all(x.isinf(), dim=0),
                torch.all(torch.logical_not(x.isinf())),
            )

        self.common(fn, [-torch.rand(64)])
        tmp = torch.randn(16, 8)
        tmp[1, 1] = float("inf")
        self.common(fn, [tmp])

    def test_inplace_activations(self):
        def fn(x):
            a = aten.hardswish_(x + 1)
            b = aten.hardtanh_(x + 1)
            c = aten.leaky_relu_(x + 1)
            d = aten.silu_(x + 1)
            e = aten.log1p(x + 1)
            f = aten.masked_fill_(x + 1, torch.zeros_like(x, dtype=torch.bool), 99.0)
            h = aten.masked_fill_(x + 1, torch.ones_like(x, dtype=torch.bool), 99.0)
            return (a, b, c, d, e, f, h)

        self.common(fn, [torch.randn(64) * 10])

    def test_baddbmm(self):
        def fn(a, b, c):
            return aten.baddbmm(a, b, c)

        self.common(
            fn,
            [
                torch.randn(6, 1, 100),
                torch.randn(6, 128, 64),
                torch.randn(6, 64, 100),
            ],
            # Mismatched elements: 1212 / 76800 (1.6%)
            # Greatest absolute difference: 0.001953125 at index (0, 0, 93) (up to 1e-05 allowed)
            # Greatest relative difference: 1.0 at index (3, 19, 4) (up to 0.001 allowed)
            atol=0.002,
            rtol=0.001,
        )

    @patch.object(config.triton, "max_tiles", 2)
    def test_fuse_tiled(self):
        def fn(a, b, c):
            return a + b, c + 1

        self.common(
            fn, [torch.randn(128, 1), torch.randn(1, 128), torch.randn(128, 128)]
        )

    def test_expand_as(self):
        def fn(a, b):
            return aten.expand_as(a, b), aten.expand_as(a + 1, b + 1) + 1

        self.common(
            fn,
            [
                torch.randn(6, 1, 100),
                torch.randn(6, 128, 100),
            ],
        )

    def test_index_put1(self):
        def fn(a, b, c):
            return (
                torch.index_put(a, [b], c),
                torch.index_put_(a + 1, [b + 1], c + 1) + 1,
            )

        self.common(
            fn,
            [
                torch.randn([800, 256, 7, 7]),
                torch.randperm(601),
                torch.randn([601, 256, 7, 7]),
            ],
        )
        self.common(
            fn, [torch.randn(1024, 4, 2), torch.arange(4), torch.randn(4, 1, 1)]
        )

    def test_index_put2(self):
        def fn(a, b, c):
            return torch.index_put(a, [b], c, True)

        self.common(
            fn,
            [
                torch.randn([100, 256, 7, 7]),
                torch.randint(0, 100, size=[600], dtype=torch.int64),
                torch.randn([600, 256, 7, 7]),
            ],
            # workaround for https://github.com/openai/triton/issues/558
            check_lowp=False,
        )

    def test_index_put3(self):
        def fn(a, b, c):
            torch.ops.aten.index_put_(a, (None, b, None), c)
            a1 = a + 1
            torch.ops.aten.index_put_(a1, (None, b + 1, None), c + 1)
            return (a, a1)

        self.common(
            fn,
            [
                torch.randn([1024, 4, 2]),
                torch.arange(3),
                torch.randn([1024, 1, 2]),
            ],
        )

    def test_index_put_as_masked_fill(self):
        def fn(a, b, c, d):
            a = a.clone()
            torch.ops.aten.index_put_(a, [b], c, d)
            return a

        self.common(
            fn,
            (
                torch.randn([1024, 4, 2]),
                torch.randn([1024, 4, 2]) > 0,
                torch.randn([]),
                False,
            ),
        )

        self.common(
            fn,
            (
                torch.randn([1024, 4, 2]),
                torch.randn([1024, 4, 2]) > 0,
                torch.randn([]),
                True,
            ),
        )

    def test_index_put_fallback1(self):
        def fn(a, b, c, d):
            a = a.clone()
            torch.ops.aten.index_put_(a, [b], c, d)
            return a

        self.common(
            fn,
            (
                torch.randn([3]),
                torch.as_tensor([True, True, False]),
                torch.randn([2]),
                False,
            ),
        )

        self.common(
            fn,
            (
                torch.randn([3]),
                torch.as_tensor([True, True, False]),
                torch.randn([2]),
                True,
            ),
        )

    def test_index_put_fallback2(self):
        def fn(a, b, c, d, e):
            a = a.clone()
            torch.ops.aten.index_put_(a, [None, b, c], d, e)
            return a

        self.common(
            fn,
            (
                torch.randn([1, 2, 3]),
                torch.as_tensor([0, 1]),
                torch.as_tensor([True, True, False]),
                torch.randn([]),
                False,
            ),
        )
        self.common(
            fn,
            (
                torch.randn([1, 2, 3]),
                torch.as_tensor([0, 1]),
                torch.as_tensor([True, True, False]),
                torch.randn([]),
                True,
            ),
        )

    @patch.object(config, "fallback_random", True)
    def test_bernoulli1(self):
        def fn(a):
            b = torch.empty_like(a)
            return aten.bernoulli_(b), b

        self.common(
            fn,
            [
                torch.randn([100]),
            ],
        )

    def test_bernoulli2(self):
        def fn(a):
            return aten.bernoulli(a)

        self.common(
            fn,
            [torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0])],
        )

    def test_narrow(self):
        def fn(x):
            return aten.narrow(x, 1, 10, 16), aten.narrow(x + 2, 0, 10, 16) + 1

        self.common(fn, [torch.randn(64, 64)])

    def test_as_strided(self):
        def fn(x):
            return (
                aten.as_strided(x, (8, 8, 64), (8 * 64, 64, 1), 0),
                aten.as_strided(x + 1, (8, 8, 64), (8 * 64, 64, 1), 0) + 2,
            )

        self.common(fn, [torch.randn(64, 64)])

    def test_as_strided_scatter(self):
        def fn(a, b):
            return aten.as_strided_scatter(
                a * 8 + 10,
                b * 2 - 4,
                size=(a.shape[0], a.shape[1] // 2),
                stride=(a.shape[1], 2),
                storage_offset=0,
            )

        self.common(fn, [torch.randn(10, 1024), torch.randn(10, 512)])

    def test_select_scatter(self):
        def fn(x, a, b):
            return (
                aten.select_scatter(x, a, 1, 0),
                aten.select_scatter(x, b, 0, 1),
            )

        self.common(
            fn,
            [
                torch.randn(8, 197, 38),
                torch.randn(8, 38),
                torch.randn(197, 38),
            ],
        )

    def test_slice_scatter(self):
        def fn(x, a):
            return (
                aten.slice_scatter(x, a, 2, 10, -10),
                aten.slice_scatter(x, a[:, :, :40], 2, 10, -10, 2),
            )

        self.common(
            fn,
            [
                torch.randn(4, 8, 100),
                torch.randn(4, 8, 80),
            ],
        )

    def test_slice_scatter2(self):
        def fn(a, b):
            return aten.slice_scatter(a, b, 0, 0, 9223372036854775807)

        self.common(
            fn,
            [
                torch.randn([8, 197, 384]),
                torch.randn([8, 197, 384]),
            ],
        )

    def test_scatter1(self):
        def fn(a, dim, index, b):
            return aten.scatter(a, dim, index, b)

        self.common(
            fn,
            [
                torch.zeros(2, 3),
                -1,
                torch.tensor([[0]]),
                torch.ones(2, 3),
            ],
        )

    def test_scatter2(self):
        if self.device == "cuda":
            raise unittest.SkipTest("unstable on sm86")

        def fn(a, dim, index, b):
            return aten.scatter.reduce(a, dim, index, b, reduce="add")

        self.common(
            fn,
            [
                torch.zeros(64, 512),
                0,
                torch.zeros((64, 512), dtype=torch.int64),
                torch.ones(64, 512),
            ],
        )

    def test_scatter3(self):
        def fn(a, dim, index, b):
            return aten.scatter(a, dim, index, b, reduce="add")

        self.common(
            fn,
            [
                torch.randn(5, 29, 13),
                2,
                torch.tensor([[[3, 5, 7, 9]]]),
                0.8,  # src can be a scalar
            ],
            # Mismatched elements: 1 / 1885 (0.1%)
            # Greatest absolute difference: 0.00018310546875 at index (0, 0, 3) (up to 1e-05 allowed)
            # Greatest relative difference: 0.0022371364653243847 at index (0, 0, 3) (up to 0.001 allowed)
            atol=2e-4,
            rtol=1e-3,
        )

    def test_scatter4(self):
        def fn(x, ind, src):
            return torch.scatter(x, 0, ind, src)

        self.common(
            fn,
            (torch.randn(196, 992), torch.randint(196, (1, 992)), torch.randn(1, 992)),
        )

    @unittest.skip("Flaky test, needs debugging")
    def test_scatter_add1(self):
        def fn(a, dim, index, b):
            return aten.scatter_add(a, dim, index, b)

        self.common(
            fn,
            [
                torch.randn(2, 3),
                0,
                torch.tensor([[0]]),
                torch.randn(2, 3),
            ],
        )

    def test_scatter_add2(self):
        def fn(a, dim, index, b):
            return aten.scatter_add(a, dim, index, b)

        self.common(
            fn,
            [
                torch.randn(2, 3),
                0,
                torch.tensor([[0, 0, 0], [1, 1, 1]]),
                torch.randn(2, 3),
            ],
        )

    def test_scatter_add3(self):
        def fn(a, dim, index, b):
            return aten.scatter_add(a, dim, index, b)

        self.common(
            fn,
            [
                torch.randn(5, 29, 13),
                2,
                torch.tensor([[[3, 5, 7, 9]]]),
                torch.randn(1, 1, 10),
            ],
        )

    def test_scatter_reduce1(self):
        def fn(a, dim, index, b):
            return aten.scatter_reduce(a, dim, index, b, "sum")

        self.common(
            fn,
            [
                torch.randn(5, 29, 13),
                2,
                torch.tensor([[[3, 5, 7, 9]]]),
                torch.randn(1, 1, 10),
            ],
        )

    def test_scatter_reduce2(self):
        def fn(a, dim, index, b):
            return aten.scatter_reduce(a, dim, index, b, "sum", include_self=False)

        self.common(
            fn,
            [
                torch.randn(2, 3),
                0,
                torch.zeros((2, 3), dtype=torch.int64),
                torch.randn(2, 3),
            ],
        )

    # issue #1150
    def test_dense_mask_index(self):
        if self.device == "cpu":
            raise unittest.SkipTest(
                "https://github.com/pytorch/torchdynamo/issues/1697"
            )

        def fn(x, y):
            y = torch.ops.aten.select.int(y, 0, 2)
            z = x * y
            return z.sum()

        self.common(fn, [torch.randn(102400), torch.randn(3)])

    def test_new_empty_strided(self):
        def fn(a):
            return aten.new_empty_strided(a, [1, 128, 128], [16384, 128, 1]).fill_(123)

        self.common(fn, [torch.randn(55)])

    @patch.object(torch._inductor.config.triton, "cudagraphs", True)
    def test_dropout(self):
        random.seed(1234)
        torch.manual_seed(1234)

        @torch._dynamo.optimize("inductor")
        def fn1(a):
            return torch.nn.functional.dropout(a)

        x = torch.ones(1000, device=self.device, dtype=torch.float32)
        result1 = fn1(x)
        self.assertTrue(400 < result1.nonzero().shape[0] < 600)
        self.assertTrue(0.9 < result1.mean().item() < 1.1)

        random.seed(1234)
        torch.manual_seed(1234)

        @torch._dynamo.optimize("inductor")
        def fn2(a):
            return torch.nn.functional.dropout(a, 0.5, True)

        result2 = fn2(x)
        self.assertTrue(400 < result2.nonzero().shape[0] < 600)
        self.assertTrue(0.9 < result2.mean().item() < 1.1)

    def test_dropout_deterministic(self):
        @torch._dynamo.optimize("inductor")
        def fn(a):
            return torch.nn.functional.dropout(a, 0.55, True)

        for cg in (False, True):
            with patch.object(torch._inductor.config.triton, "cudagraphs", cg):
                torch._dynamo.reset()

                x = torch.ones(1024, device=self.device, dtype=torch.float32)

                torch.manual_seed(1234)
                a0 = fn(x).clone()
                a1 = fn(x).clone()
                a2 = fn(x).clone()

                torch.manual_seed(1234)
                b0 = fn(x).clone()
                b1 = fn(x).clone()
                b2 = fn(x).clone()

                # same seed, same values
                self.assertTrue(torch.allclose(a0, b0))
                self.assertTrue(torch.allclose(a1, b1))
                self.assertTrue(torch.allclose(a2, b2))

                # different calls, different values
                self.assertFalse(torch.allclose(a0, a1))
                self.assertFalse(torch.allclose(a1, a2))

    def test_rand_like_deterministic(self):
        @torch._dynamo.optimize("inductor")
        def fn(a):
            return torch.rand_like(a), torch.rand_like(a)

        x = torch.ones(1024, device=self.device, dtype=torch.float32)

        torch.manual_seed(1234)
        a0 = fn(x)[0].clone()
        a1 = fn(x)[0].clone()
        a2 = fn(x)[0].clone()

        torch.manual_seed(1234)
        b0 = fn(x)[0].clone()
        b1 = fn(x)[0].clone()
        b2 = fn(x)[0].clone()

        # same seed, same values
        self.assertTrue(torch.allclose(a0, b0))
        self.assertTrue(torch.allclose(a1, b1))
        self.assertTrue(torch.allclose(a2, b2))

        # different calls, different values
        self.assertFalse(torch.allclose(a0, a1))
        self.assertFalse(torch.allclose(a1, a2))

        c, d = fn(x)
        self.assertFalse(torch.allclose(c, d))
        self.assertTrue((c >= 0).all())
        self.assertTrue((c < 1).all())
        self.assertTrue((d >= 0).all())
        self.assertTrue((d < 1).all())

    def test_max_pool2d_with_indices_backward(self):
        def fn(a, b, c):
            return aten.max_pool2d_with_indices_backward(
                a, b, [2, 2], [2, 2], [0, 0], [1, 1], False, c
            )

        x = torch.randn([2, 4, 18, 14])
        result, indices = aten.max_pool2d_with_indices(
            x,
            [2, 2],
            [2, 2],
            [0, 0],
            [1, 1],
            False,
        )

        self.common(
            fn,
            [
                torch.randn_like(result),
                x,
                indices,
            ],
        )

    def test_max_pool2d_with_indices_backward2(self):
        def fn(a, b, c):
            return aten.max_pool2d_with_indices_backward(
                a, b, [3, 3], [2, 2], [1, 1], [1, 1], True, c
            )

        x = torch.randn([2, 4, 40, 56])
        result, indices = aten.max_pool2d_with_indices(
            x,
            [3, 3],
            [2, 2],
            [1, 1],
            [1, 1],
            True,
        )

        self.common(
            fn,
            [
                torch.randn_like(result),
                x,
                indices,
            ],
        )

    # From https://github.com/pytorch/torchdynamo/issues/1200
    def test_max_pool2d_with_indices_backward3(self):
        def fn(a, b, c):
            return aten.max_pool2d_with_indices_backward(
                a, b, [1, 1], [2, 2], [0, 0], [1, 1], False, c
            )

        x = torch.randn([32, 256, 37, 38])
        result, indices = aten.max_pool2d_with_indices(
            x,
            [1, 1],
            [2, 2],
            0,
            1,
            False,
        )
        self.common(
            fn,
            [
                torch.randn_like(result),
                x,
                indices,
            ],
        )

    # From https://github.com/pytorch/torchdynamo/issues/1352
    def test_max_pool2d_with_indices_backward4(self):
        def fn(a, b, c):
            return aten.max_pool2d_with_indices_backward(
                a, b, [5, 5], [1, 1], [2, 2], [1, 1], False, c
            )

        x = torch.randn([2, 64, 3, 4])
        result, indices = aten.max_pool2d_with_indices(
            x,
            [5, 5],
            [1, 1],
            2,
            1,
            False,
        )
        self.common(
            fn,
            [
                torch.randn_like(result),
                x,
                indices,
            ],
        )

    def test_avg_pool2d_backward(self):
        def fn(a, b):
            return aten.avg_pool2d_backward(
                a,
                b,
                [2, 2],
                [2, 2],
                [0, 0],
                True,
                False,
                None,
            )

        self.common(
            fn,
            [
                torch.randn([2, 4, 7, 7]),
                torch.randn([2, 4, 14, 14]),
            ],
        )

    def test_avg_pool2d_backward2(self):
        def fn(a, b):
            return aten.avg_pool2d_backward(
                a,
                b,
                [3, 3],
                [1, 1],
                [1, 1],
                True,
                False,
                None,
            )

        self.common(
            fn,
            [
                torch.randn([1, 1, 20, 15]),
                torch.randn([1, 1, 20, 15]),
            ],
        )

    def test_avg_pool2d_backward3(self):
        def fn(a, b):
            return aten.avg_pool2d_backward(
                a,
                b,
                [1, 1],
                [2, 2],
                [0, 0],
                False,
                False,
                None,
            )

        self.common(
            fn,
            [
                torch.randn([1, 2016, 11, 11]),
                torch.randn([1, 2016, 21, 21]),
            ],
        )

    def test_mm_views(self):
        def fn(a, b):
            return torch.mm(a.view(32, 32), b.view(32, 32))

        self.common(
            fn,
            (
                torch.randn([32, 32]).transpose(0, 1),
                torch.randn([1, 32, 32]).transpose(0, 1),
            ),
            check_lowp=False,
        )
        expected_kernel = 0
        # codegen mm kernel from template
        if config.triton.mm != "aten" and self.device == "cuda":
            expected_kernel = 1
        if config.triton.mm == "autotune":
            self.assertLessEqual(
                torch._inductor.metrics.generated_kernel_count, expected_kernel
            )
        self.assertEqual(
            torch._inductor.metrics.generated_kernel_count, expected_kernel
        )

    @patch.object(config.triton, "cudagraphs", False)
    def test_lowmem_dropout1(self):
        n = 100000
        weight = torch.ones(
            n, device=self.device, dtype=torch.float32, requires_grad=True
        )
        ones = torch.ones(n, device=self.device, dtype=torch.float32)

        @torch._dynamo.optimize_assert("inductor")
        def run(x, train=True):
            return F.dropout(x * weight, 0.33, train)

        def check(r, g):
            rmean = r.mean().item()
            gmean = g.mean().item()
            rcount = len(r.nonzero())
            gcount = len(g.nonzero())

            # dropped elements should match
            self.assertTrue(same(r.nonzero(), g.nonzero()))
            self.assertEqual(rcount, gcount)

            # dropped should be close to 0.33
            self.assertGreater(rcount, 0.64 * n)
            self.assertGreater(0.68 * n, rcount)

            self.assertAlmostEqual(rmean, gmean)
            self.assertAlmostEqual(rmean, 1.0, places=2)

        r1 = run(ones, train=False)
        r1.sum().backward()
        g1 = weight.grad.clone()
        # eval mode should be all ones
        self.assertTrue(same(r1, torch.ones_like(r1)))
        self.assertTrue(same(g1, torch.ones_like(g1)))

        torch.manual_seed(1234)
        weight.grad.zero_()
        r2 = run(ones)
        r2.sum().backward()
        g2 = weight.grad.clone()
        check(r2, g2)

        torch.manual_seed(1234)
        weight.grad.zero_()
        r3 = run(ones)
        r3.sum().backward()
        g3 = weight.grad.clone()
        check(r3, g3)

        # second run is same result as first
        self.assertTrue(same(r2, r3))
        self.assertTrue(same(g2, g3))

    def test_lowmem_dropout2(self):
        m = torch.nn.Sequential(
            torch.nn.Linear(32, 32, bias=False),
            torch.nn.Dropout(),
            torch.nn.Linear(32, 32, bias=False),
            torch.nn.Dropout(),
        ).to(self.device)

        @torch._dynamo.optimize_assert("inductor")
        def run(x):
            return m(x)

        torch._inductor.metrics.generated_kernel_count = 0
        result = run(torch.randn([8, 32], device=self.device))
        result.sum().backward()

        expected_kernel = 4
        if config.triton.mm != "aten" and self.device == "cuda":
            # fwd: 2 * (mm+dropout) kernels = 2 kernels
            # bwd: dropout + (mm) + 2 * (mm+dropout) kernels = 4 kernels
            # expect 2 + 4 = 6 kernels
            expected_kernel = 6
        if config.triton.mm == "autotune":
            self.assertLessEqual(
                torch._inductor.metrics.generated_kernel_count, expected_kernel
            )
        self.assertEqual(
            torch._inductor.metrics.generated_kernel_count, expected_kernel
        )

    def test_roll(self):
        def fn(a):
            return (
                aten.roll(a, [-3, 10], [1, 2]),
                aten.roll(a, [5]),
            )

        self.common(
            fn,
            [
                torch.randn([2, 56, 56, 16]),
            ],
        )

    def test_argmax_argmin1(self):
        def fn(x):
            return (aten.argmax(x), aten.argmin(x))

        self.common(
            fn,
            [
                torch.randn([8, 256, 256]),
            ],
        )

    def test_argmax_argmin2(self):
        def fn(x):
            return (
                aten.argmax(x, 0),
                aten.argmin(x, 0),
                aten.argmax(x, 1),
                aten.argmin(x, 1),
            )

        self.common(
            fn,
            [
                torch.randn([144, 144]),
            ],
            # Mismatched elements: 1 / 144 (0.7%)
            # Greatest absolute difference: 26 at index (71,)
            # Greatest relative difference: 0.4126984179019928 at index (71,)
            atol=1e-5,
            rtol=0.5,
        )

    @unittest.skip(
        """
        FIXME: In the case of having equally max/min elements, our implementation returns
        the last index instead of the first one
        """
    )
    def test_argmax_argmin3(self):
        def fn(x):
            return (
                aten.argmax(x, 0),
                aten.argmin(x, 0),
                aten.argmax(x, -1),
                aten.argmin(x, -1),
            )

        self.common(
            fn,
            [torch.randint(0, 5, [10, 10])],
        )

    def test_vdd_clamp(self):
        def fn(x):
            return torch.clamp_min(x, 3)

        self.common(
            fn,
            [
                torch.randn([16], requires_grad=True) * 10,
            ],
        )

    def test_tmp_not_defined_issue1(self):
        def forward(
            primals_3,
            primals_4,
            add_tensor,
            convert_element_type_default,
            div_default,
            reciprocal_default,
        ):
            var_default = torch.ops.prims.var.default(
                convert_element_type_default, [2], correction=0
            )
            sub_tensor = torch.ops.aten.sub.Tensor(add_tensor, div_default)
            mul_tensor_1 = torch.ops.aten.mul.Tensor(sub_tensor, reciprocal_default)
            mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor_1, primals_3)
            add_tensor_2 = torch.ops.aten.add.Tensor(mul_tensor_2, primals_4)
            convert_element_type_default_1 = (
                torch.ops.prims.convert_element_type.default(
                    add_tensor_2, torch.float32
                )
            )
            convert_element_type_default_2 = (
                torch.ops.prims.convert_element_type.default(
                    convert_element_type_default_1, torch.float32
                )
            )
            var_default_1 = torch.ops.prims.var.default(
                convert_element_type_default_2, [2], correction=0
            )
            broadcast_in_dim_default_2 = torch.ops.prims.broadcast_in_dim.default(
                var_default_1, [1, 512, 1], [0, 1]
            )
            sum_default_1 = torch.ops.prims.sum.default(
                convert_element_type_default_2, [2]
            )
            add_tensor_3 = torch.ops.aten.add.Tensor(broadcast_in_dim_default_2, 1e-05)
            return (var_default, sum_default_1, add_tensor_3)

        inps = [
            (torch.Size([1024]), torch.float32),
            (torch.Size([1024]), torch.float32),
            (torch.Size([1, 512, 1024]), torch.float32),
            (torch.Size([1, 512, 1024]), torch.float32),
            (torch.Size([1, 512, 1]), torch.float32),
            (torch.Size([1, 512, 1]), torch.float32),
        ]
        inps = [torch.randn(shape, dtype=dtype) for (shape, dtype) in inps]
        self.common(forward, inps, atol=1e-05, rtol=2e-05)

    @unittest.skipIf(
        TEST_WITH_ASAN
        or os.environ.get("BUILD_ENVIRONMENT", "").startswith("parallelnative"),
        "TODO: debug this with asan",
    )
    def test_tmp_not_defined_issue2(self):
        def forward(arg38_1, arg81_1, getitem_17, new_zeros_default_4):
            div_tensor_7 = torch.ops.aten.div.Tensor(getitem_17, arg81_1)
            mul_tensor_24 = torch.ops.aten.mul.Tensor(div_tensor_7, arg38_1)
            sum_default_7 = torch.ops.aten.sum.default(mul_tensor_24)
            return (new_zeros_default_4, sum_default_7)

        args = [
            ((1, 88, 40, 40), (140800, 1600, 40, 1), torch.float32),
            ((), (), torch.float32),
            ((1, 88, 40, 40), (140800, 1600, 40, 1), torch.float32),
            ((3,), (1,), torch.float32),
        ]
        args = [rand_strided(shape, stride, dtype) for shape, stride, dtype in args]
        self.common(forward, args)

    def test_misaligned_address_issue1(self):
        def forward(sub_tensor_1, unsqueeze_default):
            gather_default = torch.ops.aten.gather.default(
                sub_tensor_1, 1, unsqueeze_default
            )
            return gather_default

        args = [
            ((1, 1000), (1000, 1), torch.float32),
            ((1, 1), (1, 1), torch.int64),
        ]
        args = [rand_strided(shape, stride, dtype) for shape, stride, dtype in args]
        self.common(forward, args)

    def test_invalid_operand_issue1(self):
        def forward(arg0_1, arg1_1, arg3_1, squeeze, view_1, slice_1):
            slice_scatter = torch.ops.aten.slice_scatter.default(
                slice_1, arg3_1, 1, 1, 9223372036854775807
            )
            slice_scatter_1 = torch.ops.aten.slice_scatter.default(
                arg1_1, slice_scatter, 0, 0, 9223372036854775807
            )
            slice_2 = torch.ops.aten.slice.Tensor(
                slice_scatter_1, 0, 0, 9223372036854775807
            )
            select_scatter = torch.ops.aten.select_scatter.default(
                slice_2, squeeze, 1, 0
            )
            slice_scatter_2 = torch.ops.aten.slice_scatter.default(
                slice_scatter_1, select_scatter, 0, 0, 9223372036854775807
            )
            view = torch.ops.aten.view.default(slice_scatter_2, [-1, 128])
            embedding = torch.ops.aten.embedding.default(arg0_1, view, 1)
            return [embedding, view_1]

        args = [
            ((50005, 768), (768, 1), torch.float32),
            ((8, 128), (128, 1), torch.int64),
            ((8, 127), (127, 1), torch.int64),
            ((8,), (1,), torch.int64),
            ((1024,), (1,), torch.int64),
            ((8, 128), (128, 1), torch.int64),
        ]
        args = [rand_strided(shape, stride, dtype) for shape, stride, dtype in args]
        self.common(forward, args)

    def test_sizehint_issue1(self):
        def forward(x):
            return torch.nn.functional.unfold(
                x, kernel_size=[4, 4], dilation=1, padding=0, stride=[4, 4]
            )

        args = [((2, 24, 56, 56), (75264, 3136, 56, 1), torch.float32, False)]
        args = [
            rand_strided(sh, st, dt).requires_grad_(rg) for (sh, st, dt, rg) in args
        ]
        self.common(forward, args)

    def test_zero_dim_reductions(self):
        for kd in [True, False]:
            inps0 = (torch.zeros(2, 0, device=self.device, dtype=torch.float16), 1, kd)
            failed_ops = [aten.argmin, aten.argmax, aten.max, aten.min]
            for fo in failed_ops:
                with self.assertRaisesRegex(
                    IndexError, "Expected reduction dim 1 to have non-zero size"
                ):
                    mod = make_fx(fo)(*inps0)
                    _ = compile_fx_inner(mod, inps0)

            pass_ops = [
                lambda *x: fn(*x) for fn in [aten.sum, aten.prod, aten.any, aten.all]
            ]
            for po in pass_ops:
                compiled = torch._dynamo.optimize("inductor")(po)
                expected = po(*inps0)
                actual = compiled(*inps0)

            self.assertTrue(torch.allclose(actual, expected, atol=1e-3, rtol=1e-3))

    @requires_cuda()
    def test_unspec_inputs(self):
        def fn(x, y):
            return x + y, x * y, x / y

        opt = torch._dynamo.optimize("inductor")(fn)
        dtypes = [
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.int32,
            torch.int64,
        ]

        for d in dtypes:
            inputs = (
                rand_strided((2, 3), (3, 1), dtype=torch.float32, device="cuda"),
                rand_strided((), (), dtype=d, device="cpu"),
            )
            self.assertTrue(same(opt(*inputs), fn(*inputs)))
            inputs = (inputs[1], inputs[0])
            self.assertTrue(same(opt(*inputs), fn(*inputs)))

    @patch.object(config.triton, "mm", "aten")
    def test_list_clearing(self):

        if self.device == "cpu":
            contexts = [contextlib.nullcontext]
        else:
            contexts = [
                contextlib.nullcontext,
                lambda: patch.object(config.triton, "cudagraphs", True),
            ]

        for context in contexts:
            with context():
                inps = [
                    torch.rand([5, 5]).to(self.device),
                    torch.rand([5, 5]).to(self.device),
                ]
                inp_refs = [weakref.ref(inp) for inp in inps]

                def fn(x, y):
                    a = x + y
                    return (a @ a,)

                fn_fx = make_fx(fn)(inps[0], inps[1])
                fn_compiled = compile_fx_inner(fn_fx, inps)

                test_self = self
                matmul_seen = False

                class TestRefMode(TorchDispatchMode):
                    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                        kwargs = kwargs if kwargs else {}

                        nonlocal inps
                        nonlocal inp_refs
                        nonlocal test_self
                        nonlocal matmul_seen

                        # by matmul, inputs should be deallocated
                        if func is aten.mm.out:
                            matmul_seen = True
                            test_self.assertEqual(len(inps), 0)
                            test_self.assertIsNone(inp_refs[0]())
                            test_self.assertIsNone(inp_refs[1]())

                        return func(*args, **kwargs)

                with TestRefMode():
                    fn_compiled(inps)

                # for some reason, TorchDispatch doesnt capture the
                # cuda mm call (even without cudagraphs)
                if self.device == "cpu":
                    self.assertTrue(matmul_seen)
                else:
                    self.assertEqual(len(inps), 0)

    def test_dtype_mismatch_issue(self):
        def fn(x):
            attn = torch.nn.functional.pad(x, [0, 1])
            return attn.softmax(dim=-1)

        x = torch.rand(128, 32, 63)
        res_ref = fn(x)
        res = torch._dynamo.optimize("inductor")(fn)(x)
        self.assertEqual(res, res_ref)

    @unittest.skipIf(HAS_CUDA, "histogramdd only supports cpu")
    def test_kwargs(self):
        def fn(x, y):
            return torch.histogramdd(
                x,
                bins=[3, 3],
                weight=y,
            )

        self.common(
            fn,
            [torch.randn((4, 2)), torch.randn((4))],
        )


if HAS_CPU:

    class SweepInputsCpuTest(SweepInputs2, TestCase):
        gen = InputGen(10, "cpu")

    SweepInputsCpuTest.populate()

    class CpuTests(TestCase):
        common = check_model
        device = "cpu"

    CommonTemplate.install(CpuTests, "cpu")

    class CPUReproTests(TestCase):
        def test_conv_stride_constraints(self):
            for fmt in [torch.channels_last, torch.contiguous_format]:
                # TorchDispatch doesn't work in our cuda invocation for some reason
                m = torch.nn.Conv2d(5, 6, [3, 3])

                def fn(inp, weight):
                    return (
                        F.conv2d(
                            inp, weight, None, m.stride, m.padding, m.dilation, m.groups
                        ),
                    )

                inp = torch.randn([2, 5, 16, 16])
                inps = [inp, m.weight.to(memory_format=fmt)]
                fn_fx = make_fx(fn)(*inps)
                fn_compiled = compile_fx_inner(fn_fx, inps)
                test_self = self
                conv_seen = False

                class RecordFunctions(TorchDispatchMode):
                    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                        kwargs = kwargs if kwargs else {}
                        if func == torch.ops.aten.convolution.default:
                            test_self.assertTrue(
                                args[0].is_contiguous(memory_format=fmt)
                            )
                            test_self.assertTrue(
                                args[1].is_contiguous(memory_format=fmt)
                            )
                            nonlocal conv_seen
                            conv_seen = True

                        return func(*args, **kwargs)

                with RecordFunctions():
                    out = fn_compiled(inps)

                self.assertTrue(conv_seen)

        def test_inplace_squeeze_needed(self):
            mod = torch.nn.Sequential(
                torch.nn.Linear(10, 10),
                torch.nn.LayerNorm(10),
                torch.nn.ReLU(),
            ).eval()

            @torch._dynamo.optimize("inductor")
            def fn(x):
                return mod(x)

            v = torch.randn(10)
            result = fn(v)
            assert same(result, mod(v))

        def test_inplace_add_alpha(self):
            def fn(x, y):
                aten.add_.Tensor(x, y, alpha=0.55)
                return (x,)

            x1 = torch.zeros(10)
            x2 = torch.zeros(10)
            x3 = torch.zeros(10)
            y = torch.randn(10)
            fn_fx = make_fx(fn)(x1, y)
            fn_compiled = compile_fx_inner(fn_fx, [x1, y])
            fn(x2, y)
            fn_compiled([x3, y])
            assert same(x2, x3)

        def test_no_op_squeeze(self):
            @torch._dynamo.optimize("inductor")
            def forward(arg0_1):
                return torch.ops.aten.squeeze.dim(arg0_1, 1)

            x = torch.randn((10, 20))
            assert same(x, forward(x))

        def test_parallel_num_threads(self):
            @torch._dynamo.optimize("inductor")
            def fn(x1, x2):
                return x1 + x2

            @contextlib.contextmanager
            def set_num_threads(num_threads):
                orig_num_threads = torch.get_num_threads()
                torch.set_num_threads(num_threads)
                yield
                torch.set_num_threads(orig_num_threads)

            x1 = torch.randn((10, 20))
            x2 = torch.randn((10, 20))
            with set_num_threads(1):
                assert same(x1 + x2, fn(x1, x2))
            with set_num_threads(4):
                assert same(x1 + x2, fn(x1, x2))

        @patch("torch.cuda.is_available", lambda: False)
        def test_timed_cpu_only(self):
            timed(lambda: torch.randn(10), ())

        def test_complex_memory_overlap(self):
            dense = torch.zeros(64, 32)
            self.assertFalse(complex_memory_overlap(dense))
            self.assertFalse(complex_memory_overlap(dense.t()))

            strided = dense.split(4, dim=1)
            self.assertFalse(complex_memory_overlap(strided[0]))
            self.assertFalse(complex_memory_overlap(strided[0].t()))

            unsqueezed = dense.unsqueeze(1)
            self.assertFalse(complex_memory_overlap(unsqueezed))
            self.assertFalse(complex_memory_overlap(unsqueezed.permute(1, 2, 0)))

            expanded = unsqueezed.expand(-1, 2, -1)
            self.assertTrue(complex_memory_overlap(expanded))
            self.assertTrue(complex_memory_overlap(expanded.permute(1, 2, 0)))

            gathered = dense.index_select(0, torch.IntTensor([1, 0, 1]))
            self.assertFalse(complex_memory_overlap(gathered))
            self.assertFalse(complex_memory_overlap(gathered.t()))

        @unittest.skipIf(
            not codecache.get_cpu_proc_info(), "Does not support vectorization"
        )
        @patch("torch.cuda.is_available", lambda: False)
        def test_sign_cpu_only(self):
            def fn(x):
                return (torch.sign(x),)

            x = torch.randn((2, 9))
            x[0, 0] = torch.nan
            x[1, -1] = torch.nan

            with patch.object(config.cpp, "simdlen", 8):
                torch._dynamo.reset()
                metrics.reset()
                traced = make_fx(fn)(x)
                compiled = compile_fx_inner(traced, [x])
                assert same(fn(x)[0], compiled([x])[0], equal_nan=True)
                assert metrics.generated_cpp_vec_kernel_count == 1

        # Currently, we enabled AVX2 and AVX512 for vectorization. If the platform is not
        # supported, the vectorization will not work and skip this test case. For ARM or
        # other platforms support, we just need to add the ISA info to the supported_vector_isa
        # and include proper aten vectorization head file.
        @unittest.skipIf(
            not codecache.get_cpu_proc_info(), "Does not support vectorization"
        )
        @patch("torch.cuda.is_available", lambda: False)
        def test_vec_kernel_cpu_only(self):
            def fn(x1, x2):
                # Current, there are some limitations as follows.
                #   rsqrt:
                #     assert [both a fallback and a decomp for same kernel: aten.rsqrt.default]
                #   round:
                #     couldn't find symbolic meta function/decomposition
                #   fmod/logical_and/logic_or:
                #     vec kernel has not support to_type
                x = torch.abs(x1)
                x = torch.sin(x)
                x = torch.neg(x)
                x = torch.square(x)
                x = torch.sigmoid(x)
                x = torch.relu(x)
                x = torch.cos(x)
                x = torch.exp(x)
                x = torch.sqrt(x)
                x = torch.add(x, x1)
                x = torch.sub(x, x2)
                x = torch.mul(x, x1)
                x = torch.div(x, x1)
                x = torch.pow(x, 10)
                x = torch.log(x)
                x = torch.floor(x)
                x = torch.ceil(x)
                x = torch.trunc(x)
                x = torch.lgamma(x)
                x = torch.fmod(x, x2)
                x = torch.sign(x)
                res = x + x2
                return (res,)

            x1 = torch.randn((10, 20))
            x2 = torch.randn((10, 20))

            with patch.object(config.cpp, "simdlen", 8):
                torch._dynamo.reset()
                metrics.reset()
                traced = make_fx(fn)(x1, x2)
                compiled = compile_fx_inner(traced, [x1, x2])
                assert same(fn(x1, x2)[0], compiled([x1, x2])[0], equal_nan=True)
                assert metrics.generated_cpp_vec_kernel_count == 1

                torch._dynamo.reset()
                metrics.reset()
                x1 = x1.permute(1, 0)
                x2 = torch.randn((20, 10))
                traced = make_fx(fn)(x1, x2)
                compiled = compile_fx_inner(traced, [x1, x2])
                assert same(fn(x1, x2)[0], compiled([x1, x2])[0], equal_nan=True)
                assert metrics.generated_cpp_vec_kernel_count == 1

                torch._dynamo.reset()
                metrics.reset()
                x1 = torch.randn((10, 7))
                x2 = torch.randn((10, 7))
                traced = make_fx(fn)(x1, x2)
                compiled = compile_fx_inner(traced, ([x1, x2]))
                assert same(fn(x1, x2)[0], compiled([x1, x2])[0], equal_nan=True)
                assert metrics.generated_cpp_vec_kernel_count == 1


if HAS_CUDA:
    import triton
    import triton.language as tl

    class SweepInputsCudaTest(SweepInputs2, TestCase):
        gen = InputGen(10, "cuda")

    SweepInputsCudaTest.populate()

    class CudaTests(TestCase):
        common = check_model_cuda
        device = "cuda"

        def test_simplify_dims(self):
            def fn(a):
                return (a + 1,)

            self.common(
                fn, (torch.randn(2, 3, 10, 5, 6, device="cuda")[:, :, 2::2, :, :],)
            )

    CommonTemplate.install(CudaTests, "cuda")

    class CudaReproTests(TestCase):
        common = check_model_cuda

        def test_index_put_issue(self):
            def forward(
                self,
                arg76_1,
                expand_default,
                full_like_default,
                _to_copy_default_67,
                zeros,
            ):
                sum_sym_int_19 = torch.ops.aten.sum(_to_copy_default_67, [0], True)
                view_default_57 = torch.ops.aten.view.default(
                    sum_sym_int_19, [512, 768]
                )
                where_self = torch.ops.aten.where.self(
                    expand_default, view_default_57, full_like_default
                )
                clone_default_12 = torch.ops.aten.clone.default(zeros)
                index_put__default = torch.ops.aten.index_put_.default(
                    clone_default_12, [arg76_1], where_self, True
                )
                return (index_put__default,)

            inps = [
                (torch.Size([512]), torch.int64),
                (torch.Size([512, 768]), torch.bool),
                (torch.Size([512, 768]), torch.float16),
                (torch.Size([4, 512, 768]), torch.float16),
                (torch.Size([512, 768]), torch.float16),
            ]
            inps = [torch.zeros(())] + [
                torch.ones(shape, dtype=dtype, device="cuda") for (shape, dtype) in inps
            ]
            mod = make_fx(forward)(*inps)
            compiled = compile_fx_inner(mod, inps)
            compiled(inps)

        @requires_cuda()
        def test_input_channels_last(self):
            m = torch.nn.Sequential(
                torch.nn.Conv2d(3, 3, 1, 1),
                ToTuple(),
            ).cuda()
            inp = (
                torch.randn([2, 3, 16, 16]).to(memory_format=torch.channels_last).cuda()
            )

            self.common(
                m,
                (inp,),
                check_lowp=False,
            )

            @torch._dynamo.optimize()
            def foo(m, inp):
                return m(inp)

            self.assertTrue(
                foo(m, inp)[0].is_contiguous(memory_format=torch.channels_last)
            )

        # https://github.com/pytorch/torchdynamo/issues/1681#issuecomment-1283433527
        @requires_cuda()
        def test_unspec_inputs_interop(self):
            class Repro(torch.nn.Module):
                def __init__(self):
                    super().__init__()

                def forward(self, x, y):
                    unsqueeze = torch.ops.aten.unsqueeze.default(x, 4)
                    permute = torch.ops.aten.permute.default(unsqueeze, [0, 1, 2, 4, 3])
                    add = torch.ops.aten.add.Tensor(y, 1)
                    return [permute, add]

            inps = [
                rand_strided(
                    (12, 3, 512, 64), (64, 196608, 768, 1), torch.float32, "cuda"
                ),
                rand_strided((), (), torch.int64, "cpu"),
            ]
            mod = make_fx(Repro().to(device="cuda"))(*inps)
            compiled = compile_fx_inner(mod, inps)
            compiled(inps)

        @patch.object(config, "fallback_random", True)
        def test_dtype_factory_issue(self):
            def forward():
                randn = torch.ops.aten.randn.default(
                    [12, 64, 1, 64],
                    dtype=torch.float32,
                    device=torch.device(type="cuda", index=0),
                    pin_memory=False,
                )
                unsqueeze_default_2 = torch.ops.aten.unsqueeze.default(randn, -1)
                return (unsqueeze_default_2,)

            mod = make_fx(forward)()
            compiled = compile_fx_inner(mod, ())
            assert compiled([])[0].device.type == "cuda"

        @patch.object(config.triton, "cudagraphs", True)
        def test_expanded_inputs_cudagraphs(self):
            @torch._dynamo.optimize("inductor")
            def fn(x, y):
                return x + y

            inputs = (
                rand_strided((5, 5, 5, 5), (0, 5, 0, 1), device="cuda"),
                rand_strided((5, 5, 5, 5), (0, 5, 0, 1), device="cuda"),
            )
            self.assertTrue(same(fn(*inputs), inputs[0] + inputs[1]))

        # TODO: Abstract this out, test more extensively
        @patch.object(config, "dynamic_shapes", True)
        @patch.object(torch._dynamo.config, "dynamic_shapes", True)
        @patch.object(functorch_config, "use_dynamic_shapes", True)
        def test_dynamic_shapes(self):
            torch._dynamo.reset()  # Needed since everywhere else uses "inductor"

            def f(x):
                return x.cos().view(x.shape).sin()

            cnts = torch._dynamo.testing.CompileCounterWithBackend("inductor")

            f2 = torch._dynamo.optimize(cnts)(f)

            f2(torch.randn(32))

            inp = torch.randn(16)
            real_out = f(inp)
            compiled_out = f2(inp)

            self.assertEqual(cnts.frame_count, 1)
            self.assertEqual(real_out, compiled_out)
            torch._dynamo.reset()

        @patch.object(config, "size_asserts", False)
        @patch.object(config.triton, "cudagraphs", True)
        def test_expanded_inputs_cudagraphs_no_size_asserts(self):
            @torch._dynamo.optimize("inductor")
            def fn(x, y):
                return x + y

            inputs = (
                rand_strided((5, 5, 5, 5), (0, 5, 0, 1), device="cuda"),
                rand_strided((5, 5, 5, 5), (0, 5, 0, 1), device="cuda"),
            )
            self.assertTrue(same(fn(*inputs), inputs[0] + inputs[1]))

        @patch.object(config.triton, "cudagraphs", True)
        def test_inplace_updates_cudagraphs(self):
            class Repro(torch.nn.Module):
                def __init__(self):
                    super(Repro, self).__init__()
                    self.weight1 = torch.nn.Parameter(
                        torch.randn(10, 20, requires_grad=True)
                    )

                def forward(self, x):
                    x = torch.matmul(x, self.weight1)
                    return x

            from copy import deepcopy

            model = Repro().cuda()
            model_ref = deepcopy(model)
            model_opt = torch._dynamo.optimize("inductor")(model)

            input = torch.randn(10, 10, device="cuda", requires_grad=True)

            for i in range(2):
                output_ref = model_ref(input)
                output_res = model_opt(input)
                output_ref.sum().backward()
                output_res.sum().backward()
                for (p_ref, p_res) in zip(
                    model_ref.parameters(), model_opt.parameters()
                ):
                    self.assertEqual(p_ref.grad, p_res.grad)
                with torch.no_grad():
                    for param in model_ref.parameters():
                        param.add_(1.0)
                    for param in model_opt.parameters():
                        param.add_(1.0)

        # Probably fails due to the symint math issue caught while adding
        # max_pool2d_with_indices_backward
        @unittest.skip("Accuracy failure, needs debugging")
        def test_accuracy_issue1(self):
            class Repro(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = torch.nn.Linear(
                        in_features=768, out_features=2, bias=True
                    )

                def forward(self, start_positions: torch.Tensor, x: torch.Tensor):
                    linear = self.linear(x)
                    split = linear.split(1, dim=-1)
                    getitem = split[0]
                    squeeze = getitem.squeeze(-1)
                    clamp = start_positions.clamp(0, 128)
                    cross_entropy = torch.nn.functional.cross_entropy(
                        squeeze, clamp, None, None, 128, None, "mean", 0.0
                    )
                    return cross_entropy

            mod = Repro().cuda()
            opt_mod = torch._dynamo.optimize("inductor")(mod)
            mod.eval()
            opt_mod.eval()

            args = [
                ((1,), (1,), torch.int64, "cuda", False),
                ((1, 128, 768), (98304, 768, 1), torch.float32, "cuda", True),
            ]
            args = [
                rand_strided(sh, st, dt, dev).requires_grad_(rg)
                for (sh, st, dt, dev, rg) in args
            ]
            with torch.cuda.amp.autocast(enabled=False):
                assert same_two_models(mod, opt_mod, args), "Dynamo failed"

        def test_autotune_inplace_kernel(self):
            """
            This UT tests autotune on an inplace kernel. The autotune should not contaminate
            the input buffers when tuning with multiple configs. For more details, refer to
            https://github.com/openai/triton/issues/781
            https://github.com/pytorch/torchdynamo/issues/1670
            """
            from torch._C import _cuda_getCurrentRawStream as get_cuda_stream
            from torch._inductor.triton_ops.autotune import CachingAutotuner, grid
            from torch._inductor.utils import instance_descriptor

            def autotune(configs, meta):
                def decorator(fn):
                    return CachingAutotuner(
                        # force autotune by setting save_cache_hook to False
                        fn,
                        meta=meta,
                        configs=configs,
                        save_cache_hook=False,
                    )

                return decorator

            @autotune(
                configs=[
                    triton.Config({"XBLOCK": 1}),
                    triton.Config({"XBLOCK": 2}),
                ],
                meta={
                    "signature": {0: "*fp32", 1: "*fp32", 2: "i32"},
                    "device": 0,
                    "configs": [
                        instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())
                    ],
                    "constants": {},
                },
            )
            @triton.jit
            def kernel(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
                pid = tl.program_id(0)
                block_start = pid * XBLOCK
                offsets = block_start + tl.arange(0, XBLOCK)
                mask = offsets < xnumel
                x = tl.load(in_out_ptr0 + offsets, mask=mask)
                y = tl.load(in_ptr0 + offsets, mask=mask)
                output = x + y
                tl.store(in_out_ptr0 + offsets, output, mask=mask)

            xnumel = 384
            in0 = rand_strided((xnumel,), (1,), device="cuda", dtype=torch.float32)
            inout1 = rand_strided((xnumel,), (1,), device="cuda", dtype=torch.float32)
            inout2 = inout1.clone()

            stream0 = get_cuda_stream(0)
            kernel.run(inout1, in0, xnumel, grid=grid(xnumel), stream=stream0)
            kernel.run(inout2, in0, xnumel, grid=grid(xnumel), stream=stream0)

            assert same(
                inout1, inout2, tol=0.001, equal_nan=True
            ), "failed autotune with inplace kernel"

        @requires_cuda()
        def test_sort_stride_issue(self):
            # This minified testcase comes from detectron2_maskrcnn_r_50_fpn
            # There was a false error from our size_assert code
            @torch._dynamo.optimize(nopython=True)
            def forward(pred_objectness_logits_3_: torch.Tensor):
                sort_3 = pred_objectness_logits_3_.sort(descending=True, dim=1)
                getitem_12 = sort_3[0]
                return getitem_12

            args = [((1, 100), (0, 1), torch.float16, "cuda", False)]
            args = [
                rand_strided(sh, st, dt, dev).requires_grad_(rg)
                for (sh, st, dt, dev, rg) in args
            ]
            result = forward(*args)
            assert same(result, torch.sort(args[0], descending=True, dim=1)[0])

    class TritonCodeGenTests(TestCase):
        from torch._inductor.triton_ops.autotune import CachingAutotuner

        class NoOpCompilerBackend:
            def __init__(self):
                self.example_args = None
                self.model = None

            def noop_backend(
                self,
                model_: torch.fx.GraphModule,
                example_inputs_: typing.List[torch.Tensor],
            ):
                """
                The Noop backend does not compile the fx graph it is given.
                Instead, it transforms the fx graph so that its functions are
                aten operations. It then saves this graph.
                """
                from functorch._src.aot_autograd import Interpreter
                from torch._inductor.decomposition import select_decomp_table
                from torch._subclasses import FakeTensorMode

                fake_mode = FakeTensorMode()

                def interpret(*args, **kwargs):
                    return Interpreter(model_).run(*args[0:], **kwargs)

                fake_flat_tensor_args = [
                    fake_mode.from_tensor(x) for x in example_inputs_
                ]
                fw_module = make_fx(interpret, select_decomp_table())(
                    *fake_flat_tensor_args
                )
                self.model = fw_module
                self.example_args = fake_flat_tensor_args
                return lambda x: example_inputs_

        def get_kernels(self, fn, args) -> typing.List[CachingAutotuner]:
            from torch._inductor.debug import DebugContext
            from torch._inductor.graph import GraphLowering
            from torch._inductor.virtualized import V

            cxt = TritonCodeGenTests.NoOpCompilerBackend()
            torch._dynamo.optimize(backend=cxt.noop_backend)(fn)(*args)
            graph = GraphLowering(cxt.model)
            graph.num_static_inputs = 0
            kernels = []
            with V.set_graph_handler(graph), V.set_debug_handler(DebugContext()):
                graph.run(*(cxt.example_args))
                mod = graph.compile_to_module()

                for val in mod.__dict__.values():
                    if isinstance(
                        val, torch._inductor.triton_ops.autotune.CachingAutotuner
                    ):
                        kernels.append(val)

            return kernels

        def test_divisibile_by_16_covers_numel_args(self):
            def fn(a: torch.Tensor) -> torch.Tensor:
                return torch.sum(a)

            kernels = self.get_kernels(fn, [torch.randn([256, 256], device="cuda")])
            self.assertTrue(len(kernels) == 2, "SUM should result in two kernels")

            # kernel0 reduces from 256 to (xnumel=8, rnumel=8192), which means it reduces 256 by 256 into an array of
            # size 8 by accumulating 8192 elements at once note that rnumel is equal to 512 * 16, so rnumel which is
            # at slot 3 should be in the divisible by 16 descriptor
            arguments_that_are_divisible_by_16_in_kernel0 = (
                kernels[0].meta["configs"][0].divisible_by_16
            )
            self.assertEqual(arguments_that_are_divisible_by_16_in_kernel0, (0, 1, 3))

            # kernel1 reduces from 8 elements to a single scalar.
            arguments_that_are_divisible_by_16_in_kernel1 = (
                kernels[1].meta["configs"][0].divisible_by_16
            )
            self.assertEqual(arguments_that_are_divisible_by_16_in_kernel1, (0, 1))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if (HAS_CPU or HAS_CUDA) and not TEST_WITH_ROCM:
        run_tests(needs="filelock")
