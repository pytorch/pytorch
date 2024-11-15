# Owner(s): ["module: inductor"]
# flake8: noqa: E731
# Skip do not assign a lambda expression, use a def
import functools
import logging
from unittest.mock import patch

import torch
import torch._dynamo.testing
import torch._inductor.test_case
from torch._higher_order_ops.triton_kernel_wrap import (
    generate_ttir,
    triton_kernel_wrapper_functional,
    triton_kernel_wrapper_mutation,
)
from torch._inductor import metrics
from torch._inductor.utils import run_and_get_code
from torch._library import capture_triton
from torch.testing import FileCheck
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import (
    parametrize,
    skipIfRocm,
    skipIfXpu,
    TEST_WITH_ROCM,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CUDA, HAS_GPU, HAS_XPU
from torch.testing._internal.logging_utils import logs_to_string

# Defines all the kernels for tests
from torch.testing._internal.triton_utils import *  # noqa: F403
from torch.utils._triton import has_triton_package, has_triton_tma


if HAS_GPU:
    import triton
    from triton import language as tl

    if not TEST_WITH_ROCM:
        if HAS_CUDA:
            try:
                from triton.language.extra.libdevice import (  # @manual
                    fast_dividef,
                    fast_dividef as my_fast_dividef,
                )
            except ImportError:
                from triton.language.extra.cuda.libdevice import (  # @manual
                    fast_dividef,
                    fast_dividef as my_fast_dividef,
                )
        elif HAS_XPU:
            from triton.language.extra.intel.libdevice import (  # @manual
                fast_dividef,
                fast_dividef as my_fast_dividef,
            )

    def _triton_get_ast_equal_to_str(params):
        try:
            from triton.backends.compiler import AttrsDescriptor  # noqa: F401

            return f"'tt.equal_to': {params}"
        except ImportError:
            return f"equal_to_1={params}"

    # Define shared triton constants here.
    CONSTANT_C: tl.constexpr = 4
    STRING_CONSTANT_C: tl.constexpr = "CONSTANT_C"
    BOOL_CONSTANT_C: tl.constexpr = True
    FLOAT_CONSTANT_C = tl.constexpr(3.14)  # intentionally un-annotated


class KernelTests(torch._inductor.test_case.TestCase):
    @requires_gpu
    def test_triton_kernel_with_kernel_param(self):
        @triton.jit
        def pass_kernel(kernel):
            pass

        @torch.compile(backend="eager")
        def f(x):
            grid = (x.numel(),)
            pass_kernel[grid](kernel=x)

        t1 = torch.rand(5, device=GPU_TYPE)
        f(t1)
        # No need to assert anything, the goal is to make sure dynamo does
        # not crash

    @requires_gpu
    def test_triton_kernel_higher_order_func(self):
        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

        add_kernel_id = kernel_side_table.add_kernel(add_kernel)

        t1 = torch.rand(5, device=GPU_TYPE)
        t2 = torch.rand(5, device=GPU_TYPE)

        torch_add = t1 + t2

        # Test higher order function with mutation
        output = torch.zeros_like(t1)
        n_elements = output.numel()
        constant_args_idx = kernel_side_table.add_constant_args(
            {"n_elements": n_elements, "BLOCK_SIZE": 16}
        )
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        triton_kernel_wrapper_mutation(
            kernel_idx=add_kernel_id,
            constant_args_idx=constant_args_idx,
            grid=[grid],
            tma_descriptor_metadata={},
            kwargs={
                "in_ptr0": t1,
                "in_ptr1": t2,
                "out_ptr": output,
            },
        )
        self.assertEqual(output, torch_add)
        # Make sure it is modified
        self.assertNotEqual(output, torch.zeros_like(t1))

        # Test higher order function without mutation
        output = torch.zeros_like(t1)
        out_dict = triton_kernel_wrapper_functional(
            kernel_idx=add_kernel_id,
            constant_args_idx=constant_args_idx,
            grid=[grid],
            tma_descriptor_metadata={},
            kwargs={
                "in_ptr0": t1,
                "in_ptr1": t2,
                "out_ptr": output,
            },
            tensors_to_clone=["in_ptr0", "in_ptr1", "out_ptr"],
        )
        self.assertEqual(out_dict["out_ptr"], torch_add)
        # Make sure it is NOT modified
        self.assertEqual(output, torch.zeros_like(t1))

    @requires_gpu
    def test_triton_kernel_functionalize(self):
        from functorch import make_fx
        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
        from torch._subclasses.functional_tensor import (
            CppFunctionalizeAPI,
            FunctionalTensorMode,
            PythonFunctionalizeAPI,
        )

        kernel_side_table.reset_table()

        def f(x, output):
            out = triton_kernel_wrapper_functional(
                kernel_idx=kernel_side_table.add_kernel(mul2_kernel),
                constant_args_idx=kernel_side_table.add_constant_args(
                    {"n_elements": output.numel(), "BLOCK_SIZE": 16}
                ),
                grid=[(x.numel(),)],
                tma_descriptor_metadata={},
                kwargs={
                    "in_ptr0": x,
                    "out_ptr": output,
                },
                tensors_to_clone=["in_ptr0", "out_ptr"],
            )
            return out["out_ptr"]

        t1 = torch.rand(5, device=GPU_TYPE)
        t2 = torch.rand(5, device=GPU_TYPE)
        with FunctionalTensorMode():
            gm = make_fx(PythonFunctionalizeAPI().functionalize(f))(t1, t2)
        # Make sure t2 was not modified
        self.assertNotEqual(gm(t1, t2), t2)

        gm = make_fx(CppFunctionalizeAPI().functionalize(f))(t1, t2)
        # Make sure t2 was not modified
        self.assertNotEqual(gm(t1, t2), t2)

        gm = make_fx(torch.func.functionalize(f))(t1, t2)
        # Make sure t2 was not modified
        self.assertNotEqual(gm(t1, t2), t2)

        gm = make_fx(f, tracing_mode="fake")(t1, t2)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x_1, output_1):
    triton_kernel_wrapper_functional_proxy = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 0, constant_args_idx = 3, grid = [(5,)], tma_descriptor_metadata = {}, kwargs = {'in_ptr0': x_1, 'out_ptr': output_1}, tensors_to_clone = ['in_ptr0', 'out_ptr']);  x_1 = output_1 = None
    getitem = triton_kernel_wrapper_functional_proxy['in_ptr0'];  getitem = None
    getitem_1 = triton_kernel_wrapper_functional_proxy['out_ptr'];  triton_kernel_wrapper_functional_proxy = None
    return getitem_1""",
        )

    @requires_gpu
    def test_triton_kernel_mutation_type(self):
        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch._subclasses.functional_tensor import (
            FunctionalTensor,
            FunctionalTensorMode,
        )

        def prep():
            x = torch.ones(4, device=GPU_TYPE, requires_grad=True)
            with FunctionalTensorMode():
                x_func = FunctionalTensor.to_functional(x)
            self.assertTrue(torch._is_functional_tensor(x_func.elem))
            return x_func

        # normal mutation only
        with FakeTensorMode():
            x_func = prep()

            with FunctionalTensorMode():
                x_func.mul_(2)

            self.assertFalse(
                torch._functionalize_are_all_mutations_hidden_from_autograd(x_func.elem)
            )

        # triton kernel mutation only
        with FakeTensorMode():
            x_func = prep()

            with FunctionalTensorMode():
                triton_kernel_wrapper_mutation(
                    kernel_idx=kernel_side_table.add_kernel(mul2_inplace_kernel),
                    constant_args_idx=kernel_side_table.add_constant_args(
                        {"n_elements": x_func.numel(), "BLOCK_SIZE": 16}
                    ),
                    grid=[(x_func.numel(),)],
                    tma_descriptor_metadata={},
                    kwargs={
                        "ptr": x_func,
                    },
                )

            self.assertTrue(
                torch._functionalize_are_all_mutations_hidden_from_autograd(x_func.elem)
            )

        # normal mutation + triton kernel mutation
        with FakeTensorMode():
            x_func = prep()

            with FunctionalTensorMode():
                x_func.mul_(2)
                triton_kernel_wrapper_mutation(
                    kernel_idx=kernel_side_table.add_kernel(mul2_inplace_kernel),
                    constant_args_idx=kernel_side_table.add_constant_args(
                        {"n_elements": x_func.numel(), "BLOCK_SIZE": 16}
                    ),
                    grid=[(x_func.numel(),)],
                    tma_descriptor_metadata={},
                    kwargs={
                        "ptr": x_func,
                    },
                )

            self.assertFalse(
                torch._functionalize_are_all_mutations_hidden_from_autograd(x_func.elem)
            )

    @requires_gpu
    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_triton_kernel_with_views(self, dynamic, backend):
        def call_triton_take_view(x: torch.Tensor):
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            mul2_kernel[grid](x, output, n_elements, BLOCK_SIZE=16)
            return output

        def call_triton_return_view(x: torch.Tensor):
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            mul2_kernel[grid](x, output, n_elements, BLOCK_SIZE=16)
            return output.view(4, 4)

        t = torch.rand(4, 4, device=GPU_TYPE)
        t_view = t.view(16)

        compiled_func = torch.compile(
            call_triton_take_view, backend=backend, fullgraph=True, dynamic=dynamic
        )
        self.assertEqual(2 * t_view, compiled_func(t_view))
        self.assertEqual(2 * t, compiled_func(t_view).view(4, 4))

        compiled_func = torch.compile(
            call_triton_return_view, backend=backend, fullgraph=True, dynamic=dynamic
        )
        self.assertEqual(2 * t_view, compiled_func(t).view(16))
        self.assertEqual(2 * t, compiled_func(t))

    @requires_gpu
    def test_no_nan_kernels(self):
        @triton.jit
        def add_one_kernel(
            in_ptr0,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            output = x + 1
            tl.store(out_ptr + offsets, output, mask=mask)

        def add_one(x, out):
            n_elements = x.numel()
            add_one_kernel[(n_elements,)](x, out, n_elements, BLOCK_SIZE=4)

        class AddOne(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                out = torch.empty_like(x)
                add_one(x, out)
                ctx.save_for_backward(out)
                return out

            @staticmethod
            def backward(ctx, grad):
                (saved,) = ctx.saved_tensors
                out = torch.empty_like(grad)
                add_one(saved, out)
                return out

        @torch.compile
        def f(x):
            return AddOne.apply(x)

        log_stream, ctx = logs_to_string("torch._inductor.codecache", "output_code")

        x = torch.randn(3, requires_grad=True, device=GPU_TYPE)
        with ctx():
            y = f(x)

        output_code = "\n".join(log_stream.getvalue().strip().split("\n")[3:]).strip()
        self.assertTrue(len(output_code) > 0, msg="output code is not empty")
        self.assertEqual(output_code.count('float("nan")'), 0)
        self.assertEqual(output_code.count("float('nan')"), 0)

    @requires_gpu
    @common_utils.parametrize("grad_fn", [torch.no_grad, torch.enable_grad])
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_triton_kernel_with_grad_option(self, grad_fn, backend):
        def call_triton(x: torch.Tensor):
            with grad_fn():
                output = torch.zeros_like(x)
                n_elements = output.numel()
                grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
                mul2_kernel[grid](x, output, n_elements, BLOCK_SIZE=16)
                return output

        t = torch.rand(5, device=GPU_TYPE)
        compiled_func = torch.compile(call_triton, backend=backend, fullgraph=True)
        self.assertEqual(2 * t, compiled_func(t))

    @requires_gpu
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_triton_kernel_inner_triton_function(self, backend):
        def f(x: torch.Tensor):
            @triton.jit
            def pow2_kernel(
                in_ptr0,
                out_ptr,
                n_elements,
                BLOCK_SIZE: "tl.constexpr",
            ):
                pid = tl.program_id(axis=0)
                block_start = pid * BLOCK_SIZE
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                x = tl.load(in_ptr0 + offsets, mask=mask)
                output = x * x
                tl.store(out_ptr + offsets, output, mask=mask)

            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            pow2_kernel[grid](x, output, n_elements, BLOCK_SIZE=16)
            return output

        t = torch.rand(5, device=GPU_TYPE)

        compiled_func = torch.compile(f, backend=backend, fullgraph=True)
        # TODO(oulgen): NYI - Support this
        # self.assertEqual(t * t, compiled_func(t))

    @requires_gpu
    @common_utils.parametrize("grad", [False, True])
    @common_utils.parametrize("dynamic", [False, True])
    @patch.object(torch._inductor.config, "implicit_fallbacks", False)
    def test_triton_kernel_no_clones(self, grad, dynamic):
        from torch._inductor.utils import run_and_get_code

        def call_triton(x: torch.Tensor, y: torch.Tensor, output: torch.Tensor):
            n_elements = output.numel()

            tmp = torch.add(x, 1)
            grid = (x.numel(),)
            add_kernel.run(
                x, y, output, n_elements, warmup=False, grid=grid, BLOCK_SIZE=16
            )

            return output, tmp

        t1 = torch.rand(5, device=GPU_TYPE, requires_grad=grad)
        t2 = torch.rand(5, device=GPU_TYPE, requires_grad=grad)
        o1 = torch.zeros_like(t1, requires_grad=grad)

        torch_add = call_triton(t1, t2, o1)
        metrics.reset()
        o2 = torch.zeros_like(t1, requires_grad=grad)
        test, codes = run_and_get_code(
            torch.compile(call_triton, dynamic=dynamic), t1, t2, o2
        )
        if not grad:
            self.assertEqual(metrics.generated_kernel_count, 1)
        self.assertEqual(torch_add, test)
        # These two asserts are not optimal since it requires original aten
        # to be in the metadata, so there might be false negatives
        self.assertTrue("aten.copy" not in codes[0])
        self.assertTrue("aten.clone" not in codes[0])
        # The following checks that there are only the tensor output is in
        # the compiled graph
        if dynamic and grad:
            self.assertTrue("return (buf0, s0, )" in codes[0])
        else:
            self.assertTrue("return (buf0, )" in codes[0])

    @requires_gpu
    def test_triton_kernel_caching(self):
        from torch._inductor.utils import run_and_get_code

        def add_in_loop(
            x: torch.Tensor,
            y: torch.Tensor,
        ):
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            add_kernel_autotuned[grid](x, y, output, n_elements)
            return output

        def call_triton_add(
            x: torch.Tensor,
            y: torch.Tensor,
        ):
            for i in range(4):
                x = add_in_loop(x, y)
            return x

        t1 = torch.ones(5, device=GPU_TYPE)
        t2 = torch.ones(5, device=GPU_TYPE)

        test, (code,) = run_and_get_code(torch.compile(call_triton_add), t1, t2)
        self.assertEqual(test, 5 * torch.ones(5, device=GPU_TYPE))
        self.assertTrue("add_kernel_autotuned_1.run" not in code)

    @requires_gpu
    def test_triton_kernel_caching_duplicate(self):
        from torch._inductor.utils import run_and_get_code

        class C:
            @triton.jit
            def pass_kernel(
                in_ptr0,
                out_ptr,
                n_elements,
                BLOCK_SIZE: "tl.constexpr",
            ):
                pid = tl.program_id(axis=0)
                block_start = pid * BLOCK_SIZE
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                x = tl.load(in_ptr0 + offsets, mask=mask)
                tl.store(out_ptr + offsets, x, mask=mask)

        class D:
            @triton.jit
            def pass_kernel(
                in_ptr0,
                out_ptr,
                n_elements,
                BLOCK_SIZE: "tl.constexpr",
            ):
                pid = tl.program_id(axis=0)
                block_start = pid * BLOCK_SIZE
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                x = tl.load(in_ptr0 + offsets, mask=mask)
                tl.store(out_ptr + offsets, x, mask=mask)

        def call_triton(x: torch.Tensor):
            output1 = torch.zeros_like(x)
            output2 = torch.zeros_like(x)
            n_elements = output1.numel()
            grid = (n_elements,)
            C.pass_kernel[grid](x, output1, n_elements, BLOCK_SIZE=16)
            D.pass_kernel[grid](x, output2, n_elements, BLOCK_SIZE=16)
            return output1 + output2

        t = torch.ones(5, device=GPU_TYPE)
        test, (code,) = run_and_get_code(torch.compile(call_triton), t)
        # Make sure we emitted two kernels here
        self.assertTrue("pass_kernel_0.run" in code)
        self.assertTrue("pass_kernel_1.run" in code)

    @requires_gpu
    def test_triton_kernel_various_args(self):
        @triton.autotune(
            configs=[triton.Config({"BLOCK_SIZE": 128})],
            key=[],
        )
        @triton.jit
        def pass_kernel(
            out_ptr,
            n_elements,
            dummy_None,
            dummy_empty,
            dummy_float,
            BLOCK_SIZE: "tl.constexpr",
            RANDOM_SIZE: "tl.constexpr",
        ):
            pass

        @torch.compile
        def call_triton(output):
            n_elements = output.numel()
            grid = (n_elements,)
            pass_kernel[grid](
                output,
                n_elements,
                None,
                torch.empty_like(output),
                3.1415926,
                RANDOM_SIZE=0,
            )
            return output

        output = torch.randn(5, device=GPU_TYPE)
        # Make sure this does not crash
        call_triton(output)

    @requires_gpu
    @skipIfRocm
    def test_triton_kernel_dependancies(self):
        def call_triton(
            x: torch.Tensor,
            y: torch.Tensor,
        ):
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            add_kernel_autotuned[grid](x, y, output, n_elements)
            output2 = torch.zeros_like(output)
            add_kernel_autotuned[grid](output, y, output2, n_elements)
            output3 = torch.add(output2, 1)
            return output3

        t1 = torch.rand(5, device=GPU_TYPE)
        t2 = torch.rand(5, device=GPU_TYPE)
        torch_result = call_triton(t1, t2)
        compiled_result = torch.compile(call_triton)(t1, t2)
        self.assertEqual(torch_result, compiled_result)

    @requires_gpu
    def test_triton_kernel_reinplace_inplaceable_pass(self):
        def call_triton(
            x: torch.Tensor,
            y: torch.Tensor,
        ):
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            add_kernel_autotuned[grid](x, y, output, n_elements)
            add_kernel_autotuned[grid](output, x, output, n_elements)
            return output

        t1 = torch.rand(5, device=GPU_TYPE)
        t2 = torch.rand(5, device=GPU_TYPE)
        torch_result = call_triton(t1, t2)
        compiled_result = torch.compile(call_triton)(t1, t2)
        self.assertEqual(torch_result, compiled_result)

    @requires_gpu
    @common_utils.parametrize("grad", [False, True])
    def test_triton_kernel_multi_kernel(self, grad):
        @triton.jit
        def mul2_and_add_and_zero_negatives_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
            ACTIVATION: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            indirection_kernel(
                in_ptr0,
                in_ptr0,
                n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
                ACTIVATION="mul2_inplace_kernel",
            )
            indirection_kernel(
                in_ptr1,
                in_ptr1,
                n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
                ACTIVATION="mul2_inplace_kernel",
            )
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
            if ACTIVATION == "zero_negs":
                output = zero_negs(output)
            tl.store(out_ptr + offsets, output, mask=mask)

        @torch.compile
        def call_triton(
            x: torch.Tensor,
            y: torch.Tensor,
            xi: torch.Tensor,
            yi: torch.Tensor,
            output: torch.Tensor,
            outputi: torch.Tensor,
        ):
            n_elements = output.numel()

            grid = (x.numel(),)
            mul2_and_add_and_zero_negatives_kernel[grid](
                x, y, output, n_elements, BLOCK_SIZE=16, ACTIVATION="zero_negs"
            )
            mul2_and_add_and_zero_negatives_kernel[grid](
                xi, yi, outputi, n_elements, BLOCK_SIZE=16, ACTIVATION=None
            )

            return (output, outputi)

        t1 = torch.tensor(
            [-2.0, -1.0, 0.0, 1.0, 2.0], device=GPU_TYPE, requires_grad=grad
        )
        t2 = torch.tensor(
            [-2.0, -1.0, 0.0, 1.0, 2.0], device=GPU_TYPE, requires_grad=grad
        )
        float_result = 2 * t1 + 2 * t2
        float_result = float_result.where(float_result >= 0, 0.0)

        t1i = torch.randint(-2, 2, (5,), device=GPU_TYPE)
        t2i = torch.randint(-2, 2, (5,), device=GPU_TYPE)
        o = torch.zeros_like(t1, requires_grad=grad)
        oi = torch.zeros_like(t1i)
        int_result = 2 * t1i + 2 * t2i

        (result, resulti) = call_triton(t1, t2, t1i, t2i, o, oi)
        self.assertEqual(float_result, result)
        self.assertEqual(int_result, resulti)

    @requires_gpu
    @skipIfXpu
    @skipIfRocm
    def test_triton_kernel_constants(self):
        @triton.jit
        def mulC_kernel(
            in_ptr0,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
            CONSTANT_NAME: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            if CONSTANT_NAME == STRING_CONSTANT_C:
                output = CONSTANT_C * x
            if BOOL_CONSTANT_C:
                output *= CONSTANT_C
            tl.store(out_ptr + offsets, output, mask=mask)

        def call_triton(
            x: torch.Tensor,
        ):
            output = torch.zeros_like(x)
            n_elements = output.numel()

            grid = (x.numel(),)
            mulC_kernel[grid](
                x, output, n_elements, BLOCK_SIZE=16, CONSTANT_NAME="CONSTANT_C"
            )
            return output

        # Triton kernels capture global constants by their parse time value
        # not runtime value
        global CONSTANT_C
        prev_c = CONSTANT_C
        # If the behavior of triton kernels change, this test will fail
        CONSTANT_C = 10
        assert CONSTANT_C != prev_c

        t = torch.randn(5, device=GPU_TYPE)
        torch_result = call_triton(t)
        compiled_result = torch.compile(call_triton)(t)

        self.assertEqual(torch_result, compiled_result)

        # reset back
        CONSTANT_C = prev_c

    @requires_gpu
    @common_utils.parametrize("grad", [False, True])
    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    @common_utils.parametrize("grid_type", [1, 2, 3])
    def test_triton_kernel_autotune(self, grad, dynamic, backend, grid_type):
        def call_triton(x: torch.Tensor, y: torch.Tensor, output: torch.Tensor):
            n_elements = output.numel()

            def grid_fn(meta):
                return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            if grid_type == 1:
                grid = (n_elements,)
            elif grid_type == 2:
                grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            elif grid_type == 3:
                grid = grid_fn

            add_kernel_autotuned[grid](x, y, output, n_elements)
            return output

        t1 = torch.rand(256, device=GPU_TYPE, requires_grad=grad)
        t2 = torch.rand(256, device=GPU_TYPE, requires_grad=grad)
        output = torch.zeros_like(t1, requires_grad=grad)

        torch_add = call_triton(t1, t2, output)
        compiled_func = torch.compile(
            call_triton, backend=backend, fullgraph=True, dynamic=dynamic
        )

        output2 = torch.zeros_like(t1, requires_grad=grad)
        self.assertEqual(compiled_func(t1, t2, output2), torch_add)

    @requires_gpu
    @skipIfRocm  # https://github.com/pytorch/pytorch/actions/runs/10051552819/job/27782048305?pr=131431
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    @patch.object(
        torch._inductor.config, "unsafe_ignore_unsupported_triton_autotune_args", True
    )
    def test_triton_kernel_autotune_with_unsupported_args(self, backend):
        def call_triton(x: torch.Tensor, y: torch.Tensor):
            output = torch.zeros_like(x)
            n_elements = output.numel()
            add_kernel_autotuned_with_unsupported_args[(n_elements,)](
                x, y, output, n_elements
            )
            return output

        t1 = torch.rand(256, device=GPU_TYPE)
        t2 = torch.rand(256, device=GPU_TYPE)

        torch_add = call_triton(t1, t2)
        compiled_func = torch.compile(call_triton, backend=backend, fullgraph=True)
        compiled_add = compiled_func(t1, t2)
        self.assertEqual(compiled_add, torch_add)

    @requires_gpu
    @common_utils.parametrize("grad", [False, True])
    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    @common_utils.parametrize("grid_type", [1, 2, 3])
    def test_triton_kernel_2d_autotune(self, grad, dynamic, backend, grid_type):
        def call_triton(x: torch.Tensor, y: torch.Tensor, output: torch.Tensor):
            x_elements = output.size()[0]
            y_elements = output.size()[1]

            def grid_fn(meta):
                return (
                    triton.cdiv(x_elements, meta["BLOCK_SIZE_X"]),
                    triton.cdiv(y_elements, meta["BLOCK_SIZE_Y"]),
                )

            if grid_type == 1:
                grid = (x_elements, y_elements)
            elif grid_type == 2:
                grid = lambda meta: (
                    triton.cdiv(x_elements, meta["BLOCK_SIZE_X"]),
                    triton.cdiv(y_elements, meta["BLOCK_SIZE_Y"]),
                )
            elif grid_type == 3:
                grid = grid_fn

            add_kernel_2d_autotuned[grid](x, y, output, x_elements, y_elements)
            return output

        t1 = torch.rand((512, 256), device=GPU_TYPE, requires_grad=grad)
        t2 = torch.rand((512, 256), device=GPU_TYPE, requires_grad=grad)
        output = torch.zeros_like(t1, requires_grad=grad)

        torch_result = call_triton(t1, t2, output)
        compiled_func = torch.compile(
            call_triton, backend=backend, fullgraph=True, dynamic=dynamic
        )
        output2 = torch.zeros_like(t1, requires_grad=grad)
        self.assertEqual(compiled_func(t1, t2, output2), torch_result)

    @requires_gpu
    @common_utils.parametrize("dynamic", [False, True])
    def test_triton_kernel_tracing(self, dynamic):
        def call_triton_add(
            x: torch.Tensor,
            y: torch.Tensor,
            grid_type: int,
            num=1,
            positional=False,
            autotuned=False,
        ):
            output = torch.empty_like(x)
            n_elements = output.numel()

            def grid_fn(meta):
                return (triton.cdiv(num, meta["BLOCK_SIZE"]),)

            if grid_type == 0:
                grid = (x.numel(),)
            elif grid_type == 1:
                grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            elif grid_type == 2:
                grid = grid_fn
            else:
                grid = [x.numel()]

            if autotuned:
                capture_triton(add_kernel_autotuned)[grid](x, y, output, n_elements)
            else:
                if positional:
                    capture_triton(add_kernel)[grid](x, y, output, n_elements, 16)
                else:
                    capture_triton(add_kernel)[grid](
                        x, y, output, n_elements, BLOCK_SIZE=16
                    )

            return output

        t0 = torch.rand(5, device=GPU_TYPE, requires_grad=True)
        t1 = torch.rand(5, device=GPU_TYPE, requires_grad=True)
        t2 = torch.rand(5, device=GPU_TYPE, requires_grad=True)
        t3 = torch.rand(5, device=GPU_TYPE, requires_grad=True)
        torch_add = t2 + t3

        tests = [
            functools.partial(call_triton_add, grid_type=0),
            functools.partial(call_triton_add, grid_type=1),
            functools.partial(call_triton_add, grid_type=1, num=1, positional=True),
            functools.partial(call_triton_add, grid_type=2, num=200),
            functools.partial(call_triton_add, grid_type=3),
            functools.partial(call_triton_add, grid_type=0, autotuned=True),
            functools.partial(call_triton_add, grid_type=1, num=1, autotuned=True),
            functools.partial(call_triton_add, grid_type=2, num=200, autotuned=True),
            functools.partial(call_triton_add, grid_type=3, autotuned=True),
        ]
        from functorch import make_fx

        tracing_mode = "symbolic" if dynamic else "fake"

        for test in tests:
            gm = make_fx(test, tracing_mode=tracing_mode)(t0, t1)
            result = test(t2, t3)
            self.assertEqual(result, torch_add)

    @requires_gpu
    @common_utils.parametrize("grad", [False, True])
    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    @patch.object(torch._inductor.config, "implicit_fallbacks", False)
    def test_triton_kernel_native(self, grad, dynamic, backend):
        def call_triton_add(
            x: torch.Tensor,
            y: torch.Tensor,
            output: torch.Tensor,
            grid_type: int,
            num=1,
            positional=False,
        ):
            n_elements = output.numel()

            def grid_fn(meta):
                return (triton.cdiv(num, meta["BLOCK_SIZE"]),)

            if grid_type == 0:
                grid = (x.numel(),)
            elif grid_type == 1:
                grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            else:
                grid = grid_fn

            if positional:
                add_kernel[grid](x, y, output, n_elements, 16)
            else:
                add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)

            return output

        t1 = torch.rand(5, device=GPU_TYPE, requires_grad=grad)
        t2 = torch.rand(5, device=GPU_TYPE, requires_grad=grad)
        o1 = torch.zeros_like(t1, requires_grad=grad)

        torch_add = t1 + t2

        # No Dynamo -- Make sure triton kernel works
        self.assertEqual(call_triton_add(t1, t2, o1, 1), torch_add)
        # No Dynamo -- Make sure triton kernel works (with positional BLOCK_SIZE)
        o2 = torch.zeros_like(t1, requires_grad=grad)
        self.assertEqual(call_triton_add(t1, t2, o2, 1, True), torch_add)

        # With Dynamo
        compiled_func = torch.compile(
            call_triton_add, backend=backend, fullgraph=True, dynamic=dynamic
        )
        # With simple kernel
        o3 = torch.zeros_like(t1, requires_grad=grad)
        self.assertEqual(compiled_func(t1, t2, o3, 0), torch_add)
        # With lambda kernel
        o4 = torch.zeros_like(t1, requires_grad=grad)
        self.assertEqual(compiled_func(t1, t2, o4, 1), torch_add)
        # With lambda kernel (with positional BLOCK_SIZE)
        o5 = torch.zeros_like(t1, requires_grad=grad)
        self.assertEqual(compiled_func(t1, t2, o5, 1, 1, True), torch_add)
        # With user defined function kernel
        o6 = torch.zeros_like(t1, requires_grad=grad)
        self.assertEqual(compiled_func(t1, t2, o6, 2, 200), torch_add)

    @requires_gpu
    def test_triton_kernel_mutation_not_mark_dirty(self):
        @torch.compile
        def f(x):
            n_elements = x.numel()
            add_kernel[(n_elements,)](x, x, x, n_elements, 16)
            return x

        x = torch.randn(5, device=GPU_TYPE, requires_grad=True)
        x_cloned = x.clone()
        out = x_cloned.sin()
        f(x_cloned)
        out.sum().backward()

    @requires_gpu
    @patch.object(torch._inductor.config, "allow_buffer_reuse", True)
    def test_triton_kernel_inputs_buffer_reuse(self):
        def _mul2(x):
            y = torch.empty_like(x)
            mul2_kernel[(10,)](
                in_ptr0=x,
                out_ptr=y,
                n_elements=x.numel(),
                BLOCK_SIZE=1,
            )
            return y

        @torch.compile
        def f(x):
            for _ in range(4):
                # The output of one kernel is the input to the next kernel, but
                # at some point we should re-use buffers not allocate new ones.
                x = _mul2(x)
            return x + 1

        x = torch.randn(10, device=GPU_TYPE, dtype=torch.float32)
        eager_out = f(x)
        compiled_out, (code,) = run_and_get_code(torch.compile(f), x)
        self.assertEqual(compiled_out, eager_out)

        # Check that we're allocating the minimal # of buffers.
        code_string = f"empty_strided_{GPU_TYPE}((10, ), (1, ), torch.float32)"

        num_bufs_allocated = code.count(code_string)
        self.assertEqual(num_bufs_allocated, 2)

        # Check we're re-using buffers if not allocating.
        num_bufs_reused = code.count("# reuse")
        self.assertEqual(num_bufs_reused, 3)

    @requires_gpu
    def test_triton_kernel_matmul_tracking(self):
        @triton.jit
        def ones_kernel(x_ptr, n_elements, BLOCK_SIZE: "tl.constexpr"):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = 1.0
            tl.store(x_ptr + offsets, x, mask=mask)

        @torch.compile
        def f(x):
            out = torch.zeros_like(x)
            ones_kernel[(4,)](out, 16, BLOCK_SIZE=16)
            return torch.mm(out, x) + 10

        x = torch.randn(4, 4, device=GPU_TYPE)
        torch_out = f(x)
        python_out = torch.mm(torch.ones(4, 4, device=GPU_TYPE), x) + 10
        self.assertEqual(torch_out, python_out)

    @requires_gpu
    def test_triton_kernel_strided_input(self):
        def f(inp):
            # left has strides [256, 1]
            left, right = torch.split(inp, [128, 128], dim=1)
            out = torch.empty_like(left)
            X_BLOCK_SIZE, Y_BLOCK_SIZE = 32, 16
            grid = (left.size(1) // X_BLOCK_SIZE, left.size(0) // Y_BLOCK_SIZE)
            double_strided_kernel[grid](
                in_ptr=left,
                out_ptr=out,
                in_y_stride=left.stride(0),
                out_y_stride=out.stride(0),
                X_BLOCK_SIZE=X_BLOCK_SIZE,
                Y_BLOCK_SIZE=Y_BLOCK_SIZE,
            )
            return out

        inp = torch.randn(64, 256, device=GPU_TYPE)

        eager_out = f(inp)
        compiled_out = torch.compile(f)(inp)
        self.assertEqual(compiled_out, eager_out)

    @torch._inductor.config.patch(
        triton_kernel_default_layout_constraint="needs_fixed_stride_order"
    )
    @requires_gpu
    def test_layout_constraint_needs_fixed_stride_order(self):
        # Construct a custom op whose output strides are (1, 2)
        @torch.library.custom_op("mylib::weird_op_with_lowering", mutates_args={})
        def weird_op_with_lowering(x: torch.Tensor) -> torch.Tensor:
            return torch.empty_strided((2, 2), (1, 2), dtype=x.dtype, device=x.device)

        @weird_op_with_lowering.register_fake
        def _(x):
            return torch.empty_strided((2, 2), (1, 2), dtype=x.dtype, device=x.device)

        # The lowering for the custom op produces output strides (2, 1).
        from torch._inductor.lowering import empty_strided, register_lowering

        @register_lowering(torch.ops.mylib.weird_op_with_lowering)
        def _(x):
            return empty_strided(
                x.shape, (2, 1), dtype=x.dtype, device=torch.device(GPU_TYPE, 0)
            )

        # Triton kernel that has different behavior depending on the input strides.
        @triton.jit
        def kernel(
            in_ptr0,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            output = offsets
            tl.store(out_ptr + offsets, output, mask=mask)

        def arange_out(x, out):
            n_elements = x.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            kernel[grid](x, out, n_elements, BLOCK_SIZE=4)

        def f(x):
            y = weird_op_with_lowering(x)
            # Inductor lowering will decide that y is better having strides (2, 1).
            # This is different from the strides at tracing time (1, 2).
            # Under the "needs_fixed_stride_order" config, inductor will coerce
            # y to have strides (1, 2) before passing it to arange_out.
            # If it doesn't, then the result will be different from eager mode.
            arange_out(x, y)
            return x + y

        x = torch.randn(2, 2, device=GPU_TYPE)
        eager_out = f(x)

        compiled_inductor_f = torch.compile(f, backend="inductor", fullgraph=True)
        compiled_inductor_out = compiled_inductor_f(x)
        self.assertEqual(compiled_inductor_out, eager_out)

    @requires_gpu
    def test_triton_kernel_strided_input_nonzero_offset(self):
        def f(inp):
            # right has strides [256, 1] and storage offset 128
            left, right = torch.split(inp, [128, 128], dim=1)
            out = torch.empty_like(right)
            X_BLOCK_SIZE, Y_BLOCK_SIZE = 32, 16
            grid = (right.size(1) // X_BLOCK_SIZE, right.size(0) // Y_BLOCK_SIZE)
            double_strided_kernel[grid](
                in_ptr=right,
                out_ptr=out,
                in_y_stride=right.stride(0),
                out_y_stride=out.stride(0),
                X_BLOCK_SIZE=X_BLOCK_SIZE,
                Y_BLOCK_SIZE=Y_BLOCK_SIZE,
            )
            return out

        inp = torch.randn(64, 256, device=GPU_TYPE)

        eager_out = f(inp)
        compiled_out = torch.compile(f)(inp)
        self.assertEqual(compiled_out, eager_out)

    @requires_gpu
    def test_triton_kernel_slice_and_view_input(self):
        def f(inp):
            # left has strides [256, 1]
            left = inp[:, :128]
            left = left.view(64, 4, 32)
            out = torch.empty_like(left)
            X_BLOCK_SIZE, Y_BLOCK_SIZE = 32, 16
            grid = (
                (left.size(1) * left.size(2)) // X_BLOCK_SIZE,
                left.size(0) // Y_BLOCK_SIZE,
            )
            double_strided_kernel[grid](
                in_ptr=left,
                out_ptr=out,
                in_y_stride=left.stride(0),
                out_y_stride=out.stride(0),
                X_BLOCK_SIZE=X_BLOCK_SIZE,
                Y_BLOCK_SIZE=Y_BLOCK_SIZE,
            )
            return out + left

        inp = torch.randn(64, 256, device=GPU_TYPE)

        eager_out = f(inp)
        compiled_out = torch.compile(f)(inp)
        self.assertEqual(compiled_out, eager_out)

    @requires_gpu
    def test_triton_kernel_fallback(self):
        def f(x, y):
            out = torch.zeros_like(x)
            out2 = torch.zeros_like(x)
            # torch.mm is ExternKernelOut
            add_kernel[(4,)](x, torch.mm(x, y), out, 4, 16)
            # torch.sort creates fallback kernel and hence MultiOutput
            add_kernel[(4,)](x, torch.sort(y).values, out, 4, 16)
            return out, out2

        x = torch.randn(4, 4, device=GPU_TYPE)
        y = torch.randn(4, 4, device=GPU_TYPE)
        eager_out = f(x, y)
        compiled_out = torch.compile(f)(x, y)
        self.assertEqual(compiled_out, eager_out)

    @requires_gpu
    def test_triton_kernel_out_of_order(self):
        @triton.jit
        def add_kernel(
            in_ptr0,
            in_ptr1,
            BLOCK_SIZE: "tl.constexpr",
            out_ptr,
            n_elements,
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)

        def f(x, y):
            out = torch.zeros_like(x)
            n_elements = x.numel()
            add_kernel[(n_elements,)](x, y, 4, out, n_elements)
            return out

        x = torch.randn(4, device=GPU_TYPE)
        y = torch.randn(4, device=GPU_TYPE)
        eager_out = f(x, y)
        compiled_out = torch.compile(f)(x, y)
        self.assertEqual(compiled_out, eager_out)

    @requires_gpu
    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_triton_kernel_unbacked_shape_tensor(self, backend):
        @triton.jit
        def square(
            in_ptr,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr + offsets, mask=mask)
            output = x * x
            tl.store(out_ptr + offsets, output, mask=mask)

        def f(x):
            x = x[x > 2]
            n_elements = x.numel()
            output = torch.zeros_like(x)
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            square[grid](x, output, n_elements, BLOCK_SIZE=16)
            return output

        x = torch.randn(4, device=GPU_TYPE)
        eager_out = f(x)
        compiled_out = torch.compile(f, fullgraph=True, backend=backend)(x)
        self.assertEqual(compiled_out, eager_out)

    @requires_gpu
    @common_utils.parametrize("dynamic", [False, True])
    def test_triton_kernel_equal_to_1_arg(self, dynamic):
        @triton.jit
        def add_kernel_half_n_elements(
            in_ptr0,
            in_ptr1,
            out_ptr,
            half_n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < half_n_elements * 2
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)

        def f(x, y):
            out = torch.empty_like(x)
            half_n_elements = x.numel() // 2
            add_kernel_half_n_elements[(half_n_elements,)](
                x, y, out, half_n_elements, BLOCK_SIZE=16
            )
            return out

        x = torch.randn(2, device=GPU_TYPE)
        y = torch.randn(2, device=GPU_TYPE)
        eager_out = f(x, y)
        compiled_out, sources = run_and_get_code(
            torch.compile(f, dynamic=dynamic), x, y
        )

        if dynamic:
            # when half_n_elements passed to the Triton kernel is
            # dynamic, equal_to_1 specializaiton can't be enforced
            self.assertTrue(_triton_get_ast_equal_to_str(()) in sources[0])
        else:
            self.assertTrue(_triton_get_ast_equal_to_str((3,)) in sources[0])
        self.assertEqual(compiled_out, eager_out)

    @requires_gpu
    @common_utils.parametrize("dynamic", [False, True])
    def test_triton_kernel_equal_to_1_float_arg(self, dynamic):
        def f(x, y):
            out = torch.empty_like(x)
            n_elements = x.numel()
            scaling_factor = (n_elements**0) / 1.0
            add_kernel_with_scaling[(n_elements,)](
                x,
                y,
                out,
                n_elements,
                scaling_factor,
                BLOCK_SIZE=16,
            )
            return out

        x = torch.randn(2, device=GPU_TYPE)
        y = torch.randn(2, device=GPU_TYPE)
        eager_out = f(x, y)
        compiled_out, sources = run_and_get_code(
            torch.compile(f, dynamic=dynamic), x, y
        )

        # float 1.0 (both literal or symbolic)
        # should not be added to equal_to_1
        self.assertTrue(_triton_get_ast_equal_to_str(()) in sources[0])
        self.assertEqual(compiled_out, eager_out)

    @requires_gpu
    @skipIfRocm
    def test_triton_kernel_with_imported_symbol(self):
        @triton.jit
        def add_kernel_with_imported_symbol(
            in_ptr,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr + offsets, mask=mask)
            output = fast_dividef(x, 3.14)
            tl.store(out_ptr + offsets, output, mask=mask)

        def f(x):
            out = torch.empty_like(x)
            n_elements = x.numel()
            add_kernel_with_imported_symbol[(n_elements,)](
                x, out, n_elements, BLOCK_SIZE=16
            )
            return out

        x = torch.randn(4, device=GPU_TYPE)
        eager_out = f(x)
        compiled_out = torch.compile(f)(x)

        self.assertEqual(compiled_out, eager_out)

    @requires_gpu
    @skipIfRocm
    def test_triton_kernel_with_imported_symbol_with_custom_name(self):
        @triton.jit
        def add_kernel_with_imported_symbol(
            in_ptr,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr + offsets, mask=mask)
            output = my_fast_dividef(x, 3.14)
            tl.store(out_ptr + offsets, output, mask=mask)

        def f(x):
            out = torch.empty_like(x)
            n_elements = x.numel()
            add_kernel_with_imported_symbol[(n_elements,)](
                x, out, n_elements, BLOCK_SIZE=16
            )
            return out

        x = torch.randn(4, device=GPU_TYPE)
        eager_out = f(x)
        compiled_out = torch.compile(f)(x)

        self.assertEqual(compiled_out, eager_out)

    @requires_gpu
    @common_utils.parametrize("size", [4, 16])
    @common_utils.parametrize("dynamic", [False, True])
    def test_triton_kernel_different_shapes(self, size, dynamic):
        from torch._inductor.utils import run_and_get_code

        def f(x, y, xx, yy):
            n_elements = x.numel()
            output_1 = torch.zeros_like(x)
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            add_kernel[grid](x, y, output_1, n_elements, BLOCK_SIZE=4)

            n_elements = xx.numel()
            output_2 = torch.zeros_like(xx)
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            add_kernel[grid](xx, yy, output_2, n_elements, BLOCK_SIZE=4)

            return output_1, output_2

        x = torch.rand(size, device=GPU_TYPE)
        y = torch.rand(size, device=GPU_TYPE)
        xx = torch.rand(size, size, device=GPU_TYPE)
        yy = torch.rand(size, size, device=GPU_TYPE)
        args = [x, y, xx, yy]

        eager_out = f(*args)
        compiled_out, (code,) = run_and_get_code(
            torch.compile(f, fullgraph=True, dynamic=dynamic, backend="inductor"), *args
        )
        if size == 4 and not dynamic:
            # Produce 2 kernels due to divisibility
            self.assertTrue("add_kernel_0.run" in code)
            self.assertTrue("add_kernel_1.run" in code)
        else:
            # size == 16 or dynamic
            # Only one kernel
            self.assertTrue("add_kernel_0.run" in code)
            self.assertTrue("add_kernel_1.run" not in code)

        self.assertEqual(compiled_out, eager_out)

    @requires_gpu
    def test_triton_kernel_reset_to_zero(self):
        @triton.autotune(
            configs=[
                triton.Config({"BLOCK_SIZE": 128}, num_stages=3, num_warps=8),
                triton.Config({"BLOCK_SIZE": 64}, num_stages=3, num_warps=8),
            ],
            key=["n_elements"],
            reset_to_zero=["out_ptr"],
        )
        @triton.jit
        def add_kernel_autotuned_reset(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)

        @torch.compile(fullgraph=True)
        def f(x, y):
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            add_kernel_autotuned_reset[grid](x, y, output, n_elements)
            return output

        x = torch.randn(4, device=GPU_TYPE)
        msg = "Only configs, keys, and restore_value are supported for triton.autotune"
        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, msg):
            f(x, x)

    @requires_gpu
    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_triton_kernel_triton_dtype(self, dynamic, backend):
        @triton.jit
        def add_kernel_with_dtype(
            in_ptr0,
            in_ptr1,
            out_ptr,
            dtype: "tl.constexpr",
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask).to(dtype)
            y = tl.load(in_ptr1 + offsets, mask=mask).to(dtype)
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)

        def f(x, y, dtype_torch, dtype_triton):
            output = torch.zeros_like(x).to(dtype=dtype_torch)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            add_kernel_with_dtype[grid](
                x, y, output, dtype_triton, n_elements, BLOCK_SIZE=4
            )
            return output

        x = torch.randn(4, device=GPU_TYPE)
        y = torch.randn(4, device=GPU_TYPE)
        args_list = (
            [x, y, torch.float32, tl.float32],
            [x, y, torch.bfloat16, tl.bfloat16],
        )
        for args in args_list:
            eager_out = f(*args)
            compiled_out = torch.compile(
                f, fullgraph=True, backend=backend, dynamic=dynamic
            )(*args)
            self.assertEqual(compiled_out, eager_out)

    @requires_gpu
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_triton_kernel_special_kwargs_with_autotune(self, backend):
        @triton.autotune(
            configs=[
                triton.Config({"BLOCK_SIZE": 128}),
                triton.Config({"BLOCK_SIZE": 64}),
            ],
            key=["n_elements"],
        )
        @triton.jit
        def add_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)

        @torch.compile(fullgraph=True, backend=backend)
        def f(x, y):
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            add_kernel[grid](
                x,
                y,
                output,
                n_elements,
                num_warps=8,
                num_stages=3,
            )
            return output

        x = torch.randn(4, device=GPU_TYPE)
        f(x, x)

    @requires_gpu
    @common_utils.parametrize("autotune", [False, True])
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_triton_kernel_special_params(self, autotune, backend):
        @triton.jit
        def special_params_kernel(
            in_ptr,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
            num_warps: "tl.constexpr",
            num_stages: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr + offsets, mask=mask)
            output = x * num_stages + num_warps
            tl.store(out_ptr + offsets, output, mask=mask)

        NUM_WARPS = 4
        NUM_STAGES = 3

        if autotune:
            special_params_kernel = triton.autotune(
                configs=[
                    triton.Config(
                        {"BLOCK_SIZE": 128},
                        num_stages=NUM_STAGES,
                        num_warps=NUM_WARPS,
                    ),
                    triton.Config(
                        {"BLOCK_SIZE": 64},
                        num_stages=NUM_STAGES,
                        num_warps=NUM_WARPS,
                    ),
                ],
                key=["n_elements"],
            )(special_params_kernel)
            kwargs = {}
        else:
            kwargs = {
                "BLOCK_SIZE": 128,
                "num_stages": NUM_STAGES,
                "num_warps": NUM_WARPS,
            }

        def f(x):
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            special_params_kernel[grid](
                x,
                output,
                n_elements,
                **kwargs,
            )
            return output

        x = torch.randn(4, device=GPU_TYPE)
        eager_out = f(x)
        compiled_out = torch.compile(f, fullgraph=True, backend=backend)(x)
        expected_out = x * NUM_STAGES + NUM_WARPS
        self.assertEqual(eager_out, expected_out)
        self.assertEqual(compiled_out, expected_out)

    @requires_gpu
    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_triton_kernel_multiple_outputs(self, dynamic, backend):
        @triton.jit
        def add_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            out_ptr2,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)
            tl.store(out_ptr2 + offsets, output + 1, mask=mask)

        @torch.compile(fullgraph=True, backend=backend, dynamic=dynamic)
        def f(x, y, z):
            output = torch.empty_like(x)
            output2 = torch.empty_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            add_kernel[grid](x, y, output, output2, n_elements, BLOCK_SIZE=16)
            # The z return is intentional: we're testing training
            return output, output2, z**2

        x = torch.randn(3, requires_grad=True, device=GPU_TYPE)
        y = torch.randn(3, requires_grad=True, device=GPU_TYPE)
        z = torch.randn(3, requires_grad=True, device=GPU_TYPE)
        out, out2, out3 = f(x, y, z)
        self.assertEqual(out, x + y)
        self.assertEqual(out2, x + y + 1)
        self.assertEqual(out3, z**2)

    @requires_gpu
    @unittest.skipIf(not has_triton_tma(), "requires Triton TMA support")
    @common_utils.parametrize("dynamic", [False, True])
    def test_tma_capture_and_functionalize(self, dynamic):
        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

        kernel_side_table.reset_table()

        def f(a, b):
            BLOCK_SIZE = 256
            out = torch.zeros_like(a)
            n_elements = out.numel()

            desc_a, desc_b, desc_out = (
                triton.tools.experimental_descriptor.create_1d_tma_descriptor(
                    t.data_ptr(),
                    n_elements,
                    BLOCK_SIZE,
                    t.element_size(),
                )
                for t in (a, b, out)
            )

            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            add_kernel_with_tma_1d[grid](
                desc_a,
                desc_b,
                desc_out,
                BLOCK_SIZE=BLOCK_SIZE,
            )

            return out

        a = torch.randn(301, device=GPU_TYPE)
        b = torch.randn(301, device=GPU_TYPE)

        backend = torch._dynamo.testing.AotEagerAndRecordGraphs()
        torch.compile(
            f,
            fullgraph=True,
            backend=backend,
            dynamic=dynamic,
        )(a, b)

        if dynamic:
            self.assertExpectedInline(
                backend.fw_graphs[0].code.strip(),
                """\
def forward(self, arg0_1, arg1_1, arg2_1):
    zeros_like = torch.ops.aten.zeros_like.default(arg1_1, pin_memory = False)
    add_2 = arg0_1 + 256
    sub_1 = add_2 - 1;  add_2 = None
    floordiv = sub_1 // 256;  sub_1 = None
    triton_kernel_wrapper_functional_proxy = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 0, constant_args_idx = 0, grid = [(floordiv, 1, 1)], tma_descriptor_metadata = {'in_desc_ptr0': ([arg0_1], [256], 4), 'in_desc_ptr1': ([arg0_1], [256], 4), 'out_desc_ptr': ([arg0_1], [256], 4)}, kwargs = {'in_desc_ptr0': arg1_1, 'in_desc_ptr1': arg2_1, 'out_desc_ptr': zeros_like}, tensors_to_clone = ['out_desc_ptr']);  floordiv = arg0_1 = arg1_1 = arg2_1 = zeros_like = None
    getitem = triton_kernel_wrapper_functional_proxy['out_desc_ptr'];  triton_kernel_wrapper_functional_proxy = None
    return (getitem,)""",
            )
        else:
            self.assertExpectedInline(
                backend.fw_graphs[0].code.strip(),
                """\
def forward(self, arg0_1, arg1_1):
    zeros_like = torch.ops.aten.zeros_like.default(arg0_1, pin_memory = False)
    triton_kernel_wrapper_functional_proxy = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 0, constant_args_idx = 0, grid = [(2, 1, 1)], tma_descriptor_metadata = {'in_desc_ptr0': ([301], [256], 4), 'in_desc_ptr1': ([301], [256], 4), 'out_desc_ptr': ([301], [256], 4)}, kwargs = {'in_desc_ptr0': arg0_1, 'in_desc_ptr1': arg1_1, 'out_desc_ptr': zeros_like}, tensors_to_clone = ['out_desc_ptr']);  arg0_1 = arg1_1 = zeros_like = None
    getitem = triton_kernel_wrapper_functional_proxy['out_desc_ptr'];  triton_kernel_wrapper_functional_proxy = None
    return (getitem,)""",
            )

    @requires_gpu
    @unittest.skipIf(not has_triton_tma(), "requires Triton TMA support")
    @common_utils.parametrize("after_data_ptr", [False, True])
    @common_utils.parametrize("after_create_desc", [False, True])
    def test_tma_graph_breaks(self, after_data_ptr, after_create_desc):
        def f(a, b):
            BLOCK_SIZE = 256
            out = torch.zeros_like(a)
            n_elements = out.numel()

            ptrs = [t.data_ptr() for t in (a, b, out)]

            if after_data_ptr:
                torch._dynamo.graph_break()

            descs = [
                triton.tools.experimental_descriptor.create_1d_tma_descriptor(
                    ptr,
                    n_elements,
                    BLOCK_SIZE,
                    t.element_size(),
                )
                for ptr in ptrs
            ]

            if after_create_desc:
                torch._dynamo.graph_break()

            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            add_kernel_with_tma_1d[grid](
                *descs,
                BLOCK_SIZE=BLOCK_SIZE,
            )

            return out

        a = torch.randn(301, device=GPU_TYPE)
        b = torch.randn(301, device=GPU_TYPE)

        expected_out = a + b
        eager_out = f(a, b)
        compiled_out = torch.compile(
            f,
            fullgraph=False,
            backend="eager",
            dynamic=False,
        )(a, b)

        self.assertEqual(eager_out, expected_out)
        self.assertEqual(compiled_out, expected_out)

    @requires_gpu
    @unittest.skipIf(not has_triton_tma(), "requires Triton TMA support")
    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_tma_descriptor_1d(self, dynamic, backend):
        def f(a, b):
            BLOCK_SIZE = 256
            out = torch.zeros_like(a)
            n_elements = out.numel()

            desc_a, desc_b, desc_out = (
                triton.tools.experimental_descriptor.create_1d_tma_descriptor(
                    t.data_ptr(),
                    n_elements,
                    BLOCK_SIZE,
                    t.element_size(),
                )
                for t in (a, b, out)
            )

            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            add_kernel_with_tma_1d[grid](
                desc_a,
                desc_b,
                desc_out,
                BLOCK_SIZE=BLOCK_SIZE,
            )

            return out

        a = torch.randn(301, device=GPU_TYPE)
        b = torch.randn(301, device=GPU_TYPE)

        expected_out = a + b
        eager_out = f(a, b)
        compiled_out = torch.compile(
            f,
            fullgraph=True,
            backend=backend,
            dynamic=dynamic,
        )(a, b)

        self.assertEqual(eager_out, expected_out)
        self.assertEqual(compiled_out, expected_out)

    @requires_gpu
    @unittest.skipIf(not has_triton_tma(), "requires Triton TMA support")
    def test_tma_descriptor_dedup(self):
        def f(a):
            BLOCK_SIZE = 256
            out = torch.zeros_like(a)
            n_elements = out.numel()

            desc_a, desc_out = (
                triton.tools.experimental_descriptor.create_1d_tma_descriptor(
                    t.data_ptr(),
                    n_elements,
                    BLOCK_SIZE,
                    t.element_size(),
                )
                for t in (a, out)
            )

            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            add_kernel_with_tma_1d[grid](
                desc_a,
                desc_a,
                desc_out,
                BLOCK_SIZE=BLOCK_SIZE,
            )

            return out

        a = torch.randn(301, device=GPU_TYPE)

        expected_out = a + a
        eager_out = f(a)
        compiled_out, (code,) = run_and_get_code(
            torch.compile(
                f,
                fullgraph=True,
                backend="inductor",
                dynamic=True,
            ),
            a,
        )

        self.assertEqual(eager_out, expected_out)
        self.assertEqual(compiled_out, expected_out)

        # 2 calls: one for two inputs (dedupped), one for the output
        self.assertEqual(code.count("create_1d_tma_descriptor("), 2)

    @requires_gpu
    @unittest.skipIf(not has_triton_tma(), "requires Triton TMA support")
    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("backend", ["eager", "aot_eager"])
    def test_tma_descriptor_2d(self, dynamic, backend):
        def f(a, b):
            BLOCK_SIZE_X = 16
            BLOCK_SIZE_Y = 32
            out = torch.zeros_like(a)
            x_size, y_size = out.size()

            desc_a, desc_b, desc_out = (
                triton.tools.experimental_descriptor.create_2d_tma_descriptor(
                    t.data_ptr(),
                    x_size,
                    y_size,
                    BLOCK_SIZE_X,
                    BLOCK_SIZE_Y,
                    t.element_size(),
                )
                for t in (a, b, out)
            )

            grid = lambda meta: (
                triton.cdiv(x_size, meta["BLOCK_SIZE_X"]),
                triton.cdiv(y_size, meta["BLOCK_SIZE_Y"]),
            )
            add_kernel_with_tma_2d[grid](
                desc_a,
                desc_b,
                desc_out,
                BLOCK_SIZE_X=BLOCK_SIZE_X,
                BLOCK_SIZE_Y=BLOCK_SIZE_Y,
            )

            return out

        a = torch.randn((25, 16), device=GPU_TYPE)
        b = torch.randn((25, 16), device=GPU_TYPE)

        expected_out = a + b
        eager_out = f(a, b)
        compiled_out = torch.compile(
            f,
            fullgraph=True,
            backend=backend,
            dynamic=dynamic,
        )(a, b)

        self.assertEqual(eager_out, expected_out)
        self.assertEqual(compiled_out, expected_out)

    @requires_gpu
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_triton_kernel_num_ctas(self, backend):
        @triton.jit
        def kernel(X):
            return

        @torch.compile(backend=backend)
        def f(x):
            kernel[(1,)](x, num_ctas=1)
            kernel.run(x, num_ctas=1, grid=(1,), warmup=False)
            return x

        x = torch.randn(4, device=GPU_TYPE)
        f(x)

    @requires_gpu
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_triton_kernel_special_kwargs_without_autotune(self, backend):
        @triton.jit
        def add_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)

        @torch.compile(fullgraph=True, backend=backend)
        def f(x, y):
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            add_kernel[grid](
                x,
                y,
                output,
                n_elements,
                BLOCK_SIZE=128,
                num_warps=8,
                num_stages=3,
            )
            return output

        x = torch.randn(4, device=GPU_TYPE)
        f(x, x)

    @requires_gpu
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    @common_utils.parametrize("autotune_at_compile_time", [True, False])
    def test_triton_kernel_restore_value(self, backend, autotune_at_compile_time):
        if autotune_at_compile_time and backend != "inductor":
            raise unittest.SkipTest("compile-time autotuning only exists in inductor")

        @triton.autotune(
            configs=[
                triton.Config({"BLOCK_SIZE": 16}, num_stages=3, num_warps=8),
                triton.Config({"BLOCK_SIZE": 32}, num_stages=3, num_warps=8),
            ],
            key=[],
            restore_value=["in_ptr0"],
        )
        @triton.jit
        def increment_kernel(
            in_ptr0,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            output = x + 1
            tl.store(in_ptr0 + offsets, output, mask=mask)

        @torch.compile(fullgraph=True, backend=backend)
        def f(x):
            n_elements = x.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            increment_kernel[grid](x, n_elements=n_elements)
            return x

        x = torch.rand(4, device=GPU_TYPE)
        prev = x.clone()

        with torch._inductor.config.patch(
            {"triton.autotune_at_compile_time": autotune_at_compile_time}
        ):
            f(x)

        # make sure x was restored after autotuning
        torch.testing.assert_close(x, prev + 1)

    @requires_gpu
    @parametrize("dtype", (torch.float16, torch.float32, torch.float64))
    def test_triton_kernel_float64_constant(self, dtype):
        def f(x):
            return x * (0.12 * x.shape[0])

        x = torch.ones(200, device=GPU_TYPE, dtype=dtype)

        eager_out = f(x)
        compiled_out = torch.compile(f, dynamic=True)(x)
        self.assertEqual(compiled_out, eager_out)

    # TODO enable this test case on XPU.
    @requires_cuda
    @parametrize("cfg", ["normal", "cpp_wrapper"])
    def test_triton_kernel_dtype_view(self, cfg):
        # https://github.com/pytorch/pytorch/issues/136159
        if cfg == "normal":
            config_kwargs = {"cpp_wrapper": False}
        elif cfg == "cpp_wrapper":
            config_kwargs = {"cpp_wrapper": True}

        with torch._inductor.config.patch(**config_kwargs):

            @triton.jit
            def _triton_kernel(out_ptr, numel, BLOCK_SIZE: tl.constexpr):
                pid = tl.program_id(0)
                offsets = BLOCK_SIZE * pid + tl.arange(0, BLOCK_SIZE)
                mask = offsets < numel
                ones = tl.full((BLOCK_SIZE,), 1, tl.float16)
                tl.store(out_ptr + offsets, ones, mask)

            def fn(x):
                buf = torch.empty(x.shape, device=x.device, dtype=torch.float16)
                # the buf.view() should be a view sharing the same storage as buf.
                bfloat_buf = buf.view(dtype=torch.bfloat16)
                BLOCK_SIZE = 256
                numel = buf.numel()
                grid = (triton.cdiv(numel, BLOCK_SIZE),)
                _triton_kernel[grid](bfloat_buf, numel, BLOCK_SIZE)
                return buf, bfloat_buf

            fn_c = torch.compile(fn)

            x = torch.randn(8, device=GPU_TYPE)
            out_c = fn_c(x)
            out_e = fn(x)

            # expect view() to be an actual view, sharing the same data as the original buffer
            # verify first that this is true in the eager output
            self.assertEqual(out_e[0].data_ptr(), out_e[1].data_ptr())
            # .. and also in the compiled output
            self.assertEqual(out_c[0].data_ptr(), out_c[1].data_ptr())

            self.assertEqual(out_e[0], out_c[0])
            self.assertEqual(out_e[1], out_c[1])

    # TODO enable this test case on XPU.
    @requires_gpu
    def test_i64_input(self):
        # The i64 "seed" input needs to be marked as "i64", not "i32".
        @triton.jit
        def triton_add_noise_(x_ptr, y_ptr, seed, numel, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

            x = tl.load(x_ptr + offsets, mask=(offsets < numel))
            rnd = tl.rand(seed, offsets)
            res = x + rnd
            tl.store(y_ptr + offsets, res, mask=(offsets < numel))

        def add_noise(x, seed):
            y = torch.empty_like(x)
            numel = x.numel()
            BLOCK_SIZE = 256

            def grid(meta):
                return (triton.cdiv(numel, meta["BLOCK_SIZE"]),)

            triton_add_noise_[grid](x, y, seed, numel, BLOCK_SIZE)
            return y

        def fn(x):
            x = x * x
            seed = torch.randint(
                low=2**32, high=2**62, size=(1,), dtype=torch.int64
            ).item()
            return add_noise(x, seed)

        inp = torch.rand(400, device=GPU_TYPE)
        torch._dynamo.mark_dynamic(inp, 0)

        fn_c = torch.compile(fn, fullgraph=True)
        with torch._dynamo.config.patch(capture_scalar_outputs=True):
            res = fn_c(inp)

        self.assertTrue(((res < 2) & (res >= 0)).all().item())

    @requires_gpu
    @parametrize("wrapped", [False, True])
    @parametrize("autotune", [False, True])
    def test_constexpr_dynamic_shapes(self, wrapped, autotune):
        # https://github.com/pytorch/pytorch/issues/136504
        @triton.jit
        def triton_(
            x_ptr,
            y_ptr,
            NUMEL: tl.constexpr,
            IS_ODD: tl.constexpr,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(0)
            offsets = BLOCK_SIZE * pid + tl.arange(0, BLOCK_SIZE)
            mask = offsets < NUMEL

            data = tl.load(x_ptr + offsets, mask)
            result = data * data
            if IS_ODD:
                result = result + 1

            tl.store(y_ptr + offsets, result, mask)

        if autotune:
            triton_ = triton.autotune(
                [
                    triton.Config(kwargs={"BLOCK_SIZE": 128}),
                    triton.Config(kwargs={"BLOCK_SIZE": 256}),
                ],
                key=[],
            )(triton_)

        def triton_kernel_impl(x: torch.Tensor) -> torch.Tensor:
            y = torch.empty_like(x)
            numel = x.numel()

            args = [x, y, numel, numel % 2 == 0]
            if not autotune:
                args.append(256)  # BLOCK_SIZE

            def grid(meta):
                return (triton.cdiv(numel, meta["BLOCK_SIZE"]),)

            if wrapped:
                capture_triton(triton_)[grid](*args)
            else:
                triton_[grid](*args)
            return y

        if wrapped:
            triton_kernel = torch._library.triton_op(
                "constexpr_test::square", triton_kernel_impl, mutates_args={}
            )
        else:
            triton_kernel = triton_kernel_impl

        def fn(x):
            return triton_kernel(x)

        fn_c = torch.compile(fn, dynamic=True)

        x = torch.randn(512 + 5, device=GPU_TYPE)
        res = fn_c(x)
        self.assertEqual(x * x, res)

        x2 = torch.randn(1024 + 5, device=GPU_TYPE)
        res2 = fn_c(x2)
        self.assertEqual(x2 * x2, res2)

    @requires_gpu
    def test_triton_kernel_none_args(self):
        # https://github.com/pytorch/pytorch/issues/115344
        @triton.autotune(
            configs=[
                triton.Config({"BLOCK_SIZE": 32}, num_stages=5, num_warps=2),
                triton.Config({"BLOCK_SIZE": 64}, num_stages=4, num_warps=4),
            ],
            key=["n_elements"],
        )
        @triton.jit
        def sin_kernel(
            in_ptr0,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            if in_ptr0 is not None:
                x = tl.load(in_ptr0 + offsets, mask=mask)
            else:
                x = 0.0
            output = tl.sin(x)
            tl.store(out_ptr + offsets, output, mask=mask)

        def sin_triton(x, out):
            n_elements = out.numel()
            sin_kernel[(n_elements,)](x, out, n_elements)

        x = torch.randn(65, device=GPU_TYPE)
        out = torch.empty_like(x)
        out_compiled = torch.empty_like(x)
        sin_triton_compiled = torch.compile(fullgraph=True)(sin_triton)

        sin_triton(x, out)
        sin_triton_compiled(x, out_compiled)
        self.assertEqual(out, out_compiled)

        sin_triton(None, out)
        sin_triton_compiled(None, out_compiled)
        self.assertEqual(out, out_compiled)

    @requires_gpu
    def test_triton_kernel_global_constexpr(self):
        @triton.jit
        def triton_(in_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            x = tl.load(in_ptr + offsets)
            output = x + FLOAT_CONSTANT_C
            tl.store(out_ptr + offsets, output)

        def fn(x):
            y = torch.empty_like(x)
            BLOCK_SIZE = 256
            grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
            triton_[grid](x, y, BLOCK_SIZE)
            return y

        # make sure FLOAT_CONSTANT_C is NOT annotated
        self.assertFalse("FLOAT_CONSTANT_C" in globals().get("__annotations__", {}))
        # sanity check: STRING_CONSTANT_C _should_ be annotated
        self.assertTrue("STRING_CONSTANT_C" in globals().get("__annotations__", {}))

        x = torch.randn(512, device=GPU_TYPE)
        expected = x + 3.14
        actual = torch.compile(fn)(x)
        self.assertEqual(expected, actual)


def make_mutation_test(fn):
    @requires_gpu
    def test_fn(self):
        from torch._higher_order_ops.triton_kernel_wrap import identify_mutated_tensors

        kernel, inputs, outputs = fn()
        self.assertListEqual(
            identify_mutated_tensors(kernel, inputs),
            outputs,
        )

    return test_fn


# Triton codegen suffers from scoping issues.
# Define helpers here
if HAS_GPU:

    @triton.jit
    def helper_id(p):
        return p

    @triton.jit
    def helper_add_and_out(x, y, out_ptr):
        return x + y, out_ptr


class MutationTests(torch._inductor.test_case.TestCase):
    # Tests injected below

    @make_mutation_test
    def test_out_of_order_kernel():
        @triton.jit
        def add_kernel_out_of_order(
            in_ptr0,
            n_elements,
            in_ptr1,
            out_ptr,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)

        t = torch.randn(4)
        return (
            add_kernel_out_of_order,
            {
                "in_ptr0": t,
                "n_elements": 4,
                "in_ptr1": t,
                "out_ptr": t,
                "BLOCK_SIZE": 4,
            },
            ["out_ptr"],
        )

    @make_mutation_test
    def test_out_of_order_kernel_call():
        @triton.jit
        def add_kernel_out_of_order_fn1(
            in_ptr0,
            n_elements,
            in_ptr1,
            out_ptr,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            add_kernel_out_of_order_fn2(
                in_ptr0, in_ptr1, n_elements, out_ptr, BLOCK_SIZE=BLOCK_SIZE
            )

        t = torch.randn(4)
        return (
            add_kernel_out_of_order_fn1,
            {
                "in_ptr0": t,
                "n_elements": 4,
                "in_ptr1": t,
                "out_ptr": t,
                "BLOCK_SIZE": 4,
            },
            ["out_ptr"],
        )

    @make_mutation_test
    def test_reduce_sum():
        @triton.jit
        def reduce_sum_kernel(a_ptr, c_ptr, stride_am, stride_an):
            offs_am = tl.arange(0, 4)
            offs_an = tl.arange(0, 4)
            a_ptrs = a_ptr + (
                offs_am[:, None] * stride_am + offs_an[None, :] * stride_an
            )
            a = tl.load(a_ptrs)
            m = tl.sum(a, axis=1)
            tl.store(c_ptr + tl.arange(0, 4), m)

        t = torch.randn(4)
        kernel = reduce_sum_kernel
        kwargs = {
            "a_ptr": t,
            "c_ptr": t,
            "stride_am": 4,
            "stride_an": 4,
        }

        # TODO(aakhundov): tt.reduce is now supported, but only
        # in the new MLIR-based Triton analysis pass (not in the
        # old TTIR string parsing-based one). remove this gating
        # and use ["c_ptr"] as `expected` after the new Triton
        # pin lands both in OSS and internally.
        ttir_module, _ = generate_ttir(kernel, kwargs)
        if hasattr(ttir_module, "walk"):
            # with MLIR-based Triton analysis pass
            expected = ["c_ptr"]
        else:
            # with TTIR string parsing-based Triton analysis pass
            expected = ["a_ptr", "c_ptr"]

        return (
            kernel,
            kwargs,
            expected,
        )

    @make_mutation_test
    def test_argmax():
        @triton.jit
        def argmax_kernel(a_ptr, c_ptr, stride_am, stride_an):
            offs_am = tl.arange(0, 4)
            offs_an = tl.arange(0, 4)
            a_ptrs = a_ptr + (
                offs_am[:, None] * stride_am + offs_an[None, :] * stride_an
            )
            a = tl.load(a_ptrs)
            m = tl.argmax(a, axis=1)
            tl.store(c_ptr + tl.arange(0, 4), m)

        t = torch.randn(4)
        kernel = argmax_kernel
        kwargs = {
            "a_ptr": t,
            "c_ptr": t,
            "stride_am": 4,
            "stride_an": 4,
        }

        # TODO(aakhundov): tt.reduce is now supported, but only
        # in the new MLIR-based Triton analysis pass (not in the
        # old TTIR string parsing-based one). remove this gating
        # and use ["c_ptr"] as `expected` after the new Triton
        # pin lands both in OSS and internally.
        ttir_module, _ = generate_ttir(kernel, kwargs)
        if hasattr(ttir_module, "walk"):
            # with MLIR-based Triton analysis pass
            expected = ["c_ptr"]
        else:
            # with TTIR string parsing-based Triton analysis pass
            expected = ["a_ptr", "c_ptr"]

        return (
            kernel,
            kwargs,
            expected,
        )

    @requires_gpu
    @skipIfRocm
    def test_triton_kernel_inference_mode(self):
        def f(x, y, out):
            n_elements = x.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=4)

        with torch.inference_mode():
            x = torch.ones(32, device=GPU_TYPE)
            y = torch.ones(32, device=GPU_TYPE)
            out_ref = torch.zeros_like(x)
            out_test = torch.zeros_like(x)
            f(x, y, out_ref)
            torch.compile(f)(x, y, out_test)
            self.assertEqual(out_ref, out_test)

    @make_mutation_test
    def test_cumsum():
        @triton.jit
        def cumsum_kernel(in_ptr, out_ptr, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
            rindex = tl.arange(0, RBLOCK)[None, :]
            xindex = tl.arange(0, XBLOCK)[:, None]
            data = tl.load(in_ptr + rindex)
            scan = tl.cumsum(data, 1)
            expected_max = tl.sum(data, 1)
            tl.device_assert(scan <= expected_max)
            tl.store(out_ptr + xindex * RBLOCK + rindex, scan)

        t = torch.randn(4)
        kernel = cumsum_kernel
        kwargs = {
            "in_ptr": t,
            "out_ptr": t,
            "XBLOCK": 4,
            "RBLOCK": 16,
        }

        # TODO(aakhundov): tt.scan is now supported, but only
        # in the new MLIR-based Triton analysis pass (not in the
        # old TTIR string parsing-based one). remove this gating
        # and use ["out_ptr"] as `expected` after the new Triton
        # pin lands both in OSS and internally.
        ttir_module, _ = generate_ttir(kernel, kwargs)
        if hasattr(ttir_module, "walk"):
            # with MLIR-based Triton analysis pass
            expected = ["out_ptr"]
        else:
            # with TTIR string parsing-based Triton analysis pass
            expected = ["in_ptr", "out_ptr"]

        return (
            kernel,
            kwargs,
            expected,
        )

    @make_mutation_test
    def test_fn_call_one_return():
        @triton.jit
        def add_kernel_with_fn_call(
            in_ptr0,
            in_ptr1,
            n_elements,
            out_ptr,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
            out = helper_id(out_ptr)
            tl.store(out + offsets, output, mask=mask)

        t = torch.randn(4)
        return (
            add_kernel_with_fn_call,
            {
                "in_ptr0": t,
                "in_ptr1": t,
                "n_elements": 4,
                "out_ptr": t,
                "BLOCK_SIZE": 4,
            },
            ["out_ptr"],
        )

    @make_mutation_test
    def test_fn_call_multi_return():
        @triton.jit
        def add_kernel_with_fn_call(
            in_ptr0,
            in_ptr1,
            n_elements,
            out_ptr,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output, out = helper_add_and_out(x, y, out_ptr)
            tl.store(out + offsets, output, mask=mask)

        t = torch.randn(4)
        return (
            add_kernel_with_fn_call,
            {
                "in_ptr0": t,
                "in_ptr1": t,
                "n_elements": 4,
                "out_ptr": t,
                "BLOCK_SIZE": 4,
            },
            ["out_ptr"],
        )

    @make_mutation_test
    def test_nested_cond_op_kernel():
        @triton.jit
        def nested_cond_op_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            if tl.program_id(0) == 0:
                if tl.program_id(1) == 0:
                    output = x + y
                    tl.store(out_ptr + offsets, output, mask=mask)
            else:
                pass

        t = torch.randn(4)
        return (
            nested_cond_op_kernel,
            {
                "in_ptr0": t,
                "in_ptr1": t,
                "out_ptr": t,
                "n_elements": 4,
                "BLOCK_SIZE": 4,
            },
            ["out_ptr"],
        )

    @make_mutation_test
    def test_add_for_loop():
        @triton.jit
        def add_4_times_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = tl.zeros((n_elements,), dtype=tl.float32)
            for i in range(4):
                output += x + y
            tl.store(out_ptr + offsets, output, mask=mask)

        t = torch.randn(4)
        return (
            add_4_times_kernel,
            {
                "in_ptr0": t,
                "in_ptr1": t,
                "out_ptr": t,
                "n_elements": 4,
                "BLOCK_SIZE": 4,
            },
            ["out_ptr"],
        )

    @make_mutation_test
    def test_add_for_loop2():
        @triton.jit
        def add_1_time_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            for i in range(0, BLOCK_SIZE):
                i = tl.multiple_of(i, 1)
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)

        t = torch.randn(4)
        return (
            add_1_time_kernel,
            {
                "in_ptr0": t,
                "in_ptr1": t,
                "out_ptr": t,
                "n_elements": 4,
                "BLOCK_SIZE": 4,
            },
            ["out_ptr"],
        )

    @make_mutation_test
    def test_add_nested_for_loop():
        @triton.jit
        def add_4_times_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = tl.zeros((n_elements,), dtype=tl.float32)
            for i in range(2):
                for j in range(2):
                    output += x + y
            tl.store(out_ptr + offsets, output, mask=mask)

        t = torch.randn(4)
        return (
            add_4_times_kernel,
            {
                "in_ptr0": t,
                "in_ptr1": t,
                "out_ptr": t,
                "n_elements": 4,
                "BLOCK_SIZE": 4,
            },
            ["out_ptr"],
        )

    @make_mutation_test
    def test_add_nested_for_loop_multi_return():
        @triton.jit
        def add_4_times_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output1 = tl.zeros((n_elements,), dtype=tl.float32)
            output2 = tl.zeros((n_elements,), dtype=tl.float32)
            for i in range(2):
                for j in range(2):
                    output1 += y
                    output2 += x
            output = output1 + output2
            tl.store(out_ptr + offsets, output, mask=mask)

        t = torch.randn(4)
        return (
            add_4_times_kernel,
            {
                "in_ptr0": t,
                "in_ptr1": t,
                "out_ptr": t,
                "n_elements": 4,
                "BLOCK_SIZE": 4,
            },
            ["out_ptr"],
        )

    @make_mutation_test
    def test_labels():
        @triton.jit
        def kernel_with_label(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            if pid > 1:
                return
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)

        t = torch.randn(4)
        return (
            kernel_with_label,
            {
                "in_ptr0": t,
                "in_ptr1": t,
                "out_ptr": t,
                "n_elements": 4,
                "BLOCK_SIZE": 4,
            },
            ["out_ptr"],
        )

    @make_mutation_test
    def test_for_loop_arg():
        @triton.jit
        def fwd_kernel(
            X_ptr,
            W1_ptr,
            b1_ptr,
            O_ptr,
            M: tl.constexpr,
            C1: tl.constexpr,
            C2: tl.constexpr,
            BLOCK_SIZE_M: tl.constexpr,
            BLOCK_SIZE_C2: tl.constexpr,
        ):
            # Get program ids
            pid_m = tl.program_id(0)

            # Compute offsets
            offs_c1 = tl.arange(0, C1)
            offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

            # Load input data
            x_block_ptr = X_ptr + offs_m[:, None] * C1 + offs_c1[None, :]
            x = tl.load(x_block_ptr)

            # Compute gating
            for c2 in range(0, tl.cdiv(C2, BLOCK_SIZE_C2)):
                # Compute block pointers
                offs_c2 = c2 * BLOCK_SIZE_C2 + tl.arange(0, BLOCK_SIZE_C2)
                o_block_ptr = O_ptr + offs_m[:, None] * C2 + offs_c2[None, :]
                w1_block_ptr = W1_ptr + offs_c1[:, None] * C2 + offs_c2[None, :]
                b1_block_ptr = b1_ptr + offs_c2

                # Compute output
                w = tl.load(w1_block_ptr)
                b = tl.load(b1_block_ptr)
                o = tl.dot(x, w, allow_tf32=False)
                o += b[None, :]

                # Store output
                tl.store(o_block_ptr, o)

        t = torch.randn(64)
        return (
            fwd_kernel,
            {
                "X_ptr": t,
                "W1_ptr": t,
                "b1_ptr": t,
                "O_ptr": t,
                "M": 64,
                "C1": 64,
                "C2": 64,
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_C2": 64,
            },
            ["O_ptr"],
        )

    @make_mutation_test
    def test_for_loop_arg_2():
        @triton.jit
        def fwd_kernel(
            x_ptr,
            o_ptr,
            M,
            N,
            stride_m,
            stride_n,
            BLOCK_B: tl.constexpr,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
        ):
            # Get program ids
            pid_m = tl.program_id(0)
            X_block_ptr = tl.make_block_ptr(
                base=x_ptr,
                shape=(M, N),
                strides=(stride_m, stride_n),
                offsets=(0, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0),
            )
            O_block_ptr = tl.make_block_ptr(
                base=o_ptr,
                shape=(M, N),
                strides=(stride_m, stride_n),
                offsets=(0, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0),
            )

            for _ in range(BLOCK_B):
                x = tl.load(X_block_ptr)
                tl.store(O_block_ptr, x)

                X_block_ptr = tl.advance(X_block_ptr, (BLOCK_M, 0))
                O_block_ptr = tl.advance(O_block_ptr, (BLOCK_M, 0))

        t = torch.randn((32, 64, 128))
        o = torch.empty_like(t)
        B, M, N = t.shape
        return (
            fwd_kernel,
            {
                "x_ptr": t,
                "o_ptr": o,
                "M": M,
                "N": N,
                "stride_m": N,
                "stride_n": 1,
                "BLOCK_B": B,
                "BLOCK_M": M,
                "BLOCK_N": N,
            },
            ["o_ptr"],
        )

    @make_mutation_test
    def test_while_loop():
        @triton.jit
        def fwd_kernel(
            x_ptr,
            o_ptr,
            M,
            N,
            stride_m,
            stride_n,
            BLOCK_B: tl.constexpr,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
        ):
            # Get program ids
            pid_m = tl.program_id(0)
            X_block_ptr = tl.make_block_ptr(
                base=x_ptr,
                shape=(M, N),
                strides=(stride_m, stride_n),
                offsets=(0, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0),
            )
            O_block_ptr = tl.make_block_ptr(
                base=o_ptr,
                shape=(M, N),
                strides=(stride_m, stride_n),
                offsets=(0, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0),
            )

            i = 0
            while i < BLOCK_B:
                x = tl.load(X_block_ptr)
                tl.store(O_block_ptr, x)

                X_block_ptr = tl.advance(X_block_ptr, (BLOCK_M, 0))
                O_block_ptr = tl.advance(O_block_ptr, (BLOCK_M, 0))
                i += 1

        t = torch.randn((32, 64, 128))
        o = torch.empty_like(t)
        B, M, N = t.shape
        return (
            fwd_kernel,
            {
                "x_ptr": t,
                "o_ptr": o,
                "M": M,
                "N": N,
                "stride_m": N,
                "stride_n": 1,
                "BLOCK_B": B,
                "BLOCK_M": M,
                "BLOCK_N": N,
            },
            ["o_ptr"],
        )


if HAS_GPU:
    t = torch.randn(4)
    tt = torch.randn(4, 1)
    tests = [
        [
            add_kernel,
            {
                "in_ptr0": t,
                "in_ptr1": t,
                "out_ptr": t,
                "n_elements": 4,
                "BLOCK_SIZE": 4,
            },
            ["out_ptr"],
        ],
        [
            add_kernel_2d_autotuned,
            {
                "in_ptr0": t,
                "in_ptr1": t,
                "out_ptr": t,
                "x_elements": 4,
                "y_elements": 4,
            },
            ["out_ptr"],
        ],
        [
            indirection_kernel,
            {
                "in_ptr0": t,
                "out_ptr": t,
                "n_elements": 4,
                "BLOCK_SIZE": 4,
                "ACTIVATION": "mul2_inplace_kernel",
            },
            ["in_ptr0", "out_ptr"],
        ],
        [
            indirection_kernel,
            {
                "in_ptr0": t,
                "out_ptr": t,
                "n_elements": 4,
                "BLOCK_SIZE": 4,
                "ACTIVATION": "add_kernel",
            },
            ["out_ptr"],
        ],
        [
            mul2_inplace_kernel,
            {"ptr": t, "n_elements": 4, "BLOCK_SIZE": 4},
            ["ptr"],
        ],
        # Cant optimize since the kernel contains a tl.inline_asm_elementwise
        [
            inline_asm_kernel,
            {"X": t, "Y": t, "Z": t, "n": 4, "BLOCK": 4},
            ["X", "Y", "Z"],
        ],
        [
            add_kernel_with_block_ptr,
            {
                "x_ptr": t,
                "y_ptr": t,
                "output_ptr": t,
                "n_elements": 4,
                "BLOCK_SIZE": 4,
            },
            ["output_ptr"],
        ],
        [
            kernel_with_block_ptr_2d,
            {
                "x_ptr": tt,
                "output_ptr": tt,
                "n_elements": 4,
                "BLOCK_SIZE": 4,
            },
            ["output_ptr"],
        ],
        [
            add_kernel_with_import,
            {
                "in_ptr0": t,
                "in_ptr1": t,
                "out_ptr": t,
                "n_elements": 4,
                "BLOCK_SIZE": 4,
            },
            ["out_ptr"],
        ],
        [
            atomic_add_kernel,
            {
                "in_ptr0": t,
                "in_ptr1": t,
                "out_ptr": t,
                "n_elements": 4,
                "BLOCK_SIZE": 4,
            },
            ["out_ptr"],
        ],
        [
            add_4_times_kernel,
            {
                "in_ptr0": t,
                "in_ptr1": t,
                "out_ptr": t,
                "n_elements": 4,
                "BLOCK_SIZE": 4,
            },
            ["out_ptr"],
        ],
        [
            cond_op_kernel,
            {
                "in_ptr0": t,
                "in_ptr1": t,
                "out_ptr": t,
                "n_elements": 4,
                "BLOCK_SIZE": 4,
            },
            ["out_ptr"],
        ],
    ]
    for kernel, inputs, outputs in tests:
        fn = make_mutation_test(
            # Add default arguments to avoid Python lambda capture pitfall
            # This forces the capture at lambda creation
            lambda kernel=kernel, inputs=inputs, outputs=outputs: (
                kernel,
                inputs,
                outputs,
            )
        )
        name = f"test_mutations_{kernel.fn.__name__}"
        # Poor way to make test names be unique
        while name in MutationTests.__dict__:
            name += "1"

        setattr(MutationTests, name, fn)


class CustomOpTests(torch._inductor.test_case.TestCase):
    """Tests for custom ops wrapping triton kernels"""

    @requires_gpu
    @common_utils.parametrize("autotuned", [False, True])
    @common_utils.parametrize("dynamic", [False, True])
    def test_add_kernel(self, autotuned, dynamic):
        from torch._inductor.utils import run_and_get_code

        libname = "my_cool_namespace"
        opname = "my_triton_operator"

        @torch._library.triton_op(f"{libname}::{opname}", mutates_args={})
        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            output = torch.empty_like(x)
            n_elements = output.numel()

            def grid(meta):
                return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            if autotuned:
                capture_triton(add_kernel_autotuned)[grid](x, y, output, n_elements)
            else:
                capture_triton(add_kernel)[grid](x, y, output, n_elements, 16)
            return output

        def f(x, y):
            return add(x, y)

        x = torch.randn(3, device=GPU_TYPE)
        y = torch.randn(3, device=GPU_TYPE)

        out = f(x, y)
        expected = x + y
        self.assertEqual(out, expected)
        out_compiled, codes = run_and_get_code(torch.compile(f, dynamic=dynamic), x, y)
        self.assertEqual(out_compiled, expected)
        self.assertEqual(len(codes), 1)

        # Check that we decomposed the operator away
        code = "\n".join(codes[0])
        self.assertNotIn(libname, code)
        self.assertNotIn(opname, code)

    @requires_gpu
    @patch.object(torch._dynamo.config, "cache_size_limit", 1)
    def test_triton_dynamic_grid_no_recompile(self):
        libname = "my_cool_namespace"
        opname = "my_triton_operator"

        @torch._library.triton_op(f"{libname}::{opname}", mutates_args={})
        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            output = torch.empty_like(x)
            n_elements = output.numel()
            capture_triton(add_kernel)[(n_elements,)](x, y, output, n_elements, 16)
            return output

        @torch.compile(fullgraph=True, dynamic=True)
        def f(x):
            return add(x, x)

        f(torch.randn(8, device=GPU_TYPE))
        f(torch.randn(16, device=GPU_TYPE))

    @unittest.skipIf(not has_triton_package(), "requires triton")
    def test_capture_triton_meta(self):
        import triton
        import triton.language as tl

        @triton.jit
        def add_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)

        @torch._library.triton_op("mylib::add", mutates_args=())
        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            output = torch.empty_like(x)
            n_elements = output.numel()

            def grid(meta):
                return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            capture_triton(add_kernel)[grid](x, y, output, n_elements, 16)
            return output

        def f(x, y):
            return add(x, y)

        x = torch.randn(3, device="meta")
        y = torch.randn(3, device="meta")

        out = f(x, y)
        expected = torch.empty_like(x)
        self.assertEqual(out, expected)

    @requires_gpu
    def test_capture_triton_disabled_in_triton_op(self):
        import triton  # @manual
        import triton.language as tl  # @manual

        @triton.jit
        def add_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)

        add_kernel_decorated = torch._library.capture_triton(add_kernel)

        status = []

        @torch._library.triton_op("mylib::add", mutates_args=())
        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            import torch._higher_order_ops.triton_kernel_wrap

            status.append(torch._library.triton.is_capture_triton_enabled())

            # capture_triton should return the kernel directly if disabled
            result = torch._library.capture_triton(add_kernel)
            self.assertIs(result, add_kernel)

            # Smoke test: check that with capture_triton disabled this still does something
            output = torch.empty_like(x)
            output2 = torch.empty_like(x)

            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            add_kernel_decorated[grid](x, y, output, n_elements, BLOCK_SIZE=16)

            add_kernel_decorated.run(
                x, y, output2, n_elements, BLOCK_SIZE=16, grid=grid, warmup=False
            )

            return output + output2

        x = torch.randn(3, device=GPU_TYPE)
        y = torch.randn(3, device=GPU_TYPE)
        z = add(x, y)
        self.assertEqual(status[-1], False)
        self.assertEqual(z, (x + y) * 2)

    @requires_gpu
    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("autotune", [False, True])
    def test_capture_triton_special_kwargs(self, dynamic, autotune):
        @triton.jit
        def add_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)

        if autotune:
            add_kernel = triton.autotune(
                configs=[
                    triton.Config({"BLOCK_SIZE": 128}),
                    triton.Config({"BLOCK_SIZE": 64}),
                ],
                key=["n_elements"],
            )(add_kernel)

        def f(x, y):
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            if autotune:
                kwargs = {}
            else:
                kwargs = {"BLOCK_SIZE": 128}
            capture_triton(add_kernel)[grid](
                x,
                y,
                output,
                n_elements,
                num_warps=8,
                num_stages=3,
                **kwargs,
            )
            return output

        x = torch.randn(4, device=GPU_TYPE)
        tracing_mode = "symbolic" if dynamic else "fake"

        result = f(x, x)
        self.assertEqual(result, x + x)

        from functorch import make_fx

        gm = make_fx(f, tracing_mode=tracing_mode)(x, x)
        self.assertEqual(gm(x, x), x + x)

    @requires_gpu
    @patch.object(torch._inductor.config, "cpp_wrapper", True)
    @patch.object(torch._inductor.config, "triton.autotune_at_compile_time", True)
    def test_autotune_unbacked(self):
        import triton
        import triton.language as tl

        def get_op_configs():
            return [
                triton.Config(
                    {
                        "BLOCK_M": 32,
                        "BLOCK_N": 64,
                        "BLOCK_K": 32,
                        "GROUP_M": 8,
                    },
                    num_stages=5,
                    num_warps=2,
                ),
                triton.Config(
                    {
                        "BLOCK_M": 128,
                        "BLOCK_N": 256,
                        "BLOCK_K": 64,
                        "GROUP_M": 8,
                    },
                    num_stages=3,
                    num_warps=8,
                ),
            ]

        @triton.autotune(
            configs=get_op_configs(),
            key=["N", "K"],
        )
        @triton.jit
        def op_zeros(
            x_ptr,
            w_ptr,
            z_ptr,
            M,
            N,
            K,
            stride_xm,
            stride_xk,
            stride_wk,
            stride_wn,
            stride_zm,
            stride_zn,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr,
            GROUP_M: tl.constexpr,
            ALLOW_TF32: tl.constexpr,
        ):
            pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_M)
            num_pid_n = tl.cdiv(N, BLOCK_N)
            num_pid_in_group = GROUP_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
            pid_m = first_pid_m + (pid % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m

            offs_m = tl.arange(0, BLOCK_M)
            offs_n = tl.arange(0, BLOCK_N)
            mask_m = (pid_m * BLOCK_M + offs_m)[:, None] < M
            mask_n = (pid_n * BLOCK_N + offs_n)[None, :] < N

            z_mask = mask_m & mask_n
            z = 0.0
            z_ptr += pid_m.to(tl.int64) * BLOCK_M * stride_zm
            z_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_zn
            z_ptrs = z_ptr + stride_zm * offs_m[:, None] + stride_zn * offs_n[None, :]
            tl.store(z_ptrs, z, mask=z_mask)

        @torch.compile()
        def foo(x, w):
            M, K = x.shape
            KB, N = w.shape
            assert K == KB, f"incompatible dimensions {K}, {KB}"

            z = torch.empty((M, N), device=x.device, dtype=x.dtype)

            def grid(META):
                return (
                    triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
                )

            op_zeros[grid](
                x,
                w,
                z,
                M,
                N,
                K,
                x.stride(0),
                x.stride(1),
                w.stride(0),
                w.stride(1),
                z.stride(0),
                z.stride(1),
                ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
            )
            return z

        M, K, N = 128, 64, 32
        x = torch.randn(M, K, device=GPU_TYPE)
        w = torch.randn(K, N, device=GPU_TYPE)

        torch._dynamo.decorators.mark_unbacked(x, 0)
        torch._logging.set_logs(output_code=True)
        with self.assertLogs(logger="torch._inductor", level=logging.DEBUG) as log:
            foo(x, w)

        output = "\n".join(record.getMessage() for record in log.records)
        # correct grid example values updated per block size
        FileCheck().check("Compile-time auto-tuning code").check(
            "grid_wrapper_for_op_zeros_0"
        ).check_next("return (256").check_next("return (64").run(output)

    @requires_gpu
    def test_autotune_no_pre_or_post_hook(self):
        def init_to_zero(name):
            return lambda nargs: nargs[name].zero_()

        # pre_hook requires running arbitrary code at runtime, which we cannot handle at this time
        # https://github.com/pytorch/pytorch/issues/139059
        @triton.autotune(
            configs=[
                triton.Config(
                    {"BLOCK_SIZE": 1024},
                    num_warps=4,
                    num_stages=2,
                    pre_hook=init_to_zero("output_ptr"),
                )
            ],
            key=["n_elements"],
        )
        @triton.jit
        def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)

            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements

            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = x + y
            tl.atomic_add(output_ptr + offsets, output, mask=mask)

        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            output = torch.ones(x.shape, device=x.device, dtype=x.dtype)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            add_kernel[grid](x, y, output, n_elements)
            return output

        x = torch.ones((4096,), device=GPU_TYPE, dtype=torch.float16)
        y = torch.ones((4096,), device=GPU_TYPE, dtype=torch.float16)

        # should always pass
        assert add(x, y).mean() == 2, "Problem with add kernel"

        # this should cause an exception, since pre_hook is not allowed
        msg = "pre_hook is not supported in triton.Autotune Configs"
        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, msg):
            add_compiled = torch.compile(add, mode="reduce-overhead", fullgraph=True)
            add_compiled(x, y).mean()


common_utils.instantiate_parametrized_tests(KernelTests)
common_utils.instantiate_parametrized_tests(CustomOpTests)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
