# Owner(s): ["module: dynamo"]
# flake8: noqa: E731
# Skip do not assign a lambda expression, use a def
from unittest.mock import patch

import torch

import torch._dynamo.test_case
import torch._dynamo.testing

from torch._higher_order_ops.triton_kernel_wrap import (
    triton_kernel_wrapper_functional,
    triton_kernel_wrapper_mutation,
)
from torch._inductor import metrics
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import skipIfRocm

# Defines all the kernels for tests
from torch.testing._internal.triton_utils import *  # noqa: F403

if HAS_CUDA:
    import triton
    from triton import language as tl


# Define shared triton constants here.
CONSTANT_C = 4
STRING_CONSTANT_C = "CONSTANT_C"
BOOL_CONSTANT_C = True


class KernelTests(torch._dynamo.test_case.TestCase):
    @requires_cuda()
    def test_triton_kernel_with_kernel_param(self):
        @triton.jit
        def pass_kernel(kernel):
            pass

        @torch.compile(backend="eager")
        def f(x):
            grid = (x.numel(),)
            pass_kernel[grid](kernel=x)

        t1 = torch.rand(5, device="cuda")
        f(t1)
        # No need to assert anything, the goal is to make sure dynamo does
        # not crash

    @requires_cuda()
    def test_triton_kernel_higher_order_func(self):
        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

        add_kernel_id = kernel_side_table.add_kernel(add_kernel)

        t1 = torch.rand(5, device="cuda")
        t2 = torch.rand(5, device="cuda")

        torch_add = t1 + t2

        # Test higher order function with mutation
        output = torch.zeros_like(t1)
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        triton_kernel_wrapper_mutation(
            kernel_idx=add_kernel_id,
            grid=[grid],
            kwargs={
                "in_ptr0": t1,
                "in_ptr1": t2,
                "out_ptr": output,
                "n_elements": n_elements,
                "BLOCK_SIZE": 16,
            },
        )
        self.assertEqual(output, torch_add)
        # Make sure it is modified
        self.assertNotEqual(output, torch.zeros_like(t1))

        # Test higher order function without mutation
        output = torch.zeros_like(t1)
        out_dict = triton_kernel_wrapper_functional(
            kernel_idx=add_kernel_id,
            grid=[grid],
            kwargs={
                "in_ptr0": t1,
                "in_ptr1": t2,
                "out_ptr": output,
                "n_elements": n_elements,
                "BLOCK_SIZE": 16,
            },
            tensors_to_clone=["in_ptr0", "in_ptr1", "out_ptr"],
        )
        self.assertEqual(out_dict["out_ptr"], torch_add)
        # Make sure it is NOT modified
        self.assertEqual(output, torch.zeros_like(t1))

    @requires_cuda()
    @skipIfRocm
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
                grid=[(x.numel(),)],
                kwargs={
                    "in_ptr0": x,
                    "out_ptr": output,
                    "n_elements": output.numel(),
                    "BLOCK_SIZE": 16,
                },
                tensors_to_clone=["in_ptr0", "out_ptr"],
            )
            return out["out_ptr"]

        t1 = torch.rand(5, device="cuda")
        t2 = torch.rand(5, device="cuda")
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
    triton_kernel_wrapper_functional_proxy = torch._higher_order_ops.triton_kernel_wrap.triton_kernel_wrapper_functional(kernel_idx = 0, grid = [(5,)], kwargs = {'in_ptr0': x_1, 'out_ptr': output_1, 'n_elements': 5, 'BLOCK_SIZE': 16}, tensors_to_clone = ['in_ptr0', 'out_ptr']);  x_1 = output_1 = None
    getitem = triton_kernel_wrapper_functional_proxy['in_ptr0']
    getitem_1 = triton_kernel_wrapper_functional_proxy['out_ptr'];  triton_kernel_wrapper_functional_proxy = None
    return getitem_1""",
        )

    @requires_cuda()
    @skipIfRocm
    def test_triton_kernel_mutation_type(self):
        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch._subclasses.functional_tensor import (
            FunctionalTensor,
            FunctionalTensorMode,
        )

        def prep():
            x = torch.ones(4, device="cuda", requires_grad=True)
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
                    grid=[(x_func.numel(),)],
                    kwargs={
                        "ptr": x_func,
                        "n_elements": x_func.numel(),
                        "BLOCK_SIZE": 16,
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
                    grid=[(x_func.numel(),)],
                    kwargs={
                        "ptr": x_func,
                        "n_elements": x_func.numel(),
                        "BLOCK_SIZE": 16,
                    },
                )

            self.assertFalse(
                torch._functionalize_are_all_mutations_hidden_from_autograd(x_func.elem)
            )

    @requires_cuda()
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

        t = torch.rand(4, 4, device="cuda")
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

    @requires_cuda()
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

        t = torch.rand(5, device="cuda")
        compiled_func = torch.compile(call_triton, backend=backend, fullgraph=True)
        self.assertEqual(2 * t, compiled_func(t))

    @requires_cuda()
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

        t = torch.rand(5, device="cuda")

        compiled_func = torch.compile(f, backend=backend, fullgraph=True)
        # TODO(oulgen): NYI - Support this
        # self.assertEqual(t * t, compiled_func(t))

    @requires_cuda()
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

        t1 = torch.rand(5, device="cuda", requires_grad=grad)
        t2 = torch.rand(5, device="cuda", requires_grad=grad)
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

    @requires_cuda()
    @skipIfRocm
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

        t1 = torch.ones(5, device="cuda")
        t2 = torch.ones(5, device="cuda")

        test, (code,) = run_and_get_code(torch.compile(call_triton_add), t1, t2)
        self.assertEqual(test, 5 * torch.ones(5, device="cuda"))
        self.assertTrue("add_kernel_autotuned_1.run" not in code)

    @requires_cuda()
    @skipIfRocm
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

        t = torch.ones(5, device="cuda")
        test, (code,) = run_and_get_code(torch.compile(call_triton), t)
        # Make sure we emitted two kernels here
        self.assertTrue("pass_kernel_0.run" in code)
        self.assertTrue("pass_kernel_1.run" in code)

    @requires_cuda()
    @skipIfRocm
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

        output = torch.randn(5, device="cuda")
        # Make sure this does not crash
        call_triton(output)

    @requires_cuda()
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

        t1 = torch.rand(5, device="cuda")
        t2 = torch.rand(5, device="cuda")
        torch_result = call_triton(t1, t2)
        compiled_result = torch.compile(call_triton)(t1, t2)
        self.assertEqual(torch_result, compiled_result)

    @requires_cuda()
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
            [-2.0, -1.0, 0.0, 1.0, 2.0], device="cuda", requires_grad=grad
        )
        t2 = torch.tensor(
            [-2.0, -1.0, 0.0, 1.0, 2.0], device="cuda", requires_grad=grad
        )
        float_result = 2 * t1 + 2 * t2
        float_result = float_result.where(float_result >= 0, 0.0)

        t1i = torch.randint(-2, 2, (5,), device="cuda")
        t2i = torch.randint(-2, 2, (5,), device="cuda")
        o = torch.zeros_like(t1, requires_grad=grad)
        oi = torch.zeros_like(t1i)
        int_result = 2 * t1i + 2 * t2i

        (result, resulti) = call_triton(t1, t2, t1i, t2i, o, oi)
        self.assertEqual(float_result, result)
        self.assertEqual(int_result, resulti)

    @requires_cuda()
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
            if CONSTANT_NAME.value == STRING_CONSTANT_C:
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

        t = torch.randn(5, device="cuda")
        torch_result = call_triton(t)
        compiled_result = torch.compile(call_triton)(t)

        self.assertEqual(torch_result, compiled_result)

        # reset back
        CONSTANT_C = prev_c

    @requires_cuda()
    @skipIfRocm
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

        t1 = torch.rand(256, device="cuda", requires_grad=grad)
        t2 = torch.rand(256, device="cuda", requires_grad=grad)
        output = torch.zeros_like(t1, requires_grad=grad)

        torch_add = call_triton(t1, t2, output)
        compiled_func = torch.compile(
            call_triton, backend=backend, fullgraph=True, dynamic=dynamic
        )

        output2 = torch.zeros_like(t1, requires_grad=grad)
        self.assertEqual(compiled_func(t1, t2, output2), torch_add)

    @requires_cuda()
    @skipIfRocm
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

        t1 = torch.rand((512, 256), device="cuda", requires_grad=grad)
        t2 = torch.rand((512, 256), device="cuda", requires_grad=grad)
        output = torch.zeros_like(t1, requires_grad=grad)

        torch_result = call_triton(t1, t2, output)
        compiled_func = torch.compile(
            call_triton, backend=backend, fullgraph=True, dynamic=dynamic
        )
        output2 = torch.zeros_like(t1, requires_grad=grad)
        self.assertEqual(compiled_func(t1, t2, output2), torch_result)

    @requires_cuda()
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

        t1 = torch.rand(5, device="cuda", requires_grad=grad)
        t2 = torch.rand(5, device="cuda", requires_grad=grad)
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

    @requires_cuda()
    def test_triton_kernel_mutation_not_mark_dirty(self):
        @torch.compile
        def f(x):
            n_elements = x.numel()
            add_kernel[(n_elements,)](x, x, x, n_elements, 16)
            return x

        x = torch.randn(5, device="cuda", requires_grad=True)
        x_cloned = x.clone()
        out = x_cloned.sin()
        f(x_cloned)
        out.sum().backward()

    @requires_cuda()
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

        x = torch.randn(4, 4, device="cuda")
        torch_out = f(x)
        python_out = torch.mm(torch.ones(4, 4, device="cuda"), x) + 10
        self.assertEqual(torch_out, python_out)

    @requires_cuda()
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

        inp = torch.randn(64, 256, device="cuda")

        eager_out = f(inp)
        compiled_out = torch.compile(f)(inp)
        self.assertEqual(compiled_out, eager_out)

    @requires_cuda()
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

        inp = torch.randn(64, 256, device="cuda")

        eager_out = f(inp)
        compiled_out = torch.compile(f)(inp)
        self.assertEqual(compiled_out, eager_out)


class MutationTests(torch._dynamo.test_case.TestCase):
    @requires_cuda()
    def test_find_mutations(self):
        from torch._higher_order_ops.triton_kernel_wrap import identify_mutated_tensors

        t = torch.randn(4)

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
                # TODO(oulgen): since the indirection uses @triton.jit
                # we cannot optimize this right now. Fix it.
                ["in_ptr0", "out_ptr"],
            ],
            [
                mul2_inplace_kernel,
                {"ptr": t, "n_elements": 4, "BLOCK_SIZE": 4},
                ["ptr"],
            ],
            # cant optimize since the kernel contains a tl.inline_asm_elementwise
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
                add_kernel_with_import,
                {
                    "in_ptr0": t,
                    "in_ptr1": t,
                    "out_ptr": t,
                    "n_elements": 4,
                    "BLOCK_SIZE": 4,
                },
                # TODO(oulgen): Improve this by finding a way to monkey
                # patch the imported tl functions
                ["in_ptr0", "in_ptr1", "out_ptr"],
            ],
        ]
        for kernel, inputs, outputs in tests:
            self.assertListEqual(
                identify_mutated_tensors(kernel, inputs),
                outputs,
                msg=f"while testing {kernel.fn.__name__}",
            )


common_utils.instantiate_parametrized_tests(KernelTests)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
