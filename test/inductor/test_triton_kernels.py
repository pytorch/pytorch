# Owner(s): ["module: inductor"]
# ruff: noqa: F841
# flake8: noqa: E731
# Skip do not assign a lambda expression, use a def
import functools
import logging
import os

import torch
import torch._dynamo.testing
import torch._inductor.test_case
import torch.utils._pytree as pytree
from torch._dynamo import config as dynamo_config
from torch._higher_order_ops.triton_kernel_wrap import (
    generate_ttir,
    triton_kernel_wrapper_functional,
    triton_kernel_wrapper_mutation,
)
from torch._inductor import config as inductor_config, metrics
from torch._inductor.pattern_matcher import (
    CallFunctionVarArgs,
    PatternMatcherPass,
    register_graph_pattern,
)
from torch._inductor.utils import run_and_get_code, triton_version_uses_attrs_dict
from torch._library import capture_triton
from torch.testing import FileCheck
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import parametrize, skipIfWindows, skipIfXpu
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CUDA_AND_TRITON,
    HAS_GPU,
    HAS_XPU_AND_TRITON,
)
from torch.testing._internal.logging_utils import log_settings, logs_to_string

# Defines all the kernels for tests
from torch.testing._internal.triton_utils import *  # noqa: F403
from torch.utils._triton import (
    has_triton_experimental_host_tma,
    has_triton_package,
    has_triton_tensor_descriptor_host_tma,
)


if HAS_GPU:
    import triton
    from triton import language as tl

    if HAS_CUDA_AND_TRITON:
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
    elif HAS_XPU_AND_TRITON:
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
    CONSTANT_C: tl.constexpr = tl.constexpr(4)
    STRING_CONSTANT_C: tl.constexpr = tl.constexpr("CONSTANT_C")
    BOOL_CONSTANT_C: tl.constexpr = tl.constexpr(True)
    FLOAT_CONSTANT_C = tl.constexpr(3.14)  # intentionally un-annotated
    # Compute at module level to avoid dynamo tracing issue with
    # torch.backends.cuda.matmul.fp32_precision getter
    USE_TF32 = torch.backends.cuda.matmul.fp32_precision == "tf32"

    if hasattr(triton, "constexpr_function"):

        @triton.constexpr_function
        def log2(n):
            return len(bin(n)) - 3


class KernelTests(torch._inductor.test_case.TestCase):
    def _kernel_launched_in_code(self, kernel_name: str, code: str) -> bool:
        if inductor_config.cpp_wrapper:
            return f"launchKernel({kernel_name}" in code
        return f"{kernel_name}.run(" in code

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
    def test_triton_kernel_ill_formed(self):
        @triton.jit
        def ill_formed_kernel(kernel):
            off_n = tl.program_id(0)
            seq_len_a = 5
            if off_n < seq_len_a:
                out_row_idx = off_n + seq_len_a
            else:
                out_row_idx = (off_n + seq_len_a).to(tl.int64)
            tl.store(kernel, out_row_idx)

        @torch.compile(backend="inductor")
        def f(x):
            grid = (x.numel(),)
            ill_formed_kernel[grid](kernel=x)

        catch_error = False
        try:
            t1 = torch.rand(5, device=GPU_TYPE)
            f(t1)
        except torch._inductor.exc.InductorError as e:
            if isinstance(e.inner_exception, triton.compiler.errors.CompilationError):
                catch_error = True
            if (
                isinstance(
                    e.inner_exception,
                    torch._inductor.compile_worker.subproc_pool.SubprocException,
                )
                and "triton.compiler.errors.CompilationError"
                in e.inner_exception.details
            ):
                catch_error = True
        finally:
            # we assert user custom Triton kernel compile error can be captured
            # after torch.compile
            self.assertTrue(catch_error)

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
        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
        from torch._subclasses.functional_tensor import (
            CppFunctionalizeAPI,
            FunctionalTensorMode,
            PythonFunctionalizeAPI,
        )
        from torch.fx.experimental.proxy_tensor import make_fx

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
    def test_triton_kernel_clone_wekdeps(self):
        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
        from torch._inductor.choices import InductorChoices
        from torch._inductor.compile_fx import compile_fx_inner
        from torch._inductor.virtualized import V
        from torch.fx.experimental.proxy_tensor import make_fx

        TENSOR_SIZE = 30_000_000
        BLOCK_SIZE = 1024

        @triton.jit
        def dual_output_kernel_with_inline_asm(
            in_ptr0,
            out_ptr0,
            out_ptr1,
            n_elements,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            result = x.to(tl.float32)
            tl.store(out_ptr0 + offsets, result, mask=mask)
            tl.store(out_ptr1 + offsets, result + x, mask=mask)

        kernel_side_table.reset_table()

        def f(x: torch.Tensor):
            full_default = torch.full(
                (TENSOR_SIZE,), 0.0, device=x.device, dtype=x.dtype
            )
            running_sum = torch.empty_like(x)
            running_sum.fill_(0.0)

            for _ in range(7):
                n_elements = full_default.numel()
                grid = (triton.cdiv(n_elements, BLOCK_SIZE), 1, 1)
                out = triton_kernel_wrapper_functional(
                    kernel_idx=kernel_side_table.add_kernel(
                        dual_output_kernel_with_inline_asm
                    ),
                    constant_args_idx=kernel_side_table.add_constant_args(
                        {"n_elements": n_elements, "BLOCK_SIZE": BLOCK_SIZE}
                    ),
                    grid=[grid],
                    tma_descriptor_metadata={},
                    kwargs={
                        "in_ptr0": x,
                        "out_ptr0": full_default,
                        "out_ptr1": full_default,
                    },
                    tensors_to_clone=["out_ptr0", "out_ptr1"],
                )
                dk_output = out["out_ptr0"]
                dv_output = out["out_ptr1"]
                running_sum = running_sum + dk_output + dv_output

            return (running_sum,)

        t = torch.rand(TENSOR_SIZE, device=GPU_TYPE)

        class NoFusionChoices(InductorChoices):
            def can_fuse(self, scheduler, node1, node2, shared_data_score) -> bool:
                return False

        gm = make_fx(f, tracing_mode="fake")(t)

        with V.set_choices_handler(NoFusionChoices()):
            log_stream, ctx = logs_to_string("torch._inductor.codecache", "output_code")
            with ctx():
                compiled_gm = compile_fx_inner(gm, [t])
                compiled_result = compiled_gm([t])[0]

            output_code = log_stream.getvalue()

        FileCheck().check("del buf3").check(
            "dual_output_kernel_with_inline_asm_0.run(x_1, buf0,"
        ).run(output_code)

        eager_result = f(t.clone())[0]
        self.assertEqual(eager_result, compiled_result)

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
        if inductor_config.cpp_wrapper:
            self.assertEqual(
                output_code.count("std::numeric_limits<double>::quiet_NaN()"), 0
            )
        else:
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
    @inductor_config.patch("implicit_fallbacks", False)
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
        test, (code,) = run_and_get_code(
            torch.compile(call_triton, dynamic=dynamic), t1, t2, o2
        )
        if not grad:
            self.assertEqual(metrics.generated_kernel_count, 1)
        self.assertEqual(torch_add, test)
        # These two asserts are not optimal since it requires original aten
        # to be in the metadata, so there might be false negatives
        self.assertNotIn(
            "aoti_torch_copy_" if inductor_config.cpp_wrapper else "aten.copy", code
        )
        self.assertNotIn(
            "aoti_torch_clone" if inductor_config.cpp_wrapper else "aten.clone", code
        )
        # The following checks that there are only the tensor output is in
        # the compiled graph
        if dynamic and grad:
            if inductor_config.cpp_wrapper:
                self.assertIn("output_handles[0] = ", code)
                self.assertIn("output_handles[1] = ", code)
            else:
                self.assertIn("return (buf0, s92, )", code)
        else:
            self.assertIn(
                "output_handles[0] = "
                if inductor_config.cpp_wrapper
                else "return (buf0, )",
                code,
            )

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
            for _ in range(4):
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
        self.assertTrue(self._kernel_launched_in_code("pass_kernel_0", code))
        self.assertTrue(self._kernel_launched_in_code("pass_kernel_1", code))

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
        CONSTANT_C = tl.constexpr(10)
        if CONSTANT_C == prev_c:
            raise AssertionError

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
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    @inductor_config.patch("unsafe_ignore_unsupported_triton_autotune_args", True)
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
    @common_utils.parametrize("tdlp", ["0", "1"])
    def test_triton_kernel_2d_autotune(self, grad, dynamic, backend, grid_type, tdlp):
        import os

        os.environ["TORCHINDUCTOR_DUMP_LAUNCH_PARAMS"] = tdlp

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
        from torch.fx.experimental.proxy_tensor import make_fx

        tracing_mode = "symbolic" if dynamic else "fake"

        for test in tests:
            gm = make_fx(test, tracing_mode=tracing_mode)(t0, t1)
            result = test(t2, t3)
            self.assertEqual(result, torch_add)

    @requires_gpu
    @common_utils.parametrize("grad", [False, True])
    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    @inductor_config.patch("implicit_fallbacks", False)
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
    @inductor_config.patch("allow_buffer_reuse", True)
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
                # at some point we should reuse buffers not allocate new ones.
                x = _mul2(x)
            return x + 1

        x = torch.randn(10, device=GPU_TYPE, dtype=torch.float32)
        eager_out = f(x)
        compiled_out, (code,) = run_and_get_code(torch.compile(f), x)
        self.assertEqual(compiled_out, eager_out)

        # Check that we're allocating the minimal # of buffers.
        code_string = (
            "aoti_torch_empty_strided("
            if inductor_config.cpp_wrapper
            else f"empty_strided_{GPU_TYPE}((10, ), (1, ), torch.float32)"
        )
        num_bufs_allocated = code.count(code_string)
        self.assertEqual(num_bufs_allocated, 2)

        # Check we're reusing buffers if not allocating.
        num_bufs_reused = code.count(
            "// reuse" if inductor_config.cpp_wrapper else "# reuse"
        )
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

    @inductor_config.patch(
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
    def test_triton_kernel_to_cpu(self):
        def f(x, y):
            out = torch.zeros_like(x)
            add_kernel[(1,)](x, y, out, 16, 16)
            out_cpu = out.cpu() + 1
            return out_cpu

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
    @dynamo_config.patch(capture_dynamic_output_shape_ops=True)
    @dynamo_config.patch(capture_scalar_outputs=True)
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
    @common_utils.parametrize("dump_launch_params", ["0", "1"])
    @common_utils.parametrize("dynamic", [False, True])
    def test_triton_kernel_equal_to_1_arg(self, dynamic, dump_launch_params):
        os.environ["TORCHINDUCTOR_DUMP_LAUNCH_PARAMS"] = dump_launch_params

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

        if triton_version_uses_attrs_dict():
            self.assertFalse("equal_to" in sources[0])
        else:
            if dynamic:
                # when half_n_elements passed to the Triton kernel is
                # dynamic, equal_to_1 specialization can't be enforced

                # also, equal_to_1 specialization doesn't occur (or appear in the signature)
                # for newer versions of triton (i.e. the ones where triton_version_uses_attrs_dict() == True)
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
        if not triton_version_uses_attrs_dict():
            self.assertTrue(_triton_get_ast_equal_to_str(()) in sources[0])
        self.assertEqual(compiled_out, eager_out)

    @requires_gpu
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

    @unittest.skipIf(
        not HAS_GPU or not hasattr(triton, "constexpr_function"),
        "newer triton version required",
    )
    def test_triton_kernel_with_constexpr_function(self):
        @triton.jit
        def kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)

            FIRST_DIM: tl.constexpr = x.shape[0]
            output = x + log2(FIRST_DIM)
            tl.store(output_ptr + offsets, output, mask=mask)

        def f(x):
            out = torch.zeros_like(x)
            n_elements = x.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            kernel[grid](x, out, n_elements, BLOCK_SIZE=16)
            return out

        x = torch.randn(16, device=GPU_TYPE)
        eager_out = f(x)
        compiled_out, (triton_code,) = run_and_get_code(
            torch.compile(f, fullgraph=True), x
        )

        self.assertIn("@triton.constexpr_function", triton_code)
        self.assertEqual(compiled_out, eager_out)

    @requires_gpu
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
            self.assertTrue(self._kernel_launched_in_code("add_kernel_0", code))
            self.assertTrue(self._kernel_launched_in_code("add_kernel_1", code))
        else:
            # size == 16 or dynamic
            # Only one kernel
            self.assertTrue(self._kernel_launched_in_code("add_kernel_0", code))
            self.assertFalse(self._kernel_launched_in_code("add_kernel_1", code))

        self.assertEqual(compiled_out, eager_out)

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
        args_list = [(x, y, torch.float32, tl.float32)]
        if torch.cuda.is_bf16_supported(including_emulation=False):
            args_list.append((x, y, torch.bfloat16, tl.bfloat16))

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
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_triton_kernel_empty_autotune_config_dict(self, backend):
        @triton.autotune(
            configs=[
                triton.Config({}, num_stages=2),
                triton.Config({}, num_stages=3),
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
                BLOCK_SIZE=128,
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
    @common_utils.parametrize("tma_version", ["new", "old"])
    def test_on_device_tma(self, dynamic, tma_version):
        if tma_version == "new" and not has_triton_tensor_descriptor_host_tma():
            self.skipTest("requires triton.tools.tensor_descriptor TMA support")
        if tma_version == "old" and not has_triton_experimental_host_tma():
            self.skipTest("requires triton.tools.experimental_descriptor TMA support")

        kernel = (
            add_kernel_on_device_tma_new_api
            if tma_version == "new"
            else add_kernel_on_device_tma_old_api
        )

        def f(a, b):
            BLOCK_SIZE = 32
            out = torch.zeros_like(a)
            m, n = out.size()

            # Allocate workspace for on-device TMA descriptors
            # Need 128 bytes per descriptor, 3 descriptors total
            if tma_version == "old":
                workspace = torch.zeros(3 * 128, dtype=torch.uint8, device=a.device)
            else:
                workspace = None

            grid = lambda meta: (
                triton.cdiv(m, meta["BLOCK_SIZE"]),
                triton.cdiv(n, meta["BLOCK_SIZE"]),
            )

            kernel[grid](
                a,
                b,
                out,
                m,
                n,
                workspace,
                BLOCK_SIZE=BLOCK_SIZE,
            )

            return out

        a = torch.randn((32, 32), device=GPU_TYPE)
        b = torch.randn((32, 32), device=GPU_TYPE)

        expected_out = a + b
        triton.set_allocator(
            lambda size, align, stream: torch.empty(
                size, dtype=torch.int8, device=GPU_TYPE
            )
        )
        eager_out = f(a, b)
        compiled_out = torch.compile(f, fullgraph=True, dynamic=dynamic)(a, b)

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
    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("tma_version", ["new", "old"])
    def test_tma_capture_and_functionalize(self, dynamic, tma_version):
        if tma_version == "new" and not has_triton_tensor_descriptor_host_tma():
            self.skipTest("requires triton.tools.tensor_descriptor TMA support")
        if tma_version == "old" and not has_triton_experimental_host_tma():
            self.skipTest("requires triton.tools.experimental_descriptor TMA support")

        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

        kernel_side_table.reset_table()
        kernel = (
            add_kernel_with_tma_1d_new_api
            if tma_version == "new"
            else add_kernel_with_tma_1d_old_api
        )

        def f(a, b):
            BLOCK_SIZE = 256
            out = torch.zeros_like(a)
            n_elements = out.numel()

            desc_a, desc_b, desc_out = (
                create_tensor_descriptor_shim(
                    t, [BLOCK_SIZE], new_api=(tma_version == "new")
                )
                for t in (a, b, out)
            )

            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            kernel[grid](
                desc_a,
                desc_b,
                desc_out,
                BLOCK_SIZE=BLOCK_SIZE,
            )

            return out

        a = torch.randn(301, device=GPU_TYPE)
        b = torch.randn(301, device=GPU_TYPE)

        backend = torch._dynamo.testing.AotEagerAndRecordGraphs()
        _ = f(a, b)
        torch.compile(
            f,
            fullgraph=True,
            backend=backend,
            dynamic=dynamic,
        )(a, b)

        if dynamic:
            if tma_version == "new":
                self.assertExpectedInline(
                    backend.fw_graphs[0].code.strip(),
                    """\
def forward(self, arg0_1, arg1_1, arg2_1):
    zeros_like = torch.ops.aten.zeros_like.default(arg1_1, pin_memory = False)
    add_2 = arg0_1 + 256;  arg0_1 = None
    sub_1 = add_2 - 1;  add_2 = None
    floordiv = sub_1 // 256;  sub_1 = None
    triton_kernel_wrapper_functional_proxy = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 0, constant_args_idx = 0, grid = [(floordiv, 1, 1)], tma_descriptor_metadata = {'in_desc_ptr0': ('stable', ([256],)), 'in_desc_ptr1': ('stable', ([256],)), 'out_desc_ptr': ('stable', ([256],))}, kwargs = {'in_desc_ptr0': arg1_1, 'in_desc_ptr1': arg2_1, 'out_desc_ptr': zeros_like}, tensors_to_clone = ['out_desc_ptr']);  floordiv = arg1_1 = arg2_1 = zeros_like = None
    getitem = triton_kernel_wrapper_functional_proxy['out_desc_ptr'];  triton_kernel_wrapper_functional_proxy = None
    return (getitem,)""",
                )
            elif tma_version == "old":
                self.assertExpectedInline(
                    backend.fw_graphs[0].code.strip(),
                    """\
def forward(self, arg0_1, arg1_1, arg2_1):
    zeros_like = torch.ops.aten.zeros_like.default(arg1_1, pin_memory = False)
    add_2 = arg0_1 + 256
    sub_1 = add_2 - 1;  add_2 = None
    floordiv = sub_1 // 256;  sub_1 = None
    triton_kernel_wrapper_functional_proxy = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 0, constant_args_idx = 0, grid = [(floordiv, 1, 1)], tma_descriptor_metadata = {'in_desc_ptr0': ('experimental', ([arg0_1], [256], 4)), 'in_desc_ptr1': ('experimental', ([arg0_1], [256], 4)), 'out_desc_ptr': ('experimental', ([arg0_1], [256], 4))}, kwargs = {'in_desc_ptr0': arg1_1, 'in_desc_ptr1': arg2_1, 'out_desc_ptr': zeros_like}, tensors_to_clone = ['out_desc_ptr']);  floordiv = arg0_1 = arg1_1 = arg2_1 = zeros_like = None
    getitem = triton_kernel_wrapper_functional_proxy['out_desc_ptr'];  triton_kernel_wrapper_functional_proxy = None
    return (getitem,)""",
                )
        else:
            if tma_version == "new":
                self.assertExpectedInline(
                    backend.fw_graphs[0].code.strip(),
                    """\
def forward(self, arg0_1, arg1_1):
    zeros_like = torch.ops.aten.zeros_like.default(arg0_1, pin_memory = False)
    triton_kernel_wrapper_functional_proxy = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 0, constant_args_idx = 0, grid = [(2, 1, 1)], tma_descriptor_metadata = {'in_desc_ptr0': ('stable', ([256],)), 'in_desc_ptr1': ('stable', ([256],)), 'out_desc_ptr': ('stable', ([256],))}, kwargs = {'in_desc_ptr0': arg0_1, 'in_desc_ptr1': arg1_1, 'out_desc_ptr': zeros_like}, tensors_to_clone = ['out_desc_ptr']);  arg0_1 = arg1_1 = zeros_like = None
    getitem = triton_kernel_wrapper_functional_proxy['out_desc_ptr'];  triton_kernel_wrapper_functional_proxy = None
    return (getitem,)""",
                )
            elif tma_version == "old":
                self.assertExpectedInline(
                    backend.fw_graphs[0].code.strip(),
                    """\
def forward(self, arg0_1, arg1_1):
    zeros_like = torch.ops.aten.zeros_like.default(arg0_1, pin_memory = False)
    triton_kernel_wrapper_functional_proxy = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 0, constant_args_idx = 0, grid = [(2, 1, 1)], tma_descriptor_metadata = {'in_desc_ptr0': ('experimental', ([301], [256], 4)), 'in_desc_ptr1': ('experimental', ([301], [256], 4)), 'out_desc_ptr': ('experimental', ([301], [256], 4))}, kwargs = {'in_desc_ptr0': arg0_1, 'in_desc_ptr1': arg1_1, 'out_desc_ptr': zeros_like}, tensors_to_clone = ['out_desc_ptr']);  arg0_1 = arg1_1 = zeros_like = None
    getitem = triton_kernel_wrapper_functional_proxy['out_desc_ptr'];  triton_kernel_wrapper_functional_proxy = None
    return (getitem,)""",
                )

    @requires_gpu
    @common_utils.parametrize("after_data_ptr", [False, True])
    @common_utils.parametrize("after_create_desc", [False, True])
    @common_utils.parametrize("tma_version", ["new", "old"])
    def test_tma_graph_breaks(self, after_data_ptr, after_create_desc, tma_version):
        if tma_version == "new" and not has_triton_tensor_descriptor_host_tma():
            self.skipTest("requires triton.tools.tensor_descriptor TMA support")
        if tma_version == "old" and not has_triton_experimental_host_tma():
            self.skipTest("requires triton.tools.experimental_descriptor TMA support")

        kernel = (
            add_kernel_with_tma_1d_new_api
            if tma_version == "new"
            else add_kernel_with_tma_1d_old_api
        )

        def f(a, b):
            BLOCK_SIZE = 256
            out = torch.zeros_like(a)
            n_elements = out.numel()

            if after_data_ptr:
                torch._dynamo.graph_break()

            descs = [
                create_tensor_descriptor_shim(
                    t, [BLOCK_SIZE], new_api=(tma_version == "new")
                )
                for t in (a, b, out)
            ]

            if after_create_desc:
                torch._dynamo.graph_break()

            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            kernel[grid](
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
    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    @common_utils.parametrize("tma_version", ["new", "old"])
    def test_tma_descriptor_1d(self, dynamic, backend, tma_version):
        if tma_version == "new" and not has_triton_tensor_descriptor_host_tma():
            self.skipTest("requires triton.tools.tensor_descriptor TMA support")
        if tma_version == "old" and not has_triton_experimental_host_tma():
            self.skipTest("requires triton.tools.experimental_descriptor TMA support")

        kernel = (
            add_kernel_with_tma_1d_new_api
            if tma_version == "new"
            else add_kernel_with_tma_1d_old_api
        )

        def f(a, b):
            BLOCK_SIZE = 256
            out = torch.zeros_like(a)
            n_elements = out.numel()

            desc_a, desc_b, desc_out = (
                create_tensor_descriptor_shim(
                    t, [BLOCK_SIZE], new_api=(tma_version == "new")
                )
                for t in (a, b, out)
            )

            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            kernel[grid](
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

        if backend == "inductor":
            log_stream, ctx = logs_to_string("torch._inductor.debug", "ir_post_fusion")
            with ctx():
                compiled_out = torch.compile(
                    f,
                    fullgraph=True,
                    backend=backend,
                    dynamic=dynamic,
                )(a, b)

            FileCheck().check("op4.unmet_dependencies").check(
                "WeakDep(name='buf3', mutating_buf='buf4', is_fake=False)"
            ).run(log_stream.getvalue())
        else:
            compiled_out = torch.compile(
                f,
                fullgraph=True,
                backend=backend,
                dynamic=dynamic,
            )(a, b)

        self.assertEqual(eager_out, expected_out)
        self.assertEqual(compiled_out, expected_out)

    @requires_gpu
    @common_utils.parametrize("tma_version", ["new", "old"])
    def test_tma_descriptor_dedup(self, tma_version):
        if tma_version == "new" and not has_triton_tensor_descriptor_host_tma():
            self.skipTest("requires triton.tools.tensor_descriptor TMA support")
        if tma_version == "old" and not has_triton_experimental_host_tma():
            self.skipTest("requires triton.tools.experimental_descriptor TMA support")

        kernel = (
            add_kernel_with_tma_1d_new_api
            if tma_version == "new"
            else add_kernel_with_tma_1d_old_api
        )

        def f(a):
            BLOCK_SIZE = 256
            out = torch.zeros_like(a)
            n_elements = out.numel()

            desc_a, desc_out = (
                create_tensor_descriptor_shim(
                    t, [BLOCK_SIZE], new_api=(tma_version == "new")
                )
                for t in (a, out)
            )

            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            kernel[grid](
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
        if tma_version == "new":
            self.assertEqual(code.count("TensorDescriptor.from_tensor("), 2)
        else:
            self.assertEqual(code.count("create_1d_tma_descriptor("), 2)

    @requires_gpu
    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("backend", ["eager", "aot_eager"])
    @common_utils.parametrize("tma_version", ["new", "old"])
    def test_tma_descriptor_2d(self, dynamic, backend, tma_version):
        if tma_version == "new" and not has_triton_tensor_descriptor_host_tma():
            self.skipTest("requires triton.tools.tensor_descriptor TMA support")
        if tma_version == "old" and not has_triton_experimental_host_tma():
            self.skipTest("requires triton.tools.experimental_descriptor TMA support")

        kernel = (
            add_kernel_with_tma_2d_new_api
            if tma_version == "new"
            else add_kernel_with_tma_2d_old_api
        )

        def f(a, b):
            BLOCK_SIZE_X = 16
            BLOCK_SIZE_Y = 32
            out = torch.zeros_like(a)
            x_size, y_size = out.size()

            desc_a, desc_b, desc_out = (
                create_tensor_descriptor_shim(
                    t, [BLOCK_SIZE_X, BLOCK_SIZE_Y], new_api=(tma_version == "new")
                )
                for t in (a, b, out)
            )

            grid = lambda meta: (
                triton.cdiv(x_size, meta["BLOCK_SIZE_X"]),
                triton.cdiv(y_size, meta["BLOCK_SIZE_Y"]),
            )
            kernel[grid](
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

        @torch.compile(fullgraph=True, backend=backend)
        def f(x):
            kernel[(1,)](x, num_ctas=1)
            kernel.run(x, num_ctas=1, grid=(1,), warmup=False)
            return x

        msg = "Passing num_ctas directly to the Triton kernel is not supported. Please use a Config in @triton.autotune instead."
        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, msg):
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

        with inductor_config.patch(
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
    @requires_gpu_and_triton
    @parametrize("cfg", ["normal", "cpp_wrapper"])
    def test_triton_kernel_dtype_view(self, cfg):
        # https://github.com/pytorch/pytorch/issues/136159
        if cfg == "normal":
            config_kwargs = {"cpp_wrapper": False}
        elif cfg == "cpp_wrapper":
            config_kwargs = {"cpp_wrapper": True}

        with inductor_config.patch(**config_kwargs):

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
        with dynamo_config.patch(capture_scalar_outputs=True):
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
            triton_kernel = torch.library.triton_op(
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

    @requires_gpu
    @unittest.skipIf(
        not triton_version_uses_attrs_dict(),
        "Test is only valid for new triton versions where attrs is represented by a raw dict",
    )
    def test_triton_attrs_dict_equal_1_None_format(self):
        @triton.jit
        def triton_(in_ptr, out_ptr, numel, add_amount, BLOCK_SIZE: tl.constexpr):
            offsets = tl.arange(0, BLOCK_SIZE)
            x = tl.load(in_ptr + offsets, mask=(offsets < numel))
            output = x * x
            if add_amount is not None:
                output = output + add_amount
            tl.store(out_ptr + offsets, output, mask=(offsets < numel))

        def fn(x):
            y = torch.empty_like(x)
            BLOCK_SIZE = 256
            grid = (1,)
            triton_[grid](x, y, x.numel(), None, BLOCK_SIZE)
            return y

        x = torch.full((1,), 2.5, device=GPU_TYPE)
        expected = fn(x)

        fn_c = torch.compile(fn)
        res, code = run_and_get_code(fn_c, x)
        self.assertEqual(expected, res)

        FileCheck().check("triton_meta=").check("'constants':").check("'numel': 1").run(
            code[0]
        )
        FileCheck().check("triton_meta=").check("'constants':").check(
            "'add_amount': None"
        ).run(code[0])
        FileCheck().check("triton_meta=").check("'constants':").check(
            "'BLOCK_SIZE': 256"
        ).run(code[0])

        FileCheck().check("triton_meta=").check("'signature':").check(
            "'numel': 'constexpr'"
        ).run(code[0])
        FileCheck().check("triton_meta=").check("'signature':").check(
            "'add_amount': 'constexpr'"
        ).run(code[0])
        FileCheck().check("triton_meta=").check("'signature':").check(
            "'BLOCK_SIZE': 'constexpr'"
        ).run(code[0])

    @requires_gpu
    @inductor_config.patch({"triton.autotune_at_compile_time": True})
    @parametrize("quotes", ["single", "double"])
    def test_kernel_with_docstring(self, quotes):
        kernel = (
            kernel_with_docstring_single_quotes
            if quotes == "single"
            else kernel_with_docstring_double_quotes
        )

        # https://github.com/pytorch/pytorch/issues/155006
        def fn(sz):
            x = torch.empty(sz, device=GPU_TYPE)
            BLOCK_SIZE = 32
            grid = (triton.cdiv(sz, BLOCK_SIZE),)
            kernel[grid](x, sz, BLOCK_SIZE)
            return x

        actual = fn(345)
        expected = torch.compile(fn, fullgraph=True)(345)
        self.assertEqual(actual, expected)

    @requires_gpu
    @skipIfXpu(msg="`tl.inline_asm_elementwise` is not yet supported on Intel GPUs")
    @inductor_config.patch({"triton.autotune_at_compile_time": True})
    @parametrize("quotes", ["single", "double"])
    def test_kernel_inline_asm(self, quotes):
        if torch.version.cuda:
            kernel = (
                kernel_inline_asm_single_quotes
                if quotes == "single"
                else kernel_inline_asm_double_quotes
            )
        else:
            kernel = (
                kernel_inline_asm_rocm_single_quotes
                if quotes == "single"
                else kernel_inline_asm_rocm_double_quotes
            )

        # https://github.com/pytorch/pytorch/issues/155006
        def fn(inp):
            sz = inp.size(0)
            x = torch.empty(sz, device=GPU_TYPE)
            BLOCK_SIZE = 32
            grid = (triton.cdiv(sz, BLOCK_SIZE),)
            kernel[grid](inp, x, sz, BLOCK_SIZE)
            return x

        inp = torch.randn(345, device=GPU_TYPE)
        actual = fn(inp)
        expected = torch.compile(fn, fullgraph=True)(inp)
        self.assertEqual(actual, expected)

    @requires_gpu
    @inductor_config.patch("emulate_precision_casts", True)
    def test_triton_kernel_emulate_precision_unaffected(self):
        @triton.jit
        def triton_(in_ptr, out_ptr, numel, add_amount, BLOCK_SIZE: tl.constexpr):
            offsets = tl.arange(0, BLOCK_SIZE)
            x = tl.load(in_ptr + offsets, mask=(offsets < numel))
            output = x * x
            if add_amount is not None:
                output = output + add_amount
            tl.store(out_ptr + offsets, output, mask=(offsets < numel))

        def fn(x):
            y = torch.empty_like(x)
            BLOCK_SIZE = 256
            grid = (1,)
            triton_[grid](x, y, x.numel(), None, BLOCK_SIZE)
            return y

        t1 = torch.rand(5, device=GPU_TYPE)
        fn = torch.compile(fn)
        _, (code,) = run_and_get_code(fn, t1)
        self.assertTrue("enable_fp_fusion" not in code)

    @requires_gpu
    @inductor_config.patch("emulate_precision_casts", True)
    @inductor_config.patch("max_autotune_gemm_backends", "TRITON")
    def test_triton_kernel_emulate_precision_mm_kernels_do_not_change(self):
        from torch._inductor.utils import run_and_get_code

        @torch.compile(mode="max-autotune")
        def fn(a, b):
            return a @ b

        t1 = torch.rand(512, 512, device=GPU_TYPE)
        t2 = torch.rand(512, 512, device=GPU_TYPE)
        try:
            _, (code,) = run_and_get_code(fn, t1, t2)
            self.assertTrue("enable_fp_fusion" not in code)
        except Exception as e:
            if "NoValidChoicesError" in str(e):
                raise unittest.SkipTest(
                    "where inductor has no triton mm kernels available, this test is meaningless"
                ) from e
            raise


def make_mutation_test(fn):
    @requires_gpu
    def test_fn(self):
        from torch._higher_order_ops.triton_kernel_wrap import identify_mutated_tensors

        kernel, inputs, tma_descriptor_metadata, outputs = fn()
        self.assertListEqual(
            identify_mutated_tensors(kernel, inputs, tma_descriptor_metadata),
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

    # Regression test for #169782
    @make_mutation_test
    def test_with_none_arg():
        @triton.jit
        def add_kernel_with_none_arg(
            in_ptr0,
            optional_ptr,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            if optional_ptr is not None:
                y = tl.load(optional_ptr + offsets, mask=mask)
                x = x + y
            tl.store(out_ptr + offsets, x, mask=mask)

        t = torch.randn(4)
        return (
            add_kernel_with_none_arg,
            {
                "in_ptr0": t,
                "optional_ptr": None,
                "out_ptr": t,
                "n_elements": 4,
                "BLOCK_SIZE": 4,
            },
            {},
            ["out_ptr"],
        )

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
            {},
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
            {},
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
        ttir_module, _ = generate_ttir(kernel, kwargs, tma_descriptor_metadata={})
        if hasattr(ttir_module, "walk"):
            # with MLIR-based Triton analysis pass
            expected = ["c_ptr"]
        else:
            # with TTIR string parsing-based Triton analysis pass
            expected = ["a_ptr", "c_ptr"]

        return (
            kernel,
            kwargs,
            {},
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
        ttir_module, _ = generate_ttir(kernel, kwargs, tma_descriptor_metadata={})
        if hasattr(ttir_module, "walk"):
            # with MLIR-based Triton analysis pass
            expected = ["c_ptr"]
        else:
            # with TTIR string parsing-based Triton analysis pass
            expected = ["a_ptr", "c_ptr"]

        return (
            kernel,
            kwargs,
            {},
            expected,
        )

    @requires_gpu
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
        ttir_module, _ = generate_ttir(kernel, kwargs, tma_descriptor_metadata={})
        if hasattr(ttir_module, "walk"):
            # with MLIR-based Triton analysis pass
            expected = ["out_ptr"]
        else:
            # with TTIR string parsing-based Triton analysis pass
            expected = ["in_ptr", "out_ptr"]

        return (
            kernel,
            kwargs,
            {},
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
            {},
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
            {},
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
            {},
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
            for _ in range(4):
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
            {},
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
            for i in range(BLOCK_SIZE):
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
            {},
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
            for _ in range(2):
                for _ in range(2):
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
            {},
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
            for _ in range(2):
                for _ in range(2):
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
            {},
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
            {},
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
            for c2 in range(tl.cdiv(C2, BLOCK_SIZE_C2)):
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
            {},
            ["O_ptr"],
        )

    @skipIfXpu(msg="Blocked by https://github.com/pytorch/pytorch/issues/170049")
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
            {},
            ["o_ptr"],
        )

    @skipIfXpu(msg="Blocked by https://github.com/pytorch/pytorch/issues/170049")
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
            {},
            ["o_ptr"],
        )

    @make_mutation_test
    def test_branch_with_multiple_yield_args():
        @triton.jit
        def branch_with_multiple_yield_args(
            in_ptr0,
            in_ptr1,
            out_ptr,
            conditional_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            conditional = tl.load(conditional_ptr)
            if conditional:
                in0 = in_ptr0 + 1
                in1 = in_ptr1 + 1
                out = out_ptr + 1
            else:
                in0 = in_ptr0
                in1 = in_ptr1
                out = out_ptr
            x = tl.load(in0 + offsets, mask=mask)
            y = tl.load(in1 + offsets, mask=mask)
            tl.store(out + offsets, x + y, mask=mask)

        x = torch.randn(15)
        y = torch.randn(15)
        out = torch.zeros(15)
        conditional = torch.tensor(True)
        return (
            branch_with_multiple_yield_args,
            {
                "in_ptr0": x,
                "in_ptr1": y,
                "out_ptr": out,
                "conditional_ptr": conditional,
                "n_elements": 14,
                "BLOCK_SIZE": 16,
            },
            {},
            ["out_ptr"],
        )

    def test_get_tma_stores(self):
        from torch._higher_order_ops.triton_kernel_wrap import (
            get_tma_stores,
            Intermediate,
            Op,
            Param,
        )

        functions = {
            "helper": {
                Intermediate(idx=0): [
                    Op(
                        "tt.reinterpret_tensor_descriptor",
                        None,
                        [Param(idx=0)],
                        Intermediate(idx=0),
                    )
                ],
            },
            "main": {
                Intermediate(idx=-1): [
                    Op(
                        "tt.call",
                        "helper",
                        [Param(idx=0), Param(idx=1)],
                        Intermediate(idx=-1),
                    )
                ],
            },
        }

        self.assertEqual(get_tma_stores(functions, "helper"), set())
        self.assertEqual(get_tma_stores(functions, "main"), set())

        functions["helper"][Intermediate(idx=-1)] = [
            Op(
                "tt.experimental_descriptor_store",
                None,
                [Intermediate(idx=0), Param(idx=1)],
                Intermediate(idx=-1),
            )
        ]
        get_tma_stores.reset()

        self.assertEqual(
            get_tma_stores(functions, "helper"), {Param(idx=0), Intermediate(idx=0)}
        )
        self.assertEqual(get_tma_stores(functions, "main"), {Param(idx=0)})

    @unittest.skipIf(
        not has_triton_experimental_host_tma(),
        "requires experimental TMA descriptor API",
    )
    @make_mutation_test
    def test_add_kernel_on_device_tma_old_api():
        a = torch.randn(1024, 1024)
        b = torch.randn(1024, 1024)
        c = torch.empty(1024, 1024)
        workspace = torch.empty(128 * 3, dtype=torch.int8)
        return (
            add_kernel_on_device_tma_old_api,
            {
                "a_ptr": a,
                "b_ptr": b,
                "c_ptr": c,
                "m": 1024,
                "n": 1024,
                "workspace": workspace,
                "BLOCK_SIZE": 32,
            },
            {},
            ["c_ptr", "workspace"],
        )

    @unittest.skipIf(
        not has_triton_tensor_descriptor_host_tma(),
        "requires TensorDescriptor API in Triton",
    )
    @make_mutation_test
    def test_add_kernel_on_device_tma_new_api():
        a = torch.randn(1024, 1024)
        b = torch.randn(1024, 1024)
        c = torch.empty(1024, 1024)
        workspace = torch.empty(
            128 * 3, dtype=torch.int8
        )  # Not used by the new API but kept for consistency
        return (
            add_kernel_on_device_tma_new_api,
            {
                "a_ptr": a,
                "b_ptr": b,
                "c_ptr": c,
                "m": 1024,
                "n": 1024,
                "workspace": workspace,
                "BLOCK_SIZE": 32,
            },
            {},
            ["c_ptr"],
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
            {},
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
            {},
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
            {},
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
            {},
            ["out_ptr"],
        ],
        [
            mul2_inplace_kernel,
            {"ptr": t, "n_elements": 4, "BLOCK_SIZE": 4},
            {},
            ["ptr"],
        ],
        [
            inline_asm_kernel_is_pure_true,
            {"X": t, "Y": t, "Z": t, "n": 4, "BLOCK": 4},
            {},
            ["Z"],
        ],
        [
            inline_asm_kernel_is_pure_false,
            {"X": t, "Y": t, "Z": t, "n": 4, "BLOCK": 4},
            {},
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
            {},
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
            {},
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
            {},
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
            {},
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
            {},
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
            {},
            ["out_ptr"],
        ],
    ]
    for kernel, inputs, tma_descriptor_metadata, outputs in tests:
        fn = make_mutation_test(
            # Add default arguments to avoid Python lambda capture pitfall
            # This forces the capture at lambda creation
            lambda kernel=kernel,
            inputs=inputs,
            tma_descriptor_metadata=tma_descriptor_metadata,
            outputs=outputs: (
                kernel,
                inputs,
                tma_descriptor_metadata,
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

        @torch.library.triton_op(f"{libname}::{opname}", mutates_args={})
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
    def test_subclass(self):
        libname = "my_cool_namespace"
        opname = "my_triton_operator"

        @torch.library.triton_op(f"{libname}::{opname}", mutates_args={})
        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            output = torch.empty_like(x)
            n_elements = output.numel()

            def grid(meta):
                return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            capture_triton(add_kernel)[grid](x, y, output, n_elements, 16)
            return output

        def f(x, y):
            return add(x, y)

        x0 = torch.randn(3, device=GPU_TYPE)
        y0 = torch.randn(3, device=GPU_TYPE)
        x1 = torch.randn(3, device=GPU_TYPE)
        y1 = torch.randn(3, device=GPU_TYPE)

        from torch.testing._internal.two_tensor import TwoTensor

        x = TwoTensor(x0, x1)
        y = TwoTensor(y0, y1)

        out = torch.compile(f, fullgraph=True)(x, y)
        expected = f(x, y)
        self.assertEqual(out.a, expected.a)
        self.assertEqual(out.b, expected.b)

    @requires_gpu
    @dynamo_config.patch("recompile_limit", 1)
    def test_triton_dynamic_grid_no_recompile(self):
        libname = "my_cool_namespace"
        opname = "my_triton_operator"

        @torch.library.triton_op(f"{libname}::{opname}", mutates_args={})
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

        @torch.library.triton_op("mylib::add", mutates_args=())
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
    def test_wrap_triton_disabled_in_triton_op(self):
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

        add_kernel_decorated = torch.library.wrap_triton(add_kernel)

        status = []

        @torch.library.triton_op("mylib::add", mutates_args=())
        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            import torch._higher_order_ops.triton_kernel_wrap

            status.append(torch._library.triton.is_wrap_triton_enabled())

            # capture_triton should return the kernel directly if disabled
            result = torch.library.wrap_triton(add_kernel)
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
    @common_utils.parametrize(
        "variant", ["triton_kernel", "custom_op", "mutable_custom_op"]
    )
    def test_preserves_strides(self, variant):
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

        x = torch.randn(4, 4, 2, 2, device=GPU_TYPE)
        other = torch.randn(4, 4, 2, 2, device=GPU_TYPE)

        def add_triton(y, z):
            grid = (z.numel(),)
            out = torch.empty_like(z, memory_format=torch.contiguous_format)
            add_kernel[grid](y, z, out, z.numel(), BLOCK_SIZE=16)
            return out

        class _CustomPass(PatternMatcherPass):
            def __init__(self) -> None:
                super().__init__()

            def __call__(self, g: torch.fx.Graph):
                self.apply(g)

        g = _CustomPass()
        called = False

        @register_graph_pattern(
            CallFunctionVarArgs(torch.ops.aten.permute),
            pass_dict=g,
        )
        def _(match, *args, **kwargs):
            flat_args, spec = pytree.tree_flatten((args, kwargs))

            def decomp(*flat_args):
                args, kwargs = pytree.tree_unflatten(flat_args, spec)
                return torch.ops.mylib.force_channels_last(
                    torch.ops.aten.permute(*args, **kwargs)
                )

            nonlocal called
            called = True
            match.replace_by_example(decomp, flat_args)

        from torch._inductor import config

        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define(
                "force_channels_last(Tensor x) -> Tensor",
                tags=[torch._C.Tag.flexible_layout],
            )

            def impl2(x):
                return x.clone(memory_format=torch.channels_last)

            lib.impl("force_channels_last", impl2, "CompositeExplicitAutograd")

            lib.define(
                "add_op(Tensor x, Tensor y) -> Tensor",
            )

            def impl(x, y):
                return add_triton(x, y)

            def meta(x, y):
                return torch.empty_like(y, memory_format=torch.contiguous_format)

            lib.impl("add_op", impl, "CompositeExplicitAutograd")
            lib.impl("add_op", meta, "Meta")

            lib.define(
                "add_out_op(Tensor x, Tensor y, Tensor(a!) out) -> ()",
            )

            def impl_out(x, y, out):
                grid = (y.numel(),)
                add_kernel[grid](x, y, out, y.numel(), BLOCK_SIZE=16)

            lib.impl("add_out_op", impl_out, "CompositeExplicitAutograd")
            lib.impl("add_out_op", lambda x, y, out: None, "Meta")

            def f(x, other):
                y = x.transpose(2, 3).contiguous().transpose(2, 3)
                z = y.sin().transpose(2, 3)
                if variant == "triton_kernel":
                    return add_triton(y, z)
                elif variant == "custom_op":
                    return torch.ops.mylib.add_op.default(y, z)
                elif variant == "mutable_custom_op":
                    out = torch.empty_like(y, memory_format=torch.contiguous_format)
                    torch.ops.mylib.add_out_op(y, z, out)
                    return out
                else:
                    raise AssertionError("should not be hit")

            with config.patch(
                post_grad_custom_post_pass=g,
            ):
                f_compile = torch.compile(f, fullgraph=True)
                self.assertEqual(f(x, other), f_compile(x, other))
                self.assertTrue(called)

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

        from torch.fx.experimental.proxy_tensor import make_fx

        gm = make_fx(f, tracing_mode=tracing_mode)(x, x)
        self.assertEqual(gm(x, x), x + x)

    @skipIfWindows(msg="AOTI/Cpp_Wrapper have not enabled on Windows")
    @requires_gpu
    @inductor_config.patch("cpp_wrapper", True)
    @inductor_config.patch("triton.autotune_at_compile_time", True)
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
            assert K == KB, f"incompatible dimensions {K}, {KB}"  # noqa: S101

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
                ALLOW_TF32=USE_TF32,
            )
            return z

        M, K, N = 128, 64, 32
        x = torch.randn(M, K, device=GPU_TYPE)
        w = torch.randn(K, N, device=GPU_TYPE)

        torch._dynamo.decorators.mark_unbacked(x, 0)

        with (
            log_settings("+output_code"),
            self.assertLogs(logger="torch._inductor", level=logging.DEBUG) as log,
        ):
            foo(x, w)

        output = "\n".join(record.getMessage() for record in log.records)
        # correct grid example values updated per block size
        FileCheck().check("Compile-time auto-tuning block:").check(
            "PrecomputedGrid"
        ).check("(31 + _launcher_s0) // 32").check("(127 + _launcher_s0) // 128").run(
            output
        )

    # Triton 3.2.0 adds the required flags to the Autotuner object for this test
    # PR: https://github.com/triton-lang/triton/pull/5092
    @requires_gpu
    def test_autotune_no_pre_or_post_hook_user_defined(self):
        from triton.runtime.autotuner import Autotuner

        def init_to_zero(name):
            return lambda nargs: nargs[name].zero_()

        @triton.autotune(
            configs=[
                triton.Config(
                    {"BLOCK_SIZE": 1024},
                    num_warps=4,
                    num_stages=2,
                    pre_hook=init_to_zero("output_ptr"),
                )
            ],
            pre_hook=init_to_zero("output_ptr"),
            post_hook=init_to_zero("output_ptr"),
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
        if add(x, y).mean() != 2:
            raise AssertionError("Problem with add kernel")

        # assert that the user_defined_* flags are properly set on the kernel before compilation
        self.assertEqual(isinstance(add_kernel, Autotuner), True)
        if not hasattr(add_kernel, "user_defined_pre_hook") or not hasattr(
            add_kernel, "user_defined_post_hook"
        ):
            raise unittest.SkipTest(
                "test requires Triton version >= 3.2.0 for Autotuner.user_defined* hooks"
            )

        self.assertEqual(add_kernel.user_defined_pre_hook, True)
        self.assertEqual(add_kernel.user_defined_post_hook, True)

        # this should cause an exception, since pre_hook is not allowed
        msg = "pre_hook and post_hook are not supported in triton.Autotune or triton.Config"
        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, msg):
            add_compiled = torch.compile(add, mode="reduce-overhead", fullgraph=True)
            add_compiled(x, y).mean()

    @requires_gpu
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    @common_utils.parametrize("autotune_at_compile_time", [True, False])
    def test_triton_kernel_reset_to_zero(self, backend, autotune_at_compile_time):
        if autotune_at_compile_time and backend != "inductor":
            raise unittest.SkipTest("compile-time autotuning only exists in inductor")

        @triton.autotune(
            configs=[
                triton.Config({"BLOCK_SIZE": 64}, num_stages=3, num_warps=8),
                triton.Config({"BLOCK_SIZE": 32}, num_stages=3, num_warps=8),
                triton.Config({"BLOCK_SIZE": 16}, num_stages=3, num_warps=8),
            ],
            key=[],
            reset_to_zero=["increment_ptr"],
        )
        @triton.jit
        def increment_kernel(
            in_ptr0,
            increment_ptr,  # reset this to zero every time
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements

            in_ptr_vals = tl.load(in_ptr0 + offsets, mask=mask)
            increment_val = tl.load(increment_ptr + offsets, mask=mask)
            # increment_val should always be zero
            tl.store(in_ptr0 + offsets, in_ptr_vals + increment_val, mask=mask)

        @torch.compile(fullgraph=True, backend=backend)
        def f(x, increment):
            n_elements = x.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            increment_kernel[grid](x, increment, n_elements=n_elements)
            return x

        x = torch.rand(4, device=GPU_TYPE)
        y = torch.clone(x)
        increment = torch.rand(4, device=GPU_TYPE)

        # during autotuning, x should not change in value
        with inductor_config.patch(
            {"triton.autotune_at_compile_time": autotune_at_compile_time}
        ):
            # we will add rand a single time to x
            f(x, increment)

        self.assertEqual(y + increment, x)

    @requires_gpu
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_triton_single_autotune(self, backend):
        @triton.autotune(
            configs=[
                triton.Config(
                    {"BLOCK_SIZE": 4096},
                )
            ],
            key=["n_elements"],
        )
        # Currently, this autotuning decorator will never run!
        # We only support having a single autotuning decorator on each Triton kernel
        @triton.autotune(
            configs=[
                triton.Config(
                    {"BLOCK_SIZE": 1024},
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
            tl.store(output_ptr + offsets, output, mask=mask)

        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            output = torch.ones(x.shape, device=x.device, dtype=x.dtype)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            add_kernel[grid](x, y, output, n_elements)
            return output

        x = torch.ones((4096,), device=GPU_TYPE, dtype=torch.float16)
        y = torch.ones((4096,), device=GPU_TYPE, dtype=torch.float16)

        # this should cause an exception, since pre_hook is not allowed
        msg = "Passing multiple @triton.autotune decorators is not supported. Please use a single @triton.autotune decorator instead."
        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, msg):
            add_compiled = torch.compile(
                add, mode="reduce-overhead", fullgraph=True, backend=backend
            )
            add_compiled(x, y).mean()

    @requires_gpu
    @common_utils.parametrize("non_strict", [True, False])
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    @common_utils.parametrize("with_perf_model", [True, False])
    def test_triton_kernel_prune_configs_by(self, backend, with_perf_model, non_strict):
        # for non-strict mode
        libname = "my_cool_namespace"
        opname = "my_triton_operator"

        records = {}

        def early_config_prune(configs, named_args, **kwargs):
            # we need to save the records to the returned config
            records["run_early_config_prune"] = True
            if "N" in kwargs and kwargs["N"] == 1024:
                records["capture_kwargs"] = True
            # named args are: dst, src, add_float
            if "dst" in named_args and "src" in named_args and len(named_args) == 3:
                records["capture_named_args"] = True
            return [configs[0]]

        def perf_model(*args, **kwargs):
            records["run_perf_model"] = True
            return kwargs["BLOCK_SIZE"] * -1

        if with_perf_model:
            prune_configs_by = {"perf_model": perf_model, "top_k": 1}
        else:
            prune_configs_by = {"early_config_prune": early_config_prune}

        @triton.autotune(
            configs=[
                triton.Config(kwargs={"BLOCK_SIZE": 32}),
                triton.Config(kwargs={"BLOCK_SIZE": 128}),
            ],
            key=["N"],
            prune_configs_by=prune_configs_by,
        )
        @triton.jit
        def prune_by_kernel(
            dst,
            src,
            add_float,
            N,
            BLOCK_SIZE: tl.constexpr,
        ):
            offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            x = tl.load(src + offsets, mask=offsets < N)
            # we only modify dst if our perf_model is applied (and a BLOCK_SIZE of 128 is selected)
            if BLOCK_SIZE == 128:
                x = x + add_float
            tl.store(dst + offsets, x, mask=offsets < N)

        def f(
            dst: torch.Tensor,
            src: torch.Tensor,
            add_float: float,
            N: int,
        ) -> None:
            grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
            if non_strict:
                torch.library.wrap_triton(prune_by_kernel)[grid](
                    dst, src, add_float, N=N
                )
            else:
                prune_by_kernel[grid](dst, src, add_float, N=N)

        if non_strict:
            decorator = torch.library.triton_op(
                f"{libname}::{opname}", mutates_args={"dst"}
            )(f)
        else:
            # we can just pass the function 'f' for dynamo
            decorator = f

        compiled_f = torch.compile(decorator, backend=backend)
        N = 1024
        src = torch.randn(N, device=GPU_TYPE)
        dst = torch.empty(N, device=GPU_TYPE)
        compiled_f(dst, src, 1.5, N)

        if with_perf_model:
            # when applying the perf_model: kwargs["BLOCK_SIZE"] * -1, the largest config (BLOCK_SIZE==128) is selected
            self.assertEqual(len(records), 1)
            self.assertEqual(src + 1.5, dst)
        else:
            # without the perf_model, the BLOCK_SIZE==32, and as a result dst is not modified and remains equal to src
            self.assertEqual(src, dst)
            self.assertEqual(len(records), 3)
            self.assertTrue(records["run_early_config_prune"])
            self.assertTrue(records["capture_kwargs"])
            self.assertTrue(records["capture_named_args"])

    @requires_gpu
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    @common_utils.parametrize("with_perf_model", [True, False])
    def test_triton_kernel_prune_configs_by_recompile(self, backend, with_perf_model):
        """
        We want to recompile if anyone changes configs in the autotuner object
        In short if for example the following sequence of events happens:
        1. foo = torch.compile(bar)
        1. call foo
        2. autotuner.configs = [new configs list]
        3. call foo

        A recompile event should occur, which we check with Dynamo counters
        This tests that we are installing guards on input objects properly
        """

        # We don't modify records here because we are testing whether or not
        # recompiles occur/guards are installed
        # If we modified the non-local records dict here, this would trigger
        # recompile events.
        def early_config_prune(configs, named_args, **kwargs):
            return [configs[0]]

        def perf_model(*args, **kwargs):
            return kwargs["BLOCK_SIZE"] * -1

        if with_perf_model:
            prune_configs_by = {"perf_model": perf_model, "top_k": 1}
        else:
            prune_configs_by = {"early_config_prune": early_config_prune}

        @triton.autotune(
            configs=[
                triton.Config(kwargs={"BLOCK_SIZE": 32}),
                triton.Config(kwargs={"BLOCK_SIZE": 128}),
            ],
            key=["N"],
            prune_configs_by=prune_configs_by,
        )
        @triton.jit
        def prune_by_kernel(
            dst,
            src,
            add_float,
            N,
            BLOCK_SIZE: tl.constexpr,
        ):
            offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            x = tl.load(src + offsets, mask=offsets < N)
            # Let's make sure we always select a block size of 128 based on our perf_model
            if BLOCK_SIZE == 128:
                x = x + add_float
            tl.store(dst + offsets, x, mask=offsets < N)

        torch._dynamo.reset()
        counter = torch._dynamo.testing.CompileCounterWithBackend(backend=backend)

        @torch.compile(fullgraph=True, backend=counter)
        def f(dst, src, add_float, N):
            grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
            prune_by_kernel[grid](dst, src, add_float, N=N)

        N = 1024
        src = torch.randn(N, device=GPU_TYPE)
        dst = torch.empty(N, device=GPU_TYPE)

        # first compilation, this prunes the configs
        f(dst, src, 1.5, N)

        self.assertEqual(counter.op_count, 1)

        f(dst, src, 1.5, N)

        # this should not trigger a recompilation
        # this is because we modified the test to not touch the records dict
        # as we do in test_triton_kernel_prune_configs_by. If we kept it, it would trigger a recompile here.
        self.assertEqual(counter.op_count, 1)

        # Modify the autotuner object
        prune_by_kernel.configs = [triton.Config(kwargs={"BLOCK_SIZE": 64})]

        # Calling the kernel after modifying the autotuner should
        # trigger a recompile
        f(dst, src, 1.5, N)

        self.assertEqual(counter.op_count, 2)

        # there should be no recompile here
        f(dst, src, 1.5, N)

        self.assertEqual(counter.op_count, 2)

    # see: https://github.com/triton-lang/triton/blob/67ea999935f4511a535a25bdecb27e79e3c3af41/python/test/unit/language/test_decorator.py#L31
    @requires_gpu
    @common_utils.parametrize("non_strict", [True, False])
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    @common_utils.parametrize("autotune_at_compile_time", [True, False])
    def test_triton_kernel_heuristic(
        self, backend, autotune_at_compile_time, non_strict
    ):
        # for non-strict mode
        libname = "my_cool_namespace"
        opname = "my_triton_operator"

        @triton.autotune(
            configs=[
                triton.Config(kwargs={"BLOCK_SIZE": 32}),
            ],
            key=["N"],
        )
        # we should be able to modify existing keys in kwargs
        @triton.heuristics({"BLOCK_SIZE": lambda nargs: nargs["BLOCK_SIZE"] * 2})
        # test kwargs
        @triton.heuristics({"EVEN_N": lambda nargs: nargs["N"] + 10})
        @triton.heuristics({"EVEN_N": lambda nargs: nargs["EVEN_N"] * 2})
        # test args
        # There are differences here from OSS Triton because we run these functions in Dynamo
        # We don't have access to the .data_ptr() of TensorVariables
        @triton.heuristics({"NDIM_src": lambda nargs: nargs["src"] is None})
        # test that heuristics are applied in the correct order
        @triton.heuristics({"EVEN_N": lambda nargs: nargs["EVEN_N"] - 10})
        @triton.jit
        def heuristics_kernel(
            dst,
            src,
            N,
            BLOCK_SIZE: tl.constexpr,
            EVEN_N: tl.constexpr,
            NDIM_src: tl.constexpr,
        ):
            tl.store(dst, EVEN_N + BLOCK_SIZE)
            tl.store(dst + 1, NDIM_src)

        grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)

        def f(
            dst: torch.Tensor,
            src: torch.Tensor,
            N: int,
        ) -> None:
            grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
            if non_strict:
                torch.library.wrap_triton(heuristics_kernel)[grid](dst, src, N=N)
            else:
                heuristics_kernel[grid](dst, src, N=N)

        if non_strict:
            decorator = torch.library.triton_op(
                f"{libname}::{opname}", mutates_args={"dst"}
            )(f)
        else:
            # we can just pass the function 'f' for dynamo
            decorator = f

        compiled_f = torch.compile(decorator, backend=backend)

        N = 1023
        src = torch.empty(N, device=GPU_TYPE)
        dst = torch.zeros(N, device=GPU_TYPE)

        with inductor_config.patch(
            {"triton.autotune_at_compile_time": autotune_at_compile_time}
        ):
            compiled_f(dst, src, N=N)

        # now let's run without torch.compile to compare
        triton_src = torch.empty(N, device=GPU_TYPE)
        triton_dst = torch.zeros(N, device=GPU_TYPE)
        heuristics_kernel[grid](triton_dst, triton_src, N=N)

        # triton_dst[0].item() is 2120
        # (1023 + 10) * 2 - 10 + BLOCK_SIZE = 2056 + 64 = 2120
        # this is to test that we apply the heuristics in the correct order
        self.assertEqual(triton_dst[0].item(), 2120)
        self.assertEqual(triton_dst[1].item(), 0.0)

        # Results should match
        self.assertEqual(dst[0].item(), triton_dst[0].item())
        self.assertEqual(dst[1].item(), triton_dst[1].item())

        # @triton.heuristics cannot return non-constant values
        # check for the exception
        if not non_strict:

            @triton.autotune(
                configs=[
                    triton.Config(kwargs={"BLOCK_SIZE": 32}),
                ],
                key=["N"],
            )
            # torch.randint(...)[0] will produce a non-constant value
            @triton.heuristics({"EVEN_N": lambda nargs: torch.randint(1, (1, 1))[0]})
            @triton.jit
            def heuristics_kernel(
                dst,
                src,
                N,
                BLOCK_SIZE: tl.constexpr,
                EVEN_N: tl.constexpr,
            ):
                tl.store(dst, N)

            grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)

            def f(
                dst: torch.Tensor,
                src: torch.Tensor,
                N: int,
            ) -> None:
                grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
                heuristics_kernel[grid](dst, src, N=N)

            compiled_f = torch.compile(f, backend=backend, fullgraph=True)
            N = 1023
            src = torch.empty(N, device=GPU_TYPE)
            dst = torch.zeros(N, device=GPU_TYPE)
            msg = "@triton.heuristics must return constant values because configs can only contain constant values."
            with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, msg):
                compiled_f(dst, src, N=N)


common_utils.instantiate_parametrized_tests(KernelTests)
common_utils.instantiate_parametrized_tests(CustomOpTests)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
