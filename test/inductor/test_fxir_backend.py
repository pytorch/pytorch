# Owner(s): ["module: inductor"]
"""
Test the FX IR backend.
"""

import itertools
import operator
import unittest
from collections.abc import Callable
from typing import Optional

import sympy

import torch
import torch._inductor.codegen.common as common
import torch.utils._pytree as pytree
from torch._dynamo.exc import BackendCompilerFailed
from torch._dynamo.utils import same
from torch._higher_order_ops.triton_kernel_wrap import triton_kernel_wrapper_mutation
from torch._inductor import config
from torch._inductor.codegen.cpp import CppScheduling
from torch._inductor.codegen.triton import TritonScheduling
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.codegen.wrapper_fxir import (
    FxConverter,
    replace_floor_div,
    WrapperFxCodegen,
)
from torch._inductor.test_case import TestCase as InductorTestCase
from torch.export import Dim
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_GPU,
    patch_custom_fallback_pass,
    requires_gpu,
    TRITON_HAS_CPU,
)
from torch.utils._sympy.functions import FloorDiv


try:
    from .test_control_flow import CondModels
except ImportError:
    from test_control_flow import (
        CondModels,  # @manual=fbcode//caffe2/test/inductor:control_flow-library
    )

if HAS_GPU:
    import triton
    import triton.language as tl

    from torch.testing._internal.triton_utils import add_kernel_2d_autotuned

test_config = {
    "compile_threads": 1,
    "alignment_asserts": False,
    "size_asserts": False,
    "scalar_asserts": False,
    "nan_asserts": False,
}


@requires_gpu()
@config.patch(test_config)
@instantiate_parametrized_tests
class FxirTestCase(InductorTestCase):
    device = GPU_TYPE

    def _count_ops(self, gm: torch.fx.GraphModule, target: Callable) -> int:
        return len(gm.graph.find_nodes(op="call_function", target=target))

    def _run_and_capture_graphs(self, opt, args) -> torch.fx.GraphModule:
        gms = []

        orig_generate = FxConverter.generate

        def generate(self) -> torch.fx.GraphModule:
            nonlocal gms
            gm = orig_generate(self)
            gms.append(gm)
            return gm

        with unittest.mock.patch.object(
            torch._inductor.codegen.wrapper_fxir.FxConverter, "generate", generate
        ):
            opt(*args)

        return gms

    def _compile_and_check(
        self,
        func,
        args,
        expected_num_triton_kernels: int = 1,
        metadata_only: bool = False,
        compile_kwargs: Optional[dict] = None,
    ):
        if compile_kwargs is None:
            compile_kwargs = {}

        opt = torch.compile(func, **compile_kwargs)

        # Get the FX graph from the backend.
        gms = self._run_and_capture_graphs(opt, args)

        # Check the code for triton kernels.
        num_kernels = sum(
            self._count_ops(gm, triton_kernel_wrapper_mutation) for gm in gms
        )
        self.assertEqual(num_kernels, expected_num_triton_kernels)

        # Check accuracy.
        result = opt(*args)
        ref = func(*args)
        if metadata_only:
            # When we only want to check metadata, fill in zeros for tensor data.
            ref, result = tuple(
                pytree.tree_map(torch.zeros_like, x) for x in (ref, result)
            )

        self.assertTrue(same(ref, result))

        return gms

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # Register the FX backend, storing the default for later.
        common.init_backend_registration()
        cls._default_backend = common.device_codegens[cls.device]
        common.register_backend_for_device(
            cls.device, TritonScheduling, WrapperFxCodegen
        )

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

        # Restore the default backend.
        common.device_codegens[cls.device] = cls._default_backend

    def test_basic(self):
        args = [torch.randn(8, device=self.device) for _ in range(2)]
        self._compile_and_check(torch.add, args)

    def test_device_type(self):
        """
        Test that we allocate on a device type instead of a specific index.
        """
        # Pass in a tensor on an indexed device.
        device_runtime = getattr(torch, self.device)
        indexed_device = torch.device(self.device, device_runtime.current_device())
        args = [torch.randn(8, device=indexed_device) for _ in range(2)]
        (gm,) = self._compile_and_check(torch.add, args)
        (empty_strided,) = gm.graph.find_nodes(
            op="call_function", target=torch.empty_strided
        )

        # Check that the device of the output allocation is not indexed.
        output_device = torch.device(empty_strided.kwargs["device"])
        self.assertIs(output_device.index, None)
        self.assertEqual(output_device.type, indexed_device.type)

    def test_multiple_kernels(self):
        def foo(x, y):
            return x.sum() + y.sum()

        args = [torch.randn(length, device=self.device) for length in [517, 1029]]
        self._compile_and_check(foo, args, expected_num_triton_kernels=2)

    def test_free(self):
        """
        Test a program that frees a buffer which is no longer in use.
        """

        def foo(x, y, z):
            w = x.sum() + y
            return z.sum() + w.sum()

        args = [torch.randn(length, device=self.device) for length in [517, 1029, 123]]
        (gm,) = self._compile_and_check(foo, args, expected_num_triton_kernels=3)

        # Check the generated code for frees.
        num_frees = gm.code.count("= None")
        self.assertGreater(num_frees, 0)

    def test_extern(self):
        """
        Test a program that calls an extern kernel.
        """

        def foo(x, y):
            return x @ y + y.sum()

        args = [
            torch.randn(size, device=self.device) for size in [(129, 129), (129, 1)]
        ]
        (gm,) = self._compile_and_check(foo, args, expected_num_triton_kernels=1)

        # Check for the extern kernel
        num_extern = self._count_ops(gm, torch.ops.aten.addmm.out)
        self.assertEqual(num_extern, 1)

    def test_fallback(self):
        """
        Test a program that calls aten fallbacks.
        """

        def foo(x):
            batch1 = torch.randn(2, 3, 5, device=self.device)
            batch2 = torch.randn(2, 5, 4, device=self.device)
            return torch.addbmm(x, batch1, batch2)

        args = (torch.randn(3, 4, device=self.device),)

        # Since the program has a random output, just check metadata.
        # Don't check for an exact value.
        (gm,) = self._compile_and_check(
            foo, args, expected_num_triton_kernels=2, metadata_only=True
        )

        # Check for the fallback kernel.
        num_fallback = self._count_ops(
            gm, torch.ops.aten.randint.low_out
        ) + self._count_ops(gm, torch.ops.aten.addbmm.default)
        self.assertEqual(num_fallback, 2)

    def test_cat_inputs(self):
        """
        Test concatenation of graph inputs.
        """

        def foo(x, y):
            return torch.cat((x, y)) + 1

        args = [torch.randn(8, device=self.device) for _ in range(2)]
        self._compile_and_check(foo, args, expected_num_triton_kernels=1)

    def test_cat_views(self):
        """
        Test concatenation with multiple kernels writing to the same buffer.
        """

        def foo(x, y):
            a = x - 2
            b = y.sum(0, keepdim=True)
            c = torch.cat((a, b)).clone()
            return a, b, c

        args = [torch.randn(8, device=self.device) for _ in range(2)]
        (gm,) = self._compile_and_check(foo, args, expected_num_triton_kernels=2)

        def get_offset(node: torch.fx.Node) -> int:
            (input_, shape, stride, offset) = node.args
            assert isinstance(offset, int)
            return offset

        # Check for 2 views, one of which is offset.
        as_strided_nodes = list(
            gm.graph.find_nodes(op="call_function", target=torch.as_strided)
        )
        self.assertEqual(len(as_strided_nodes), 2)
        num_offset_views = sum(get_offset(node) > 0 for node in as_strided_nodes)
        self.assertEqual(num_offset_views, 1)

    def test_cat_to_alloc(self):
        """
        Test concatenation that's optimized out to an allocation.
        """
        length = 8

        def foo(x):
            y, z = tuple(
                torch.arange(length // 2, device=self.device) for _ in range(2)
            )
            return x + torch.cat((y, z))

        args = [torch.randn(length, device=self.device)]
        (gm,) = self._compile_and_check(foo, args, expected_num_triton_kernels=1)

        # Expect a single allocation, even though eager mode would use 2.
        num_allocs = self._count_ops(gm, torch.empty_strided)
        self.assertEqual(num_allocs, 1)

    def test_cat_reinterpret_view(self):
        """
        Test torch.cat using ReinterpretView.
        """
        length = 8

        def foo(x):
            y, z = tuple(torch.randn(length // 2, device=self.device) for _ in range(2))
            return x + torch.cat((y, z))

        args = [torch.randn(length, device=self.device)]

        # Since this test generates random numbers, check metadata only.
        (gm,) = self._compile_and_check(
            foo, args, expected_num_triton_kernels=3, metadata_only=True
        )

        # Check for as_strided. We map ReinterpretView to this.
        num_as_strided = self._count_ops(gm, torch.as_strided)
        self.assertEqual(num_as_strided, 2)

    def test_reshape_output(self):
        """
        Test reshaping the output, which maps to a ReinterpretView.
        """

        def foo(x, y):
            return torch.reshape(x + y, (8,))

        args = [torch.randn((2, 4), device=self.device) for _ in range(2)]
        (gm,) = self._compile_and_check(foo, args, expected_num_triton_kernels=1)

        # Check for as_strided. We map ReinterpretView to this.
        num_as_strided = self._count_ops(gm, torch.as_strided)
        self.assertEqual(num_as_strided, 1)

    def test_reshape_fallback(self):
        """
        Test falling back to aten.reshape. This uses a custom pass to enable more fallbacks.
        """

        def always_fallback(node: torch.fx.Node) -> bool:
            return True

        def foo(x):
            return x.reshape((2, 5))

        args = (torch.randn(10, device=self.device),)
        with patch_custom_fallback_pass(always_fallback):
            (gm,) = self._compile_and_check(foo, args, expected_num_triton_kernels=0)

        # Check for the reshape.
        (reshape_node,) = gm.graph.find_nodes(
            op="call_function", target=torch.ops.aten.reshape.default
        )

    def test_extern_multi_output(self):
        """
        Test an extern kernel with multiple outputs.
        Also test a graph with multiple outputs.
        """

        def foo(x):
            top, idx = torch.topk(x, 2)
            return top + 1, idx * 2

        args = [torch.randn(8, device=self.device)]
        (gm,) = self._compile_and_check(foo, args, expected_num_triton_kernels=2)

        # Check for multiple kernel outputs via getitems.
        num_getitems = self._count_ops(gm, operator.getitem)
        self.assertEqual(num_getitems, 2)

        # Check for multiple graph outputs.
        output_node = gm.graph.find_nodes(op="output")[0]
        self.assertEqual(len(output_node.args[0]), 2)

    def test_duplicate_input(self):
        """
        Test duplicated inputs. This will collapse into a single input in the GM.
        """

        args = [torch.randn(4, device=self.device)] * 2
        (gm,) = self._compile_and_check(torch.add, args, expected_num_triton_kernels=1)

        num_placeholders = len(gm.graph.find_nodes(op="placeholder"))
        self.assertEqual(num_placeholders, 1)

    def test_backward(self):
        """
        Test a program with a backward pass.
        """

        x = torch.ones(5, device=self.device)  # input tensor
        y = torch.zeros(3, device=self.device)  # expected output
        w = torch.randn(5, 3, requires_grad=True, device=self.device)
        b = torch.randn(3, requires_grad=True, device=self.device)

        def foo(x, y):
            z = torch.matmul(x, w) + b
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
            loss.backward()
            return w.grad, b.grad

        # Expect separate forward and backward graphs.
        (forward_gm, backward_gm) = self._compile_and_check(
            foo, (x, y), expected_num_triton_kernels=3
        )

    def test_custom_compiler(self):
        """
        Test a derived backend with a custom compiler.
        """
        offset = 1

        class CustomWrapperCodegen(WrapperFxCodegen):
            def compile_graph(self, gm):
                def compiled_fn(*args):
                    # Adds an offset to the program's outputs.
                    outputs = gm(*args)
                    return pytree.tree_map(lambda x: x + 1, outputs)

                return compiled_fn

        args = [torch.randn(8, device=self.device) for _ in range(2)]
        custom_backend = common.DeviceCodegen(
            TritonScheduling, CustomWrapperCodegen, None
        )
        with unittest.mock.patch.dict(
            common.device_codegens, {self.device: custom_backend}
        ):
            func = torch.add
            opt = torch.compile(func)
            result = opt(*args)

        # Check the output is offset from eager mode.
        ref = func(*args)
        self.assertFalse(same(result, ref))
        self.assertNotEqual(offset, 0)
        self.assertTrue(same(result - offset, ref))

    def test_dynamic_shapes_and_strides(self):
        """
        Test a graph with dynamic shapes and strides.
        """

        static_dims = (8, 8)

        def get_input():
            full_size = (16, 8)
            full = torch.randn(full_size, device=self.device)
            view = torch.as_strided(full, static_dims, full.stride())
            return view

        func = torch.add
        args = [get_input() for _ in range(2)]
        (gm,) = self._compile_and_check(func, args, compile_kwargs={"dynamic": True})

        # Check for a symbolic output shape.
        (empty_strided,) = gm.graph.find_nodes(
            op="call_function", target=torch.empty_strided
        )
        example_tensor = empty_strided.meta["val"]
        symbolic_dims = example_tensor.shape
        self.assertEqual(len(symbolic_dims), len(static_dims))

        # Check for symbolic output strides.
        (stride, one) = example_tensor.stride()
        self.assertEqual(one, sympy.S.One)

        # Find the size symbols, and check for a corresponding placeholders defining them.
        for symbol in itertools.chain(symbolic_dims, [stride]):
            self.assertTrue(isinstance(symbol, torch.SymInt))
            (placeholder,) = [
                node
                for node in gm.graph.find_nodes(op="placeholder")
                if node.name == str(symbol)
            ]
            self.assertEqual(placeholder.meta["val"], symbol)

    @parametrize(
        "shape",
        [
            (20,),
            (50, 30),
            (50, 30, 40),
        ],
    )
    @torch._inductor.config.patch(
        {
            "pad_dynamic_shapes": True,
            "comprehensive_padding": True,
            "padding_alignment_bytes": 32,
            "pad_outputs": True,
        }
    )
    def test_dynamic_shapes_with_padding(self, shape):
        """
        Test a graph with dynamic shapes with padding.
        """

        def get_input(shape):
            pad_size = list(shape)
            pad_size[-1] = ((shape[-1] + 7) // 8) * 8
            pad = torch.randn(pad_size, dtype=torch.float32, device=self.device)
            view = torch.as_strided(pad, shape, pad.stride())
            return view

        args = [get_input(shape) for _ in range(2)]
        (gm,) = self._compile_and_check(
            torch.add, args, compile_kwargs={"dynamic": True}
        )

        # Check for a symbolic output shape.
        (empty_strided,) = gm.graph.find_nodes(
            op="call_function", target=torch.empty_strided
        )
        example_tensor = empty_strided.meta["val"]
        symbolic_dims = example_tensor.shape
        symbolic_strides = example_tensor.stride()

        align_elems = 32 // args[0].dtype.itemsize
        expected_strides = [1 for _ in range(len(shape))]
        for i in range(len(shape) - 1, 0, -1):
            expected_strides[i - 1] = align_elems * (
                ((expected_strides[i] * symbolic_dims[i]) + align_elems - 1)
                // align_elems
            )
        for i, j in zip(symbolic_strides, expected_strides):
            self.assertEqual(i, j)

    def test_dynamic_shapes_precomputed_size(self):
        """
        Test dynamic shapes where a kernel's size arg is precomputed.
        """
        func = torch.add
        args = [
            torch.randn(shape, device=self.device) for shape in [(7, 12, 9), (7, 1, 1)]
        ]
        (gm,) = self._compile_and_check(func, args, compile_kwargs={"dynamic": True})

        # Check for the precomputed size arg.
        (triton_node,) = gm.graph.find_nodes(
            op="call_function", target=triton_kernel_wrapper_mutation
        )
        self.assertIn("ks0", triton_node.kwargs["kwargs"])

    def test_dynamic_launch_grid_calc(self):
        """
        Test the dynamic launch grid calculation.
        """

        func = torch.add
        args = [torch.randn(shape, device=self.device) for shape in [(7, 12), (7, 1)]]
        (gm,) = self._compile_and_check(func, args, compile_kwargs={"dynamic": True})

        # Check for the precomputed size arg.
        (triton_node,) = gm.graph.find_nodes(
            op="call_function", target=triton_kernel_wrapper_mutation
        )
        self.assertIn("grid", triton_node.kwargs)
        self.assertIn("xnumel", triton_node.kwargs["kwargs"])
        self.assertIn("XBLOCK", triton_node.kwargs["kwargs"])
        grid = triton_node.kwargs["grid"][0]
        xnumel = triton_node.kwargs["kwargs"]["xnumel"].meta["val"]
        xblock = triton_node.kwargs["kwargs"]["XBLOCK"]
        self.assertEqual(grid[0].meta["val"], -(-xnumel // xblock))
        self.assertEqual(grid[1], 1)
        self.assertEqual(grid[2], 1)

    @config.patch({"trace.enabled": True})
    @unittest.mock.patch("torch._inductor.debug.DebugFormatter.output_code")
    def test_debug(self, mock_output_code):
        # Compile in debug mode.
        args = [torch.randn(11, device=self.device) for _ in range(2)]
        self._compile_and_check(torch.sub, args)

        # Check the output code for a Triton kernel call.
        mock_output_code.assert_called_once()
        (output_filename,) = mock_output_code.call_args.args
        with open(output_filename) as f:
            output_code = f.read()
        self.assertIn("triton_kernel_wrapper_mutation", output_code)

    @parametrize(
        "const",
        (1, 1.5),
    )
    def test_export_const_placeholder(self, const):
        """
        Test that we can compile a graph coming from torch.export with a constant input.
        """

        class TestModule(torch.nn.Module):
            def forward(self, x, y):
                return x - y

        args = (torch.randn(8, device=self.device), const)
        mod = TestModule()
        export_gm = torch.export.export(mod, args).module()

        def compile_module(*inps):
            torch._inductor.compile(export_gm, inps)

        (inductor_gm,) = self._run_and_capture_graphs(compile_module, args)
        result = inductor_gm(*args)
        ref = mod(*args)

        self.assertTrue(same(ref, result))

    def test_scatter_fallback_scalar_src(self):
        """
        Test a special case where ScatterFallback takes a scalar 'src' argument.
        """

        def foo(input_):
            dim = 0
            src = 1.5
            return torch.ops.aten.scatter(input_, dim, index, src)

        length = 8
        index = torch.randint(length, (length,), device=self.device)
        input_ = torch.randn(length, device=self.device)
        with DeterministicGuard(True):
            (gm,) = self._compile_and_check(
                foo,
                (input_,),
            )

        # Check for the fallback op.
        num_fallback = self._count_ops(gm, torch.ops.aten.scatter_.value)
        self.assertEqual(num_fallback, 1)

    def test_index_put_fallback(self):
        """
        Test the deterministic fallback for index_put.
        """
        length = 8
        out, values = [torch.randn(length, device=self.device) for _ in range(2)]
        indices = (torch.randint(length, (length,), device=self.device),)
        accumulate = True
        with DeterministicGuard(True):
            (gm,) = self._compile_and_check(
                torch.index_put,
                (out, indices, values, accumulate),
                expected_num_triton_kernels=1,
            )

        # Check for the fallback op.
        self.assertEqual(self._count_ops(gm, torch.ops.aten.index_put_.default), 1)

    def test_scatter_reduce_fallback(self):
        """
        Test the customized wrapper codegen for ScatterFallback ops.
        """
        fallback_op = torch.ops.aten.scatter_reduce_.two

        def foo(out, index, src):
            dim = 0
            out = fallback_op(out, dim, index, src, reduce="amax", include_self=False)
            return out + 1

        length = 8
        out, src = [torch.randn(length, device=self.device) for _ in range(2)]
        index = torch.randint(length, (length,), device=self.device)
        (gm,) = self._compile_and_check(
            foo, (out, index, src), expected_num_triton_kernels=2
        )

        # Check for the fallback.
        self.assertEqual(self._count_ops(gm, fallback_op), 1)

    @parametrize("pred", (False, True))
    def test_cond_subgraph(self, pred: bool):
        """
        Test a model with subgraphs.
        """

        def foo(pred, x):
            return torch.cond(pred, torch.cos, torch.sin, [x]) + 1

        x = torch.randn((2, 3), device=self.device)
        pred_tensor = torch.tensor([pred], device=self.device)
        gm = self._compile_and_check(
            foo, [pred_tensor, x], expected_num_triton_kernels=3
        )[-1]

        # Check for subgraphs.
        subgm_getattrs = list(gm.graph.find_nodes(op="get_attr"))
        self.assertEqual(len(subgm_getattrs), 2)
        for subgm_getattr in subgm_getattrs:
            target = subgm_getattr.name
            self.assertTrue(isinstance(getattr(gm, target), torch.fx.GraphModule))

    @parametrize("pred", (False, True))
    def test_cond_no_operands(self, pred: bool):
        """
        Test torch.cond when the subgraphs take no inputs.
        """

        length = 8

        def true_fn():
            return torch.zeros(length, device=self.device)

        def false_fn():
            return true_fn() + 5

        def foo(pred):
            return torch.cond(pred, true_fn, false_fn, ())

        pred_tensor = torch.tensor([pred], device=self.device)
        self._compile_and_check(foo, [pred_tensor], expected_num_triton_kernels=2)

    def test_cpp_raises(self):
        """
        Test the C++ CPU backend. C++ kernels are not yet supported, so for now check
        that we get the expected exception.
        """

        def foo(x, y):
            return x + y * 5

        device = torch.device("cpu")
        args = [torch.randn(5, device=device) for _ in range(2)]

        cpp_backend = common.DeviceCodegen(CppScheduling, WrapperFxCodegen, None)
        with (
            unittest.mock.patch.dict(
                common.device_codegens, {device.type: cpp_backend}
            ),
            self.assertRaisesRegex(BackendCompilerFailed, "Triton"),
        ):
            self._compile_and_check(foo, args)

    @parametrize("enable_tuning", (False, True))
    @parametrize("use_dynamic_shapes", (False, True))
    def test_autotune(self, use_dynamic_shapes: bool, enable_tuning: bool):
        orig_run = torch._inductor.runtime.triton_heuristics.CachingAutotuner.run
        called = False

        def run(*args, **kwargs):
            nonlocal called
            called = True
            return orig_run(*args, **kwargs)

        args = [torch.randn(8, device=self.device) for _ in range(2)]

        with (
            config.patch("triton.autotune_at_compile_time", enable_tuning),
            unittest.mock.patch.object(
                torch._inductor.runtime.triton_heuristics.CachingAutotuner, "run", run
            ),
        ):
            # Compile and check that the tuner was called.
            self.assertFalse(called)
            (gm,) = self._compile_and_check(
                torch.mul, args, compile_kwargs={"dynamic": use_dynamic_shapes}
            )
            self.assertEqual(called, enable_tuning)

        # Check for a symbolic output shape.
        (empty_strided,) = gm.graph.find_nodes(
            op="call_function", target=torch.empty_strided
        )
        (shape, stride) = empty_strided.args
        if use_dynamic_shapes:
            self.assertEqual(type(shape[0]), torch.fx.Node)

    def test_custom_triton(self):
        @triton.jit
        def add_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)

        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            output = torch.empty_like(x)
            n_elements = output.numel()

            def grid(meta):
                return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
            return output

        args = [torch.randn(32, device=self.device) for _ in range(2)]
        self._compile_and_check(add, args)

    def test_output_slice_view(self):
        """
        Test when the output is a view of the input.
        The sliced strides create a TensorBox in the output IR.
        """

        def foo(x):
            return x[0:2:2].T[3:].squeeze(0)

        args = [torch.rand([4, 4, 4, 4], device=self.device)]
        self._compile_and_check(foo, args, expected_num_triton_kernels=0)

    def test_fallback_tuple_constant_arg(self):
        """
        Test a fallback op with tuple constant argument.
        Check that tuple arguments are not flattened during codegen.
        """

        def foo(x):
            # permute with a tuple argument
            return torch.permute(x, (0, 2, 1))

        # Use complex64 to force permute to become a fallback op
        args = [torch.randn(2, 3, 4, dtype=torch.complex64, device=self.device)]

        (gm,) = self._compile_and_check(foo, args, expected_num_triton_kernels=0)

        # Check for the fallback kernel with permute
        num_fallback = self._count_ops(gm, torch.ops.aten.permute.default)
        self.assertEqual(num_fallback, 1)

        # Verify the permute node has the correct tuple argument
        permute_node = next(
            iter(
                gm.graph.find_nodes(
                    op="call_function", target=torch.ops.aten.permute.default
                )
            )
        )

        # The second argument should be the permutation (0, 2, 1)
        # Check that it's not flattened
        perm_arg = permute_node.args[1]
        self.assertIsInstance(
            perm_arg, list, "Permutation argument should not be flattened"
        )
        self.assertEqual(len(perm_arg), 3)
        self.assertEqual(tuple(perm_arg), (0, 2, 1))


@instantiate_parametrized_tests
class AOTFxirTestCase(InductorTestCase):
    device = GPU_TYPE

    def check(
        self, model, inp, dynamic_shapes=None, strict=False
    ) -> torch.fx.GraphModule:
        with torch.no_grad():
            ep = torch.export.export(
                model, inp, dynamic_shapes=dynamic_shapes, strict=strict
            )
            gm = torch._inductor.aot_compile(
                ep.module(), inp, options={"fx_wrapper": True, **test_config}
            )
            # Flatten args for fx_wrapper gm
            flat_args, _ = pytree.tree_flatten(inp)
            self.assertTrue(same(model(*inp), gm(*flat_args)))

            for node in gm.graph.nodes:
                if (
                    node.op == "call_function"
                    and node.target != triton_kernel_wrapper_mutation
                ):
                    self.assertTrue(node.meta.get("val", None) is not None)

            return gm

    def test_aoti_fx_add(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        inp = (torch.ones(3, device=self.device), torch.ones(3, device=self.device))
        self.check(M(), inp)

    def test_aoti_fx_const(self):
        class M(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.device = device
                self.a = torch.nn.Parameter(torch.ones(3, device=self.device))
                self.b = torch.ones(3, device=self.device)

            def forward(self, x, y):
                return x + y + self.a + self.b + torch.tensor(3, device=self.device)

        inp = (torch.ones(3, device=self.device), torch.ones(3, device=self.device))
        self.check(M(self.device), inp)

    def test_aoti_fx_linear(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                return self.linear(x)

        inp = (torch.ones(3, 3, device=self.device),)
        self.check(M().to(self.device), inp)

    def test_aoti_fx_dynamic(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x + y

        inp = (torch.ones(3, device=self.device), torch.ones(3, device=self.device))
        self.check(
            M().to(device=self.device),
            inp,
            dynamic_shapes=({0: Dim.DYNAMIC}, {0: Dim.DYNAMIC}),
        )

    def test_custom_triton_autotune_dynamic(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                output = torch.zeros_like(x)
                x_elements = output.size()[0]
                y_elements = output.size()[1]

                def grid(meta):
                    return (
                        triton.cdiv(x_elements, meta["BLOCK_SIZE_X"]),
                        triton.cdiv(y_elements, meta["BLOCK_SIZE_Y"]),
                    )

                add_kernel_2d_autotuned[grid](x, y, output, x_elements, y_elements)

                return output

        num_dims = 2
        dims = [10] * num_dims
        x = torch.randn(*dims, device=self.device)
        y = torch.randn(*dims, device=self.device)
        dim0_x = Dim("dim0_x", min=1, max=10)
        dim0_y = Dim("dim0_y", min=1, max=10)
        dynamic_shapes = {"x": {0: dim0_x}, "y": {0: dim0_y}}
        self.check(
            Model().to(device=self.device),
            (x, y),
            dynamic_shapes=dynamic_shapes,
            strict=True,
        )

    def test_custom_backend(self):
        """
        Test registering a custom FX backend.
        """
        called = False

        class CustomWrapperCodegen(WrapperFxCodegen):
            def compile_graph(self, gm):
                """
                Simply records whether this override was called.
                """
                nonlocal called
                called = True
                return super().compile_graph(gm)

        class M(torch.nn.Module):
            def forward(self, x):
                return x + 1

        # Register a custom FX backend.
        custom_backend = common.DeviceCodegen(
            TritonScheduling,
            PythonWrapperCodegen,
            fx_wrapper_codegen=CustomWrapperCodegen,
        )
        with unittest.mock.patch.dict(
            common.device_codegens, {self.device: custom_backend}
        ):
            # The backend should not have been called yet.
            self.assertFalse(called)

            inp = (torch.randn(8, device=self.device),)
            self.check(M().to(self.device), inp)

        # Now the backend should have been called.
        self.assertTrue(called)

    @parametrize(
        "expr",
        [
            (2 * Dim("x") + 1),
            (Dim("x", min=3) - 3),
        ],
    )
    def test_dynamic_input_expr(self, expr: sympy.Expr):
        """
        Test dynamic shapes with a nontrivial input expression.
        """

        class M(torch.nn.Module):
            def forward(self, x):
                return x.reshape(x.shape[0] * x.shape[1]) + x.shape[1]

        dynamic_shapes = {"x": {0: expr}}
        inp = (torch.randn((5, 4), device=self.device),)
        gm = self.check(M().to(self.device), inp, dynamic_shapes=dynamic_shapes)

        # Check for dynamic size ops.
        self.assertEqual(
            len(
                gm.graph.find_nodes(
                    op="call_function", target=torch.ops.aten.sym_size.int
                )
            ),
            1,
        )

    @parametrize("pred", (False, True))
    def test_cond_multi_inputs_and_outputs(self, pred):
        """
        Test torch.cond and check the output graphs.
        """

        class M(torch.nn.Module):
            def forward(self, pred, x, y):
                def true_fn(x, y):
                    return torch.tanh(x), torch.relu(y)

                def false_fn(x, y):
                    return tuple(t / 2 for t in true_fn(x, y))

                return torch.cond(pred, true_fn, false_fn, (x, y))

        pred = torch.tensor([True], device=self.device)
        (x, y) = [torch.randn(8, device=self.device) for _ in range(2)]
        gm = self.check(M(), (pred, x, y))

        # Check the graph.
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(arg0_1, true_graph_0, false_graph_0, (arg1_1, arg2_1));  arg0_1 = true_graph_0 = false_graph_0 = arg1_1 = arg2_1 = None
    buf1 = cond[0]
    buf2 = cond[1];  cond = None
    return [buf1, buf2]""",  # noqa: B950
        )

    def test_dims_dynamic_outer_static_padded_inner(self):
        """
        Test padding on inner dimensions, with dynamic outer dimensions.
        """

        class M(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        def get_input_padded_inner(shape):
            full_shape = shape[:-1] + (shape[-1] * 2,)
            full = torch.randn(full_shape, dtype=torch.float32, device=self.device)
            view = torch.as_strided(full, shape, full.stride())
            return view

        shape = (4, 4, 4)
        args = tuple(get_input_padded_inner(shape) for _ in range(2))
        self.check(
            M(),
            args,
            dynamic_shapes=({0: Dim.DYNAMIC, 1: Dim.DYNAMIC, 2: Dim.STATIC},) * 2,
        )

    @parametrize("length", (4, 8))
    def test_cond_dynamic_shape_pred_scalar_closure(self, length: int):
        """
        Test cond using a predicate computed from dynamic shapes.
        Also test a dynamic scalar computed outside the branches.
        """

        class M(torch.nn.Module):
            def forward(self, x, y):
                z = x.reshape(-1)
                a = y.shape[0]

                def true_fn(x):
                    return x + a

                def false_fn(x):
                    return true_fn(x) / 2

                return torch.cond(x.shape[0] > 5, true_fn, false_fn, (z,))

        (x, y) = [
            torch.randn(shape, device=self.device)
            for shape in [(length // 2,) * 2, (length,)]
        ]
        dynamic_shapes = {
            "x": {0: Dim.DYNAMIC},
            "y": {0: Dim.DYNAMIC},
        }
        self.check(M(), (x, y), dynamic_shapes=dynamic_shapes)

    def test_dynamic_scalar_output(self):
        """
        Test an output scalar from dynamic shapes.
        """

        class M(torch.nn.Module):
            def forward(self, x):
                return x.shape[0] * 3

        x = torch.randn(7, device=self.device)
        self.check(M(), (x,), dynamic_shapes=({0: Dim.DYNAMIC},))

    @parametrize("dynamic", (False, True))
    @parametrize("input_", (1.5, 2, False))
    def test_item(self, input_, dynamic: bool):
        """
        Test calling Tensor.item.
        """

        class M(torch.nn.Module):
            def forward(self, x):
                return x[1].item()

        x = torch.tensor((input_,) * 10)
        d = Dim("s0", min=1)
        dynamic_shapes = ({0: 2 * d},) if dynamic else None
        self.check(M(), (x,), dynamic_shapes=dynamic_shapes)

    @parametrize("pred", (False, True))
    def test_mismatched_branch_dynamic(self, pred: bool):
        """
        Test cond branches with mismatched dynamic shapes.
        """

        # Apply an offset to guarantee the truith of the predicate.
        pred_offset = 1 if pred else -1

        inputs = [
            torch.tensor([pred], device=self.device),
        ] + [torch.randn(10, 20, device=self.device) + pred_offset for _ in range(3)]
        dim0_a = Dim("s0", min=4, max=1024)
        dim0_b = Dim("s1", min=4, max=1024)
        dynamic_shapes = {
            "p": {},
            "x": {0: dim0_a, 1: None},
            "y": {0: dim0_b, 1: None},
            "z": {0: dim0_a, 1: None},
        }

        self.check(
            CondModels.MismatchedOutputSize(),
            tuple(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    def test_const_folded_subgraph(self):
        """
        If a graph only contains a call_module node to a subgraph,
        where the subgraph can be const-folded away,
        validate the fake mode used in FXConverter generation is not None.
        """
        device = self.device
        shape = (5, 10)

        class Submodule(torch.nn.Module):
            def forward(self):
                return torch.randn(*shape, device=device) + 1

        # Create a parent graph with this module as a subgraph and output
        ep = torch.export.export(Submodule(), ())
        parent_graph = torch.fx.Graph()
        call_mod = parent_graph.call_module("sub", args=())
        get_item = parent_graph.call_function(
            operator.getitem, args=(call_mod, slice(None))
        )
        parent_graph.output((get_item,))
        parent = torch.fx.GraphModule({"sub": ep.module()}, parent_graph)

        # Verify FXConverter.generate uses non-null fake mode
        # Intercept _set_node_metadata_hook to ensure fake_mode is not None
        orig_set_hook = torch._inductor.codegen.wrapper_fxir._set_node_metadata_hook
        called = False

        def mock_set_hook(gm: torch.fx.GraphModule, fn):
            nonlocal called
            called = True
            # Please update this check if `fake_mode` is
            # no longer used in FXConverter call to _node_metadata_hook
            self.assertTrue("fake_mode" in fn.keywords)
            self.assertIsNotNone(fn.keywords["fake_mode"])
            return orig_set_hook(gm, fn)

        self.assertFalse(called)
        with unittest.mock.patch.object(
            torch._inductor.codegen.wrapper_fxir,
            "_set_node_metadata_hook",
            mock_set_hook,
        ):
            args = ()
            compiled = torch._inductor.aot_compile(
                parent, args, options={"fx_wrapper": True}
            )
            self.assertTrue(called)

            compiled_out = compiled(*args)
            self.assertEqual(compiled_out.shape, shape)

    def test_reshape_dynamic_ph(self):
        """
        Test dynamic scalars using SymInts placeholder
        """

        class TestModule(torch.nn.Module):
            def forward(self, x, shape):
                return torch.reshape(x, shape) + 2

        ds = {
            "x": (torch.export.Dim.AUTO, torch.export.Dim.AUTO),
            "shape": [torch.export.Dim.AUTO, torch.export.Dim.AUTO],
        }
        args = (torch.randn((12, 14), device=self.device), [6, 28])
        self.check(TestModule(), args, ds)

    def test_reshape_dynamic_tmd(self):
        """
        Test dynamic reshape using shape dependent information
        """

        class TestModule(torch.nn.Module):
            def forward(self, x):
                new_shape = [x.shape[0] // 2, x.shape[1] * 2]
                return torch.reshape(x, new_shape) + 2

        ds = {
            "x": (torch.export.Dim.AUTO, torch.export.Dim.AUTO),
        }
        args = (torch.randn((12, 14), device=self.device),)
        self.check(TestModule(), args, ds)


class TestReplaceFloorDiv(InductorTestCase):
    """
    Tests for floor -> FloorDiv conversion.
    """

    def _check(self, expr: sympy.Expr) -> sympy.Expr:
        # Check that we started with floor's.
        num_floors = expr.count(sympy.floor)
        self.assertGreater(num_floors, 0)

        replaced = replace_floor_div(expr)

        # Check that all floor's were replaced.
        # We should have no more new FloorDiv's than floor's in the original expression,
        # although we can have less due to simplification.
        self.assertEqual(replaced.count(sympy.floor), 0)
        self.assertLessEqual(
            replaced.count(FloorDiv) - expr.count(FloorDiv), num_floors
        )

        def expand_floor_div(
            numerator: sympy.Expr, denominator: sympy.Expr
        ) -> sympy.Expr:
            return sympy.floor(numerator / denominator)

        # Expand FloorDiv back into floor and check for equality.
        self.assertEqual(
            *[
                sympy.simplify(e.replace(FloorDiv, expand_floor_div))
                for e in (replaced, expr)
            ]
        )

        return replaced

    def test_rewrite_floor_div_mul_pow(self):
        x, y = sympy.symbols("x y")
        expr = sympy.floor(x / y)
        self.assertEqual(expr.count(FloorDiv), 0)
        self.assertEqual(expr.count(sympy.core.mul.Mul), 1)
        self.assertEqual(expr.count(sympy.Pow), 1)

        rewritten = self._check(expr)
        self.assertTrue(isinstance(rewritten, FloorDiv))
        self.assertEqual(rewritten.args, (x, y))

    def test_rewrite_floor_div_mul_rational(self):
        x = sympy.Symbol("x")
        expr = sympy.floor(x / 5)
        self.assertEqual(expr.count(FloorDiv), 0)
        self.assertEqual(expr.count(sympy.core.mul.Mul), 1)
        self.assertEqual(expr.count(sympy.Rational), 1)

        rewritten = self._check(expr)
        self.assertTrue(isinstance(rewritten, FloorDiv))
        self.assertEqual(rewritten.args, (x, 5))

    def test_no_rewrite_div(self):
        x, y = sympy.symbols("x y")
        expr = x / y
        self.assertEqual(expr.count(FloorDiv), 0)

        rewritten = replace_floor_div(expr)
        self.assertEqual(rewritten, expr)

    def test_rewrite_floor_div_nested(self):
        x, y = sympy.symbols("x y")
        expr = sympy.floor((sympy.floor(x / 5) + 1) / y)
        self.assertEqual(expr.count(FloorDiv), 0)

        rewritten = self._check(expr)
        self.assertEqual(rewritten.count(FloorDiv), 2)

    def test_rewrite_floor_div_rational_const(self):
        expr = sympy.floor(sympy.S.One / 5, evaluate=False)
        self.assertEqual(expr.count(FloorDiv), 0)
        self.assertEqual(expr.count(sympy.Mul), 0)
        self.assertEqual(expr.count(sympy.Rational), 1)

        # Expression evaluates to a compile time constant
        rewritten = self._check(expr)
        self.assertEqual(rewritten, sympy.S.Zero)

    def test_no_distribute_mul_floordiv(self):
        """
        Test that multiplication doesn't distribute with floor division.
        """
        x = sympy.Symbol("x")
        expr = 2 * sympy.floor(x / 2)
        rewritten = self._check(expr)
        self.assertEqual(rewritten.count(sympy.Mul), 1)
        self.assertEqual(rewritten.count(FloorDiv), 1)

    def test_rational_multi_pows(self):
        """
        Test an expression with a rational and multiple pows.
        """
        x, y, z = sympy.symbols("x y z")
        expr = sympy.floor((x / 5) * (y**2) * (z**3))
        mul = expr.args[0]
        self.assertTrue(isinstance(mul, sympy.Mul))
        self.assertTrue(isinstance(mul.args[0], sympy.Rational))
        self.assertEqual(expr.count(sympy.Pow), 2)
        rewritten = self._check(expr)
        self.assertEqual(rewritten.count(FloorDiv), 1)

    def test_variable_exp(self):
        """
        Test pow when the exponent is a variable.
        """
        x = sympy.Symbol("x", positive=True)
        expr = sympy.floor(2**-x)
        replaced = self._check(expr)

        # Check that x went to the denominator.
        self.assertEqual(replaced.args, (1, 2**x))

    def test_launch_grid_dynamic_padding(self):
        """
        Test a complex launch grid expression arising from padding with dynamic shapes.
        """
        x, y = sympy.symbols("x y")
        expr = sympy.floor(-FloorDiv(x * y, 2) / FloorDiv(-x * y, 131070))
        self._check(expr)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU or TRITON_HAS_CPU:
        run_tests(needs="filelock")
