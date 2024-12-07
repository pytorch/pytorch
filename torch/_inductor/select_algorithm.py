# mypy: allow-untyped-defs
import builtins
import contextlib
import dataclasses
import functools
import inspect
import itertools
import json
import logging
import math
import operator
import os
import sys
import textwrap
import time
from collections import namedtuple
from concurrent.futures import as_completed, ThreadPoolExecutor
from io import StringIO
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union
from unittest.mock import patch

import sympy
from filelock import FileLock

import torch
import torch._inductor.async_compile  # noqa: F401 required to warm up AsyncCompile pools
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import counters, dynamo_timed, identity, preserve_rng_state

from . import config, ir
from .autotune_process import (
    TensorMeta,
    TritonBenchmarkRequest,
    TritonCPUBenchmarkRequest,
    TritonGPUBenchmarkRequest,
)
from .codecache import code_hash, PersistentCache, PyCodeCache
from .codegen.common import IndentedBuffer, KernelTemplate, OpOverrides, WorkspaceArg
from .codegen.simd_kernel_features import SIMDKernelFeatures
from .codegen.triton import (
    gen_common_triton_imports,
    texpr,
    TritonKernel,
    TritonScheduling,
)
from .codegen.triton_utils import config_of, signature_to_meta
from .exc import CUDACompileError
from .ir import ChoiceCaller, PrimitiveInfoType
from .runtime.benchmarking import benchmarker
from .runtime.hints import DeviceProperties
from .utils import (
    FakeIndentedBuffer,
    get_dtype_size,
    Placeholder,
    restore_stdout_stderr,
    sympy_dot,
    sympy_index_symbol,
    sympy_product,
    unique,
)
from .virtualized import V


log = logging.getLogger(__name__)

# correctness checks struggle with fp16/tf32
VERIFY: Dict[str, Any] = {}
PRINT_AUTOTUNE = True
DEBUG = False


class KernelNamespace:
    pass


# these objects are imported from the generated wrapper code
extern_kernels = KernelNamespace()


_T = TypeVar("_T", bound="AutotuneArgs")


@dataclasses.dataclass
class BenchmarkTensors:
    """Represents a set of inputs and outputs for autotuning with a template"""

    input_tensors: List[torch.Tensor]
    output_tensor: Optional[torch.Tensor]

    def unpack(self):
        return self.input_tensors, self.output_tensor


@dataclasses.dataclass
class AutotuneArgs:
    """During autotuning, we need to pass the same inputs to all choices.
    Note:
        Since we typically have a mix of external choices and triton choices, we create
        two lists of inputs for the same underlying buffers:
        - External inputs (for aten kernels): Include offset for sliced tensors
        - Triton inputs: Use base pointer for sliced tensors, without offset
    """

    triton: BenchmarkTensors
    extern: BenchmarkTensors
    expected: Optional[torch.Tensor] = None

    def get_benchmark_tensors(self, extern=False) -> BenchmarkTensors:
        """Returns the inputs and output tensors for a given choice."""
        bench_tensors = self.extern if extern else self.triton
        return bench_tensors

    @classmethod
    def from_choice_args(
        cls: Type[_T],
        example_inputs: List[torch.Tensor],
        example_inputs_extern: List[torch.Tensor],
        out: torch.Tensor,
        out_extern: torch.Tensor,
        expected: Optional[torch.Tensor] = None,
    ) -> _T:
        """Factory method to create AutotuneInputs from separate inputs/outputs"""
        return cls(
            triton=BenchmarkTensors(example_inputs, out),
            extern=BenchmarkTensors(example_inputs_extern, out_extern),
            expected=expected,
        )

    def verify(self, **kwargs):
        """Verify the correctness of the benchmarking results"""

        torch.testing.assert_close(self.extern.output_tensor, self.expected, **kwargs)


class PartialRender:
    """
    Some parts of a template need to be generated at the end, but
    inserted into the template at the start.  This allows doing a bunch
    of replacements after the initial render.
    """

    def __init__(self, code, replacement_hooks) -> None:
        super().__init__()
        self.code = code
        self.replacement_hooks = replacement_hooks

    def finalize_hook(self, hook_key: str, strict=True) -> None:
        if hook_key not in self.replacement_hooks:
            if strict:
                raise RuntimeError(
                    f"{hook_key} not registered in self.replacement_hooks"
                )
            else:
                return
        assert (
            self.replacement_hooks[hook_key] is not None
        ), "hook_key can only be called once"
        self.code = self.code.replace(hook_key, self.replacement_hooks[hook_key]())
        self.replacement_hooks[hook_key] = None

    def finalize_all(self) -> str:
        for key, fn in self.replacement_hooks.items():
            self.code = self.code.replace(key, fn())
        return self.code


# This is used to store info needed for lowering each subgraph in triton
# templates
SubgraphInfo = namedtuple(
    "SubgraphInfo",
    [
        "body",
        "template_mask",
        "template_out",
    ],
)


class ModificationWrapper(V.WrapperHandler):  # type: ignore[name-defined]
    """Handles placeholder substitutions during subgraph processing."""

    def __init__(
        self,
        kernel,
        subgraph_number: int,
        fixed_inputs: Dict[str, Any],
        mask: Optional[str],
    ):
        super().__init__(V.ops)
        self.name = f"PlaceholderSubstitution_{subgraph_number}"
        self.kernel = kernel
        self.fixed_inputs = fixed_inputs
        self.mask = mask

    def load(self, name: str, index: sympy.Expr):
        """Handle loading from tensor or fixed input."""
        if name not in self.fixed_inputs:
            index_str = self._process_indexing(index)
            var = self._add_kernel_input(name)
            return f"tl.load({var} + {index_str})"
        return f"({self.fixed_inputs[name]})"

    def indirect_indexing(self, index_var: str, size, check, wrap_neg=True):
        """Convert index variable to symbolic form."""
        return sympy_index_symbol(str(index_var))

    def store(self, name, index, value, mode):
        """Store value and track the store's mask and output value on the kernel.

        The template_mask and template_out are used by the indexing() method to properly
        mask store operations in the generated Triton code. The mask ensures stores only
        affect elements matching the mask condition. This is currently only used for scatter node's store
        """
        assert (
            self.mask is not None
        ), "Mask is required for inner stores in modifications"
        self.kernel.template_out = value
        self.kernel.template_mask = self.mask
        return self._inner.store(name, index, value, mode)

    def _add_kernel_input(self, name: str):
        """Add name as input to kernel and return input ref."""
        return self.kernel.args.input(name)

    def _process_indexing(self, index):
        """Process and rename indexing, adding symbols as kernel inputs."""
        return self.kernel.kexpr(self.kernel.rename_indexing(index))


class TritonTemplateKernel(TritonKernel):
    def __init__(
        self,
        kernel_name,
        input_nodes,
        output_node,
        defines,
        num_stages,
        num_warps,
        grid_fn,
        meta,
        call_sizes,
        use_jit=False,
        prefix_args=0,
        suffix_args=0,
        epilogue_fn=identity,
        subgraphs: Optional[List[ir.ComputedBuffer]] = None,
        workspace_arg: Optional[WorkspaceArg] = None,
    ) -> None:
        numel = sympy_product(output_node.get_size())
        super().__init__(
            {
                "x": numel,
                "r": sympy.S.One,
            },
            features=SIMDKernelFeatures([], numel),
        )
        self.input_nodes = input_nodes
        self.output_node = output_node
        self.named_input_nodes = {}  # type: ignore[var-annotated]
        self.defines = defines
        self.kernel_name = kernel_name
        self.use_jit = use_jit
        self.num_stages = num_stages
        self.num_warps = num_warps
        self.grid_fn = grid_fn
        self.meta = meta
        self.call_sizes = call_sizes
        # for templates with fixed epilogues
        self.prefix_args = prefix_args
        self.suffix_args = suffix_args
        self.epilogue_fn = epilogue_fn
        self.render_hooks = {}  # type: ignore[var-annotated]
        self.triton_meta: Optional[Dict[str, object]] = None
        # For Templated Attention this can be a list of ir.Subgraph
        self.subgraphs: Optional[List[ir.ComputedBuffer]] = subgraphs

        # Some templates use extra global memory as a workspace
        self.workspace_arg = workspace_arg
        if workspace_arg is not None:
            self.args.workspace_args.append(workspace_arg)

        # The following attributes (body, template_mask, output_val) are all
        # used for triton kernel codegen.
        # They are swapped onto the TritonTemplateKernel object by
        # `set_subgraph_body`
        self.subgraph_bodies: Dict[str, SubgraphInfo] = {}

        self.body: IndentedBuffer = FakeIndentedBuffer()
        self.template_mask: Optional[str] = None
        self.template_out: Optional[str] = None

    @contextlib.contextmanager
    def set_subgraph_body(self, body_name: str):
        old_body, old_mask, old_out = self.body, self.template_mask, self.template_out
        assert body_name in self.subgraph_bodies, body_name
        self.body, self.template_mask, self.template_out = self.subgraph_bodies[
            body_name
        ]
        yield
        self.subgraph_bodies[body_name] = SubgraphInfo(
            self.body, self.template_mask, self.template_out
        )
        self.body, self.template_mask, self.template_out = old_body, old_mask, old_out

    @contextlib.contextmanager
    def create_subgraph_body(self, body_name: str):
        assert body_name not in self.subgraph_bodies
        self.subgraph_bodies[body_name] = SubgraphInfo(IndentedBuffer(), None, None)
        with self.set_subgraph_body(body_name):
            yield

    def need_numel_args(self):
        return False

    def estimate_kernel_num_bytes(self):
        """
        Estimate the total number of bytes this kernel takes.
        For in/out nodes, sizes are counted twice: once for reading and
        once for writing.
        """
        ninplace_args = len(unique(self.args.inplace_buffers.values()))
        num_bytes = []
        for i, inp in enumerate(itertools.chain(self.input_nodes, (self.output_node,))):
            size = V.graph.sizevars.size_hints(inp.get_size())
            numel = functools.reduce(operator.mul, size, 1)
            dtype_size = get_dtype_size(inp.get_dtype())
            num_bytes.append(numel * dtype_size * (1 + int(i < ninplace_args)))
        return sum(num_bytes)

    def jit_lines(self):
        if self.use_jit:
            return "@triton.jit"

        argdefs, _, signature, _ = self.args.python_argdefs()
        triton_meta = {
            "signature": signature_to_meta(
                signature, size_dtype=self.index_dtype, argdefs=argdefs
            ),
            "device": DeviceProperties.create(self.output_node.get_device()),
            "constants": {},
        }
        triton_meta["configs"] = [config_of(signature)]
        for arg_num in triton_meta["configs"][0].equal_to_1:  # type: ignore[index]
            triton_meta["constants"][signature[arg_num].name] = 1  # type: ignore[index]
        matrix_instr_nonkdim = self.meta.get("matrix_instr_nonkdim", 0)
        if matrix_instr_nonkdim != 0:
            triton_meta["matrix_instr_nonkdim"] = matrix_instr_nonkdim

        self.triton_meta = triton_meta

        inductor_meta = {
            "kernel_name": str(Placeholder.DESCRIPTIVE_NAME),
            **TritonKernel.inductor_meta_common(),
        }
        if config.profile_bandwidth or config.benchmark_kernel:
            num_gb = self.estimate_kernel_num_bytes() / 1e9
            inductor_meta["kernel_num_gb"] = num_gb
        return f"""
            @triton_heuristics.template(
                num_stages={self.num_stages},
                num_warps={self.num_warps},
                triton_meta={triton_meta!r},
                inductor_meta={inductor_meta!r},
            )
            @triton.jit
        """

    def gen_argdefs(self):
        def hook():
            # python_argdefs() cannot be run until after the rest of the template lazily adds more args
            arg_defs, *_ = self.args.python_argdefs()
            return f"{', '.join(arg_defs)}"

        self.render_hooks["<ARGDEFS>"] = hook
        return "<ARGDEFS>"

    def gen_defines(self):
        return self.defines

    def def_kernel(self, *argnames):
        """
        Hook called from template code to generate function def and
        needed args.
        """
        assert all(isinstance(x, str) for x in argnames)
        renames = IndentedBuffer(initial_indent=1)

        named_args = self.input_nodes[
            self.prefix_args : len(self.input_nodes) - self.suffix_args
        ]

        assert len(argnames) == len(named_args), (
            len(argnames),
            len(named_args),
            self.prefix_args,
            len(self.input_nodes),
        )

        for input_node in self.input_nodes[: self.prefix_args]:
            # get args in correct order
            self.args.input(input_node.get_name())

        for name, input_node in zip(argnames, named_args):
            arg_name = f"arg_{name}"
            self.named_input_nodes[name] = input_node
            self.args.input_buffers[input_node.get_name()] = arg_name

        # The args may be duplicated, so renaming must be after args are de-duplicated.
        for name in argnames:
            input_node = self.named_input_nodes[name]
            arg_name = self.args.input_buffers[input_node.get_name()]
            if input_node.get_layout().offset == 0:
                renames.writeline(f"{name} = {arg_name}")
            else:
                offset = texpr(self.rename_indexing(input_node.get_layout().offset))
                renames.writeline(f"{name} = {arg_name} + {offset}")

        for input_node in self.input_nodes[len(self.input_nodes) - self.suffix_args :]:
            # get args in correct order
            self.args.input(input_node.get_name())

        def hook():
            # python_argdefs() cannot be run until after the rest of the template lazily adds more args
            arg_defs, *_ = self.args.python_argdefs()
            code = IndentedBuffer()
            code.splice(gen_common_triton_imports())
            code.splice(self.jit_lines())
            code.writeline(f"def {self.kernel_name}({', '.join(arg_defs)}):")
            with code.indent():
                code.splice(self.defines)
                code.splice(renames.getvalue())
            return code.getvalue()

        assert "<DEF_KERNEL>" not in self.render_hooks
        self.render_hooks["<DEF_KERNEL>"] = hook
        return "<DEF_KERNEL>"

    def size(self, name: str, index: int):
        """
        Hook called from template code to get the size of an arg.
        Will add needed args to pass it in if it is dynamic.
        """
        assert isinstance(index, int)
        if name is None:
            val = self.output_node.get_size()[index]
        else:
            assert isinstance(name, str)
            val = self.named_input_nodes[name].get_size()[index]
        return texpr(self.rename_indexing(val))

    def stride(self, name, index=None):
        """
        Hook called from template code to get the stride of an arg.
        Will add needed args to pass it in if it is dynamic.
        """
        if name is None:
            val = self.output_node.get_stride()
        else:
            assert isinstance(name, str)
            val = self.named_input_nodes[name].get_stride()

        if isinstance(index, int):
            return texpr(self.rename_indexing(val[index]))
        return ", ".join([texpr(self.rename_indexing(i)) for i in val])

    def _get_subgraph(self, subgraph_number: int):
        assert isinstance(subgraph_number, int)
        assert isinstance(self.subgraphs, list)
        assert subgraph_number < len(
            self.subgraphs
        ), f"Invalid subgraph number provided to create_modification, {subgraph_number} must be < {len(self.subgraphs)}"
        assert (
            self.body.getvalue() == ""
        ), "Body should be clear before adding a modification"
        return self.subgraphs[subgraph_number]

    def _handle_scatter_graph(self, scatter_graph):
        """Handle processing for a single scatter graph.

        Args:
            scatter_graph: The scatter graph to process
        """
        assert isinstance(
            scatter_graph, ir.ComputedBuffer
        ), f"scatter_graph must be an instance of ComputeBuffer but got {type(scatter_graph)}"

        def contiguous_strides(x):
            # We always create a fresh contiguous grad for scattering into
            return sum(
                x_i * stride for x_i, stride in zip(x, scatter_graph.get_stride())
            )

        scatter_graph.data.store_output(scatter_graph.name, contiguous_strides, [])  # type: ignore[attr-defined]

    def modification(
        self,
        subgraph_number: int,
        output_name: Optional[str],
        mask: Optional[str] = None,
        **fixed_inputs,
    ) -> str:
        """This creates a modification function for a subgraph.
        To use this inside a template, the first argument should specify which subgraph to codegen for

        Args:
            subgraph_number (int): The index of the subgraph in self.subgraphs
            output_name (Optional[str]): The name of the output variable to store the result in
            mask (Optional[str]): An optional mask to use for the store operation. If provided, this mask
                will be applied to the store.
        """
        num = 0
        out = None
        while f"mod_{subgraph_number}_{num}" in self.subgraph_bodies:
            num += 1
        with self.create_subgraph_body(f"mod_{subgraph_number}_{num}"):
            subgraph = self._get_subgraph(subgraph_number)
            modification_handler = ModificationWrapper(
                self, subgraph_number, fixed_inputs, mask
            )
            with V.set_ops_handler(modification_handler):
                assert isinstance(
                    subgraph, (ir.ComputedBuffer, List)
                ), f"Expected the subgraph to be a ComputedBuffer or a List[ComputedBuffer], got {type(subgraph)}"
                # Handle scatter stores
                if isinstance(subgraph, list):
                    for scatter_graph in subgraph:
                        self._handle_scatter_graph(scatter_graph)
                elif isinstance(subgraph.data, ir.InputBuffer):
                    out = subgraph.data.make_loader()(())
                else:
                    out = subgraph.data.inner_fn(())

            self.codegen_body()
            if output_name is not None:
                assert isinstance(output_name, str)
                assert out is not None
                self.body.writeline(f"{output_name} = {out.value}")

            body_val = self.body.getvalue()
            self.cse.invalidate(set())  # type: ignore[arg-type]
            return body_val

    def store_output(
        self,
        indices: Union[List[Any], Tuple[Any]],
        val: str,
        mask: Optional[str] = None,
        indent_width: int = 4,
    ):
        """Stores the final output and appends any epilogue fusions if the buffer hasn't been optimized away.

        Args:
            indices (Union[List, Tuple]): The index for each dimension of the output. The dot product of
                these indices and output strides must match `val`.
            val (str): The value to store.
            mask (Optional[str]): An optional mask to use for the store operation. If provided, this mask
                will be applied to the store.
            indent_width (int): The number of spaces to use for indentation. This is used when the call to
                store_output is indented in the kernel definition.
        """
        with self.create_subgraph_body("<STORE_OUTPUT>"):
            assert isinstance(indices, (list, tuple))
            assert isinstance(val, str)
            assert isinstance(mask, (str, type(None)))
            assert self.template_mask is None
            indices = list(map(OpOverrides.paren, indices))
            index_symbols = [sympy.Symbol(x, integer=True) for x in indices]
            lengths = [
                V.graph.sizevars.simplify(s) for s in self.output_node.get_size()
            ]
            assert len(indices) == len(lengths)

            # glue to make generated code use same indexing from template
            for name, range_tree_entry in zip(
                indices, self.range_trees[0].construct_entries(lengths)
            ):
                range_tree_entry.set_name(name)
            contiguous_index = sympy_dot(
                ir.FlexibleLayout.contiguous_strides(lengths), index_symbols
            )
            contiguous_index = self.rename_indexing(contiguous_index)
            self.body.writeline("xindex = " + texpr(contiguous_index))
            self.range_trees[0].lookup(sympy.S.One, sympy_product(lengths)).set_name(
                "xindex"
            )
            self.template_mask = mask
            self.template_out = val
            self.template_indices = indices
            output_index = self.output_node.get_layout().make_indexer()(index_symbols)
            output_index = self.rename_indexing(output_index)
            if output_index == contiguous_index:
                output_index = sympy.Symbol("xindex", integer=True)

            epilogue_args = [val]
            for input_node in itertools.chain(
                self.input_nodes[: self.prefix_args],
                self.input_nodes[len(self.input_nodes) - self.suffix_args :],
            ):
                input_node.freeze_layout()
                epilogue_args.append(input_node.make_loader()(index_symbols))

            V.ops.store(
                self.output_node.get_name(),
                output_index,
                self.epilogue_fn(*epilogue_args),
            )
            self.codegen_body()

        def hook():
            # more stuff might have been added since the codegen_body above
            self.codegen_body()

            return textwrap.indent(self.body.getvalue(), " " * indent_width).strip()

        assert "<STORE_OUTPUT>" not in self.render_hooks
        self.render_hooks["<STORE_OUTPUT>"] = hook
        return "<STORE_OUTPUT>"

    def render(self, template, kwargs):
        return PartialRender(
            template.render(**self.template_env(), **kwargs),
            self.render_hooks,
        )

    def make_load(self, name, indices, mask):
        """
        Optional helper called from template code to generate the code
        needed to load from an tensor.
        """
        assert isinstance(indices, (list, tuple))
        assert isinstance(name, str)
        assert isinstance(mask, str)
        stride = self.named_input_nodes[name].get_stride()
        indices = list(map(OpOverrides.paren, indices))
        assert len(indices) == len(stride)
        index = " + ".join(
            f"{texpr(self.rename_indexing(s))} * {i}" for s, i in zip(stride, indices)
        )
        return f"tl.load({name} + ({index}), {mask}, other=0.0)"

    def template_env(self):
        """
        Generate the namespace visible in the template.
        """
        return {
            fn.__name__: fn
            for fn in [
                self.def_kernel,
                self.size,
                self.stride,
                self.store_output,
                self.make_load,
                self.modification,
                self.gen_argdefs,
                self.gen_defines,
            ]
        }

    def indexing(
        self,
        index: sympy.Expr,
        *,
        dense_indexing=False,
        copy_shape=None,
        override_mask=None,
        block_ptr=False,
    ):
        """
        Override the default indexing to use our custom mask and force
        dense indexing.
        """
        return super().indexing(
            index,
            dense_indexing=False,
            # We pass template_out as the shape to broadcast the indexing to as
            # the mask might be broadcast to the output shape
            copy_shape=self.template_out,
            override_mask=self.template_mask,
            block_ptr=block_ptr,
        )

    def codegen_range_tree(self):
        pass  # ignore default codegen

    def call_kernel(self, name: str, node: Optional[ir.IRNode] = None):
        wrapper = V.graph.wrapper_code
        _, call_args, _, arg_types = self.args.python_argdefs()

        # Handle workspace allocation
        if self.workspace_arg is not None:
            wrapper.generate_workspace_allocation(self.workspace_arg)

        if V.graph.cpp_wrapper:
            # In the cpp_wrapper case, we have to compute CUDA launch grid at runtime
            # if any dynamic dimension is involved. We rely on the Python version
            # of the grid function to generate those grid configs, which may contain
            # symbolic values. The wrapper will use cexpr to print out C++ code
            # appropriately for the grid configs.
            grid = self.call_sizes + [self.meta]
            wrapper.generate_kernel_call(
                name,
                call_args,
                grid=self.grid_fn(*grid),
                arg_types=arg_types,
                triton_meta=self.triton_meta,
            )
        else:
            wrapper.add_import_once(f"import {self.grid_fn.__module__}")
            meta = wrapper.add_meta_once(self.meta)
            grid = self.call_sizes + [meta]
            wrapper.generate_kernel_call(
                name,
                call_args,
                grid=grid,
                grid_fn=f"{self.grid_fn.__module__}.{self.grid_fn.__name__}",
                arg_types=arg_types,
                triton_meta=self.triton_meta,
                gpu="cpu" not in V.graph.device_types,
            )

        if self.workspace_arg is not None:
            wrapper.generate_workspace_deallocation(self.workspace_arg)


@functools.lru_cache(None)
def _jinja2_env():
    try:
        import jinja2

        return jinja2.Environment(
            undefined=jinja2.StrictUndefined,
        )
    except ImportError:
        return None


class TritonTemplate(KernelTemplate):
    index_counter = itertools.count()
    all_templates: Dict[str, "TritonTemplate"] = {}

    def __init__(self, name: str, grid: Any, source: str, debug=False) -> None:
        super().__init__(name)
        self.grid = grid
        self.template = self._template_from_string(source)
        assert name not in self.all_templates, "duplicate template name"
        self.all_templates[name] = self
        self.debug = debug

    def generate(  # type: ignore[override]
        self,
        input_nodes,
        layout,
        num_stages,
        num_warps,
        prefix_args=0,
        suffix_args=0,
        epilogue_fn=identity,
        subgraphs=None,
        mutated_inputs=None,
        call_sizes=None,
        workspace_arg: Optional[WorkspaceArg] = None,
        **kwargs,
    ):
        """This function generates a TritonTemplateCaller

        Args:
            input_nodes: List of input nodes
            layout: Output layout
            num_stages: Number of stages for triton launch
            num_warps: Number of warps for triton launch
            prefix_args: Number of input nodes to be passed as arguments
            suffix_args: Number of input nodes to be passed as arguments
            epilogue_fn: Optional epilogue function to be called on the output
            subgraphs: Optional subgraphs to be passed as arguments, these will be inlined
                into the triton template string
            mutated_inputs: Optional list of input nodes that are mutated by the kernel, this is helpful
                if you need to return multiple outputs. You can pass them as inputs and mark them as
                being mutated by the kernel.
        """
        assert self.template, "requires jinja2"
        defines = StringIO()
        for name, val in kwargs.items():
            defines.write(f"{name} : tl.constexpr = {val}\n")
        defines = defines.getvalue()

        fake_out = ir.Buffer(name="buf_out", layout=layout)
        kernel_name = f"triton_{self.name}"

        numel = sympy_product(layout.size)
        buffers = itertools.chain(input_nodes, (fake_out,))
        if not TritonScheduling.can_use_32bit_indexing(numel, buffers):
            raise NotImplementedError(
                "64-bit indexing is not yet implemented for triton templates"
            )

        if call_sizes is None:
            call_sizes = layout.size

        kernel_options = {
            "input_nodes": input_nodes,
            "defines": defines,
            "num_stages": num_stages,
            "num_warps": num_warps,
            "grid_fn": self.grid,
            "meta": kwargs,
            "call_sizes": call_sizes,
            "prefix_args": prefix_args,
            "suffix_args": suffix_args,
            "epilogue_fn": epilogue_fn,
            "subgraphs": subgraphs,
        }

        with patch.object(
            V.graph, "get_dtype", self._fake_get_dtype(fake_out)
        ), V.graph.set_current_device(layout.device), TritonTemplateKernel(
            kernel_name=kernel_name,
            output_node=fake_out,
            workspace_arg=workspace_arg,
            use_jit=False,
            **kernel_options,
        ) as kernel:
            try:
                template = kernel.render(self.template, kwargs)
                with kernel.set_subgraph_body("<STORE_OUTPUT>"):
                    code = template.finalize_all()
            except ZeroDivisionError:
                # TODO(nmacchioni): fix sympy division by zero
                return None
            if self.debug:
                print("Generated Code:\n", code)
            extra = (
                "-".join(
                    [
                        *[
                            f"{kwarg}={repr(kwargs[kwarg])}"
                            for kwarg in sorted(kwargs.keys())
                        ],
                        f"num_stages={num_stages}",
                        f"num_warps={num_warps}",
                    ]
                )
                + "-"
            )
            mod = PyCodeCache.load(code, extra)

        input_call_args = tuple(kernel.args.input_buffers.keys())

        # We expect the input_buffer order to be [*input_nodes, *captured_buffers]
        expected_input_args = tuple(unique(x.get_name() for x in input_nodes))
        assert input_call_args[: len(expected_input_args)] == expected_input_args, (
            input_call_args,
            expected_input_args,
        )

        full_input_nodes = tuple([V.graph.get_buffer(k) for k in input_call_args])
        extra_args = V.graph.sizevars.size_hints(
            map(sympy.expand, tuple(kernel.args.sizevars.keys())),
            fallback=config.unbacked_symint_fallback,
        )

        kernel_hash_name = f"triton_{self.name}_{next(self.index_counter)}"

        def make_kernel_render(out_node):
            kernel = TritonTemplateKernel(
                kernel_name=str(Placeholder.KERNEL_NAME),
                output_node=out_node,
                workspace_arg=workspace_arg,
                use_jit=False,
                **kernel_options,
            )
            render = functools.partial(
                kernel.render,
                self.template,
                kwargs,
            )
            return kernel, render

        # create the BenchmarkRequest
        assert mod.__file__ is not None
        grid = self.grid(
            *V.graph.sizevars.size_hints(
                call_sizes,
                fallback=config.unbacked_symint_fallback,
            ),
            kwargs,
        )
        bmreq_cls: Type[TritonBenchmarkRequest]
        if layout.device.type == "cpu":
            bmreq_cls = TritonCPUBenchmarkRequest
        else:
            bmreq_cls = TritonGPUBenchmarkRequest
        bmreq = bmreq_cls(
            module_path=mod.__file__,
            module_cache_key=mod.key,
            kernel_name=kernel_name,
            grid=grid,
            extra_args=extra_args,
            num_stages=num_stages,
            num_warps=num_warps,
            matrix_instr_nonkdim=kwargs.get("matrix_instr_nonkdim", 0),
            input_tensor_meta=TensorMeta.from_irnodes(full_input_nodes),  # type: ignore[arg-type]
            output_tensor_meta=TensorMeta.from_irnodes(layout),
            workspace_arg=workspace_arg,
        )

        return TritonTemplateCaller(
            kernel_hash_name,
            full_input_nodes,
            layout,
            make_kernel_render,
            extra.strip("-").replace("-", ", "),
            bmreq,
            log_info={
                "tile_shape": str(
                    (
                        kwargs.get("BLOCK_M", -1),
                        kwargs.get("BLOCK_K", -1),
                        kwargs.get("BLOCK_N", -1),
                    )
                ),
                "num_stages": num_stages,
                "num_warps": num_warps,
                "allow_tf32": str(kwargs.get("ALLOW_TF32", None)),
                "acc_type": str(kwargs.get("ACC_TYPE", None)),
            },
            mutated_inputs=mutated_inputs,
            workspace_arg=workspace_arg,
        )


class ExternKernelChoice:
    def __init__(
        self,
        kernel,
        cpp_kernel=None,
        *,
        name=None,
        has_out_variant=True,
        op_overload=None,
        use_fallback_kernel=False,
        kernel_creator=None,
    ) -> None:
        super().__init__()
        name = name or kernel.__name__
        assert callable(kernel)
        assert not hasattr(extern_kernels, name), f"duplicate extern kernel: {name}"
        self.name = name
        self.cpp_kernel_name = cpp_kernel
        self.has_out_variant = has_out_variant
        setattr(extern_kernels, name, kernel)
        self.op_overload = op_overload
        self.use_fallback_kernel = use_fallback_kernel
        self.kernel_creator = kernel_creator

    def to_callable(self):
        return getattr(extern_kernels, self.name)

    def call_name(self):
        return f"extern_kernels.{self.name}"

    @functools.lru_cache(None)  # noqa: B019
    def hash_key(self):
        fn = self.to_callable()
        parts = [
            self.name,
            getattr(fn, "__name__", ""),
            getattr(fn, "__module__", ""),
        ]
        try:
            parts.append(inspect.getsource(fn))
        except Exception:
            pass
        return code_hash("-".join(parts))

    def bind(
        self,
        input_nodes,
        layout,
        ordered_kwargs_for_cpp_kernel=(),
        **kwargs,
    ):
        self.ordered_kwargs_for_cpp_kernel = ordered_kwargs_for_cpp_kernel
        return ExternKernelCaller(
            self, input_nodes, layout, kwargs, has_out_variant=self.has_out_variant
        )


class TritonTemplateCaller(ir.TritonTemplateCallerBase):
    def __init__(
        self,
        name,
        input_nodes,
        layout,
        make_kernel_render,
        description,
        bmreq,
        log_info: Optional[
            Dict[str, Union[PrimitiveInfoType, List[PrimitiveInfoType]]]
        ] = None,
        mutated_inputs=None,
        workspace_arg: Optional[WorkspaceArg] = None,
    ) -> None:
        super().__init__(name, input_nodes, layout, description)
        self.make_kernel_render = make_kernel_render
        self.bmreq: TritonBenchmarkRequest = bmreq
        if log_info is None:
            log_info = {}
        self.log_info: Dict[str, Any] = log_info
        self.log_info.update(
            {
                "backend": "Triton",
                "grid": str(self.bmreq.grid),
                "num_stages": self.bmreq.num_stages,
                "num_warps": self.bmreq.num_warps,
            }
        )
        self.mutated_inputs = mutated_inputs
        self.workspace_arg = workspace_arg

    def benchmark(self, *args, out):
        assert self.bmreq is not None
        return self.bmreq.benchmark(*args, output_tensor=out)

    def precompile(self):
        assert self.bmreq is not None
        self.bmreq.precompile()

    def __str__(self) -> str:
        return f"TritonTemplateCaller({self.bmreq.module_path}, {self.description})"

    def call_name(self):
        return f"template_kernels.{self.name}"

    def hash_key(self):
        return "-".join(
            [
                self.name.rsplit("_", 1)[0],
                self.bmreq.module_cache_key,
            ]
        )

    def output_node(self):
        return ir.TensorBox.create(
            ir.TritonTemplateBuffer(
                layout=self.layout,
                inputs=self.input_nodes,
                make_kernel_render=self.make_kernel_render,
                mutated_inputs=self.mutated_inputs,
            )
        )

    def info_dict(self) -> Dict[str, Union[PrimitiveInfoType, List[PrimitiveInfoType]]]:
        """Information returned here is logged to the autotune log file when that is enabled."""
        return self.log_info

    def get_make_kernel_render(self):
        return self.make_kernel_render

    def autoheuristic_id(self):
        type_name = "triton"
        info = self.info_dict()
        # TODO(AlnisM): Does tile_shape always exist?
        tile = info["tile_shape"]
        tile_vals = eval(tile)  # type: ignore[arg-type]
        BLOCK_M = tile_vals[0]
        BLOCK_K = tile_vals[1]
        BLOCK_N = tile_vals[2]
        num_stages = info["num_stages"]
        num_warps = info["num_warps"]
        return f"type={type_name}_BLOCK-M={BLOCK_M}_BLOCK-K={BLOCK_K}_BLOCK-N={BLOCK_N}_numstages={num_stages}_numwarps={num_warps}"


class ExternKernelCaller(ChoiceCaller):
    def __init__(
        self,
        choice: ExternKernelChoice,
        input_nodes,
        layout,
        kwargs=None,
        *,
        has_out_variant=True,
    ) -> None:
        super().__init__(choice.name, input_nodes, layout, description="")
        self.choice = choice
        self.kwargs = kwargs or {}
        self.has_out_variant = has_out_variant

    def __str__(self) -> str:
        return f"ExternKernelCaller({self.choice.call_name()})"

    def benchmark(self, *args, out):
        if out.numel() == 0:
            # no need to run the kerrnel of do benchmarking
            return 0.0
        if self.has_out_variant:
            return super().benchmark(*args, out=out)
        else:
            algo = self.to_callable()
            out_new = algo(*args)
            torch._C._dynamo.guards.assert_size_stride(
                out_new, tuple(out.size()), tuple(out.stride())
            )
            out.copy_(out_new)  # for correctness checking
            return benchmarker.benchmark(algo, args, {})

    def to_callable(self):
        fn = self.choice.to_callable()
        if self.kwargs:
            return functools.partial(fn, **self.kwargs)
        return fn

    def hash_key(self):
        return "-".join(
            [
                self.choice.name,
                *[
                    f"{kwarg}={repr(self.kwargs[kwarg])}"
                    for kwarg in sorted(self.kwargs.keys())
                ],
                self.choice.hash_key(),
            ]
        )

    def output_node(self):
        if self.choice.use_fallback_kernel:
            assert (
                self.choice.op_overload is not None
            ), "Please provide an op_overload to use ir.FallbackKernel"
            inner = ir.FallbackKernel.create(
                self.choice.op_overload, *self.input_nodes, **self.kwargs
            )
        elif self.choice.kernel_creator is not None:
            inner = self.choice.kernel_creator(*self.input_nodes, **self.kwargs)
        else:
            cls = ir.ExternKernelOut if self.has_out_variant else ir.ExternKernelAlloc
            inner = cls(
                layout=self.layout,
                inputs=self.input_nodes,
                python_kernel_name=self.choice.call_name(),
                cpp_kernel_name=self.choice.cpp_kernel_name,
                ordered_kwargs_for_cpp_kernel=self.choice.ordered_kwargs_for_cpp_kernel,
                op_overload=self.choice.op_overload,
                kwargs=self.kwargs,
            )

        return ir.TensorBox.create(inner)

    def info_dict(self) -> Dict[str, Union[PrimitiveInfoType, List[PrimitiveInfoType]]]:
        """Information returned here is logged to the autotune log file when that is enabled."""
        return {
            "backend": "extern",
            "kernel_call_name": self.choice.call_name(),
        }

    def autoheuristic_id(self):
        return f"extern_{self.choice.name}"


@functools.lru_cache(None)
def get_mm_log_filename() -> Optional[str]:
    mm_file_name = os.environ.get("TORCHINDUCTOR_MM_LOGGING_FILE", None)
    if not mm_file_name:
        return None

    if "json" not in mm_file_name:
        mm_file_name = f"{mm_file_name}.json"

    return mm_file_name


def append_to_log(filename, data):
    lock_file = filename.replace(".json", ".lock")
    lock = FileLock(lock_file)
    with lock:
        try:
            with open(filename) as f:
                log_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            log_data = []

        log_data.append(data)

        with open(filename, "w") as f:
            json.dump(log_data, f, indent=4)


class DataProcessorChoiceCallerWrapper:
    def __init__(self, wrapped, preprocessor, postprocessor) -> None:
        self._wrapped = wrapped
        if preprocessor is not None:
            self._preprocessor = preprocessor
        else:
            self._preprocessor = lambda x, y: (x, y)
        if postprocessor is not None:
            self._postprocessor = postprocessor
        else:
            self._postprocessor = lambda x: x

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def benchmark(self, *args, out) -> float:
        new_args, new_out = self._preprocessor(args, out)
        result = self._wrapped.benchmark(*new_args, out=new_out)
        new_out = self._postprocessor(new_out)
        if out is not new_out:
            out.copy_(new_out)
        return result

    def output_node(self) -> ir.TensorBox:
        result = self._wrapped.output_node()
        return self._postprocessor(result)

    def __repr__(self) -> str:
        return f"DataProcessorChoiceCallerWrapper({self._wrapped})"


class DataProcessorTemplateWrapper:
    """
    A wrapper class for a kernel template.

    This class together with `DataProcessorChoiceCallerWrapper` provides a convenient way to
    preprocess and postprocess data before and after using the wrapped template. A typical
    usage is to reorder or filter the input nodes in order to match the expected input of other
    kernel choices like a ATen kernel. A more complicated usage is to prepack the weights.
    See the example from :mod:`cpp_gemm_template` for more details.
    """

    def __init__(
        self,
        wrapped_template_cls,
        preprocessor,
        postprocessor,
        **kwargs,
    ) -> None:
        if preprocessor is not None:
            self._preprocessor = preprocessor
        else:
            self._preprocessor = lambda x, y: (x, y)
        if postprocessor is not None:
            self._postprocessor = postprocessor
        else:
            self._postprocessor = lambda x: x
        assert "input_nodes" in kwargs
        assert "layout" in kwargs
        kwargs["input_nodes"], kwargs["layout"] = preprocessor(
            kwargs["input_nodes"], kwargs["layout"]
        )
        self._wrapped = wrapped_template_cls(**kwargs)

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def maybe_append_choice(self, choices, **kwargs):
        return type(self._wrapped).maybe_append_choice(self, choices, **kwargs)

    def generate(self, **kwargs):
        choice_caller = self._wrapped.generate(**kwargs)
        return DataProcessorChoiceCallerWrapper(
            choice_caller, self._preprocessor, self._postprocessor
        )

    def __repr__(self) -> str:
        return f"DataProcessorTemplateWrapper({self._wrapped})"


class ErrorFromChoice(RuntimeError):
    def __init__(self, msg, choice: ChoiceCaller, inputs_str) -> None:
        msg += f"\nFrom choice {choice}\n{inputs_str}"
        super().__init__(msg)
        self.choice = choice


class NoValidChoicesError(RuntimeError):
    pass


@functools.lru_cache(None)
def get_env_num_workers() -> Optional[int]:
    if "TORCHINDUCTOR_COMPILE_THREADS" in os.environ:
        return int(os.environ["TORCHINDUCTOR_COMPILE_THREADS"])
    return None


def create_inputs_key(input_nodes) -> str:
    return repr([AlgorithmSelectorCache.key_of(x) for x in input_nodes])


def create_precompile_key(
    name: str, inputs_key: str, choices: List[ChoiceCaller]
) -> str:
    return ":".join(
        [
            name,
            inputs_key,
            torch.get_float32_matmul_precision(),
        ]
        + [choice.hash_key() for choice in choices]
    )


class AlgorithmSelectorCache(PersistentCache):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # the autotuning will get occur in the scheduler, so there is
        # no guarantee that the first lowering for a given key will also be the
        # first to benchmark it. share a single precompilation function for all lowerings
        # of a particular key
        self.precompile_cache: Dict[str, Callable[[], None]] = {}
        # list of callbacks that are called after benchmarking
        self.feedback_saver_fns: List[
            Callable[
                [Dict[ChoiceCaller, float], str, List[Any], List[ChoiceCaller]], None
            ]
        ] = []

    def __call__(
        self,
        name,
        choices: List[ChoiceCaller],
        input_nodes,
        layout,
        # optional dict mapping arg indices to the functions
        # generating a torch.Tensor for that input from the
        # corresponding ir.Buffer. if passed for a given
        # arg, the function will be called instead of
        # generating a random torch.Tensor for benchmarking.
        input_gen_fns: Optional[Dict[int, Callable[[ir.Buffer], torch.Tensor]]] = None,
        precompilation_timeout_seconds: int = 60 * 60,
        return_multi_template=False,
    ):
        from .codegen.cuda.cuda_kernel import CUDATemplateCaller

        # Templates selected with input_gen_fns require specific input data to avoid IMA
        # Passing custom input gen fns to benchmark_fusion NYI, so skip deferred template selection
        # TODO(jgong5): support multi-template on CPU
        if input_gen_fns is not None or layout.device.type == "cpu":
            return_multi_template = False

        # TODO - assert that we have not mutating kernels here

        # TODO(nmacchioni): remove once CI tests are fixed
        choices = [choice for choice in choices if choice is not None]

        if mm_file_name := get_mm_log_filename():
            M, K = input_nodes[-2].get_size()[:2]
            N = input_nodes[-1].get_size()[-1]
            append_to_log(mm_file_name, {"invoke": str((M, K, N))})

        if len(choices) == 0:
            backend_config = (
                "max_autotune_gemm_backends"
                if name != "convolution"
                else "max_autotune_conv_backends"
            )
            raise NoValidChoicesError(
                f"No choices to select, please consider adding ATEN into {backend_config} "
                "config (defined in torch/_inductor/config.py) to allow at least one choice. "
            )
        log.debug("Max autotune selects from %s choices.", str(len(choices)))

        if len(choices) == 1:
            if not isinstance(choices[0], CUDATemplateCaller):
                # CUDATemplateCaller still needs to go through autotuning process to retrieve workspace size.
                return choices[0].output_node()

        @functools.lru_cache(None)
        def make_benchmark_fn():
            return self.make_benchmark_fn(choices, input_nodes, layout, input_gen_fns)

        inputs_key = create_inputs_key(input_nodes)

        def precompile(choices) -> Callable[[], None]:
            def no_op(*args, **kwargs):
                return

            if (
                precompilation_timeout_seconds is None
                or precompilation_timeout_seconds <= 0
            ):
                return no_op

            env_workers = get_env_num_workers()
            num_workers = env_workers if env_workers is not None else (len(choices))

            if num_workers <= 0:
                return no_op

            # https://github.com/python/cpython/issues/106905
            if (
                sys.version_info.major == 3
                and sys.version_info.minor == 11
                and sys.version_info.micro <= 8
            ):
                return no_op

            # check local and global cache before precompiling
            timings = self.lookup(
                choices,
                name,
                inputs_key,
                benchmark=None,
            )

            if timings:
                return no_op

            if config.search_autotune_cache and not (
                config.max_autotune or config.max_autotune_gemm
            ):
                return no_op

            precompile_key = create_precompile_key(name, inputs_key, choices)
            if precompile_func := self.precompile_cache.get(precompile_key):
                return precompile_func

            log.info(
                "Multithreaded precompilation for %d choices using %d worker threads",
                len(choices),
                num_workers,
            )

            # In rare circumstances, because python threads inherit global state,
            # thread pool executor can race and leave stdout/stderr in a state
            # different than the original values. we explicitly restore the state
            # here to avoid this issue.

            initial_stdout = sys.stdout
            initial_stderr = sys.stderr

            def precompile_with_captured_stdout(choice):
                with restore_stdout_stderr(initial_stdout, initial_stderr):
                    start_time = time.time()
                    choice.precompile()
                    return time.time() - start_time

            executor = ThreadPoolExecutor(max_workers=num_workers)

            futures = {}
            for c in choices:
                if hasattr(c, "precompile"):
                    future = executor.submit(precompile_with_captured_stdout, c)
                    futures[future] = c

            @functools.lru_cache(None)
            @restore_stdout_stderr(initial_stdout, initial_stderr)
            def wait_on_futures():
                counters["inductor"]["select_algorithm_precompile"] += 1
                for future in as_completed(
                    futures,
                    timeout=precompilation_timeout_seconds,
                ):
                    if e := future.exception():
                        log.error(
                            "Exception %s for benchmark choice %s", e, futures[future]
                        )
                    else:
                        log.info(
                            "Precompiling benchmark choice %s took %.02fs",
                            futures[future],
                            future.result(),
                        )

                executor.shutdown(wait=True)

            self.precompile_cache[precompile_key] = wait_on_futures

            return wait_on_futures

        def autotune(choices):
            with dynamo_timed(f"{name}_template_autotuning"):
                return make_benchmark_fn()(choices)

        if config.autotune_in_subproc:
            from .autotune_process import tuning_pool

            # do the optional warmup
            tuning_pool.initialize()

        def do_autotuning(precompile_fn):
            precompile_start_ts = time.time()
            with dynamo_timed(f"{name}_template_precompiling"):
                precompile_fn()
            precompile_elapse = time.time() - precompile_start_ts

            autotune_start_ts = time.time()
            timings = self.lookup(
                choices,
                name,
                inputs_key,
                autotune,
            )
            autotune_elapse = time.time() - autotune_start_ts

            if timings and all(
                not math.isfinite(timing) for timing in timings.values()
            ):
                raise NoValidChoicesError

            if make_benchmark_fn.cache_info().currsize:
                counters["inductor"]["select_algorithm_autotune"] += 1

            if (
                make_benchmark_fn.cache_info().currsize
                or log.getEffectiveLevel() == logging.DEBUG
                or config.trace.log_autotuning_results
            ):
                self.log_results(
                    name, input_nodes, timings, autotune_elapse, precompile_elapse
                )

            for feedback_fn in self.feedback_saver_fns:
                feedback_fn(timings, name, input_nodes, choices)

            return timings

        precompile_fn = precompile(choices)

        if return_multi_template and (config.max_autotune or config.max_autotune_gemm):

            def get_timings():
                timings = do_autotuning(precompile_fn)
                min_extern_choice = float("inf")
                for choice, timing in timings.items():
                    if isinstance(choice, ExternKernelCaller):
                        min_extern_choice = min(min_extern_choice, timing)

                timings = {
                    choice: time
                    for choice, time in timings.items()
                    if (
                        time <= min_extern_choice
                        or not isinstance(choice, ExternKernelCaller)
                    )
                }

                return timings

            return torch._inductor.ir.TensorBox.create(
                torch._inductor.ir.MultiTemplateBuffer(
                    layout,
                    input_nodes,
                    get_timings,
                    choices,
                )
            )

        # TODO - dont want to precompile if we have a cache hit
        timings = do_autotuning(precompile_fn)
        if timings == {} or choices[0] not in timings:
            return choices[0].output_node()

        selected_key = builtins.min(timings, key=timings.__getitem__)
        selected_time = timings[selected_key]
        selected_choice = selected_key.output_node()
        log.debug("selected choice: %s", str(selected_choice))
        return selected_choice

    @classmethod
    def make_benchmark_fn(
        cls,
        choices,
        input_nodes,
        layout,
        input_gen_fns=None,
    ):
        if input_gen_fns is None:
            input_gen_fns = {}

        def get_inputs(
            choices: Union[List[ExternKernelCaller], List[TritonTemplateCaller]]
        ) -> AutotuneArgs:
            # de-duplicate args
            unique_example_inputs = {
                x.get_name(): input_gen_fns.get(i, cls.benchmark_example_value)(x)
                for i, x in enumerate(input_nodes)
            }
            example_inputs = list(unique_example_inputs.values())
            example_inputs_extern = [
                (
                    unique_example_inputs[input_node.get_name()]
                    if unique_example_inputs[input_node.get_name()].is_mkldnn
                    else torch.as_strided(
                        unique_example_inputs[input_node.get_name()],
                        V.graph.sizevars.size_hints(
                            input_node.get_size(),
                            fallback=config.unbacked_symint_fallback,
                        ),
                        V.graph.sizevars.size_hints(
                            input_node.get_stride(),
                            fallback=config.unbacked_symint_fallback,
                        ),
                        V.graph.sizevars.size_hint(
                            input_node.get_layout().offset,
                            fallback=config.unbacked_symint_fallback,
                        ),
                    )
                )
                for input_node in input_nodes
            ]
            out = cls.benchmark_example_value(layout)
            out_extern = torch.as_strided(
                out, out.size(), out.stride(), V.graph.sizevars.size_hint(layout.offset)
            )
            expected = None
            if VERIFY:
                choices[0].benchmark(*example_inputs_extern, out=out_extern)
                expected = out_extern.clone()

            return AutotuneArgs.from_choice_args(
                example_inputs,
                example_inputs_extern,
                out,
                out_extern,
                expected,
            )

        if DEBUG:
            print(f"{len(choices)} tuning requests:")

        def benchmark_choice_in_current_process(
            choice: ChoiceCaller, autotune_args: AutotuneArgs
        ) -> float:
            is_extern = isinstance(choice, ExternKernelCaller)
            benchmark_tensors = autotune_args.get_benchmark_tensors(is_extern)
            inpts, output = benchmark_tensors.unpack()
            output.zero_()
            result = choice.benchmark(*inpts, out=output)
            if VERIFY and autotune_args.expected is not None:
                autotune_args.verify(**VERIFY)
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # shake out any CUDA errors
            return result

        def benchmark_in_current_process(
            choices: Union[List[ExternKernelCaller], List[TritonTemplateCaller]],
        ) -> Dict[Union[ExternKernelCaller, TritonTemplateCaller], float]:
            inputs = get_inputs(choices)
            timings = {}
            for choice in choices:
                try:
                    timing = benchmark_choice_in_current_process(choice, inputs)
                except CUDACompileError as e:
                    log.error(
                        "CUDA compilation error during autotuning: \n%s. \nIgnoring this choice.",
                        str(e),
                    )
                    timing = float("inf")
                except NotImplementedError as e:
                    log.warning("Not yet implemented: %s", e)
                    timing = float("inf")
                except RuntimeError as e:
                    msg = str(e)
                    if "invalid argument" in msg:
                        msg += "\n\nThis may mean this GPU is too small for max_autotune mode.\n\n"
                    else:
                        if "illegal memory access" in msg:
                            msg += "\n\nEither error in template or triton bug.\n"
                    log.error(
                        "Runtime error during autotuning: \n%s. \nIgnoring this choice.",
                        msg,
                    )
                    timing = float("inf")
                except AssertionError as e:
                    raise AssertionError(  # noqa: B904
                        f"Incorrect result from choice {choice}\n\n{e}"
                    )
                except Exception as e:
                    try:
                        from triton.runtime.autotuner import OutOfResources

                        if isinstance(e, OutOfResources):
                            log.warning(e)
                            timing = float("inf")
                        else:
                            raise e
                    except ImportError:
                        raise e from None

                timings[choice] = timing

            return timings

        def benchmark_in_sub_process(
            choices: Union[List[ExternKernelCaller], List[TritonTemplateCaller]]
        ):
            from . import autotune_process

            # only benchmark triton kernel in sub process for now.
            # ATen/Extern kernel are still benchmarked in the current process.
            extern = [c for c in choices if isinstance(c, ExternKernelCaller)]
            triton = [c for c in choices if not isinstance(c, ExternKernelCaller)]

            timings = benchmark_in_current_process(extern)
            timings.update(autotune_process.benchmark_in_sub_process(triton))  # type: ignore[arg-type]
            return timings

        benchmark = (
            benchmark_in_sub_process
            if config.autotune_in_subproc
            else benchmark_in_current_process
        )

        return benchmark

    @staticmethod
    def log_results(
        name: str,
        input_nodes: List[ir.IRNode],
        timings: Dict[ChoiceCaller, float],
        elapse: float,
        precompile_elapse: float,
    ):
        V.debug.log_autotuning_results(
            name, input_nodes, timings, elapse, precompile_elapse
        )
        if not (config.max_autotune or config.max_autotune_gemm) or not PRINT_AUTOTUNE:
            return
        sizes = ", ".join(
            [
                "x".join(
                    map(
                        str,
                        V.graph.sizevars.size_hints(
                            n.get_size(), fallback=config.unbacked_symint_fallback  # type: ignore[arg-type]
                        ),
                    )
                )
                for n in input_nodes
            ]
        )
        if config.autotune_num_choices_displayed == 0:
            return
        elif config.autotune_num_choices_displayed is None:
            n = -1
        else:
            n = config.autotune_num_choices_displayed

        top_k = sorted(timings, key=timings.__getitem__)[:n]

        best = top_k[0]

        def get_choice_info(choice):
            if isinstance(choice, torch._inductor.select_algorithm.ExternKernelCaller):
                return {"type": "cublas", "time": timings[choice]}

            assert isinstance(
                choice, torch._inductor.select_algorithm.TritonTemplateCaller
            )

            info = choice.info_dict()
            tile = info["tile_shape"]

            tile_vals = eval(tile)  # type: ignore[arg-type]
            BLOCK_M = tile_vals[0]
            BLOCK_K = tile_vals[1]
            BLOCK_N = tile_vals[2]

            return {
                "type": "triton",
                "time": timings[choice],
                "BLOCK_M": BLOCK_M,
                "BLOCK_K": BLOCK_K,
                "BLOCK_N": BLOCK_N,
                "num_stages": info["num_stages"],
                "num_warps": info["num_warps"],
            }

        mm_filename = get_mm_log_filename()
        if mm_filename and "mm" in name:
            M, K = input_nodes[-2].get_size()[:2]
            N = input_nodes[-1].get_size()[-1]

            out_dict = {
                str((M, K, N)): [get_choice_info(choice) for choice in timings.keys()]
            }

            append_to_log(mm_filename, out_dict)

        best_time = timings[best]
        sys.stderr.write(f"AUTOTUNE {name}({sizes})\n")
        for choice in top_k:
            result = timings[choice]
            if result:
                kernel_description = choice.description
                sys.stderr.write(
                    f"  {choice.name} {result:.4f} ms {best_time / result:.1%} {kernel_description}\n"
                )
            else:
                sys.stderr.write(
                    f"  {choice.name} {result:.4f} ms <DIVIDED BY ZERO ERROR>\n"
                )

        autotune_type_str = (
            "SubProcess" if config.autotune_in_subproc else "SingleProcess"
        )
        sys.stderr.write(
            f"{autotune_type_str} AUTOTUNE benchmarking takes {elapse:.4f} seconds and {precompile_elapse:.4f}"
            f" seconds precompiling for {len(timings)} choices\n"
        )

    @staticmethod
    def benchmark_example_value(node):
        """
        Convert an ir.Buffer into a concrete torch.Tensor we can use for
        benchmarking.
        """
        if isinstance(node, ir.Layout):
            node = ir.Buffer(name="fake", layout=node)
        # triton templates want the base tensor.
        if isinstance(node, ir.BaseView):
            node = node.unwrap_view()
        return AlgorithmSelectorCache.generate_example_value(
            V.graph.sizevars.size_hints(
                node.get_size(),
                fallback=config.unbacked_symint_fallback,
            ),
            V.graph.sizevars.size_hints(
                node.get_stride(),
                fallback=config.unbacked_symint_fallback,
            ),
            node.get_device(),
            node.get_dtype(),
            node.layout.offset,
        )

    @staticmethod
    def generate_example_value(size, stride, device, dtype, extra_size):
        # preserve rng states to avoid the rand_strided call below changes
        # the rng states for the real model code.
        with preserve_rng_state():
            return rand_strided(
                size,
                stride,
                device=device,
                dtype=dtype,
                extra_size=extra_size,
            )

    @staticmethod
    def key_of(node):
        """
        Extract the pieces of an ir.Buffer that we should invalidate cached
        autotuning results on.
        """
        sizevars = V.graph.sizevars
        return (
            node.get_device().type,
            str(node.get_dtype()),
            *sizevars.size_hints(
                node.get_size(),
                fallback=config.unbacked_symint_fallback,
            ),
            *sizevars.size_hints(
                node.get_stride(),
                fallback=config.unbacked_symint_fallback,
            ),
            sizevars.size_hint(
                node.get_layout().offset,
                fallback=config.unbacked_symint_fallback,
            ),
        )

    def add_feedback_saver(
        self,
        fn: Callable[
            [Dict[ChoiceCaller, float], str, List[Any], List[ChoiceCaller]], None
        ],
    ):
        self.feedback_saver_fns.append(fn)


_ALGORITHM_SELECTOR_CACHE: Optional[AlgorithmSelectorCache] = None


def autotune_select_algorithm(*args, **kwargs):
    global _ALGORITHM_SELECTOR_CACHE
    if _ALGORITHM_SELECTOR_CACHE is None:
        _ALGORITHM_SELECTOR_CACHE = AlgorithmSelectorCache()

    if "return_multi_template" not in kwargs:
        kwargs[
            "return_multi_template"
        ] = torch._inductor.config.benchmark_epilogue_fusion

    return _ALGORITHM_SELECTOR_CACHE(*args, **kwargs)


def add_feedback_saver(
    fn: Callable[[Dict[ChoiceCaller, float], str, List[Any], List[ChoiceCaller]], None]
):
    global _ALGORITHM_SELECTOR_CACHE
    if _ALGORITHM_SELECTOR_CACHE is None:
        _ALGORITHM_SELECTOR_CACHE = AlgorithmSelectorCache()
    _ALGORITHM_SELECTOR_CACHE.add_feedback_saver(fn)


def realize_inputs(*args):
    if len(args) == 1:
        return ir.ExternKernel.require_stride1(ir.ExternKernel.realize_input(args[0]))
    return [realize_inputs(x) for x in args]


# ensure lowering is imported so that `extern_kernels.*` is populated
from . import lowering  # noqa: F401
