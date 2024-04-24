import builtins
import functools
import inspect
import itertools
import logging
import operator
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from io import StringIO

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from unittest.mock import patch

import sympy

import torch
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import counters, identity, preserve_rng_state

from . import config, ir
from .autotune_process import TensorMeta, TritonBenchmarkRequest
from .codecache import code_hash, PersistentCache, PyCodeCache
from .codegen.common import IndentedBuffer, KernelTemplate

from .codegen.triton import (
    gen_common_triton_imports,
    texpr,
    TritonKernel,
    TritonPrinter,
    TritonScheduling,
)

from .codegen.triton_utils import config_of, signature_to_meta
from .exc import CUDACompileError
from .ir import ChoiceCaller, PrimitiveInfoType
from .runtime.hints import DeviceProperties
from .runtime.runtime_utils import do_bench
from .utils import (
    get_dtype_size,
    Placeholder,
    sympy_dot,
    sympy_index_symbol,
    sympy_product,
    unique,
)
from .virtualized import V

log = logging.getLogger(__name__)

# correctness checks struggle with fp16/tf32
VERIFY: Dict[str, Any] = dict()
PRINT_AUTOTUNE = True
DEBUG = False


class KernelNamespace:
    pass


# these objects are imported from the generated wrapper code
extern_kernels = KernelNamespace()


class PartialRender:
    """
    Some parts of a template need to be generated at the end, but
    inserted into the template at the start.  This allows doing a bunch
    of replacements after the initial render.
    """

    def __init__(self, code, replacement_hooks):
        super().__init__()
        self.code = code
        self.replacement_hooks = replacement_hooks

    def finalize(self):
        code = self.code
        assert code is not None, "can only be called once"
        self.code = None
        for key, fn in self.replacement_hooks.items():
            code = code.replace(key, fn())
        return code


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
        subgraphs=None,
        *,
        index_dtype,
    ):
        super().__init__(
            sympy_product(output_node.get_size()),
            sympy.Integer(1),
            index_dtype=index_dtype,
        )
        self.input_nodes = input_nodes
        self.output_node = output_node
        self.named_input_nodes = {}
        self.defines = defines
        self.kernel_name = kernel_name
        self.template_mask = None
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
        self.render_hooks = dict()
        self.triton_meta: Optional[Dict[str, object]] = None
        # For Templated Attention
        self.subgraphs = subgraphs

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
            numel = functools.reduce(operator.mul, size)
            dtype_size = get_dtype_size(inp.get_dtype())
            num_bytes.append(numel * dtype_size * (1 + int(i < ninplace_args)))
        return sum(num_bytes)

    def jit_lines(self):
        if self.use_jit:
            return "@triton.jit"

        argdefs, _, signature = self.args.python_argdefs()
        triton_meta = {
            "signature": signature_to_meta(signature, size_dtype=self.index_dtype),
            "device": DeviceProperties.create(self.output_node.get_device()),
            "constants": {},
        }
        triton_meta["configs"] = [config_of(signature)]
        for arg_num in triton_meta["configs"][0].equal_to_1:  # type: ignore[index]
            triton_meta["constants"][arg_num] = 1  # type: ignore[index]
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

    def stride(self, name, index):
        """
        Hook called from template code to get the stride of an arg.
        Will add needed args to pass it in if it is dynamic.
        """
        assert isinstance(index, int)
        if name is None:
            val = self.output_node.get_stride()[index]
        else:
            assert isinstance(name, str)
            val = self.named_input_nodes[name].get_stride()[index]
        return texpr(self.rename_indexing(val))

    def modification(self, **fixed_inputs) -> str:
        """This function generates the code body to populate
        a 'modification' placeholder within a template

        TODO come up with standardized way to modify templates, with
        potential multiple modifications
        """

        def add_input(name):
            return self.args.input(name)

        class PlaceholderSubstitution(V.WrapperHandler):  # type: ignore[name-defined]
            self.name = "PlaceholderSubstitution"

            def load(self, name: str, index: sympy.Expr):
                if name not in fixed_inputs:
                    # If it's not a fixed input, it's a load from a captured
                    # tensor
                    var = add_input(name)
                    return f"tl.load({var} + {index})"

                return f"({fixed_inputs[name]})"

            def indirect_indexing(self, index_var, size, check):
                return sympy_index_symbol(str(index_var))

        # if self.modification_cache is None:
        with V.set_ops_handler(PlaceholderSubstitution(V.ops)):
            assert isinstance(
                self.subgraphs, ir.ComputedBuffer
            ), "Expected the subgraph to be a ComputedBuffer"
            if isinstance(self.subgraphs.data, ir.InputBuffer):
                out = self.subgraphs.data.make_loader()((1,))
            else:
                out = self.subgraphs.data.inner_fn((1,))

        self.codegen_body()
        self.body.writeline(f"{fixed_inputs['out']} = {out.value}")

        body_val = self.body.getvalue()
        self.body.clear()
        self.cse.invalidate(set())
        return body_val

    def store_output(
        self,
        indices: Union[List[Any], Tuple[Any]],
        val: str,
        mask: Optional[str] = None,
    ):
        """
        Hook called from template code to store the final output
        (if the buffer hasn't been optimized away), then append any
        epilogue fusions.
        """
        assert isinstance(indices, (list, tuple))
        assert isinstance(val, str)
        assert isinstance(mask, (str, type(None)))
        assert self.template_mask is None
        indices = list(map(TritonPrinter.paren, indices))
        index_symbols = [sympy.Symbol(x) for x in indices]
        lengths = [V.graph.sizevars.simplify(s) for s in self.output_node.get_size()]
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
        self.range_trees[0].lookup(sympy.Integer(1), sympy_product(lengths)).set_name(
            "xindex"
        )
        self.template_mask = mask
        self.template_indices = indices
        output_index = self.output_node.get_layout().make_indexer()(index_symbols)
        output_index = self.rename_indexing(output_index)
        if output_index == contiguous_index:
            output_index = sympy.Symbol("xindex")

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
            return textwrap.indent(self.body.getvalue(), "    ").strip()

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
        indices = list(map(TritonPrinter.paren, indices))
        assert len(indices) == len(stride)
        index = " + ".join(
            f"{texpr(self.rename_indexing(s))} * {i}" for s, i in zip(stride, indices)
        )
        return f"tl.load({name} + ({index}), {mask})"

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
            copy_shape=self.template_mask,
            override_mask=self.template_mask,
            block_ptr=block_ptr,
        )

    def initialize_range_tree(self, pid_cache):
        super().initialize_range_tree(pid_cache)
        # ignore default codegen
        self.body.clear()
        self.indexing_code.clear()

    def call_kernel(self, name: str, node: Optional[ir.IRNode] = None):
        wrapper = V.graph.wrapper_code
        _, call_args, _ = self.args.python_argdefs()
        call_args = [str(a) for a in call_args]

        for i in range(len(call_args)):
            if V.graph.is_unspec_arg(call_args[i]):
                call_args[i] = call_args[i] + ".item()"
            if isinstance(call_args[i], sympy.Symbol):
                call_args[i] = texpr(call_args[i])

        if V.graph.cpp_wrapper:
            # In the cpp_wrapper case, we have to compute CUDA launch grid at runtime
            # if any dynamic dimension is involved. We rely on the Python version
            # of the grid function to generate those grid configs, which may contain
            # symbolic values. The wrapper will use cexpr to print out C++ code
            # appropriately for the grid configs.
            grid_args = [V.graph.sizevars.simplify(s) for s in self.call_sizes] + [
                self.meta
            ]
            grid = self.grid_fn(*grid_args)

            wrapper.generate_kernel_call(
                name,
                call_args,
                device_index=V.graph.scheduler.current_device.index,
                grid=grid,
                triton_meta=self.triton_meta,
            )
        else:
            stream_name = wrapper.write_get_raw_stream(
                V.graph.scheduler.current_device.index
            )

            wrapper.add_import_once(f"import {self.grid_fn.__module__}")
            meta = wrapper.add_meta_once(self.meta)

            grid_call = [
                texpr(V.graph.sizevars.simplify(s)) for s in self.call_sizes
            ] + [meta]
            grid_call = f"{self.grid_fn.__module__}.{self.grid_fn.__name__}({', '.join(grid_call)})"
            wrapper.writeline(
                f"{name}.run({', '.join(call_args)}, grid={grid_call}, stream={stream_name})"
            )


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
    all_templates: Dict[str, "TritonTemplate"] = dict()

    def __init__(self, name: str, grid: Any, source: str, debug=False):
        super().__init__(name)
        self.grid = grid
        self.template = self._template_from_string(source)
        assert name not in self.all_templates, "duplicate template name"
        self.all_templates[name] = self
        self.debug = debug

    def generate(
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
            defines.write(f"    {name} : tl.constexpr = {val}\n")
        defines = defines.getvalue()

        fake_out = ir.Buffer("buf_out", layout)
        kernel_name = f"triton_{self.name}"

        numel = sympy_product(layout.size)
        buffers = itertools.chain(input_nodes, (fake_out,))
        if not TritonScheduling.can_use_32bit_indexing(numel, buffers):
            raise NotImplementedError(
                "64-bit indexing is not yet implemented for triton templates"
            )

        kernel_options = dict(
            input_nodes=input_nodes,
            defines=defines,
            num_stages=num_stages,
            num_warps=num_warps,
            grid_fn=self.grid,
            meta=kwargs,
            call_sizes=layout.size,
            prefix_args=prefix_args,
            suffix_args=suffix_args,
            epilogue_fn=epilogue_fn,
            index_dtype="tl.int32",
            subgraphs=subgraphs,
        )
        with patch.object(
            V.graph, "get_dtype", self._fake_get_dtype(fake_out)
        ), TritonTemplateKernel(
            kernel_name=kernel_name,
            output_node=fake_out,
            use_jit=False,
            **kernel_options,
        ) as kernel:
            try:
                code = kernel.render(self.template, kwargs).finalize()
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
        output_call_args = tuple(kernel.args.output_buffers.keys())

        # We expect the input_buffer order to be [*input_nodes, *captured_buffers]
        expected_input_args = tuple(unique(x.get_name() for x in input_nodes))
        expected_output_args = (fake_out.get_name(),)
        assert input_call_args[: len(expected_input_args)] == expected_input_args, (
            input_call_args,
            expected_input_args,
        )
        assert output_call_args == expected_output_args, (
            output_call_args,
            expected_output_args,
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
                layout.size,
                fallback=config.unbacked_symint_fallback,
            ),
            kwargs,
        )
        bmreq = TritonBenchmarkRequest(
            module_path=mod.__file__,
            module_cache_key=mod.key,
            kernel_name=kernel_name,
            grid=grid,
            extra_args=extra_args,
            num_stages=num_stages,
            num_warps=num_warps,
            matrix_instr_nonkdim=kwargs.get("matrix_instr_nonkdim", 0),
            input_tensor_meta=TensorMeta.from_irnodes(full_input_nodes),
            output_tensor_meta=TensorMeta.from_irnodes(layout),
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
    ):
        super().__init__()
        name = name or kernel.__name__
        assert callable(kernel)
        assert not hasattr(extern_kernels, name), "duplicate extern kernel"
        self.name = name
        self.cpp_kernel_name = cpp_kernel
        self.has_out_variant = has_out_variant
        setattr(extern_kernels, name, kernel)
        self.op_overload = op_overload
        self.use_fallback_kernel = use_fallback_kernel

    def to_callable(self):
        return getattr(extern_kernels, self.name)

    def call_name(self):
        return f"extern_kernels.{self.name}"

    @functools.lru_cache(None)
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
        debug_extra,
        bmreq,
        log_info: Optional[
            Dict[str, Union[PrimitiveInfoType, List[PrimitiveInfoType]]]
        ] = None,
        mutated_inputs=None,
    ):
        super().__init__(name, input_nodes, layout)
        self.make_kernel_render = make_kernel_render
        self.debug_extra = debug_extra
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

    def benchmark(self, *args, out):
        assert self.bmreq is not None
        return self.bmreq.benchmark(*args, output_tensor=out)

    def precompile(self):
        assert self.bmreq is not None
        self.bmreq.precompile()

    def __str__(self):
        return f"TritonTemplateCaller({self.bmreq.module_path}, {self.debug_extra})"

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
                debug_extra=self.debug_extra,
                mutated_inputs=self.mutated_inputs,
            )
        )

    def info_dict(self) -> Dict[str, Union[PrimitiveInfoType, List[PrimitiveInfoType]]]:
        """Information returned here is logged to the autotune log file when that is enabled."""
        return self.log_info

    def get_make_kernel_render(self):
        return self.make_kernel_render


class ExternKernelCaller(ChoiceCaller):
    def __init__(
        self,
        choice: ExternKernelChoice,
        input_nodes,
        layout,
        kwargs=None,
        *,
        has_out_variant=True,
    ):
        super().__init__(choice.name, input_nodes, layout)
        self.choice = choice
        self.kwargs = kwargs or {}
        self.has_out_variant = has_out_variant

    def __str__(self):
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
            return do_bench(lambda: algo(*args))

    def to_callable(self):
        fn = self.choice.to_callable()
        if self.kwargs:
            return functools.partial(fn, **self.kwargs)
        else:
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
        if config.abi_compatible and self.choice.use_fallback_kernel:
            assert (
                self.choice.op_overload is not None
            ), "Please provide an op_overload to use ir.FallbackKernel"
            inner = ir.FallbackKernel.create(
                self.choice.op_overload, *self.input_nodes, **self.kwargs
            )
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


class ErrorFromChoice(RuntimeError):
    def __init__(self, msg, choice: ChoiceCaller, inputs_str):
        msg += f"\nFrom choice {choice}\n{inputs_str}"
        super().__init__(msg)
        self.choice = choice


class NoValidChoicesError(RuntimeError):
    pass


class AlgorithmSelectorCache(PersistentCache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # the autotuning will get occur in the scheduler, so there is
        # no guarantee that the first lowering for a given key will also be the
        # first to benchmark it. share a single precompilation function for all lowerings
        # of a particular key
        self.precompile_cache: Dict[str, Callable[[], None]] = {}

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
        if input_gen_fns is not None:
            return_multi_template = False

        # TODO - assert that we have not mutating kernels here

        # TODO(nmacchioni): remove once CI tests are fixed
        choices = [choice for choice in choices if choice is not None]

        if len(choices) == 0:
            raise RuntimeError(
                "No choices to select, please consider adding ATEN into max_autotune_gemm_backends "
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

        inputs_key = repr([self.key_of(x) for x in input_nodes])

        def precompile(choices) -> Callable[[], None]:
            def no_op(*args, **kwargs):
                return

            if (
                precompilation_timeout_seconds is None
                or precompilation_timeout_seconds <= 0
            ):
                return no_op
            num_workers = min(
                config.compile_threads,
                torch.get_num_threads(),
                len(choices),
            )
            if num_workers <= 0:
                return no_op

            # TODO - debug issue
            if torch.version.hip:
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

            precompile_key = (
                f"{name}: {inputs_key} : {torch.get_float32_matmul_precision()}"
            )
            if precompile_func := self.precompile_cache.get(precompile_key):
                return precompile_func

            log.info(
                "Multithreaded precompilation for %d choices using %d worker threads",
                len(choices),
                num_workers,
            )

            executor = ThreadPoolExecutor(max_workers=num_workers)
            futures = executor.map(
                lambda c: c.precompile(),
                [c for c in choices if hasattr(c, "precompile")],
                timeout=precompilation_timeout_seconds,
            )

            @functools.lru_cache(None)
            def wait_on_futures():
                counters["inductor"]["select_algorithm_precompile"] += 1
                try:
                    iterator = iter(futures)
                    while True:
                        try:
                            next(iterator)
                        except CUDACompileError:
                            log.error(  # noqa: G201
                                "CUDA Compilation error", exc_info=True
                            )
                except TimeoutError:
                    log.warning(
                        f"Precompilation timed out after {precompilation_timeout_seconds} seconds."  # noqa: G004
                    )
                except StopIteration:
                    pass

                executor.shutdown(wait=True)

            self.precompile_cache[precompile_key] = wait_on_futures

            return wait_on_futures

        def autotune(choices):
            return make_benchmark_fn()(choices)

        if config.autotune_in_subproc:
            from .autotune_process import tuning_pool

            # do the optional warmup
            tuning_pool.initialize()

        def do_autotuning(precompile_fn):
            precompile_start_ts = time.time()
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
                )
            )

        # TODO - dont want to precompile if we have a cache hit
        timings = do_autotuning(precompile_fn)
        if timings == {} or choices[0] not in timings:
            return choices[0].output_node()

        selected_choice = builtins.min(timings, key=timings.__getitem__).output_node()
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

        def get_inputs():
            # de-duplicate args
            unique_example_inputs = {
                x.get_name(): input_gen_fns.get(i, cls.benchmark_example_value)(x)
                for i, x in enumerate(input_nodes)
            }
            example_inputs = list(unique_example_inputs.values())
            example_inputs_extern = [
                torch.as_strided(
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

            return example_inputs, example_inputs_extern, out, out_extern, expected

        if DEBUG:
            print(f"{len(choices)} tuning requests:")

        def debug_str(example_inputs, out):
            def tensor_repr(x):
                return (
                    f"torch.empty_strided({tuple(x.size())!r}, {tuple(x.stride())!r}, "
                    f"dtype={x.dtype!r}, device={x.device.type!r})"
                )

            lines = [
                "inputs = [",
            ]
            for x in example_inputs:
                lines.append(f"    {tensor_repr(x)},")
            lines += ["]", f"out = {tensor_repr(out)}", ""]
            return "\n".join(lines)

        def benchmark_choice_in_current_process(
            choice, example_inputs, example_inputs_extern, out, out_extern, expected
        ):
            out.zero_()
            if isinstance(choice, ExternKernelCaller):
                # aten kernels want the offset baked in for sliced tensors
                result = choice.benchmark(*example_inputs_extern, out=out_extern)
            else:
                # triton templates want the base pointer for sliced tensors
                result = choice.benchmark(*example_inputs, out=out)
            if VERIFY:
                torch.testing.assert_close(out_extern, expected, **VERIFY)
            torch.cuda.synchronize()  # shake out any CUDA errors
            return result

        def benchmark_in_current_process(choices):
            from triton.runtime.autotuner import OutOfResources

            inputs = get_inputs()
            example_inputs, _, out, _, _ = inputs
            timings = {}
            for choice in choices:
                try:
                    timing = benchmark_choice_in_current_process(choice, *inputs)
                except CUDACompileError as e:
                    log.warning(
                        "CUDA compilation error: \n%s. \nIgnore this choice.", str(e)
                    )
                    timing = float("inf")
                except RuntimeError as e:
                    msg = str(e)
                    if "invalid argument" in msg:
                        msg += "\n\nThis may mean this GPU is too small for max_autotune mode.\n\n"
                        log.warning(msg)
                        timing = float("inf")
                    else:
                        if "illegal memory access" in msg:
                            msg += "\n\nEither error in template or triton bug.\n"
                        raise ErrorFromChoice(
                            msg, choice, debug_str(example_inputs, out)
                        ) from e
                except OutOfResources as e:
                    log.warning(e)
                    timing = float("inf")

                except AssertionError as e:
                    raise AssertionError(  # noqa: TRY200
                        f"Incorrect result from choice {choice}\n\n{e}"
                    )

                timings[choice] = timing

            return timings

        def benchmark_in_sub_process(choices):
            from . import autotune_process

            # only benchmark triton kernel in sub process for now.
            # ATen/Extern kernel are still benchmarked in the current process.
            extern = [c for c in choices if isinstance(c, ExternKernelCaller)]
            triton = [c for c in choices if not isinstance(c, ExternKernelCaller)]

            timings = benchmark_in_current_process(extern)
            timings.update(autotune_process.benchmark_in_sub_process(triton))
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
        V.debug.log_autotuning_results(name, input_nodes, timings, elapse)
        if not (config.max_autotune or config.max_autotune_gemm) or not PRINT_AUTOTUNE:
            return
        sizes = ", ".join(
            [
                "x".join(
                    map(
                        str,
                        V.graph.sizevars.size_hints(
                            n.get_size(), fallback=config.unbacked_symint_fallback
                        ),
                    )
                )
                for n in input_nodes
            ]
        )
        n = None if log.getEffectiveLevel() == logging.DEBUG else 10
        top_k = sorted(timings, key=timings.__getitem__)[:n]
        best = top_k[0]
        best_time = timings[best]
        sys.stderr.write(f"AUTOTUNE {name}({sizes})\n")
        for choice in top_k:
            result = timings[choice]
            if result:
                sys.stderr.write(
                    f"  {choice.name} {result:.4f} ms {best_time/result:.1%}\n"
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
            " seconds precompiling\n"
        )

    @staticmethod
    def benchmark_example_value(node):
        """
        Convert an ir.Buffer into a concrete torch.Tensor we can use for
        benchmarking.
        """
        if isinstance(node, ir.Layout):
            node = ir.Buffer("fake", node)
        # triton templates want the base tensor.
        if isinstance(node, ir.BaseView):
            node = node.unwrap_view()
        # preserve rng states to avoid the rand_strided call below changes
        # the rng states for the real model code.
        with preserve_rng_state():
            return rand_strided(
                V.graph.sizevars.size_hints(
                    node.get_size(),
                    fallback=config.unbacked_symint_fallback,
                ),
                V.graph.sizevars.size_hints(
                    node.get_stride(),
                    fallback=config.unbacked_symint_fallback,
                ),
                device=node.get_device(),
                dtype=node.get_dtype(),
                extra_size=node.layout.offset,
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


_ALGORITHM_SELECTOR_CACHE: Optional[AlgorithmSelectorCache] = None


def autotune_select_algorithm(*args, **kwargs):
    global _ALGORITHM_SELECTOR_CACHE
    if _ALGORITHM_SELECTOR_CACHE is None:
        _ALGORITHM_SELECTOR_CACHE = AlgorithmSelectorCache()

    if "return_multi_template" not in kwargs:
        kwargs[
            "return_multi_template"
        ] = torch._inductor.config.benchmark_multi_templates

    return _ALGORITHM_SELECTOR_CACHE(*args, **kwargs)


def realize_inputs(*args):
    if len(args) == 1:
        return ir.ExternKernel.require_stride1(ir.ExternKernel.realize_input(args[0]))
    return [realize_inputs(x) for x in args]


# ensure lowering is imported so that `extern_kernels.*` is populated
from . import lowering  # noqa: F401
