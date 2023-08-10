import builtins
import functools
import inspect
import itertools
import logging
import sys
import textwrap
import time
from io import StringIO

from typing import Any, Dict, List, Type, Union
from unittest.mock import patch

import sympy

import torch
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import counters, identity

from . import config, ir
from .autotune_process import BenchmarkRequest, TensorMeta
from .codecache import code_hash, PersistentCache, PyCodeCache

from .codegen.common import IndentedBuffer
from .codegen.triton import texpr, TritonKernel, TritonPrinter, TritonScheduling

from .codegen.triton_utils import config_of, signature_of

from .utils import do_bench, sympy_dot, sympy_product, unique
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
        use_jit=True,
        prefix_args=0,
        suffix_args=0,
        epilogue_fn=identity,
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

    def jit_line(self):
        if self.use_jit:
            return "@triton.jit"

        argdefs, _, signature = self.args.python_argdefs()
        triton_meta = {
            "signature": dict(enumerate(map(signature_of, signature))),
            "device": V.graph.scheduler.current_device.index,
            "constants": {},
        }
        triton_meta["configs"] = [config_of(signature)]
        return (
            f"@template(num_stages={self.num_stages}, num_warps={self.num_warps}, meta={triton_meta!r})\n"
            + "@triton.jit"
        )

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
            return "\n".join(
                [
                    "import triton.language as tl",
                    "import triton",
                    "from torch._inductor.triton_heuristics import template",
                    "from torch._inductor.utils import instance_descriptor",
                    "from torch._inductor import triton_helpers",
                    "",
                    self.jit_line(),
                    f"def {self.kernel_name}({', '.join(arg_defs)}):",
                    self.defines,
                    renames.getvalue(),
                ]
            )

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

    def store_output(self, indices, val, mask):
        """
        Hook called from template code to store the final output
        (if the buffer hasn't been optimized away), then append any
        epilogue fusions.
        """
        assert isinstance(indices, (list, tuple))
        assert isinstance(val, str)
        assert isinstance(mask, str)
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

        V.ops.store(  # type: ignore[attr-defined]
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
            ]
        }

    def indexing(
        self,
        index: sympy.Expr,
        *,
        copy_shape=None,
        dense_indexing=False,
        override_mask=None,
    ):
        """
        Override the default indexing to use our custom mask and force
        dense indexing.
        """
        result, *mask = super().indexing(
            index,
            dense_indexing=False,
            copy_shape=self.template_mask,
            override_mask=self.template_mask,
        )
        return (result, *mask)

    def initialize_range_tree(self, pid_cache):
        super().initialize_range_tree(pid_cache)
        # ignore default codegen
        self.body.clear()
        self.indexing_code.clear()

    def call_kernel(self, name: str):
        wrapper = V.graph.wrapper_code
        _, call_args, _ = self.args.python_argdefs()

        for i in range(len(call_args)):
            if V.graph.is_unspec_arg(call_args[i]):
                call_args[i] = call_args[i] + ".item()"
            if isinstance(call_args[i], sympy.Symbol):
                call_args[i] = texpr(call_args[i])

        if V.graph.cpp_wrapper:
            wrapper.generate_kernel_call(
                name,
                call_args,
                device_index=V.graph.scheduler.current_device.index,
            )
        else:
            call_args = ", ".join(call_args)
            stream_name = wrapper.write_get_cuda_stream(
                V.graph.scheduler.current_device.index
            )

            wrapper.add_import_once(f"import {self.grid_fn.__module__}")
            meta = wrapper.add_meta_once(self.meta)

            grid_call = [
                texpr(V.graph.sizevars.simplify(s)) for s in self.call_sizes
            ] + [meta]
            grid_call = f"{self.grid_fn.__module__}.{self.grid_fn.__name__}({', '.join(grid_call)})"
            wrapper.writeline(
                f"{name}.run({call_args}, grid={grid_call}, stream={stream_name})"
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


class TritonTemplate:
    index_counter = itertools.count()
    all_templates: Dict[str, "TritonTemplate"] = dict()

    @staticmethod
    def _template_from_string(source):
        env = _jinja2_env()
        if env is not None:
            return env.from_string(source)
        return None

    def __init__(self, name: str, grid: Any, source: str, debug=False):
        super().__init__()
        self.name = name
        self.grid = grid
        self.template = self._template_from_string(source)
        assert name not in self.all_templates, "duplicate template name"
        self.all_templates[name] = self
        self.debug = debug

    def maybe_append_choice(
        self,
        choices,
        input_nodes,
        layout,
        num_stages,
        num_warps,
        prefix_args=0,
        suffix_args=0,
        epilogue_fn=identity,
        **kwargs,
    ):
        try:
            choices.append(
                self.generate(
                    input_nodes=input_nodes,
                    layout=layout,
                    num_stages=num_stages,
                    num_warps=num_warps,
                    prefix_args=prefix_args,
                    suffix_args=suffix_args,
                    epilogue_fn=epilogue_fn,
                    **kwargs,
                )
            )
        except NotImplementedError:
            pass

    def generate(
        self,
        input_nodes,
        layout,
        num_stages,
        num_warps,
        prefix_args=0,
        suffix_args=0,
        epilogue_fn=identity,
        **kwargs,
    ):
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
        )
        with patch.object(
            V.graph, "get_dtype", self.fake_get_dtype(fake_out)
        ), TritonTemplateKernel(
            kernel_name=kernel_name,
            output_node=fake_out,
            use_jit=True,
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
            _, call_args, _ = kernel.args.python_argdefs()

        expected_args = list(unique(x.get_name() for x in input_nodes))
        expected_args.extend([fake_out.get_name()])
        assert list(call_args)[: len(expected_args)] == expected_args, (
            call_args,
            expected_args,
        )
        extra_args = V.graph.sizevars.size_hints(
            map(sympy.expand, call_args[len(expected_args) :])
        )

        kernel_hash_name = f"triton_{self.name}_{next(self.index_counter)}"

        def make_kernel_render(out_node):
            kernel = TritonTemplateKernel(
                kernel_name="KERNEL_NAME",
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
        grid = self.grid(*V.graph.sizevars.size_hints(layout.size), kwargs)
        bmreq = BenchmarkRequest(
            module_path=mod.__file__,
            module_cache_key=mod.key,
            kernel_name=kernel_name,
            grid=grid,
            extra_args=extra_args,
            num_stages=num_stages,
            num_warps=num_warps,
            input_tensors=TensorMeta.from_irnodes(input_nodes),
            output_tensor=TensorMeta.from_irnodes(layout),
        )

        return TritonTemplateCaller(
            kernel_hash_name,
            input_nodes,
            layout,
            make_kernel_render,
            extra.strip("-").replace("-", ", "),
            bmreq,
        )

    @staticmethod
    def fake_get_dtype(fake_out):
        _get_dtype_real = V.graph.get_dtype

        def get_dtype(name):
            if name == fake_out.get_name():
                return fake_out.get_dtype()
            return _get_dtype_real(name)

        return get_dtype


class ExternKernelChoice:
    def __init__(
        self,
        kernel,
        cpp_kernel=None,
        *,
        name=None,
        has_out_variant=True,
    ):
        super().__init__()
        name = name or kernel.__name__
        assert callable(kernel)
        assert not hasattr(extern_kernels, name), "duplicate extern kernel"
        self.name = name
        self.cpp_kernel = cpp_kernel
        self.has_out_variant = has_out_variant
        setattr(extern_kernels, name, kernel)

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

    def bind(self, input_nodes, layout, ordered_kwargs_for_cpp_kernel=(), **kwargs):
        self.ordered_kwargs_for_cpp_kernel = ordered_kwargs_for_cpp_kernel
        return ExternKernelCaller(
            self, input_nodes, layout, kwargs, has_out_variant=self.has_out_variant
        )


class ChoiceCaller:
    def __init__(self, name, input_nodes, layout):
        super().__init__()
        self.name = name
        self.layout = layout
        self.input_nodes = input_nodes

    def benchmark(self, *args, out):
        algo = self.to_callable()
        return do_bench(lambda: algo(*args, out=out))

    def call_name(self):
        raise NotImplementedError()

    def to_callable(self):
        raise NotImplementedError()

    def hash_key(self):
        raise NotImplementedError()

    def output_node(self):
        raise NotImplementedError()


class TritonTemplateCaller(ChoiceCaller):
    def __init__(
        self, name, input_nodes, layout, make_kernel_render, debug_extra, bmreq
    ):
        super().__init__(name, input_nodes, layout)
        self.make_kernel_render = make_kernel_render
        self.debug_extra = debug_extra
        self.bmreq = bmreq

    def benchmark(self, *args, out):
        assert self.bmreq is not None
        return self.bmreq.benchmark(*args, output_tensor=out)

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
            ir.TemplateBuffer(
                layout=self.layout,
                inputs=self.input_nodes,
                make_kernel_render=self.make_kernel_render,
            )
        )


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
        if self.has_out_variant:
            return super().benchmark(*args, out=out)
        else:
            algo = self.to_callable()
            out_new = algo(*args)
            torch._C._dynamo.guards.assert_size_stride(  # type: ignore[attr-defined]
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
        cls: Union[Type[ir.ExternKernelOut], Type[ir.ExternKernelAlloc]]
        if self.has_out_variant:
            cls = ir.ExternKernelOut
        else:
            cls = ir.ExternKernelAlloc
        return ir.TensorBox.create(
            cls(
                layout=self.layout,
                inputs=self.input_nodes,
                kernel=self.choice.call_name(),
                cpp_kernel=self.choice.cpp_kernel,
                ordered_kwargs_for_cpp_kernel=self.choice.ordered_kwargs_for_cpp_kernel,
                kwargs=self.kwargs,
            )
        )


class ErrorFromChoice(RuntimeError):
    def __init__(self, msg, choice: ChoiceCaller, inputs_str):
        msg += f"\nFrom choice {choice}\n{inputs_str}"
        super().__init__(msg)
        self.choice = choice


class AlgorithmSelectorCache(PersistentCache):
    def __call__(self, name, choices: List[ChoiceCaller], input_nodes, layout):
        # TODO(nmacchioni): remove once CI tests are fixed
        choices = [choice for choice in choices if choice is not None]
        if len(choices) == 0:
            raise RuntimeError(
                "No choices to select, please consider adding ATEN into max_autotune_gemm_backends "
                "config (defined in torch/_inductor/config.py) to allow at least one choice. "
            )

        if len(choices) == 1:
            return choices[0].output_node()

        @functools.lru_cache(None)
        def make_benchmark_fn():
            return self.make_benchmark_fn(choices, input_nodes, layout)

        def autotune(choice):
            benchmark_fn = make_benchmark_fn()
            try:
                timing = benchmark_fn(
                    choice,
                )
            except RuntimeError as e:
                msg = str(e)
                if "invalid argument" in msg:
                    msg += "\n\nThis may mean this GPU is too small for max_autotune mode.\n\n"
                    log.warning(msg)
                    return float("inf")
                elif "illegal memory access" in msg:
                    msg += "\n\nEither error in template or triton bug.\n"
                raise ErrorFromChoice(msg, choice, benchmark_fn.debug_str())
            except AssertionError as e:
                raise AssertionError(f"Incorrect result from choice {choice}\n\n{e}")
            return timing

        if config.autotune_in_subproc:
            from .autotune_process import tuning_process

            # do the optional warmup
            tuning_process.initialize()
            assert tuning_process.valid()

        autotune_start_ts = time.time()
        timings = self.lookup(
            choices,
            name,
            repr([self.key_of(x) for x in input_nodes]),
            autotune,
        )
        autotune_elapse = time.time() - autotune_start_ts
        if timings == {} or choices[0] not in timings:
            return choices[0].output_node()

        if make_benchmark_fn.cache_info().currsize:
            counters["inductor"]["select_algorithm_autotune"] += 1
            self.log_results(name, input_nodes, timings, autotune_elapse)
        return builtins.min(timings, key=timings.__getitem__).output_node()

    @classmethod
    def make_benchmark_fn(
        cls,
        choices,
        input_nodes,
        layout,
    ):
        # de-duplicate args
        unique_example_inputs = {
            x.get_name(): cls.benchmark_example_value(x) for x in input_nodes
        }
        example_inputs = list(unique_example_inputs.values())
        example_inputs_extern = [
            torch.as_strided(
                unique_example_inputs[input_node.get_name()],
                V.graph.sizevars.size_hints(input_node.get_size()),
                V.graph.sizevars.size_hints(input_node.get_stride()),
                V.graph.sizevars.size_hint(input_node.get_layout().offset),
            )
            for input_node in input_nodes
        ]

        out = cls.benchmark_example_value(layout)
        out_extern = torch.as_strided(
            out, out.size(), out.stride(), V.graph.sizevars.size_hint(layout.offset)
        )
        if VERIFY:
            choices[0].benchmark(*example_inputs_extern, out=out_extern)
            expected = out_extern.clone()

        if DEBUG:
            print(f"{len(choices)} tuning requests:")

        def benchmark_in_current_process(choice):
            if DEBUG:
                start_ts = time.time()
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

        def benchmark_in_sub_process(choice):
            # only benchmark triton kernel in sub process for now.
            # ATen/Extern kernel are still benchmarked in the current process.
            if isinstance(choice, ExternKernelCaller):
                return benchmark_in_current_process(choice)

            from . import autotune_process

            if DEBUG:
                start_ts = time.time()

            out = autotune_process.benchmark_in_sub_process(
                choice,
            )
            if DEBUG:
                elapse = time.time() - start_ts
                print(f"MultiProcessTuning {choice}: {elapse}")
            return out

        benchmark = (
            benchmark_in_sub_process
            if config.autotune_in_subproc
            else benchmark_in_current_process
        )

        def debug_str():
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

        benchmark.debug_str = debug_str  # type: ignore[attr-defined]
        return benchmark

    @staticmethod
    def log_results(name, input_nodes, timings, elapse):
        if not (config.max_autotune or config.max_autotune_gemm) or not PRINT_AUTOTUNE:
            return
        sizes = ", ".join(
            [
                "x".join(map(str, V.graph.sizevars.size_hints(n.get_size())))
                for n in input_nodes
            ]
        )
        top_k = sorted(timings, key=timings.__getitem__)[:10]
        best = top_k[0]
        best_time = timings[best]
        sys.stderr.write(f"AUTOTUNE {name}({sizes})\n")
        for choice in top_k:
            result = timings[choice]
            sys.stderr.write(
                f"  {choice.name} {result:.4f} ms {best_time/result:.1%}\n"
            )

        autotune_type_str = (
            "SubProcess" if config.autotune_in_subproc else "SingleProcess"
        )
        sys.stderr.write(f"{autotune_type_str} AUTOTUNE takes {elapse:.4f} seconds\n")

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
        return rand_strided(
            V.graph.sizevars.size_hints(node.get_size()),
            V.graph.sizevars.size_hints(node.get_stride()),
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
            *sizevars.size_hints(node.get_size()),
            *sizevars.size_hints(node.get_stride()),
            sizevars.size_hint(node.get_layout().offset),
        )


_ALGORITHM_SELECTOR_CACHE = None


def autotune_select_algorithm(*args, **kwargs):
    global _ALGORITHM_SELECTOR_CACHE
    if _ALGORITHM_SELECTOR_CACHE is None:
        _ALGORITHM_SELECTOR_CACHE = AlgorithmSelectorCache()
    return _ALGORITHM_SELECTOR_CACHE(*args, **kwargs)


def realize_inputs(*args):
    if len(args) == 1:
        return ir.ExternKernel.require_stride1(ir.ExternKernel.realize_input(args[0]))
    return [realize_inputs(x) for x in args]


# ensure lowering is imported so that `extern_kernels.*` is populated
from . import lowering  # noqa: F401
