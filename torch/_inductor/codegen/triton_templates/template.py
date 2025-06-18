# mypy: allow-untyped-defs
import functools
import itertools
import logging
from io import StringIO
from types import ModuleType
from typing import Any, Callable, NamedTuple, Optional
from unittest.mock import patch

import sympy

import torch
from torch._dynamo.utils import identity
from torch._inductor.utils import clear_on_fresh_inductor_cache, sympy_product, unique
from torch.utils._ordered_set import OrderedSet

from ... import config, ir
from ...autotune_process import (
    TensorMeta,
    TritonBenchmarkRequest,
    TritonCPUBenchmarkRequest,
    TritonGPUBenchmarkRequest,
)
from ...codecache import PyCodeCache
from ...codegen.common import KernelTemplate, WorkspaceArg, WorkspaceZeroMode
from ...codegen.triton import TritonScheduling
from ...runtime.triton_compat import HAS_WARP_SPEC
from ...utils import Placeholder
from ...virtualized import V
from .caller import TritonTemplateCaller
from .kernel import TritonTemplateKernel


log = logging.getLogger(__name__)


class GenerateAndLoadResult(NamedTuple):
    """
    Return type of TritonTemplate.generate_and_load.
    """

    mod: ModuleType
    extra: str
    input_call_args: tuple[str, ...]
    prologue_supported_inputs: OrderedSet[str]
    kernel_args_sizevars_keys: tuple[sympy.Expr]
    kernel_options: dict[str, Any]


class GeneratedCodeCacheEntry(NamedTuple):
    code: str
    extra: str
    events: list[Any]


class GeneratedCodeCache:
    """
    Cache for generated code. The cache key is a string representation of the input nodes,
    number of stages, number of warps, and call sizes. The cache value is a tuple of the
    generated code, extra code, and events.
    """

    def __init__(self, *args, **kwargs):
        self._cache: dict[str, GeneratedCodeCacheEntry] = {}

    def cache_clear(self) -> None:
        self._cache.clear()

    def __repr__(self):
        return repr(self._cache)

    def make_key(
        self,
        input_nodes: tuple[ir.IRNode],
        num_stages: int,
        num_warps: int,
        call_sizes: list[sympy.core.symbol.Symbol],
        prefix_args: int,
        suffix_args: int,
        epilogue_fn: Optional[Callable[..., Any]],
        epilogue_fn_hash: Optional[str],
        subgraphs: Optional[list[ir.Buffer]],  # has to be none to cache
        workspace_arg: Optional[WorkspaceArg],  # has to be none to cache
        layout: ir.Layout,
        num_consumer_groups: int,
        num_buffers_warp_spec: int,
        kwargs: dict[str, Any],
    ) -> Optional[str]:
        def layout_key(layout: ir.Layout) -> str:
            assert not isinstance(layout, ir.FlexibleLayout)
            return repr(
                [
                    layout.size,
                    layout.stride,
                    layout.dtype,
                    layout.device,
                    layout.offset,
                ]
            )

        def has_flexible_layout() -> bool:
            if isinstance(layout, ir.FlexibleLayout):
                return True

            for input in input_nodes:
                if isinstance(input.get_layout(), ir.FlexibleLayout):
                    return True
            return False

        if epilogue_fn is identity:
            assert epilogue_fn_hash is None
            epilogue_fn_hash = "identity"

        # we do not cache under those conditions right now.
        if (
            has_flexible_layout()
            or subgraphs is not None
            or workspace_arg is not None
            or epilogue_fn_hash is None
        ):
            return None

        return repr(
            {
                "input_nodes": [
                    layout_key(input.get_layout()) for input in input_nodes
                ],
                "num_stages": num_stages,
                "num_warps": num_warps,
                "prefix_args": prefix_args,
                "suffix_args": suffix_args,
                "call_sizes": call_sizes,
                "layout": layout_key(layout),
                "num_consumer_groups": num_consumer_groups,
                "num_buffers_warp_spec": num_buffers_warp_spec,
                "epilogue_fn_hash": epilogue_fn_hash,
                "kwargs": kwargs,
            }
        )

    def get_entry(self, cache_key: Optional[str]) -> Optional[GeneratedCodeCacheEntry]:
        if cache_key is None:
            return None

        entry = self._cache.get(cache_key, None)
        if entry is None:
            torch._dynamo.utils.counters["inductor"]["generated_module_cache_miss"] += 1
        else:
            torch._dynamo.utils.counters["inductor"]["generated_module_cache_hit"] += 1
        return entry

    def put_entry(
        self,
        cache_key: Optional[str],
        code: str,
        extra: str,
        events: list[Any],
    ) -> None:
        if cache_key is None:
            return
        entry = GeneratedCodeCacheEntry(code, extra, events)
        self._cache.update({cache_key: entry})


class TritonTemplate(KernelTemplate):
    """
    A Triton template is a template that can be used to generate a Triton kernel.
    """

    # Allow subclasses to override the kernel type
    kernel_type: type[Any] = TritonTemplateKernel
    index_counter = itertools.count()
    all_templates: dict[str, "TritonTemplate"] = {}

    def __init__(
        self,
        name: str,
        grid: Any,
        source: str,
        debug=False,
        cache_codegen_enabled_for_template=False,
        prologue_loads_all_inputs=False,
    ) -> None:
        super().__init__(name)
        self.grid = grid
        self.template = self._template_from_string(source)
        assert name not in self.all_templates, "duplicate template name"
        TritonTemplate.all_templates[name] = self
        self.debug = debug
        self._cache_codegen_enabled_for_template = cache_codegen_enabled_for_template
        self._generated_code_cache: GeneratedCodeCache = GeneratedCodeCache()
        clear_on_fresh_inductor_cache(self._generated_code_cache)
        # When prologue_loads_all_inputs is true, prologue_supported_inputs is populated during def_kernel
        # by adding all inputs.
        self.prologue_loads_all_inputs = prologue_loads_all_inputs

    # When this flag is on, we ensure that the cached results and the generated result if cache
    # was not used are the same.
    test_cache = False

    def maybe_append_choice(
        self, choices: list[Any], **kwargs: Any
    ) -> Optional[NotImplementedError]:
        """
        Maybe generates a new ChoiceCaller and appends it into existing choices.
        Returns None if success, otherwise returns the error.

        choices: A list of ChoiceCallers.
        kwargs: Additional kwargs to be passed to self.generate() to generate a new ChoiceCaller.
        """

        try:
            choices.append(self.generate(generate_with_caching=True, **kwargs))
            return None
        except NotImplementedError as e:
            log.info(
                "Cannot Append Choice: %s. KernelTemplate type is %s",
                e,
                type(self),
                stack_info=log.getEffectiveLevel() < logging.INFO,
            )
            return e

    # NOTE: MAKE SURE THAT ANY ARGUMENT ADDED TO THIS FUNCTION IS PROPERLY HANDLED IN _generated_code_cache.make_key.
    def generate_and_load(
        self,
        input_nodes: tuple[ir.IRNode],
        num_stages: int,
        num_warps: int,
        call_sizes: list[sympy.core.symbol.Symbol],
        prefix_args: int,
        suffix_args: int,
        epilogue_fn: Optional[Callable[..., Any]],
        epilogue_fn_hash: Optional[str],
        subgraphs: Optional[list[ir.Buffer]],
        workspace_arg: Optional[WorkspaceArg],
        num_consumer_groups: int,
        num_buffers_warp_spec: int,
        layout: ir.Layout,
        kwargs: dict[str, Any],
        generate_with_caching,
    ) -> Optional[GenerateAndLoadResult]:
        """Generate the python code and load it into the current process"""
        caching_enabled = (
            generate_with_caching
            and torch._inductor.config.enable_caching_generated_triton_templates
        )

        cache_key = None
        if caching_enabled:
            cache_key = self._generated_code_cache.make_key(
                input_nodes,
                num_stages,
                num_warps,
                call_sizes,
                prefix_args,
                suffix_args,
                epilogue_fn,
                epilogue_fn_hash,
                subgraphs,
                workspace_arg,
                layout,
                num_consumer_groups,
                num_buffers_warp_spec,
                kwargs,
            )

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
            "prologue_loads_all_inputs": self.prologue_loads_all_inputs,
        }

        if HAS_WARP_SPEC:
            kernel_options.update(
                {
                    "num_consumer_groups": num_consumer_groups,
                    "num_buffers_warp_spec": num_buffers_warp_spec,
                }
            )

        def make_kernel():
            return self.kernel_type(
                kernel_name=kernel_name,
                output_node=fake_out,
                workspace_arg=workspace_arg,
                use_jit=False,
                **kernel_options,
            )

        def generate_code(kernel) -> Optional[tuple[str, str]]:
            def make_extra() -> str:
                extra_parts = [
                    f"{kwarg}={repr(kwargs[kwarg])}" for kwarg in sorted(kwargs.keys())
                ]

                extra_parts.extend(
                    [
                        f"num_stages={num_stages}",
                        f"num_warps={num_warps}",
                    ]
                )
                if HAS_WARP_SPEC:
                    extra_parts.extend(
                        [
                            f"num_consumer_groups={num_consumer_groups}",
                            f"num_buffers_warp_spec={num_buffers_warp_spec}",
                        ]
                    )
                extra = "-".join(extra_parts) + "-"
                return extra

            try:
                template = kernel.render(self.template, kwargs, caching_enabled)
                with kernel.set_subgraph_body("<STORE_OUTPUT>"):
                    code = template.finalize_all()
            except ZeroDivisionError:
                # TODO(nmacchioni): fix sympy division by zero
                return None
            if self.debug:
                print("Generated Code:\n", code)

            extra = make_extra()
            return code, extra

        def maybe_test_cache(code: str, extra: str, kernel):
            if self.test_cache or self.debug:
                with (
                    patch.object(V.graph, "get_dtype", self._fake_get_dtype(fake_out)),
                    V.graph.set_current_device(layout.device),
                    make_kernel() as kernel_test,
                ):
                    result2 = generate_code(kernel_test)
                    assert result2 is not None
                    code_test, extra_test = result2
                    assert (
                        code == code_test
                        and extra == extra_test
                        and kernel.args.input_buffers == kernel_test.args.input_buffers
                        and kernel.prologue_supported_inputs
                        == kernel_test.prologue_supported_inputs
                        and kernel.args.sizevars == kernel_test.args.sizevars
                    ), "Generated code cache results in wrong output"

        # Generate code, extra.
        code: Optional[str] = None
        extra: Optional[str] = None
        with (
            patch.object(V.graph, "get_dtype", self._fake_get_dtype(fake_out)),
            V.graph.set_current_device(layout.device),
            make_kernel() as kernel,
        ):
            cache_entry = self._generated_code_cache.get_entry(cache_key)
            cache_hit = False

            if cache_entry is not None:
                code, extra, events = cache_entry
                kernel.replay_cached_events(events)
                cache_hit = True

            else:
                result = generate_code(kernel)
                if result is None:  # happens at ZeroDivisionError:
                    return None
                code, extra = result
                self._generated_code_cache.put_entry(
                    cache_key, code, extra, kernel.cached_replay_events
                )

        assert code is not None and extra is not None

        mod = PyCodeCache.load(code, extra)

        input_call_args = tuple(kernel.args.input_buffers.keys())
        prologue_supported_inputs = kernel.prologue_supported_inputs.copy()
        kernel_args_sizevars_keys = tuple(kernel.args.sizevars.keys())

        if cache_hit:
            maybe_test_cache(code, extra, kernel)

        return GenerateAndLoadResult(
            mod,
            extra,
            input_call_args,
            prologue_supported_inputs,
            kernel_args_sizevars_keys,
            kernel_options,
        )

    def generate(  # type: ignore[override]
        self,
        input_nodes: tuple[ir.IRNode],
        layout: ir.Layout,
        num_stages: int,
        num_warps: int,
        num_consumer_groups: int = 0,
        num_buffers_warp_spec: int = 0,
        prefix_args: int = 0,
        suffix_args: int = 0,
        epilogue_fn: Optional[Callable[..., Any]] = identity,
        epilogue_fn_hash: Optional[str] = None,
        subgraphs: Optional[list[ir.Buffer]] = None,
        mutated_inputs: Optional[list[ir.IRNode]] = None,
        call_sizes: Optional[list[sympy.core.symbol.Symbol]] = None,
        workspace_arg: Optional[WorkspaceArg] = None,
        generate_with_caching=False,
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
        # HACK: Triton currently breaks if TF32 floats are requested, but the CUDA
        # capability doesn't support them.  This is a bug in Triton, but for now we'll
        # patch around it here.  See https://github.com/triton-lang/triton/issues/3011
        # for one example issue with this problem.
        if torch.cuda.is_available() and not torch.cuda.is_tf32_supported():
            kwargs["ALLOW_TF32"] = "False"

        if call_sizes is None:
            call_sizes = layout.size

        result = self.generate_and_load(
            input_nodes,
            num_stages,
            num_warps,
            call_sizes,
            prefix_args,
            suffix_args,
            epilogue_fn,
            epilogue_fn_hash,
            subgraphs,
            workspace_arg,
            num_consumer_groups,
            num_buffers_warp_spec,
            layout,
            kwargs,
            generate_with_caching and self._cache_codegen_enabled_for_template,
        )

        # May happen as result of dev by 0.
        if result is None:
            return None

        # We expect the input_buffer order to be [*input_nodes, *captured_buffers]
        expected_input_args = tuple(unique(x.get_name() for x in input_nodes))
        assert (
            result.input_call_args[: len(expected_input_args)] == expected_input_args
        ), (
            result.input_call_args,
            expected_input_args,
        )

        full_input_nodes = tuple(
            [V.graph.get_buffer(k) for k in result.input_call_args]
        )
        extra_args = V.graph.sizevars.size_hints(
            map(sympy.expand, result.kernel_args_sizevars_keys),
            fallback=config.unbacked_symint_fallback,
        )

        kernel_hash_name = f"triton_{self.name}_{next(self.index_counter)}"

        workspace_args = []
        if workspace_arg is not None:
            # Create workspace tensor
            workspace_size = workspace_arg.count
            workspace_tensor = torch.empty_strided(
                (workspace_size,),
                (1,),
                dtype=torch.uint8,
                device=layout.device.type,
            )

            # Handle zero initialization if needed
            if workspace_arg.zero_mode != WorkspaceZeroMode.UNINITIALIZED:
                workspace_tensor.zero_()

            workspace_args.append(workspace_tensor)

        options = result.kernel_options

        def make_kernel_render(out_node):
            assert result is not None
            kernel = self.kernel_type(
                kernel_name=str(Placeholder.KERNEL_NAME),
                output_node=out_node,
                workspace_arg=workspace_arg,
                use_jit=False,
                **options,
            )
            render = functools.partial(
                kernel.render,
                self.template,
                kwargs,
            )
            return kernel, render

        # create the BenchmarkRequest
        assert result.mod.__file__ is not None
        grid = self.grid(
            *V.graph.sizevars.size_hints(
                call_sizes,
                fallback=config.unbacked_symint_fallback,
            ),
            kwargs,
        )
        bmreq_cls: type[TritonBenchmarkRequest]
        if layout.device.type == "cpu":
            bmreq_cls = TritonCPUBenchmarkRequest
        else:
            bmreq_cls = TritonGPUBenchmarkRequest
        bmreq = bmreq_cls(
            module_path=result.mod.__file__,
            module_cache_key=result.mod.key,
            kernel_name=f"triton_{self.name}",
            extra_args=[*extra_args, *workspace_args, *grid],
            num_stages=num_stages,
            num_warps=num_warps,
            num_consumer_groups=num_consumer_groups,
            num_buffers_warp_spec=num_buffers_warp_spec,
            matrix_instr_nonkdim=kwargs.get("matrix_instr_nonkdim", 0),
            waves_per_eu=kwargs.get("waves_per_eu", 0),
            kpack=kwargs.get("kpack", 2),
            input_tensor_meta=TensorMeta.from_irnodes(full_input_nodes),  # type: ignore[arg-type]
            output_tensor_meta=TensorMeta.from_irnodes(layout),
        )

        return TritonTemplateCaller(
            kernel_hash_name,
            full_input_nodes,
            layout,
            make_kernel_render,
            result.extra.strip("-").replace("-", ", "),
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
                "GROUP_M": kwargs.get("GROUP_M", -1),
                "allow_tf32": str(kwargs.get("ALLOW_TF32", None)),
                "acc_type": str(kwargs.get("ACC_TYPE", None)),
                "matrix_instr_nonkdim": kwargs.get("matrix_instr_nonkdim", 0),
                "waves_per_eu": kwargs.get("waves_per_eu", 0),
                "kpack": kwargs.get("kpack", 2),
            },
            mutated_inputs=mutated_inputs,
            workspace_arg=workspace_arg,
            allowed_prologue_inps=result.prologue_supported_inputs,
        )
