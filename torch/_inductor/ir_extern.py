from __future__ import annotations

from .ir_base import (
    AbstractContextManager,
    Any,
    Argument,
    BackendFeature,
    Buffer,
    CUTLASSTemplate,
    Callable,
    ComputedBuffer,
    Constant,
    ConstantBuffer,
    Dep,
    Expr,
    FakeScriptObject,
    FixedLayout,
    FlexibleLayout,
    GPU_ALIGN_BYTES,
    GeneratorState,
    GraphLowering,
    GraphModuleSerializer,
    IRNode,
    InputBuffer,
    Iterable,
    Iterator,
    Kernel,
    Layout,
    LoopBody,
    MutationLayoutSHOULDREMOVE,
    NHWC_STRIDE_ORDER,
    NHWDC_STRIDE_ORDER,
    Never,
    Node,
    NonOwningLayout,
    NonTensorObj,
    NoneAsConstantBuffer,
    NoneLayout,
    OpaqueObjectState,
    Operation,
    OperationBuffer,
    OrderedSet,
    OutputSpec,
    PythonWrapperCodegen,
    Sequence,
    ShapeAsConstantBuffer,
    SymTypes,
    Symbol,
    SympyBoolean,
    TorchBindObject,
    TypeIs,
    V,
    _IntLike,
    _OpOverloads,
    _P,
    _T,
    _disable_current_modes,
    _remove_effect_token_unbacked_bindings,
    aten,
    benchmarker,
    cache_on_self_and_args,
    can_auto_functionalize,
    cast,
    compute_unbacked_bindings,
    config,
    contextlib,
    convert_shape_to_inductor,
    dataclasses,
    dependencies,
    developer_warning,
    do_bench_using_profiling,
    export_schema,
    free_symbols,
    free_unbacked_symbols,
    get_free_symbols,
    get_kernel_metadata,
    get_stride_order,
    get_symbolic_inputs,
    ir_dataclass,
    ir_node_to_tensor,
    is_contiguous_for_memory_format_or_false,
    is_dynamic,
    is_gpu,
    is_opaque_value,
    itertools,
    library_utils,
    log,
    make_channels_last_strides_for,
    nullcontext,
    ops,
    override,
    pytree,
    rebind_unbacked,
    resolve_unbacked_bindings,
    significant_strides_equal,
    sympy,
    sympy_dot,
    sympy_index_symbol,
    sympy_product,
    sympy_subs,
    tensor_is_aligned,
    torch,
    try_get_name,
    try_match_insignificant_strides,
    var_builder,
)
from .ir_compute import Loops, Pointwise
from .ir_views import (
    BaseView,
    ReinterpretView,
    SliceView,
    as_storage_and_layout,
    is_storage_and_layout,
    is_stride_order_storage_and_layout,
    is_unaligned,
)
from .ir_containers import MutableBox, StorageBox, TensorBox


@dataclasses.dataclass(frozen=True)
class FinalizeCodegenResult:
    """Structured result from TemplateBuffer._finalize_codegen for external backends."""

    source: str
    imports: list[str]
    call_preamble: list[str]
    call_args: list[str]


class TemplateBuffer(OperationBuffer):
    """
    Base class for template operators that support epilogue and prologue fusion.
    Subclasses: TritonTemplateBuffer (built-in Triton templates),
    HelionTemplateBuffer (Helion kernels), etc.
    """

    def __init__(
        self,
        layout: OutputSpec,
        inputs: Sequence[IRNode],
        make_kernel_render: Callable[..., Any] | None,
        mutated_inputs: Iterable[IRNode] | None = None,
        allowed_prologue_inps: OrderedSet[str] | None = None,
        named_inputs: dict[str, IRNode] | None = None,
    ) -> None:
        super().__init__(name=None, layout=layout)
        self.inputs = InputsKernel.unwrap_storage(inputs)
        self.make_kernel_render = make_kernel_render
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)
        # Annotations dict for storing metadata (e.g., KernelTemplateChoice)
        self.annotations: dict[str, Any] = {}

        # Output buffer names eligible for epilogue fusion.
        # Maps buffer name → kernel parameter name (e.g. "buf3" → "result").
        self.epilogue_fusable_outputs: dict[str, str] = {}
        # For multi-output kernels: maps child buffer name → MultiOutput
        # node.  Used by call_kernel to emit tuple-unpacking lines.
        self._multi_output_children: dict[str, MultiOutput] = {}
        # Maps kernel parameter name → IRNode for each tensor input.
        # Used by ExternalTritonTemplateKernel to set up prologue fusion and
        # by HelionTemplateBuffer to resolve call arguments.
        self._named_inputs: dict[str, IRNode] = (
            dict(named_inputs) if named_inputs else {}
        )

        # Inputs that the kernel mutates in-place
        self.mutated_inputs = mutated_inputs
        self.mutation_outputs: list[MutationOutput] = []
        if mutated_inputs is not None:
            first_input = self.inputs[0]
            assert isinstance(first_input, IRNode), type(first_input)
            device = first_input.get_device()
            self.mutation_outputs = [
                MutationOutput(NoneLayout(device=device), buf, self)
                for buf in mutated_inputs
            ]
        # Input buffer names eligible for prologue fusion.
        self.allowed_prologue_inps: OrderedSet[str] = (
            allowed_prologue_inps or OrderedSet()
        )
        # Per-template fusion overrides.  None means fall back to global
        # config.epilogue_fusion / config.prologue_fusion.
        self.allow_epilogue_fusion: bool | None = None
        self.allow_prologue_fusion: bool | None = None

    @property
    def dtype(self) -> torch.dtype:
        if isinstance(self.layout, MultiOutputLayout):
            raise NotImplementedError(
                "Multi-output templates do not have a single dtype"
            )
        return self.get_layout().dtype

    def get_read_writes(self) -> dependencies.ReadWrites:
        return self.extract_read_writes(normalize=True)

    def _read_deps_from_inputs(self, normalize: bool) -> OrderedSet[dependencies.Dep]:
        """Build read dependencies from all inputs."""
        reads: OrderedSet[dependencies.Dep] = OrderedSet()
        for inp_raw in self.inputs:
            assert isinstance(inp_raw, (ReinterpretView, Buffer)), type(inp_raw)
            inp: ReinterpretView | Buffer = inp_raw
            assert isinstance(inp.layout, Layout), type(inp.layout)
            inp_indexer = inp.layout.make_indexer()

            def dummy(index: Sequence[Any], rindex: Sequence[Any]) -> Any:
                assert len(rindex) == 0
                return ops.load(inp.get_name(), inp_indexer(index))

            reads |= dependencies.extract_read_writes(
                dummy, inp.get_size(), (), normalize=normalize
            ).reads
        return reads

    def extract_read_writes(self, normalize: bool = False) -> dependencies.ReadWrites:
        """Extract read/write dependencies for this TemplateBuffer.

        When the layout is MultiOutputLayout (multi-output templates), the
        buffer itself has no data layout, so we cannot build an indexer.
        Instead, synthesize a trivial write dep and derive read deps from
        the named tensor inputs (``_named_inputs``).  For single-output
        templates with a concrete layout, fall through to the standard path.
        """
        if isinstance(self.layout, MultiOutputLayout):
            writes: OrderedSet[dependencies.Dep] = OrderedSet(
                [
                    dependencies.MemoryDep(
                        self.get_name(), sympy.Integer(0), var_names=(), size=()
                    ),
                ]
            )
            return dependencies.ReadWrites(
                reads=self._read_deps_from_inputs(normalize),
                writes=writes,
                index_exprs=OrderedSet(),
                range_vars=None,
                var_ranges=None,
            )

        name = self.get_name()
        indexer = self.get_layout().make_indexer()

        def dummy(index: Sequence[Any], rindex: Sequence[Any]) -> Any:
            assert len(rindex) == 0
            return ops.store(name, indexer(index), "fake")

        deps = dependencies.extract_read_writes(
            dummy, self.get_size(), (), normalize=normalize
        )
        deps.reads |= self._read_deps_from_inputs(normalize)
        return deps

    def get_reduction_size(self) -> Sequence[Expr]:
        return sympy.S.One

    def get_reduction_type(self) -> str | None:
        return None

    def should_allocate(self) -> bool:
        return True

    def simplify_and_reorder(
        self,
        extra_indexing_constraints: tuple[dict[Any, Any], list[Any]] | None = None,
        recompute_sizes_body_func: Callable[..., Any] | None = None,
    ) -> tuple[tuple[Sequence[Expr], list[Expr]], LoopBody | None]:
        return (
            (
                self.get_size(),
                [],
            ),
            None,
        )

    def is_multi_outputs_template(self) -> bool:
        """Whether this template produces multiple outputs via MultiOutputLayout."""
        return isinstance(self.layout, MultiOutputLayout)

    def get_allowed_prologue_inps(self) -> OrderedSet[str]:
        return self.allowed_prologue_inps

    def _finalize_codegen(
        self, hook_outputs: dict[str, str]
    ) -> FinalizeCodegenResult | None:
        """Called after epilogue/prologue subgraph codegen with rendered hook outputs.

        ``hook_outputs`` maps placeholder keys (e.g. ``<STORE_OUTPUT_0>``,
        ``<LOAD_INPUT_x>``) to the Triton code generated by Inductor for
        each fused subgraph.

        Return a ``FinalizeCodegenResult`` to provide custom source code and
        call metadata, or ``None`` to use the default codegen path.
        """
        return None

    @classmethod
    def realize_template_input(cls, tb: TensorBox) -> IRNode:
        """Realize a TensorBox, preserving MultiOutput layout (unlike ExternKernel.realize_input)."""
        if isinstance(tb, TensorBox) and isinstance(tb.data, MultiOutput):
            return tb.data
        result = ExternKernel.realize_input(tb)
        if isinstance(result, StorageBox):
            result = result.data
        if isinstance(result.layout, FlexibleLayout):  # type: ignore[union-attr]
            result.freeze_layout()
        return result

    @classmethod
    def build_multi_outputs(
        cls,
        template_buf: TemplateBuffer,
        structured: object,
        *,
        direct_alias_at_leaf: dict[int, IRNode] | None = None,
        on_tensor_leaf: Callable[[str, MultiOutput, list[tuple[type, int]], int], None]
        | None = None,
        on_non_tensor_leaf: Callable[[int], None] | None = None,
    ) -> tuple[TensorBox, ...]:
        """Walk a structured output tree, creating MultiOutput nodes for tensor leaves."""
        seen_outputs: dict[int, TensorBox] = {}
        leaf_counter = itertools.count()

        def walk(output: object, indices: list[tuple[type, int]]) -> list[TensorBox]:
            if isinstance(output, (list, tuple)):
                results: list[TensorBox] = []
                for i, item in enumerate(output):
                    results.extend(walk(item, [*indices, (type(output), i)]))
                return results
            leaf_idx = next(leaf_counter)
            if isinstance(output, torch.Tensor):
                if direct_alias_at_leaf and leaf_idx in direct_alias_at_leaf:
                    return [TensorBox.create(direct_alias_at_leaf[leaf_idx])]
                tid = id(output)
                if tid in seen_outputs:
                    return [seen_outputs[tid]]
                mo = MultiOutput(
                    FallbackKernel.tensor_to_layout(output), template_buf, indices
                )
                template_buf._multi_output_children[mo.get_name()] = mo
                if on_tensor_leaf is not None:
                    on_tensor_leaf(mo.get_name(), mo, indices, leaf_idx)
                tb = TensorBox(mo)
                seen_outputs[tid] = tb
                return [tb]
            # Non-tensor leaf (int, SymInt, None, etc.)
            if on_non_tensor_leaf is not None:
                on_non_tensor_leaf(leaf_idx)
            return []

        return tuple(walk(structured, []))


class TritonTemplateBuffer(TemplateBuffer):
    def __init__(
        self,
        layout: Layout,
        inputs: Sequence[IRNode],
        make_kernel_render: Callable[_P, _T] | None,
        mutated_inputs: Iterable[IRNode] | None = None,
        allowed_prologue_inps: OrderedSet[str] | None = None,
    ) -> None:
        """
        NOTE:[TritonTemplates with multiple outputs]
        We want the ability for TritonTemplates to output multiple tensors. Triton
        kernels have no notion of outputs and this is done by creating tensors that
        are then mutated by the kernel. Currently our STORE_OUTPUT codegen doesn't
        support creating multinode outputs for triton templates.
        We work around this by creating an extra input buffer during the lowering
        and we mark them as mutated inputs.
        """
        super().__init__(
            layout,
            inputs,
            make_kernel_render,
            mutated_inputs=mutated_inputs,
            allowed_prologue_inps=allowed_prologue_inps,
        )
        assert self.name is not None
        self.epilogue_fusable_outputs = {self.name: self.name}

        self.subgraph_inps: list[IRNode | sympy.Expr | None] | None = None
        self.subgraph_outs: list[IRNode | None] | None = None

    @cache_on_self_and_args("TritonTemplateBuffer")
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        res = super().get_free_symbol_uses(unbacked_only)
        subgraph_outs = self.subgraph_outs if self.subgraph_outs else []
        subgraph_inps = self.subgraph_inps if self.subgraph_inps else []

        for inp in subgraph_inps:
            if isinstance(inp, sympy.Expr):
                res.update(get_free_symbols(inp, unbacked_only))
            elif isinstance(inp, IRNode):
                res.update(inp.get_free_symbol_uses(unbacked_only))
            else:
                assert inp is None

        for out in subgraph_outs:
            if isinstance(out, IRNode):
                res.update(out.get_free_symbol_uses(unbacked_only))
            else:
                assert out is None

        return res

    def get_outputs(self) -> list[Buffer]:
        return [self, *self.mutation_outputs]

    def __str__(self) -> str:
        out = f"TritonTemplateBuffer(layout={self.layout})"
        return out


PrimitiveInfoType = int | float | bool | str | list[int | str | float | bool]


class ChoiceCaller:
    """
    Represents a possible choice used in autotune_process.py.
    During autotuning, self.benchmark() is first called to get benchmark result,
    and if this choice is selected, self.output_node() is called to get the output_node.

    Children classes: TritonTemplateCaller, CUTLASSTemplateCaller.
    """

    def __init__(
        self,
        name: str,
        input_nodes: list[Buffer],
        layout: Layout,
        description: str,
    ) -> None:
        super().__init__()
        self.name = name
        self.layout = layout
        self.input_nodes = input_nodes
        # An additional description used to describe the choice (useful for
        # knowing what autotuning is choosing)
        self.description = description
        self.failed: bool = False
        # When True, benchmark using CUDA graph capture/replay
        self._benchmark_with_cudagraphs: bool = False
        # A place to store annotations that can be read post benchmarking
        # Use this to shuttle information between ChoieCaller generation
        # and the end of benchmarking
        self.annotations: dict[Any, Any] = {}
        # Subclass-overridden attributes for subgraph-based choices
        self.gm: torch.fx.GraphModule | None = None
        self.decomposition: Callable[..., Any] | None = None
        self.decomposition_kwargs: dict[str, Any] = {}
        self.config_patches: dict[str, Any] = {}

    def benchmark(self, *args: Any, out: torch.Tensor) -> float:
        algo = self.to_callable()
        if self._benchmark_with_cudagraphs:
            return benchmarker.benchmark_gpu_with_cuda_graph(lambda: algo(*args))
        if config.profile_bandwidth_with_do_bench_using_profiling:
            return do_bench_using_profiling(lambda: algo(*args))  # type: ignore[arg-type]
        return benchmarker.benchmark(algo, args, {"out": out}, device=None)

    def call_name(self) -> str:
        raise NotImplementedError

    def to_callable(self) -> Callable[..., Any]:
        raise NotImplementedError

    def kernel_hash_key(self) -> str:
        """
        Hash key for the underlying kernel. By default, we assume there are no
        runtime params, so kernel hash key defaults to choice caller's hash key.
        """
        return self.hash_key()

    def hash_key(self) -> str:
        raise NotImplementedError

    def output_node(self) -> TensorBox:
        raise NotImplementedError

    def info_dict(self) -> dict[str, PrimitiveInfoType | list[PrimitiveInfoType]]:
        """Information returned here is logged to the autotune log file when that is enabled."""
        return {}

    def autoheuristic_id(self) -> str:
        return "unsupported_choice"

    def mark_failed(self) -> None:
        """
        Mark the choice as failed so that it can be
        removed later. Useful for when we decouple
        compilation and tuning.
        """
        self.failed = True


class TritonTemplateCallerBase(ChoiceCaller):
    def get_make_kernel_render(self) -> Any:
        raise NotImplementedError


class MultiTemplateBuffer(TritonTemplateBuffer):
    """
    Represents a Buffer with multiple backing implementation choices.

    Choices can be TritonTemplates or ExternKernels. During scheduling if there is a potential
    epilogue we will benchmark each of the choices with the epilogue to determine an implementation.
    Otherwise, the fastest base choice will be chosen.
    """

    def __init__(
        self,
        layout: Layout,
        inputs: Sequence[IRNode],
        choice_timings_fn: Callable[[int | None], dict[ChoiceCaller, float]],
        unfiltered_choices: list[ChoiceCaller],
        allowed_prologue_inps: OrderedSet[str],
    ) -> None:
        super().__init__(
            layout=layout,
            inputs=inputs,
            make_kernel_render=None,
            allowed_prologue_inps=allowed_prologue_inps,
        )
        self._choice_timings_fn = choice_timings_fn
        self._choice_timings: dict[int | None, dict[ChoiceCaller, float]] = {}
        self._choices: list[ChoiceCaller] = unfiltered_choices
        self.original_inputs = inputs
        self._output_plannable = all(
            isinstance(choice, TritonTemplateCallerBase)
            or (
                isinstance(choice, torch._inductor.select_algorithm.ExternKernelCaller)
                and choice.has_out_variant
            )
            for choice in unfiltered_choices
        )
        self._make_kernel_renders: dict[int | None, Any] = {}

    @property
    def output_plannable(self) -> bool:
        """
        Are all possible choices TritonTemplates or Extern Kernels with out variants
        """
        return self._output_plannable

    @property
    def choices(self) -> list[ChoiceCaller]:
        return self._choices

    def choice_timings(
        self, hint_override: int | None = None
    ) -> dict[ChoiceCaller, float]:
        if hint_override not in self._choice_timings:
            self._choice_timings[hint_override] = self._choice_timings_fn(hint_override)
        return self._choice_timings[hint_override]

    @contextlib.contextmanager
    def swap_as_triton_caller(self, caller: TritonTemplateCallerBase) -> Iterator[None]:
        assert isinstance(
            caller, torch._inductor.select_algorithm.TritonTemplateCaller
        ), type(caller)
        assert self.layout == caller.layout

        render = self.make_kernel_render
        self.make_kernel_render = caller.get_make_kernel_render()
        try:
            yield
        finally:
            self.make_kernel_render = render

    def finalize_as_triton_caller(self, caller: TritonTemplateCallerBase) -> None:
        assert isinstance(
            caller, torch._inductor.select_algorithm.TritonTemplateCaller
        ), type(caller)
        assert self.get_size() == caller.layout.size
        assert self.get_stride() == caller.layout.stride
        self.make_kernel_render = caller.get_make_kernel_render()

    def get_min_choice(
        self, hint_override: int | None = None
    ) -> tuple[ChoiceCaller, float]:
        timings = self.choice_timings(hint_override=hint_override)
        min_choice = min(timings, key=timings.get)  # type: ignore[arg-type]
        return (min_choice, timings[min_choice])

    def finalize_as_triton_callers(
        self, callers: dict[int | None, TritonTemplateCallerBase]
    ) -> None:
        """Finalize with multiple callers for different hint overrides"""
        for hint_override, caller in callers.items():
            self._make_kernel_renders[hint_override] = caller.get_make_kernel_render()

        # Set the default to be the one without hint override
        self.make_kernel_render = self._make_kernel_renders[None]


class CUTLASSTemplateBuffer(TemplateBuffer):
    def __init__(
        self,
        layout: Layout,
        inputs: Sequence[IRNode],
        make_kernel_render: Callable[_P, _T],
        workspace_size: int,
        template: CUTLASSTemplate,
        supports_epilogue_fusion: bool,
    ) -> None:
        super().__init__(layout, inputs, make_kernel_render)
        # Global memory (in bytes) needed for this template.
        self.workspace_size = workspace_size
        self.template = template
        self.supports_epilogue_fusion = supports_epilogue_fusion

    def get_workspace_size(self) -> int:
        return self.workspace_size if self.workspace_size is not None else 0

    def emulate_store_fn(self) -> None:
        for output in self.get_outputs():
            ops.store(output.get_name(), None, None)


class CppTemplateBuffer(TemplateBuffer):
    def __init__(
        self,
        layout: Layout,
        inputs: Sequence[IRNode],
        make_kernel_render: Callable[_P, _T],
        template: CUTLASSTemplate,
        choice: Any,
    ) -> None:
        super().__init__(layout, inputs, make_kernel_render)
        self.template = template
        self.choice = choice
        self.outputs: list[Buffer] | None = None

    def get_layout(self) -> Layout:
        if isinstance(self.layout, MultiOutputLayout):
            assert isinstance(self.outputs, Iterable), type(self.outputs)

            first_output = self.outputs[0]
            assert isinstance(first_output, Buffer), type(first_output)
            layout = first_output.layout
            assert isinstance(layout, Layout), type(layout)
            return layout
        else:
            return super().get_layout()


class CuteDSLTemplateBuffer(TemplateBuffer):
    """
    Buffer for CuteDSL (CUTLASS Python DSL) template kernels.
    Similar to other template buffers but specialized for CuteDSL operations.
    """

    def __init__(
        self,
        layout: Layout,
        inputs: Sequence[IRNode],
        make_kernel_render: Callable[_P, _T],
        template: Any,
        mutated_inputs: Iterable[IRNode] | None = None,
    ) -> None:
        super().__init__(layout, inputs, make_kernel_render)
        self.template = template
        self.mutated_inputs = mutated_inputs
        self.outputs: list[Buffer] = [self]

        if mutated_inputs is not None:
            assert isinstance(self.inputs[0], IRNode), type(self.inputs[0])
            device = self.inputs[0].get_device()
            self.outputs += [
                MutationOutput(NoneLayout(device=device), buf, self)
                for buf in mutated_inputs
            ]

    def get_outputs(self) -> list[Buffer]:
        return self.outputs


class NVUniversalGemmBuffer(TemplateBuffer):
    """
    Buffer for NVIDIA Universal GEMM kernels.

    Unlike CuteDSL templates which use Jinja templates, this generates
    simpler Python code that directly calls the cutlass_api library.
    """

    def __init__(
        self,
        layout: Layout,
        inputs: Sequence[IRNode],
        kernel: Any,
        accumulator_type: Any,
        variant: Any,  # GemmVariant, use Any to avoid circular import
        workspace_size: int = 0,
        scale_type_a: Any | None = None,
        scale_type_b: Any | None = None,
        swizzle_type_a: Any | None = None,
        swizzle_type_b: Any | None = None,
    ) -> None:
        # We pass None initially, then override with our method below
        super().__init__(layout, inputs, make_kernel_render=None)
        self.kernel = kernel
        self.accumulator_type = accumulator_type
        self.outputs: list[Buffer] = [self]
        self.workspace_size = workspace_size
        self.variant = variant
        self.scale_type_a = scale_type_a
        self.scale_type_b = scale_type_b
        self.swizzle_type_a = swizzle_type_a
        self.swizzle_type_b = swizzle_type_b
        # Store kernel metadata for code generation since kernels aren't serializeable yet
        self.kernel_metadata = {
            "kernel_name": kernel.metadata.kernel_name,
            "min_cc": kernel.metadata.min_cc,
        }
        # Override the instance attribute set by parent with our method
        # This is necessary because TemplateBuffer stores make_kernel_render as instance attr
        self.make_kernel_render = self._make_kernel_render

    def get_workspace_size(self) -> int:
        """Return the workspace size in bytes."""
        return self.workspace_size

    def get_outputs(self) -> list[Buffer]:
        return self.outputs

    def _make_kernel_render(
        self, out_node: Any, hint_override: int | None = None
    ) -> tuple[Any, Any]:
        """
        Create a kernel renderer for code generation.

        Returns (kernel, render) tuple where:
        - kernel: NVUniversalGemmKernel object with call_kernel() method
        - render: function that returns source code string
        """
        from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_kernel import (
            NVUniversalGemmKernel,
        )
        from torch._inductor.utils import Placeholder

        input_nodes: list[Any] = []
        for inp in self.inputs:
            if isinstance(inp, TensorBox):
                inp = inp.data
            if isinstance(inp, StorageBox):
                inp = inp.data
            input_nodes.append(inp)

        kernel_name = str(Placeholder.KERNEL_NAME)

        render_kernel = NVUniversalGemmKernel(
            kernel_name=kernel_name,
            input_nodes=input_nodes,
            output_node=out_node,
            kernel_metadata=self.kernel_metadata,
            accumulator_type=self.accumulator_type,
            workspace_size=self.workspace_size,
            variant=self.variant,
            scale_type_a=self.scale_type_a,
            scale_type_b=self.scale_type_b,
            swizzle_type_a=self.swizzle_type_a,
            swizzle_type_b=self.swizzle_type_b,
        )

        def render():
            return render_kernel.render()

        return render_kernel, render


def is_node_sequence(
    nodes: Sequence[IRNode | Sequence[IRNode]],
) -> TypeIs[Sequence[IRNode]]:
    return all(isinstance(n, IRNode) for n in nodes)


@ir_dataclass(frozen=False)
class InputsKernel(OperationBuffer):
    inputs: Sequence[IRNode | Sequence[IRNode]]

    def input_name(self, i: int) -> str:
        input = self.inputs[i]
        assert isinstance(input, IRNode)
        return input.get_name()

    def get_read_writes(self) -> dependencies.ReadWrites:
        reads = OrderedSet[dependencies.Dep]()
        StarDep = dependencies.StarDep
        for input in self.inputs:
            if isinstance(input, Sequence):
                reads.update(StarDep(x.get_name()) for x in input)
            elif isinstance(input, ShapeAsConstantBuffer):
                # Skip creating dependency for symbolics as they're visible globally
                continue
            else:
                reads.add(StarDep(input.get_name()))

        writes = OrderedSet[dependencies.Dep](
            StarDep(buf.get_name()) for buf in self.get_outputs()
        )

        return dependencies.ReadWrites(
            reads=reads,
            writes=writes,
            index_exprs=OrderedSet(),
        )

    def get_reads(self) -> OrderedSet[Dep]:
        return self.get_read_writes().reads

    @classmethod
    def unwrap_storage_for_input(cls, x: IRNode) -> IRNode:
        if isinstance(x, TensorBox):
            x = x.data
        if isinstance(x, StorageBox):
            x = x.data
        if isinstance(x, BaseView) and not isinstance(x, ReinterpretView):
            x = ExternKernel.realize_input(x)
        if isinstance(x, TensorBox):
            # when converting to ReinterpretView fails in the
            # realize_input call above, the result will be wrapped
            # into TensorBox / StorageBox pair as a result of the
            # cls.copy_input call; so we should unwrap recursively
            return cls.unwrap_storage_for_input(x)
        if isinstance(x, TorchBindObject):
            return x
        assert isinstance(x, (Buffer, ReinterpretView)), type(x)
        return x

    @staticmethod
    def unwrap_storage(
        inputs: Sequence[IRNode | Sequence[IRNode]],
    ) -> list[IRNode | Sequence[IRNode]]:
        inputs_new: list[IRNode | Sequence[IRNode]] = []
        for x in inputs:
            if isinstance(x, Sequence):
                x = [InputsKernel.unwrap_storage_for_input(i) for i in x]
            else:
                x = InputsKernel.unwrap_storage_for_input(x)
            inputs_new.append(x)
        return inputs_new

    def is_extern(self) -> bool:
        return True

    def num_reads(self) -> int:
        return 1

    @cache_on_self_and_args("InputsKernel")
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        r = OrderedSet[sympy.Symbol]()
        for inp in self.inputs:
            if isinstance(inp, IRNode):
                r |= inp.get_free_symbol_uses(unbacked_only)
            else:
                for inner_inp in inp:
                    r |= inner_inp.get_free_symbol_uses(unbacked_only)
        return r


class NopKernel(InputsKernel):
    def is_no_op(self) -> bool:
        return True

    def get_reads(self) -> OrderedSet[Dep]:
        return OrderedSet()


class ConcatKernel(NopKernel):
    """
    There isn't actually a real kernel for concat, we just change the
    storage for the upstream data.
    """

    @classmethod
    def create(cls, inputs: Sequence[IRNode], dim: int) -> StorageBox:
        """
        Create the concat kernel from inputs
        """
        device = inputs[0].get_device()
        dtype = inputs[0].get_dtype()
        new_size = list(inputs[0].get_size())
        offsets_start = [0]
        offsets_end = [new_size[dim]]
        assert 0 <= dim < len(new_size)
        for i in range(1, len(inputs)):
            input_size = inputs[i].get_size()
            offsets_start.append(new_size[dim])
            assert len(input_size) == len(new_size)
            assert inputs[i].get_dtype() == dtype
            assert inputs[i].get_device() == device
            for j in range(len(new_size)):
                if j == dim:
                    new_size[j] = new_size[j] + input_size[j]
                else:
                    new_size[j] = V.graph.sizevars.check_equals_and_simplify(
                        new_size[j], input_size[j]
                    )
            offsets_end.append(new_size[dim])

        output_stride: Sequence[int] = FlexibleLayout.contiguous_strides(new_size)
        if config.comprehensive_padding:
            # Ensure the output stride matches the alignment requirements
            output_stride = Layout._pad_strides(
                output_stride, new_size, inputs[0].dtype
            )

        # If any of the inputs is in CL format, use CL format for the output
        for i in range(len(inputs)):
            x = inputs[i]
            if is_storage_and_layout(x):
                layout = x.get_layout()
                if isinstance(
                    layout, FixedLayout
                ) and Layout.is_channels_last_contiguous(layout.size, layout.stride):
                    # use CL stride for the output
                    output_stride = make_channels_last_strides_for(new_size)
                    break
        any_input_is_storage_and_layout = any(is_storage_and_layout(x) for x in inputs)
        fx_node_args = V.graph.current_node.args[0]
        # If any of the inputs has meta tensor and the meta tensor is in CL format, use CL format for the output
        # Skip this check when fx_node_args is not a list (e.g., called from _pad_as_cat).
        if (
            any_input_is_storage_and_layout is False
            and isinstance(fx_node_args, list)
            and any(
                # pyrefly: ignore [missing-attribute]
                "val" in arg.meta
                and (
                    # pyrefly: ignore [missing-attribute]
                    arg.meta["val"].is_contiguous(memory_format=torch.channels_last)
                    # pyrefly: ignore [missing-attribute]
                    or arg.meta["val"].is_contiguous(
                        memory_format=torch.channels_last_3d
                    )
                )
                for arg in fx_node_args
            )
        ):
            output_stride = make_channels_last_strides_for(new_size)

        is_pinned = all(
            is_storage_and_layout(x) and x.get_layout().is_pinned for x in inputs
        )

        assert device is not None
        concat_kernel = ConcatKernel(
            name=None,
            layout=FixedLayout(
                device=device,
                dtype=dtype,
                size=new_size,
                stride=output_stride,
                is_pinned=is_pinned,
            ),
            inputs=[],
        )
        kernel = StorageBox(concat_kernel)
        op_names = []
        for i, inp in enumerate(inputs):
            assert isinstance(inp, (BaseView, MutableBox)), type(inp)
            input_buffer = cls.realize_into(
                inp,
                SliceView.create(
                    kernel, dim, offsets_start[i], offsets_end[i], clamp=False
                ),
            )
            assert isinstance(input_buffer, Buffer), type(input_buffer)
            assert isinstance(concat_kernel.inputs, list), type(concat_kernel.inputs)
            concat_kernel.inputs.append(input_buffer)

            if isinstance(inp.data, BaseView):
                input_unwrapped = inp.data.unwrap_view()
            else:
                input_unwrapped = inp.data

            if (
                isinstance(input_unwrapped, StorageBox)
                and input_unwrapped.is_input_buffer()
                and (dev := inp.get_device()) is not None
                and is_gpu(dev.type)
                and not is_dynamic(input_buffer)
            ):
                op_names.append(input_buffer.get_operation_name())

        if len(op_names) > 1 and V.graph.has_feature(device, BackendFeature.FOREACH):
            V.graph.register_operation_list(op_names)

        concat_kernel.name = V.graph.register_buffer(concat_kernel)
        concat_kernel.inputs = cls.unwrap_storage(concat_kernel.inputs)
        V.graph.register_operation(concat_kernel)

        return kernel

    @classmethod
    def can_realize_into_without_copy(
        cls, src: IRNode, dst: IRNode | None = None
    ) -> bool:
        if isinstance(src, TensorBox):
            # unwrap a TensorBox
            return cls.can_realize_into_without_copy(src.data, dst)

        assert isinstance(src, (BaseView, StorageBox)), type(src)
        if isinstance(src.data, MultiTemplateBuffer):
            if (
                not isinstance(src.data.layout, FixedLayout)
                or not src.data.output_plannable
            ):
                return False

            # we call can_realize_into_without_copy in cat lowering before we've decided
            # on output format, optimistically assume layout matches
            if dst is None:
                return True

            # otherwise, check equality of layouts
            if len(src.get_stride()) != len(dst.get_stride()):
                return False

            return all(
                V.graph.sizevars.statically_known_equals(s1, s2)
                for s1, s2 in zip(src.get_stride(), dst.get_stride())
            )

        return (
            hasattr(src.data, "layout")
            and isinstance(src.data.layout, FlexibleLayout)
            and not isinstance(src.data, ExternKernelAlloc)
        )

    @cache_on_self_and_args("ConcatKernel")
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        return NopKernel.get_free_symbol_uses(self, unbacked_only)

    @classmethod
    def realize_into(cls, src: IRNode, dst: IRNode) -> IRNode:
        # Attempt to turn this into a ReinterpretView rather than assert.
        # This has concessions around layout, as as_storage_and_layout
        # can cause us to go from flexible to fixed layout.
        if not isinstance(dst, ReinterpretView):
            if is_storage_and_layout(dst):
                storage, layout = as_storage_and_layout(dst)
                dst = ReinterpretView(data=storage, layout=layout)
        assert isinstance(dst, ReinterpretView), type(dst)
        if isinstance(src, TensorBox):
            # unwrap a TensorBox
            return cls.realize_into(src.data, dst)

        if isinstance(src, StorageBox):
            src.realize()
            # ExternKernelAlloc has specific requirements for output layout, should create a copy
            assert hasattr(src.data, "layout")
            if cls.can_realize_into_without_copy(src, dst):
                # pyrefly: ignore [missing-attribute]
                src.data.layout = NonOwningLayout(dst)
                return src.data
        # introduce a copy
        pw = Pointwise.create(
            device=src.get_device(),
            dtype=src.get_dtype(),
            inner_fn=src.make_loader(),
            ranges=[
                V.graph.sizevars.check_equals_and_simplify(a, b)
                for a, b in zip(src.get_size(), dst.get_size())
            ],
        )
        return cls.realize_into(pw, dst)

    def should_allocate(self) -> bool:
        return True


@ir_dataclass(frozen=False)
class ExternKernel(InputsKernel):
    """
    A class that represents Kernels which are not directly lowered to Inductor
    Loop Level IR, such as custom operators, or aten operators which we fallback to.
    """

    constant_args: Sequence[Any] = ()
    kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    output_view: ReinterpretView | None = None
    python_kernel_name: str | None = None
    cpp_kernel_name: str | None = None
    # FIXME: in some cases we sill need to explicitly pass in ordered_kwargs_for_cpp_kernel
    # We shouldn't need to do this since the information can be retrieved from op_overload._schema.
    ordered_kwargs_for_cpp_kernel: Iterable[str] = dataclasses.field(
        default_factory=list
    )
    op_overload: _OpOverloads | None = None
    arg_properties: list[dict[str, Any]] | None = None
    allarg_properties: dict[str, dict[str, Any]] = dataclasses.field(
        default_factory=dict
    )
    kwarg_properties: dict[str, dict[str, Any]] | None = None
    unbacked_bindings: dict[sympy.Symbol, pytree.KeyPath] = dataclasses.field(
        default_factory=dict
    )
    mutation_outputs: list[MutationOutput] = dataclasses.field(default_factory=list)

    def __init__(
        self,
        name: str | None,
        layout: OutputSpec,
        inputs: Sequence[IRNode | Sequence[IRNode]],
        constant_args: Sequence[Any] = (),
        kwargs: dict[str, Any] | None = None,
        output_view: ReinterpretView | None = None,
        python_kernel_name: str | None = None,
        cpp_kernel_name: str | None = None,
        ordered_kwargs_for_cpp_kernel: Iterable[str] = (),
        op_overload: _OpOverloads | None = None,
    ) -> None:
        super().__init__(
            name=name,
            layout=layout,
            inputs=inputs,
        )
        self.constant_args = constant_args
        self.kwargs = kwargs if kwargs else {}
        self.output_view = output_view
        self.op_overload = op_overload
        self.set_cpp_kernel_name(cpp_kernel_name)
        self.set_python_kernel_name(python_kernel_name)
        self.ordered_kwargs_for_cpp_kernel = ordered_kwargs_for_cpp_kernel
        self.collect_arg_kwarg_properties()
        self.unbacked_bindings = {}
        self.mutation_outputs = []
        self.fx_node = V.graph.current_node
        # Annotations dict for storing metadata (e.g., KernelTemplateChoice)
        self.annotations: dict[str, Any] = {}

    def get_outputs(self) -> list[Buffer]:
        return [self, *self.mutation_outputs]

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()

    def collect_arg_kwarg_properties(self) -> None:
        # if self.op_overload is torch._ops.OpOverload, we can use its schema to collect additional
        # information for args and kwargs, e.g. type and default value, to help with the cpp wrapper codegen
        self.arg_properties = (
            [
                {
                    "name": x.name,
                    "type": x.real_type,
                    "default_value": x.default_value,
                }
                for x in self.op_overload._schema.arguments
                if not x.kwarg_only
            ]
            if isinstance(self.op_overload, torch._ops.OpOverload)
            else [{} for i in range(len(self.inputs))]
        )
        self.allarg_properties = (
            {
                x.name: {"type": x.real_type, "default_value": x.default_value}
                for x in self.op_overload._schema.arguments
            }
            if isinstance(self.op_overload, torch._ops.OpOverload)
            else {}
        )
        # FIXME: self.kwargs does not always match kwargs defined in schema, so sometimes
        # ordered_kwargs_for_cpp_kernel is explicitly passed in.
        if isinstance(self.op_overload, torch._ops.OpOverload):
            if not self.ordered_kwargs_for_cpp_kernel:
                self.ordered_kwargs_for_cpp_kernel = [
                    x.name for x in self.op_overload._schema.arguments if x.kwarg_only
                ]
            self.schema_kwargs = [
                x for x in self.op_overload._schema.arguments if x.kwarg_only
            ]
        else:
            self.schema_kwargs = []

    def decide_layout(self) -> None:
        if isinstance(self.layout, FlexibleLayout):
            self.apply_constraint()
            self.freeze_layout()

    def codegen_comment(
        self, wrapper: PythonWrapperCodegen, kernel_name: str | None = None
    ) -> None:
        origin_str, _detailed_origin_str = get_kernel_metadata(self, wrapper)
        if origin_str:
            wrapper.make_comment(origin_str)

        if not kernel_name:
            kernel_name = self.try_get_kernel_name()
        if kernel_name:
            from .debug import set_kernel_post_grad_provenance_tracing

            debug_handle = set_kernel_post_grad_provenance_tracing(
                self, kernel_name, is_extern=True
            )
            wrapper.write_provenance_debug_handle(kernel_name, debug_handle)

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        raise NotImplementedError

    def set_cpp_kernel_name(self, cpp_kernel_name: str | None = None) -> None:
        self.cpp_kernel_name = cpp_kernel_name
        if not V.graph.cpp_wrapper or not isinstance(
            self.op_overload, torch._ops.OpOverload
        ):
            return

        kernel = self.op_overload
        if self.cpp_kernel_name is None:
            # Try to construct cpp_kernel_name from op_overload
            if kernel.namespace == "aten":
                # Calling with the default kernel name can lead to ambiguous behavior like the following example.
                # repeat_interleave(const at::Tensor & repeats, std::optional<int64_t> output_size=std::nullopt)
                # repeat_interleave(const at::Tensor & self, int64_t repeats,
                #       std::optional<int64_t> dim=std::nullopt, std::optional<int64_t> output_size=std::nullopt)
                opname = (
                    kernel.__name__.split(".")[0]
                    if kernel._overloadname == "default"
                    else kernel.__name__.replace(".", "_")
                )
                self.cpp_kernel_name = f"at::_ops::{opname}::call"
            else:
                self.cpp_kernel_name = kernel._schema.name

    def set_python_kernel_name(self, python_kernel_name: str | None) -> None:
        self.python_kernel_name = python_kernel_name
        if python_kernel_name is not None:
            return

        kernel = self.op_overload
        if kernel is None:
            pass
        elif isinstance(kernel, torch._ops.HigherOrderOperator):
            self.python_kernel_name = f"torch.ops.higher_order.{kernel.__name__}"
        else:
            self.python_kernel_name = (
                f"{kernel.__module__.replace('._ops.', '.ops.')}.{kernel.__name__}"
            )

    def try_get_kernel_name(self) -> str | None:
        from .codegen.cpp_wrapper_cpu import CppWrapperCpu

        device = d.type if (d := self.get_device()) else V.graph.device_type
        if V.graph.fx_wrapper:
            return self.python_kernel_name
        elif V.graph.cpp_wrapper:
            assert isinstance(V.graph.wrapper_code, CppWrapperCpu), type(
                V.graph.wrapper_code
            )
            if self.cpp_kernel_name is None:
                return None
            return V.graph.wrapper_code.get_c_shim_func_name(
                self.cpp_kernel_name, device
            )
        else:
            return self.python_kernel_name

    def get_kernel_name(self) -> str:
        name = self.try_get_kernel_name()
        assert name is not None
        return name

    @staticmethod
    def copy_input(x: IRNode) -> TensorBox:
        pw = Pointwise.create(
            device=x.get_device(),
            dtype=x.get_dtype(),
            inner_fn=x.make_loader(),
            ranges=x.get_size(),
            origin_node=x.get_origin_node(),
            traceback=x.get_traceback(),
        )
        pw.realize()
        return pw

    @classmethod
    def process_kernel(
        cls, kernel: _OpOverloads, *args: Any, **kwargs: Any
    ) -> tuple[
        Any,
        list[Any],
        list[Any],
        Callable[[Any, Any], Any],
        dict[sympy.Symbol, pytree.KeyPath] | None,
    ]:
        """Partition kernel args into tensor and non-tensor, realize tensor inputs,
        re-run fake tensor propagation with the realized strides, and return
        (example_output, tensor_args, non_tensor_args, unflatten_args, unbacked_bindings).

        unflatten_args(new_tensor_args, new_non_tensor_args) reconstructs the
        original (args, kwargs) tree from replacement lists.
        """
        binded_args = {"args": args, "kwargs": kwargs}

        args_flat, args_spec = pytree.tree_flatten(binded_args)

        args_flat_is_tensor: list[bool] = []
        # tensor_args can be either tensor or torchbind objects
        tensor_args: list[IRNode] = []
        non_tensor_args: list[object] = []
        real_non_tensor_args: list[
            FakeScriptObject | torch._C.Generator | torch._C.ScriptObject | torch.Tensor
        ] = []
        for arg in args_flat:
            match arg:
                case Expr():
                    node = V.graph.sizevars.shape_env.create_symintnode(arg, hint=None)
                    args_flat_is_tensor.append(False)
                    non_tensor_args.append(node)
                    real_non_tensor_args.append(node)

                case GeneratorState():
                    args_flat_is_tensor.append(False)
                    non_tensor_args.append(arg)
                    device_index = arg.device.index
                    assert arg.device.type == "cuda" and device_index is not None
                    real_non_tensor_args.append(
                        torch.cuda.default_generators[device_index].clone_state()
                    )

                case OpaqueObjectState():
                    args_flat_is_tensor.append(False)
                    non_tensor_args.append(arg)
                    real_non_tensor_args.append(arg.value)

                case IRNode():
                    args_flat_is_tensor.append(True)
                    tensor_args.append(arg)

                case _:
                    args_flat_is_tensor.append(False)
                    non_tensor_args.append(arg)
                    real_non_tensor_args.append(arg)

        def unflatten_args(
            new_tensor_args: Sequence[_T], new_non_tensor_args: Sequence[_T]
        ) -> tuple[list[_T], dict[str, _T]]:
            result = []
            it_tensors = iter(new_tensor_args)
            it_non_tensors = iter(new_non_tensor_args)
            for is_tensor in args_flat_is_tensor:
                if is_tensor:
                    result.append(next(it_tensors))
                else:
                    result.append(next(it_non_tensors))
            r = pytree.tree_unflatten(result, args_spec)
            return r.get("args", []), r.get("kwargs", {})

        tensor_args = [cls.realize_input(x) for x in tensor_args]

        # freeze layout otherwise our output stride calculation might
        # become incorrect
        for x in tensor_args:
            if is_storage_and_layout(x):
                as_storage_and_layout(x, freeze=True)

        # Rerun fake tensor propagation, because Inductor may have changed the
        # strides of inputs and we need to determine accurately what the
        # output stride will be.
        example_args: list[
            torch.Tensor | torch._C.ScriptObject | FakeScriptObject | torch.Generator
        ] = []

        # We need to retain the constant values of fake tensors that we originally
        # propagated the graph with, because for some operators running without a
        # constant would trigger an error / DataDependentException
        for x in tensor_args:
            # if x is a view of a constant, we need to realize the view
            # (we can't pass the constant into the kernel directly)
            if not isinstance(x, BaseView) and x.get_name() in V.graph.constants:
                example_args.append(V.graph.constants[x.get_name()])
            elif (
                not isinstance(x, BaseView)
                and x.get_name() in V.graph.torchbind_constants
            ):
                example_args.append(V.graph.torchbind_constants[x.get_name()])
            elif isinstance(x, TorchBindObject):
                example_args.append(x.get_value())
            elif isinstance(x, OpaqueMultiOutput):
                example_args.append(x.opaque_example_value)
            elif isinstance(x, torch._inductor.ir.GeneratorState):
                device_index = x.device.index
                assert x.device.type == "cuda" and device_index is not None
                example_args.append(
                    torch.cuda.default_generators[device_index].clone_state()
                )
            else:
                example_args.append(ir_node_to_tensor(x))

        new_args, new_kwargs = unflatten_args(example_args, real_non_tensor_args)
        example_output = kernel(*new_args, **new_kwargs)

        unbacked_bindings: dict[sympy.Symbol, pytree.KeyPath] | None = None
        if shape_env := V.fake_mode.shape_env:
            node_meta_val = V.current_node.meta.get("val")
            ctx: AbstractContextManager[None] = nullcontext()
            if V.current_node.target is torch._higher_order_ops.effects.with_effects:
                # remove the first effect token in meta["val"] and meta["unbacked_bindings"]
                node_meta_val = node_meta_val[1]
                ctx = _remove_effect_token_unbacked_bindings(V.current_node)

            with ctx:
                rebind_unbacked(shape_env, V.current_node, example_output)
            unbacked_bindings = compute_unbacked_bindings(
                shape_env, example_output, node_meta_val
            )

        example_out_li = (
            [example_output]
            if not isinstance(example_output, (list, tuple))
            else example_output
        )
        # When graph_partition is enabled, skip - partitioning handles sparse outputs
        for t in example_out_li:
            if (
                isinstance(t, torch.Tensor)
                and t.is_sparse
                and not config.graph_partition
            ):
                msg = "sparsity not handled. Please file issue for sparse inference weights."
                if stack_trace := V.graph.current_node.meta.get("stack_trace", None):
                    msg = f"{msg} Found from : \n {stack_trace}"
                V.graph.disable_cudagraphs_reason = msg

        return (
            example_output,
            tensor_args,
            non_tensor_args,
            unflatten_args,
            unbacked_bindings,
        )

    @classmethod
    def convert_to_reinterpret_view(cls, x: IRNode) -> ReinterpretView:
        """
        In order to pass this to an extern kernel we need a
        ReinterpretView not a View.  This allows us to avoid some
        unneeded copies.
        """
        assert isinstance(x, BaseView), type(x)
        if isinstance(x, ReinterpretView):
            return x

        # NOTE: Don't use extract_read_writes here as it fails when
        # make_loader() inlines the computation
        x_unwrap_view = x.unwrap_view()
        buf = V.graph.get_buffer(x_unwrap_view.get_name())
        assert buf is not None
        x_unwrap_view_fx_node = buf.get_origin_node()
        # Prefer channels last format according to how the format is set from eager.
        if (
            x_unwrap_view_fx_node is not None
            and "val" in x_unwrap_view_fx_node.meta
            and isinstance(x_unwrap_view, (ReinterpretView, Buffer, MutableBox))
            and isinstance(x_unwrap_view.layout, FlexibleLayout)
            and (
                is_contiguous_for_memory_format_or_false(
                    x_unwrap_view_fx_node.meta["val"],
                    memory_format=torch.channels_last,
                )
                or is_contiguous_for_memory_format_or_false(
                    x_unwrap_view_fx_node.meta["val"],
                    memory_format=torch.channels_last_3d,
                )
            )
        ):
            x_unwrap_view.freeze_layout_with_same_order(
                make_channels_last_strides_for(x_unwrap_view.get_size())
            )
        else:
            x_unwrap_view.freeze_layout()

        index_args, var_ranges = dependencies.index_vars_squeeze(
            x.get_size(), prefix="r"
        )
        range_vars = index_args[0]
        index = x.make_indexer()(range_vars)

        index = V.graph.sizevars.simplify_with_ranges(index, var_ranges)
        strides = V.graph.sizevars.stride_vars(index, range_vars)
        offset = V.graph.sizevars.offset_var(index, range_vars)
        expected = sympy_dot(range_vars, strides) + offset

        if index != expected:
            log.debug(
                "convert_to_reinterpret_view failed: stride=%s offset=%s index=%s",
                strides,
                offset,
                index,
            )
            raise NotImplementedError

        return ReinterpretView(
            data=x.data,
            layout=FixedLayout(
                device=x.get_device_or_error(),
                dtype=x.get_dtype(),
                size=x.get_size(),
                stride=strides,
                offset=offset,
                is_pinned=False,
            ),
        )

    @classmethod
    def realize_input(cls, x: IRNode) -> IRNode:
        if x is None:
            return NoneAsConstantBuffer()
        if isinstance(x, (Expr, sympy.logic.boolalg.Boolean, int)):
            return ShapeAsConstantBuffer(expr=x)
        if isinstance(x, Constant):
            # We need to unset fake mode, or else the torch.tensor() call will
            # turn into a FakeTensor
            with _disable_current_modes():
                return V.graph.add_tensor_constant(
                    torch.tensor(x.value, dtype=x.get_dtype(), device=x.get_device())
                )
        if isinstance(x, ConstantBuffer):
            return x
        if isinstance(x, TensorBox):
            return cls.realize_input(x.data)
        if isinstance(x, ReinterpretView):
            return ReinterpretView(
                data=cls.realize_input(x.data), layout=x.get_layout()
            )
        if isinstance(x, BaseView):
            x.realize()
            if is_storage_and_layout(x.unwrap_view()):
                try:
                    return cls.convert_to_reinterpret_view(x)
                except NotImplementedError:
                    pass
        if isinstance(x, StorageBox):
            # TODO(jansel): impose layout preference on realized buffer
            x.realize()
            return x
        if isinstance(x, (NonTensorObj, ShapeAsConstantBuffer, OpaqueMultiOutput)):
            return x
        return cls.copy_input(x)

    @classmethod
    def require_stride1(cls, x: IRNode) -> IRNode:
        if is_storage_and_layout(x):
            if len(x.get_stride()) == 0:
                return x
            for stride in x.get_stride():
                if stride == 1:
                    return x
        return cls.copy_input(x)

    @classmethod
    def require_strides(
        cls,
        x: IRNode,
        order: Sequence[int] | None = None,
        exact_strides: Sequence[_IntLike] | None = None,
        allow_padding: bool = False,
    ) -> IRNode:
        assert order is not None or exact_strides is not None
        # Layout generally doesn't matter, but some consuming external ops might have requirements
        if x.get_numel() in (0, 1) and not exact_strides:
            return x

        # require x to have the layout
        if is_storage_and_layout(x):
            if isinstance(x.get_layout(), FlexibleLayout):
                if order:
                    # If the FlexibleLayout already has the size and stride in the required order,
                    # freeze it to a FixedLayout by using its current size and stride.
                    # The behavior of using its current size and stride or the given order can be different
                    # if the size and stride has ambiguilty, for example for a 4D input where the iC = 1:
                    # size=[s0, 1, 28, 28], stride=[784, 784, 28, 1]. If the required order is [3, 0, 2, 1] (channels last),
                    # the current size and stride already satisfies this order.
                    # However by freezing it to the required order, the layout will be changed to:
                    # size=[s0, 1, 28, 28], stride=[784, 1, 28, 1]), which is not actually necessary.
                    use_current_stride_order = is_stride_order_storage_and_layout(
                        x, order
                    ) and not free_unbacked_symbols(x.get_layout().stride)
                    # fix flexiblelayout to be FixedLayout with stride_order
                    as_storage_and_layout(
                        x,
                        freeze=True,
                        want_contiguous=False,
                        stride_order=(
                            get_stride_order(
                                V.graph.sizevars.guarding_hints_or_throw(
                                    x.get_layout().stride
                                )
                            )
                            if use_current_stride_order
                            else order
                        ),
                        allow_padding=allow_padding,
                    )
                    return x
                else:
                    # If the exact_strides is given, freeze the FlexibleLayout to a FixedLayout with the exact_strides.
                    as_storage_and_layout(
                        x,
                        freeze=True,
                        want_contiguous=False,
                        stride_order=None,
                        allow_padding=allow_padding,
                        exact_strides=exact_strides,
                    )
                    return x
            elif isinstance(x.get_layout(), (FixedLayout, NonOwningLayout)) and (
                (order and x.get_layout().is_stride_ordered(order))
                or (
                    exact_strides
                    and significant_strides_equal(
                        exact_strides, x.get_layout().stride, x.get_size()
                    )
                )
            ):
                return (
                    try_match_insignificant_strides(x, exact_strides)
                    if exact_strides is not None
                    else x
                )
            elif isinstance(
                (mutation_layout := x.get_layout()), MutationLayoutSHOULDREMOVE
            ):
                if isinstance(
                    (real_layout := mutation_layout.real_layout()), FlexibleLayout
                ):
                    raise AssertionError(
                        "the MutationLayoutSHOULDREMOVE's real layout shouldn't be FlexibleLayout"
                    )
                elif isinstance(real_layout, FixedLayout) and (
                    (order and real_layout.is_stride_ordered(order))
                    or (
                        exact_strides
                        and significant_strides_equal(
                            exact_strides, real_layout.stride, x.get_size()
                        )
                    )
                ):
                    return x

        # TODO - Storage to InputBuffer
        if isinstance(x, InputBuffer) and (
            (order and x.get_layout().is_stride_ordered(order))
            or (
                exact_strides
                and significant_strides_equal(
                    exact_strides, x.get_layout().stride, x.get_size()
                )
            )
        ):
            return x
        if (
            isinstance(x, TensorBox)
            and isinstance(x.data, BaseView)
            and not isinstance(x.data, ReinterpretView)
            and is_storage_and_layout(unwrap_view := x.unwrap_view())
            and hasattr(unwrap_view, "data")
            and not isinstance(unwrap_view.data, ExternKernelAlloc)
        ):
            try:
                x.data = cls.convert_to_reinterpret_view(x.data)
                if order:
                    return cls.require_stride_order(
                        x, order, allow_padding=allow_padding
                    )
                elif exact_strides:
                    return cls.require_exact_strides(
                        x, exact_strides, allow_padding=allow_padding
                    )
            except NotImplementedError:
                pass

        # Preserve ExpandView representation that would be lost during copy_input
        # Without representation of the expand in inductor IR, in codegen we end up
        # launching a grid for the full size tensor and doing redundant computation
        # across expanded dims.
        # TODO: could also be good to have a codegen fix to recognize overlapping elements

        expanded_dims: list[int] | None = None
        orig_size = x.get_size()
        if exact_strides is not None:
            sizevars = V.graph.sizevars
            expanded_dims = [
                i
                for i in range(len(x.get_size()))
                if sizevars.statically_known_equals(exact_strides[i], 0)
                and sizevars.statically_known_geq(x.get_size()[i], 2)
            ]

            for dim in expanded_dims:
                x = torch._inductor.lowering.slice_(x, dim, 0, 1)

        # Although this is a clone, inductor is good about fusing clones into previous
        # operations if they weren't realized and their layouts were flexible.
        x = cls.copy_input(x)

        as_storage_and_layout(
            x,
            freeze=True,
            want_contiguous=False,
            stride_order=order,
            allow_padding=allow_padding,
            exact_strides=exact_strides,
        )
        if order:
            assert is_stride_order_storage_and_layout(x, order)
        elif expanded_dims:
            assert orig_size is not None and exact_strides is not None
            x = torch._inductor.lowering.expand(x, orig_size)
            # the expand will sometimes may change insignificant strides, so match them back
            return try_match_insignificant_strides(x, exact_strides)

        return x

    @classmethod
    def require_exact_strides(
        cls, x: IRNode, exact_strides: Sequence[_IntLike], allow_padding: bool = False
    ) -> IRNode:
        return cls.require_strides(
            x,
            exact_strides=[
                s.node.expr if isinstance(s, torch.SymInt) else s for s in exact_strides
            ],
            allow_padding=allow_padding,
        )

    @classmethod
    def require_stride_order(
        cls, x: IRNode, order: Sequence[int], allow_padding: bool = False
    ) -> IRNode:
        return cls.require_strides(x, order=order, allow_padding=allow_padding)

    @classmethod
    def require_channels_last(cls, x: IRNode) -> IRNode:
        return cls.require_stride_order(x, NHWC_STRIDE_ORDER)

    @classmethod
    def require_channels_last_3d(cls, x: IRNode) -> IRNode:
        return cls.require_stride_order(x, NHWDC_STRIDE_ORDER)

    @classmethod
    def require_contiguous(cls, x: IRNode) -> IRNode:
        def is_mkldnn_tensor(x: IRNode) -> bool:
            try:
                name = x.get_name()
            except (AttributeError, NotImplementedError):
                return False

            return name in V.graph.constants and V.graph.constants[name].is_mkldnn

        # TODO move this to the more proper places
        if is_mkldnn_tensor(x):
            return x
        else:
            return cls.require_exact_strides(
                x, FlexibleLayout.contiguous_strides(x.get_size())
            )

    @classmethod
    def require_contiguous_strides(cls, x: IRNode) -> IRNode:
        # TODO: combine this with require_contiguous after
        # https://github.com/pytorch/pytorch/pull/148235 lands.
        return cls.require_exact_strides(
            x, FlexibleLayout.contiguous_strides(x.get_size())
        )

    def apply_constraint(self) -> None:
        pass

    def fill_non_provided_args(
        self, args: Sequence[Any], kwargs: dict[str, Any]
    ) -> Sequence[Any]:
        # Previously, we want to maintain forward-compatibility by skipping
        # default args in the serialized artifacts in fbcode. However,
        # some of our shim interfaces require default values being OrderedSet.
        # Discussed with Sherlock offline and we decided to allow serializing
        # default args into the C++ wrapper code for now. We will refine this
        # part if we see real FC requirement. More details related to FC
        # can be found at:
        # https://docs.google.com/document/d/1FzWm-sHYwmRi3x_g036kOxd99KaYquUsA-L5JwOn8ys/edit?usp=sharing
        assert isinstance(args, Sequence), type(args)
        if not isinstance(args, list):
            args = list(args)
        assert self.arg_properties, "ExternKernel.arg_properties should not be empty"

        n_args = len(args)
        n_pos_args = len(self.arg_properties)
        # For cpp wrapper, if some positional args are not provided, we need to check
        # if they're in the kwargs or use their default value
        if n_args < n_pos_args:
            log.debug(
                "%s has %d unprovided positional arguments. "
                "Will check if they are in the keyword arguments or will use default values.",
                self.op_overload,
                n_pos_args - n_args,
            )
            for i in range(n_args, n_pos_args):
                arg_name = self.arg_properties[i]["name"]
                args.append(
                    kwargs[arg_name]
                    if arg_name in kwargs
                    else self.arg_properties[i]["default_value"]
                )
        return args

    def codegen_const_args(self, names: list[str] | None = None) -> list[str]:
        if V.graph.cpp_wrapper:
            result = []
            # Aten ops follow the convention that tensor args are before non-tensor args,
            # in which case the following 'len(self.inputs) + i' logic works. But this
            # may not be true for other ops, and if that is the case, caller needs to
            # pass in a list of const arg names for arg_properties lookup.
            name_to_arg_properties = None
            if names and self.arg_properties:
                assert len(self.constant_args) == len(names), (
                    "names passed to codegen_const_args does not match self.constant_args"
                )
                name_to_arg_properties = {
                    arg.get("name"): arg for arg in self.arg_properties
                }

            for i, x in enumerate(self.constant_args):
                if name_to_arg_properties is not None:
                    assert names is not None
                    prop = name_to_arg_properties.get(names[i])
                    type_ = prop.get("type") if prop else None
                else:
                    idx = len(self.inputs) + i
                    type_ = (
                        self.arg_properties[idx].get("type")
                        if self.arg_properties and idx < len(self.arg_properties)
                        else None
                    )
                result.append(V.graph.wrapper_code.val_to_arg_str(x, type_))
            return result
        else:
            return [V.graph.wrapper_code.val_to_arg_str(a) for a in self.constant_args]

    def codegen_args(self) -> list[str]:
        if V.graph.cpp_wrapper and self.op_overload is not None:
            # cpp wrapper needs special logic to fill in missing args with default values
            inputs = self.fill_non_provided_args(
                [*self.inputs, *self.constant_args], self.kwargs
            )
            # fill_non_provided_args has handled constant args, so no need to codegen for that later
            need_codegen_constant_args = False
        else:
            inputs = self.inputs
            need_codegen_constant_args = True

        args = []
        for i, x in enumerate(inputs):
            if V.graph.cpp_wrapper:
                assert self.arg_properties and i < len(self.arg_properties), (
                    "Invalid access to ExternKernel.arg_properties"
                )
                type_ = self.arg_properties[i].get("type")
                args.append(V.graph.wrapper_code.val_to_arg_str(x, type_))
            else:
                args.append(V.graph.wrapper_code.val_to_arg_str(x))
        if need_codegen_constant_args:
            args.extend(self.codegen_const_args())
        return args

    def get_kwargs_value(self, arg_name: str, **kwargs: Any) -> Any:
        """Given an argument name, queries for values in (in order):
        1. any provided kwargs for this function.
        2. the class self.kwargs member.
        3. any available default arguments in self.allarg_properties."""
        if arg_name in kwargs:
            return kwargs.get(arg_name)
        if arg_name in self.kwargs:
            return self.kwargs.get(arg_name)
        if (arg := self.allarg_properties.get(arg_name)) is not None:
            return arg.get("default_value")
        raise AssertionError(f"{arg_name} not in self.allarg_properties")

    def codegen_kwargs(self, skip_out: bool = False) -> list[str]:
        if V.graph.cpp_wrapper:
            if self.op_overload is not None and len(self.schema_kwargs) == 0:
                # All the args should have been generated by fill_non_provided_args in codegen_args
                return []

            kwargs = []
            for arg_name in self.ordered_kwargs_for_cpp_kernel:
                if skip_out and arg_name == "out":
                    # ExternKernelOut has its own logic for inserting the out parameter
                    continue

                v = self.get_kwargs_value(arg_name)
                if isinstance(v, Expr):
                    kwargs.append(v)
                else:
                    assert self.allarg_properties is not None
                    type_ = self.allarg_properties.get(arg_name, {}).get("type")
                    kwargs.append(V.graph.wrapper_code.val_to_arg_str(v, type_))
        else:
            kwargs = [
                f"{k}={V.graph.wrapper_code.val_to_arg_str(v)}"
                for k, v in self.kwargs.items()
            ]
        return kwargs

    def get_op_name(self) -> str:
        if self.fx_node is not None:
            target = self.fx_node.target
            op_namespace = getattr(target, "__module__", "unknown_namespace")
            op_namespace = op_namespace.replace("._ops.", ".ops.")
            op_namespace = op_namespace.rsplit(".", 1)[0]
            op_name = f"{op_namespace}.{target}"
        else:
            op_name = "unknown_op"
        return op_name

    def codegen_size_asserts(self, wrapper: PythonWrapperCodegen) -> None:
        if config.size_asserts and not V.graph.cpp_wrapper:
            # comparing strides for 0 size tensor is tricky. Ignore them for now.
            if sympy_product(self.get_size()) == 0:
                return
            size = V.graph.wrapper_code.codegen_shape_tuple(self.get_size())
            stride = V.graph.wrapper_code.codegen_shape_tuple(self.get_stride())
            op_name = self.get_op_name()
            wrapper.writeline(
                f"assert_size_stride({self.get_name()}, {size}, {stride}, {op_name!r})"
            )

    def codegen_alignment_asserts(self, wrapper: PythonWrapperCodegen) -> None:
        if config.alignment_asserts and not V.graph.cpp_wrapper:
            name = self.get_name()
            aligned = name not in V.graph.unaligned_buffers
            op_name = self.get_op_name()
            if aligned:
                wrapper.writeline(
                    f"assert_alignment({name}, {GPU_ALIGN_BYTES}, {op_name!r})"
                )
            else:
                wrapper.writeline(
                    f"# buffer {name} (op: {op_name}) is assumed to be not aligned"
                )

    def codegen_memory_tracking(self, wrapper: PythonWrapperCodegen) -> None:
        """
        Track outputs of fallback operators if config.test_configs.track_memory_lifecycle
        """
        if not config.test_configs.track_memory_lifecycle or V.graph.cpp_wrapper:
            return

        wrapper.write_memory_track_allocation_once()
        name = self.get_name()
        wrapper.writeline(f"track_tensor({name}, '{name}')")

    def get_group_stride(self) -> tuple[list[Sequence[Expr]], list[Expr]]:
        """
        get output sizes and strides, for template_codegen
        """
        _size = self.get_size()
        _stride = self.get_stride()
        # iter_ranges = _size of output tensor, reduce_range = [] because no reduction
        return [_size, []], _stride

    def canonicalize(self) -> tuple[Expr, Sequence[Expr]]:
        """
        Manually get canonicalization of the output index
        """
        # manually generate index formula for conv
        sizevars = V.graph.sizevars
        sizes = self.get_size()
        strides = self.get_stride()
        # Stride hints are only used as sort keys to determine dimension
        # ordering for canonicalization, not for correctness.
        strides = [sizevars.optimization_hint(x) for x in strides]
        # TODO: I can't tell if the symbols here are temporary
        index_vars = [sympy_index_symbol(f"d{i}") for i in range(len(sizes))]
        # reorder index vars according to stride
        index_order = sorted(range(len(strides)), key=strides.__getitem__, reverse=True)
        lookup = {pos: idx for idx, pos in enumerate(index_order)}
        order = [lookup[i] for i in range(len(lookup))]
        index_vars = [index_vars[i] for i in order]
        indexer = self.make_indexer()
        index = indexer(index_vars)

        new_sizes, reindex, _prune = V.graph.sizevars._simplify_loops(
            index_vars, sizes, [index]
        )

        # assign new variables each dimension to deal with numbering mismatches
        # d0, d1, d2 could become d0, d2 -- which won't match d0, d1
        _, add_var = var_builder("c")
        replacement = dict(zip(index_vars, reindex([add_var(x) for x in new_sizes])))

        index = sympy_subs(sympy.expand(index), replacement)
        return index, tuple(new_sizes)

    @cache_on_self_and_args("ExternKernel")
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        # NB: It's not necessary to check regular inputs as we automatically
        # have dependencies on them
        maybe_get_symbols = (
            maybe_free_unbacked_symbols if unbacked_only else maybe_free_symbols
        )
        r = InputsKernel.get_free_symbol_uses(self, unbacked_only)
        for arg in self.constant_args:
            r |= maybe_get_symbols(arg)
        for arg in self.kwargs.values():
            r |= maybe_get_symbols(arg)
        return r

    def __str__(self) -> str:
        kernel_name = getattr(self, "python_kernel_name", None)
        lines = [
            f"python_kernel_name={kernel_name!r}",
        ]
        lines += [
            f"{field.name}={getattr(self, field.name)}"
            for field in dataclasses.fields(self)
        ]
        lines.append(f"origin_node={self.origin_node!r}")
        return self.str_helper(lines)

    __repr__ = __str__


@ir_dataclass(frozen=False)
class ExternKernelOut(ExternKernel):
    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        wrapper.generate_extern_kernel_out(self)

    def __init__(
        self,
        layout: Layout,
        inputs: Sequence[IRNode],
        constant_args: Sequence[Any] = (),
        kwargs: dict[str, Any] | None = None,
        output_view: ReinterpretView | None = None,
        python_kernel_name: str | None = None,
        cpp_kernel_name: str | None = None,
        ordered_kwargs_for_cpp_kernel: Sequence[Any] = (),
        op_overload: _OpOverloads | None = None,
    ) -> None:
        unwrapped_inputs = self.unwrap_storage(inputs)
        assert isinstance(unwrapped_inputs, Sequence), type(unwrapped_inputs)
        super().__init__(
            None,
            layout,
            unwrapped_inputs,
            constant_args,
            kwargs or {},
            None,
            python_kernel_name,
            cpp_kernel_name,
            ordered_kwargs_for_cpp_kernel,
            op_overload,
        )
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

    def should_allocate(self) -> bool:
        return True


class RandomSeeds(ExternKernelOut):
    def __init__(self, count: int, device: torch.device) -> None:
        limits = torch.iinfo(torch.int64)
        super().__init__(
            layout=FixedLayout(
                device=device,
                dtype=torch.int64,
                size=[count],
            ),
            inputs=[],
            constant_args=[limits.min, limits.max, [count]],
            python_kernel_name="aten.randint.low_out",
            # FIXME: Ideally we should only use at::_ops::randint_low_out::call here,
            # but the signature is different from is at::randint_out. Again,
            # we can simplify the code when only keeping an ABI-compatible version.
            cpp_kernel_name="at::_ops::randint_low_out::call",
            op_overload=aten.randint.low_out,
        )


class ExternKernelAlloc(ExternKernel):
    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        wrapper.generate_extern_kernel_alloc(self)

    def __init__(
        self,
        layout: OutputSpec,
        inputs: Sequence[IRNode],
        constant_args: Sequence[Any] = (),
        kwargs: dict[str, Any] | None = None,
        python_kernel_name: str | None = None,
        cpp_kernel_name: str | None = None,
        ordered_kwargs_for_cpp_kernel: Sequence[Any] = (),
        op_overload: _OpOverloads | None = None,
    ) -> None:
        unwrapped_inputs = self.unwrap_storage(inputs)
        assert all(isinstance(i, IRNode) for i in unwrapped_inputs)
        super().__init__(
            None,
            layout,
            cast(Sequence[IRNode], unwrapped_inputs),
            constant_args,
            kwargs or {},
            None,
            python_kernel_name,
            cpp_kernel_name,
            ordered_kwargs_for_cpp_kernel,
            op_overload,
        )
        # We need output buffers for generating kernel arguments in the
        # abi-compatible mode, where we retrieve outputs by pass each individual
        # output through the abi-compatible interface.
        self.outputs: Sequence[Any] = []
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

    def should_allocate(self) -> bool:
        return False

    def apply_constraint(self) -> None:
        raise NotImplementedError


class MutationOutput(Buffer):
    """
    An output buffer that represents the mutation of a pre-existing buffer
    """

    def __init__(
        self, layout: OutputSpec, mutated_node: IRNode, mutating_node: Operation
    ) -> None:
        super().__init__(name=None, layout=layout)
        mutated_node_name = mutated_node.get_name()
        V.graph.mark_buffer_mutated(mutated_node_name)
        self.mutation_names = [mutated_node_name]
        self.mutating_node: Operation = mutating_node
        self.name = V.graph.register_buffer(self)

    def get_defining_op(self) -> Operation:
        return self.mutating_node

    def get_mutation_names(self) -> Sequence[str]:
        return self.mutation_names

    def should_allocate(self) -> bool:
        return False

    def get_mutation_buffers(self) -> Sequence[IRNode]:
        mutation_names = self.get_mutation_names()
        return [
            buf
            for buf in (V.graph.try_get_buffer(name) for name in mutation_names)
            if buf is not None
        ]


class TMADescriptor(ExternKernel):
    """
    An IR node representing a generic host-side TMA descriptor in the Triton API
    Mostly useful for user-defined Triton kernels relying on host-side TMA;
    but can, in principle, be used for Inductor's Triton templates, too.

    See TMADescriptorExperimental and TMADescriptorStable for the two implementations
    (the old API and the new API)
    """

    # as TMA descriptors are immutable,
    # we can dedup them by the input args
    _CACHE: dict[Any, TMADescriptor] = {}

    @classmethod
    def _create_impl(
        cls, tensor: IRNode, tma_meta: tuple[str, tuple[Any, ...]]
    ) -> TMADescriptor:
        assert len(tma_meta) == 2
        if tma_meta[0] == "experimental":
            return TMADescriptorExperimental(tensor, *tma_meta[1])
        else:
            assert tma_meta[0] == "stable"
            return TMADescriptorStable(tensor, *tma_meta[1])

    @classmethod
    def create(
        cls, tensor: IRNode, tma_meta: tuple[str, tuple[Any, ...]]
    ) -> TMADescriptor:
        key = (id(tensor), tma_meta)
        if key not in cls._CACHE:
            cls._CACHE[key] = cls._create_impl(tensor, tma_meta)
        return cls._CACHE[key]

    def __init__(
        self, tensor: IRNode, inputs: Sequence[Any], constant_args: Sequence[Any]
    ) -> None:
        super().__init__(
            None,
            # link back to the underlying tensor in terms of ownership
            # to avoid getting the underlying tensor deleted *before*
            # the TMADescriptor node can be deleted.
            NonOwningLayout(
                ReinterpretView(
                    data=tensor,
                    layout=tensor.get_layout(),
                )
            ),
            cast(Sequence[Buffer], inputs),
            tuple(constant_args),
            None,
        )

        self.tensor = tensor
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        wrapper.generate_tma_descriptor(self)

    def get_tensor(self) -> IRNode:
        return self.tensor


class TMADescriptorExperimental(TMADescriptor):
    """
    the new host-side TMA Descriptor API:
    (the ones obtained via create_{1d,2d}_tma_descriptor calls).

    See also TMADescriptorStable for the new API.
    """

    def __init__(
        self,
        tensor: IRNode,
        dims: list[int | torch.SymInt],
        block_dims: list[int | torch.SymInt],
        element_size: int | None = None,
    ) -> None:
        assert len(dims) in (1, 2)
        assert len(dims) == len(block_dims)

        if element_size is None:
            element_size = tensor.get_dtype().itemsize

        self.dims = dims
        self.block_dims = block_dims
        self.element_size = element_size
        self.rank = len(self.dims)

        inputs = [tensor]
        constant_args = [
            *self.dims,
            *self.block_dims,
            self.element_size,
        ]

        super().__init__(
            tensor=tensor,
            inputs=inputs,
            constant_args=constant_args,
        )


class TMADescriptorStable(TMADescriptor):
    """
    the new host-side TMA descriptor API
    (the ones obtained via TensorDescriptor.from_tensor).

    See also TMADescriptorExperimental for the old API.
    """

    def __init__(self, tensor: IRNode, block_shape: list[int | torch.SymInt]):
        self.block_shape = block_shape

        super().__init__(
            tensor=tensor,
            inputs=[tensor],
            constant_args=block_shape,
        )


class SubgraphBuffer(ExternKernel):
    def __init__(
        self,
        layout: Layout,
        input_nodes: list[Buffer],
        gm: torch.fx.GraphModule,
        example_inputs: list[Any],
        subgraph_name: str,
        config_patches: dict[str, Any] | None = None,
    ):
        super().__init__(None, layout, input_nodes)
        self.gm = gm
        self.example_inputs = example_inputs
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

        self.subgraph = V.graph.make_subgraph(self.gm, example_inputs, subgraph_name)

        assert is_node_sequence(self.inputs)
        sym_inputs = get_symbolic_inputs(self.inputs)

        for sym_inp in sym_inputs:
            self.subgraph.graph_inputs[sym_inp.name] = sym_inp
            self.subgraph.graph_input_names.append(sym_inp.name)

        self.sym_inputs = [sym_var.name for sym_var in sym_inputs]

        import torch._inductor.config as inductor_config

        with V.set_graph_handler(self.subgraph):
            # Base config: don't autotune Triton, but allow other optimizations
            base_patches = {
                "max_autotune": False,
                "max_autotune_gemm": False,
                "max_autotune_gemm_backends": "ATEN",
            }
            # Merge with user config_patches (e.g., coordinate_descent_tuning)
            merged_patches: dict[str, Any] = {**base_patches, **(config_patches or {})}
            with inductor_config.patch(merged_patches):
                self.subgraph.run(*self.example_inputs)

            # Tag all operations in subgraph with config_patches
            # These will be applied during kernel codegen via SIMD scheduling
            if config_patches:
                for op in self.subgraph.operations:
                    op.set_config_patches(config_patches.copy())

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        class CodegenGraph:
            def __init__(self, graph: GraphLowering):
                self.graph = graph
                self.name = graph.name

        assert is_node_sequence(self.inputs)
        outer_inputs = [t.codegen_reference() for t in self.inputs]

        wrapper.codegen_subgraph_with_flattened_outputs(
            CodegenGraph(self.subgraph),
            [*self.sym_inputs, *outer_inputs],
            [self.name],
        )


class UserDefinedTritonKernel(ExternKernel):
    """
    A user-defined triton kernel (e.g. via @triton.jit).
    """

    def get_kernel_and_metadata(self) -> tuple[Kernel, Any, list[str], list[str]]:
        from triton.runtime.autotuner import Autotuner

        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

        kernel = kernel_side_table.get_kernel(self.kernel_idx)
        configs = []
        restore_value_args: list[str] = []
        reset_to_zero_args: list[str] = []
        if isinstance(kernel, Autotuner):
            # https://github.com/triton-lang/triton/pull/5083
            # changes kernel.restore_idx to kernel.restore_value
            if hasattr(kernel, "restore_idx"):
                restore_value_args.extend(
                    kernel.fn.arg_names[i] for i in kernel.restore_idx
                )
            else:
                assert hasattr(kernel, "restore_value")
                restore_value_args.extend(kernel.restore_value)

            if hasattr(kernel, "reset_idx"):
                for i in kernel.reset_idx:
                    reset_to_zero_args.append(kernel.fn.arg_names[i])
            else:
                assert hasattr(kernel, "reset_to_zero")
                reset_to_zero_args.extend(kernel.reset_to_zero)

            configs = kernel.configs
            kernel = kernel.fn

        return kernel, configs, restore_value_args, reset_to_zero_args

    def can_fuse_epilogue(self) -> bool:
        """
        For kernels like

        @triton.jit
        def add_kernel(in_ptr0, in_ptr1, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < n_elements
            x = tl.load(in_ptr0 + offs, mask=mask)
            y = tl.load(in_ptr1 + offs, mask=mask)
            tl.store(out_ptr + offs, x + y, mask=mask)

        @torch.compile
        def fn(a, b):
            out = torch.empty_like(a)
            grid = (triton.cdiv(a.numel(), 1024),)
            add_kernel[grid](a, b, out, a.numel(), BLOCK_SIZE=1024)
            return out.relu()

        We can potentially fuse the relu epilogue into the add_kernel.
        We do this by pruning the `out` tensor allocation and directly writing the relu-output.
        """

        if not config.epilogue_fusion_user_defined_triton_kernel:
            return False

        if not self.arg_accesses.can_fuse_epilogue:
            return False

        # We achieve fusion by parsing the original src into a python AST,
        # then identify the expr containing the original value written via tl.store().
        # We generate an expr for the value after the epilogue and replace that into the tl.store.
        # So far we only support the simple case where there is a single tl.store in the kernel.
        if len(self.kernel_stores.stores) != 1:
            return False

        # Only fuse if the mutated arg is originally an "empty" tensor.
        # This is because we don't know exactly which element of that tensor is being written to.
        # If the kernel only writes to a subset of the tensor, then we only apply the epilogue to that subset.
        # In these edge cases, our fusion is only correct if the original tensor is empty,
        # where the semantics is that content values are UB, and we can rely on the fact that `epilogue(UB) == UB`.
        assert len(self.mutable_args) == 1
        if not isinstance(self.mutable_args[0], TensorBox):
            return False
        if not isinstance(self.mutable_args[0].data, StorageBox):
            return False
        if not isinstance(self.mutable_args[0].data.data, ComputedBuffer):
            return False
        if not isinstance(self.mutable_args[0].data.data.data, Pointwise):
            return False
        if not all(r == 0 for r in self.mutable_args[0].data.data.data.ranges):
            return False

        return True

    @override
    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        return self._codegen(wrapper, epilogue_fusion=None)

    def codegen_with_epilogue_fusion(
        self, wrapper: PythonWrapperCodegen, epilogue_fusion: tuple[ComputedBuffer, str]
    ) -> None:
        """
        epilogue_fusion: (fused epilogue node, modified kerel src code)
        """
        return self._codegen(wrapper, epilogue_fusion)

    def _codegen(
        self,
        wrapper: PythonWrapperCodegen,
        epilogue_fusion: tuple[ComputedBuffer, str] | None,
    ) -> None:
        """Overrides the parent member.
        See https://github.com/pytorch/pytorch/issues/151692"""

        from torch._inductor.utils import triton_version_uses_attrs_dict

        (
            kernel,
            configs,
            restore_value_args,
            reset_to_zero_args,
        ) = self.get_kernel_and_metadata()

        # Definition of kernel
        (
            new_name,
            triton_meta,
            inductor_meta,
            extra_launch_args,
        ) = wrapper.define_user_defined_triton_kernel(
            kernel,
            configs,
            self.kwargs,
            restore_value_args,
            reset_to_zero_args,
            self.grid,
            epilogue_fusion,
        )
        named_args = {
            k: self.get_kwargs_value(k) for k in self.ordered_kwargs_for_cpp_kernel
        }

        if epilogue_fusion:
            assert len(self.arg_accesses.read_writes.writes) == 1
            mutable_arg_name = next(iter(self.arg_accesses.read_writes.writes)).name
            assert mutable_arg_name in named_args
            epilogue_computed_buffer, _ = epilogue_fusion
            named_args[mutable_arg_name] = epilogue_computed_buffer

        arg_names = [p.name for p in kernel.params]  # type: ignore[attr-defined]
        constexprs = [p.num for p in kernel.params if p.is_constexpr]  # type: ignore[attr-defined]
        constexpr_names = OrderedSet(arg_names[i] for i in constexprs)

        args: list[Any] = []
        arg_types: list[Any] = []
        raw_keys_filtered: list[Any] = []
        raw_args_filtered: list[Any] = []
        for name, arg in itertools.chain(
            named_args.items(), zip(itertools.repeat(""), extra_launch_args)
        ):
            if name in constexpr_names and triton_version_uses_attrs_dict():
                # see #160000 - we don't pass in constexpr args to speed up runtime.
                continue
            raw_keys_filtered.append(name)
            raw_args_filtered.append(arg)
            if isinstance(arg, IRNode):
                args.append(arg.codegen_reference())
                arg_types.append(arg.get_dtype())
            elif isinstance(arg, (int, float, bool, sympy.Expr)):
                args.append(arg)
                arg_types.append(type(arg))
            elif name in constexpr_names:
                # insert a dummy value for constexpr args of unsupported type
                # constexprs will end up getting baked into the kernel at compile time
                args.append(-1)
                arg_types.append(int)
            elif arg is None:
                """
                Filter out None args.

                see https://github.com/pytorch/pytorch/issues/115344

                Two cases for a None arg:
                1. The arg is already tl.constexpr, so leave it in
                2. The arg is not tl.constexpr so we have to remove it
                """
                if triton_version_uses_attrs_dict():
                    args.append(-1)
                    arg_types.append(int)
                else:
                    raw_keys_filtered.pop()
                    raw_args_filtered.pop()
            else:
                raise NotImplementedError(f"Unsupported arg type: {type(arg)}: {arg}")

        self.codegen_comment(wrapper, new_name)
        wrapper.generate_kernel_call(
            new_name,
            args,
            arg_types=arg_types,
            raw_args=raw_args_filtered,
            raw_keys=raw_keys_filtered,
            triton_meta=triton_meta,
            inductor_meta=inductor_meta,
            triton=True,
            device=self.get_device(),
            original_fxnode_name=self.fx_node.name,
        )

    @cache_on_self_and_args("UserDefinedTritonKernel")
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        # add unbacked symbols used in the grid to the ones used
        # in the kwargs (the latter is generated by ExternKernel)
        return super().get_free_symbol_uses(unbacked_only) | get_free_symbols(
            self.grid, unbacked_only
        )

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()

    def __init__(
        self,
        *,
        kernel_idx: int,
        grid: Any,
        tma_descriptor_metadata: dict[str, Any],
        kernel_args: dict[str, Any],
    ) -> None:
        inputs: list[IRNode] = []
        kwargs: dict[str, IRNode] = {}
        constant_args: list[IRNode] = []

        for k, v in kernel_args.items():
            if isinstance(v, TensorBox):
                t = InputsKernel.unwrap_storage_for_input(self.realize_input(v))
                if k in tma_descriptor_metadata:
                    t = TMADescriptor.create(t, tma_descriptor_metadata[k])
                inputs.append(t)
                kwargs[k] = t
            else:
                constant_args.append(v)
                kwargs[k] = v

        assert len(inputs) != 0
        self.device = inputs[0].get_device()

        assert isinstance(inputs, Sequence), type(inputs)
        super().__init__(
            None,
            NoneLayout(device=self.device),
            inputs,
            tuple(constant_args),
            kwargs,
        )
        self.kernel_idx = kernel_idx
        self.grid = grid

        kernel, configs, _, _ = self.get_kernel_and_metadata()

        # If we are autotuning, not all arguments will be passed
        assert hasattr(kernel, "arg_names")
        self.ordered_kwargs_for_cpp_kernel = [
            arg for arg in kernel.arg_names if arg in kernel_args
        ]

        from torch._higher_order_ops.triton_kernel_wrap import (
            identify_accessed_tensors,
            identify_triton_stores,
        )

        autotuned_kwargs = configs[0].kwargs if len(configs) > 0 else {}

        import ast

        # pyrefly: ignore [missing-attribute]
        self.kernel_src = kernel.src
        self.kernel_ast = ast.parse(self.kernel_src)
        self.kernel_stores = identify_triton_stores(self.kernel_ast)
        self.kernel_args = kernel_args
        # names in `arg_accesses.read_writes` are names of formal arguments in the kernel's prototype
        self.arg_accesses = identify_accessed_tensors(
            kernel,
            {**kernel_args, **autotuned_kwargs},
            tma_descriptor_metadata,
        )

        # Filter to only tensor args: with Triton 3.7+, ordered_arg_names
        # includes scalars, so writes may reference non-tensor args like SymInts.
        self.mutable_args = [
            kernel_args[key.name]
            for key in self.arg_accesses.read_writes.writes
            if isinstance(kernel_args.get(key.name), TensorBox)
        ]

        self.mutation_outputs = [
            MutationOutput(NoneLayout(device=self.device), buf, self)
            for buf in self.mutable_args
        ]
        V.graph.register_operation(self)

    @override
    def get_read_writes(self) -> dependencies.ReadWrites:
        # Limit the new `get_read_writes` to `epilogue_fusion_user_defined_triton_kernel`
        # to avoid potential regression to existing models.
        if not config.epilogue_fusion_user_defined_triton_kernel:
            return super().get_read_writes()

        # maps formal arg name to actual arg name
        read_renames = {
            formal_arg_dep.name: self.kernel_args[formal_arg_dep.name].get_name()
            for formal_arg_dep in self.arg_accesses.read_writes.reads
        }

        formal_arg_writes = list(self.arg_accesses.read_writes.writes)
        write_renames = {
            formal_arg_dep.name: mut_output.get_name()
            for formal_arg_dep, mut_output in zip(
                formal_arg_writes, self.mutation_outputs
            )
        }

        read_writes = dependencies.ReadWrites(
            reads=OrderedSet(
                [
                    dep.rename(read_renames)
                    for dep in self.arg_accesses.read_writes.reads
                ]
            ),
            writes=OrderedSet(
                [
                    dep.rename(write_renames)
                    for dep in self.arg_accesses.read_writes.writes
                ]
            ),
            index_exprs=OrderedSet(),
        )
        return read_writes

    def get_outputs(self) -> list[Buffer]:
        return list(self.mutation_outputs)

    def get_device(self) -> torch.device | None:
        return self.device


class InplaceBernoulliFallback(ExternKernel):
    """
    This needs to be a custom class to handle mutation properly
    """

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        assert all(isinstance(t, IRNode) for t in self.inputs)
        (x,) = (cast(IRNode, t).codegen_reference() for t in self.inputs)

        if V.graph.cpp_wrapper:
            # Inductor doesn't really support aten Generator, so the Generator kwarg is always NULL here,
            # which needs to be explicitly generated for cpp wrapper
            wrapper.writeline(
                f"{self.get_kernel_name()}({x}, {', '.join(map(repr, self.constant_args))}, NULL){wrapper.ending}"
            )
        else:
            wrapper.writeline(
                f"{self.get_kernel_name()}({x}, {', '.join(map(repr, self.constant_args))}){wrapper.ending}"
            )

    def should_allocate(self) -> bool:
        return False

    def get_mutation_names(self) -> Sequence[str]:
        return [self.input_name(0)]

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()

    def __init__(
        self, op_overload: _OpOverloads, x: IRNode, *constant_args: Any
    ) -> None:
        super().__init__(
            None,
            NoneLayout(device=x.get_device()),
            self.unwrap_storage([x]),
            constant_args,
            op_overload=op_overload,
        )
        V.graph.mark_buffer_mutated(x.get_name())
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)


# Used to deal with torch.complex types
class InplaceCopyFallback(ExternKernel):
    """
    This needs to be a custom class to handle mutation properly
    """

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        (dst, src, non_blocking) = self.codegen_args()
        wrapper.codegen_device_copy(src, dst, non_blocking)

    def should_allocate(self) -> bool:
        return False

    def get_mutation_names(self) -> Sequence[str]:
        return [self.input_name(0)]

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()

    def __init__(
        self,
        layout: OutputSpec,
        inputs: Sequence[IRNode],
        constant_args: Sequence[Any],
    ) -> None:
        super().__init__(
            None,
            layout,
            inputs,
            constant_args,
            python_kernel_name="aten.copy_",
            cpp_kernel_name="aoti_torch_copy_",
        )
        V.graph.mark_buffer_mutated(inputs[0].get_name())
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

    @classmethod
    def create(
        cls, dst: IRNode, src: IRNode, non_blocking: bool = False
    ) -> InplaceCopyFallback:
        inputs = [cls.realize_input(t) for t in [dst, src]]
        constant_args = (non_blocking,)
        result = InplaceCopyFallback(
            NoneLayout(device=dst.get_device()),
            inputs,
            constant_args,
        )
        return result


class MutatingFirstArgExternKernel(ExternKernel):
    """
    This needs to be a custom class to handle mutation properly
    """

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        assert is_node_sequence(self.inputs)
        argrefs = [
            *(t.codegen_reference() for t in self.inputs),
            *map(repr, self.constant_args),
        ]
        wrapper.writeline(
            f"{self.get_kernel_name()}({', '.join(argrefs)}){wrapper.ending}"
        )

    def should_allocate(self) -> bool:
        return False

    def get_mutation_names(self) -> Sequence[str]:
        return [self.input_name(0)]

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()

    def has_side_effects(self) -> bool:
        return True


class ResizeStorageBytes(MutatingFirstArgExternKernel):
    def __init__(self, variable: IRNode, new_size: int) -> None:
        assert isinstance(new_size, int), "TODO: dynamic shapes"
        super().__init__(
            None,
            NoneLayout(device=variable.get_device()),
            self.unwrap_storage([variable]),
            constant_args=(new_size,),
        )
        V.graph.mark_buffer_mutated(variable.get_name())
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)
        self.python_kernel_name = "inductor_ops.resize_storage_bytes_"
        self.cpp_kernel_name = "torch::inductor::resize_storage_bytes_"
        assert isinstance(variable, (BaseView, StorageBox, TensorBox)), type(variable)
        V.graph.never_reuse_buffers.add(variable.data.get_name())


class SetSourceTensorKernel(ExternKernelAlloc):
    def __init__(self, self_tensor: IRNode, storage_tensor: IRNode) -> None:
        storage_tensor.freeze_layout()
        super().__init__(
            storage_tensor.get_layout(),
            [self_tensor, storage_tensor],
            python_kernel_name="torch.ops.aten.set_.source_Tensor",
            op_overload=torch.ops.aten.set_.source_Tensor,
        )
        assert isinstance(self_tensor, (BaseView, StorageBox, TensorBox)), type(
            self_tensor
        )
        V.graph.never_reuse_buffers.add(self_tensor.data.get_name())
        V.graph.never_reuse_buffers.add(storage_tensor.get_name())
        V.graph.never_reuse_buffers.add(self.get_name())
        device = storage_tensor.get_device()
        self.mutation_outputs = [
            MutationOutput(NoneLayout(device=device), self_tensor, self),
            MutationOutput(NoneLayout(device=device), storage_tensor, self),
        ]

    def get_inputs_that_alias_output(self) -> Sequence[str]:
        return [self.input_name(0), self.input_name(1)]


class ScatterFallback(ExternKernel):
    """
    This needs to be a custom class to handle mutation properly.
    This class handles both aten.scatter_ and aten.scatter_reduce_.
    It also handle the case `src` being a scalar properly.
    """

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        wrapper.generate_scatter_fallback(self)

    def should_allocate(self) -> bool:
        return False

    def get_mutation_names(self) -> list[str]:
        inp = self.inputs[0]
        assert isinstance(inp, IRNode)
        return [inp.get_name()]

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()

    def __init__(
        self,
        op_overload: _OpOverloads,
        x: IRNode,
        dim: int,
        index: IRNode,
        src: IRNode,
        *,
        reduce: str | None = None,
        include_self: bool = True,
    ) -> None:
        self.src_is_tensor = isinstance(src, TensorBox)

        constant_args: tuple[Any, ...]
        if self.src_is_tensor:
            tensors = [self.realize_input(t) for t in [x, index, src]]
            constant_args = (dim,)
        else:
            tensors = [self.realize_input(t) for t in [x, index]]
            constant_args = (dim, src)

        super().__init__(
            None,
            NoneLayout(device=x.get_device()),
            self.unwrap_storage(tensors),
            constant_args,
            {"reduce": reduce, "include_self": include_self},
            python_kernel_name=str(op_overload),
            ordered_kwargs_for_cpp_kernel=["reduce", "include_self"],
            op_overload=op_overload,
        )
        V.graph.mark_buffer_mutated(x.get_name())
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)


class IndexPutFallback(ExternKernel):
    """
    This needs to be a custom class to handle mutation and indices properly
    """

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        wrapper.generate_index_put_fallback(self)

    def should_allocate(self) -> bool:
        return False

    def get_mutation_names(self) -> Sequence[str]:
        return [self.input_name(0)]

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()

    def __init__(
        self,
        op_overload: torch._ops.OpOverload,
        x: IRNode,
        indices: list[Any],
        values: Sequence[Any],
        accumulate: Any,
    ) -> None:
        self.indices = indices
        valid_indices = [i for i in indices if i is not None]
        # pyrefly: ignore [bad-argument-type]
        tensors = [self.realize_input(x) for x in [x, values, *valid_indices]]
        cpp_kernel_name = "aoti_torch_index_put_out"
        super().__init__(
            None,
            NoneLayout(device=x.get_device()),
            self.unwrap_storage(tensors),
            (accumulate,),
            python_kernel_name="aten.index_put_",
            cpp_kernel_name=cpp_kernel_name,
            op_overload=op_overload,
        )
        V.graph.mark_buffer_mutated(self.input_name(0))
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)


class DeviceCopy(ExternKernelOut):
    @classmethod
    def create(cls, x: IRNode, device: torch.device, non_blocking: bool) -> IRNode:
        x_device = x.get_device()
        assert x_device is not None
        if (
            not x.is_extern()
            # Can not apply this optimization if x has been mutated
            and try_get_name(x) not in V.graph.mutated_buffers
            and all(r in V.graph.constants for r in x.get_read_names())
            and not config.aot_inductor.use_runtime_constant_folding
        ):
            if V.graph.cpp_wrapper:
                # Even if x is promoted to be a device constant, we still need to
                # register device info to construct the correct CppWrapper class later
                V.graph.add_device_info(device)
                V.graph.add_device_info(x_device)
            return x.constant_to_device(device)

        V.graph.add_device_info(device)
        V.graph.add_device_info(x_device)
        developer_warning("DeviceCopy in input program")
        constant_args = (non_blocking,)
        # Device Copy should keep the same layout as input
        x = ExternKernel.require_contiguous(x)
        stride = None
        if x.get_size():
            # x.get_stride() may be unimplemented if x's size is empty
            stride = x.get_stride()
        is_destination_pinned = (
            is_gpu(x_device.type) and device.type == "cpu" and non_blocking
        )
        is_source_pinned = (
            x_device.type == "cpu" and is_gpu(device.type) and non_blocking
        )
        if is_source_pinned and is_storage_and_layout(x):
            x.get_layout().is_pinned = True
        return DeviceCopy(
            FixedLayout(
                device,
                x.get_dtype(),
                x.get_size(),
                stride,
                is_pinned=is_destination_pinned,
            ),
            [cls.realize_input(x)],
            constant_args,
        )

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        args = self.codegen_args()
        assert len(args) == 2
        if self.output_view:
            wrapper.codegen_device_copy(
                args[0], self.output_view.codegen_reference(), args[1]
            )
        else:
            wrapper.codegen_device_copy(args[0], self.codegen_reference(), args[1])


class DynamicSelectStorageOffset(ExternKernel):
    """
    The result of computing a dynamic selection index is determined as follows: when the index in the
    select operation is unbacked, the actual index calculation is ambiguous for negative indices
    (index + size) versus non-negative indices (just index). To resolve this, we allocate an unbacked
    SymInt to represent the storage offset and decompose the select operation into a call to as_strided,
    computing the storage offset at runtime with this node.
    """

    def get_reads(self) -> OrderedSet[Dep]:
        return OrderedSet()

    def should_allocate(self) -> bool:
        return False

    def __init__(
        self,
        unbacked_offset_symbol: sympy.Symbol,
        index: sympy.Symbol,
        base_offset: sympy.Symbol | int,
        base_dim_stride: sympy.Symbol | int,
        size: sympy.Symbol | int,
        clamp: bool,
    ) -> None:
        super().__init__(None, NoneLayout(device=torch.device("cpu")), [])
        # This node codegen the following:
        # unbacked_offset_symbol = base_offset + base_dim_stride * (index if index >=0 else index + size)
        self.unbacked_offset_symbol = unbacked_offset_symbol
        self.index = index
        self.base_offset = base_offset
        self.base_dim_stride = base_dim_stride
        self.size = size
        self.clamp = clamp

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        return OrderedSet([self.unbacked_offset_symbol])

    @cache_on_self_and_args("DynamicSelectStorageOffset")
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        return get_free_symbols(self.index, unbacked_only)

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        wrapper.codegen_dynamic_select_index(self, clamp=self.clamp)


class DynamicSliceSize(ExternKernel):
    """
    Computes the output size of a slice call, handling the correct semantics in codegen.
    We do this for flexible handling for unbacked indices (to not data-dependent error).

    Slicing has 4 semantics for indices, i.e. x[start:] could be:
    1) start < -x.size(0)            -> x[0:]                    # negative out-of-bounds
    2) start in [-x.size(0), 0)      -> x[x.size(0) + start:]    # negative slicing
    3) start in [0, x.size(0))       -> x[start:]                # standard slicing
    4) start >= x.size(0)            -> empty slice              # positive out-of-bounds

    If the appropriate semantics are known beforehand, the output size is computed based on
    the start & end indices. If not (with unbacked indices), a new unbacked symbol is created
    to represent the output size, and codegen handles computing the correct case.
    """

    def get_reads(self) -> OrderedSet[Dep]:
        return OrderedSet()

    def should_allocate(self) -> bool:
        return False

    def __init__(
        self,
        unbacked_size_symbol: sympy.Symbol,
        start: sympy.Symbol | int,
        end: sympy.Symbol | int,
        step: sympy.Symbol | int,
        size: sympy.Symbol | int,
    ):
        super().__init__(None, NoneLayout(device=torch.device("cpu")), [])
        # This node codegen
        self.unbacked_size_symbol = unbacked_size_symbol
        self.start = start
        self.end = end
        self.step = step
        self.size = size

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        return OrderedSet([self.unbacked_size_symbol])

    @cache_on_self_and_args("DynamicSliceSize")
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        return get_free_symbols(self.start, unbacked_only).union(
            get_free_symbols(self.end, unbacked_only)
        )

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        wrapper.codegen_dynamic_slice_size(self)


class DynamicScalar(ExternKernel):
    """
    The result of a call to aten._local_scalar_dense.
    """

    def get_reads(self) -> OrderedSet[Dep]:
        return OrderedSet()

    def should_allocate(self) -> bool:
        return False

    def __init__(
        self, sym: sympy.Symbol, keypath: pytree.KeyPath, data: IRNode
    ) -> None:
        data.realize()
        super().__init__(
            None, NoneLayout(device=torch.device("cpu")), self.unwrap_storage([data])
        )
        self.sym = sym
        self.keypath = keypath

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        return OrderedSet([self.sym])

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        wrapper.codegen_dynamic_scalar(self)


class AssertScalar(ExternKernel):
    """
    The result of a call to aten._assert_scalar
    """

    def get_reads(self) -> OrderedSet[Dep]:
        return OrderedSet()

    def should_allocate(self) -> bool:
        return False

    def __init__(self, scalar: SympyBoolean, msg: str) -> None:
        super().__init__(
            # Buffer(name, layotu)
            None,
            NoneLayout(device=torch.device("cpu")),
            # InputsKernel(inputs)
            [],
        )
        self.scalar = scalar
        self.msg = msg

    def has_side_effects(self) -> bool:
        return True

    @cache_on_self_and_args("AssertScalar")
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        return get_free_symbols(self.scalar, unbacked_only)

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        if not config.scalar_asserts:
            return
        # NB: It is EXTREMELY important not to simplify the scalar under assertion here,
        # because simplify is done with respect to runtime asserts.  So if you have
        # "u0 == 0" in the runtime asserts, if you subsequently try to
        # simplify(u0 == 0), you will get True (because we've already runtime assert'ed
        # that it's true).  But we're code generating the actual runtime assert here!!
        symbol = next(iter(self.get_free_symbol_uses(unbacked_only=False)))
        if V.graph.fx_wrapper:
            # TODO fix
            pass
        elif V.graph.cpp_wrapper:
            symbol_str = f"std::to_string({symbol})"
            sizevar = V.graph.wrapper_code.codegen_cpp_sizevar(
                self.scalar, simplify=False
            )
            # TODO: when we start compiling in C++20, annotate with [[unlikely]].
            wrapper.writeline(
                f'if (!({sizevar})) {{ throw std::runtime_error("Expected {self.msg} but received " + {symbol_str}); }}'
            )
        else:
            sizevar = V.graph.wrapper_code.codegen_python_sizevar(
                self.scalar, simplify=False
            )
            wrapper.writeline(f"if not ({sizevar}):")
            wrapper.writeline(f"    raise RuntimeError({repr(self.msg)})")
            # No one should ever use this buffer, but for uniformity
            # define the variable and assign it None
            wrapper.writeline(f"{self.get_name()} = None")


@ir_dataclass(frozen=False)
class ExternKernelNode:
    name: str
    node: export_schema.Node


class FallbackKernel(ExternKernelAlloc):
    """
    A class that represents a fallback kernel for handling operators that are not
    directly support by inductor. It currently supports functional ops, view ops,
    inplace aten ops, and mutating ops that are auto-functionalizable.
    """

    def __init__(
        self,
        layout: OutputSpec,
        kernel: _OpOverloads,
        tensor_args: Sequence[IRNode],
        nontensor_args: Sequence[Any],
        unflatten_args: Callable[..., Any],
        kwargs: dict[str, Any] | None = None,
        *,
        unbacked_bindings: dict[sympy.Symbol, pytree.KeyPath] | None = None,
    ) -> None:
        super().__init__(
            layout,
            tuple(tensor_args),
            tuple(nontensor_args),
            op_overload=kernel,
        )

        self.use_runtime_dispatch = False
        self.unbacked_bindings = unbacked_bindings or {}

        assert isinstance(
            kernel, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)
        ), f"Fails to create FallbackKernel for {kernel}: {type(kernel)} not supported"
        self.op_overload = kernel
        self.unflatten_args = unflatten_args
        self.kwargs = {} if kwargs is None else kwargs
        assert self.python_kernel_name is not None
        V.graph.warn_fallback(self.python_kernel_name)

        # args that are aliased
        self.alias_names: list[str] = []
        # args that are mutated AND returned from the op
        self.mutation_names: list[str] = []

        if isinstance(self.op_overload, torch._ops.HigherOrderOperator):
            # We assume here that HOPs with FallbackKernel are functional.
            # This may not always be true! HOPs must individually opt-in to
            # FallbackKernel, so please check this if you opt-in.
            return

        if "_c10d_functional" in self.op_overload.name():
            # _c10d_functional kernels are lowered into _CollectiveKernel which
            # derives from FallbackKernel for the cpp codegen. The kernels
            # don't pass the can_auto_functionalize check, but their mutation
            # is handled properly by _CollectiveKernel.
            return

        schema = self.op_overload._schema

        # NOTE: [FallbackKernel supported operators]
        # We only support three types of operators:
        # - functional ops
        # - view ops
        # - inplace aten ops
        # - mutating ops that are auto-functionalizable. That is,
        # the operator may mutate any number of inputs, but its outputs
        # may not alias any of the inputs.
        #
        # The unsupported cases usually do not show up here (because
        # AOTAutograd functionalized them away); the only way for an in-place
        # op to show up here is if a lowering or pass introduced it.
        if torch._library.utils.mutates_and_returns_first_arg(self.op_overload):
            self.mutation_names.append(tensor_args[0].get_name())
            return

        def has_functionalize_impl(op: torch._ops.OpOverload) -> bool:
            return torch._C._dispatch_has_kernel_for_dispatch_key(
                op.name(), torch._C.DispatchKey.Functionalize
            ) or (
                hasattr(op, "py_kernels")
                and torch._C.DispatchKey.Functionalize in op.py_kernels
            )

        if (
            schema.is_mutable
            and not can_auto_functionalize(self.op_overload)
            and not has_functionalize_impl(self.op_overload)
        ):
            raise NotImplementedError(
                f"NYI: Can't generate FallbackKernel for {self.op_overload}"
            )

        args, kwargs = self.unflatten_args(self.inputs, self.constant_args)

        def handle_aliasing_and_mutation(info: torch._C.Argument, arg: Any) -> None:
            # Assertions to make sure we didn't mismatch args
            if isinstance(info.type, torch.ListType):
                assert isinstance(arg, (list, tuple)), type(arg)
            if library_utils.is_tensor_like_type(info.type):
                # PyTorch also accepts None and scalar types for args marked as "Tensor".
                # We're not going to check all of them here.
                assert not isinstance(arg, (tuple, list))

            if arg is None:
                return
            if info.alias_info is None:
                return

            def add_alias(t: IRNode) -> None:
                self.alias_names.append(t.get_name())
                assert info.alias_info is not None
                if info.alias_info.is_write:
                    self.mutation_outputs.append(
                        MutationOutput(NoneLayout(device=t.get_device()), t, self)
                    )

            if library_utils.is_tensorlist_like_type(info.type):
                if arg is not None:
                    for optional_tensor_arg in arg:
                        add_alias(optional_tensor_arg)
            else:
                assert library_utils.is_tensor_like_type(info.type)

                add_alias(arg)

        for info, arg in torch._library.utils.zip_schema(schema, args, kwargs):
            handle_aliasing_and_mutation(info, arg)

    def get_read_writes(self) -> dependencies.ReadWrites:
        read_writes = super().get_read_writes()

        if self.op_overload is torch._prims.rng_prims.graphsafe_run_with_rng_state:
            for arg in self.constant_args:
                if isinstance(arg, GeneratorState):
                    read_writes = read_writes.with_read(
                        dependencies.StarDep(arg.get_name())
                    )

        return read_writes

    def codegen_unbacked_symbol_defs(self, wrapper: PythonWrapperCodegen) -> None:
        return wrapper.codegen_unbacked_symbol_defs_for_outputs(
            self.get_name(), self.outputs, getattr(self, "unbacked_bindings", None)
        )

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        if unbacked_bindings := getattr(self, "unbacked_bindings", None):
            resolved = resolve_unbacked_bindings(
                V.graph.sizevars.shape_env, unbacked_bindings
            )
            assert resolved is not None
            return OrderedSet(resolved.keys())
        else:
            return OrderedSet()

    def codegen_args(self) -> list[str]:
        @dataclasses.dataclass
        class Shim:
            ref: Any

            def __repr__(self) -> str:
                return self.ref

        assert is_node_sequence(self.inputs)
        tensor_args = [Shim(x.codegen_reference()) for x in self.inputs]
        args, kwargs = self.unflatten_args(tensor_args, self.constant_args)
        if V.graph.cpp_wrapper and isinstance(self.op_overload, torch._ops.OpOverload):
            args = self.fill_non_provided_args(args, kwargs)
            args = [
                V.graph.wrapper_code.val_to_arg_str(x, param.real_type)
                for param, x in zip(self.op_overload._schema.arguments, args)
            ]
        else:
            args = [V.graph.wrapper_code.val_to_arg_str(x) for x in args]

        # let self.codegen_kwargs handle kwargs
        self.kwargs.update(kwargs)
        return args

    @staticmethod
    def find_device(
        tensor_args: Sequence[torch.Tensor] | None, example_output: Sequence[Any]
    ) -> Any:
        non_torch_bind_tensor_args = (
            [t for t in tensor_args if not isinstance(t, TorchBindObject)]
            if tensor_args
            else None
        )
        if non_torch_bind_tensor_args:
            assert tensor_args
            devices = [arg.get_device() for arg in tensor_args if arg.get_device()]
            return devices[0]
        if isinstance(example_output, torch.Tensor):
            return example_output.device
        if isinstance(
            example_output, (torch._C.ScriptObject, FakeScriptObject)
        ) or is_opaque_value(example_output):
            return torch.device("cpu")
        if isinstance(example_output, (list, tuple)):
            device_set = OrderedSet(
                # pyrefly: ignore [bad-argument-type]
                FallbackKernel.find_device(None, x)
                for x in example_output
            )
            # Remove None
            devices = [device for device in device_set if device]
            if len(devices) == 1:
                return devices[0]
            if not devices:
                return None
            for device in devices:
                assert isinstance(device, torch.device)
                if is_gpu(device.type):
                    return device
            return devices[0]
        return None

    def has_side_effects(self) -> bool:
        from torch._library.utils import is_impure

        # Note: We don't pass args/kwargs here because they're IRNodes, not actual values
        # The check is done on the op_overload itself
        return is_impure(self.op_overload)  # pyrefly: ignore[bad-argument-type]

    def get_inputs_that_alias_output(self) -> Sequence[str]:
        assert isinstance(
            self.op_overload, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)
        ), (
            f"Fails to create FallbackKernel for {self.op_overload}: "
            f"{type(self.op_overload)} not supported"
        )

        # See [Note: FallbackKernel supported operators]: for a mutating
        # op that is auto-functionalizable, its outputs does NOT
        # alias any of the inputs.
        if (
            not isinstance(self.op_overload, torch._ops.HigherOrderOperator)
            and "_c10d_functional" not in self.op_overload.name()
            and self.op_overload._schema.is_mutable
            and can_auto_functionalize(self.op_overload)
        ):
            return []
        else:
            return self.alias_names

    def get_mutation_names(self) -> Sequence[str]:
        assert len(self.mutation_names) <= 1
        return self.mutation_names

    def export_extern_kernel_node(self):  # type: ignore[no-untyped-def]
        """
        ProxyExecutor Design Note
        We export the ExternFallbackNodes (for custom ops) into a serialized file
        and run it with a host side proxy executor to address the ABI problem
        This is currently only implemented for fbcode. Eventually, we will also make this work for OSS.
        Detailed design doc can be found at
        https://docs.google.com/document/d/1wC4DOZFaYym2t1Esz0X5yxlLI3RDnSiyRbUus3bkJ64/edit?usp=sharing
        """
        log.debug(
            "Extern kernel node added for node %s with target %s.",
            self.get_name(),
            self.op_overload,
        )

        assert isinstance(self, FallbackKernel), type(self)
        args, kwargs = self.unflatten_args(self.inputs, self.constant_args)
        args = self.fill_non_provided_args(args, kwargs)
        ordered_kwargs = [
            self.get_kwargs_value(key, **kwargs)
            for key in self.ordered_kwargs_for_cpp_kernel
        ]
        target = self.op_overload

        if not V.graph.aot_mode:
            # No need to serialize in the cpp wrapper JIT mode
            return [*args, *ordered_kwargs]

        serializer = GraphModuleSerializer(None, [])  # type: ignore[arg-type]
        named_arguments = serializer.serialize_inputs(target, args, kwargs)

        # serialize_outputs
        def handle_single_output(
            return_type: torch.TensorType | torch.ListType | torch.JitType,
            output: IRNode | Sequence[IRNode],
        ) -> export_schema.Argument:
            if isinstance(return_type, (torch.TensorType, torch.NoneType)):
                # For single Tensor or None
                out = output
                if isinstance(output, (list, tuple)):
                    assert len(output) == 1
                    out = output[0]
                if isinstance(return_type, torch.TensorType):
                    assert isinstance(out, IRNode)
                    return export_schema.Argument.create(
                        as_tensor=export_schema.TensorArgument(name=out.get_name())
                    )
                else:  # NoneType
                    assert out is None
                    return export_schema.Argument.create(as_none=True)
            elif isinstance(return_type, torch.ListType) and isinstance(
                return_type.getElementType(), torch.TensorType
            ):
                assert isinstance(output, Sequence), type(output)
                # For single TensorList
                return export_schema.Argument.create(
                    as_tensors=[
                        export_schema.TensorArgument(name=out.get_name())
                        for out in output
                    ]
                )
            elif isinstance(return_type, torch.OptionalType) and isinstance(
                return_type.getElementType(), torch.TensorType
            ):
                # For OptionalTensor
                if output is None:
                    return export_schema.Argument.create(
                        as_optional_tensor=export_schema.OptionalTensorArgument.create(
                            as_none=True
                        )
                    )
                else:
                    assert isinstance(output, IRNode)
                    return export_schema.Argument.create(
                        as_optional_tensor=export_schema.OptionalTensorArgument.create(
                            as_tensor=export_schema.TensorArgument(
                                name=output.get_name()
                            )
                        )
                    )
            elif isinstance(return_type, torch.IntType):
                return export_schema.Argument.create(as_int=output)
            else:
                raise RuntimeError(f"Unsupported return type {type(return_type)}")

        if isinstance(target, torch._higher_order_ops.torchbind.CallTorchBind):
            returns = target.schema(args[0], args[1]).returns
        else:
            returns = target._schema.returns  # type: ignore[union-attr]
        if len(returns) == 1:
            # NOTE: [special handling of all_reduce_coalesced_'s return value]
            # all_reduce_coalesced_ return a list of tensors via self.mutation_outputs
            outputs = self.outputs if self.outputs else self.mutation_outputs
            return_type = returns[0].real_type
            output_arguments = [handle_single_output(return_type, outputs)]
        else:
            # For tuple returns, e.g "-> (Tensor, Tensor)" or "-> (Tensor, Tensor[])"
            # Not generating output args for self.mutation_outputs
            output_arguments = [
                handle_single_output(
                    return_schema.real_type,  # type: ignore[attr-defined]
                    output,
                )
                for return_schema, output in zip(returns, self.outputs)
            ]

        assert self.op_overload is not None
        node = ExternKernelNode(
            name=self.get_name(),
            node=export_schema.Node(
                target=self.op_overload.name(),
                inputs=named_arguments,
                outputs=output_arguments,
                metadata={},
            ),
        )

        V.extern_kernel_nodes.append(node)

        return [*args, *ordered_kwargs]

    @override
    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        """Overrides the parent member.
        See https://github.com/pytorch/pytorch/issues/151692"""
        kernel = self.op_overload
        assert kernel is not None
        if kernel.namespace == "aten":
            # Aten Fallback Ops
            assert isinstance(kernel, torch._ops.OpOverload), type(kernel)
            if V.graph.cpp_wrapper:
                from torchgen.aoti.fallback_ops import inductor_fallback_ops

                if str(kernel) not in inductor_fallback_ops:
                    # C shim v2 is torchgen-ed, which should cover all aten ops.
                    # If you do hit a missed op, please update fallback_ops.py.
                    log.warning(
                        "%s is missing a c-shim implementation, using proxy executor as fallback",
                        kernel,
                    )
                    self.use_runtime_dispatch = True
        elif kernel.namespace == "_quantized":
            # Internal Quantized Fallback Ops
            assert isinstance(kernel, torch._ops.OpOverload), type(kernel)
        elif V.graph.cpp_wrapper:
            # For non-aten OpOverload, i.e. custom ops
            # If the op is in custom_ops_to_c_shims, generate direct function call
            self.use_runtime_dispatch = (
                kernel not in config.aot_inductor.custom_ops_to_c_shims
            )

        # Handle the special case where a complex number is input to a C-shim kernel for
        # a scalar input.  The torchgen'ed shim API will use type "double", which is
        # incompatible with complex numbers, forcing a fallback to runtime dispatch.
        if (
            V.graph.cpp_wrapper
            and isinstance(kernel, torch._ops.OpOverload)
            and not self.use_runtime_dispatch
        ):

            def is_number(t: torch.JitType) -> bool:
                if isinstance(t, torch.OptionalType):
                    return is_number(t.getElementType())
                return isinstance(t, torch.NumberType)

            # Using unflatten_args is a bit of a hack, but all the complex arguments we
            # care about are in self.constant_args, and calling unflatten_args puts them
            # in the correct order without triggering codegen.
            args, kwargs = self.unflatten_args(self.inputs, self.constant_args)
            # Append kwarg values to args.  ordered_kwargs_for_cpp_kernel is guaranteed
            # to be set, since this is an OpOverload kernel.
            args_iter = itertools.chain(
                args,
                (
                    self.get_kwargs_value(k, **kwargs)
                    for k in self.ordered_kwargs_for_cpp_kernel
                ),
            )
            self.use_runtime_dispatch = any(
                isinstance(v, complex) and is_number(a.real_type)
                for v, a in zip(args_iter, kernel._schema.arguments)
            )

        self.codegen_comment(wrapper)
        if self.use_runtime_dispatch:
            exported_args = self.export_extern_kernel_node()
            assert self.python_kernel_name is not None
            assert self.op_overload is not None

            wrapper.generate_fallback_kernel_with_runtime_lookup(
                self.get_name(),
                self.python_kernel_name,
                lambda: [*self.codegen_args(), *self.codegen_kwargs()],
                self.op_overload,
                exported_args,
                # NOTE: [special handling of all_reduce_coalesced_'s return value]
                self.outputs if self.outputs else self.mutation_outputs,
            )
        else:
            wrapper.generate_fallback_kernel(self)
            if isinstance(self.layout, Layout):
                self.codegen_size_asserts(wrapper)
                self.codegen_alignment_asserts(wrapper)
                self.codegen_memory_tracking(wrapper)

        self.codegen_unbacked_symbol_defs(wrapper)

    @staticmethod
    def tensor_to_layout(output: torch.Tensor) -> FixedLayout:
        is_pinned = False
        try:
            is_pinned = output.is_pinned()
        except RuntimeError:
            # dispatch not implemented
            pass
        return FixedLayout(
            output.device,
            output.dtype,
            convert_shape_to_inductor(output.size()),
            convert_shape_to_inductor(output.stride()),
            is_pinned=is_pinned,
        )

    @classmethod
    def create(cls, kernel: _OpOverloads, *args: Any, **kwargs: Any) -> FallbackKernel:
        """Create an instance of FallbackKernel from an _OpOverloads"""
        fake_incorrect_kernels = (aten._fused_moving_avg_obs_fq_helper_functional,)
        if kernel not in fake_incorrect_kernels:
            context = cast(AbstractContextManager[None], V.graph.fake_mode)
        else:
            context = nullcontext()

        with context:
            (
                example_output,
                tensor_args,
                non_tensor_args,
                unflatten_args,
                unbacked_bindings,
            ) = cls.process_kernel(kernel, *args, **kwargs)

        # Try to lower single output functional custom ops to their out-variant.
        if isinstance(kernel, torch._ops.OpOverload) and isinstance(
            example_output, torch.Tensor
        ):
            from torch._library._out_variant import (
                _is_functional,
                get_out_arg_names,
                lookup_manual_out_variant,
                to_out_variant,
            )

            out_op = None
            if _is_functional(kernel._schema):
                out_op = to_out_variant(kernel)
            if out_op is None:
                out_op = lookup_manual_out_variant(kernel)

            if out_op is not None and len(get_out_arg_names(out_op)) == 1:
                layout = FixedLayout(
                    device=example_output.device,
                    dtype=example_output.dtype,
                    size=[*example_output.shape],
                    stride=[*example_output.stride()],
                )
                return ExternKernelOut(  # type: ignore[return-value]
                    layout=layout,
                    inputs=list(tensor_args),
                    constant_args=list(non_tensor_args),
                    kwargs=kwargs,
                    python_kernel_name=_make_out_variant_kernel_name(out_op),
                    op_overload=out_op,
                )

        # We need this extra check for input alignment since the example
        # inputs we created are always aligned.
        has_unaligned_input = any(is_unaligned(arg) for arg in tensor_args)

        device = cls.find_device(tensor_args, example_output)

        # Default to CPU for torchbind methods or HOPs that don't produce tensors
        if not device and (
            isinstance(kernel, torch._higher_order_ops.torchbind.CallTorchBind)
            or kernel is torch.ops.higher_order.print
        ):
            device = torch.device("cpu")

        # Try multi-output .out() lowering for ops with out_variant tag.
        if (
            isinstance(kernel, torch._ops.OpOverload)
            and not V.graph.cpp_wrapper
            and device
        ):
            out_result = ExternKernelMultiOut.try_create(
                kernel,
                example_output,
                device,
                tensor_args,
                non_tensor_args,
                unflatten_args,
                kwargs,
                unbacked_bindings=unbacked_bindings,
                has_unaligned_input=has_unaligned_input,
            )
            if out_result is not None:
                return out_result  # type: ignore[return-value]

        if example_output is None:
            packed = cls(
                NoneLayout(device=device),
                kernel,
                tensor_args,
                non_tensor_args,
                unflatten_args,
                kwargs=kwargs,
                unbacked_bindings=unbacked_bindings,
            )

        else:
            assert device, "Not sure where to find device info"
            packed = cls(
                MultiOutputLayout(device=device),
                kernel,
                tensor_args,
                non_tensor_args,
                unflatten_args,
                kwargs=kwargs,
                unbacked_bindings=unbacked_bindings,
            )

        def generate_output(output: Any, indices: list[tuple[Any, int]]) -> Any:
            if isinstance(output, (list, tuple)):
                return type(output)(
                    generate_output(output[i], indices + [(type(output), i)])
                    for i in range(len(output))
                )
            elif isinstance(output, dict):
                return {
                    key: generate_output(val, indices + [(type(output), key)])
                    for key, val in output.items()
                }
            elif isinstance(output, torch.Tensor):
                buf = MultiOutput(
                    cls.tensor_to_layout(output),
                    packed,
                    indices,
                )
                if (
                    config.assume_unaligned_fallback_output
                    or has_unaligned_input
                    or not tensor_is_aligned(output)
                ):
                    V.graph.unaligned_buffers.add(buf.name)  # type: ignore[arg-type]
                return buf
            elif isinstance(output, int):
                return output
            elif isinstance(output, torch.SymInt):
                return output.node.expr
            elif isinstance(
                output, (torch._C.ScriptObject, FakeScriptObject)
            ) or is_opaque_value(output):
                return OpaqueMultiOutput(
                    NoneLayout(device=device),
                    packed,
                    indices,
                    output,
                )
            else:
                assert output is None, (
                    f"FallbackKernel output type {type(output)} is not supported"
                )
                return None

        outputs = generate_output(example_output, [])
        if isinstance(outputs, (list, tuple)):
            packed.outputs = outputs
        elif isinstance(outputs, dict):
            packed.outputs = tuple(outputs)
        else:
            packed.outputs = [outputs]

        return outputs


@ir_dataclass(frozen=False)
class ComplexView(FallbackKernel):
    """View a complex number as two dtyped numbers or vice versa"""

    def should_allocate(self) -> bool:
        return False

    def get_inputs_that_alias_output(self) -> Sequence[str]:
        # Signal to codegen that our output buffer isn't safe to reuse
        return [self.input_name(0)]

    def __init__(
        self,
        layout: OutputSpec,
        kernel: _OpOverloads,
        tensor_args: Sequence[IRNode],
        nontensor_args: Sequence[Any],
        unflatten_args: Callable[..., Any],
        *,
        kwargs: dict[str, Any] | None = None,
        unbacked_bindings: dict[sympy.Symbol, pytree.KeyPath] | None = None,
    ) -> None:
        super().__init__(
            layout,
            kernel,
            tensor_args,
            nontensor_args,
            unflatten_args,
            kwargs=kwargs,
            unbacked_bindings=unbacked_bindings,
        )


class MemoryCheckKernel(FallbackKernel):
    """
    Custom kernel for memory checking that generates direct function calls

    TODO - the custom op was erroring with str inputs. should be able to custom op directly.
    """

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        """Override codegen to write direct function call"""
        # Extract our arguments from nontensor_args
        wrapper.write_memory_track_allocation_once()
        alive_list, dead_list, is_final_step = self.constant_args

        alive_repr = repr(alive_list)
        dead_repr = repr(dead_list)
        if is_final_step:
            wrapper.writeline(
                "# note: dont currently distinguish between buffers returned and dealloc'd in last step"
            )
            call = f"check_memory_step(allocated={alive_repr}, freed={dead_repr}, is_final_step={is_final_step})"
        else:
            call = f"check_memory_step(allocated={alive_repr}, freed={dead_repr})"
        wrapper.writeline(call)


@ir_dataclass
class MultiOutputLayout(OutputSpec):
    device: torch.device

    def get_device(self) -> torch.device | None:
        return self.device


class MultiOutput(ExternKernel):
    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        wrapper.codegen_multi_output(self)
        if not self.skip_size_stride_alignment_checks:
            self.codegen_size_asserts(wrapper)
            self.codegen_alignment_asserts(wrapper)

    def __init__(
        self,
        layout: OutputSpec,
        input: IRNode,
        indices: list[tuple[Any, ...]],
        skip_size_stride_alignment_checks: bool = False,
    ) -> None:
        super().__init__(None, layout, [input], ())
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)
        self.indices = indices
        self.skip_size_stride_alignment_checks = skip_size_stride_alignment_checks

    @cache_on_self_and_args("MultiOutput")
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        input_node = self.inputs[0]
        assert isinstance(input_node, IRNode), input_node
        return input_node.get_free_symbol_uses(unbacked_only)

    def should_allocate(self) -> bool:
        return len(self.inputs) == 1 and (
            isinstance(self.inputs[0], CppTemplateBuffer)  # Grouped GEMM
        )

    def get_inputs_that_alias_output(self) -> Sequence[str]:
        return [
            inp.get_name()
            for inp in self.inputs
            if isinstance(inp, FallbackKernel)
            and len(inp.get_inputs_that_alias_output()) > 0
        ]

    def get_read_writes(self) -> dependencies.ReadWrites:
        # Reads: StarDep on parent (we don't know which elements of the
        # packed output we index into — conservative is correct).
        reads: OrderedSet[dependencies.Dep] = OrderedSet()
        for inp in self.inputs:
            if isinstance(inp, IRNode):
                reads.add(dependencies.StarDep(inp.get_name()))

        # Writes: build proper MemoryDep from our FixedLayout so the
        # scheduler can match our write with downstream epilogue reads.
        # Normalize using the same policy as SchedulerNode so that the
        # index expressions are directly comparable during fusion checks.
        name = self.get_name()
        indexer = self.get_layout().make_indexer()

        def dummy(index: Sequence[Any], rindex: Sequence[Any]) -> Any:
            assert len(rindex) == 0
            return ops.store(name, indexer(index), "fake")

        device = self.get_device()
        should_normalize = (
            not config.loop_ordering_after_fusion
            or device is None
            or not is_gpu(device.type)
        )
        write_rw = dependencies.extract_read_writes(
            dummy, self.get_size(), (), normalize=should_normalize
        )
        return dependencies.ReadWrites(
            reads=reads,
            writes=write_rw.writes,
            index_exprs=OrderedSet(),
        )


class OpaqueMultiOutput(MultiOutput):
    """MultiOutput for opaque objects."""

    def __init__(
        self,
        layout: OutputSpec,
        input: IRNode,
        indices: list[tuple[Any, ...]],
        opaque_value: Any,
    ) -> None:
        super().__init__(layout, input, indices, skip_size_stride_alignment_checks=True)
        self.opaque_example_value = opaque_value

    @property  # type: ignore[override]
    def dtype(self) -> Never:
        raise AttributeError("OpaqueMultiOutput has no dtype")

    def wrap_for_lowering(self) -> OpaqueMultiOutput:
        return self

    def get_read_writes(self) -> dependencies.ReadWrites:
        reads: OrderedSet[dependencies.Dep] = OrderedSet()
        for inp in self.inputs:
            if isinstance(inp, IRNode):
                reads.add(dependencies.StarDep(inp.get_name()))
        writes: OrderedSet[dependencies.Dep] = OrderedSet(
            [dependencies.StarDep(self.get_name())]
        )
        return dependencies.ReadWrites(
            reads=reads,
            writes=writes,
            index_exprs=OrderedSet(),
        )


class AllocatingMultiOutput(MultiOutput):
    """MultiOutput with Inductor-controlled allocation for .out() variant ops.

    Overrides should_allocate()=True so Inductor allocates the output buffer,
    and skips tuple-indexing codegen since .out() writes directly into these buffers.
    """

    def should_allocate(self) -> bool:
        return True

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        if not self.skip_size_stride_alignment_checks:
            self.codegen_size_asserts(wrapper)
            self.codegen_alignment_asserts(wrapper)


def _make_out_variant_kernel_name(out_op: torch._ops.OpOverload) -> str:
    """Build fully-qualified kernel name for an out-variant op."""
    ns = out_op.namespace
    op_name = out_op._schema.name.split("::")[1]
    overload = out_op._overloadname
    return f"torch.ops.{ns}.{op_name}.{overload}"


class ExternKernelMultiOut(FallbackKernel):
    """Multi-output .out() variant lowering.

    Subclass of FallbackKernel that emits .out() calls with pre-allocated
    output buffers. Uses AllocatingMultiOutput child nodes for each output.
    """

    out_arg_names: list[str]
    out_variant_output_nodes: list[AllocatingMultiOutput]

    def __init__(
        self,
        *args: Any,
        out_op: torch._ops.OpOverload,
        out_arg_names: list[str],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.out_arg_names = out_arg_names
        self.out_variant_output_nodes = []
        self.python_kernel_name = _make_out_variant_kernel_name(out_op)
        self.op_overload = out_op

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        self.codegen_comment(wrapper)
        wrapper.generate_extern_kernel_multi_out(self)

    @classmethod
    def try_create(
        cls,
        kernel: torch._ops.OpOverload,
        example_output: Any,
        device: torch.device,
        tensor_args: Sequence[IRNode],
        non_tensor_args: Sequence[Any],
        unflatten_args: Callable[..., Any],
        kwargs: dict[str, Any] | None,
        *,
        unbacked_bindings: dict[sympy.Symbol, pytree.KeyPath] | None = None,
        has_unaligned_input: bool = False,
    ) -> Sequence[AllocatingMultiOutput] | None:
        """Create an ExternKernelMultiOut if the op has a matching .out() variant."""
        from torch._library._out_variant import (
            _is_functional,
            get_out_arg_names,
            to_out_variant,
        )

        if not _is_functional(kernel._schema):
            return None

        if not isinstance(example_output, (tuple, list)):
            return None

        out_op = to_out_variant(kernel)
        if out_op is None:
            return None

        out_arg_names = get_out_arg_names(out_op)
        if not all(isinstance(t, torch.Tensor) for t in example_output):
            return None
        if len(example_output) != len(out_arg_names):
            return None

        packed = cls(
            MultiOutputLayout(device=device),
            kernel,
            tensor_args,
            non_tensor_args,
            unflatten_args,
            kwargs=kwargs,
            unbacked_bindings=unbacked_bindings,
            out_op=out_op,
            out_arg_names=out_arg_names,
        )

        outputs: list[AllocatingMultiOutput] = []
        for i, tensor_out in enumerate(example_output):
            layout = FixedLayout(
                device=tensor_out.device,
                dtype=tensor_out.dtype,
                size=[*tensor_out.shape],
                stride=[*tensor_out.stride()],
            )
            multi_out = AllocatingMultiOutput(
                layout=layout,
                input=packed,
                indices=[(type(example_output), i)],
            )
            if (
                config.assume_unaligned_fallback_output
                or has_unaligned_input
                or not tensor_is_aligned(tensor_out)
            ):
                V.graph.unaligned_buffers.add(multi_out.name)  # type: ignore[arg-type]
            outputs.append(multi_out)

        packed.out_variant_output_nodes = outputs
        packed.outputs = outputs

        if isinstance(example_output, tuple):
            return tuple(outputs)  # type: ignore[return-value]
        return list(outputs)


# We just use a normal dataclass for MutableBox/TensorBox/StorageBox since
# they're mainly lowering-time constructs that we expect to mutate and such.
@ir_dataclass(frozen=False)
class Subgraph(IRNode):
    name: str
    graph_module: torch.fx.GraphModule
    graph: GraphLowering | None = None


def _has_aliased_buffers(buffers: Sequence[IRNode]) -> bool:
    buffers = [
        buffer.unwrap_view() if isinstance(buffer, ReinterpretView) else buffer
        for buffer in buffers
    ]
    # assuming the same buffer is represented by the same IRNode object
    return len(OrderedSet(id(buffer) for buffer in buffers)) < len(buffers)


@ir_dataclass(frozen=False)
class InvokeSubgraph(ExternKernel):
    """
    Ir node for the invoke_subgraph HOP.
    """

    subgraph: Subgraph | None = None
    operands: Sequence[IRNode] | None = None
    outputs: Sequence[IRNode] | None = None

    def __init__(
        self, subgraph: Subgraph, operands: Sequence[IRNode], layout: MultiOutputLayout
    ) -> None:
        super().__init__(
            name=None,
            layout=layout,
            inputs=operands,
        )
        self.subgraph = subgraph
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

    def get_subgraphs(self) -> list[Subgraph]:
        return [self.subgraph] if self.subgraph else []

    @classmethod
    def create(
        cls, subgraph: Subgraph, *operands: IRNode
    ) -> list[ShapeAsConstantBuffer | NoneAsConstantBuffer | MultiOutput]:
        """For each operand, get a realized input, force it to have the same
        strides as the subgraph inputs, then use an InvokeSubgraph"""
        from .lowering import constrain_to_fake_tensor

        # TODO(anijain2305) - Support sym expr as operands in future.
        current_node = V.graph.current_node

        fake_operands = None
        if eager_input_vals := current_node.meta.get("eager_input_vals"):
            # eager_input_vals is (args_values, kwargs_values). We need args for invoke_subgraph
            offset = 2
            if current_node.target is torch.ops.higher_order.with_effects:
                # Aruguments eagerly are (token, subgraph, identifier, *operands)
                assert current_node.args[1] is torch.ops.higher_order.invoke_subgraph
                offset = 3
            fake_operands = eager_input_vals[0][offset:]
        else:
            offset = 2
            if current_node.target is torch.ops.higher_order.with_effects:
                # with_effects args: (token, invoke_subgraph, subgraph, identifier, *operands)
                assert current_node.args[1] is torch.ops.higher_order.invoke_subgraph
                offset = 4

            # For the partitioned backward graph, we do not have
            # eager_input_vals. Here, we rely on the recorded example values.
            fx_operands = current_node.args[offset:]
            fake_operands = [x.meta["val"] for x in fx_operands]  # type: ignore[union-attr]

        # Realize the inputs. Also intermediates can have different strides than
        # the inputs of the subgraph. So, force the intermediates to have same
        # strides as that of subgraph inputs.
        # pyrefly: ignore [annotation-mismatch, redefinition]
        operands: list[IRNode] = [cls.realize_input(x) for x in operands]
        new_operands: list[IRNode] = []

        for idx, operand in enumerate(operands):
            if isinstance(
                operand, (ShapeAsConstantBuffer, GeneratorState, OpaqueObjectState)
            ):
                new_operands.append(operand)
            else:
                new_operands.append(
                    constrain_to_fake_tensor(operand, fake_operands[idx])
                )

        # pyrefly: ignore [bad-assignment]
        operands = new_operands

        if subgraph.graph is None:
            # create and lower subgraphs
            subgraph.graph = V.graph.make_subgraph(
                gm=subgraph.graph_module,
                example_inputs=fake_operands,
                subgraph_name=subgraph.name,
            )
            with V.set_graph_handler(subgraph.graph):
                subgraph.graph.run(*fake_operands)

        outputs = subgraph.graph.graph_outputs

        # Find the device - operands could be integers from shapes, so we can't
        # use operands[0]
        device = None
        for operand in operands:
            if not isinstance(operand, ShapeAsConstantBuffer):
                device = operand.get_device()
                break
        assert device is not None
        invoke_subgraph = InvokeSubgraph(
            subgraph=subgraph,
            operands=operands,
            layout=MultiOutputLayout(device=device),
        )

        def create_output(
            output: IRNode, ind: int
        ) -> ShapeAsConstantBuffer | NoneAsConstantBuffer | MultiOutput:
            if isinstance(output, (ShapeAsConstantBuffer, NoneAsConstantBuffer)):
                return output
            else:
                device = output.get_device()
                assert device is not None

                return MultiOutput(
                    FixedLayout(
                        device=device,
                        dtype=output.get_dtype(),
                        size=output.get_size(),
                        stride=output.get_stride(),
                        offset=output.get_layout().offset,
                        is_pinned=output.get_layout().is_pinned,
                    ),
                    invoke_subgraph,  # type: ignore[has-type]
                    [(list, ind)],
                    skip_size_stride_alignment_checks=True,
                )

        outs = [create_output(output, i) for i, output in enumerate(outputs)]
        invoke_subgraph.outputs = outs  # type: ignore[assignment]
        return outs

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        wrapper.codegen_invoke_subgraph(self)


@ir_dataclass(frozen=False)
class Conditional(ExternKernel):
    """
    IR node representing torch.cond

    Attributes:
        predicate: A boolean scalar tensor determining which branch to execute.
        operands: Input tensors passed to both true and false subgraphs.
        true_subgraph: Subgraph executed when predicate is True.
        false_subgraph: Subgraph executed when predicate is False.
        outputs: MultiOutput nodes representing the conditional's outputs.
    """

    predicate: IRNode | None = None
    operands: Sequence[IRNode] | None = None
    true_subgraph: Subgraph | None = None
    false_subgraph: Subgraph | None = None
    outputs: Sequence[MultiOutput] | None = None

    def __init__(
        self,
        predicate: IRNode,
        operands: Sequence[IRNode],
        true_subgraph: Subgraph,
        false_subgraph: Subgraph,
        layout: MultiOutputLayout,
        unbacked_bindings: dict[sympy.Symbol, pytree.KeyPath] | None,
    ) -> None:
        self.predicate = predicate
        self.operands = operands
        self.true_subgraph = true_subgraph
        self.false_subgraph = false_subgraph

        sym_args, tensor_args = _split_by_sym_type([predicate, *operands])

        super().__init__(
            name=None,
            layout=layout,
            inputs=tensor_args,
            constant_args=sym_args,
        )
        if unbacked_bindings is not None:
            self.unbacked_bindings = unbacked_bindings

        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

    def get_subgraphs(self) -> list[Subgraph]:
        subgraphs = []
        if self.true_subgraph:
            subgraphs.append(self.true_subgraph)
        if self.false_subgraph:
            subgraphs.append(self.false_subgraph)
        return subgraphs

    @staticmethod
    def _maybe_expr(s: int | torch.SymInt) -> int | sympy.Expr:
        if isinstance(s, int):
            return s
        return s.node.expr

    @classmethod
    def create(
        cls,
        predicate: TensorBox,
        true_fn: Subgraph,
        false_fn: Subgraph,
        operands: list[TensorBox],
    ) -> list[MultiOutput]:
        """Create a Sequence of IRNodes from a conditional statement (see .lowering.cond)"""
        # pyrefly: ignore [bad-assignment]
        predicate = cls.realize_input(predicate)
        # pyrefly: ignore [bad-assignment]
        operands = [cls.realize_input(x) for x in operands]
        fx_operands: Argument = V.graph.current_node.args[-1]

        assert isinstance(fx_operands, Sequence), type(fx_operands)
        # Build fake_operands from FX nodes' metadata
        # For FX Nodes, get the fake tensor from meta["val"]
        # For non-Nodes (e.g., symbolic integers from sym_size lowering), pass directly
        fake_operands: list[Any] = []
        for fx_op in fx_operands:
            if isinstance(fx_op, Node):
                fake_operands.append(fx_op.meta["val"])
            else:
                # Symbolic integer or constant - pass directly
                fake_operands.append(fx_op)
        fake_outputs = V.graph.current_node.meta["val"]

        def _require_exact_strides(
            graph_outputs: Sequence[IRNode],
            fake_tensors: Sequence[torch.Tensor],
        ) -> list[IRNode]:
            ret = []
            for output, fake in zip(graph_outputs, fake_tensors):
                if isinstance(output, ShapeAsConstantBuffer):
                    ret.append(output)
                else:
                    ret.append(
                        # pyrefly: ignore [bad-argument-type]
                        ExternKernel.require_exact_strides(
                            TensorBox(output), fake.stride(), allow_padding=False
                        )
                    )
            # pyrefly: ignore [bad-return]
            return ret

        for subgraph in (true_fn, false_fn):
            if subgraph.graph is None:
                # create and lower subgraphs
                subgraph.graph = V.graph.make_subgraph(
                    gm=subgraph.graph_module,
                    example_inputs=fake_operands,
                    subgraph_name=subgraph.name,
                )
                with V.set_graph_handler(subgraph.graph):
                    subgraph.graph.run(*fake_operands)
                    # Force subgraph outputs to have the expected strides from
                    # FakeTensor metadata. This ensures both branches produce
                    # outputs with consistent strides.
                    subgraph.graph.graph_outputs = _require_exact_strides(
                        subgraph.graph.graph_outputs, fake_outputs
                    )

        assert true_fn.graph is not None
        assert false_fn.graph is not None
        true_outputs = true_fn.graph.graph_outputs
        false_outputs = false_fn.graph.graph_outputs

        for name, outputs in (("true_fn", true_outputs), ("false_fn", false_outputs)):
            if _has_aliased_buffers(true_outputs):
                raise AssertionError(
                    "Output aliasing is currently not supported in compiled torch.cond. "
                    f"The outputs of the {name} subgraph of torch.cond are aliased: {outputs}"
                )

        # make sure true and false outputs are structurally equivalent
        assert len(true_outputs) == len(false_outputs), (true_outputs, false_outputs)
        for i, (t_o, f_o) in enumerate(zip(true_outputs, false_outputs)):
            assert t_o.get_device() == f_o.get_device(), (i, t_o, f_o)
            assert t_o.get_dtype() == f_o.get_dtype(), (i, t_o, f_o)
            assert t_o.get_layout().offset == f_o.get_layout().offset, (i, t_o, f_o)

        # Determine device from operands and predicate
        # The predicate can be on a different device (e.g., CPU for control flow)
        # while the data operands and outputs should be on the compute device, so
        # using predicate device as a fallback.
        device = next(
            o.get_device()
            for o in operands + [predicate]
            if not isinstance(o, ShapeAsConstantBuffer)
        )
        unbacked_bindings = resolve_unbacked_bindings(
            V.graph.sizevars.shape_env,
            V.graph.current_node.meta.get("unbacked_bindings", None),
        )
        assert device is not None, "cannot determine device"
        conditional = Conditional(
            predicate=predicate,
            operands=operands,
            true_subgraph=true_fn,
            false_subgraph=false_fn,
            layout=MultiOutputLayout(device=device),
            unbacked_bindings=unbacked_bindings,
        )

        outputs = [
            MultiOutput(
                FixedLayout(
                    # pyrefly: ignore [bad-argument-type]
                    device=output.get_device()
                    if output.get_device() is not None
                    else device,  # type: ignore[arg-type]
                    dtype=output.get_dtype(),
                    size=[Conditional._maybe_expr(sz) for sz in merged_output.size()],
                    stride=[
                        Conditional._maybe_expr(sz) for sz in merged_output.stride()
                    ],
                    offset=output.get_layout().offset,
                    is_pinned=output.get_layout().is_pinned,
                ),
                conditional,
                [(list, i)],
            )
            # as the true and false outputs are equivalent,
            # we can use either of them here as a "template"
            for i, (output, merged_output) in enumerate(
                zip(true_outputs, V.graph.current_node.meta["val"])
            )
        ]

        conditional.outputs = outputs  # type: ignore[assignment]

        from torch._higher_order_ops.utils import (
            check_input_alias_and_mutation_return_outputs,
        )

        (_, _, _, true_mutated_inputs, _) = (
            check_input_alias_and_mutation_return_outputs(true_fn.graph_module)
        )
        (_, _, _, false_mutated_inputs, _) = (
            check_input_alias_and_mutation_return_outputs(false_fn.graph_module)
        )

        mutated_operand_indices = OrderedSet(true_mutated_inputs) | OrderedSet(
            false_mutated_inputs
        )

        # Create MutationOutput for each mutated operand (for scheduler dependencies)
        conditional.mutation_outputs = [
            MutationOutput(operands[idx].layout, operands[idx], conditional)  # type: ignore[union-attr]
            for idx in sorted(mutated_operand_indices)
        ]

        return outputs

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        wrapper.codegen_conditional(self)
        wrapper.codegen_unbacked_symbol_defs_for_outputs(
            self.get_name(), self.outputs, getattr(self, "unbacked_bindings", {})
        )

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        if unbacked_bindings := getattr(self, "unbacked_bindings", None):
            resolved = resolve_unbacked_bindings(
                V.graph.sizevars.shape_env, unbacked_bindings
            )
            assert resolved is not None
            return OrderedSet(resolved.keys())
        else:
            return OrderedSet()


def _split_by_sym_type(
    args: list[Any],
) -> tuple[list[ShapeAsConstantBuffer], list[Any]]:
    non_sym_args = []
    sym_args = []
    for arg in args:
        if isinstance(arg, ShapeAsConstantBuffer):
            sym_args.append(arg.expr)
        else:
            non_sym_args.append(arg)

    return sym_args, non_sym_args


@ir_dataclass(frozen=False)
class WhileLoop(ExternKernel):
    """The IR node for while_loop and while_loop_stack_output. It supports input mutation."""

    carried_inputs: Sequence[IRNode] | None = None
    additional_inputs: Sequence[IRNode] | None = None
    cond_subgraph: Subgraph | None = None
    body_subgraph: Subgraph | None = None
    outputs: Sequence[MultiOutput] | None = None

    def __init__(
        self,
        carried_inputs: Sequence[IRNode],
        additional_inputs: Sequence[IRNode],
        cond_subgraph: Subgraph,
        body_subgraph: Subgraph,
        layout: MultiOutputLayout,
        unbacked_bindings: dict[sympy.Symbol, pytree.KeyPath] | None,
        stack_output: bool,
    ) -> None:
        self.carried_inputs = carried_inputs
        self.additional_inputs = additional_inputs
        self.cond_subgraph = cond_subgraph
        self.body_subgraph = body_subgraph

        sym_args, tensor_args = _split_by_sym_type(
            [*carried_inputs, *additional_inputs]
        )
        super().__init__(
            name=None,
            layout=layout,
            inputs=tensor_args,
            constant_args=sym_args,
        )
        if unbacked_bindings is not None:
            self.unbacked_bindings = unbacked_bindings
        self.stack_output = stack_output

        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

    def get_subgraphs(self) -> list[Subgraph]:
        subgraphs = []
        if self.cond_subgraph:
            subgraphs.append(self.cond_subgraph)
        if self.body_subgraph:
            subgraphs.append(self.body_subgraph)
        return subgraphs

    # Accidental aliasing can be created due to cse, where the empty buffers we
    # allocated for backward to use gets csed into the same buffer in function fx_graph_cse.
    # See test_scan_multiple_layers_gradient for a concrete example.
    @staticmethod
    def _clone_aliased_inputs(carried_inputs: Sequence[IRNode]) -> Sequence[IRNode]:
        if not _has_aliased_buffers(carried_inputs):
            return carried_inputs

        # Import clone from lowering module

        # Unwrap views to get the underlying buffers for comparison
        unwrapped_buffers = [
            buffer.unwrap_view() if isinstance(buffer, ReinterpretView) else buffer
            for buffer in carried_inputs
        ]

        # Track which buffers we've seen and their indices
        seen_buffers: OrderedSet[int] = OrderedSet()
        result: list[IRNode | TensorBox] = []

        for original_input, unwrapped_buffer in zip(carried_inputs, unwrapped_buffers):
            if id(unwrapped_buffer) in seen_buffers:
                result.append(ExternKernel.copy_input(original_input))
            else:
                seen_buffers.add(id(unwrapped_buffer))
                result.append(original_input)

        return result

    @staticmethod
    def _maybe_wrap_as_tensor_box(out: IRNode) -> IRNode:
        if isinstance(out, TensorBox):
            return out
        elif isinstance(out, (StorageBox, ReinterpretView)):
            return TensorBox(out)
        elif isinstance(out, MultiOutput):
            return TensorBox.create(out)
        else:
            raise RuntimeError(f"NYI unsupported output type: {type(out)}")

    @classmethod
    def create(
        cls,
        cond_fn: Subgraph,
        body_fn: Subgraph,
        carried_inputs: Sequence[IRNode],
        additional_inputs: Sequence[IRNode],
        stack_output: bool,
    ) -> IRNode | Sequence[IRNode]:
        """create the while_loop IR node. stack_output controls whether it stack
        each iterations' output, which is necessary for training.
        """
        from torch._higher_order_ops.utils import check_input_alias_and_mutation

        def _require_exact_strides(
            tensor_boxes: Sequence[IRNode],
            fake_tensors: list[int | torch.SymInt | torch.Tensor],
        ) -> list[IRNode]:
            assert len(tensor_boxes) == len(fake_tensors)
            ret = []
            for tb, fk in zip(tensor_boxes, fake_tensors):
                if isinstance(fk, torch.Tensor):
                    # Subgraph lowering always return StorageBox as graph_outputs because
                    # it realizes the outputs.
                    #
                    # However, require_exact_strides is expecting TensorBox
                    # e.g. in require_exact_strides when an expand happens,
                    # the fake tensor's stride is (0, 0, 0) but the storage
                    # box might have a different stride so lowering.slice_
                    # is used to make the stride consistent and it expects input to
                    # be TensorBox.
                    #
                    # So we wrap the inputs as tensor boxes if they're not yet.
                    new_tb = WhileLoop._maybe_wrap_as_tensor_box(tb)
                    ret.append(
                        ExternKernel.require_exact_strides(
                            new_tb, fk.stride(), allow_padding=False
                        )
                    )
                else:
                    ret.append(tb)
            return ret

        fx_carried_inputs = V.graph.current_node.args[-2]
        fx_additional_inputs = V.graph.current_node.args[-1]
        fx_all_inputs = fx_carried_inputs + fx_additional_inputs  # type: ignore[operator]
        fake_all_inputs = [x.meta["val"] for x in fx_all_inputs]  # type: ignore[union-attr]
        fake_carried_inputs = [x.meta["val"] for x in fx_carried_inputs]  # type: ignore[union-attr]
        fake_additional_inputs = [x.meta["val"] for x in fx_additional_inputs]  # type: ignore[union-attr]

        carried_inputs_ = [cls.realize_input(x) for x in carried_inputs]
        carried_inputs_ = WhileLoop._clone_aliased_inputs(carried_inputs_)
        carried_inputs_ = _require_exact_strides(carried_inputs_, fake_carried_inputs)
        additional_inputs_ = [cls.realize_input(x) for x in additional_inputs]
        additional_inputs_ = _require_exact_strides(
            additional_inputs_, fake_additional_inputs
        )
        all_inputs = carried_inputs_ + additional_inputs_

        for subgraph in (cond_fn, body_fn):
            if subgraph.graph is None:
                # create and lower subgraphs
                assert isinstance(fx_all_inputs, Sequence), type(fx_all_inputs)
                subgraph.graph = V.graph.make_subgraph(
                    gm=subgraph.graph_module,
                    example_inputs=fx_all_inputs,  # type: ignore[arg-type]
                    subgraph_name=subgraph.name,
                )
                with V.set_graph_handler(subgraph.graph):
                    subgraph.graph.run(*fake_all_inputs)
                    # For body_fn, we require its output to have the exact same stride
                    # as inputs because the previous output is the input of next iteration.
                    #
                    # This cannot be automatically done in graph lowering because body_fn's graph outputs
                    # are not user-facing so the special handling for strides of user-facing output in graph
                    # lowering is not applicable.
                    if subgraph is body_fn:
                        assert len(subgraph.graph.graph_outputs) == len(
                            fake_carried_inputs
                        )
                        subgraph.graph.graph_outputs = _require_exact_strides(  # type: ignore[assignment]
                            subgraph.graph.graph_outputs,
                            fake_carried_inputs,
                        )

        assert cond_fn.graph and body_fn.graph
        cond_outputs = cond_fn.graph.graph_outputs
        body_outputs = body_fn.graph.graph_outputs

        if _has_aliased_buffers(body_outputs):
            raise AssertionError(
                "Output aliasing is currently not supported in compiled torch.while_loop. "
                f"The outputs of the body_fn subgraph of torch.while_loop are aliased: {body_outputs}"
            )

        # make sure cond_fn returns a boolean scalar Tensor
        assert len(cond_outputs) == 1, cond_outputs
        p = cond_outputs[0]
        if not isinstance(p, ShapeAsConstantBuffer):
            assert p.get_dtype() == torch.bool, p
            assert len(p.get_size()) == 0, p

        assert len(all_inputs) > 0, (
            "torch.while_loop is assumed to have at least one operand."
        )

        device = all_inputs[0].get_device()

        assert device is not None  # to make linter happy
        # make sure carried_inputs_ and body outputs are structurally equivalent
        assert len(carried_inputs_) == len(body_outputs), (
            carried_inputs_,
            body_outputs,
        )
        for i, (op, bo) in enumerate(zip(carried_inputs_, body_outputs)):

            def _guard_list_equals(
                lhs_exprs: Sequence[int | sympy.Expr],
                rhs_exprs: Sequence[int | sympy.Expr],
            ) -> None:
                assert len(lhs_exprs) == len(rhs_exprs)
                for lhs, rhs in zip(lhs_exprs, rhs_exprs):
                    V.graph.sizevars.check_equals(lhs, rhs)

            _guard_list_equals(op.get_size(), bo.get_size())
            _guard_list_equals(op.get_stride(), bo.get_stride())
            # assume all carried_inputs_ and outputs are on the same device
            # as the MultiOutputLayout below requires single device
            assert op.get_device() == bo.get_device(), (i, op, bo, device)
            assert op.get_dtype() == bo.get_dtype(), (i, op, bo)

        assert device is not None

        unbacked_bindings = resolve_unbacked_bindings(
            V.graph.sizevars.shape_env,
            V.graph.current_node.meta.get("unbacked_bindings", None),
        )

        while_loop = WhileLoop(
            carried_inputs=carried_inputs_,
            additional_inputs=additional_inputs_,
            cond_subgraph=cond_fn,
            body_subgraph=body_fn,
            # asserted above that there is at least one operand
            layout=MultiOutputLayout(device=device),
            unbacked_bindings=unbacked_bindings,
            stack_output=stack_output,
        )

        assert body_fn.graph is not None and isinstance(
            body_fn.graph.module, torch.fx.GraphModule
        )  # to make linter happy

        # Handling input mutations
        mutated_idxs = check_input_alias_and_mutation(
            body_fn.graph.module, fake_all_inputs
        )[3]
        mutated_idx_set = OrderedSet(mutated_idxs)
        mutated_inputs = [all_inputs[idx] for idx in mutated_idx_set]

        # Create all outputs first
        mutated_inputs_iter = iter(mutated_inputs)
        all_outputs: list[IRNode] = []
        while_loop.outputs = []
        while_loop.mutation_outputs = []
        if stack_output:
            assert len(mutated_idx_set) == 0, (
                "NYI: while_loop_stack_output input mutations."
            )
            for idx, output in enumerate(V.graph.current_node.meta["val"]):
                # Create MultiOutput for regular outputs
                multi_out = MultiOutput(
                    FixedLayout(
                        device=output.device,  # type: ignore[arg-type]
                        dtype=output.dtype,
                        size=[Conditional._maybe_expr(sz) for sz in output.size()],
                        stride=[Conditional._maybe_expr(st) for st in output.stride()],
                    ),
                    while_loop,
                    [(list, idx)],
                )
                while_loop.outputs.append(multi_out)
                all_outputs.append(multi_out)
        else:
            for idx, output in enumerate(body_outputs):
                if idx in mutated_idx_set:
                    assert idx < len(carried_inputs), "only carries can be mutated."
                    # Create MutationOutput for mutated inputs
                    mutated_input = next(mutated_inputs_iter)
                    while_loop.mutation_outputs.append(
                        MutationOutput(mutated_input.layout, mutated_input, while_loop)  # type: ignore[attr-defined, union-attr]
                    )
                    all_outputs.append(mutated_input)
                else:
                    multi_out = MultiOutput(
                        FixedLayout(
                            device=output.get_device(),  # type: ignore[arg-type]
                            dtype=output.get_dtype(),
                            size=output.get_size(),
                            stride=output.get_stride(),
                            offset=output.get_layout().offset,
                        ),
                        while_loop,
                        [(list, idx)],
                    )
                    while_loop.outputs.append(multi_out)
                    all_outputs.append(multi_out)

        for inp, out in zip(carried_inputs, all_outputs):
            if inp.get_name() in V.graph.graph_inputs:
                # if a carried input of the while_loop is a graph input,
                # it can be returned as is when the number of iterations
                # is zero. due to this, we can't (generally) reuse the
                # output buffers corresponding to the graph inputs, as
                # the inputs may end up being mutated.
                V.graph.never_reuse_buffers.add(out.get_name())
        return all_outputs

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        wrapper.codegen_while_loop(self, self.stack_output)
        wrapper.codegen_unbacked_symbol_defs_for_outputs(
            self.get_name(), self.outputs, getattr(self, "unbacked_bindings", {})
        )

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        if unbacked_bindings := getattr(self, "unbacked_bindings", None):
            resolved = resolve_unbacked_bindings(
                V.graph.sizevars.shape_env, unbacked_bindings
            )
            assert resolved is not None
            return OrderedSet(resolved.keys())
        else:
            return OrderedSet()


class EffectfulKernel(FallbackKernel):
    def __init__(
        self,
        layout: OutputSpec,
        kernel: _OpOverloads,
        tensor_args: Sequence[IRNode],
        nontensor_args: Sequence[Any],
        unflatten_args: Callable[..., Any],
        kwargs: dict[str, Any] | None = None,
        *,
        unbacked_bindings: dict[sympy.Symbol, pytree.KeyPath] | None = None,
    ) -> None:
        super().__init__(
            layout,
            kernel,
            tensor_args,
            nontensor_args,
            unflatten_args,
            kwargs=None,
            unbacked_bindings=unbacked_bindings,
        )

        from torch._higher_order_ops.effects import _get_effect

        effect_type = _get_effect(kernel)
        assert effect_type is not None
        self.effect_type = effect_type
        self.prev_effect_buffer = V.graph.effectful_ops.get(effect_type, None)
        V.graph.effectful_ops[effect_type] = self

    def get_read_writes(self) -> dependencies.ReadWrites:
        read_writes = super().get_read_writes()

        if self.prev_effect_buffer is not None:
            read_writes.reads.add(
                dependencies.StarDep(self.prev_effect_buffer.get_name())
            )

        return read_writes

    def has_side_effects(self) -> bool:
        return True


class _CollectiveKernel(FallbackKernel):
    def should_allocate(self) -> bool:
        return False

    def has_side_effects(self) -> bool:
        return True

    # This is identical to FallbackKernel.set_cpp_kernel(), minus the
    # part that checks against input aliasing and mutation.
    def set_cpp_kernel_name(self, cpp_kernel_name: str | None = None) -> None:
        assert type(self.op_overload) is torch._ops.OpOverload, (
            "Setting cpp kernel needs a valid op_overload"
        )
        kernel = self.op_overload
        if cpp_kernel_name is not None:
            self.cpp_kernel_name = cpp_kernel_name
        else:
            self.cpp_kernel_name = kernel._schema.name

        self.ordered_kwargs_for_cpp_kernel = [
            x.name for x in kernel._schema.arguments if x.kwarg_only
        ]

    # NOTE: [In-Place Collective Safety]
    # Between the initiation and completion of an in-place collective, the
    # input buffers are subject to both volatile reads and volatile writes.
    # They must not be read, written to or reused by another kernel. To ensure
    # the constraints, we model collective -> wait_tensor as as two-step
    # mutation of the input buffers.
    @classmethod
    def create_inplace(
        cls,
        kernel: _OpOverloads,
        inputs: IRNode | list[IRNode],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        with V.graph.fake_mode:
            (
                _example_output,
                tensor_args,
                non_tensor_args,
                unflatten_args,
                unbacked_bindings,
            ) = cls.process_kernel(kernel, inputs, *args, **kwargs)
        assert not unbacked_bindings, f"{kernel} {unbacked_bindings}"
        for tensor_arg in tensor_args:
            tensor_arg.realize()
            V.graph.mark_buffer_mutated(tensor_arg.get_name())

        device = tensor_args[0].get_device()
        packed = cls(
            NoneLayout(device=device),
            kernel,
            tensor_args,
            non_tensor_args,
            unflatten_args,
        )

        inps = pytree.tree_leaves(inputs)
        packed.mutation_outputs.extend(
            [MutationOutput(NoneLayout(device=device), buf, packed) for buf in inps]
        )

        # For inplace collective ops, the input is guaranteed to be alias of the returned value of op.
        packed.alias_names.extend([inp.get_name() for inp in inps])
        if "out" in kwargs:
            packed.mutation_outputs.append(
                MutationOutput(NoneLayout(device=device), kwargs["out"], packed)
            )
            # For out-variant collective ops, the `out=` arg is guaranteed to be alias of the returned value of op.
            packed.alias_names.append(kwargs["out"].get_name())

    # NOTE: [Out-of-Place Collective Safety]
    # Between the initiation and completion of an out-of-place collective:
    #
    # Input buffers:
    # - Are subject to volatile reads
    # - Can be read by another kernel
    # - Must not be written to or reused by another kernel
    #
    # Output buffers:
    # - Are subject to volatile writes
    # - Must not be read, written to or reused by another kernel
    #
    # To ensure the safety of input buffers without sacrificing read
    # availability, we add input buffers as read deps of wait_tensor kernels.
    #
    # To ensure the safety of output buffers, we model wait_tensor as a
    # mutation to the output buffer. Note we also assumes the user program being
    # correct and the output buffer is not consumed by kernels other than
    # wait_tensor.
    #
    # TODO(yifu): add a pre-grad pass to validate the correctness of collective
    # usage in the user program.
    @classmethod
    def create_out_of_place(
        cls,
        kernel: _OpOverloads,
        inputs: TensorBox | list[TensorBox],
        *args: Any,
        **kwargs: Any,
    ) -> list[MultiOutput] | _CollectiveKernel:
        with V.graph.fake_mode:
            (
                example_output,
                tensor_args,
                non_tensor_args,
                unflatten_args,
                unbacked_bindings,
            ) = cls.process_kernel(kernel, inputs, *args, **kwargs)
        assert not unbacked_bindings, f"{kernel}, {unbacked_bindings}"
        for tensor_arg in tensor_args:
            if not isinstance(tensor_arg, TorchBindObject):
                tensor_arg.realize()

        if isinstance(example_output, list):
            device = cls.find_device(tensor_args, example_output)
            assert device is not None
            packed = cls(
                MultiOutputLayout(device=device),
                kernel,
                tensor_args,
                non_tensor_args,
                unflatten_args,
            )
            packed.outputs = [
                MultiOutput(
                    cls.tensor_to_layout(tensor),
                    packed,
                    [(list, i)],
                )
                for i, tensor in enumerate(example_output)
            ]
            for buf, tensor in zip(packed.outputs, example_output):
                if config.assume_unaligned_fallback_output or not tensor_is_aligned(
                    tensor
                ):
                    V.graph.unaligned_buffers.add(buf.name)  # type: ignore[arg-type]
            return packed.outputs
        else:
            packed = cls(
                cls.tensor_to_layout(example_output),
                kernel,
                tensor_args,
                non_tensor_args,
                unflatten_args,
            )
            if config.assume_unaligned_fallback_output or not tensor_is_aligned(
                example_output
            ):
                V.graph.unaligned_buffers.add(packed.name)  # type: ignore[arg-type]
            packed.outputs = [packed]
            return packed


class _AllReduce_Kernel(_CollectiveKernel):
    def __init__(
        self,
        layout: OutputSpec,
        kernel: _OpOverloads,
        tensor_args: Sequence[IRNode],
        nontensor_args: Sequence[Any],
        unflatten_args: Callable[..., Any],
        kwargs: dict[str, Any] | None = None,
        *,
        unbacked_bindings: dict[sympy.Symbol, pytree.KeyPath] | None = None,
    ) -> None:
        super().__init__(
            layout,
            kernel,
            tensor_args,
            nontensor_args,
            unflatten_args,
            kwargs=None,
            unbacked_bindings=unbacked_bindings,
        )
        self.set_cpp_kernel_name("aoti_torch_cpu__c10d_functional_all_reduce_")

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        wrapper.include_extra_header("torch/csrc/inductor/aoti_torch/c/shim_cpu.h")
        wrapper.generate_extern_kernel_alloc(self)

        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)


class _AllReduceKernel(_CollectiveKernel):
    def __init__(
        self,
        layout: OutputSpec,
        kernel: _OpOverloads,
        tensor_args: Sequence[IRNode],
        nontensor_args: Sequence[Any],
        unflatten_args: Callable[..., Any],
        kwargs: dict[str, Any] | None = None,
        *,
        unbacked_bindings: dict[sympy.Symbol, pytree.KeyPath] | None = None,
    ) -> None:
        super().__init__(
            layout,
            kernel,
            tensor_args,
            nontensor_args,
            unflatten_args,
            kwargs=None,
            unbacked_bindings=unbacked_bindings,
        )
        self.set_cpp_kernel_name("aoti_torch_cpu__c10d_functional_all_reduce")

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        wrapper.include_extra_header("torch/csrc/inductor/aoti_torch/c/shim_cpu.h")
        wrapper.generate_extern_kernel_alloc(self)

        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)


class _WaitKernel(_CollectiveKernel):
    def __init__(
        self,
        layout: OutputSpec,
        kernel: _OpOverloads,
        tensor_args: Sequence[IRNode],
        nontensor_args: Sequence[Any],
        unflatten_args: Callable[..., Any],
        kwargs: dict[str, Any] | None = None,
        *,
        unbacked_bindings: dict[sympy.Symbol, pytree.KeyPath] | None = None,
    ) -> None:
        super().__init__(
            layout,
            kernel,
            tensor_args,
            nontensor_args,
            unflatten_args,
            kwargs=None,
            unbacked_bindings=unbacked_bindings,
        )
        self.set_cpp_kernel_name("aoti_torch_cpu__c10d_functional_wait_tensor")

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        wrapper.include_extra_header("torch/csrc/inductor/aoti_torch/c/shim_cpu.h")
        wrapper.generate_extern_kernel_alloc(self)

        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    def get_volatile_reads(self) -> Sequence[IRNode]:
        inp = self.inputs[0]
        assert isinstance(inp, IRNode)
        if isinstance(inp, _CollectiveKernel):
            # Out-of-place single-output
            i = inp.inputs[0]
            assert isinstance(i, IRNode), type(i)
            return [i]
        elif isinstance(inp, MultiOutput):
            # This can be two things:
            # 1. Out-of-place multi-output coll
            # 2. In-place coll with inputs coming from another MultiOutput
            coll = inp.inputs[0]
            # Case 1
            if isinstance(coll, _CollectiveKernel):
                _, idx = inp.indices[0]

                return [coll.inputs[idx]]
            # Case 2
            return []
        else:
            # In-place requires no additional deps handling for volatile
            # reads since the inputs are mutated.
            return []

    @classmethod
    def create_wait(cls, kernel: _OpOverloads, inp: TensorBox) -> None:
        with V.graph.fake_mode:
            (
                _example_output,
                tensor_args,
                non_tensor_args,
                unflatten_args,
                unbacked_bindings,
            ) = cls.process_kernel(kernel, inp)
        assert not unbacked_bindings, f"{kernel} {unbacked_bindings}"
        packed = cls(
            NoneLayout(device=inp.get_device()),
            kernel,
            tensor_args,
            non_tensor_args,
            unflatten_args,
        )
        packed.mutation_outputs.append(
            MutationOutput(NoneLayout(device=inp.get_device()), inp, packed)
        )

    def get_read_writes(self) -> dependencies.ReadWrites:
        read_writes = super().get_read_writes()
        # See [Out-of-Place Collective Safety].
        volatile_reads = self.get_volatile_reads()
        for vr in volatile_reads:
            read_writes.reads.add(dependencies.StarDep(vr.get_name()))
        return read_writes


# NB: recursive structure here reflects val_to_arg_str, avoid
# calling free_unbacked_symbols on "exotic" types that don't get pexpr
# treatment
def maybe_free_unbacked_symbols(s: object) -> OrderedSet[Symbol]:
    if isinstance(s, (SymTypes, Expr)):
        # This branch should be impossible in return position
        return free_unbacked_symbols(s)
    elif isinstance(s, (tuple, list)):
        r = OrderedSet[sympy.Symbol]()
        for t in s:
            r |= maybe_free_unbacked_symbols(t)
        return r
    elif isinstance(s, torch.Tensor):
        # This branch is impossible in constant-args position
        return free_unbacked_symbols(s)
    else:
        return OrderedSet()


def maybe_free_symbols(s: object) -> OrderedSet[Symbol]:
    if isinstance(s, (SymTypes, Expr)):
        # This branch should be impossible in return position
        return free_symbols(s)
    elif isinstance(s, (tuple, list)):
        r = OrderedSet[sympy.Symbol]()
        for t in s:
            r |= maybe_free_symbols(t)
        return r
    elif isinstance(s, torch.Tensor):
        # This branch is impossible in constant-args position
        return free_symbols(s)
    else:
        return OrderedSet()


def assign_origin_node(result: Any, n: torch.fx.Node) -> None:
    # This is not complete, but it doesn't have to be: origin_node
    # tracking is best effort.  The logic here critically relies on direct
    # TensorBox -> StorageBox denoting a non-view; we don't bother trying
    # to get views to work.  Feel free to add any extra cases as needed.
    #
    # Note: we can't YOLO tree_map over this result, because if there are
    # buffers or a view involved, we might not be able to validly assign
    # the origin_node here.
    if isinstance(result, TensorBox) and isinstance(result.data, StorageBox):
        if isinstance(result.data.data, Loops):
            result.data.data._post_init_setattr("origin_node", n)
        elif isinstance(result.data.data, Buffer):
            result.data.data._post_init_setattr("origin_node", n)
            if isinstance(result.data.data, ComputedBuffer) and isinstance(
                result.data.data.data, Loops
            ):
                result.data.data.data._post_init_setattr("origin_node", n)
            # Not really multi-output, can straightforwardly recurse in
            elif (
                isinstance(result.data.data, MultiOutput)
                and not result.data.data.indices
            ):
                if isinstance(result.data.data.inputs[0], Buffer):
                    result.data.data.inputs[0]._post_init_setattr("origin_node", n)
