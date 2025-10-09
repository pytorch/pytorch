# mypy: allow-untyped-defs
import contextlib
import dataclasses
import functools
import hashlib
import inspect
import itertools
import json
import logging
import math
import operator
import os
import re
import sys
import textwrap
import time
from collections.abc import Sequence
from concurrent.futures import as_completed, ThreadPoolExecutor
from io import StringIO
from types import ModuleType
from typing import Any, Callable, NamedTuple, Optional, TYPE_CHECKING, Union
from typing_extensions import Self
from unittest.mock import patch

import sympy

import torch
import torch._inductor.async_compile  # noqa: F401 required to warm up AsyncCompile pools
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import (
    counters,
    dynamo_timed,
    get_chromium_event_logger,
    identity,
    preserve_rng_state,
)
from torch._inductor.await_utils import await_sync
from torch._inductor.utils import clear_on_fresh_cache
from torch.utils._filelock import FileLock
from torch.utils._ordered_set import OrderedSet

from ..utils._sympy.functions import CeilDiv
from . import config, ir
from .autotune_process import (
    TensorMeta,
    TritonBenchmarkRequest,
    TritonCPUBenchmarkRequest,
    TritonGPUBenchmarkRequest,
)
from .codecache import code_hash, PersistentCache, PyCodeCache
from .codegen.common import (
    CSEVariable,
    IndentedBuffer,
    KernelTemplate,
    OpOverrides,
    WorkspaceArg,
    WorkspaceZeroMode,
)
from .codegen.simd_kernel_features import SIMDKernelFeatures
from .codegen.subgraph import SubgraphChoiceCaller
from .codegen.triton import (
    gen_common_triton_imports,
    texpr,
    TMACompatibilityChecker,
    TritonKernel,
    TritonScheduling,
)
from .codegen.triton_utils import config_of, equal_1_arg_indices, signature_to_meta
from .codegen.wrapper import pexpr
from .exc import CUDACompileError
from .fx_utils import count_flops_fx
from .ir import ChoiceCaller, PrimitiveInfoType
from .ops_handler import StoreMode
from .runtime.benchmarking import benchmarker
from .runtime.hints import DeviceProperties
from .runtime.triton_compat import HAS_WARP_SPEC
from .runtime.triton_heuristics import FixedGrid
from .utils import (
    ceildiv,
    do_bench_using_profiling,
    FakeIndentedBuffer,
    get_dtype_size,
    is_gpu,
    Placeholder,
    restore_stdout_stderr,
    sympy_dot,
    sympy_index_symbol,
    sympy_product,
    triton_type,
    triton_type_to_torch,
    unique,
)
from .virtualized import V


log = logging.getLogger(__name__)

# correctness checks struggle with fp16/tf32
VERIFY: dict[str, Any] = {}
PRINT_AUTOTUNE = True
DEBUG = False


if TYPE_CHECKING:
    import concurrent

    from torch._inductor.codegen.simd import IterationRangesEntry, IterationRangesRoot

    from .codegen.common import CSE


class KernelNamespace:
    pass


# these objects are imported from the generated wrapper code
extern_kernels = KernelNamespace()


@dataclasses.dataclass
class BenchmarkTensors:
    """Represents a set of inputs and outputs for autotuning with a template"""

    input_tensors: list[torch.Tensor]
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
        cls,
        example_inputs: list[torch.Tensor],
        example_inputs_extern: list[torch.Tensor],
        out: torch.Tensor,
        out_extern: torch.Tensor,
        expected: Optional[torch.Tensor] = None,
    ) -> Self:
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

    HookFn = Callable[[], str]

    def __init__(
        self, code: str, replacement_hooks: dict[str, Optional[HookFn]]
    ) -> None:
        super().__init__()
        self._code: str = code
        self.replacement_hooks: dict[str, Optional[PartialRender.HookFn]] = (
            replacement_hooks
        )

    @property
    def code(self) -> str:
        """
        The fully rendered code. Will **error** if any hooks have yet to be
        finalized.
        """
        remaining_active_hooks = [
            key for key, fn in self.replacement_hooks.items() if fn is not None
        ]
        assert len(remaining_active_hooks) == 0, (
            f"The following hooks have not yet been finalized:\n {remaining_active_hooks=}"
        )
        return self._code

    def finalize_hook(self, hook_key: str, strict: bool = True) -> None:
        """
        Finalize a hook by name.

        :param strict: If ``True``, raise an error if the hook wasn't found.

        NOTE: Will **error** if the hook has already been finalized.
        """
        if hook_key not in self.replacement_hooks:
            if strict:
                raise RuntimeError(
                    f"{hook_key} not registered in self.replacement_hooks"
                )
            else:
                return

        hook = self.replacement_hooks[hook_key]
        assert hook is not None, f"Hook key {hook_key} can only be called once"
        self._code = self._code.replace(hook_key, hook())

        self.replacement_hooks[hook_key] = None

    def finalize_remaining(self) -> str:
        """
        Finalize the remaining active hooks. This function can be used in cases
        where the caller uses `finalize_hook` rather than `finalize_all`.
        Note: `finalize_all` errors if a hook that has already been finalized
        is attempted to be called again. This function only attempts to
        finalize active hooks.
        """
        for key, fn in self.replacement_hooks.items():
            if fn is not None:
                self.finalize_hook(key)
        return self.code

    def finalize_all(self) -> str:
        """
        Finalize all active hooks.

        NOTE: unlike ``finalize_remaining``, this method will **error** if any
        hook has already been finalized.
        """
        for key in self.replacement_hooks:
            self.finalize_hook(key)
        return self.code


# This is used to store info needed for lowering each subgraph in triton
# templates


@dataclasses.dataclass()
class SubgraphInfo:
    body: IndentedBuffer
    template_mask: Optional[str] = None
    template_out_shape: Optional[Union[str, tuple[str]]] = None
    compute: IndentedBuffer = dataclasses.field(default_factory=IndentedBuffer)
    indexing_code: IndentedBuffer = dataclasses.field(default_factory=IndentedBuffer)
    loads: IndentedBuffer = dataclasses.field(default_factory=IndentedBuffer)
    stores: IndentedBuffer = dataclasses.field(default_factory=IndentedBuffer)
    ops_handler: Optional[V.WrapperHandler] = None  # type: ignore[name-defined]
    cse: Optional["CSE[Any]"] = None

    # only copied over if not None
    range_trees: Optional[list["IterationRangesRoot"]] = None
    range_tree_nodes: Optional[dict[sympy.Symbol, "IterationRangesEntry"]] = None
    numels: Optional[dict[str, sympy.Expr]] = None

    def __post_init__(self):
        self.only_copy_if_non_none_fields = (
            "range_trees",
            "range_tree_nodes",
            "numels",
            "cse",
        )

    def to_dict(self):
        return {
            field.name: getattr(self, field.name) for field in dataclasses.fields(self)
        }


class ModificationWrapper(V.WrapperHandler):  # type: ignore[name-defined]
    """Handles placeholder substitutions during subgraph processing."""

    def __init__(
        self,
        kernel,
        subgraph_number: int,
        fixed_inputs: dict[str, Any],
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
            buffer = V.graph.get_buffer(name)
            var_dtype = buffer.dtype
            line = f"tl.load({var} + {index_str})"

            if (
                var_dtype in (torch.float16, torch.bfloat16)
                and config.triton.codegen_upcast_to_fp32
            ):
                line += ".to(tl.float32)"
                var_dtype = torch.float32

            out = self.kernel.cse.generate(
                self.kernel.compute, line, dtype=var_dtype, shape=()
            )
            return out

        return self.kernel.cse.generate(
            self.kernel.compute,
            f"({self.fixed_inputs[name]})",
            dtype=torch.float32,
            shape=(),
        )

    def indirect_indexing(self, index_var: str, size, check, wrap_neg=True):
        """Convert index variable to symbolic form."""
        return sympy_index_symbol(str(index_var))

    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
    ) -> str:
        """Currently only supports stores for atomic adds coming from scatter nodes
        This is used by flex_attention's backwards grad for captured buffers, see
        zeros_and_scatter lowering
        """
        assert self.mask is not None, (
            "Mask is required for inner stores in modifications"
        )
        assert mode == "atomic_add", "Only atomic_add is supported for inner stores"

        buf_name = self._add_kernel_input(name)
        index_str = self._process_indexing(index)
        index_str = f"tl.broadcast_to({index_str}, {value}.shape)"
        store = f"tl.atomic_add({buf_name} + {index_str}, {value}, {self.mask}, sem='relaxed')"
        return store

    def _add_kernel_input(self, name: str):
        """Add name as input to kernel and return input ref."""
        return self.kernel.args.input(name)

    def _process_indexing(self, index):
        """Process and rename indexing, adding symbols as kernel inputs."""
        return self.kernel.kexpr(self.kernel.rename_indexing(index))


# Function name, followed by args and kwargs.
RecordedEventsType = list[tuple[str, list[Any], dict[str, Any]]]


class TritonTemplateKernel(TritonKernel):
    """
    A specialized kernel class for Triton templates that handles code generation
    for templated Triton kernels.

    This class extends TritonKernel to provide additional functionality for
    template-based kernel generation, including support for subgraphs, workspace
    arguments, and prologue/epilogue fusion.
    """

    def __init__(
        self,
        kernel_name,
        input_nodes: tuple[ir.IRNode],
        output_node,
        defines,
        num_stages,
        num_warps,
        grid_fn,
        meta,
        call_sizes,
        num_consumer_groups=0,
        num_buffers_warp_spec=0,
        use_jit=False,
        tma_store=False,
        prefix_args=0,
        suffix_args=0,
        epilogue_fn=identity,
        subgraphs: Optional[list[ir.ComputedBuffer]] = None,
        workspace_arg: Optional[WorkspaceArg] = None,
        prologue_loads_all_inputs=False,
        hint_override: Optional[int] = None,
    ) -> None:
        if tma_store:
            pass
        numel = sympy_product(output_node.get_size())
        if tma_store:
            assert len(output_node.get_size()) == 2, (
                "TMA store only supported for 2D with templates"
            )
            tiling = {
                "x": output_node.get_size()[0],
                "y": output_node.get_size()[1],
                "r0_": sympy.S.One,
            }
        else:
            tiling = {
                "x": numel,
                "r0_": sympy.S.One,
            }
        super().__init__(
            tiling,
            features=SIMDKernelFeatures([], numel),
            hint_override=hint_override,
        )
        self.input_nodes = input_nodes
        self.output_node = output_node
        self.named_input_nodes = {}  # type: ignore[var-annotated]
        self.defines = defines
        self.kernel_name = kernel_name
        self.use_jit = use_jit
        self.tma_store = tma_store
        self.num_stages = num_stages
        self.num_warps = num_warps
        self.num_consumer_groups = num_consumer_groups
        self.num_buffers_warp_spec = num_buffers_warp_spec
        self.grid_fn = grid_fn
        self.meta = meta
        self.call_sizes = call_sizes
        # for templates with fixed epilogues
        self.prefix_args = prefix_args
        self.suffix_args = suffix_args
        self.epilogue_fn = epilogue_fn
        self.render_hooks = {}  # type: ignore[var-annotated]
        self.triton_meta: Optional[dict[str, object]] = None
        # For Templated Attention this can be a list of ir.Subgraph
        self.subgraphs: Optional[list[ir.ComputedBuffer]] = subgraphs

        # Some templates use extra global memory as a workspace
        self.workspace_arg = workspace_arg
        if workspace_arg is not None:
            self.args.workspace_args.append(workspace_arg)

        # The following attributes (body, template_mask, output_val) are all
        # used for triton kernel codegen.
        # They are swapped onto the TritonTemplateKernel object by
        # `set_subgraph_body`
        self.subgraph_bodies: dict[str, SubgraphInfo] = {}

        # input buffers which we are allowed to prologue fuse into
        self.prologue_supported_inputs: OrderedSet[str] = OrderedSet()

        # input buffers which we are fusing into
        self.prologue_fused_inputs: OrderedSet[str] = OrderedSet()
        # input buffers which we are fusing into, which preserve a zero mask
        self.prologue_fused_inputs_preserve_zero: OrderedSet[str] = OrderedSet()

        # The following attributes are all used for triton kernel codegen.
        # They are swapped onto the TritonTemplateKernel object by
        # `set_subgraph_body`
        # NB: the names here must match the fields in SubgraphInfo
        self.body: IndentedBuffer = FakeIndentedBuffer()
        self.compute: IndentedBuffer = FakeIndentedBuffer()
        self.indexing_code: IndentedBuffer = FakeIndentedBuffer()
        self.loads: IndentedBuffer = FakeIndentedBuffer()
        self.stores: IndentedBuffer = FakeIndentedBuffer()
        self.template_mask: Optional[str] = None
        self.template_out_shape: Optional[Union[str, tuple[str]]] = None
        self.ops_handler: Optional[V.WrapperHandler] = None  # type: ignore[name-defined]

        # When caching is enabled, the generated code is not dependent on the input nodes names, or
        # symbolic sizes names.
        # However, some of the variables returned by generate_and_load that are computed during the
        # triton template expansions (code generation) are dependent on those.
        # In order to cache the code generation and avoid redoing it for similar inputs that varies only by
        # input names or symbol names, we do a record and replay method.
        # During template expansions we record all function calls that change input_dependent_preserved_state
        # and replay them on a cache hit to regenerate them.
        self.cached_replay_events: Optional[RecordedEventsType] = None

        # Update each time an input is marked frozen, used to replay the freezing of inputs on a cache hit.
        self.frozen_layouts_cnt = 0

        # When prologue_loads_all_inputs is true, prologue_supported_inputs is populated during def_kernel
        # by adding all inputs.
        self.prologue_loads_all_inputs = prologue_loads_all_inputs

        # Extra functions to be exposed during partial template rendering.
        self.extra_template_env_fns: list[Callable[..., Any]] = []

        # Tracking for intermediate variables
        self.tmp_var_ctr = itertools.count()

    def _gen_tmp_var(self) -> str:
        return f"_tmp_var{next(self.tmp_var_ctr)}"

    def input_dependent_preserved_state(self) -> str:
        # Not adding self.args.output_buffers on purpose. But we do not need to reproduce it on a cache hit.
        # (never accessed).
        return repr(
            [
                self.args.input_buffers,
                self.args.sizevars,
                self.args.workspace_args,
                self.prologue_supported_inputs,
                self.frozen_layouts_cnt,
            ]
        )

    def record_input_dependent_tracked_event(self) -> Callable[..., Any]:
        def decorator(fn) -> Callable[..., Any]:
            def wrapper(*args, **kwargs) -> Any:
                pre_state = self.input_dependent_preserved_state()
                result = fn(*args, **kwargs)
                post_state = self.input_dependent_preserved_state()
                if pre_state != post_state:
                    assert self.cached_replay_events is not None
                    self.cached_replay_events.append((fn.__name__, [*args], {**kwargs}))
                return result

            return wrapper

        return decorator

    def replay_cached_events(self, events: RecordedEventsType) -> None:
        for f, args, kwargs in events:
            getattr(self, f)(*args, **kwargs)

    @contextlib.contextmanager
    def set_subgraph_body(self, body_name: str):
        assert all(
            hasattr(self, field.name) for field in dataclasses.fields(SubgraphInfo)
        )
        old_state = {
            key.name: getattr(self, key.name)
            for key in dataclasses.fields(SubgraphInfo)
        }

        assert body_name in self.subgraph_bodies, body_name

        subgraph = self.subgraph_bodies[body_name]
        for key, value in subgraph.to_dict().items():
            if value is None and key in subgraph.only_copy_if_non_none_fields:
                continue
            setattr(self, key, value)

        context = (
            contextlib.nullcontext
            if not self.ops_handler
            else lambda: V.set_ops_handler(self.ops_handler(V.get_ops_handler()))
        )
        with context():  # type: ignore[operator]
            yield
        self.subgraph_bodies[body_name] = SubgraphInfo(
            **{
                key.name: getattr(self, key.name)
                for key in dataclasses.fields(SubgraphInfo)
            }
        )
        for key, value in old_state.items():
            setattr(self, key, value)

    @contextlib.contextmanager
    def create_subgraph_body(self, body_name: str, clear_cse: bool = False):
        assert body_name not in self.subgraph_bodies
        self.subgraph_bodies[body_name] = SubgraphInfo(
            IndentedBuffer(), None, None, cse=self.cse.clone() if clear_cse else None
        )
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
            size = V.graph.sizevars.size_hints(inp.get_size(), fallback=0)
            numel = functools.reduce(operator.mul, size, 1)
            dtype_size = get_dtype_size(inp.get_dtype())
            num_bytes.append(numel * dtype_size * (1 + int(i < ninplace_args)))
        return sum(num_bytes)

    def estimate_flops(self) -> int:
        for node in self.input_nodes:
            for fx_node in node._current_origins:
                f = count_flops_fx(fx_node)
                if f is not None:
                    return V.graph.sizevars.size_hint(f, fallback=0)
        return 0

    def jit_lines(self):
        if self.use_jit:
            return "@triton.jit"

        argdefs, _, signature, _ = self.args.python_argdefs()
        triton_meta: dict[str, Any] = {
            "signature": signature_to_meta(
                signature,
                size_dtype=self.index_dtype,
                argdefs=argdefs,
                is_template=True,
            ),
            "device": DeviceProperties.create(self.output_node.get_device()),
            "constants": {},
        }
        triton_meta["configs"] = [config_of(signature)]
        for arg_num in equal_1_arg_indices(signature):  # type: ignore[index]
            triton_meta["constants"][signature[arg_num].name] = 1  # type: ignore[index,union-attr]
        matrix_instr_nonkdim = self.meta.get("matrix_instr_nonkdim", None)
        waves_per_eu = self.meta.get("waves_per_eu", None)
        kpack = self.meta.get("kpack", None)
        if matrix_instr_nonkdim:
            triton_meta["matrix_instr_nonkdim"] = matrix_instr_nonkdim
        if waves_per_eu:
            triton_meta["waves_per_eu"] = waves_per_eu
        if kpack:
            triton_meta["kpack"] = kpack

        self.triton_meta = triton_meta

        inductor_meta = {
            "kernel_name": str(Placeholder.DESCRIPTIVE_NAME),
            **self.inductor_meta_common(),
            **FixedGrid.setup_grid_as_args(),
        }
        if config.profile_bandwidth or config.benchmark_kernel:
            num_gb = self.estimate_kernel_num_bytes() / 1e9
            inductor_meta["kernel_num_gb"] = num_gb
        if config.benchmark_kernel:
            flops = self.estimate_flops()
            inductor_meta["kernel_flop"] = flops

        inductor_meta["config_args"] = self.meta

        template_args = f"""
            num_stages={self.num_stages},
            num_warps={self.num_warps},
            triton_meta={triton_meta!r},
            inductor_meta={inductor_meta!r},
        """

        if HAS_WARP_SPEC:
            template_args += f"""
            num_consumer_groups={self.num_consumer_groups},
            num_buffers_warp_spec={self.num_buffers_warp_spec},
        """

        return f"""
            @triton_heuristics.template(
                {template_args}
            )
            @triton.jit
        """

    def gen_argdefs(self):
        def hook():
            # python_argdefs() cannot be run until after the rest of the template lazily adds more args
            arg_defs, *_ = self.args.python_argdefs()
            return f"{', '.join(x.full_name() for x in arg_defs)}"

        return self._register_hook("<ARGDEFS>", hook, allow_overwriting=True)

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
            if input_node.get_name() in V.graph.removed_buffers:
                continue
            if input_node.get_name() in self.prologue_fused_inputs:
                continue

            self.args.input_buffers[input_node.get_name()] = arg_name

        # The args may be duplicated, so renaming must be after args are de-duplicated.
        for name in argnames:
            input_node = self.named_input_nodes[name]
            if self.prologue_loads_all_inputs:
                self.prologue_supported_inputs.add(input_node.get_name())
            if input_node.get_name() in V.graph.removed_buffers:
                continue
            if input_node.get_name() in self.prologue_fused_inputs:
                continue

            arg_name = self.args.input_buffers[input_node.get_name()]
            if input_node.get_layout().offset == 0:
                renames.writeline(f"{name} = {arg_name}")
            else:
                offset = texpr(self.rename_indexing(input_node.get_layout().offset))
                renames.writeline(f"{name} = {arg_name} + {offset}")

        for input_node in self.input_nodes[len(self.input_nodes) - self.suffix_args :]:
            # get args in correct order
            if input_node.get_name() in V.graph.removed_buffers:
                continue
            if input_node.get_name() in self.prologue_fused_inputs:
                continue

            self.args.input(input_node.get_name())

        def hook():
            # python_argdefs() cannot be run until after the rest of the template lazily adds more args
            arg_defs, *_ = self.args.python_argdefs()
            code = IndentedBuffer()
            code.splice(gen_common_triton_imports())
            code.splice(self.jit_lines())
            code.writeline(
                f"def {self.kernel_name}({', '.join(x.full_name() for x in arg_defs)}):"
            )
            with code.indent():
                code.splice(self.defines)
                code.splice(renames.getvalue())
                self.codegen_prologue(code)
            return code.getvalue()

        return self._register_hook("<DEF_KERNEL>", hook)

    def size(self, name: Optional[str], index: int):
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
        assert subgraph_number < len(self.subgraphs), (
            f"Invalid subgraph number provided to create_modification, {subgraph_number} must be < {len(self.subgraphs)}"
        )
        assert self.body.getvalue() == "", (
            "Body should be clear before adding a modification"
        )
        return self.subgraphs[subgraph_number]

    def _handle_scatter_graph(self, scatter_graph):
        """Handle processing for a single scatter graph.

        Args:
            scatter_graph: The scatter graph to process
        """
        assert isinstance(scatter_graph, ir.ComputedBuffer), (
            f"scatter_graph must be an instance of ComputeBuffer but got {type(scatter_graph)}"
        )

        def contiguous_strides(x):
            # We always create a fresh contiguous grad for scattering into
            return sum(
                x_i * stride for x_i, stride in zip(x, scatter_graph.get_stride())
            )

        return scatter_graph.data.store_output(  # type: ignore[attr-defined]
            scatter_graph.name, contiguous_strides, []
        )

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
        scatters = []
        while f"mod_{subgraph_number}_{num}" in self.subgraph_bodies:
            num += 1
        with self.create_subgraph_body(f"mod_{subgraph_number}_{num}"):
            subgraph = self._get_subgraph(subgraph_number)
            modification_handler = ModificationWrapper(
                self, subgraph_number, fixed_inputs, mask
            )
            with V.set_ops_handler(modification_handler):
                assert isinstance(subgraph, (ir.ComputedBuffer, list)), (
                    f"Expected the subgraph to be a ComputedBuffer or a List[ComputedBuffer], got {type(subgraph)}"
                )
                # Handle scatter stores
                if isinstance(subgraph, list):
                    for scatter_graph in subgraph:
                        scatters.append(self._handle_scatter_graph(scatter_graph))
                elif isinstance(subgraph.data, ir.InputBuffer):
                    out = subgraph.data.make_loader()(())
                else:
                    out = subgraph.data.inner_fn(())

            self.codegen_body()
            if output_name is not None:
                assert isinstance(output_name, str)
                assert out is not None
                self.body.writeline(f"{output_name} = {out.value}")
            else:
                assert out is None
                for scatter in scatters:
                    self.body.writeline(str(scatter))

            body_val = self.body.getvalue()
            self.cse.invalidate(OrderedSet())
            return body_val

    def load_input(
        self,
        input_name: str,
        output_name: str,
        indices: Union[list[Any], tuple[Any]],
        mask: Optional[str] = None,
        other: Optional[Union[float, int]] = 0.0,
        indent_width: int = 4,
        index_shape: Optional[tuple[str]] = None,
    ):
        """Loads an input and applies any necessary preprocessing or masking.

        Args:
            input_name (str): The name of the input to load.
            indices (Union[List, Tuple]): The index for each dimension of the input.
            val (str): The name of the variable to store the loaded value.
            mask (Optional[str]): An optional mask to use for the load operation.
            other (Optional[Union[float, int]]): The value to use for masked elements. Default is 0.0.
            indent_width (int): The number of spaces to use for indentation.
        """

        input_node = self.named_input_nodes[input_name]
        if not self.prologue_loads_all_inputs:
            self.prologue_supported_inputs.add(input_node.get_name())

        tilings = (sympy_product(input_node.get_size()), sympy.Integer(1))
        groups = {
            "x": tilings[0],
            "r0_": tilings[1],
        }

        range_trees = self.construct_range_trees(
            pid_cache=None,
            inside_reduction=False,
            is_reduction=False,
            numels=groups,
            no_x_dim=False,
        )
        load_code = None

        with self.create_subgraph_body(f"<LOAD_INPUT_{input_name}>"):
            assert isinstance(indices, (list, tuple))
            assert isinstance(output_name, str)
            assert isinstance(mask, (str, type(None)))
            self.range_trees = range_trees
            self.numels = {k: V.graph.sizevars.simplify(v) for k, v in groups.items()}
            indices = list(map(OpOverrides.paren, indices))
            index_symbols = [sympy.Symbol(x, integer=True) for x in indices]

            lengths = [V.graph.sizevars.simplify(s) for s in input_node.get_size()]
            assert len(indices) == len(lengths)

            index_symbols = [sympy.Symbol(x, integer=True) for x in indices]
            assert len(indices) == len(lengths)

            # glue to make generated code use same indexing from template

            # TODO (from reviewers as well)
            # in codegen_template,
            # prologue_node.codegen(kernel.split_and_set_ranges(prologue_node.get_ranges()))
            # the ranges need to reflect the group of the prologue input or it will error
            # not sure if there is any difference between original range_tree_entry in
            # and new one from correct lengths/groups... both actually seem to work
            for name, range_tree_entry in zip(
                indices, self.range_trees[0].construct_entries(lengths)
            ):
                range_tree_entry.set_name(name)
            contiguous_index = sympy_dot(
                ir.FlexibleLayout.contiguous_strides(lengths), index_symbols
            )
            contiguous_index = self.rename_indexing(contiguous_index)
            self.body.writeline("xindex = " + texpr(contiguous_index))

            xindex_range_root = self.range_trees[0].lookup(
                sympy.Integer(1), sympy_product(lengths)
            )
            xindex_range_root.set_name("xindex")

            # Note - ["None" override_mask]
            # MM Templates work by taking out of bounds index values and wrapping them around to 0
            # so that no mask is required on the load: offs_a_m = `rm % M`
            # We should to override the mask to be "None" instead of inheriting the mask that would
            # have been loaded otherwise.
            # We are using "None" for clarity in output code, but
            # we could alternatively emit `xmask = tl.full([xindex.shape], True, tl.int1)`
            self.template_mask = mask if mask is not None else "None"
            self.template_out_shape = index_shape if index_shape else "xindex"
            self.template_indices = indices
            self.named_input_nodes[input_name].data.freeze_layout()
            self.cse.invalidate(OrderedSet())

            template_mask = self.template_mask

            class StoreOutputSubstitution(V.WrapperHandler):  # type: ignore[name-defined]
                name = "StoreOutputSubstitution"

                def store(
                    self,
                    name: str,
                    index: sympy.Expr,
                    value: "CSEVariable",
                    mode: "StoreMode" = None,
                ):
                    V.kernel.store_buffer_names.add(name)
                    V.kernel.cse.store_cache[name] = value
                    if name in V.kernel.prologue_fused_inputs:
                        # We load masked out values with 0, then apply a prologue.
                        # The masked out values may not necessariliy be 0 any more
                        # so we need to reapply the mask.
                        value_dtype = value.dtype
                        value_str = str(value)
                        if template_mask != "None" and (
                            name not in V.kernel.prologue_fused_inputs_preserve_zero
                            or other != 0
                        ):
                            value_str = (
                                f"tl.where({template_mask}, {value_str}, {other})"
                            )

                        if value_dtype != V.graph.get_buffer(name).dtype:
                            value_str = f"{value_str}.to({triton_type(V.graph.get_buffer(name).dtype)})"

                        # TODO: we should have intermediary var shapes
                        V.kernel.compute.writeline(
                            f"{output_name} = {value_str}.broadcast_to(xindex.shape)"
                        )

            self.ops_handler = StoreOutputSubstitution

            input_node = self.named_input_nodes[input_name]
            output_index = input_node.make_indexer()(index_symbols)

            # in def_kernel above we define the inputs with the storage offset adjusted
            # creating the load in input_node.make_indexer() will also adjust by storage offset
            # so subtract here to not double increment
            if not V.graph.sizevars.statically_known_equals(
                input_node.layout.offset, 0
            ):
                output_index = output_index - self.rename_indexing(
                    input_node.get_layout().offset
                )

            output_index = self.rename_indexing(output_index)

            if output_index == contiguous_index:
                output_index_str = "xindex"
            else:
                out_indexing = self.indexing(
                    output_index,
                    copy_shape=self.template_out_shape,
                    override_mask=self.template_mask,
                )
                from .codegen.triton import IndexingOptions

                assert isinstance(out_indexing, IndexingOptions)
                output_index_str = (
                    f"({out_indexing.index_str}).broadcast_to(xindex.shape)"
                )

            # Generate load code
            load_code = f"{output_name} = tl.load({input_name} + ({output_index_str})"

            if mask:
                load_code += f", mask={mask}, other={other})"
            else:
                load_code += ")"

        hook_key = f"<LOAD_INPUT_{input_name}>"

        def hook():
            with self.set_subgraph_body(hook_key):
                self.cse.invalidate(OrderedSet())
                self.codegen_body()
                self.cse.invalidate(OrderedSet())
                if input_node.get_name() not in self.prologue_fused_inputs:
                    assert load_code is not None
                    self.body.writeline(load_code)

                return textwrap.indent(self.body.getvalue(), " " * indent_width).strip()

        return self._register_hook(hook_key, hook)

    def _generate_index_from_tma_index(
        self,
        output_name: str,
        offset_name: str,
        tma_index: sympy.Symbol,
        block_size: str,
        dim: int,
        num_dims: int,
        block_name: Optional[str] = None,
    ) -> list[str]:
        """
        Generate the logic to compute the regular tl.load index from the provided
        tma index. This is used to ensure variables can support fusions.

        Args:
            output_name (str): The output variable name.
            offset_name (str): The name used for the intermediate offset.
            tma_index (sympy.Symbol): The symbol used for the original TMA index.
            block_size (str): The block size of the index.
            dim (int): Which dimension to project the index in.
            num_dims (int): The total number of dimensions in the output.
            block_name (Optional[str]): The name of the block variable. If not passed
                in then we aren't reusing standard symbol names.

        Returns:
            list[str]: The lines used to generate the index.

        """
        if block_name:
            # Generate the expected names for the structure:
            # XBLOCK/YBLOCK and xoffset/yoffset. We append XBLOCK/YBLOCK
            # to the top of the kernel so we can safely extract the tensor
            # descriptor construction to the top of the kernel.
            if block_name in self.prologue_cache:
                assert self.prologue_cache[block_name] == block_size, (
                    f"Constant {block_name} must be used for all stores"
                )
            else:
                self.prologue_cache[block_name] = block_size
                self.prologue.writeline(f"{block_name}: tl.constexpr = {block_size}")
        else:
            block_name = block_size
        line0 = f"{offset_name} = {texpr(tma_index)}"
        expr = f"({offset_name} + tl.arange(0, {block_name}))"
        prefix_none = "".join(["None, "] * dim)
        suffix_none = ", ".join(["None"] * (num_dims - (dim + 1)))
        line1 = f"{output_name} = {expr}[{prefix_none}:, {suffix_none}]"
        return [line0, line1]

    def _generated_mask_for_tma(
        self,
        index_name: str,
        shape_val: str,
        output_name: str,
    ) -> str:
        """
        Generate the mask logic to feed to fusions for mask. The expectation
        is that if we have X/Y there will be a variable named xmask and ymask.

        Args:
            index_name (str): The index used in the mask. Should be one of
                xindex or yindex.
            shape_val (str): The expression for the upper bound shape.
            output_name (str): The expression used for the output.

        Returns:
            str: The mask generation line.
        """
        return f"{output_name} = {index_name} < {shape_val}"

    def store_output(
        self,
        indices: Union[list[Any], tuple[Any]],
        val: str,
        mask: Optional[str] = None,
        indent_width: int = 4,
        val_shape: Optional[tuple[str]] = None,
        block_indexing: bool = False,
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
            block_indexing (bool): Are the input indices presented as offsets for creating the block (e.g.
                inputs to TMA) or are they tensors that should be passed in directly.
        """
        subgraph_name = self._get_store_output_subgraph_name(
            next(self.store_output_ctr)
        )
        with self.create_subgraph_body(subgraph_name, clear_cse=True):
            assert isinstance(indices, (list, tuple))
            assert isinstance(val, str)
            assert isinstance(mask, (str, type(None)))
            assert isinstance(val_shape, (tuple, type(None)))
            assert isinstance(block_indexing, bool)
            assert self.template_mask is None
            indices = list(map(OpOverrides.paren, indices))
            index_symbols = [sympy.Symbol(x, integer=True) for x in indices]
            lengths = [
                V.graph.sizevars.simplify(s) for s in self.output_node.get_size()
            ]
            assert len(indices) == len(lengths)

            output_layout = self.output_node.get_layout()
            self.template_out = val
            if block_indexing:
                assert val_shape, "Blocking indexing requires passing in val_shape"
                assert len(val_shape) == 2, (
                    "Blocking indexing only supports 2D data at this time"
                )
                assert not mask, "Mask is not supported with blocking indexing"
                intermediate_lines: list[str] = []
                epilogue_index_symbols: list[sympy.Symbol] = []
                if self.tma_store:
                    # Generate the expected indexing symbols.
                    # Note: TMA indices are expected to be in the
                    # format (x, y), but the range_tree is always
                    # (yindex, xindex).
                    index_order = [1, 0]
                    val_shape_copy = list(val_shape)
                    for i, range_tree in zip(index_order, self.range_trees[:-1]):
                        name = range_tree.name
                        symbol = range_tree.symbol()
                        epilogue_index_symbols.append(symbol)
                        lookup_output = range_tree.lookup(sympy.S.One, lengths[i])
                        old_name = lookup_output.symbol()
                        lookup_output.set_name(name)
                        # Update var_list and var_range
                        range_tree.var_list[range_tree.var_list.index(old_name)] = (
                            symbol
                        )
                        range_val = range_tree.var_ranges[old_name]
                        del range_tree.var_ranges[old_name]
                        range_tree.var_ranges[symbol] = range_val
                        intermediate_lines.extend(
                            self._generate_index_from_tma_index(
                                name,
                                "xoffset" if name == "xindex" else "yoffset",
                                index_symbols[i],
                                val_shape[i],
                                i,
                                len(index_order),
                                block_name=range_tree.symt.name,
                            )
                        )
                        # Generate the xmask and ymask
                        intermediate_lines.append(
                            self._generated_mask_for_tma(
                                name,
                                self.size(None, i),
                                "xmask" if name == "xindex" else "ymask",
                            )
                        )
                        # Update the val_shape information to use consistent naming
                        # after the remapping.
                        val_shape_copy[i] = range_tree.symt.name
                    # Reverse the index symbols because TMA is indexed
                    # as (x, y) whereas the variables will naturally be indexed
                    # as (y, x)
                    epilogue_index_symbols.reverse()
                    val_shape = tuple(val_shape_copy)
                else:
                    mask_vars: list[str] = []
                    for i, (index, shape) in enumerate(zip(index_symbols, val_shape)):
                        index_name = self._gen_tmp_var()
                        offset_name = self._gen_tmp_var()
                        intermediate_lines.extend(
                            self._generate_index_from_tma_index(
                                index_name,
                                offset_name,
                                index,
                                shape,
                                i,
                                len(index_symbols),
                            )
                        )
                        epilogue_index_symbols.append(
                            sympy.Symbol(index_name, integer=True)
                        )
                        mask_name = self._gen_tmp_var()
                        intermediate_lines.append(
                            self._generated_mask_for_tma(
                                index_name,
                                self.size(None, i),
                                mask_name,
                            )
                        )
                        mask_vars.append(mask_name)
                    final_mask_var = self._gen_tmp_var()
                    final_mask_rhs = " & ".join(
                        f"{mask_name}" for mask_name in mask_vars
                    )
                    intermediate_lines.append(f"{final_mask_var} = {final_mask_rhs}")
                    self.template_mask = final_mask_var
                index_symbols = epilogue_index_symbols
                contiguous_index = sympy_dot(output_layout.stride, index_symbols)
                if not self.tma_store:
                    # Convert to just use xindex.
                    contiguous_index = self.rename_indexing(contiguous_index)
                    intermediate_lines.append(f"xindex = {texpr(contiguous_index)}")
                    self.range_trees[0].lookup(
                        sympy.S.One, sympy_product(lengths)
                    ).set_name("xindex")
                index_symbols = epilogue_index_symbols
                output_index = contiguous_index
                # Write out the intermediate lines
                for line in intermediate_lines:
                    self.body.writeline(line)
            else:
                assert not self.tma_store, "TMA store requires block indexing"
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
                self.range_trees[0].lookup(
                    sympy.S.One, sympy_product(lengths)
                ).set_name("xindex")
                self.template_mask = mask
                self.template_indices = indices
                output_index = self.output_node.get_layout().make_indexer()(
                    index_symbols
                )
                output_index = self.rename_indexing(output_index)
                if output_index == contiguous_index:
                    output_index = sympy.Symbol("xindex", integer=True)

            self.template_out_shape = val_shape if val_shape else val
            acc_dtype = (
                triton_type_to_torch(self.meta["ACC_TYPE"])
                if "ACC_TYPE" in self.meta
                else torch.float32
            )
            epilogue_args = [
                V.kernel.cse.namedvar(val, dtype=acc_dtype, shape=val_shape)
            ]
            for input_node in itertools.chain(
                self.input_nodes[: self.prefix_args],
                self.input_nodes[len(self.input_nodes) - self.suffix_args :],
            ):
                input_node.freeze_layout()
                epilogue_arg = V.kernel.cse.generate(
                    self.compute,
                    input_node.make_loader()(index_symbols),
                    dtype=acc_dtype,
                    shape=input_node.get_size(),
                )
                epilogue_args.append(epilogue_arg)
                # We update frozen_layouts_cnt in order to replay this function on a cache hit.
                self.frozen_layouts_cnt += 1

            V.ops.store(
                self.output_node.get_name(),
                output_index,
                self.epilogue_fn(*epilogue_args),
                mode="tma" if self.tma_store else None,
            )
            self.codegen_body()

        def hook():
            with self.set_subgraph_body(subgraph_name):
                # more stuff might have been added since the codegen_body above
                self.codegen_body()
                self.cse.invalidate(OrderedSet())

                return textwrap.indent(self.body.getvalue(), " " * indent_width).strip()

        return self._register_hook(subgraph_name, hook)

    def _register_hook(
        self,
        hook_name: str,
        hook_fn: PartialRender.HookFn,
        *,
        allow_overwriting: bool = False,
    ) -> str:
        """
        Register a hook function with a name.

        ``hook_name`` should match the string that will be replaced via
        ``hook_fn``, and should not already be in use for a hook.

        If ``allow_overwriting`` is ``False``, will assert that there isn't
        currently a registered hook of the same name before registering the new
        one.
        """

        if not allow_overwriting:
            assert hook_name not in self.render_hooks, (
                f"Tried to register the hook {hook_name} multiple times. If "
                "desired, pass allow_overwriting=True to _register_hook"
            )
        self.render_hooks[hook_name] = hook_fn
        return hook_name

    def _register_extra_template_env_fns(self, *fns: Callable[..., Any]):
        """
        Register some extra functions to expose when performing the initial
        template render, so that they're in scope to by used by jinja
        expressions.

        These can be used to, for example, implement extra replacement hooks,
        if the given function:

        * Returns the name of their hook, which should also be the string to
          replace via the hook function. The convention is to use the format
          <HOOK_NAME>.
        * Assigns the corresponding entry in ``self.render_hooks`` to a hook
          function.
        """
        self.extra_template_env_fns.extend(fns)

    def render(self, template, kwargs, record_input_dependent_tracked_event=False):
        if record_input_dependent_tracked_event:
            self.cached_replay_events = []

        template_env = {
            fn.__name__: (
                self.record_input_dependent_tracked_event()(fn)
                if record_input_dependent_tracked_event
                else fn
            )
            for fn in [
                self.def_kernel,
                self.size,
                self.stride,
                self.store_output,
                self.load_input,
                self.make_load,
                self.modification,
                self.gen_argdefs,
                self.gen_defines,
                *self.extra_template_env_fns,
            ]
        }
        return PartialRender(
            template.render(**template_env, **kwargs),
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

    def indexing(
        self,
        index: sympy.Expr,
        *,
        dense_indexing=False,
        copy_shape=None,
        override_mask=None,
        block_ptr=False,
        tma_compatibility_checker: Optional[TMACompatibilityChecker] = None,
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
            copy_shape=self.template_out_shape,
            override_mask=self.template_mask,
            block_ptr=block_ptr,
            tma_compatibility_checker=tma_compatibility_checker,
        )

    def codegen_range_tree(self):
        pass  # ignore default codegen

    def additional_call_args_and_types(self):
        if isinstance(self.grid_fn, SymbolicGridFn):
            grid_args = self.grid_fn.sympy_call(*self.call_sizes, self.meta)
            assert len(grid_args) in (0, 3), "grid_fn should return 3 values"
            return (grid_args, map(type, grid_args))
        elif all(isinstance(x, (int, sympy.Integer)) for x in self.call_sizes):
            grid_args = self.grid_fn(*map(int, self.call_sizes), self.meta)
            assert len(grid_args) in (0, 3), "grid_fn should return 3 values"
            return (grid_args, map(type, grid_args))
        return ((), ())

    def call_kernel(self, name: str, node: Optional[ir.IRNode] = None):
        wrapper = V.graph.wrapper_code
        _, call_args, _, arg_types = self.args.python_argdefs()

        additional_call_args, additional_arg_types = (
            self.additional_call_args_and_types()
        )

        if not additional_call_args:
            assert not V.graph.cpp_wrapper, "cpp_wrapper requires SymbolicGridFn"
            wrapper.add_import_once(f"import {self.grid_fn.__module__}")
            meta = wrapper.add_meta_once(self.meta)
            fn_name = f"{self.grid_fn.__module__}.{self.grid_fn.__name__}"
            call_args.append(
                f"*{fn_name}({', '.join(map(pexpr, self.call_sizes))}, {meta})"
            )
            arg_types.append(None)

        call_args.extend(additional_call_args)
        arg_types.extend(additional_arg_types)

        if self.workspace_arg is not None:
            wrapper.generate_workspace_allocation(self.workspace_arg)
        wrapper.generate_kernel_call(
            name,
            call_args,
            arg_types=arg_types,
            triton_meta=self.triton_meta,
            triton=True,
        )
        if self.workspace_arg is not None:
            wrapper.generate_workspace_deallocation(self.workspace_arg)

    def kernel_benchmark_extra_args(self) -> list[str]:
        return [
            str(x)
            for x in self.grid_fn(
                *V.graph.sizevars.size_hints(self.call_sizes), self.meta
            )
        ]


@functools.cache
def _jinja2_env():
    try:
        import jinja2

        return jinja2.Environment(
            undefined=jinja2.StrictUndefined,
        )
    except ImportError:
        return None


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
        call_sizes: Sequence[sympy.core.symbol.Symbol],
        prefix_args: int,
        suffix_args: int,
        epilogue_fn: Optional[Callable[..., Any]],
        epilogue_fn_hash: Optional[str],
        tma_store: bool,
        subgraphs: Optional[list[ir.Buffer]],  # has to be none to cache
        workspace_arg: Optional[WorkspaceArg],  # has to be none to cache
        layout: ir.Layout,
        num_consumer_groups: int,
        num_buffers_warp_spec: int,
        kwargs: dict[str, Any],
        hint_override: Optional[int] = None,
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
                "tma_store": tma_store,
                "kwargs": kwargs,
                "hint_override": hint_override,
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
        super().__init__(name, hash=hashlib.sha256(source.encode("utf-8")).hexdigest())
        self.grid = grid
        self.template = self._template_from_string(source)
        assert name not in self.all_templates, "duplicate template name"
        TritonTemplate.all_templates[name] = self
        self.debug = debug
        self._cache_codegen_enabled_for_template = cache_codegen_enabled_for_template
        self._generated_code_cache: GeneratedCodeCache = GeneratedCodeCache()
        clear_on_fresh_cache(self._generated_code_cache)
        # When prologue_loads_all_inputs is true, prologue_supported_inputs is populated during def_kernel
        # by adding all inputs.
        self.prologue_loads_all_inputs = prologue_loads_all_inputs

    # When this flag is on, we ensure that the cached results and the generated result if cache
    # was not used are the same.
    test_cache = False

    @property
    def uid(self) -> str:
        # unique by prefixing with triton
        return f"triton::{self.name}"

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
            choice = self.generate(generate_with_caching=True, **kwargs)
            if choice is not None:
                choices.append(choice)
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
        call_sizes: Sequence[sympy.core.symbol.Symbol],
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
        hint_override: Optional[int] = None,
        tma_store: bool = False,
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
                tma_store,
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

        fake_out = ir.Buffer(name="buf_out", layout=layout)
        kernel_name = f"triton_{self.name}"

        numel = sympy_product(layout.size)
        buffers = itertools.chain(input_nodes, (fake_out,))

        if TritonScheduling.can_use_32bit_indexing(numel, buffers):
            index_dtype = "tl.int32"
        else:
            index_dtype = "tl.int64"

        # Add index dtype to defines so it's available in the template
        defines.write(f"INDEX_DTYPE : tl.constexpr = {index_dtype}\n")
        defines = defines.getvalue()

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
                hint_override=hint_override,
                tma_store=tma_store,
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
        call_sizes: Optional[Sequence[sympy.core.symbol.Symbol]] = None,
        workspace_arg: Optional[WorkspaceArg] = None,
        generate_with_caching=False,
        hint_override: Optional[int] = None,
        tma_store: bool = False,
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
            hint_override=hint_override,
            tma_store=tma_store,
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

        # `kernel_input_nodes` are the actual inputs that will be passed to the kernel,
        # so e.g. views of the same input are not included. `codegen_input_nodes`
        # includes views of inputs to preserve the kernel semantics. The shape and
        # strides of `codegen_input_nodes` will be used to infer read/writes in
        # TemplateBuffer.extract_read_writes
        kernel_input_nodes = tuple(
            [V.graph.get_buffer(k) for k in result.input_call_args]
        )
        # Here we have (*input_nodes, *captured_buffers)
        codegen_input_nodes = (
            tuple(input_nodes) + kernel_input_nodes[len(expected_input_args) :]
        )
        extra_args = V.graph.sizevars.size_hints(
            map(sympy.expand, result.kernel_args_sizevars_keys),
            fallback=config.unbacked_symint_fallback,
            hint_override=hint_override,
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

        def make_kernel_render(out_node, hint_override: Optional[int] = None):
            assert result is not None
            kernel = self.kernel_type(
                kernel_name=str(Placeholder.KERNEL_NAME),
                output_node=out_node,
                workspace_arg=workspace_arg,
                use_jit=False,
                hint_override=hint_override,
                tma_store=tma_store,
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
                hint_override=hint_override,
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
            input_tensor_meta=TensorMeta.from_irnodes(kernel_input_nodes),  # type: ignore[arg-type]
            output_tensor_meta=TensorMeta.from_irnodes(layout),
        )

        return TritonTemplateCaller(
            kernel_hash_name,
            codegen_input_nodes,
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
            hint_override=hint_override,
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
        # match the API for KernelTemplate as they can be treated the same
        # There is no src hash for ExternKernelChoice in the traditional sense
        # so we indicate this by returning None
        self.src_hash = None

    def to_callable(self):
        return getattr(extern_kernels, self.name)

    def call_name(self):
        return f"extern_kernels.{self.name}"

    @functools.cache  # noqa: B019
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

    @property
    def uid(self) -> str:
        # unique by prefixing with aten
        return f"aten::{self.name}"

    def choice_or_none(self, **kwargs: Any) -> Optional[ChoiceCaller]:
        """
        Maybe generates a new ChoiceCaller and returns it, or None if generation fails.

        kwargs: Additional kwargs to be passed to generate a new ChoiceCaller.
        """
        temp_choices: list[Any] = []
        result = self.maybe_append_choice(temp_choices, **kwargs)
        if result is None and len(temp_choices) == 1:
            return temp_choices[0]
        return None

    def maybe_append_choice(
        self, choices: list[Any], **kwargs: Any
    ) -> Optional[NotImplementedError]:
        # convenience function to match the Template interface, so that
        # templates and ExternKernelChoice can be treated the same when
        # generating choice callers
        assert "input_nodes" in kwargs, "input_nodes argument required"
        assert "layout" in kwargs, "layout argument required"
        input_nodes = kwargs.pop("input_nodes")
        layout = kwargs.pop("layout")
        choices.append(self.bind(input_nodes=input_nodes, layout=layout, **kwargs))
        return None


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
            dict[str, Union[PrimitiveInfoType, list[PrimitiveInfoType]]]
        ] = None,
        mutated_inputs=None,
        workspace_arg: Optional[WorkspaceArg] = None,
        allowed_prologue_inps: Optional[OrderedSet[str]] = None,
        hint_override: Optional[int] = None,
    ) -> None:
        super().__init__(name, input_nodes, layout, description)
        self.make_kernel_render = make_kernel_render
        self.bmreq: TritonBenchmarkRequest = bmreq
        if log_info is None:
            log_info = {}
        self.log_info: dict[str, Any] = log_info
        self.log_info.update(
            {
                "backend": "Triton",
                "num_stages": self.bmreq.num_stages,
                "num_warps": self.bmreq.num_warps,
            }
        )
        self.mutated_inputs = mutated_inputs
        self.workspace_arg = workspace_arg
        self.allowed_prologue_inps = (
            allowed_prologue_inps if allowed_prologue_inps is not None else OrderedSet()
        )
        self.hint_override = hint_override

    def benchmark(self, *args, out):
        assert self.bmreq is not None
        if config.profile_bandwidth_with_do_bench_using_profiling:
            algo = self.bmreq.make_run_fn(*args, out=out)
            return do_bench_using_profiling(algo)
        return self.bmreq.benchmark(*args, out=out)

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
                allowed_prologue_inps=self.allowed_prologue_inps,
            )
        )

    def info_dict(self) -> dict[str, Union[PrimitiveInfoType, list[PrimitiveInfoType]]]:
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
            if config.profile_bandwidth_with_do_bench_using_profiling:
                return do_bench_using_profiling(lambda: algo(*args))
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
            assert self.choice.op_overload is not None, (
                "Please provide an op_overload to use ir.FallbackKernel"
            )
            inner: ir.IRNode = ir.FallbackKernel.create(
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

    def info_dict(self) -> dict[str, Union[PrimitiveInfoType, list[PrimitiveInfoType]]]:
        """Information returned here is logged to the autotune log file when that is enabled."""
        return {
            "backend": "extern",
            "kernel_call_name": self.choice.call_name(),
        }

    def autoheuristic_id(self):
        return f"extern_{self.choice.name}"


@functools.cache
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


@functools.cache
def get_num_workers() -> int:
    if "TORCHINDUCTOR_COMPILE_THREADS" in os.environ:
        return int(os.environ["TORCHINDUCTOR_COMPILE_THREADS"])

    cpu_count = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count()
    )
    assert cpu_count

    # Divide the number of CPUs by the number of GPUs for distributed workloads
    if (
        config.is_fbcode()
        and torch.cuda.is_available()
        and torch.cuda.device_count() > 0
    ):
        cpu_count = cpu_count // torch.cuda.device_count()

    return cpu_count


def create_inputs_key(input_nodes) -> str:
    return repr([AlgorithmSelectorCache.key_of(x) for x in input_nodes])


def create_precompile_key(
    name: str, inputs_key: str, choices: list[ChoiceCaller]
) -> str:
    return ":".join(
        [
            name,
            inputs_key,
            torch.get_float32_matmul_precision(),
        ]
        + [choice.kernel_hash_key() for choice in choices]
    )


# Args to FeedbackFunctions
# timings: mapping from choices to the benchmark time
# name: name of the op
# input_nodes: list of input ir.py Nodes
# choices: list of choices
# profiled time: Callable that returns a dict mapping from choices to the profiled time
FeedbackFunction = Callable[
    [
        dict[ChoiceCaller, float],
        str,
        list[Any],
        list[ChoiceCaller],
        Callable[[], dict[ChoiceCaller, float]],
    ],
    None,
]

# Args to PreprocessingFunctions
# choices: list of ChoiceCaller objects to preprocess
# Returns: modified list of ChoiceCaller objects
PreprocessingFunction = Callable[[list[ChoiceCaller]], list[ChoiceCaller]]


def filter_choices_by_name_regex(choices: list[ChoiceCaller]) -> list[ChoiceCaller]:
    """Filter choices based on autotune_choice_name_regex config."""
    if config.test_configs.autotune_choice_name_regex is not None:
        return [
            c
            for c in choices
            if re.search(
                config.test_configs.autotune_choice_name_regex,
                c.name,
            )
        ]
    return choices


def filter_choices_by_desc_regex(choices: list[ChoiceCaller]) -> list[ChoiceCaller]:
    """Filter choices based on autotune_choice_desc_regex config."""
    if config.test_configs.autotune_choice_desc_regex is not None:
        return [
            c
            for c in choices
            if re.search(
                config.test_configs.autotune_choice_desc_regex,
                c.description,
            )
        ]
    return choices


class AlgorithmSelectorCache(PersistentCache):
    """
    A persistent cache for algorithm selection results used in autotuning of GEMMs
    and convolutions.

    This classes includes precompilation and benchmarking of the kernels.

    The cache is keyed by input characteristics (sizes, strides, dtypes, etc.) but
    doesn't depend on the output layout.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # the autotuning will get occur in the scheduler, so there is
        # no guarantee that the first lowering for a given key will also be the
        # first to benchmark it. share a single precompilation function for all lowerings
        # of a particular key
        self.precompile_cache: dict[str, Callable[[], None]] = {}
        # cache for prescreening results to ensure deterministic candidate selection
        self.prescreening_cache: dict[str, OrderedSet[str]] = {}
        # list of callbacks that are called after benchmarking
        self.feedback_saver_fns: list[FeedbackFunction] = []
        # list of callbacks that are called to preprocess choices
        self.preprocessing_fns: list[PreprocessingFunction] = []

        self._register_default_preprocessing_fns()

        # registers `self.cache_clear(...)` to be called when a fresh Inductor cache is requested
        clear_on_fresh_cache(self)

    def _register_default_preprocessing_fns(self):
        """Register default preprocessing functions."""
        # Note: broken out into its own function so that we can avoid clearing
        # them (i.e. so we can restore them after clearing user provided ones)
        self.add_preprocessing_fn(filter_choices_by_name_regex)
        self.add_preprocessing_fn(filter_choices_by_desc_regex)

    def cache_clear(self) -> None:
        self.precompile_cache.clear()
        self.prescreening_cache.clear()

    def pick_deterministic_choice(self, choices: list[ChoiceCaller]) -> ChoiceCaller:
        assert len(choices) >= 2
        externs = [
            choice for choice in choices if isinstance(choice, ExternKernelChoice)
        ]
        if len(externs) > 0:
            return externs[0]
        else:
            return choices[0]

    def __call__(
        self,
        name,
        choices: list[ChoiceCaller],
        input_nodes,
        layout,
        # optional dict mapping arg indices to the functions
        # generating a torch.Tensor for that input from the
        # corresponding ir.Buffer. if passed for a given
        # arg, the function will be called instead of
        # generating a random torch.Tensor for benchmarking.
        input_gen_fns: Optional[dict[int, Callable[[ir.Buffer], torch.Tensor]]] = None,
        precompilation_timeout_seconds: int = 60 * 60,
        return_multi_template=False,
        best_config_future=None,
    ):
        from .codegen.cuda.cuda_kernel import CUDATemplateCaller

        # Run preprocessing functions on choices
        for preprocessing_fn in self.preprocessing_fns:
            choices = preprocessing_fn(choices)

        # Templates selected with input_gen_fns require specific input data to avoid IMA
        # Passing custom input gen fns to benchmark_fusion NYI, so skip deferred template selection
        # TODO(jgong5): support multi-template on CPU
        if input_gen_fns is not None or layout.device.type == "cpu":
            return_multi_template = False

        # TODO - assert that we have not mutating kernels here

        if mm_file_name := get_mm_log_filename():
            M, K = input_nodes[-2].get_size()[:2]
            N = input_nodes[-1].get_size()[-1]
            append_to_log(mm_file_name, {"invoke": str((M, K, N))})

        def create_no_valid_choices(reason: str) -> NoValidChoicesError:
            backend_config = (
                "max_autotune_gemm_backends"
                if name != "convolution"
                else "max_autotune_conv_backends"
            )
            return NoValidChoicesError(
                f"No choices to select. Provided reason: {reason} "
                f"please consider adding ATEN into {backend_config} "
                "config (defined in torch/_inductor/config.py) to allow at least one choice. "
            )

        if len(choices) == 0:
            raise create_no_valid_choices("No choices exist for backend.")
        log.debug("Max autotune selects from %s choices.", str(len(choices)))

        if len(choices) == 1:
            if not isinstance(choices[0], CUDATemplateCaller):
                # CUDATemplateCaller still needs to go through autotuning process to retrieve workspace size.
                return choices[0].output_node()

        if config.deterministic:
            return self.pick_deterministic_choice(choices).output_node()

        inputs_key = create_inputs_key(input_nodes)

        # TODO(nmacchioni): remove this hacky way to tell if we ran benchmarking
        has_autotuned = False

        def benchmark(choices, hint_override: Optional[int] = None):
            nonlocal has_autotuned
            # TODO(nmacchioni): remove this hacky way to tell if we ran benchmarking
            has_autotuned = True
            counters["inductor"]["select_algorithm_autotune"] += 1
            # TODO(nmacchioni): remove this layer of abstraction
            # construct `benchmark_fn` which should pick between in-process and sub-process autotuning
            benchmark_fn = self.make_benchmark_fn(
                choices, input_nodes, layout, input_gen_fns, hint_override=hint_override
            )
            # `benchmark_fn(choices)` will execute each choice, and return a dict[choice, timing] which
            # maps each choice to its runtime, calculated by the specified benchmarker, in milliseconds
            return benchmark_fn(choices)

        def autotune(choices, hint_override: Optional[int] = None):
            log.debug("Starting autotuning")

            with dynamo_timed(
                f"{name}_template_autotuning",
                log_pt2_compile_event=True,
                dynamo_compile_column_us="compile_time_autotune_time_us",
                metadata=_autotune_metadata(input_nodes),
            ):
                benchmark_results = benchmark(choices, hint_override=hint_override)
                if config.max_autotune_report_choices_stats:
                    _log_autotune_choices_stats(
                        f"{name}_template_autotuning", benchmark_results
                    )
                return benchmark_results

        if config.autotune_in_subproc:
            # Initialize the suprocess pool so it will warmup early.
            torch._inductor.autotune_process.get_tuning_process_pool()

        def do_autotuning(choices, precompile_fn, hint_override: Optional[int] = None):
            precompile_start_ts = time.time()
            with dynamo_timed(
                f"{name}_template_precompiling",
                log_pt2_compile_event=True,
                dynamo_compile_column_us="compile_time_autotune_time_us",
            ):
                precompile_fn()
            precompile_elapse = time.time() - precompile_start_ts
            log.debug("Precompilation elapsed time: %.02fs", precompile_elapse)
            # Prune anything that failed to compile
            choices = [c for c in choices if not c.failed]
            if len(choices) == 0:
                raise create_no_valid_choices(
                    "All choices failed to compile for backend."
                )

            candidates = self.prescreen_choices(
                choices, name, inputs_key, self.prescreening_cache
            )
            prescreening_elapse: Optional[float] = None
            if candidates:
                prescreening_start_ts = time.time()
                timings = self.lookup(
                    candidates,
                    name,
                    inputs_key,
                    lambda choices: autotune(choices, hint_override=hint_override),
                    hint_override=hint_override,
                )
                choices = self.prune_choices_postscreen(
                    choices, timings, name, inputs_key, self.prescreening_cache
                )
                prescreening_elapse = time.time() - prescreening_start_ts
                log.debug("Prescreening elapsed time: %.02fs", prescreening_elapse)

            autotune_start_ts = time.time()

            if best_config_future is not None:
                best_config = await_sync(best_config_future)

                important_keys = [
                    "ACC_TYPE",
                    "ALLOW_TF32",
                    "BLOCK_K",
                    "BLOCK_M",
                    "BLOCK_N",
                    "EVEN_K",
                    "GROUP_M",
                    "USE_FAST_ACCUM",
                    "num_stages",
                    "num_warps",
                    "num_consumer_groups",
                    "num_buffers_warp_spec",
                ]
                choices = [
                    choice
                    for choice in choices
                    if all(
                        f"{k}={best_config[k]}" in choice.description
                        for k in important_keys
                    )
                    for k in important_keys
                ]
                log.info("Filtered to %d choices based on best_config", len(choices))

            timings = self.lookup(
                choices,
                name,
                inputs_key,
                lambda choices: autotune(choices, hint_override=hint_override),
                hint_override=hint_override,
            )

            autotune_elapse = time.time() - autotune_start_ts
            log.debug("Autotuning elapsed time: %.02fs", autotune_elapse)

            if timings and all(
                not math.isfinite(timing) for timing in timings.values()
            ):
                raise NoValidChoicesError

            if (
                has_autotuned
                or log.getEffectiveLevel() == logging.DEBUG
                or config.trace.log_autotuning_results
            ):
                self.log_results(
                    name,
                    input_nodes,
                    timings,
                    autotune_elapse,
                    precompile_elapse,
                    prescreening_elapse,
                    hint_override=hint_override,
                )

            def profiler_bench_function():
                # we're not running through the normal caching autotuner method here because we want to avoid returning
                # the cached value.
                # Avoid benchmarking in a separate process because it's not easy to signal to the TuningProcess that we
                # should use the profiler.
                with config.patch(
                    profile_bandwidth_with_do_bench_using_profiling=True,
                    autotune_in_subproc=False,
                ):
                    return benchmark(choices)

            for feedback_fn in self.feedback_saver_fns:
                # re-benchmarking the same choices with profiler is a bit expensive, so pass it in as a thunk.
                feedback_fn(
                    timings,
                    name,
                    input_nodes,
                    choices,
                    profiler_bench_function,
                )

            return timings

        precompile_fn = self.make_precompile_fn(
            choices,
            name,
            inputs_key,
            precompilation_timeout_seconds=precompilation_timeout_seconds,
        )

        if return_multi_template and (config.max_autotune or config.max_autotune_gemm):

            def get_timings(hint_override: Optional[int] = None):
                filtered_choices = [
                    c
                    for c in choices
                    if not hasattr(c, "hint_override")
                    or c.hint_override == hint_override
                ]
                timings = do_autotuning(
                    filtered_choices, precompile_fn, hint_override=hint_override
                )
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

            # We take the union of allowed prologue inputs from all choices,
            # and, within benchmark fusion, don't allow prologue fusion for
            # choices which don't support the whole union.
            allowed_prologue_inps: OrderedSet[str] = OrderedSet()
            for c in choices:
                if isinstance(c, TritonTemplateCaller):
                    allowed_prologue_inps |= c.allowed_prologue_inps

            return torch._inductor.ir.TensorBox.create(
                torch._inductor.ir.MultiTemplateBuffer(
                    layout,
                    input_nodes,
                    get_timings,
                    choices,
                    allowed_prologue_inps,
                )
            )

        timings = do_autotuning(choices, precompile_fn)

        # if timings is empty, we really have no choice but to return a semi-random
        # choice. returning the first `ExternKernelCaller` is probably the safest bet
        # in this case, since it will generally be the ATen kernel. if there are no
        # `ExternKernelCaller`s to return, then returning the 0th kernel is our next
        # best option (ideally we'd fail whenever there is no ATen kernel to fallback
        # to, but that's not trivial to figure out)
        if timings == {}:
            for choice in choices:
                if isinstance(choice, ExternKernelCaller):
                    node = choice.output_node()
                    log.debug(
                        "Autotuning returned empty timings, falling back to first `ExternKernelCaller`: %s",
                        node,
                    )
                    return node
            node = choices[0].output_node()
            log.debug(
                "Autotuning returned empty timings, falling back to first choice: %s",
                node,
            )
            return node

        # if we got any timings at all, pick the best of those
        choice = min(timings, key=timings.__getitem__)
        node = choice.output_node()
        log.debug("Autotuning selected choice: %s", node)
        return node

    def make_precompile_fn(
        self,
        choices,
        name: str,
        inputs_key: str,
        precompilation_timeout_seconds: Optional[int] = 60 * 60,
    ) -> Callable[[], None]:
        """
        Returns a function that precompiles the given choices.
        """
        log.debug("Starting precompilation")

        def no_op(*args, **kwargs):
            return

        if (
            precompilation_timeout_seconds is None
            or precompilation_timeout_seconds <= 0
        ):
            log.debug("Precompilation timeout is None or <= 0, returning no_op")
            return no_op

        num_workers = min(get_num_workers(), len(choices))

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

        if timings and len(timings) == len(choices):
            # compilation in precompile stage is much cheaper than that in
            # autotuning stage
            log.debug("Found all %d timings in cache, returning no_op", len(timings))
            return no_op

        precompile_key = create_precompile_key(name, inputs_key, choices)
        if precompile_func := self.precompile_cache.get(precompile_key):
            log.debug("Precompile function found in cache, returning it")
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

        def precompile_with_captured_stdout(choice) -> tuple[None, int]:
            log.debug("Precompiling choice with captured stdout: %s", choice)
            start_ns = time.time_ns()
            with restore_stdout_stderr():
                choice.precompile()
            elapsed_ns = time.time_ns() - start_ns
            # Return tuple as triton async compile (_worker_compile_triton)
            # returns tuple[CachingAutotuner, int]
            return None, elapsed_ns // 1000

        def on_complete(future):
            if not future.exception():
                _, precompile_elapsed_us = future.result()
                elapsed_seconds = precompile_elapsed_us / 1e6
                elapsed_times[future] = elapsed_seconds
                log.debug(
                    "Precompilation complete for future: %s, elapsed time: %.02fs",
                    future,
                    elapsed_seconds,
                )

        executor = ThreadPoolExecutor(max_workers=num_workers)
        async_compile = torch._inductor.async_compile.AsyncCompile()

        futures: dict[concurrent.futures.Future[Any], ChoiceCaller] = {}
        elapsed_times: dict[concurrent.futures.Future[Any], float] = {}

        # Some choices only differ in runtime arguments, so we
        # skip a choice if it has the same hash as a previously seen choice
        seen_choices: OrderedSet[str] = OrderedSet()
        for c in choices:
            # Skip choices which we have already issued a precompile
            if c.kernel_hash_key() in seen_choices:
                log.debug("Skipping already seen choice: %s", c)
                continue
            else:
                seen_choices.add(c.kernel_hash_key())

            if hasattr(c, "precompile"):
                triton_cuda_choice = isinstance(c, TritonTemplateCaller) and isinstance(
                    c.bmreq, TritonGPUBenchmarkRequest
                )
                if triton_cuda_choice and async_compile.use_process_pool():
                    with open(c.bmreq.module_path) as file:
                        source_code = file.read()
                    future = async_compile.triton(
                        kernel_name=c.bmreq.kernel_name, source_code=source_code
                    ).future
                    log.debug("Submitted triton async compile for choice: %s", c)
                else:
                    future = executor.submit(precompile_with_captured_stdout, c)
                    log.debug("Submitted precompile for choice: %s", c)

                future.add_done_callback(on_complete)
                futures[future] = c

        @functools.cache
        @restore_stdout_stderr()
        def wait_on_futures():
            log.debug("Waiting on futures")
            counters["inductor"]["select_algorithm_precompile"] += 1
            exceptions: list[tuple[ChoiceCaller, BaseException]] = []
            for future in as_completed(
                futures,
                timeout=precompilation_timeout_seconds,
            ):
                if e := future.exception():
                    counters["inductor"][
                        "select_algorithm_num_precompilation_exceptions"
                    ] += 1
                    exceptions.append((futures[future], e))
                    from torch._inductor.codegen.cuda.cuda_kernel import (
                        CUDATemplateCaller,
                    )

                    if isinstance(e, CUDACompileError) and isinstance(
                        futures[future], CUDATemplateCaller
                    ):
                        log.debug(
                            "Exception %s for benchmark choice %s",
                            e,
                            futures[future],
                            exc_info=e,
                        )
                        futures[future].mark_failed()
                    else:
                        log.exception(  # noqa: G202
                            "Exception %s for benchmark choice %s",
                            e,
                            futures[future],
                            exc_info=e,
                        )
                        futures[future].mark_failed()
                else:
                    counters["inductor"]["select_algorithm_num_precompiles"] += 1
                    log.info(
                        "Precompiling benchmark choice %s took %.02fs",
                        futures.get(future),
                        elapsed_times.get(future),
                    )
            if exceptions:
                _log_autotune_exceptions(exceptions)

            executor.shutdown(wait=True)

        self.precompile_cache[precompile_key] = wait_on_futures

        return wait_on_futures

    @classmethod
    def get_inputs(
        cls,
        choices: Sequence[ChoiceCaller],
        input_nodes: list[ir.IRNode],
        layout: ir.Layout,
        input_gen_fns: Optional[dict[int, Callable[[ir.Buffer], torch.Tensor]]],
        hint_override: Optional[int] = None,
    ) -> AutotuneArgs:
        """
        Factory method to create AutotuneArgs from a list of ChoiceCallers.
        """
        if input_gen_fns is None:
            input_gen_fns = {}

        # de-duplicate args
        unique_example_inputs = {
            x.get_name(): input_gen_fns.get(
                i, lambda x: cls.benchmark_example_value(x, hint_override=hint_override)
            )(x)
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
                        hint_override=hint_override,
                    ),
                    V.graph.sizevars.size_hints(
                        input_node.get_stride(),
                        fallback=config.unbacked_symint_fallback,
                        hint_override=hint_override,
                    ),
                    V.graph.sizevars.size_hint(
                        input_node.get_layout().offset,
                        fallback=config.unbacked_symint_fallback,
                        hint_override=hint_override,
                    ),
                )
            )
            for input_node in input_nodes
        ]
        out = cls.benchmark_example_value(layout, hint_override=hint_override)
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

    @staticmethod
    def _is_extern(choice: ChoiceCaller) -> bool:
        return isinstance(choice, (ExternKernelCaller, SubgraphChoiceCaller))

    @classmethod
    def benchmark_choice(
        cls, choice: ChoiceCaller, autotune_args: AutotuneArgs
    ) -> float:
        benchmark_tensors = autotune_args.get_benchmark_tensors(cls._is_extern(choice))
        inputs, output = benchmark_tensors.unpack()
        output.zero_()
        result = choice.benchmark(*inputs, out=output)
        device_type = next(
            (tensor.device.type for tensor in inputs if is_gpu(tensor.device.type)),
            "cuda",
        )
        device_interface = get_interface_for_device(device_type)
        if device_interface.is_available():
            device_interface.synchronize()  # shake out any CUDA errors

        if VERIFY and autotune_args.expected is not None:
            autotune_args.verify(**VERIFY)
        return result

    @classmethod
    def benchmark_choices(
        cls,
        choices: Sequence[ChoiceCaller],
        autotune_args: AutotuneArgs,
    ) -> dict[ChoiceCaller, float]:
        timings = {}
        for choice in choices:
            try:
                timing = cls.benchmark_choice(choice, autotune_args)
            except CUDACompileError as e:
                from torch._inductor.codegen.cuda.cuda_kernel import CUDATemplateCaller

                if not isinstance(choice, CUDATemplateCaller):
                    log.error(
                        "CUDA compilation error during autotuning: \n%s. \nIgnoring this choice.",
                        e,
                    )
                timing = float("inf")
            except NotImplementedError as e:
                log.warning("Not yet implemented: %s", e)
                timing = float("inf")
            except RuntimeError as e:
                from torch._inductor.codegen.cuda.cuda_kernel import CUDATemplateCaller

                msg = str(e)
                if "invalid argument" in msg:
                    msg += "\n\nThis may mean this GPU is too small for max_autotune mode.\n\n"
                else:
                    if "illegal memory access" in msg:
                        msg += "\n\nEither error in template or triton bug.\n"

                if isinstance(choice, CUDATemplateCaller):
                    log.debug(
                        "Runtime error during autotuning: \n%s. \nIgnoring this choice.",
                        msg,
                        exc_info=True,
                    )
                else:
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

    @classmethod
    def benchmark_in_current_process(
        cls,
        choices: Sequence[ChoiceCaller],
        input_nodes: list[ir.IRNode],
        layout: ir.Layout,
        input_gen_fns: Optional[dict[int, Callable[[ir.Buffer], torch.Tensor]]],
        hint_override: Optional[int] = None,
    ) -> dict[ChoiceCaller, float]:
        inputs = cls.get_inputs(
            choices, input_nodes, layout, input_gen_fns, hint_override=hint_override
        )
        return cls.benchmark_choices(choices, inputs)

    @classmethod
    def benchmark_in_sub_process(
        cls,
        choices: Sequence[ChoiceCaller],
        input_nodes: list[ir.IRNode],
        layout: ir.Layout,
        input_gen_fns: Optional[dict[int, Callable[[ir.Buffer], torch.Tensor]]],
        hint_override: Optional[int] = None,
    ):
        from . import autotune_process

        # only benchmark triton kernel in sub process for now.
        # ATen/Extern kernel are still benchmarked in the current process.
        extern = [c for c in choices if cls._is_extern(c)]
        triton = [c for c in choices if not cls._is_extern(c)]

        timings = cls.benchmark_in_current_process(
            extern, input_nodes, layout, input_gen_fns, hint_override=hint_override
        )
        timings.update(autotune_process.benchmark_in_sub_process(triton))  # type: ignore[arg-type]
        return timings

    @classmethod
    def make_benchmark_fn(
        cls,
        choices: Sequence[ChoiceCaller],
        input_nodes: list[ir.IRNode],
        layout: ir.Layout,
        input_gen_fns: Optional[dict[int, Callable[[ir.Buffer], torch.Tensor]]],
        hint_override: Optional[int] = None,
    ):
        if DEBUG:
            print(f"{len(choices)} tuning requests:")

        if config.autotune_in_subproc:
            return functools.partial(
                cls.benchmark_in_sub_process,
                input_nodes=input_nodes,
                layout=layout,
                input_gen_fns=input_gen_fns,
                hint_override=hint_override,
            )
        else:
            return functools.partial(
                cls.benchmark_in_current_process,
                input_nodes=input_nodes,
                layout=layout,
                input_gen_fns=input_gen_fns,
                hint_override=hint_override,
            )

    @staticmethod
    def prescreen_choices(
        choices: list[ChoiceCaller],
        name: str,
        inputs_key: str,
        prescreen_cache: dict[str, OrderedSet[str]],
    ) -> list[ChoiceCaller]:
        """
        Figure out what choices need to be prescreened before autotuning with runtime
        params.

        Prescreening is a process of reducing the number of autotuning for choices with
        runtime params via a two stage autotuning process. First, we fix a set of runtime
        params (here we use swizzle=2) and run autotuning to get a set of candidates.
        Then, we run autotuning again with the candidates and the full set of runtime
        params.

        Since have the concept of runtime params, we need to differentiate between
        choice's hash_key and choice's kernel_hash_key. The former includes information
        like runtime params, while the latter does not. prescreen_cache, if exists, stores
        the set of hash_key that should win the prescreening.

        Right now, only CUTLASS choices have runtime params.
        """
        # Create a cache key for prescreening results
        prescreen_key = f"{name}:{inputs_key}"

        # Check if we have cached prescreening results (prescreen_winners)
        if prescreen_key in prescreen_cache:
            prescreen_winners = [
                choice
                for choice in choices
                if choice.hash_key() in prescreen_cache[prescreen_key]
            ]
            return prescreen_winners

        # prescreen cutlass
        from .codegen.cuda.cuda_kernel import CUDATemplateCaller

        candidates = []
        if (
            config.cuda.cutlass_prescreening
            and len(config.cuda.cutlass_max_profiling_swizzle_options) > 1
        ):
            candidates.extend(
                [
                    c
                    for c in choices
                    if isinstance(c, CUDATemplateCaller)
                    # hardcoded to only look at swizzle=2
                    if c.info_dict().get("swizzle") == "2"
                ]
            )

        # skip prescreening if the number of candidates is too small
        if len(candidates) < 10:
            return []

        return candidates  # type: ignore[return-value]

    @staticmethod
    def prune_choices_postscreen(
        choices: list[ChoiceCaller],
        candidate_timings: dict[ChoiceCaller, float],
        name: str,
        inputs_key: str,
        prescreen_cache: dict[str, OrderedSet[str]],
    ) -> list[ChoiceCaller]:
        """
        Prune the choices after prescreening.
        """
        from .codegen.cuda.cuda_kernel import CUDATemplateCaller

        prescreen_key = f"{name}:{inputs_key}"

        # Check if we have cached postscreen results
        if prescreen_key in prescreen_cache:
            # candidate_timings are from choices that have won prescreening already
            winner_kernel_hashes = [
                candidate.kernel_hash_key() for candidate in candidate_timings
            ]

            pruned_choices = [
                choice
                for choice in choices
                if not isinstance(choice, CUDATemplateCaller)
                or choice.kernel_hash_key() in winner_kernel_hashes
            ]
            return pruned_choices

        log.debug("Before pruning using prescreening timings, %d choices", len(choices))
        sorted_candidates = sorted(
            candidate_timings.keys(), key=lambda choice: candidate_timings[choice]
        )

        # Print prescreening timings
        if (
            candidate_timings
            and PRINT_AUTOTUNE
            and config.autotune_num_choices_displayed != 0
        ):
            n = config.autotune_num_choices_displayed
            top_k = sorted_candidates[:n]
            best = top_k[0]
            best_time = candidate_timings[best]

            lines = ["PRESCREENING CANDIDATE TIMINGS"]
            for choice in top_k:
                result = candidate_timings[choice]
                if result:
                    lines.append(
                        f"  {choice.name} {result:.4f} ms {best_time / result:.1%} {choice.description}"
                    )
                else:
                    lines.append(
                        f"  {choice.name} {result:.4f} ms <DIVIDED BY ZERO ERROR>"
                    )

            log.info("\n".join(lines))
        num_to_keep = max(int(math.sqrt(len(choices)) / 4), 8)

        # prune choices based on prescreening timings
        candidates_to_prune = OrderedSet(
            candidate.kernel_hash_key() for candidate in sorted_candidates[num_to_keep:]
        )
        winner_hashes: OrderedSet[str] = OrderedSet()
        for candidate in sorted_candidates[:num_to_keep]:
            if candidate_timings[candidate] == float("inf"):
                candidates_to_prune.add(candidate.kernel_hash_key())
            else:
                winner_hashes.add(candidate.hash_key())
                if isinstance(candidate, CUDATemplateCaller):
                    candidate.bmreq.ensure_dll_loaded()

        pruned_choices = [
            choice
            for choice in choices
            if choice.kernel_hash_key() not in candidates_to_prune  # type: ignore[attr-defined]
        ]

        # Cache the hash_key of winners of prescreening
        prescreen_cache[prescreen_key] = winner_hashes

        log.debug(
            "After pruning using prescreening timings, %d choices", len(pruned_choices)
        )
        return pruned_choices

    @staticmethod
    def log_results(
        name: str,
        input_nodes: list[ir.IRNode],
        timings: dict[ChoiceCaller, float],
        elapse: float,
        precompile_elapse: float,
        prescreening_elapse: Optional[float] = None,
        hint_override: Optional[int] = None,
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
                            n.get_size(),
                            fallback=config.unbacked_symint_fallback,  # type: ignore[arg-type]
                            hint_override=hint_override,
                        ),
                    )
                )
                for n in input_nodes
            ]
        )

        strides = ", ".join([str(n.get_stride()) for n in input_nodes])
        dtypes = ", ".join([str(n.get_dtype()) for n in input_nodes])
        if config.autotune_num_choices_displayed == 0:
            return
        # when autotune_num_choices_displayed is None, [:None] means all
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
        sys.stderr.write(f"strides: {strides}\n")
        sys.stderr.write(f"dtypes: {dtypes}\n")

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
        prescreening_msg = (
            f" and {prescreening_elapse:.4f} seconds prescreening"
            if prescreening_elapse is not None
            else ""
        )
        sys.stderr.write(
            f"{autotune_type_str} AUTOTUNE benchmarking takes {elapse:.4f} seconds and {precompile_elapse:.4f}"
            f" seconds precompiling for {len(timings)} choices"
            + prescreening_msg
            + "\n"
        )

    @staticmethod
    def benchmark_example_value(node, hint_override: Optional[int] = None):
        """
        Convert an ir.Buffer into a concrete torch.Tensor we can use for
        benchmarking.
        """
        if isinstance(node, ir.Layout):
            node = ir.Buffer(name="fake", layout=node)
        # triton templates want the base tensor.
        if isinstance(node, ir.BaseView):
            node = node.unwrap_view()

        # Inplace padding may reinterpret a tensor to a larger tensor if the
        # stride is large enough. The V.graph.get_allocation_size takes this into account.
        # So we need call as_strided in the end to 'view' the tensor with the correct
        # sizes/strides
        return AlgorithmSelectorCache.generate_example_value(
            V.graph.sizevars.size_hints(
                node.get_size(),
                fallback=config.unbacked_symint_fallback,
                hint_override=hint_override,
            ),
            V.graph.sizevars.size_hints(
                node.get_stride(),
                fallback=config.unbacked_symint_fallback,
                hint_override=hint_override,
            ),
            node.get_device(),
            node.get_dtype(),
            node.layout.offset,
            V.graph.sizevars.size_hints(
                V.graph.get_allocation_size(node),
                fallback=config.unbacked_symint_fallback,
                hint_override=hint_override,
            ),
        )

    @staticmethod
    def generate_example_value(
        size, stride, device, dtype, extra_size, allocation_size=None
    ):
        # preserve rng states to avoid the rand_strided call below changes
        # the rng states for the real model code.
        with preserve_rng_state():
            if allocation_size is None or allocation_size == size:
                return rand_strided(
                    size,
                    stride,
                    device=device,
                    dtype=dtype,
                    extra_size=extra_size,
                )
            else:
                return rand_strided(
                    allocation_size,
                    stride,
                    device=device,
                    dtype=dtype,
                    extra_size=extra_size,
                ).as_strided(size, stride)

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

    def add_feedback_saver(self, fn: FeedbackFunction):
        self.feedback_saver_fns.append(fn)

    def clear_feedback_savers(self):
        self.feedback_saver_fns = []

    def add_preprocessing_fn(self, fn: PreprocessingFunction):
        self.preprocessing_fns.append(fn)

    def clear_preprocessing_fns(self, clear_defaults: bool = False):
        """Clear preprocessing functions.

        Args:
            clear_defaults: If True, clears all functions including defaults.
                           If False, clears only user-added functions and re-registers defaults.
        """
        self.preprocessing_fns.clear()
        if not clear_defaults:
            self._register_default_preprocessing_fns()


_ALGORITHM_SELECTOR_CACHE: Optional[AlgorithmSelectorCache] = None


def get_algorithm_selector_cache() -> AlgorithmSelectorCache:
    """Get the global algorithm selector cache, creating it if it doesn't exist."""
    global _ALGORITHM_SELECTOR_CACHE
    if _ALGORITHM_SELECTOR_CACHE is None:
        _ALGORITHM_SELECTOR_CACHE = AlgorithmSelectorCache()
    return _ALGORITHM_SELECTOR_CACHE


def autotune_select_algorithm(*args, **kwargs):
    cache = get_algorithm_selector_cache()

    if "return_multi_template" not in kwargs:
        kwargs["return_multi_template"] = (
            torch._inductor.config.benchmark_epilogue_fusion
        )

    if "precompilation_timeout_seconds" not in kwargs:
        kwargs["precompilation_timeout_seconds"] = config.precompilation_timeout_seconds

    return cache(*args, **kwargs)


def add_feedback_saver(
    fn: FeedbackFunction,
):
    cache = get_algorithm_selector_cache()
    cache.add_feedback_saver(fn)


def clear_feedback_savers():
    """Clear all feedback saver functions."""
    cache = get_algorithm_selector_cache()
    cache.clear_feedback_savers()


def add_preprocessing_fn(
    fn: PreprocessingFunction,
):
    """Add a preprocessing function to be applied to choices before autotuning.

    Preprocessing functions are called sequentially in the order they were registered,
    with each function receiving the output of the previous one. They can filter,
    reorder, transform, or modify the list of choices in any way.

    Args:
        fn: A function that takes a list of ChoiceCaller objects and returns
            a modified list of ChoiceCaller objects.

    Example:
        def my_filter(choices):
            # Filter out choices with certain names
            return [c for c in choices if 'slow' not in c.name.lower()]

        add_preprocessing_fn(my_filter)
    """
    cache = get_algorithm_selector_cache()
    cache.add_preprocessing_fn(fn)


def clear_preprocessing_fns(clear_defaults: bool = False):
    """Clear preprocessing functions at module level.

    Args:
        clear_defaults: If True, clears all functions including defaults.
                       If False, clears only user-added functions and re-registers defaults.
    """
    cache = get_algorithm_selector_cache()
    cache.clear_preprocessing_fns(clear_defaults)


def realize_inputs(*args):
    if len(args) == 1:
        return ir.ExternKernel.require_stride1(ir.ExternKernel.realize_input(args[0]))
    return [realize_inputs(x) for x in args]


class SymbolicGridFn:
    """
    Wrapper around a grid function that allows either int or sympy inputs.

        @SymbolicGridFn
        def grid(x, meta, *, cdiv):
            return cdiv(x, meta["BLOCK_X"])
    """

    def __init__(self, fn: Callable[..., tuple[Any, Any, Any]]):
        self.fn = fn
        self.kwargs_int = {}
        self.kwargs_sym = {}
        params = inspect.signature(fn).parameters
        for name, fn_sym, fn_int in [
            ("cdiv", CeilDiv, ceildiv),
            ("min", sympy.Min, min),
            ("max", sympy.Max, max),
        ]:
            if name in params:
                self.kwargs_int[name] = fn_int
                self.kwargs_sym[name] = fn_sym

    def __call__(self, *args, **kwargs) -> tuple[int, int, int]:
        return self.fn(*args, **kwargs, **self.kwargs_int)

    def sympy_call(self, *args, **kwargs):
        return self.fn(*args, **kwargs, **self.kwargs_sym)


def _autotune_metadata(input_nodes):
    """Helper function to extract autotune metadata from input nodes."""
    return {
        "autotune_strides": ", ".join([str(n.get_stride()) for n in input_nodes]),
        "autotune_dtypes": ", ".join([str(n.get_dtype()) for n in input_nodes]),
        "autotune_shape": ", ".join(
            ["x".join(map(str, n.get_size())) for n in input_nodes]
        ),
        "autotune_offset": ", ".join([str(n.get_layout().offset) for n in input_nodes]),
        # TODO(coconutruben): replace this with taking KernelInputs as the
        # argument, and extracting those out there directly
        "autotune_strides_hinted": ", ".join(
            [
                str(
                    V.graph.sizevars.size_hints(
                        n.get_stride(),
                        fallback=config.unbacked_symint_fallback,
                    )
                )
                for n in input_nodes
            ]
        ),
        "autotune_shape_hinted": ", ".join(
            [
                "x".join(
                    map(
                        str,
                        V.graph.sizevars.size_hints(
                            n.get_size(),
                            fallback=config.unbacked_symint_fallback,
                        ),
                    )
                )
                for n in input_nodes
            ]
        ),
    }


def _log_autotune_choices_stats(
    event_name: str, timings: dict[ChoiceCaller, float]
) -> None:
    """Helper function to extract autotune metadata from benchmark results."""
    if not timings:
        return None

    metadata: dict[str, Union[int, float, str]] = {
        "num_choices": len(timings),
        "num_triton_choices": len(
            [c for c in timings if isinstance(c, TritonTemplateCaller)]
        ),
    }

    sorted_choices = sorted(timings, key=timings.__getitem__)
    best_choice = sorted_choices[0]
    metadata["best_kernel"] = best_choice.name
    if best_choice.description:
        metadata["best_kernel_desc"] = best_choice.description
    metadata["best_time"] = timings[best_choice]

    best_triton_pos = next(
        (
            i
            for i, choice in enumerate(sorted_choices)
            if isinstance(choice, TritonTemplateCaller)
        ),
        None,
    )
    if best_triton_pos is not None:
        metadata["best_triton_pos"] = best_triton_pos
        best_triton_kernel = sorted_choices[best_triton_pos]
        if best_triton_pos != 0:
            metadata["best_triton_time"] = timings[best_triton_kernel]
            metadata["best_triton_kernel"] = best_triton_kernel.name
            if best_triton_kernel.description:
                metadata["best_triton_kernel_desc"] = best_triton_kernel.description

    payload = json.dumps(metadata)
    get_chromium_event_logger().add_event_data(
        event_name, autotune_choices_stats=payload
    )
    sys.stderr.write(f"Autotune Choices Stats:\n{payload}\n")


def _log_autotune_exceptions(
    exceptions: list[tuple[ChoiceCaller, BaseException]],
) -> None:
    """Log autotune exceptions to chromium event logger."""
    if not exceptions:
        return

    try:
        pt2_compile_substack = get_chromium_event_logger().get_pt2_compile_substack()
        if not pt2_compile_substack:
            return

        current_event = pt2_compile_substack[-1]
        if not current_event.endswith("_template_precompiling"):
            return

        exception_details = []
        for choice, exc in exceptions:
            try:
                choice_type = (
                    "triton" if isinstance(choice, TritonTemplateCaller) else "other"
                )
                data = {
                    "choice_type": choice_type,
                    "choice": choice.description,
                    "exception_message": str(exc),
                }

                exc_type_match = re.search(r"(\w+):", str(exc))
                if exc_type_match:
                    data["exception"] = exc_type_match.group(1)

                if "OutOfMemoryError" in str(exc):
                    required_match = re.search(r"Required: (\d+)", str(exc))
                    if required_match:
                        data["required_memory"] = required_match.group(1)

                    limit_match = re.search(r"Hardware limit:\s*(\d+)", str(exc))
                    if limit_match:
                        data["hardware_limit"] = limit_match.group(1)

                exception_details.append(data)
            except Exception:
                # Don't let logging errors break the main flow
                continue

        if exception_details:
            metadata = json.dumps({"exceptions": exception_details})
            get_chromium_event_logger().try_add_event_data(
                current_event, metadata=metadata
            )
    except Exception:
        # Silently ignore logging errors to avoid breaking autotune
        pass


# ensure lowering is imported so that `extern_kernels.*` is populated
from . import lowering  # noqa: F401
