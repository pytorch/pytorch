# mypy: allow-untyped-defs
import contextlib
import dataclasses
import functools
import itertools
import operator
import textwrap
import typing
from typing import Any, Callable, Optional, Union

import sympy

import torch
from torch._dynamo.utils import identity
from torch._inductor.utils import (
    FakeIndentedBuffer,
    get_dtype_size,
    sympy_dot,
    sympy_product,
    triton_type,
    triton_type_to_torch,
    unique,
)
from torch.utils._ordered_set import OrderedSet

from ... import config, ir
from ...codegen.common import CSEVariable, IndentedBuffer, OpOverrides, WorkspaceArg
from ...codegen.simd_kernel_features import SIMDKernelFeatures
from ...codegen.triton import gen_common_triton_imports, texpr, TritonKernel
from ...codegen.triton_utils import config_of, equal_1_arg_indices, signature_to_meta
from ...codegen.wrapper import pexpr
from ...ops_handler import StoreMode
from ...runtime.hints import DeviceProperties
from ...runtime.triton_compat import HAS_WARP_SPEC
from ...runtime.triton_heuristics import FixedGrid
from ...utils import Placeholder
from ...virtualized import V
from .common import SymbolicGridFn


if typing.TYPE_CHECKING:
    from torch._inductor.codegen.simd import IterationRangesRoot

# Function name, followed by args and kwargs.
RecordedEventsType = list[tuple[str, list[Any], dict[str, Any]]]


@dataclasses.dataclass()
class SubgraphInfo:
    body: IndentedBuffer
    template_mask: Optional[str] = None
    template_out: Optional[str] = None
    compute: IndentedBuffer = dataclasses.field(default_factory=IndentedBuffer)
    indexing_code: IndentedBuffer = dataclasses.field(default_factory=IndentedBuffer)
    loads: IndentedBuffer = dataclasses.field(default_factory=IndentedBuffer)
    stores: IndentedBuffer = dataclasses.field(default_factory=IndentedBuffer)
    ops_handler: Optional[V.WrapperHandler] = None  # type: ignore[name-defined]

    # only copied over if not None
    range_trees: Optional[list["IterationRangesRoot"]] = None
    numels = None  # type: ignore[var-annotated]

    def __post_init__(self):
        self.only_copy_if_non_none_fields = ("range_trees", "numels")

    def to_dict(self):
        return {
            field.name: getattr(self, field.name) for field in dataclasses.fields(self)
        }


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
            var_dtype = V.graph.get_buffer(name).dtype
            line = f"tl.load({var} + {index_str})"

            if (
                var_dtype in (torch.float16, torch.bfloat16)
                and config.triton.codegen_upcast_to_fp32
            ):
                line += ".to(tl.float32)"
                var_dtype = torch.float32

            out = self.kernel.cse.generate(self.kernel.compute, line, dtype=var_dtype)
            return out

        return self.kernel.cse.generate(
            self.kernel.compute, f"({self.fixed_inputs[name]})", dtype=torch.float32
        )

    def indirect_indexing(self, index_var: str, size, check, wrap_neg=True):
        """Convert index variable to symbolic form."""
        return sympy.Symbol(str(index_var), integer=True)

    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
    ) -> str:
        """Currently only supports stores for atomic adds coming from scatter nodes
        This is used by flex_attention's backwards grad for captured buffers, see
        zeros_and_scatter lowering
        """
        assert (
            self.mask is not None
        ), "Mask is required for inner stores in modifications"
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
        num_consumer_groups=0,
        num_buffers_warp_spec=0,
        use_jit=False,
        prefix_args=0,
        suffix_args=0,
        epilogue_fn=identity,
        subgraphs: Optional[list[ir.ComputedBuffer]] = None,
        workspace_arg: Optional[WorkspaceArg] = None,
        prologue_loads_all_inputs=False,
    ) -> None:
        numel = sympy_product(output_node.get_size())
        super().__init__(
            {
                "x": numel,
                "r0_": sympy.S.One,
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
        self.template_out: Optional[str] = None
        self.ops_handler: Optional[V.WrapperHandler] = None  # type: ignore[name-defined]

        # Whe caching is enabled, the generated code is not dependent on the input nodes names, or
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
    def create_subgraph_body(self, body_name: str):
        assert body_name not in self.subgraph_bodies
        self.subgraph_bodies[body_name] = SubgraphInfo(
            IndentedBuffer(),
            None,
            None,
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
            size = V.graph.sizevars.size_hints(inp.get_size())
            numel = functools.reduce(operator.mul, size, 1)
            dtype_size = get_dtype_size(inp.get_dtype())
            num_bytes.append(numel * dtype_size * (1 + int(i < ninplace_args)))
        return sum(num_bytes)

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
            **TritonKernel.inductor_meta_common(),
            **FixedGrid.setup_grid_as_args(),
        }
        if config.profile_bandwidth or config.benchmark_kernel:
            num_gb = self.estimate_kernel_num_bytes() / 1e9
            inductor_meta["kernel_num_gb"] = num_gb

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
                assert isinstance(
                    subgraph, (ir.ComputedBuffer, list)
                ), f"Expected the subgraph to be a ComputedBuffer or a List[ComputedBuffer], got {type(subgraph)}"
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
            self.template_out = "xindex"
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
                    copy_shape=self.template_out,
                    override_mask=self.template_mask,
                )
                from ..triton import IndexingOptions

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

        assert hook_key not in self.render_hooks
        self.render_hooks[hook_key] = hook
        return hook_key

    def store_output(
        self,
        indices: Union[list[Any], tuple[Any]],
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

            acc_dtype = (
                triton_type_to_torch(self.meta["ACC_TYPE"])
                if "ACC_TYPE" in self.meta
                else torch.float32
            )
            epilogue_args = [V.kernel.cse.namedvar(val, dtype=acc_dtype)]
            for input_node in itertools.chain(
                self.input_nodes[: self.prefix_args],
                self.input_nodes[len(self.input_nodes) - self.suffix_args :],
            ):
                input_node.freeze_layout()
                epilogue_args.append(input_node.make_loader()(index_symbols))
                # We update frozen_layouts_cnt in order to replay this function on a cache hit.
                self.frozen_layouts_cnt += 1

            V.ops.store(
                self.output_node.get_name(),
                output_index,
                self.epilogue_fn(*epilogue_args),
            )
            self.codegen_body()

        def hook():
            # more stuff might have been added since the codegen_body above
            self.codegen_body()
            self.cse.invalidate(OrderedSet())

            return textwrap.indent(self.body.getvalue(), " " * indent_width).strip()

        assert "<STORE_OUTPUT>" not in self.render_hooks
        self.render_hooks["<STORE_OUTPUT>"] = hook
        return "<STORE_OUTPUT>"

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

        grid_args = ()
        if isinstance(self.grid_fn, SymbolicGridFn):
            grid_args = self.grid_fn.sympy_call(*self.call_sizes, self.meta)
        elif all(isinstance(x, (int, sympy.Integer)) for x in self.call_sizes):
            grid_args = self.grid_fn(*map(int, self.call_sizes), self.meta)
        else:
            assert not V.graph.cpp_wrapper, "cpp_wrapper requires SymbolicGridFn"
            wrapper.add_import_once(f"import {self.grid_fn.__module__}")
            meta = wrapper.add_meta_once(self.meta)
            fn_name = f"{self.grid_fn.__module__}.{self.grid_fn.__name__}"
            call_args.append(
                f"*{fn_name}({', '.join(map(pexpr, self.call_sizes))}, {meta})"
            )
            arg_types.append(None)
        assert len(grid_args) in (0, 3), "grid_fn should return 3 values"
        call_args.extend(grid_args)
        arg_types.extend(map(type, grid_args))

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
