# mypy: allow-untyped-defs
import functools
import itertools
import logging
import operator
import os
import re
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TYPE_CHECKING,
    Union,
)

import sympy

import torch
import torch._logging
import torch.fx
from torch._decomp import get_decompositions
from torch._dynamo.utils import defake, dynamo_timed
from torch._logging import LazyString, trace_structured
from torch._prims_common import make_channels_last_strides_for
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental._backward_state import BackwardState
from torch.fx.experimental.sym_node import magic_methods, method_to_operator
from torch.fx.experimental.symbolic_shapes import (
    free_unbacked_symbols,
    has_free_symbols,
    resolve_unbacked_bindings,
    RuntimeAssert,
    ShapeEnv,
    SymTypes,
)
from torch.utils._mode_utils import no_dispatch
from torch.utils._sympy.numbers import int_oo

from . import config, ir
from .codegen.common import (
    BackendFeature,
    DeviceOpOverrides,
    get_backend_features,
    get_device_op_overrides,
    get_wrapper_codegen_for_device,
    init_backend_registration,
)
from .exc import (
    CppWrapperCodeGenError,
    LoweringException,
    MissingOperatorWithDecomp,
    MissingOperatorWithoutDecomp,
)
from .ir import (
    Constant,
    FixedLayout,
    get_device_type,
    InputBuffer,
    Pointwise,
    Reduction,
    StorageBox,
    TensorBox,
    TorchBindObject,
)
from .lowering import (
    constrain_to_fx_strides,
    FALLBACK_ALLOW_LIST,
    fallback_handler,
    fallback_node_due_to_unsupported_type,
    layout_constraints,
    lowerings,
    make_fallback,
    needs_realized_inputs,
    unsupported_output_tensor,
)
from .sizevars import SizeVarAllocator
from .utils import (
    convert_shape_to_inductor,
    gather_origins,
    get_cloned_parameter_buffer_name,
    get_sympy_Expr_dtype,
    maybe_get_suppress_shape_guards_ctx,
    should_assume_input_aligned,
)
from .virtualized import NullHandler, V

if TYPE_CHECKING:
    from torch._higher_order_ops.effects import _EffectType
    from .codegen.wrapper import WrapperCodeGen

from torch._inductor.codecache import output_code_log

log = logging.getLogger(__name__)
perf_hint_log = torch._logging.getArtifactLogger(__name__, "perf_hints")

aten = torch.ops.aten

_post_grad_graph_counter = itertools.count()

if config.is_fbcode():
    from torch._inductor.fb.utils import log_module_code
else:

    def log_module_code(*args, **kwargs):
        pass


def supported_dtype_of_cpp_wrapper(dtype, cuda):
    supported_dtype = {
        torch.float32,
        torch.float64,
        torch.int64,
        torch.int32,
        torch.int16,
        torch.int8,
        torch.uint8,
        torch.bool,
        torch.bfloat16,
        torch.complex32,
        torch.complex64,
        torch.complex128,
        torch.float16,
    }
    if cuda:
        supported_dtype.add(torch.float8_e4m3fn)
        supported_dtype.add(torch.float8_e5m2)
        supported_dtype.add(torch.float8_e4m3fnuz)
        supported_dtype.add(torch.float8_e5m2fnuz)

    return dtype in supported_dtype


def may_get_constant_buffer_dtype(constant_buffer):
    assert isinstance(
        constant_buffer, (sympy.Symbol, sympy.Expr, sympy.core.numbers.Integer)
    ), "get_constant_buffer_dtype only supports input of sympy.Symbol, sympy.Expr or sympy.core.numbers.Integer"
    if isinstance(constant_buffer, sympy.core.numbers.Integer):
        return torch.int64

    if isinstance(constant_buffer, sympy.Expr):
        return get_sympy_Expr_dtype(constant_buffer)

    if constant_buffer.is_integer:
        return torch.int64
    elif constant_buffer.is_float:
        return torch.float32
    else:
        return None


def is_magic_method(op):
    magic_ops = {method_to_operator(m) for m in magic_methods}
    return op in magic_ops


def getattr_recursive(obj, target):
    target_atoms = target.split(".")
    attr_itr = obj
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(
                f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}"
            )
        attr_itr = getattr(attr_itr, atom)
    return attr_itr


def mark_nodes_dislike_padding(g):
    """
    Nodes like convolution/convolution_backward want its input to be dense.
    If we pad their inputs, we result in extra calls to copy kernels!  On the other hand, padding usually helps reduction.

    The pass finds nodes that dislike padding. These are nodes that can be reached
    from a convolution/convolution_backward in the backward direction without
    going thru a reduction.
    """
    if not config.comprehensive_padding:
        return
    ops_dislike_padding = {
        aten.convolution,
        aten.convolution_backward,
    }
    # what's a better way to collect the reduction ops?
    ops_like_padding = {
        aten.var_mean,
        aten.sum,
        aten.mean,
        aten.prod,
        aten.any,
        aten.amin,
        aten.amax,
        aten.min,
        aten.max,
        aten.argmin,
        aten.argmax,
        aten.scatter_reduce,
    }

    def _get_overload_packet(node):
        return (
            node.target._overloadpacket
            if node.op == "call_function" and hasattr(node.target, "_overloadpacket")
            else None
        )

    for cur in reversed(g.nodes):
        op = _get_overload_packet(cur)
        if not op:
            continue
        if op in ops_dislike_padding:
            cur.meta["dislike_padding"] = True

        if cur.meta.get("dislike_padding", False):
            # propagate
            for prior in cur.all_input_nodes:
                prior_op = _get_overload_packet(prior)
                if not prior_op:
                    continue
                if prior_op not in ops_like_padding:
                    prior.meta["dislike_padding"] = True


class GraphLowering(torch.fx.Interpreter):
    graph_outputs: List[ir.IRNode]

    def symbolic_sizes_strides(self, ex: torch.Tensor):
        """
        Support dynamic shapes and dynamic strides by assigning variables
        to each dimension.  We duck-shape tensors, so if two tensors
        have the same size they get assigned the same symbolic variable.
        """
        if self.reuse_shape_env:
            return convert_shape_to_inductor(ex.size()), convert_shape_to_inductor(
                ex.stride()
            )
        else:
            from torch._dynamo.source import ConstantSource

            # TODO: this should not be needed once #93059 lands
            # https://github.com/pytorch/pytorch/pull/94031#discussion_r1096044816
            # TODO: make a dedicated UnknownSource for this?
            # NB: This is using the legacy default behavior from
            # create_symbolic_sizes_strides_storage_offset but we hope we can
            # just delete this entirely
            source = ConstantSource(
                f"__inductor_unknown_tensor_{len(self._shape_env.var_to_val)}"
            )
            (
                size,
                stride,
                _,
            ) = self._shape_env.create_symbolic_sizes_strides_storage_offset(
                ex,
                source,
            )

        size = [i.node.expr if isinstance(i, torch.SymInt) else i for i in size]
        stride = [i.node.expr if isinstance(i, torch.SymInt) else i for i in stride]
        return size, stride

    def static_sizes_strides(self, ex: torch.Tensor):
        """
        Primarily used to weights
        """
        size = [sympy.Integer(i) for i in ex.size()]
        stride = [sympy.Integer(i) for i in ex.stride()]
        return size, stride

    def __init__(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: Optional[List[torch.Tensor]] = None,
        shape_env=None,
        graph_id=None,
        cpp_wrapper=False,
        aot_mode=False,
        user_visible_outputs=None,
        layout_opt=None,
        extern_node_serializer=None,
        is_inference=False,
        is_const_graph=False,
        const_output_index=None,
        const_code=None,
        const_module=None,
        name=None,
    ):
        super().__init__(gm)
        self.example_inputs = example_inputs
        self.layout_opt = (
            layout_opt
            if layout_opt is not None
            else self.decide_layout_opt(gm, is_inference=is_inference)
        )
        self.num_channels_last_conv = 0
        self.is_inference = is_inference
        self.is_const_graph = is_const_graph
        self.const_code = const_code
        self.const_module = const_module

        self.extra_traceback = False  # we do our own error wrapping
        if shape_env is None:
            shape_env = ShapeEnv()
            self.reuse_shape_env = False
        else:
            self._shape_env = shape_env
            self.reuse_shape_env = True
        self._shape_env = shape_env
        # We are going to start code generating runtime asserts, so make sure
        # you don't start adding new ones in the lowering process
        shape_env.freeze_runtime_asserts()
        # We're going to mutate ras_by_symbol as we finish generating them
        self.ras_by_symbol: Dict[
            sympy.Symbol, List[RuntimeAssert]
        ] = shape_env.deferred_runtime_asserts.copy()
        self.bound_unbacked_symbols: Set[sympy.Symbol] = set()
        self.sizevars = SizeVarAllocator(shape_env)
        self.graph_input_names: List[str] = []
        self.graph_inputs: Dict[str, TensorBox] = {}
        self.graph_inputs_original: Dict[str, InputBuffer] = {}
        self.device_types: Set[str] = (
            const_module.device_types if const_module else set()
        )
        self.device_idxs: Set[int] = const_module.device_idxs if const_module else set()
        self.cuda = False
        self.buffers: List[ir.Buffer] = []
        self.operations: List[ir.Operation] = []
        self.const_output_index: Dict[str, int] = (
            const_output_index if const_output_index else {}
        )
        self.folded_constants: Set[str] = (
            set(const_output_index.keys()) if const_output_index else set()
        )
        self.constants: Dict[str, torch.Tensor] = (
            const_module.constants if const_module else {}
        )
        self.torchbind_constants: Dict[str, torch._C.ScriptObject] = {}
        self.constant_reprs: Dict[str, str] = {}
        self.removed_operations: Set[str] = set()
        self.removed_buffers: Set[str] = set()
        self.removed_inplace_buffers: Set[str] = set()
        self.mutated_buffers: Set[str] = set()
        self.never_reuse_buffers: Set[str] = set()
        self.inplaced_to_remove: Set[str] = set()
        self.device_ops: DeviceOpOverrides = None  # type: ignore[assignment]
        self.wrapper_code: WrapperCodeGen = None  # type: ignore[assignment]
        # See `ProxyExecutor Design Note` in ir.py for more details
        self.extern_kernel_nodes: List[ir.ExternKernelNode] = []
        self.extern_node_serializer: Optional[
            Callable[[List[ir.ExternKernelNode]], Any]
        ] = extern_node_serializer
        self.current_node: torch.fx.Node = None  # type: ignore[assignment]
        self.lists: Dict[str, List[str]] = {}
        self.mutated_inputs: Set[str] = set()
        self.mutated_input_idxs: List[int] = []
        self.name_to_buffer: Dict[str, ir.Buffer] = {}
        self.name_to_users: DefaultDict[str, List[ir.IRNode]] = defaultdict(list)
        self.name_to_op: Dict[str, ir.Operation] = {}
        self.creation_time = time.time()
        self.name = name
        self.cpp_wrapper = cpp_wrapper

        # record multi_kernel choice for cpp_wrapper so the second pass knows
        # which sub-kernel is picked. Copy cpp_wrapper to another variable
        # since cpp_wrapper flag is set to false for the first pass of codegen.
        self.record_multi_kernel_choice = cpp_wrapper
        self.multi_kernel_to_choice: Dict[str, int] = {}

        self.aot_mode = aot_mode
        self.graph_id = graph_id
        self.post_grad_graph_id = next(_post_grad_graph_counter)
        self.scheduler: torch._inductor.scheduler.Scheduler = None  # type: ignore[assignment]
        self.nodes_prefer_channels_last = (
            self.find_nodes_prefer_channels_last() if self.layout_opt else set()
        )
        mark_nodes_dislike_padding(gm.graph)
        self._warned_fallback = {"aten.convolution_backward"}
        self.user_visible_outputs = (
            user_visible_outputs if user_visible_outputs is not None else {}
        )
        self.cache_key: str = ""  # This is the cache key for the compiled artifact
        self.cache_path: str = ""  # This is the path in the filesystem where the compiled artifact is stored
        self.cache_linemap: List[
            Tuple[int, str]
        ] = (
            []
        )  # This is the linemap used by the profiler to mark custom compiled kernels getting run
        # Used if lowering encounters cases where cudagraphs are not supported
        self.disable_cudagraphs_reason: Optional[str] = None

        # only keeping one node per device for stack trace purposes
        self.device_node_mapping: Dict[torch.device, torch.fx.Node] = {}
        self.orig_gm: torch.fx.GraphModule = gm.__copy__()
        self.dynamo_flat_name_to_original_fqn = self.module.meta.get(
            "dynamo_flat_name_to_original_fqn", {}
        )
        self.allocated_constant_name = (
            const_module.allocated_constant_name if const_module is not None else {}
        )
        init_backend_registration()
        self.get_backend_features = functools.lru_cache(None)(get_backend_features)

        self.effectful_ops: Dict[_EffectType, ir.Buffer] = {}
        self.aligned_inputs: Set[str] = set()
        self.no_fuse_buffer_names: Set[str] = set()

    def has_feature(self, device, feature):
        assert isinstance(feature, BackendFeature), feature
        return feature in self.get_backend_features(get_device_type(device))

    @staticmethod
    def decide_layout_opt(gm, *, is_inference) -> bool:
        """
        Decide if we should enable layout optimization for this graph based on
        heuristics.
        """
        if not config.layout_optimization:
            return False

        if config.force_layout_optimization:
            return True

        conv_nodes = [
            n for n in gm.graph.nodes if n.target == torch.ops.aten.convolution.default
        ]
        nconv = len(conv_nodes)

        if nconv == 0:
            return False

        # For cpu backend and mkldnn enabled, we always use channels_last for better performance.
        if (
            torch.backends.mkldnn.enabled
            and torch.backends.mkldnn.is_available()
            and all(
                n.args[idx].meta["val"].device == torch.device("cpu")
                for n in conv_nodes
                for idx in [0, 1]
            )
        ):
            return True

        # Following models are skipped due to this:
        # jx_nest_base
        # volo_d1_224
        if len(list(gm.graph.nodes)) >= 300 * nconv:
            log.debug("Skipped layout opt because only a few conv")
            return False

        if any(
            has_free_symbols(n.args[idx].meta["val"])
            for n in conv_nodes
            for idx in [0, 1]
        ):
            log.debug(
                "See perf regression with dynamic shape. Follow up in https://github.com/pytorch/pytorch/issues/102670"
            )
            return False

        def is_grouped(n):
            return n.args[-1] > 1 and n.args[1].meta["val"].size(1) > 1

        def is_in_out_channel(n):
            return (
                n.args[1].meta["val"].size(0) * 2 <= n.args[1].meta["val"].size(1)
                and n.args[1].meta["val"].size(2) > 1
            )

        def is_small_channel(n):
            return (
                n.args[1].meta["val"].size(0) <= 64
                and n.args[1].meta["val"].size(1) <= 64
            )

        # only grouped convolutions benchmarked as slower in conv samples for inference only
        if is_inference:
            from torch.utils.flop_counter import FlopCounterMode

            flop_counts: Dict[str, float] = defaultdict(float)
            for node in conv_nodes:
                success, args, kwargs = torch._inductor.fx_utils.get_fake_args_kwargs(
                    node
                )

                if success:
                    with FlopCounterMode(display=False) as flop_counter_mode:
                        with V.fake_mode:
                            node.target(*args, **kwargs)

                    counted_flops = flop_counter_mode.get_total_flops()
                    if is_grouped(node):
                        node_type = "grouped"
                    elif is_small_channel(node):
                        node_type = "small"
                    elif is_in_out_channel(node):
                        node_type = "in_out"
                    else:
                        node_type = "default"

                    flop_counts[node_type] += counted_flops
                else:
                    log.debug("Conv inputs meta not found")

            # average benchmarked channels last speedup / slowdown, < 1 is speedup.
            # taken from the set of convolution inputs in benchmarks/dynamo/microbenchmarks/operator_inp_logs/torchbench_train/
            # To regenerate these numbers follow https://gist.github.com/eellison/55d7a6ed6f39829d68ac56f95f4df5bb
            GROUPED_MULTIPLIER = 1.358
            DEFAULT_MULTIPLIER = 0.823
            IN_OUT_MULTIPLIER = 0.725
            SMALL_MULTIPLIER = 0.783

            total_flops = sum(flop_counts.values())
            # TODO - get different values per hardware
            weighted_flops = (
                flop_counts["grouped"] * GROUPED_MULTIPLIER
                + flop_counts["small"] * SMALL_MULTIPLIER
                + flop_counts["in_out"] * IN_OUT_MULTIPLIER
                + flop_counts["default"] * DEFAULT_MULTIPLIER
            )
            do_layout_opt = weighted_flops <= total_flops
            if not do_layout_opt:
                log.debug(
                    "Skipped layout opt in inference because weighted flops indicate slowdown, default: %d, channels last: %d",
                    total_flops,
                    weighted_flops,
                )
            return do_layout_opt

        # Channels last layout can dramatically hurt grouped conv perf. E.g.
        # Conv with arguments like
        #   {"input_shape": [32, 224, 112, 112], "weight_shape": [224, 112, 3, 3],
        #    "stride": [2, 2], "padding": [1, 1], "groups": 2}
        # slows down 31x using channels last..

        # But a lot of timm models use depthwise separable convolution which will
        # result in grouped convolution with in-channel size == 1.
        # For those grouped convolution, channels last still helps a lot.
        # E.g.
        # Conv with arguments
        #   {"input_shape": [128, 58, 56, 56], "weight_shape": [58, 1, 3, 3],
        #    "stride": [2, 2], "padding": [1, 1], "groups": 58}
        # get 1.86x speedup with channels last layout.
        #
        # The following heuristics skip using channels-last if the model contains
        # grouped convolution with in-channels > 1.
        if any(map(is_grouped, conv_nodes)):
            log.debug(
                "Skip layout opt because found grouped convolution with >1 in_channels!"
            )
            return False

        # For some models that contain convolution with larger in-channel than out-channel, applying
        # channels last hurts performance.
        # Following models are skipped due to this:
        # - pytorch_unet
        # - phlippe_densenet (slightly worse)
        # - Background_Matting (1.22x -> 0.821x)
        # - pytorch_CycleGAN_and_pix2pix (1.597x -> 1.294x)
        if any(map(is_in_out_channel, conv_nodes)):
            log.debug(
                "Skip layout opt because some convolutions have smaller out_channel"
            )
            return False

        # Following models are skipped due to this:
        # - functorch_maml_omniglot
        if all(map(is_small_channel, conv_nodes)):
            log.debug("Skip layout opt because all convolution channels are too small")
            return False

        return True

    def qualify_name(self, name: str) -> str:
        """Prepend the given name with the graph name if any."""
        if self.name is not None:
            return f"{self.name}_{name}"
        return name

    def make_subgraph(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: List[torch.Tensor],
        subgraph_name: str,
    ) -> "GraphLowering":
        """
        Make a subgraph of the current graph with all inherited
        parts, except the graph module (`gm`) and `example_inputs`.
        The subgraphs are lowered separately, but intended to be
        inlined in the parent graph's codegening. Hence the need
        for maintaining the same `shape_env` and other properties.
        The subgraph name is qualified by the parent graph's name.
        """
        return GraphLowering(
            gm=gm,
            example_inputs=example_inputs,
            shape_env=self._shape_env,
            cpp_wrapper=self.cpp_wrapper,
            aot_mode=self.aot_mode,
            extern_node_serializer=self.extern_node_serializer,
            is_inference=self.is_inference,
            name=self.qualify_name(subgraph_name),
        )

    def find_nodes_prefer_channels_last(self):
        """
        The rule to decide if an node prefer channels last is simple.
        1. if it's input/output of a convolution
        2. if one of its user prefers channels last

        We have rule 1 because cudnn runs a faster convolution kernel for channels last inputs;
        Rule 2 is also important. It makes sure that indirect inputs to convolution also prefers
        channels last.

        Consider the scenario: conv -> batch-norm -> relu -> conv
        Without rule 2, batch-norm output may use a contiguous layout. That will cause 2 extra copies:
        1. the output of batch-norm should be channels last initially since its input is a conv's output.
           Forcing the batch-norm's output to be contiguous results in the first copy
        2. The second conv's input is initially contiguous. This layout is propagated from the batch-norm's output.
           We need convert it to channels last layout which results in the second copy.
        With rule 2, we makes sure all the tensors in the chain uses channels last layout. So both copies
        can be saved.
        """
        output_set = set()
        for n in reversed(self.module.graph.nodes):
            if n.target == torch.ops.aten.convolution.default:
                output_set.add(n)
                continue

            for user in n.users:
                if user in output_set:
                    output_set.add(n)
                    break

        # need a second pass to add downstream nodes of those channel last nodes to the sets.
        # This pass is especially needed to avoid mix-layout kernel inputs in backward pass.
        #
        # Let's say a conv-batchnorm 's output is passed to relu whose output is in turn returned
        # from the fwd graph. Without this second pass, we will force relu's output to be contiguous.
        # Then in the kernel in backward pass, the contiguous output of relu may be mix with other channels last
        # tensors and passed to a kernel.
        #
        # This pass improve yolov3 training speedup from 1.116x (worse than disabling layout optimization speedup 1.196x) to 1.457x.
        # It also improves dla102 training speedup from 1.240x (worse than disabling layout optimization speedup 1.523x) to 1.835x .
        # This also helps the following models:
        # - res2net101_26w_4s
        # - res2net50_14w_8s
        # - sebotnet33ts_256
        for n in self.module.graph.nodes:
            if n in output_set:
                output_set.update(n.users)

        return output_set

    def warn_fallback(self, name):
        if name not in self._warned_fallback:
            self._warned_fallback.add(name)
            perf_hint_log.info("Using FallbackKernel: %s", name)

    def add_device_info(self, device: torch.device):
        self.device_types.add(device.type)
        if device.index is not None:
            self.device_idxs.add(device.index)
        if V.graph.current_node and device not in self.device_node_mapping:
            self.device_node_mapping[device] = V.graph.current_node

    @property
    def fake_mode(self):
        return V.fake_mode

    def get_buffer(self, buffer_name: str):
        if buffer_name in self.name_to_buffer:
            return self.name_to_buffer[buffer_name]
        if buffer_name in self.graph_inputs:
            return self.graph_inputs[buffer_name]
        if buffer_name in self.constants:
            data = V.graph.constants[buffer_name]
            return ir.ConstantBuffer(
                buffer_name,
                ir.FixedLayout(
                    data.device, data.dtype, *V.graph.static_sizes_strides(data)
                ),
            )
        return None

    def get_dtype(self, buffer_name: str):
        if buffer_name in self.constants:
            return self.constants[buffer_name].dtype
        if buffer_name in self.name_to_buffer:
            return self.name_to_buffer[buffer_name].get_dtype()
        if buffer_name in self.graph_inputs:
            return self.graph_inputs[buffer_name].get_dtype()
        m = re.match(r"(as_strided|reinterpret_tensor)\(([a-zA-Z0-9_]+),", buffer_name)
        if m:
            return self.get_dtype(m.group(1))
        raise KeyError(f"could not find {buffer_name}")

    def get_numel(self, buffer_name: str):
        from .ir import MultiOutputLayout

        if buffer_name in self.constants:
            return self.constants[buffer_name].numel()
        if buffer_name in self.name_to_buffer:
            buf = self.name_to_buffer[buffer_name]
            if isinstance(getattr(buf, "layout", None), MultiOutputLayout):
                return 1
            return buf.get_numel()
        if buffer_name in self.graph_inputs:
            return self.graph_inputs[buffer_name].get_numel()
        raise KeyError(f"could not find {buffer_name}")

    @dynamo_timed
    def run(self, *args):
        return super().run(*args)

    def register_operation(self, op: ir.Operation):
        assert op.operation_name is None, f"Operation registered twice: {op}"
        assert isinstance(op, ir.Operation)
        name = self.qualify_name(f"op{len(self.operations)}")
        self.operations.append(op)
        self.name_to_op[name] = op
        op.operation_name = name
        return name

    def register_buffer(self, buffer: ir.Buffer, *, set_name: bool = False):
        name = self.qualify_name(f"buf{len(self.buffers)}")
        self.buffers.append(buffer)
        self.name_to_buffer[name] = buffer
        # Skip empty CPU tensor so that CUDA graphs can succeed, see https://github.com/pytorch/pytorch/pull/114144
        if (
            not (isinstance(buffer, ir.ComputedBuffer) and buffer.is_zero_elements())
            and buffer.get_device() is not None
        ):
            self.add_device_info(buffer.get_device())

        if set_name:
            buffer.name = name
        return name

    def register_operation_list(self, operation_names: List[str]) -> str:
        name = self.qualify_name("list_" + "_".join(operation_names))
        self.lists[name] = operation_names
        return name

    def register_users_of(self, node_output):
        def register(value):
            if isinstance(value, (list, tuple)):
                for x in value:
                    register(x)
            if isinstance(value, ir.TensorBox):
                for read_name in value.get_read_names():
                    self.name_to_users[read_name].append(value)

        register(node_output)

    def mark_buffer_mutated(self, name: str):
        """
        When a buffer is mutated we need to make sure all the reads to
        the old version are realized before the mutation happens.
        """
        assert isinstance(name, str)
        self.mutated_buffers.add(name)

        if name not in self.name_to_users:
            return

        for user in self.name_to_users[name]:
            user.realize()

    def get_original_value_of_constant(self, name: str):
        """
        In AOTI, module buffers may have been mutated during the tracing and compilation.
        Thus we need to read from previously stored original buffers, to make sure the
        generated model.so uses correct initial values.
        """
        assert name in self.allocated_constant_name and name in self.constants, (
            "Can not find the original value for " + name
        )
        orig_name = get_cloned_parameter_buffer_name(self.allocated_constant_name[name])
        return (
            self.module.meta[orig_name]
            if orig_name in self.module.meta
            else self.constants[name]
        )

    def allocate_non_dup_const_name(self, name, data):
        orig_name = name
        if not config.aot_inductor.use_runtime_constant_folding:
            for constant_name, value in self.constants.items():
                if (
                    not data.is_mkldnn
                    and data.size() == value.size()
                    and data.stride() == value.stride()
                    and data.dtype == value.dtype
                    and data.device == value.device
                    and data.untyped_storage().data_ptr()
                    == value.untyped_storage().data_ptr()
                    and data.storage_offset() == value.storage_offset()
                ):
                    return constant_name

        if name is None:
            name = f"constant{len(self.constants)}"
        if name[0].isdigit():
            name = f"constant_{name}"
        name = self.qualify_name(name)
        # We may generate a var name for each constant in the codegen.
        # Let's only keep sane characters.
        prefix = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        name = prefix
        cnt = 0
        while name in self.constants:
            name = f"{prefix}_{cnt}"
            cnt += 1
        self.constants[name] = data
        self.constant_reprs[name] = (
            f"{data.device!r} {data.dtype!r} "
            f"{tuple(data.size())!r} {tuple(data.stride())!r} "
            f"{hash(data):x}"
        )
        self.allocated_constant_name[name] = orig_name
        return name

    def add_tensor_constant(self, data, name=None):
        new_name = self.allocate_non_dup_const_name(name, data)
        return TensorBox.create(
            ir.ConstantBuffer(
                new_name,
                FixedLayout(data.device, data.dtype, *self.static_sizes_strides(data)),
            )
        )

    def constant_name(self, name: str, device_override: Optional[torch.device]):
        """
        We AOT copy constants to the devices they are needed on.
        If device_override doesn't match the constant's device, then
        copy it and return a different name.
        """
        if self.constants[name].device == device_override or device_override is None:
            return name
        with torch.utils._python_dispatch._disable_current_modes():
            # caller might have set fake tensor mode which will create a fake tensor
            # when calling .to, so unset modes here
            return self.allocate_non_dup_const_name(
                f"{name}_{device_override.type}{device_override.index or 0}",
                self.constants[name].to(device_override),
            )

    def placeholder(self, target: str, args, kwargs):
        example = super().placeholder(target, args, kwargs)
        self.graph_input_names.append(target)
        if isinstance(example, SymTypes):
            expr = example.node.expr
            self.graph_inputs[target] = expr
            return expr
        elif isinstance(example, (int, bool, float)):
            expr = sympy.sympify(example)
            self.graph_inputs[target] = expr
            return expr
        if isinstance(example, BackwardState):
            # Ignored arg, must be unused
            # Alternately we could filter this out in AotAutograd
            return None
        assert isinstance(example, torch.Tensor), example
        # todo(chilli): We can remove the last check once we turn buffers into
        # static shape tensors. That's a hack to workaround Inductor believing
        # the buffer should be static but us passing in a fake tensor with
        # symbolic shapes.
        if not example._has_symbolic_sizes_strides:
            # the first N inputs are weights
            sizes, strides = self.static_sizes_strides(example)
        else:
            sizes, strides = self.symbolic_sizes_strides(example)
        # TODO(jansel): handle input aliasing
        target = self.qualify_name(target)
        tensor = TensorBox.create(
            InputBuffer(
                target,
                FixedLayout(example.device, example.dtype, sizes, strides),
            )
        )
        self.graph_inputs[target] = tensor
        self.graph_inputs_original[target] = tensor.data.data
        self.add_device_info(example.device)

        # Note: [Input Alignment handling in Inductor]
        # Alignment matters for generating efficient code. Some operations,
        # e.g. vectorized loads, can only be performed on aligned inputs.
        #
        # But if we codegen assuming aligned inputs and then get unaligned
        # inputs at runtime, then we are forced to clone - which is bad for
        # both perf and memory usage.
        #
        # One option would be to guard on storage_offset%ALIGNMENT, and then
        # codegen based on this. But storage_offset guards turned out to be
        # expensive and cause recompiles; Instead, we're generating code
        # based on the alignment of the example input without guarding.
        with maybe_get_suppress_shape_guards_ctx():
            if should_assume_input_aligned(example):
                self.aligned_inputs.add(target)
        return tensor

    def call_function(self, target, args, kwargs):
        if target is operator.getitem and isinstance(args[0], (list, tuple, dict)):
            return super().call_function(target, args, kwargs)

        if hasattr(target, "_inductor_lowering_function"):
            # passthrough lowerings from .pattern_matcher
            return target(*args, **kwargs)

        def get_custom_op_layout_constraints(target, args, kwargs):
            # Custom operations that require preserving stride order
            # which run through implicit fallback must constrain their
            # arguments' fx strides
            layout_constraint = None
            if torch._C.Tag.needs_fixed_stride_order in target.tags:
                # We have to set the current args because call_function will immediately
                # evaluate this lowering after creating the fallback, without evaluating
                # the layout constraint
                args, kwargs = constrain_to_fx_strides(
                    self.current_node, *args, **kwargs
                )
                # Also register the layout constraint so when the fallback
                # is used again, we can constrain the args to the same layout
                layout_constraint = constrain_to_fx_strides
            return layout_constraint, args, kwargs

        if target not in lowerings:
            assert isinstance(
                target, torch._ops.OpOverload
            ), f"{target} is not an OpOverload"
            base_name = target.name().split(".")[0]
            if base_name in FALLBACK_ALLOW_LIST:
                make_fallback(target)
            elif config.implicit_fallbacks:
                layout_constraint, args, kwargs = get_custom_op_layout_constraints(
                    target, args, kwargs
                )
                error = (
                    MissingOperatorWithDecomp
                    if get_decompositions([target])
                    else MissingOperatorWithoutDecomp
                )
                log.info(
                    "Creating implicit fallback for:\n%s",
                    error.operator_str(target, args, kwargs),
                )
                make_fallback(target, layout_constraint)

            elif get_decompositions([target]):
                # There isn't a good way to dynamically patch this in
                # since AOT Autograd already ran.  The error message tells
                # the user how to fix it.
                raise MissingOperatorWithDecomp(target, args, kwargs)
            else:
                raise MissingOperatorWithoutDecomp(target, args, kwargs)

        try:
            log.debug("  via %s", lowerings[target])
            out = lowerings[target](*args, **kwargs)
            return out
        except Exception as e:
            raise LoweringException(e, target, args, kwargs).with_traceback(
                e.__traceback__
            ) from None

    @staticmethod
    def can_inline_constant(t: torch.Tensor) -> bool:
        """
        True if this is a small constant attr that will be inlined.
        """
        return len(t.shape) == 1 and t.shape[0] <= 8

    def get_attr(self, target, args, kwargs):
        # this is a constant
        value = getattr_recursive(self.module, target)

        if isinstance(value, torch.fx.GraphModule):
            return ir.Subgraph(name=target, graph_module=value)

        if isinstance(value, torch._C.ScriptObject):
            self.torchbind_constants[target] = value
            self.constant_reprs[target] = ""
            return TorchBindObject(target, value)

        if (
            config.aot_inductor.use_runtime_constant_folding
            or config.always_keep_tensor_constants
            or unsupported_output_tensor(value)
        ):
            return self.add_tensor_constant(value, target)

        with no_dispatch():
            if value.shape == ():
                return Constant(value.item(), value.dtype, value.device)
            if self.can_inline_constant(value):
                # tensor lowering has constant inlining logic
                from .lowering import tensor

                return tensor(value.tolist(), dtype=value.dtype, device=value.device)

        return self.add_tensor_constant(value, target)

    def call_module(self, target, args, kwargs):
        raise AssertionError

    def call_method(self, target, args, kwargs):
        raise AssertionError

    def output(self, target, args, kwargs):
        result = super().output(target, args, kwargs)
        if not isinstance(result, (tuple, list)):
            # nested subgraphs can have singleton outputs
            result = (result,)
        assert isinstance(result, (tuple, list)), type(result)
        assert all(
            isinstance(
                x,
                (
                    TensorBox,
                    ir.Constant,
                    type(None),
                    ir.ConstantBuffer,
                    sympy.Expr,
                    sympy.logic.boolalg.Boolean,
                    int,
                    ir.EffectfulKernel,
                ),
            )
            for x in result
        ), result

        fx_node_args = V.graph.current_node.args[0]  # type: ignore[arg-type]
        if not isinstance(fx_node_args, (tuple, list)):
            # nested subgraphs can have singleton outputs
            fx_node_args = (fx_node_args,)
        result = [ir.ExternKernel.realize_input(x) for x in result]
        result_correct_strides = []

        assert len(fx_node_args) == len(result)
        for r, fx_node in zip(result, fx_node_args):
            if not isinstance(r, (ir.TensorBox, ir.BaseView)):
                result_correct_strides.append(r)
            else:
                # AOT Autograd tries to detect stride divergence of inductor from output metadata.
                # Here, we try to avoid spurious divergence by matching insignificant strides such as
                result_correct_strides.append(
                    self.try_match_insignificant_strides(
                        r, fx_node.meta["val"].stride()
                    )
                )

        self.graph_outputs = result_correct_strides
        value: ir.IRNode
        for name, value in self.graph_inputs.items():
            assert isinstance(
                value, (TensorBox, sympy.Expr)
            ), f"Unsupported inductor graph input type: {type(value)}"
            if not isinstance(value, TensorBox):
                continue
            value.realize()
            assert isinstance(value, TensorBox)
            value = value.data
            assert isinstance(value, ir.StorageBox)
            value_storage_box = value
            value = value.data
            if not isinstance(value, InputBuffer) or value.get_name() != name:
                # one of our inputs was mutated, need to turn that into a copy
                ir.MutationLayoutSHOULDREMOVE.realize_into(
                    value, self.graph_inputs_original[name]
                )
                # replace output with mutated input
                try:
                    ind = self.graph_outputs.index(value_storage_box)
                    self.graph_outputs[ind] = self.graph_inputs_original[name]
                except ValueError:
                    pass

        self.finalize()
        log.debug(
            "Force channels last inputs for %d conv for the current graph with id %d",
            self.num_channels_last_conv,
            self.graph_id if self.graph_id is not None else -1,
        )

    def finalize(self):
        for buf in self.buffers:
            buf.decide_layout()

    @contextmanager
    def set_current_node(self, node: torch.fx.Node):
        old = self.current_node
        try:
            self.current_node = node
            yield
        finally:
            self.current_node = old

    def try_match_insignificant_strides(
        self,
        tensor,
        meta_strides_inp: Tuple[Union[int, torch.SymInt], ...],
    ) -> ir.TensorBox:
        """
        Tries to match the strides of the tensor to those in the meta_strides. Strides of insignificant
        dimensions - size 0 or 1 - will be updated.

        If there are real stride differences (NHWC vs NCHW) then the input will be returned.
        """

        # should have already been realized
        assert torch._inductor.ir.is_storage_and_layout(tensor)

        meta_strides = [
            s.node.expr if isinstance(s, torch.SymInt) else s for s in meta_strides_inp
        ]

        if all(
            self.sizevars.statically_known_equals(s1, s2)
            for s1, s2 in zip(meta_strides, tensor.get_stride())
        ):
            return tensor

        def significant_strides_equal(shape, meta_strides, tensor_strides):
            for dim, s1, s2 in zip(shape, meta_strides, tensor_strides):
                if self.sizevars.statically_known_leq(dim, 1):  # type: ignore[arg-type]
                    continue

                if not self.sizevars.statically_known_equals(s1, s2):
                    return False

            return True

        if not significant_strides_equal(
            tensor.get_size(), meta_strides, tensor.get_stride()
        ):
            return tensor

        storage, old_layout = torch._inductor.ir.as_storage_and_layout(tensor)
        new_stride = list(old_layout.stride)
        for i, s in enumerate(tensor.get_size()):
            if self.sizevars.statically_known_leq(s, 1):  # type: ignore[arg-type]
                new_stride[i] = meta_strides[i]

        new_layout = torch._inductor.ir.FixedLayout(
            old_layout.device,
            old_layout.dtype,
            old_layout.size,
            new_stride,
            old_layout.offset,
        )
        return ir.TensorBox(torch._inductor.ir.ReinterpretView(storage, new_layout))

    def run_node(self, n: torch.fx.Node):
        def debug(msg):
            log.debug("lowering %s %s", LazyString(n.format_node), msg)

        buffer_watermark = len(self.buffers)
        operation_watermark = len(self.operations)

        origins = {n}
        if n.op == "call_function":
            args, kwargs = self.fetch_args_kwargs_from_env(n)
            origins |= gather_origins(args, kwargs)
        with ir.IRNode.current_origins(origins), self.set_current_node(
            n
        ), V.set_current_node(n):
            if (
                n.op == "call_function"
                and n.target is not operator.getitem
                and fallback_node_due_to_unsupported_type(n)
            ):
                debug("fallback_handler")
                result = fallback_handler(n.target, add_to_fallback_set=False)(
                    *args, **kwargs  # type: ignore[possibly-undefined]
                )
            elif n.op == "call_function" and n.target in layout_constraints:
                debug("layout_constraints")
                args, kwargs = layout_constraints[n.target](n, *args, **kwargs)  # type: ignore[index]
                result = self.call_function(n.target, args, kwargs)
            elif is_magic_method(n.target):
                # TODO: this is sus, it probably should be handled in the
                # lowerings themselves similarly to sym_size/sym-stride
                # https://github.com/pytorch/pytorch/issues/127789
                debug("is_magic_method")
                if isinstance(
                    n.meta["val"], (torch.SymInt, torch.SymFloat, torch.SymBool)
                ):
                    result = n.meta["val"].node.expr
                else:
                    result = super().run_node(n)
            else:
                debug("")
                result = super().run_node(n)

            # require the same stride order for dense outputs,
            # 1. user-land view() will not throw because inductor
            # output different strides than eager
            # long term the solution is to make view() always succeed
            # with infallible strides.
            # 2: as_strided ops, we need make sure its input has same size/stride with
            # eager model to align with eager behavior.
            as_strided_ops = [
                torch.ops.aten.as_strided.default,
                torch.ops.aten.as_strided_.default,
                torch.ops.aten.as_strided_scatter.default,
                torch.ops.aten.resize.default,
                torch.ops.aten.resize_as.default,
            ]
            is_output = any(user.op == "output" for user in n.users)
            is_input_for_as_strided = any(
                user.target in as_strided_ops for user in n.users
            )

            if n.meta.get("inductor_realize_to_strides", False) and isinstance(
                result, TensorBox
            ):
                result.realize()
                strides = n.meta["val"].stride()
                sym_strides = torch._inductor.utils.any_is_symbolic(*strides)
                if (
                    not hasattr(result, "get_stride")
                    or result.get_stride() != strides
                    and not sym_strides
                ):
                    stride_order = ir.get_stride_order(strides)
                    result = ir.ExternKernel.require_stride_order(result, stride_order)
            if (
                is_output
                and isinstance(result, TensorBox)
                and isinstance(result.data, ir.BaseView)
            ):
                # Realize so that outputs are correctly aliased
                result.realize()

            if (is_output or is_input_for_as_strided) and isinstance(
                n.meta["val"], torch.Tensor
            ):
                strides = n.meta["val"].stride()
                dense = torch._prims_common.is_non_overlapping_and_dense(n.meta["val"])
                unbacked_symbols_in_strides = len(free_unbacked_symbols(strides)) > 0
                # requiring a stride order for a non-dense output wouldn't
                # recreate the same strides, and would fail with view, defer for now.
                if not unbacked_symbols_in_strides and dense and len(strides):
                    stride_order = ir.get_stride_order(strides)
                    if (
                        len(result.get_size()) == 4
                        and n in self.nodes_prefer_channels_last
                        and n.name not in self.user_visible_outputs
                        and not is_input_for_as_strided
                    ):
                        stride_order = ir.NHWC_STRIDE_ORDER

                    allow_padding = (
                        n.name not in self.user_visible_outputs
                        and not is_input_for_as_strided
                    )
                    result = ir.ExternKernel.require_stride_order(
                        result, stride_order, allow_padding=allow_padding
                    )

            # Realize if (1) any user need inputs realized, or (2) there is
            # already too many reads and rematerializing can be bad.
            num_users = len(set(n.users))
            if num_users > 1 and isinstance(result, TensorBox):
                for user in n.users:
                    if user.target in needs_realized_inputs:
                        result.realize_hint()
                        # This inclusion is somewhat controversial (from
                        # discussion between Horace, Natalia, and Elias).
                        # Currently, it's not very clear why this is helpful.
                        # The general idea here is that even though a node may
                        # have FlexibleLayout, we still often *treat* it as if
                        # it was contiguous. This appears to sometimes result in
                        # suboptimal behavior.
                        #
                        # When we do a better job selecting layout, we should
                        # revisit this.
                        need_fixed_layout = [
                            torch.ops.aten.convolution_backward.default,
                            torch.ops.aten.mm.default,
                            torch.ops.aten._int_mm.default,
                        ]
                        need_fixed_channels_last_layout = []
                        if not self.layout_opt:
                            need_fixed_layout.append(torch.ops.aten.convolution.default)
                        if torch._C._has_mkldnn:
                            need_fixed_layout += [
                                torch.ops.mkldnn._linear_pointwise.default,
                                torch.ops.mkldnn._linear_pointwise.binary,
                                torch.ops.aten.mkldnn_rnn_layer.default,
                                torch.ops.onednn.qlinear_pointwise.default,
                                torch.ops.onednn.qlinear_pointwise.tensor,
                                torch.ops.onednn.qlinear_pointwise.binary,
                                torch.ops.onednn.qlinear_pointwise.binary_tensor,
                            ]
                            need_fixed_channels_last_layout += [
                                torch.ops.mkldnn._convolution_pointwise.default,
                                torch.ops.mkldnn._convolution_pointwise.binary,
                                torch.ops.mkldnn._convolution_pointwise_.binary,
                                torch.ops.mkldnn._convolution_transpose_pointwise.default,
                                torch.ops.onednn.qconv2d_pointwise.default,
                                torch.ops.onednn.qconv2d_pointwise.binary,
                            ]
                            if torch._C.has_mkl:
                                need_fixed_layout += [torch.ops.mkl._mkl_linear.default]
                        if user.target in need_fixed_layout:
                            result = ir.ExternKernel.require_stride_order(
                                result,
                                ir.get_stride_order(n.meta["val"].stride()),
                                allow_padding=True,
                            )
                        if (
                            user.target in need_fixed_channels_last_layout
                            and n is user.args[0]
                        ):
                            result = ir.ExternKernel.require_stride_order(
                                result,
                                ir.get_stride_order(
                                    make_channels_last_strides_for(n.meta["val"].shape)
                                ),
                            )
                    if user.op == "output":
                        if isinstance(result.data.data, (Pointwise, Reduction)):
                            result.realize()

                # TODO(jansel): introduce a store vs inline choice
                result.mark_reuse(len(n.users))

            # Realize if the IRNode already has accumulated lots of reads
            if isinstance(result, TensorBox) and result.has_exceeded_max_reads():
                # Prevent excessive accumulation in a computed buffer, when
                # there are multiple branches each with small number of memory
                # reads, but they converge to a user.
                result.realize_hint()

            # Realize if a Pointwise has too much stuff to be inlined.
            # As this may cause RecursionError during Inductor's evaluation.
            if isinstance(result, TensorBox) and isinstance(result.data, StorageBox):
                curr = result.data.data
                if isinstance(curr, Pointwise):
                    # Use inner fn as a rough proxy. Good enough.
                    if curr.has_large_inner_fn():
                        result.realize()

        # This is not complete, but it doesn't have to be: origin_node
        # tracking is best effort.  The logic here critically relies on direct
        # TensorBox -> StorageBox denoting a non-view; we don't bother trying
        # to get views to work.  Feel free to add any extra cases as needed.
        #
        # Note: we can't YOLO tree_map over this result, because if there are
        # buffers or a view involved, we might not be able to validly assign
        # the origin_node here.
        if isinstance(result, TensorBox) and isinstance(result.data, ir.StorageBox):
            if isinstance(result.data.data, ir.Loops):
                result.data.data.origin_node = n
            elif isinstance(result.data.data, ir.Buffer):
                result.data.data.origin_node = n
                if isinstance(result.data.data, ir.ComputedBuffer) and isinstance(
                    result.data.data.data, ir.Loops
                ):
                    result.data.data.data.origin_node = n
                # Not really multi-output, can straightforwardly recurse in
                elif (
                    isinstance(result.data.data, ir.MultiOutput)
                    and not result.data.data.indices
                ):
                    if isinstance(result.data.data.inputs[0], ir.Buffer):
                        result.data.data.inputs[0].origin_node = n

        self.register_users_of(result)

        new_unbacked_defs = set()
        for buf in self.buffers[buffer_watermark:]:
            new_unbacked_defs |= buf.get_unbacked_symbol_defs()
        for op in self.operations[operation_watermark:]:
            new_unbacked_defs |= op.get_unbacked_symbol_defs()

        def format_new_defs():
            r = []
            for buf in self.buffers[buffer_watermark:]:
                r.append(
                    f"unbacked_symbol_defs={buf.get_unbacked_symbol_defs()} in:\n{buf}\n"
                )
            for op in self.operations[operation_watermark:]:
                r.append(
                    f"unbacked_symbol_defs={op.get_unbacked_symbol_defs()} in:\n{op}\n"
                )
            return "***\n".join(r)

        if n.op != "placeholder":
            # Note [Backwards runtime asserts]
            # Backwards poses an interesting problem for deferred runtime
            # asserts.  In the easy case, we may solely close over data
            # dependent sized tensors, and there are no binding sites for
            # unbacked SymInts.  In this case, we can just drop all the
            # runtime asserts on the floor: no non-placeholder bindings, no
            # problem.
            #
            # However, it is *possible* for a fresh runtime assert to show up
            # between forwards and backwards.  Right now, the freezing process
            # that happens when we lower forwards means that we will freeze
            # runtime asserts, and then the moment the backwards lowering
            # process attempts to add a new deferred runtime assert, we will
            # fail.  Let's say you remove that assert.  Now when we get here,
            # we need to make sure we actually emit these asserts (because we
            # can't emit them in forwards, we already compiled it).  So we
            # have to do something here.  But we don't want to reemit ALL
            # deferred runtime asserts, we only want to emit the NEW ones.
            # Therefore needing some sort of stratification in the ShapeEnv.
            # This is all doable, it just hasn't been done yet.
            shape_env = V.graph.sizevars.shape_env

            def make_assert(expr, msg):
                assert_op = ir.AssertScalar(expr, msg)
                self.register_buffer(assert_op, set_name=True)
                self.register_operation(assert_op)

            for i0 in new_unbacked_defs:
                ras = self.ras_by_symbol.pop(i0, [])
                # NB: size-like not needed, we won't retrace
                vr = shape_env.var_to_range[i0]
                if not shape_env._default_unspecified_value_range().issubset(vr):

                    def is_convertible(s):
                        if s in (int_oo, -int_oo):
                            return False
                        try:
                            int(s)
                            return True
                        except TypeError:
                            return False

                    if is_convertible(vr.lower):
                        make_assert(i0 >= vr.lower, f"{i0} >= {vr.lower}")
                    if is_convertible(vr.upper):
                        make_assert(i0 <= vr.upper, f"{i0} <= {vr.upper}")

                for ra in ras:
                    fvs = free_unbacked_symbols(ra.expr)
                    missing = fvs - self.bound_unbacked_symbols
                    if missing:
                        i1 = sorted(missing, key=lambda x: str(x))[0]
                        self.ras_by_symbol.setdefault(i1, []).append(ra)
                    else:
                        make_assert(ra.expr, f"{ra.expr}")

            self.bound_unbacked_symbols |= new_unbacked_defs

            unbacked_bindings = resolve_unbacked_bindings(
                V.graph.sizevars.shape_env, n.meta.get("unbacked_bindings", {})
            )
            # When we do lowering, it is possible we reallocate unbacked SymInts.
            # So we need to line up the unbacked SymInts when performing the test
            # here
            #
            # In principle, we could permit lowering to introduce MORE unbacked
            # SymInts: as long as all the old unbacked ones are accounted for,
            # it's fine for inductor to introduce extra calls to item()/unbacked()
            # whatever.  This actually happens in practice when an unbacked SymInt
            # gets memoized away; naively, when Inductor reprocesses a kernel, it
            # doesn't know that the memo still applies, and ends up allocating a
            # new symbol.  However, this is generally a bad thing: we may still
            # end up needing to test equalities on the symbols, and a fresh
            # symbol is likely to hit lots of GuardOnDataDependent errors that
            # we already know facts for.
            renamed_unbacked_bindings = {
                V.fake_mode.shape_env.unbacked_renamings.get(s, s)
                for s in unbacked_bindings.keys()
            }
            assert new_unbacked_defs >= renamed_unbacked_bindings, (
                f"failed {new_unbacked_defs} >= {renamed_unbacked_bindings} (inductor >= fx)\n"
                f"fx node is: {n.format_node()}\n"
                f"new operations are:\n\n{format_new_defs()}"
            )

        return result

    def validate_can_generate_cpp_wrapper(self):
        if config.disable_cpp_codegen:
            raise CppWrapperCodeGenError("C++ codegen is disabled")

        if sys.platform not in ["linux", "darwin"]:
            raise CppWrapperCodeGenError(f"Unsupported platform {sys.platform}")

        for value in self.graph_inputs.values():
            dtype = None
            if isinstance(value, TensorBox):
                dtype = value.get_dtype()
            elif isinstance(
                value, (sympy.Symbol, sympy.Expr, sympy.core.numbers.Integer)
            ):
                dtype = may_get_constant_buffer_dtype(value)

            if not supported_dtype_of_cpp_wrapper(dtype, self.cuda):
                raise CppWrapperCodeGenError(f"Unsupported input dtype {dtype}")

    def init_wrapper_code(self):
        self.cuda = "cuda" in self.device_types
        if self.cpp_wrapper:
            self.validate_can_generate_cpp_wrapper()

        device_types = self.device_types.copy()
        device_types.discard("cpu")
        device_types.discard("meta")
        # TODO(Eikan): Only support mixing cpu and other device now.
        assert len(device_types) <= 1, "Does not support mixing {}".format(
            "+".join(device_types)
        )
        only_cpu = len(device_types) == 0
        device_type = "cpu" if only_cpu else device_types.pop()

        self.device_ops = get_device_op_overrides(device_type)
        wrapper_code_gen_cls = get_wrapper_codegen_for_device(
            device_type, self.cpp_wrapper
        )
        assert wrapper_code_gen_cls is not None, f"Device {device_type} not supported"
        self.wrapper_code = wrapper_code_gen_cls()

        if self.const_module:
            # If we have const module, we could reuse the kernels
            # This could avoid duplication and save time on doing recompilation (if Triton.)
            self.wrapper_code._names_iter = self.const_module.wrapper_code._names_iter
            self.wrapper_code.src_to_kernel = (
                self.const_module.wrapper_code.src_to_kernel
            )

    def codegen_with_cpp_wrapper(self):
        """
        For CPU, the cpp wrapper codegen is done in one pass.
        For GPU, the cpp wrapper codegen is done in two steps: JIT-compile the model with python
        wrapper code and run it to generate autotuned kernel binaries in the first pass; and then
        generate cpp wrapper code and compile it to a dynamic library in the second pass.
        """
        if "cuda" in self.device_types:
            # first pass
            self.cpp_wrapper = False
            # Although triton.store_cubin was set in compile_fx, the backward pass didn't pick
            # that up. In theory it should work by only setting triton.store_cubin to True here,
            # but that will cause a problem when use_runtime_constant_folding is set.
            with config.patch({"triton.store_cubin": True}):
                compiled = self.compile_to_module().call

            if not config.triton.autotune_at_compile_time:

                def materialize(x):
                    if isinstance(x, (torch.SymInt, torch.SymFloat)):
                        # Need concrete value to run dynamic shapes and tune the result
                        return x.node.hint
                    elif isinstance(x, FakeTensor):
                        return defake(x)
                    else:
                        assert isinstance(
                            x, torch.Tensor
                        ), "Unknown type when creating real inputs" + str(type(x))
                        return x

                tracing_context = torch._guards.TracingContext.try_get()
                if tracing_context is not None and not isinstance(
                    V.real_inputs, NullHandler
                ):
                    if tracing_context.output_strides:
                        tracing_context.output_strides.clear()

                    params_flat = [
                        param
                        for param in tracing_context.params_flat  # type: ignore[union-attr]
                        if param is not None
                    ]
                    real_inputs = [
                        materialize(x)
                        for x in itertools.chain(params_flat, V.real_inputs)
                    ]
                else:
                    # In the backward pass, V.real_inputs is not set.
                    # Generating random inputs based on self.example_inputs sometimes can be problematic,
                    # e.g. illegal memory access. A comprehensive fix is to autotune in a separate process.
                    real_inputs = [
                        materialize(x)
                        for x in (
                            self.example_inputs
                            if isinstance(V.real_inputs, NullHandler)
                            else V.real_inputs
                        )
                    ]

                if self.mutated_inputs:
                    from .compile_fx import clone_preserve_strides

                    mutated_input_idxs = [
                        idx
                        for idx, name in enumerate(self.graph_inputs)
                        if name in self.mutated_inputs
                        and isinstance(real_inputs[idx], torch.Tensor)
                    ]
                    for idx in mutated_input_idxs:
                        # clone mutated Tensor inputs to avoid mutating them in
                        # the first pass of the CPP wrapper-based compilation, as
                        # this will lead to a side effect on the example inputs:
                        # e.g. if torch.compile(f)(x) if called on input-mutating
                        # f, the inputs x will be mutated twice in the process:
                        # once here, and again when running the compiled model;
                        # this will also lead to a numerically incorrect output
                        real_inputs[idx] = clone_preserve_strides(real_inputs[idx])

                with torch.utils._python_dispatch._disable_current_modes():
                    compiled(real_inputs)
                del real_inputs

            # second pass
            self.cpp_wrapper = True
            self.removed_buffers.clear()
            self.inplaced_to_remove.clear()
            V.graph.sizevars.precomputed_replacements.clear()
            V.graph.sizevars.inv_precomputed_replacements.clear()
            with config.patch({"triton.autotune_at_compile_time": False}):
                return self.codegen()
        else:
            # cpu
            return self.codegen()

    def codegen(self):
        from .scheduler import Scheduler

        self.init_wrapper_code()

        self.scheduler = Scheduler(self.operations)
        V.debug.draw_orig_fx_graph(self.orig_gm, self.scheduler.nodes)

        self.wrapper_code.push_codegened_graph(self)
        self.scheduler.codegen()
        result = self.wrapper_code.generate(self.is_inference)
        self.wrapper_code.pop_codegened_graph()
        return result

    def codegen_subgraph(self, parent_graph):
        """
        This is a more compact version of the `codegen()` above
        where we codegen this graph as a subgraph of some parent
        graph. The parent graph is passed as an argument: the
        intention is to inline codegening of the subgraph in
        the parent graph's wrapper code (including the generated
        kerenls). The wrapper code is not finalized (via `.generate()`
        call), as this will be done in the parent graph's `codegen()`.
        """
        from .scheduler import Scheduler

        self.wrapper_code = parent_graph.wrapper_code
        self.device_ops = parent_graph.device_ops
        self.cpp_wrapper = parent_graph.cpp_wrapper

        self.scheduler = Scheduler(self.operations)
        self.scheduler.codegen()

    def count_bytes(self):
        total_bytes = 0
        node_counts = []
        node_runtimes = []
        for node in self.scheduler.nodes:
            num_bytes = node.get_read_write_buffers_sizes()
            total_bytes += num_bytes
            node_counts.append((node, num_bytes // 4))
            node_runtimes.append((node, node.get_estimated_runtime()))
        return total_bytes, node_counts, node_runtimes

    @staticmethod
    def save_output_code(code: str):
        # No-op to be patched for unit tests
        pass

    @dynamo_timed(phase_name="code_gen", fwd_only=False)
    def compile_to_module(self):
        from .codecache import PyCodeCache

        code, linemap = (
            self.codegen_with_cpp_wrapper() if self.cpp_wrapper else self.codegen()
        )

        GraphLowering.save_output_code(code)
        output_code_log.debug("Output code: \n%s", code)
        try:
            linemap = [(line_no, node.stack_trace) for line_no, node in linemap]
            key, path = PyCodeCache.write(code)
        except Exception:
            trace_structured(
                "inductor_output_code",
                # Just omit the filename, I still want the code though!
                payload_fn=lambda: code,
            )
            raise
        else:
            trace_structured(
                "inductor_output_code",
                lambda: {"filename": path},
                payload_fn=lambda: code,
            )

        mod = PyCodeCache.load_by_key_path(
            key,
            path,
            linemap=linemap,
            attrs={**self.constants, **self.torchbind_constants},
        )
        self.cache_key = key
        self.cache_path = path
        self.cache_linemap = linemap

        # Logged twice as per https://github.com/pytorch/pytorch/pull/99038#discussion_r1167826029
        # TODO. Revisit this once the logging API is more mature
        assert mod.__file__ is not None

        log_module_code(mod.__file__)
        log.debug("Output code written to: %s", mod.__file__)
        output_code_log.info("Output code written to: %s", mod.__file__)
        if config.benchmark_kernel:
            print(f"Compiled module path: {mod.__file__}", file=sys.stderr)
        V.debug.output_code(mod.__file__)
        V.debug.copy(os.path.splitext(mod.__file__)[0] + ".debug")
        return mod

    def compile_to_fn(self):
        if self.aot_mode:
            from .codecache import AotCodeCompiler

            assert self.cpp_wrapper, "AOT mode only supports C++ wrapper"
            code, linemap = self.codegen_with_cpp_wrapper()
            output_code_log.debug("Output code: \n%s", code)

            serialized_extern_kernel_nodes = None
            if (
                config.is_fbcode()
                and self.extern_kernel_nodes
                and self.extern_node_serializer
            ):
                serialized_extern_kernel_nodes = self.extern_node_serializer(
                    self.extern_kernel_nodes
                )
                output_code_log.debug(
                    "Serialized Extern Kernel Nodes: \n%s",
                    serialized_extern_kernel_nodes,
                )

            # Directly return the file path with the compiled code
            return AotCodeCompiler.compile(
                self, code, serialized_extern_kernel_nodes, cuda=self.cuda
            )
        else:
            return self.compile_to_module().call

    def get_output_names(self):
        return [
            node.get_name()
            for node in self.graph_outputs
            if not isinstance(node, ir.NoneAsConstantBuffer)
            and not isinstance(node, ir.ShapeAsConstantBuffer)
        ]

    def is_unspec_arg(self, name: str):
        # dynamo wraps unspec variable as 0d CPU tensor,
        # need to convert to scalar during codegen (triton only)
        return (
            name in self.graph_inputs.keys()
            and self.graph_inputs[name].get_numel() == 1
            and self.graph_inputs[name].get_device().type == "cpu"
        )
