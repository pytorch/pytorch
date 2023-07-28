import hashlib
import logging
import operator
import os
import re
import sys
import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Set, Tuple

import sympy

import torch
import torch._logging
import torch.fx
from torch._decomp import get_decompositions
from torch._dynamo.utils import dynamo_timed
from torch.fx.experimental.symbolic_shapes import (
    free_symbols,
    magic_methods,
    method_to_operator,
    ShapeEnv,
    SymTypes,
)
from torch.utils._mode_utils import no_dispatch

from . import config, ir, metrics
from .codegen.wrapper import CppWrapperCodeGen, CudaWrapperCodeGen, WrapperCodeGen
from .exc import (
    LoweringException,
    MissingOperatorWithDecomp,
    MissingOperatorWithoutDecomp,
)
from .ir import Constant, FixedLayout, InputBuffer, Pointwise, Reduction, TensorBox
from .lowering import (
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
    get_dtype_size,
    get_sympy_Expr_dtype,
    sympy_product,
)
from .virtualized import V

log = logging.getLogger(__name__)
perf_hint_log = torch._logging.getArtifactLogger(__name__, "perf_hints")
output_code_log = torch._logging.getArtifactLogger(__name__, "output_code")


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
        torch.complex64,
        # torch.float16, # TODO: implement this
    }
    if cuda:
        supported_dtype.add(torch.float16)

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


class GraphLowering(torch.fx.Interpreter):
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
        shape_env=None,
        num_static_inputs=None,
        graph_id=None,
        cpp_wrapper=False,
        aot_mode=False,
        user_visible_outputs=frozenset(),
        layout_opt=None,
    ):
        super().__init__(gm)

        self.layout_opt = (
            layout_opt if layout_opt is not None else self.decide_layout_opt(gm)
        )
        self.num_channels_last_conv = 0

        self.extra_traceback = False  # we do our own error wrapping
        if shape_env is None:
            shape_env = ShapeEnv()
            self.reuse_shape_env = False
        else:
            self._shape_env = shape_env
            self.reuse_shape_env = True
        self._shape_env = shape_env
        self.sizevars = SizeVarAllocator(shape_env)
        self.graph_inputs: Dict[str, TensorBox] = {}
        self.graph_inputs_original: Dict[str, InputBuffer] = {}
        self.graph_outputs: Optional[List[ir.IRNode]] = None
        self.device_types: Set[str] = set()
        self.device_idxs: Set[int] = set()
        self.cuda = False
        self.buffers: List[ir.ComputedBuffer] = []
        self.constants: Dict[str, torch.Tensor] = {}
        self.constant_reprs: Dict[str, str] = {}
        self.removed_buffers: Set[str] = set()
        self.removed_inplace_buffers: Set[str] = set()
        self.mutated_buffers: Set[str] = set()
        self.inplaced_to_remove: Set[str] = set()
        self.wrapper_code: Optional[WrapperCodeGen] = None
        self.current_node: Optional[torch.fx.Node] = None
        self.num_static_inputs = num_static_inputs
        self.lists: Dict[str, List[str]] = {}
        self.mutated_inputs: Set[str] = set()
        self.mutated_input_idxs: List[int] = []
        self.unaligned_buffers: Set[str] = set()
        self.name_to_buffer: Dict[str, ir.ComputedBuffer] = {}
        self.creation_time = time.time()
        self.name = "GraphLowering"
        self.cpp_wrapper = cpp_wrapper
        self.aot_mode = aot_mode
        self.graph_id = graph_id
        self.scheduler = None
        self.nodes_prefer_channels_last = (
            self.find_nodes_prefer_channels_last() if self.layout_opt else set()
        )
        self._warned_fallback = {"aten.convolution_backward"}
        self.user_visible_outputs = user_visible_outputs
        self.cache_key: str = ""  # This is the cache key for the compiled artifact
        self.cache_path: str = ""  # This is the path in the filesystem where the compiled artifact is stored
        self.cache_linemap: List[
            Tuple[int, str]
        ] = (
            []
        )  # This is the linemap used by the profiler to mark custom compiled kernels getting run
        # Used if lowering encounters cases where cudagraphs are not supported
        self.disable_cudagraphs = False

    @staticmethod
    def decide_layout_opt(gm) -> bool:
        """
        Decide if we should enable layout optimization for this graph based on
        heuristics.
        """
        if not config.layout_optimization:
            return False

        conv_nodes = [
            n for n in gm.graph.nodes if n.target == torch.ops.aten.convolution.default
        ]
        nconv = len(conv_nodes)

        if nconv == 0:
            return False

        # For cpu backend and mkldnn enabled, we always using channels_last for a better performance.
        if (
            all(
                n.args[idx].meta["val"].device == torch.device("cpu")
                for n in conv_nodes
                for idx in [0, 1]
            )
            and torch.backends.mkldnn.enabled
            and torch.backends.mkldnn.is_available()
        ):
            return True

        # Followering models are skipped due to this:
        # jx_nest_base
        # volo_d1_224
        if len(list(gm.graph.nodes)) >= 300 * nconv:
            log.debug("Only a few conv, skip layout optimization")
            return False

        if any(
            free_symbols(n.args[idx].meta["val"]) for n in conv_nodes for idx in [0, 1]
        ):
            log.debug(
                "See perf regression with dynamic shape. Follow up in https://github.com/pytorch/pytorch/issues/102670"
            )
            return False

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
        if any(
            n.args[-1] > 1 and n.args[1].meta["val"].size(1) > 1 for n in conv_nodes
        ):
            log.debug("Found grouped convolution with >1 in_channels!")
            return False

        # For some models that contain convolution with larger in-channel than out-channel, applying
        # channels last hurts performance.
        # Following models are skipped due to this:
        # - pytorch_unet
        # - phlippe_densenet (slightly worse)
        # - Background_Matting (1.22x -> 0.821x)
        # - pytorch_CycleGAN_and_pix2pix (1.597x -> 1.294x)
        if any(
            n.args[1].meta["val"].size(0) * 2 <= n.args[1].meta["val"].size(1)
            and n.args[1].meta["val"].size(2) > 1
            for n in conv_nodes
        ):
            log.debug(
                "Skip layout optimization because some convolutions have smaller out_channel"
            )
            return False

        # Following models are skipped due to this:
        # - functorch_maml_omniglot
        if all(
            n.args[1].meta["val"].size(0) <= 64 and n.args[1].meta["val"].size(1) <= 64
            for n in conv_nodes
        ):
            log.debug("Skip layout opt because all convolution channels are too small")
            return False

        # aten._scaled_dot_product_flash_attention requires the last stride of query/key/value
        # to be 1. Check https://gist.github.com/shunting314/fa6eeab2aad8d1265c4d5e50b560d94f
        # for more details.
        #
        # When a model contains aten._scaled_dot_product_flash_attention and we enable layout optimization,
        # the op may get channels last input and fail. Example include: twins_pcpvt_base, xcit_large_24_p8_224
        # for _scaled_dot_product_flash_attention and xcit_large_24_p8_224 for _scaled_dot_product_efficient_attention.
        #
        # We disable layout optimization if a model contains aten._scaled_dot_product_flash_attention.
        #
        # An alternative is to do necessary layout convertion to make sure aten._scaled_dot_product_flash_attention's
        # inputs have the layout needed. But that seems to have worse perf than disabing the layout opt.
        # TODO(shunting) revisit if we can still apply layout optimization to models containing sdpa while
        # bringing perf gains.
        for n in gm.graph.nodes:
            if n.target in (
                torch.ops.aten._scaled_dot_product_flash_attention.default,
                torch.ops.aten._scaled_dot_product_efficient_attention.default,
            ):
                log.debug(
                    "Skip layout optimization because sdpa (scaled dot product attention) is found"
                )
                return False

        return True

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
                for child in n.users:
                    output_set.add(child)

        return output_set

    def warn_fallback(self, name):
        if name not in self._warned_fallback:
            self._warned_fallback.add(name)
            perf_hint_log.info("Using FallbackKernel: %s", name)

    def add_device_idx(self, idx: Optional[int]):
        if idx is not None:
            self.device_idxs.add(idx)

    @property
    def fake_mode(self):
        return V.fake_mode

    def get_buffer(self, buffer_name: str):
        if buffer_name in self.name_to_buffer:
            return self.name_to_buffer[buffer_name]
        if buffer_name in self.graph_inputs:
            return self.graph_inputs[buffer_name]
        return None

    def get_dtype(self, buffer_name: str):
        if buffer_name in self.constants:
            return self.constants[buffer_name].dtype
        if buffer_name in self.name_to_buffer:
            return self.name_to_buffer[buffer_name].get_dtype()
        if buffer_name in self.graph_inputs:
            return self.graph_inputs[buffer_name].get_dtype()
        m = re.match(r"as_strided\(([a-zA-Z0-9_]+),", buffer_name)
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

    def disable_cpp_wrapper(self, cond):
        metrics.disable_cpp_wrapper += 1
        self.cpp_wrapper = False
        log.debug("Set cpp_wrapper to False due to %s", cond)

    def register_buffer(self, buffer: ir.ComputedBuffer):
        name = f"buf{len(self.buffers)}"
        self.buffers.append(buffer)
        self.name_to_buffer[name] = buffer
        return name

    def register_list(self, buffer_names: List[str]):
        name = "list_" + "_".join(buffer_names)
        self.lists[name] = buffer_names
        return name

    def mark_buffer_mutated(self, name: str):
        """
        When a buffer is mutated we need to make sure all the reads to
        the old version are realized before the mutation happens.
        """
        assert isinstance(name, str)
        self.mutated_buffers.add(name)

        def visit(value):
            if isinstance(value, (list, tuple)):
                return [visit(x) for x in value]
            if isinstance(value, ir.IRNode):
                if value.is_user_of(name):
                    value.realize()
            return value

        for value in self.env.values():
            try:
                visit(value)
            except Exception:
                log.warning("error in mark_buffer_mutated", exc_info=True)

    def add_tensor_constant(self, data):
        def allocate():
            for name, value in self.constants.items():
                if (
                    not data.is_mkldnn
                    and data.size() == value.size()
                    and data.stride() == value.stride()
                    and data.dtype == value.dtype
                    and data.device == value.device
                    and torch.eq(data, value).all()
                ):
                    return name
            name = f"constant{len(self.constants)}"
            self.constants[name] = data
            self.constant_reprs[name] = hashlib.sha256(
                repr(data).encode("utf-8")
            ).hexdigest()
            return name

        return TensorBox.create(
            ir.ConstantBuffer(
                allocate(),
                FixedLayout(data.device, data.dtype, *self.static_sizes_strides(data)),
            )
        )

    def constant_name(self, name: str, device_override: torch.device):
        """
        We AOT copy constants to the devices they are needed on.
        If device_override doesn't match the constant's device, then
        copy it and return a different name.
        """
        if self.constants[name].device == device_override or device_override is None:
            return name
        alt_name = f"{name}_{device_override.type}{device_override.index or 0}"
        if alt_name not in self.constants:
            self.constants[alt_name] = self.constants[name].to(device_override)
        return alt_name

    def placeholder(self, target: str, args, kwargs):
        example = super().placeholder(target, args, kwargs)
        if isinstance(example, SymTypes):
            expr = example.node.expr
            self.graph_inputs[target] = expr
            return expr
        elif isinstance(example, (int, bool, float)):
            expr = sympy.sympify(example)
            self.graph_inputs[target] = expr
            return expr
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
        tensor = TensorBox.create(
            InputBuffer(
                target,
                FixedLayout(example.device, example.dtype, sizes, strides),
            )
        )
        self.graph_inputs[target] = tensor
        self.graph_inputs_original[target] = tensor.data.data
        self.device_types.add(example.device.type)
        self.add_device_idx(example.device.index)
        return tensor

    def call_function(self, target, args, kwargs):
        if target is operator.getitem and isinstance(args[0], (list, tuple)):
            return super().call_function(target, args, kwargs)

        if hasattr(target, "_inductor_lowering_function"):
            # passthrough lowerings from .pattern_matcher
            return target(*args, **kwargs)

        if target not in lowerings:
            base_name = target.name().split(".")[0]
            if base_name in FALLBACK_ALLOW_LIST:
                make_fallback(target)
            elif config.implicit_fallbacks:
                error = (
                    MissingOperatorWithDecomp
                    if get_decompositions([target])
                    else MissingOperatorWithoutDecomp
                )
                log.info(
                    "Creating implicit fallback for:\n%s",
                    error.operator_str(target, args, kwargs),
                )
                make_fallback(target)
            elif get_decompositions([target]):
                # There isn't a good way to dynamically patch this in
                # since AOT Autograd already ran.  The error message tells
                # the user how to fix it.
                raise MissingOperatorWithDecomp(target, args, kwargs)
            else:
                raise MissingOperatorWithoutDecomp(target, args, kwargs)

        try:
            out = lowerings[target](*args, **kwargs)
            return out
        except Exception as e:
            raise LoweringException(e, target, args, kwargs).with_traceback(
                e.__traceback__
            ) from None

    def get_attr(self, target, args, kwargs):
        # this is a constant
        value = getattr(self.module, target)

        if unsupported_output_tensor(value):
            return self.add_tensor_constant(value)

        with no_dispatch():
            if value.shape == ():
                return Constant(value.item(), value.dtype, value.device)
            if len(value.shape) == 1 and value.shape[0] <= 8:
                # tensor lowering has constant inlining logic
                from .lowering import tensor

                return tensor(value.tolist(), dtype=value.dtype, device=value.device)

        return self.add_tensor_constant(value)

    def call_module(self, target, args, kwargs):
        raise AssertionError()

    def call_method(self, target, args, kwargs):
        raise AssertionError()

    def output(self, target, args, kwargs):
        result = super().output(target, args, kwargs)
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
                ),
            )
            for x in result
        ), result
        self.graph_outputs = [ir.ExternKernel.realize_input(x) for x in result]
        value: ir.IRNode
        for name, value in self.graph_inputs.items():
            assert isinstance(value, (TensorBox, sympy.Expr))
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
                ir.MutationLayout.realize_into(value, self.graph_inputs_original[name])
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
            self.graph_id,
        )

    def finalize(self):
        for buf in self.buffers:
            buf.decide_layout()

    def run_node(self, n: torch.fx.Node):
        origins = {n}
        if n.op == "call_function":
            args, kwargs = self.fetch_args_kwargs_from_env(n)
            origins |= gather_origins(args, kwargs)
        with ir.IRNode.current_origins(origins), self.set_current_node(n):
            if (
                n.op == "call_function"
                and n.target is not operator.getitem
                and fallback_node_due_to_unsupported_type(n)
            ):
                result = fallback_handler(n.target, add_to_fallback_set=False)(
                    *args, **kwargs
                )
            elif n.op == "call_function" and n.target in layout_constraints:
                args, kwargs = layout_constraints[n.target](n, *args, **kwargs)
                result = self.call_function(n.target, args, kwargs)
            elif n.target == torch.ops.aten.sym_stride:
                # inductor graphs can occasionally return sizes/strides,
                # e.g. if we need to save symints for the backward graph.
                if isinstance(n.meta["val"], torch.SymInt):
                    result = n.meta["val"].node.expr
                else:
                    result = super().run_node(n)
            elif is_magic_method(n.target):
                if isinstance(n.meta["val"], torch.SymInt):
                    result = n.meta["val"].node.expr
                else:
                    result = super().run_node(n)
            else:
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
            ]
            is_output = any(user.op == "output" for user in n.users)
            is_input_for_as_strided = any(
                user.target in as_strided_ops for user in n.users
            )
            if (is_output or is_input_for_as_strided) and isinstance(
                n.meta["val"], torch.Tensor
            ):
                strides = n.meta["val"].stride()
                dense = torch._prims_common.is_non_overlapping_and_dense(n.meta["val"])
                # requiring a stride order for a non-dense output wouldn't
                # recreate the same strides, and would fail with view, defer for now.
                if dense and len(strides):
                    stride_order = ir.get_stride_order(strides)
                    if (
                        len(result.get_size()) == 4
                        and n in self.nodes_prefer_channels_last
                        and n.name not in self.user_visible_outputs
                        and not is_input_for_as_strided
                    ):
                        stride_order = ir.NHWC_STRIDE_ORDER
                    result = ir.ExternKernel.require_stride_order(result, stride_order)

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
                        if not self.layout_opt:
                            need_fixed_layout.append(torch.ops.aten.convolution.default)
                        if torch._C._has_mkldnn:
                            need_fixed_layout += [
                                torch.ops.mkldnn._convolution_pointwise.default,
                                torch.ops.mkldnn._convolution_pointwise.binary,
                                torch.ops.mkldnn._convolution_pointwise_.binary,
                                torch.ops.mkldnn._convolution_transpose_pointwise.default,
                                torch.ops.mkldnn._linear_pointwise.default,
                                torch.ops.mkldnn._linear_pointwise.binary,
                                torch.ops.aten.mkldnn_rnn_layer.default,
                            ]
                            if torch._C.has_mkl:
                                need_fixed_layout += [torch.ops.mkl._mkl_linear.default]
                        if user.target in need_fixed_layout:
                            result = ir.ExternKernel.require_stride_order(
                                result, ir.get_stride_order(n.meta["val"].stride())
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

        return result

    def check_cpp_codegen_disabled(self):
        if config.disable_cpp_codegen:
            self.disable_cpp_wrapper("cpp codegen disabled")

    def check_platform(self):
        if sys.platform != "linux":
            self.disable_cpp_wrapper("platform not linux")

    def check_input_for_cpp_buffer(self):
        for value in self.graph_inputs.values():
            dtype = None
            if isinstance(value, TensorBox):
                dtype = value.get_dtype()
            elif isinstance(
                value, (sympy.Symbol, sympy.Expr, sympy.core.numbers.Integer)
            ):
                dtype = may_get_constant_buffer_dtype(value)

            if not supported_dtype_of_cpp_wrapper(dtype, self.cuda):
                self.disable_cpp_wrapper("unsupported inputs dtype")

    @contextmanager
    def set_current_node(self, node: torch.fx.Node):
        old = self.current_node
        try:
            self.current_node = node
            yield
        finally:
            self.current_node = old

    def check_cpp_wrapper(self):
        self.check_cpp_codegen_disabled()
        self.check_platform()
        self.check_input_for_cpp_buffer()

    def init_wrapper_code(self):
        self.cuda = "cuda" in self.device_types
        if self.cpp_wrapper:
            self.check_cpp_wrapper()
            # Re-check self.cpp_wrapper because it might be disabled due to failed checking
            if self.cuda:
                assert self.cpp_wrapper, "CudaWrapperCodeGen hit unsupported case"

            if self.cpp_wrapper:
                self.wrapper_code = (
                    CudaWrapperCodeGen() if self.cuda else CppWrapperCodeGen()
                )
                return

        self.wrapper_code = WrapperCodeGen()

    def codegen(self):
        from .scheduler import Scheduler

        self.init_wrapper_code()

        self.scheduler = Scheduler(self.buffers)
        assert self.scheduler is not None  # mypy can't figure this out
        self.scheduler.codegen()
        assert self.wrapper_code is not None
        return self.wrapper_code.generate()

    def count_bytes(self):
        from .scheduler import FusedSchedulerNode, NopKernelSchedulerNode, Scheduler

        scheduler = Scheduler(self.buffers)

        def get_read_write_buffers_sizes(node):
            if isinstance(node, NopKernelSchedulerNode):
                return 0
            reads = {dep.name for dep in node.read_writes.reads}
            writes = {dep.name for dep in node.read_writes.writes}

            def is_materialized(buf):
                buf_uses = {user.node for user in scheduler.name_to_node[buf].users}
                return len(buf_uses - set(node.snodes)) > 0

            if isinstance(node, FusedSchedulerNode):
                removed_buffers = {dep for dep in writes if not is_materialized(dep)}
                writes = writes - removed_buffers
                reads = reads - removed_buffers
            node_bytes = 0
            for buf in reads | writes:
                if buf in self.name_to_buffer:
                    buf = self.name_to_buffer[buf]
                elif buf in self.graph_inputs:
                    buf = self.graph_inputs[buf]
                else:
                    continue

                node_bytes += V.graph.sizevars.size_hint(
                    sympy_product(buf.get_size())
                ) * get_dtype_size(buf.get_dtype())
            return node_bytes

        total_bytes = 0
        node_counts = []
        for node in scheduler.nodes:
            num_bytes = get_read_write_buffers_sizes(node)
            node_counts.append((node, num_bytes // 4))
            total_bytes += num_bytes
        return total_bytes, node_counts

    @dynamo_timed
    def compile_to_module(self):
        from .codecache import PyCodeCache

        code, linemap = self.codegen()
        linemap = [(line_no, node.stack_trace) for line_no, node in linemap]
        key, path = PyCodeCache.write(code)
        mod = PyCodeCache.load_by_key_path(key, path, linemap=linemap)
        self.cache_key = key
        self.cache_path = path
        self.cache_linemap = linemap

        for name, value in self.constants.items():
            setattr(mod, name, value)

        # Logged twice as per https://github.com/pytorch/pytorch/pull/99038#discussion_r1167826029
        # TODO. Revisit this once the logging API is more mature
        output_code_log.info("Output code written to: %s", mod.__file__)
        log.debug("Output code written to: %s", mod.__file__)
        output_code_log.debug("Output code: \n%s", code)
        if config.benchmark_kernel:
            print(f"Compiled module path: {mod.__file__}", file=sys.stderr)
        V.debug.output_code(mod.__file__)
        V.debug.copy(os.path.splitext(mod.__file__)[0] + ".debug")
        return mod

    def compile_to_fn(self):
        if self.aot_mode and self.cpp_wrapper:
            from .codecache import AotCodeCache

            code, linemap = self.codegen()
            output_code_log.debug("Output code: \n%s", code)

            return AotCodeCache.compile(self, code, cuda=self.cuda)
        else:
            return self.compile_to_module().call

    def get_output_names(self):
        assert self.graph_outputs is not None
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
