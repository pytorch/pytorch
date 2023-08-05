import dataclasses
import importlib
import logging

import torch
import torch._C
import torch._ops
import torch._prims.executor
import torch.fx
import torch.onnx
import torch.onnx._internal
import torch.onnx._internal.diagnostics
import torch.onnx._internal.exporter
import torch.onnx._internal.fx.decomposition_table
import torch.onnx._internal.fx.passes

from functorch.compile import min_cut_rematerialization_partition
from torch._dynamo.backends.common import aot_autograd
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS
from torch.onnx._internal.exporter import ExportOptions
from torch.utils import _pytree

try:
    # Use try-except to initialize package-dependent global variables.
    from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Union

    import numpy as np
    import onnx
    import onnxruntime  # type: ignore[import]
    from onnxruntime.capi import _pybind_state as ORTC  # type: ignore[import]

    # This is not use directly in DORT but needed by underlying exporter,
    # so we still need to check if it exists.
    importlib.import_module("onnxscript")

    _NP_DTYPE = {
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.uint8: np.uint8,
        torch.int8: np.int8,
        torch.int16: np.int16,
        torch.int32: np.int32,
        torch.int64: np.longlong,
        torch.bool: np.bool_,
    }
    _ONNX_ELEMENT_TYPE_TO_TORCH_DTYPE = {
        onnx.TensorProto.FLOAT: torch.float32,  # type: ignore[attr-defined]
        onnx.TensorProto.FLOAT16: torch.float16,  # type: ignore[attr-defined]
        onnx.TensorProto.DOUBLE: torch.float64,  # type: ignore[attr-defined]
        onnx.TensorProto.BOOL: torch.bool,  # type: ignore[attr-defined]
        onnx.TensorProto.UINT8: torch.uint8,  # type: ignore[attr-defined]
        onnx.TensorProto.INT8: torch.int8,  # type: ignore[attr-defined]
        onnx.TensorProto.INT16: torch.int16,  # type: ignore[attr-defined]
        onnx.TensorProto.INT32: torch.int32,  # type: ignore[attr-defined]
        onnx.TensorProto.INT64: torch.int64,  # type: ignore[attr-defined]
    }
    _TORCH_DTYPE_TO_ONNX_ELEMENT_TYPE = {
        value: key for key, value in _ONNX_ELEMENT_TYPE_TO_TORCH_DTYPE.items()
    }

    _SUPPORT_ONNXRT = True
except ImportError:
    _NP_DTYPE = {}
    _ONNX_ELEMENT_TYPE_TO_TORCH_DTYPE = {}
    _TORCH_DTYPE_TO_ONNX_ELEMENT_TYPE = {}
    _SUPPORT_ONNXRT = False


def has_onnxruntime():
    return _SUPPORT_ONNXRT


def _nvtx_range_push(name: str):
    """If PyTorch is installed with CUDA support, this starts NVTX range.

    Check torch.cuda.nvtx.range_push's document for more details.
    """
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)


def _nvtx_range_pop():
    """If PyTorch is installed with CUDA support, this terminates NVTX range.

    Check torch.cuda.nvtx.range_pop's document for more details.
    """
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_pop()


def _get_ort_device_type(device_type: str):
    if device_type == "cuda":
        return ORTC.OrtDevice.cuda()
    if device_type == "cpu":
        return ORTC.OrtDevice.cpu()
    # ort pytorch device is mapped to NPU OrtDevice type
    if device_type == "ort":
        return ORTC.OrtDevice.npu()
    raise ValueError("Unsupported device type: " + device_type)


logger = logging.getLogger(__name__)
# Uncomment the following lines to print out development info.
# logging.basicConfig(level=logging.INFO)
# logger.setLevel(logging.INFO)


class OrtOperatorSupport(OperatorSupport):
    """
    Operator support for ONNXRuntime backend. It has two-level of support decision.
    One is via support_dict and the other one is via extra_support_dict. The logic
    of using support_dict is implemented in OrtOperatorSupport and extra_support_dict
    is used by OperatorSupport.is_node_supported.
    """

    def __init__(self, support_dict: Set[Any], extra_support_dict: Dict[str, Any]):
        # Use extra_support_dict[op_name] = None to indicate
        # we support op_name with all input types. Otherwise,
        # see support_dict (type: SupportDict) in operator_support.py
        # for specifying supported types.
        super().__init__(extra_support_dict)
        self._onnx_support_dict = support_dict

    def is_node_supported(
        self, submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        # OperatorSupport.is_node_supported returns True for non-callable nodes.
        # Since ORT can't execute them, we return False here to override the base
        # behavior.
        if node.op not in CALLABLE_NODE_OPS:
            return False
        # This is the and the only place to decide if aten op is supported.
        if node.op == "call_function" and node.target in self._onnx_support_dict:
            logger.info(
                "support_dict supports node.target: %s (type: %s)",
                node.target,
                type(node.target),
            )
            return True
        logger.info(
            "support_dict doesn't support node.target: %s (type: %s)",
            node.target,
            type(node.target),
        )
        # If node.target is not in support_dict, we still want to check if torch.jit.script
        # can convert it to ONNX equivalence. Let's use base mechanism to do this.
        # See extra_support_dict  for supported ops.
        if super().is_node_supported(submodules, node):
            logger.info(
                "extra_support_dict supports node.target: %s (type: %s)",
                node.target,
                type(node.target),
            )
            return True
        logger.info(
            "extra_support_dict doesn't supports node.target: %s (type: %s)",
            node.target,
            type(node.target),
        )
        return False


def _move_placeholder_to_front(graph_module: torch.fx.GraphModule) -> None:
    """
    In torch.fx.Graph, placehoder is a special assignment node. If it's not
    executed in the beginning, it could overwrite values computed by upstream
    nodes.
    """

    graph = graph_module.graph
    placeholders = []
    first_not_placeholder = None
    for node in graph.nodes:
        if node.op == "placeholder":
            placeholders.append(node)
        if first_not_placeholder is None and node.op != "placeholder":
            first_not_placeholder = node
    if first_not_placeholder is None:
        return
    for placeholder in placeholders:
        first_not_placeholder.prepend(placeholder)


def _replace_to_copy_with_to(fx_module: torch.fx.GraphModule) -> None:
    # aten._to_copy doesn't have exporter so we replace it with aten.to.
    for node in fx_module.graph.nodes:
        if (
            isinstance(node.target, torch._ops.OpOverload)
            and node.target.overloadpacket == torch.ops.aten._to_copy
        ):
            is_default_layout = True
            is_on_same_device = True
            is_cast = True
            are_kwargs_supported = True
            if "layout" in node.kwargs and node.kwargs["layout"] != torch.strided:
                is_default_layout = False
            if (
                "device" in node.kwargs
                and node.kwargs["device"] != node.args[0].meta["val"].device
            ):
                is_on_same_device = False
            if "dtype" not in node.kwargs:
                is_cast = False
            for kwarg in node.kwargs:
                if kwarg not in ["layout", "device", "dtype"]:
                    are_kwargs_supported = False

            if (
                len(node.args) == 1
                and is_default_layout
                and is_on_same_device
                and is_cast
                and are_kwargs_supported
            ):
                # This aten::_to_copy looks like ONNX Cast, so other kwargs are ignored.
                # This change could lead to invalid FX graph but it doesn't matter, as long as the downstream backend,
                # ONNXRuntime, can execute the exported ONNX graph.
                node.kwargs = {"dtype": node.kwargs["dtype"]}

                node.target = torch.ops.aten.to.dtype
            else:
                raise RuntimeError(
                    f"aten._to_copy must be replaced with other ONNX-supported aten ops. \
                         args={[arg.meta for arg in node.args]}, kwargs={node.kwargs}"
                )
    fx_module.recompile()


def _create_onnx_session(onnx_proto, eps: Tuple[str, ...], session_options):
    # TODO(wechi): Add more EPs per PyTorch device types.
    # TODO(wechi): enable external allocators.
    return onnxruntime.InferenceSession(
        onnx_proto, providers=eps, sess_options=session_options
    )


def _infer_ep_from_device(*args) -> Tuple[str, ...]:
    """Return the first valid device (i.e., GPU or CPU) in argument list."""
    eps = []
    for arg in args:
        if hasattr(arg, "device"):
            device = arg.device
            if device.type == "cuda":
                eps.append("CUDAExecutionProvider")
            elif device.type == "cpu":
                eps.append("CPUExecutionProvider")
    return tuple(eps)


def _extract_graph_module_inputs(graph_module: torch.fx.GraphModule) -> Tuple[Any, ...]:
    placeholders = []
    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            if hasattr(node, "meta") and "val" in node.meta:
                assert isinstance(node.meta["val"], torch.Tensor)
            placeholders.append(node)
    return tuple(placeholders)


def _extract_graph_module_outputs(graph_module: torch.fx.GraphModule) -> Any:
    """Collect "val" fields from outputs metadata in this torch.fx.GraphModule."""
    for node in graph_module.graph.nodes:
        if node.op == "output":
            # Output node is unique. Let's retrieve output values from
            # this node's input list. And then just return.
            return node.args[0]
    raise ValueError("No output node found in this torch.fx.GraphModule.")


def _infer_ep_from_graph_module(graph_module: torch.fx.GraphModule) -> Tuple[str, ...]:
    """Return the all valid devices (i.e., GPU or CPU) among outputs of this torch.fx.GraphModule."""
    flattened_output_args, _ = _pytree.tree_flatten(
        _extract_graph_module_outputs(graph_module)
    )
    # Output arguments with example value (type: torch.Tensor) in the `graph_module`.
    selected_output_args = [
        output_arg.meta["val"]
        for output_arg in flattened_output_args
        # output_arg must have tensor for its device information.
        # Otherwise, skip it.
        if (hasattr(output_arg, "meta") and "val" in output_arg.meta)
    ]
    return _infer_ep_from_device(*selected_output_args)


def _sort_eps(eps: Tuple[str, ...]) -> Tuple[str, ...]:
    """Sort execution providers in eps based on pre-set priority."""

    def get_execution_provider_priority(ep: str) -> int:
        if ep == "CPUExecutionProvider":
            # Lowest priority.
            return 2
        if ep == "CUDAExecutionProvider":
            # Higher priority than CPU but lower than
            # other specialized EPs.
            return 1
        # Highest priority.
        return 0

    unique_eps = set(eps)
    return tuple(sorted(unique_eps, key=get_execution_provider_priority, reverse=True))


def _get_onnx_devices(values: Tuple[torch.Tensor, ...]) -> Tuple[ORTC.OrtDevice, ...]:
    assert all(
        value.device == values[0].device for value in values
    ), "All values must be on the same device."

    def _device_id_or_zero(device_id: int) -> int:
        return device_id or 0

    devices: Tuple[ORTC.OrtDevice, ...] = tuple(
        ORTC.OrtDevice(
            _get_ort_device_type(value.device.type),
            ORTC.OrtDevice.default_memory(),
            _device_id_or_zero(value.device.index),
        )
        for value in values
    )
    return devices


def _get_ortvalues_from_torch_tensors(
    tensors: Tuple[torch.Tensor, ...], devices: Tuple[ORTC.OrtDevice, ...]
) -> Tuple[torch.Tensor, ...]:
    ortvalues = ORTC.OrtValueVector()
    ortvalues.reserve(len(tensors))
    dtypes = []
    shapes = []
    data_ptrs = []

    for tensor in tensors:
        dtypes.append(_NP_DTYPE[tensor.dtype])
        shapes.append(tensor.size())
        data_ptrs.append(tensor.data_ptr())
    ortvalues.push_back_batch(tensors, data_ptrs, dtypes, shapes, devices)
    return ortvalues


def _to_real_tensor(tensor: FakeTensor) -> torch.Tensor:
    if tensor.is_sparse:
        raise ValueError("sparse tensor is not yet supported.")
    out = torch.empty(tensor.size(), dtype=tensor.dtype, device=tensor.device)
    return out


def _run_onnx_session_with_ortvaluevector(
    sess: onnxruntime.InferenceSession,
    input_names: Tuple[str, ...],
    inputs: Tuple[torch.Tensor, ...],
    input_devices: Tuple[ORTC.OrtDevice, ...],
    output_names: Tuple[str, ...],
    outputs: Tuple[torch.Tensor, ...],
    output_devices: Tuple[ORTC.OrtDevice, ...],
    preallocate_output: bool,
) -> Tuple[torch.Tensor, ...]:
    _nvtx_range_push("contiguous")
    inputs = tuple(a.contiguous() for a in inputs)
    _nvtx_range_pop()

    _nvtx_range_push("push_back_batch")

    ort_inputs = _get_ortvalues_from_torch_tensors(inputs, input_devices)

    # preallocate output pytorch Tensors and use the buffers affined to the torch device for the output ortvalue.
    # Because the output ortvalue is not allocated and owned by ort, it does not need to convert the output ortvalue
    # to torch Tensor transferring the ownership.
    if preallocate_output:
        pth_outputs = tuple(
            _to_real_tensor(t) if isinstance(t, FakeTensor) else t for t in outputs
        )
        ort_outputs = _get_ortvalues_from_torch_tensors(pth_outputs, output_devices)
    else:
        ort_outputs = ORTC.OrtValueVector()
    _nvtx_range_pop()

    _nvtx_range_push("run_with_ortvaluevector")
    run_options = onnxruntime.RunOptions()
    run_options.add_run_config_entry("disable_synchronize_execution_providers", "1")
    sess.run_with_ortvaluevector(
        run_options, input_names, ort_inputs, output_names, ort_outputs, output_devices
    )
    _nvtx_range_pop()

    if preallocate_output:
        return pth_outputs
    else:
        _nvtx_range_push("after run_with_ortvaluevector")
        pth_outputs = onnxruntime.training.ortmodule._utils._ortvalues_to_torch_tensor(
            ort_outputs
        )
        _nvtx_range_pop()
        return pth_outputs


def _assert_allclose_with_detailed_error_message(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = 1e-03,
    atol: float = 1e-04,
):
    diff = actual - expected
    real_atol = torch.max(torch.abs(diff))
    max_value = torch.max(torch.abs(actual), torch.abs(expected))
    max_value[max_value == 0.0] = 1.0
    real_rtol = torch.max(diff / max_value)
    allclose = bool(real_atol <= atol or real_rtol <= rtol)
    if not allclose:
        raise RuntimeError(
            "ONNX output doesn't match baseline output with "
            f"actual rtol={real_rtol} and actual atol={real_atol} "
            f"but expected rtol={rtol} and expected atol={atol}."
        )


class OrtExecutionInfoPerSession:
    """Information required to execute torch.fx.GraphModule using onnxruntime.InferenceSession"""

    def __init__(
        self,
        session: onnxruntime.InferenceSession,
        input_names: Tuple[str, ...],
        input_value_infos: Tuple[onnx.ValueInfoProto, ...],  # type: ignore[name-defined]
        output_names: Tuple[str, ...],
        output_value_infos: Tuple[onnx.ValueInfoProto, ...],  # type: ignore[name-defined]
        input_devices: Tuple[ORTC.OrtDevice, ...],
        output_devices: Tuple[ORTC.OrtDevice, ...],
        example_outputs: Union[Tuple[torch.Tensor, ...], torch.Tensor],
    ):
        # Carrier of ONNX model and its executor.
        self.session: onnxruntime.InferenceSession = session
        # For the ONNX model stored in self.session, self.input_names[i] is the
        # name of the i-th positional input.
        self.input_names: Tuple[str, ...] = input_names
        # self.input_name[i]'s type information is stored in self.input_value_infos[i].
        self.input_value_infos: Tuple[onnx.ValueInfoProto, ...] = input_value_infos  # type: ignore[name-defined]
        # Similar to self.input_names, but for outputs.
        self.output_names: Tuple[str, ...] = output_names
        # Similar to self.input_value_infos but for outputs.
        self.output_value_infos: Tuple[onnx.ValueInfoProto, ...] = output_value_infos  # type: ignore[name-defined]
        # For the ONNX model stored in self.session, self.input_devices[i] is the
        # i-th positional input's device.
        self.input_devices: Tuple[ORTC.OrtDevice, ...] = input_devices
        # Similar to self.input_devices, but for outputs.
        self.output_devices: Tuple[ORTC.OrtDevice, ...] = output_devices
        # This is the outputs of executing the original torch.fx.GraphModule with example inputs
        # (i.e., args passed into OrtBackend._ort_acclerated_call).
        self.example_outputs: Union[
            Tuple[torch.Tensor, ...], torch.Tensor
        ] = example_outputs

    def is_supported(self, *args):
        # Compare the args and the input schema in ONNX model and
        # return the first match.
        if len(args) != len(self.input_value_infos):
            return False
        for arg, value_info in zip(args, self.input_value_infos):
            if not isinstance(arg, torch.Tensor):
                return False
            onnx_dtype = _TORCH_DTYPE_TO_ONNX_ELEMENT_TYPE[arg.dtype]
            if onnx_dtype != value_info.type.tensor_type.elem_type:
                return False
            for dim, onnx_dim in zip(arg.shape, value_info.type.tensor_type.shape.dim):
                if isinstance(dim, int) and (
                    onnx_dim.dim_value == dim or onnx_dim.dim_param
                ):
                    continue
                elif isinstance(dim, torch.SymInt) and onnx_dim.dim_param:
                    continue
                else:
                    return False
        return True


@dataclasses.dataclass
class OrtExecutionInfoForAllGraphModules:
    def __init__(self):
        # All sessions (and their related information) created by exporting the same GraphModule
        # with different inputs.
        self.execution_info_per_graph_module: Dict[
            torch.fx.GraphModule, List[OrtExecutionInfoPerSession]
        ] = {}

    def search_reusable_session_execution_info(
        self, graph_module: torch.fx.GraphModule, *args
    ):
        if graph_module not in self.execution_info_per_graph_module:
            return None
        # All execution information for ONNX models exported from the same `graph_module`
        # with different inputs.
        candidates = self.execution_info_per_graph_module[graph_module]

        for candidate in candidates:
            if candidate.is_supported(*args):
                # Returns the first session that accepts this input schema.
                return candidate
        # No reusable session found.
        return None

    def cache_session_execution_info(
        self, graph_module: torch.fx.GraphModule, info: OrtExecutionInfoPerSession
    ):
        if graph_module not in self.execution_info_per_graph_module:
            self.execution_info_per_graph_module[graph_module] = [info]
        else:
            self.execution_info_per_graph_module[graph_module].append(info)


class OrtBackend:
    """A backend compiles (sub-)graphs in torch.fx.GraphModule to onnxruntime.InferenceSession calls.

    The compiler entry point is OrtBackend.compile, which
        1. partitions the original graph into supported sub-graphs (type: torch.fx.GrpahModule) and unsupported
           sub-graphs.
        2. For each supported sub-graph, it replaces its _wrapped_call function with _ort_accelerated_call.
        3. Inside _ort_accelerated_call, it creates onnxruntime.InferenceSession and calls it to execute the sub-graph.
    """

    def __init__(
        self,
        ep: str = "CPUExecutionProvider",
        preallocate_output: bool = False,
        session_options=None,
        onnx_exporter_options: Optional[
            "torch.onnx._internal.exporter.ExportOptions"
        ] = None,
    ):
        # onnx_exporter_options contains information shared between exporter and DORT.
        # For example, they should use the same decomposition table when
        #  1. capturing FX graph in torch.compile (see how we create aot_ort in register_backend.py)
        #  2. call exporter's API to convert `torch.fx.GraphModule` to ONNX model
        #     (see onnxfunction_dispatcher passed to FxOnnxInterpreter.run below).
        if onnx_exporter_options is None:
            onnx_exporter_options = torch.onnx._internal.exporter.ExportOptions()
        # Convert user-facing option to internal option used by ONNX exporter
        # to access required information.
        # Some useful fields:
        # - Decomposition table for decomposing FX operators in exporter is
        #   self.resolved_onnx_exporter_options.decomposition_table.
        # - self.resolved_onnx_exporter_options.onnx_registry records what
        #   aten/prim ops are supported by exporter and their exporters (type: callable).
        self.resolved_onnx_exporter_options = (
            torch.onnx._internal.exporter.ResolvedExportOptions(onnx_exporter_options)
        )

        # TODO(wechi): This line must generate result identical to the call of
        # _create_onnx_supports_op_overload_table(...) inside
        # create_onnx_friendly_decomposition_table(...) in
        # torch/onnx/_internal/fx/decomposition_table.py.
        support_dict = torch.onnx._internal.fx.decomposition_table._create_onnx_supports_op_overload_table(
            # This is identical to self.resolved_onnx_exporter_options.onnxfunction_dispatcher.onnx_registry.
            self.resolved_onnx_exporter_options.onnx_registry
        )

        extra_support_dict: Dict[str, Any] = {
            "getattr": None,
            "_operator.getitem": None,
        }

        self._supported_ops = OrtOperatorSupport(support_dict, extra_support_dict)
        # TODO: this is a naive implementation of cache without proper guard
        self._partitioner_cache: Dict[torch.fx.GraphModule, torch.fx.GraphModule] = {}
        # Conceptually, this filed is a 2-layer dictionary
        #   GraphModule 0
        #     ONNX Model 0 (with ORT InferenceSession and related information. type: OrtExecutionInfoPerSession)
        #     ONNX Model 1
        #     ...
        #   GraphModule 1
        #     ONNX Model 2 (with ORT InferenceSession and related information. type: OrtExecutionInfoPerSession)
        #     ONNX Model 3
        #     ...
        #   ...
        # , which caches all previous compilation result so that we can reuse them.
        # ONNX Model 0 and 1 are exported from the same GraphModule 0 but with different inputs
        # (e.g., tensors with different ranks). GraphModule 0 and GraphModule 1 are different
        # graphs captured by Dynamo and sent to OrtBackend.compile.
        self._all_ort_execution_info = OrtExecutionInfoForAllGraphModules()

        self._assert_allclose_to_baseline = False

        self.ep = ep
        self.session_options = session_options
        self.execution_count = 0

        # preallocate_output allows for allocating output torch Tensor buffers and feeding them to InferenceSession
        # in order to avoid internal allocation of output buffers in InferenceSession.
        # If output ortvalue returned from InferenceSession is allocated internally,
        # it needs to be converted to torch Tensor for return, and the torch Tensor should hold the ownership.
        # When a custom torch device is used with a custom aten allocator, the conversion from ortvalue to torch Tensor
        # should be supported, which is currently done through dlpack. Note that dlpack might not support a custom torch device.
        # It can be avoided by allowing for preallocation for output buffers allocated by a custom aten allocator,
        # and use the preallocated output buffers for InferenceSession not holding any ownership for them.
        self.preallocate_output = preallocate_output

    def _ort_acclerated_call(self, graph_module: torch.fx.GraphModule, *args, **kwargs):
        cached_execution_info_per_session = (
            self._all_ort_execution_info.search_reusable_session_execution_info(
                graph_module, *args
            )
        )
        if cached_execution_info_per_session:
            onnx_session = cached_execution_info_per_session.session
            input_names = cached_execution_info_per_session.input_names
            output_names = cached_execution_info_per_session.output_names
            input_devices = cached_execution_info_per_session.input_devices
            output_devices = cached_execution_info_per_session.output_devices
            prim_outputs = cached_execution_info_per_session.example_outputs
        else:
            # It's first time seeing such as graph. Let's make a new session
            # (type: onnxruntime.InferenceSession) for it.

            # TODO(wechi): this is a workaround for pytorch/pytorch#84311.
            _move_placeholder_to_front(graph_module)
            # Generate reference outputs. They are used to indicate output
            # tensors' types and devices when calling ORT.
            #
            # WARNING: The downstream code should not change prim_outputs and
            # this backend should always produces output with schema identical to prim_outputs'.

            if self.resolved_onnx_exporter_options.dynamic_shapes:
                # No pre-allocation when dynamic shape is enabled.
                self.preallocate_output = False
                extracted_outputs = _extract_graph_module_outputs(graph_module)

                def maybe_map_to_meta_val(value):
                    if hasattr(value, "meta") and "val" in value.meta:
                        # Select outputs with "val" information. Without "val",
                        # it's not possible access output_arg.meta["val"].device.
                        return value.meta["val"]
                    else:
                        return value

                prim_outputs = _pytree.tree_map(
                    maybe_map_to_meta_val, extracted_outputs
                )
            else:
                try:
                    prim_outputs = FakeTensorProp(graph_module).propagate(
                        *args, **kwargs
                    )
                except Exception:
                    logger.info("FakeTensorProb failed for %s", graph_module)
                    # When FakeTensorProp fails, it is not possible to preallocate output buffers
                    # because the output shapes are not inferred.
                    self.preallocate_output = False

                    # rethrow FakeTensorProb failure because it is not yet currently handled.
                    raise

            from torch.onnx._internal.fx import fx_onnx_interpreter

            # Create the object to iterate through the nodes in graph one-by-one
            # and calls the corresponding ONNX exporter for each node.
            fx_interpreter = fx_onnx_interpreter.FxOnnxInterpreter(
                diagnostic_context=self.resolved_onnx_exporter_options.diagnostic_context
            )
            # Cast FX variables if they will result schema-mismatch when searching
            # for ONNX operator. E.g., add(double_tensor, int_tensor) is fine in PyTorch,
            # but ONNX expects add(double_tensor, double_tensor).
            graph_module = torch.onnx._internal.fx.passes.InsertTypePromotion(
                self.resolved_onnx_exporter_options.diagnostic_context, graph_module
            ).run()
            # Start the per-node exporting process. It's conceptually a for loop
            # scanning through the nodes in the graph.
            exported = fx_interpreter.run(
                fx_graph_module=graph_module,
                onnxfunction_dispatcher=self.resolved_onnx_exporter_options.onnxfunction_dispatcher,
                op_level_debug=self.resolved_onnx_exporter_options.op_level_debug,
            )
            # Convert the exported result to ONNX ModelProto.
            onnx_model = exported.to_model_proto(
                opset_version=self.resolved_onnx_exporter_options.onnx_registry.opset_version,
            )

            # Initialize a ORT session to execute this ONNX model.
            # Note that TorchDynamo assumes all inputs/outputs are on the
            # same device, but it's subject to change (very likely with
            # dynamic shape support), so we add execution providers
            # based on the all inputs/outputs plus a default OrtBackend.ep.
            eps_from_args = _infer_ep_from_device(args)
            eps_from_graph_module = _infer_ep_from_graph_module(graph_module)
            if eps_from_args:
                # If user feeds CUDA tensor as input argument,
                # we want to use CUDA EP.
                # Thus, `eps_from_args` (deduced from input arguments)
                # has highest priority.
                selected_eps = _sort_eps((*eps_from_args, self.ep))
            elif eps_from_graph_module:
                # If there is no EP in input arguments, we deduce EP from
                # graph_module's outputs. Those outputs may come from
                # FakeTensorProp or Dynamo's built-in symbolic shape inference.
                selected_eps = _sort_eps((*eps_from_graph_module, self.ep))
            else:
                # No EP found in inputs and outputs, let's use default.
                selected_eps = (self.ep,)

            onnx_session = _create_onnx_session(
                onnx_model.SerializeToString(), selected_eps, self.session_options
            )
            # Cache ORT session. It's reused for the same "graph_module".
            # Generate ONNX model and extract its input and output names.
            # TODO(wechi): ORT session should provide a API to extract
            # input and output names from the underlying model.
            input_names = tuple(input.name for input in onnx_model.graph.input)
            output_names = tuple(output.name for output in onnx_model.graph.output)
            input_devices = _get_onnx_devices(args)
            # Cache devices for inputs and outputs. They are used to invoke
            # ORT session. Output devices indicate where (e.g., GPU or CPU)
            # to store outputs
            if isinstance(prim_outputs, tuple):
                output_devices = _get_onnx_devices(prim_outputs)
            else:
                output_devices = _get_onnx_devices((prim_outputs,))

            execution_info_per_session = OrtExecutionInfoPerSession(
                session=onnx_session,
                input_names=input_names,
                input_value_infos=tuple(input for input in onnx_model.graph.input),
                output_names=output_names,
                output_value_infos=tuple(output for output in onnx_model.graph.output),
                input_devices=input_devices,
                output_devices=output_devices,
                example_outputs=prim_outputs,
            )

            self._all_ort_execution_info.cache_session_execution_info(
                graph_module, execution_info_per_session
            )

        self.execution_count += 1

        if isinstance(prim_outputs, tuple):
            assert all(isinstance(elem, torch.Tensor) for elem in prim_outputs)
            # ORT always returns a tuple of outputs. If the original is a tuple, just returning
            # ORT output is ok.
            _nvtx_range_push("run_onnx_session_with_ortvaluevector")
            onnx_outputs = _run_onnx_session_with_ortvaluevector(
                onnx_session,
                input_names,
                args,
                input_devices,
                output_names,
                prim_outputs,
                output_devices,
                self.preallocate_output,
            )
            _nvtx_range_pop()
            if self._assert_allclose_to_baseline:
                # Compute baseline.
                baseline_outputs = torch._prims.executor.execute(
                    graph_module, *args, executor="aten"
                )
                # Ensure every output tensor is close to the corresponding baseline.
                for onnx_output, baseline_output in zip(onnx_outputs, baseline_outputs):
                    _assert_allclose_with_detailed_error_message(
                        onnx_output, baseline_output
                    )
            return onnx_outputs
        else:
            assert isinstance(prim_outputs, torch.Tensor)
            # ORT always returns a tuple of outputs. If the original output is a tensor,
            # ORT output's first element must be extracted and returned. Otherwise, type
            # mismatch may happen in downstream computation.
            onnx_outputs = _run_onnx_session_with_ortvaluevector(
                onnx_session,
                input_names,
                args,
                input_devices,
                output_names,
                (prim_outputs,),
                output_devices,
                self.preallocate_output,
            )
            assert len(onnx_outputs) == 1
            if self._assert_allclose_to_baseline:
                # Compute baseline.
                baseline_outputs = torch._prims.executor.execute(
                    graph_module, *args, executor="aten"
                )
                # Ensure output tensor is close to the corresponding baseline.
                _assert_allclose_with_detailed_error_message(
                    onnx_outputs[0], baseline_outputs
                )
            return onnx_outputs[0]

    def compile(self, graph_module: torch.fx.GraphModule, args) -> torch.fx.GraphModule:
        # FX graph based partitioning based on ONNX supported ops.
        if graph_module in self._partitioner_cache:
            partitioned_prim_graph_module = self._partitioner_cache[graph_module]
        else:
            prim_graph_module = graph_module
            # TODO(wechi): this is required for removing aten::_to_copy in _replace_to_copy_with_to.
            _replace_to_copy_with_to(prim_graph_module)
            partitioner = CapabilityBasedPartitioner(
                prim_graph_module,
                self._supported_ops,
                allows_single_node_partition=True,
            )
            partitioned_prim_graph_module = partitioner.partition_and_fuse()
            self._partitioner_cache[graph_module] = partitioned_prim_graph_module

            # Overriding fused_module's __call__() function with ort_acclerated_call()
            # This loop goes through all graph partitions (each of them is an ONNX-representable graph)
            # and override their _wrappped_call function with _ort_accelerated_call.
            # Inside _ort_accelerated_call, the partition's graph is exported into ONNX and executed by ORT.
            for node in partitioned_prim_graph_module.graph.nodes:
                # TODO: use a better way to identify fused submodule
                if node.op == "call_module" and "fused_" in node.name:
                    fused_module = getattr(partitioned_prim_graph_module, node.name)
                    # self.ort_acclerated_call is responsible for exporting graph to ONNX,
                    # creating ORT session, and running ORT session.
                    fused_module._wrapped_call = self._ort_acclerated_call

        return partitioned_prim_graph_module

    def __call__(
        self, graph_module: torch.fx.GraphModule, args
    ) -> torch.fx.GraphModule:
        return self.compile(graph_module, args)


def make_aot_ort(dynamic: bool = True):
    """Wrap OrtBackend as PyTorch's AOT compiler.

    Example usages:
        import torch
        from onnxruntime.training.torchdynamo.register_backend import make_aot_ort
        use_dynamic = True
        local_aot_ort, _ = make_aot_ort(dynamic = use_dynamic)

        @torch._dynamo.optimize(local_aot_ort, dynamic=use_dynamic)
        def foo(x: torch.Tensor):
            return torch.sigmoid(x)

        x = torch.rand(2, 2, dtype=torch.float)
        torch.testing.assert_close(torch.sigmoid(x), foo(x))
    """
    ort_backend = OrtBackend(
        onnx_exporter_options=ExportOptions(dynamic_shapes=dynamic)
    )
    return (
        aot_autograd(
            fw_compiler=ort_backend,
            partition_fn=min_cut_rematerialization_partition,
            decompositions=ort_backend.resolved_onnx_exporter_options.decomposition_table,
        ),
        ort_backend,
    )
