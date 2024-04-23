# @generated from torch/_C/__init__.pyi.in
# mypy: disable-error-code="type-arg"

import builtins
from enum import Enum, IntEnum
from pathlib import Path
from typing import (
    Any,
    AnyStr,
    BinaryIO,
    Callable,
    ContextManager,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Set,
    SupportsIndex,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
    runtime_checkable,
)
from typing_extensions import ParamSpec

import numpy

import torch
from torch import inf, SymInt, Tensor
from torch.autograd.graph import Node as _Node
from torch.package import PackageExporter
from torch.storage import UntypedStorage, TypedStorage
from torch.types import (
    _bool,
    _complex,
    _device,
    _dispatchkey,
    _dtype,
    _float,
    _int,
    _layout,
    _qscheme,
    _size,
    Device,
    Number,
    Storage,
)

from torch._prims_common import DeviceLikeType

# This module is defined in torch/csrc/Module.cpp

from . import _functorch, _lazy, _lazy_ts_backend, _nn, _onnx, _VariableFunctions, _cpu, _aoti, _verbose

K = TypeVar("K")
T = TypeVar("T")
S = TypeVar("S", bound="torch.Tensor")
P = ParamSpec("P")
ReturnVal = TypeVar("ReturnVal", covariant=True)  # return value (always covariant)
_T_co = TypeVar("_T_co", covariant=True)


@runtime_checkable
class _NestedSequence(Protocol[_T_co]):
    """A protocol for representing nested sequences.

    References::
        `numpy._typing._NestedSequence`
        <https://github.com/numpy/numpy/blob/main/numpy/_typing/_nested_sequence.py>
    """

    def __len__(self, /) -> builtins.int: ...
    def __getitem__(self, index: builtins.int, /) -> _T_co | _NestedSequence[_T_co]: ...
    def __contains__(self, x: builtins.object, /) -> builtins.bool: ...
    def __iter__(self, /) -> Iterator[_T_co | _NestedSequence[_T_co]]: ...
    def __reversed__(self, /) -> Iterator[_T_co | _NestedSequence[_T_co]]: ...
    def count(self, value: Any, /) -> builtins.int: ...
    def index(self, value: Any, /) -> builtins.int: ...


# Defined in torch/csrc/Device.cpp
class device:
    type: str  # THPDevice_type
    index: _int  # THPDevice_index

    def __get__(self, instance, owner=None) -> device: ...

    # THPDevice_pynew
    @overload
    def __init__(self, device: DeviceLikeType) -> None: ...
    @overload
    def __init__(self, type: str, index: _int) -> None: ...

    # Uncomment if we ever make torch.device a decorator
    # def __call__(self, func: T) -> T: ...

    def __enter__(self) -> device: ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...
    def __reduce__(self) -> Tuple[Any, ...]: ...  # THPDevice_reduce

# Defined in torch/csrc/Stream.cpp
class Stream:
    stream_id: _int  # Stream id
    device_index: _int
    device_type: _int

    device: device  # The device of the stream

# Defined in torch/csrc/Size.cpp
class Size(Tuple[_int, ...]):
    # TODO: __reduce__

    @overload  # type: ignore[override]
    def __getitem__(self: Size, key: _int) -> _int: ...
    @overload
    def __getitem__(self: Size, key: slice) -> Size: ...
    def numel(self: Size) -> _int: ...

# Defined in torch/csrc/Dtype.cpp
class dtype:
    # TODO: __reduce__
    is_floating_point: _bool
    is_complex: _bool
    is_signed: _bool
    itemsize: _int
    def to_real(self) -> dtype: ...
    def to_complex(self) -> dtype: ...

# Defined in torch/csrc/TypeInfo.cpp
class iinfo:
    bits: _int
    min: _int
    max: _int
    dtype: str

    def __init__(self, dtype: _dtype) -> None: ...

class finfo:
    bits: _int
    min: _float
    max: _float
    eps: _float
    tiny: _float
    smallest_normal: _float
    resolution: _float
    dtype: str

    @overload
    def __init__(self, dtype: _dtype) -> None: ...
    @overload
    def __init__(self) -> None: ...

float32: dtype = ...
float: dtype = ...
float64: dtype = ...
double: dtype = ...
float16: dtype = ...
bfloat16: dtype = ...
float8_e4m3fn: dtype = ...
float8_e4m3fnuz: dtype = ...
float8_e5m2: dtype = ...
float8_e5m2fnuz: dtype = ...
half: dtype = ...
uint8: dtype = ...
uint16: dtype = ...
uint32: dtype = ...
uint64: dtype = ...
int8: dtype = ...
int16: dtype = ...
short: dtype = ...
int32: dtype = ...
int: dtype = ...
int64: dtype = ...
long: dtype = ...
complex32: dtype = ...
complex64: dtype = ...
chalf: dtype = ...
cfloat: dtype = ...
complex128: dtype = ...
cdouble: dtype = ...
quint8: dtype = ...
qint8: dtype = ...
qint32: dtype = ...
bool: dtype = ...
quint4x2: dtype = ...
quint2x4: dtype = ...
bits1x8: dtype = ...
bits2x4: dtype = ...
bits4x2: dtype = ...
bits8: dtype = ...
bits16: dtype = ...

# Defined in torch/csrc/Layout.cpp
class layout: ...

# Defined in torch/csrc/utils/disable_torch_function.cpp
def DisableTorchFunction(): ...
def DisableTorchFunctionSubclass(): ...

# Defined in torch/csrc/utils/tensor_layouts.cpp
strided: layout = ...
sparse_coo: layout = ...
sparse_csr: layout = ...
sparse_csc: layout = ...
sparse_bsr: layout = ...
sparse_bsc: layout = ...
_mkldnn: layout = ...
jagged: layout = ...

# Defined in torch/csrc/MemoryFormat.cpp
class memory_format: ...

# Defined in torch/csrc/utils/tensor_memoryformats.cpp
contiguous_format: memory_format = ...
channels_last: memory_format = ...
channels_last_3d: memory_format = ...
preserve_format: memory_format = ...

# Defined in torch/csrc/QScheme.cpp
class qscheme: ...

# Defined in torch/csrc/utils/tensor_qschemes.h
per_tensor_affine: qscheme = ...
per_channel_affine: qscheme = ...
per_tensor_symmetric: qscheme = ...
per_channel_symmetric: qscheme = ...
per_channel_affine_float_qparams: qscheme = ...

# Defined in torch/csrc/autograd/python_function.cpp
class _FunctionBase:
    saved_tensors: Tuple[Tensor]
    _raw_saved_tensors: Tuple[Any]
    next_functions: Tuple[Tuple[Any, _int], ...]
    needs_input_grad: Tuple[_bool]
    metadata: dict
    _materialize_non_diff_grads: _bool
    # skip adding type hints for the fields that have wrappers defined
    # in torch/autograd/function.py

# Defined in torch/csrc/autograd/python_legacy_variable.cpp
class _LegacyVariableBase(Tensor):  # inherits from Tensor to appease mypy
    def __init__(
        self,
        data: Optional[Tensor] = ...,
        requires_grad: Optional[_bool] = ...,
        volatile: Optional[_bool] = ...,
        _grad_fn: Optional[_FunctionBase] = ...,
    ) -> None: ...

# Defined in torch/csrc/jit/python/init.cpp
class IODescriptor: ...
class JITException: ...

class Future(Generic[T]):
    def __init__(self, devices: List[device]) -> None: ...
    def done(self) -> _bool: ...
    def value(self) -> T: ...
    def wait(self) -> T: ...
    def add_done_callback(self, callback: Callable) -> None: ...
    def then(self, callback: Callable) -> Future[T]: ...
    def set_result(self, result: T) -> None: ...
    def _set_unwrap_func(self, callback: Callable) -> None: ...

class _Await:
    def __init__(self) -> None: ...
    def fn(self) -> Callable: ...
    def args(self) -> Tuple[Any, ...]: ...
    def is_nowait(self) -> _bool: ...

def _jit_set_num_profiled_runs(num: _size) -> _size: ...

# Defined in torch/csrc/jit/passes/mobile_optimizer_type.h
class _MobileOptimizerType: ...

CONV_BN_FUSION: _MobileOptimizerType
INSERT_FOLD_PREPACK_OPS: _MobileOptimizerType
REMOVE_DROPOUT: _MobileOptimizerType
FUSE_ADD_RELU: _MobileOptimizerType
HOIST_CONV_PACKED_PARAMS: _MobileOptimizerType
VULKAN_AUTOMATIC_GPU_TRANSFER: _MobileOptimizerType

def fork(*args: Any, **kwargs: Any) -> Future: ...
def wait(fut: Future) -> Any: ...
def _awaitable(*args: Any, **kwargs: Any) -> _Await: ...
def _awaitable_wait(aw: _Await) -> Any: ...
def _awaitable_nowait(x: Any) -> _Await: ...
def _collect_all(futures: List[Future]) -> Future: ...
def _set_print_stack_traces_on_fatal_signal(print: _bool) -> None: ...
def unify_type_list(types: List[JitType]) -> JitType: ...
def _freeze_module(
    module: ScriptModule,
    preserved_attrs: List[str] = [],
    freeze_interfaces: _bool = True,
    preserveParameters: _bool = True,
) -> ScriptModule: ...
def _jit_pass_optimize_frozen_graph(Graph, optimize_numerics: _bool = True) -> None: ...
def _jit_pass_optimize_for_inference(
    module: torch.jit.ScriptModule,
    other_methods: List[str] = [],
) -> None: ...
def _jit_pass_fold_frozen_conv_bn(graph: Graph): ...
def _jit_pass_fold_frozen_conv_add_or_sub(graph: Graph): ...
def _jit_pass_fold_frozen_conv_mul_or_div(graph: Graph): ...
def _jit_pass_fuse_frozen_conv_add_relu(graph: Graph): ...
def _jit_pass_concat_frozen_linear(graph: Graph): ...
def _jit_pass_convert_frozen_ops_to_mkldnn(graph: Graph): ...
def _jit_pass_transpose_frozen_linear(graph: Graph): ...
def _jit_pass_remove_dropout(module: torch.jit.ScriptModule): ...
def _is_tracing() -> _bool: ...
def _jit_init() -> _bool: ...
def _jit_flatten(arg: Any) -> Tuple[List[Tensor], IODescriptor]: ...
def _jit_unflatten(vars: List[Tensor], desc: IODescriptor) -> Any: ...
def _jit_get_operation(op_name: str) -> Tuple[Callable, List[str]]: ...
def _get_operation_overload(
    op_name: str,
    op_overload_name: str,
) -> Tuple[Callable, Callable, List[Any]]: ...
def _get_schema(op_name: str, overload_name: str) -> FunctionSchema: ...
def _jit_pass_optimize_for_mobile(
    module: torch.jit.ScriptModule,
    optimization_blocklist: Set[_MobileOptimizerType],
    preserved_methods: List[AnyStr],
) -> torch.jit.ScriptModule: ...
def _clone_module_with_class(
    module: torch.jit.ScriptModule,
    ignored_methods: List[AnyStr],
    ignored_attributes: List[AnyStr],
) -> torch.jit.ScriptModule: ...
def _jit_pass_vulkan_optimize_for_mobile(
    module: torch.jit.ScriptModule,
    optimization_blocklist: Set[_MobileOptimizerType],
    preserved_methods: List[AnyStr],
) -> torch.jit.ScriptModule: ...
def _jit_pass_metal_optimize_for_mobile(
    module: torch.jit.ScriptModule,
    preserved_methods: List[AnyStr],
) -> torch.jit.ScriptModule: ...
def _jit_pass_inline(Graph) -> None: ...
def _jit_pass_constant_propagation(Graph) -> None: ...
def _jit_pass_propagate_shapes_on_graph(Graph) -> None: ...
def _jit_register_decomposition_for_schema(schema: FunctionSchema, Graph) -> None: ...
def _jit_erase_non_input_shape_information(Graph) -> None: ...
def _jit_get_schemas_for_operator(name: str) -> List[FunctionSchema]: ...
def _jit_get_all_schemas() -> List[FunctionSchema]: ...
def _jit_check_alias_annotation(
    g: Graph,
    args: Tuple[Any, ...],
    unqualified_op_name: str,
): ...
def _jit_can_fuse_on_cpu() -> _bool: ...
def _jit_can_fuse_on_gpu() -> _bool: ...
def _jit_can_fuse_on_cpu_legacy() -> _bool: ...
def _debug_get_fusion_group_inlining() -> _bool: ...
def _debug_set_fusion_group_inlining(enable: _bool): ...
def _jit_texpr_fuser_enabled() -> _bool: ...
def _jit_nvfuser_enabled() -> _bool: ...
def _jit_llga_enabled() -> _bool: ...
def _jit_set_llga_enabled(enable: _bool): ...
def _llvm_enabled() -> _bool: ...
def _jit_override_can_fuse_on_cpu(override: _bool): ...
def _jit_override_can_fuse_on_gpu(override: _bool): ...
def _jit_override_can_fuse_on_cpu_legacy(override: _bool): ...
def _jit_set_symbolic_shapes_test_mode(override: _bool): ...
def _jit_symbolic_shapes_test_mode_enabled() -> _bool: ...
def _jit_set_texpr_fuser_enabled(enable: _bool): ...
def _jit_set_te_must_use_llvm_cpu(use_llvm: _bool): ...
def _jit_set_nvfuser_enabled(enable: _bool) -> _bool: ...
def _jit_cat_wo_conditionals(optimize_cat: _bool): ...
def _jit_opt_conditionals(opt_conds: _bool): ...
def _jit_pass_canonicalize(graph: Graph, keep_unique_names: _bool = True): ...
def _jit_pass_erase_shape_information(graph: Graph): ...
def _jit_pass_fold_convbn(module: torch.jit.ScriptModule): ...
def _jit_pass_insert_observers(
    module: torch.jit.ScriptModule,
    method_name: str,
    qconfig_dict: Dict[str, Any],
    inplace: _bool,
    quant_type: _int,
): ...
def _jit_pass_insert_quant_dequant(
    module: torch.jit.ScriptModule,
    method_name: str,
    inplace: _bool,
    debug: _bool,
    quant_type: _int,
): ...
def _jit_pass_insert_quant_dequant_for_ondevice_ptq(
    module: torch.jit.ScriptModule,
    method_name: str,
    inplace: _bool,
    debug: _bool,
    quant_type: _int,
): ...
def _jit_pass_quant_finalize(
    module: torch.jit.ScriptModule,
    quant_type: _int,
    preserved_attrs: Sequence[str],
): ...
def _jit_pass_quant_finalize_for_ondevice_ptq(
    module: torch.jit.ScriptModule,
    quant_type: _int,
    method_name: str,
): ...
def _jit_pass_insert_observer_method_for_ondevice_ptq(
    module: torch.jit.ScriptModule,
    method_name: str,
    qconfig_dict: Dict[str, Any],
    inplace: _bool,
    quant_type: _int,
): ...
def _jit_set_profiling_executor(profiling_flag: _bool) -> _bool: ...
def _jit_set_profiling_mode(profiling_flag: _bool) -> _bool: ...
def _jit_set_fusion_strategy(
    strategy: List[Tuple[str, _int]],
) -> List[Tuple[str, _int]]: ...
def _jit_try_infer_type(obj: Any) -> InferredType: ...
def _jit_get_trigger_value(trigger_name: str) -> _int: ...

# Defined in torch/csrc/jit/python/script_init.cpp
ResolutionCallback = Callable[[str], Callable[..., Any]]

# Defined in torch/csrc/jit/python/script_init.cpp
#        and torch/csrc/jit/python/init.cpp
def _maybe_call_torch_function_for_op_packet(
    op_overload_packet: Any,
    args: Any,
    kwargs: Any,
) -> Any: ...
def _check_schema_allow_fake_script_object(
    schema: FunctionSchema,
    args: Any,
    kwargs: Any,
) -> _bool: ...
def _create_function_from_graph(qualname: str, graph: Graph) -> ScriptFunction: ...
def _debug_set_autodiff_subgraph_inlining(disabled: _bool) -> None: ...
def _ivalue_tags_match(lhs: ScriptModule, rhs: ScriptModule) -> _bool: ...
def _jit_assert_is_instance(obj: Any, type: JitType): ...
def _jit_clear_class_registry() -> None: ...
def _jit_set_emit_hooks(
    ModuleHook: Optional[Callable],
    FunctionHook: Optional[Callable],
) -> None: ...
def _jit_get_emit_hooks() -> Tuple[Callable, Callable]: ...
def _load_for_lite_interpreter(
    filename: Union[str, Path],
    map_location: Optional[DeviceLikeType],
): ...
def _load_for_lite_interpreter_from_buffer(
    buffer: BinaryIO,
    map_location: Optional[DeviceLikeType],
): ...
def _export_operator_list(module: LiteScriptModule): ...
def _quantize_ondevice_ptq_dynamic(module: LiteScriptModule, method_name: str): ...
def _get_model_bytecode_version(filename: Union[str, Path]) -> _int: ...
def _get_model_bytecode_version_from_buffer(buffer: BinaryIO) -> _int: ...
def _backport_for_mobile(
    filename_input: Union[str, Path],
    filename_output: Union[str, Path],
    to_version: _int,
) -> None: ...
def _backport_for_mobile_from_buffer(
    buffer: BinaryIO,
    filename_output: Union[str, Path],
    to_version: _int,
) -> None: ...
def _backport_for_mobile_to_buffer(
    filename_input: Union[str, Path],
    to_version: _int,
) -> bytes: ...
def _backport_for_mobile_from_buffer_to_buffer(
    buffer: BinaryIO,
    to_version: _int,
) -> bytes: ...
def _get_model_ops_and_info(filename: Union[str, Path]): ...
def _get_model_ops_and_info_from_buffer(buffer: BinaryIO): ...
def _get_mobile_model_contained_types(filename: Union[str, Path]): ...
def _get_mobile_model_contained_types_from_buffer(buffer: BinaryIO): ...
def _logging_set_logger(logger: LoggerBase) -> LoggerBase: ...
def _get_graph_executor_optimize(optimize: Optional[_bool] = None) -> _bool: ...
def _set_graph_executor_optimize(optimize: _bool): ...
def _export_opnames(module: ScriptModule) -> List[str]: ...
def _create_function_from_trace(
    qualname: str,
    func: Callable[..., Any],
    input_tuple: Tuple[Any, ...],
    var_lookup_fn: Callable[[Tensor], str],
    strict: _bool,
    force_outplace: _bool,
    argument_names: List[str],
) -> Tuple[Graph, Stack]: ...
def _create_function_from_trace_with_dict(
    qualname: str,
    func: Callable[..., Any],
    input_dict: Dict[str, Any],
    var_lookup_fn: Callable[[Tensor], str],
    strict: _bool,
    force_outplace: _bool,
    argument_names: List[str],
) -> Tuple[Graph, Stack]: ...
def _jit_is_script_object(obj: Any) -> _bool: ...
def _last_executed_optimized_graph() -> Graph: ...
def parse_type_comment(comment: str) -> Decl: ...
def _get_upgraders_map_size() -> _int: ...
def _get_upgraders_entry_map() -> Dict[str, str]: ...
def _dump_upgraders_map() -> Dict[str, str]: ...
def _test_only_populate_upgraders(content: Dict[str, str]) -> None: ...
def _test_only_remove_upgraders(content: Dict[str, str]) -> None: ...
def merge_type_from_type_comment(
    decl: Decl,
    type_annotation_decl: Decl,
    is_method: _bool,
) -> Decl: ...
def parse_ir(input: str, parse_tensor_constants: _bool = False) -> Graph: ...
def parse_schema(schema: str) -> FunctionSchema: ...
def get_device(input: Tensor) -> _int: ...
def _resolve_type_from_object(
    obj: Any,
    range: SourceRange,
    rcb: ResolutionCallback,
) -> JitType: ...
def _create_module_with_type(ty: JitType) -> ScriptModule: ...
def _create_object_with_type(ty: ClassType) -> ScriptObject: ...
def _run_emit_module_hook(m: ScriptModule): ...
def _replace_overloaded_method_decl(
    overload_decl: Decl,
    implementation_def: Def,
    new_name: str,
) -> Def: ...
def _jit_pass_lower_all_tuples(graph: Graph) -> None: ...
def _jit_pass_onnx_set_dynamic_input_shape(
    graph: Graph,
    dynamic_axes: Dict[str, Dict[_int, str]],
    input_names: List[str],
) -> None: ...
def _jit_pass_onnx_graph_shape_type_inference(
    graph: Graph,
    params_dict: Dict[str, IValue],
    opset_version: _int,
) -> None: ...
def _jit_pass_onnx_assign_output_shape(
    graph: Graph,
    tensors: List[Tensor],
    desc: IODescriptor,
    onnx_shape_inference: _bool,
    is_script: _bool,
    opset_version: _int,
) -> None: ...
def _jit_pass_onnx_remove_inplace_ops_for_onnx(
    graph: Graph,
    module: Optional[ScriptModule] = None,
) -> None: ...
def _jit_pass_remove_inplace_ops(graph: Graph) -> None: ...
def _jit_pass_canonicalize_graph_fuser_ops(graph: Graph) -> None: ...
def _jit_pass_peephole(
    graph: Graph,
    disable_shape_peepholes: _bool = False,
) -> None: ...
def _jit_pass_onnx_autograd_function_process(graph: Graph) -> None: ...
def _jit_pass_fuse_addmm(graph: Graph) -> None: ...
def _jit_pass_onnx_preprocess(graph: Graph) -> None: ...
def _jit_pass_prepare_division_for_onnx(graph: Graph) -> None: ...
def _jit_pass_onnx_remove_print(graph: Graph) -> None: ...
def _jit_pass_onnx_preprocess_caffe2(graph: Graph) -> None: ...
def _jit_pass_onnx_unpack_quantized_weights(
    graph: Graph,
    paramsDict: Dict[str, IValue],
    caffe2: _bool,
) -> Dict[str, IValue]: ...
def _jit_pass_onnx_quantization_insert_permutes(
    graph: Graph,
    paramsDict: Dict[str, IValue],
) -> Dict[str, IValue]: ...
def _jit_pass_custom_pattern_based_rewrite_graph(
    pattern: str,
    fused_node_name: str,
    graph: Graph,
) -> None: ...
def _jit_onnx_list_model_parameters(
    module: ScriptModule,
) -> Tuple[ScriptModule, List[IValue]]: ...
def _jit_pass_erase_number_types(graph: Graph) -> None: ...
def _jit_pass_onnx_lint(graph: Graph) -> None: ...
def _jit_pass_onnx(
    graph: Graph,
    _jit_pass_onnx: _onnx.OperatorExportTypes,
) -> Graph: ...
def _jit_pass_onnx_scalar_type_analysis(
    graph: Graph,
    lowprecision_cast: _bool,
    opset_version: _int,
) -> None: ...
def _jit_pass_onnx_peephole(
    graph: Graph,
    opset_version: _int,
    fixed_batch_size: _bool,
) -> None: ...
def _jit_pass_dce_allow_deleting_nodes_with_side_effects(graph: Graph) -> None: ...
def _jit_pass_onnx_function_substitution(graph: Graph) -> None: ...
def _jit_pass_onnx_function_extraction(
    graph: Graph,
    module_names: Set[str],
    param_names: List[str],
) -> Dict[Node, Dict[str, str]]: ...
def _jit_pass_onnx_clear_scope_records() -> None: ...
def _jit_pass_onnx_track_scope_attributes(
    graph: Graph,
    onnx_attrs: Dict[str, Any],
) -> None: ...
def _jit_is_onnx_log_enabled() -> _bool: ...
def _jit_set_onnx_log_enabled(enabled: _bool) -> None: ...
def _jit_set_onnx_log_output_stream(stream_name: str) -> None: ...
def _jit_onnx_log(*args: Any) -> None: ...
def _jit_pass_lower_graph(graph: Graph, m: Module) -> Tuple[Graph, List[IValue]]: ...
def _jit_pass_inline_fork_wait(graph: Graph) -> None: ...
def _jit_pass_onnx_deduplicate_initializers(
    graph: Graph,
    params_dict: Dict[str, IValue],
    is_train: _bool,
) -> Dict[str, IValue]: ...
def _jit_pass_onnx_eval_peephole(
    graph: Graph,
    paramsDict: Dict[str, IValue],
) -> Dict[str, IValue]: ...
def _jit_pass_onnx_constant_fold(
    graph: Graph,
    paramsDict: Dict[str, IValue],
    opset_version: _int,
) -> Dict[str, IValue]: ...
def _jit_pass_onnx_eliminate_unused_items(
    graph: Graph,
    paramsDict: Dict[str, IValue],
) -> Dict[str, IValue]: ...
def _jit_pass_onnx_cast_all_constant_to_floating(graph: Graph) -> None: ...
def _jit_pass_filter_non_tensor_arguments(
    params: Dict[str, IValue],
) -> Dict[str, Tensor]: ...
def _jit_decay_packed_param_input_types(graph: Graph) -> None: ...
def _jit_pass_onnx_node_shape_type_inference(
    n: Node,
    paramsDict: Dict[str, IValue],
    opset_version: _int,
) -> None: ...
def _jit_onnx_convert_pattern_from_subblock(
    block: Block,
    n: Node,
    env: Dict[Value, Value],
) -> List[Value]: ...
def _jit_pass_onnx_block(
    old_block: Block,
    new_block: Block,
    operator_export_type: _onnx.OperatorExportTypes,
    env: Dict[Value, Value],
    is_sub_block: _bool,
) -> Dict[Value, Value]: ...
def _jit_pass_onnx_assign_scoped_names_for_node_and_value(graph: Graph) -> None: ...
def _jit_pass_fixup_onnx_controlflow_node(
    n: Node,
    opset_version: _int,
) -> List[Value]: ...
def _jit_onnx_create_full_scope_name(class_name: str, variable_name: str) -> str: ...
def _compile_graph_to_code_table(name: str, graph: Graph) -> IValue: ...
def _generate_upgraders_graph() -> Dict[str, Graph]: ...
def _calculate_package_version_based_on_upgraders(val: _bool): ...
def _get_version_calculator_flag() -> _bool: ...
def _jit_script_interface_compile(
    name: str,
    class_def: ClassDef,
    rcb: ResolutionCallback,
    is_module: _bool,
): ...
def _jit_script_compile_overload(
    qualname: str,
    overload_decl: Decl,
    implementation_def: Def,
    rcb: ResolutionCallback,
    implementation_defaults: Dict[str, Any],
    signature: Any,
): ...
def _jit_script_compile(
    qual_name: str,
    definition: Def,
    rcb: ResolutionCallback,
    defaults: Dict[str, Any],
): ...
def _jit_script_class_compile(
    qual_name: str,
    definition: ClassDef,
    defaults: Dict[str, Dict[str, Any]],
    rcb: ResolutionCallback,
): ...
def _parse_source_def(src: str) -> Def: ...
def import_ir_module(
    cu: CompilationUnit,
    filename: Union[str, Path],
    map_location: Optional[DeviceLikeType],
    extra_files: Dict[str, Any],
) -> ScriptModule: ...
def import_ir_module_from_buffer(
    cu: CompilationUnit,
    buffer: BinaryIO,
    map_location: Optional[DeviceLikeType],
    extra_files: Dict[str, Any],
) -> ScriptModule: ...
def _import_ir_module_from_package(
    cu: CompilationUnit,
    reader: PyTorchFileReader,
    storage_context: DeserializationStorageContext,
    map_location: Optional[DeviceLikeType],
    ts_id: str,
) -> ScriptModule: ...
def _assign_output_shapes(graph: Graph, inputs: List[Tensor]) -> Graph: ...
def _check_onnx_proto(proto: str) -> None: ...
def _propagate_and_assign_input_shapes(
    graph: Graph,
    inputs: Tuple[Tensor, ...],
    param_count_list: List[_int],
    with_grad: _bool,
    propagate: _bool,
) -> Graph: ...

# Defined in torch/csrc/jit/runtime/graph_executor.h
class GraphExecutorState: ...

# Defined in torch/torch/csrc/jit/ir/alias_analysis.h
class AliasDb:
    def __str__(self) -> str: ...

class _InsertPoint:
    def __enter__(self) -> None: ...
    def __exit__(self, *args) -> None: ...

# Defined in torch/csrc/jit/ir/ir.h
class Use:
    @property
    def user(self) -> Node: ...
    @property
    def offset(self) -> _int: ...
    def isAfter(self, other: Use) -> _bool: ...

# Defined in torch/csrc/jit/ir/ir.h
class Value:
    def type(self) -> JitType: ...
    def setType(self, t: JitType) -> Value: ...
    def setTypeAs(self, other: Value) -> Value: ...
    def inferTypeFrom(self, t: Tensor) -> None: ...
    def debugName(self) -> str: ...
    def setDebugName(self, name: str) -> None: ...
    def unique(self) -> _int: ...
    def offset(self) -> _int: ...
    def node(self) -> Node: ...
    def uses(self) -> List[Use]: ...
    def replaceAllUsesWith(self, val: Value) -> None: ...
    def replaceAllUsesAfterNodeWith(self, node: Node, val: Value) -> None: ...
    def requires_grad(self) -> _bool: ...
    def requiresGrad(self) -> _bool: ...
    def copyMetadata(self, other: Value) -> Value: ...
    def isCompleteTensor(self) -> _bool: ...
    def toIValue(self) -> IValue: ...

# Defined in torch/csrc/jit/ir/ir.h
class Block:
    def inputs(self) -> Iterator[Value]: ...
    def outputs(self) -> Iterator[Value]: ...
    def nodes(self) -> Iterator[Node]: ...
    def paramNode(self) -> Node: ...
    def returnNode(self) -> Node: ...
    def owningNode(self) -> Node: ...
    def registerOutput(self, n: Value) -> _int: ...
    def addNode(self, name: str, inputs: Sequence[Value]) -> Node: ...

# Defined in torch/csrc/jit/ir/ir.h
class Node:
    def __getitem__(self, key: str) -> Any: ...
    def schema(self) -> str: ...
    def input(self) -> Value: ...
    def inputs(self) -> Iterator[Value]: ...
    def inputsAt(self, idx: _int) -> Value: ...
    def inputsSize(self) -> _int: ...
    def output(self) -> Value: ...
    def outputs(self) -> Iterator[Value]: ...
    def outputsAt(self, idx: _int) -> Value: ...
    def outputsSize(self) -> _int: ...
    def hasMultipleOutputs(self) -> _bool: ...
    def blocks(self) -> List[Block]: ...
    def addBlock(self) -> Block: ...
    def mustBeNone(self) -> _bool: ...
    def matches(self, pattern: str) -> _bool: ...
    def kind(self) -> str: ...
    def kindOf(self, name: str) -> str: ...
    def addInput(self, name: str) -> Value: ...
    def replaceInput(self, i: _int, newValue: Value) -> Value: ...
    def replaceInputWith(self, from_: Value, to: Value) -> None: ...
    def replaceAllUsesWith(self, n: Node) -> None: ...
    def insertBefore(self, n: Node) -> Node: ...
    def insertAfter(self, n: Node) -> Node: ...
    def isBefore(self, n: Node) -> _bool: ...
    def isAfter(self, n: Node) -> _bool: ...
    def moveBefore(self, n: Node) -> None: ...
    def moveAfter(self, n: Node) -> None: ...
    def removeInput(self, i: _int) -> None: ...
    def removeAllInputs(self, i: _int) -> None: ...
    def hasUses(self) -> _bool: ...
    def eraseOutput(self, i: _int) -> None: ...
    def addOutput(self) -> Value: ...
    def scopeName(self) -> str: ...
    def isNondeterministic(self) -> _bool: ...
    def copyAttributes(self, rhs: Node) -> Node: ...
    def copyMetadata(self, rhs: Node) -> Node: ...
    def hasAttributes(self) -> _bool: ...
    def hasAttribute(self, name: str) -> _bool: ...
    def removeAttribute(self, attr: str) -> Node: ...
    def namedInput(self, name: str) -> Value: ...
    def sourceRange(self) -> SourceRange: ...
    def owningBlock(self) -> Block: ...
    def findNode(self, kind: str, recurse: _bool = True) -> Node: ...
    def findAllNodes(self, kind: str, recurse: _bool = True) -> List[Node]: ...
    def getModuleHierarchy(self) -> str: ...
    def prev(self) -> Node: ...
    def destroy(self) -> None: ...
    def attributeNames(self) -> List[str]: ...

    # Accessors for attributes as types.
    def f(self, name: str) -> _float: ...
    def f_(self, name: str, val: _float) -> Node: ...
    def fs(self, name: str) -> List[_float]: ...
    def fs_(self, name: str, val: List[_float]) -> Node: ...
    def c(self, name: str) -> complex: ...
    def c_(self, name: str, val: complex) -> Node: ...
    def s(self, name: str) -> str: ...
    def s_(self, name: str, val: str) -> Node: ...
    def ss(self, name: str) -> List[str]: ...
    def ss_(self, name: str, val: List[str]) -> Node: ...
    def i(self, name: str) -> _int: ...
    def i_(self, name: str, val: _int) -> Node: ...
    # Cannot define "is" like this because it's a reserved keyword in python.
    # def is(self, name: str) -> List[_int]: ...
    # def is_(self, name: str, val: List[_int]) -> Node: ...
    def g(self, name: str) -> Graph: ...
    def g_(self, name: str, val: Graph) -> Node: ...
    def gs(self, name: str) -> List[Graph]: ...
    def gs_(self, name: str, val: List[Graph]) -> Node: ...
    def ival(self, name: str) -> IValue: ...
    def ival_(self, name: str, val: IValue) -> Node: ...
    def t(self, name: str) -> Tensor: ...
    def t_(self, name: str, val: Tensor) -> Node: ...
    def ts(self, name: str) -> List[Tensor]: ...
    def ts_(self, name: str, val: List[Tensor]) -> Node: ...
    def ty(self, name: str) -> JitType: ...
    def ty_(self, name: str, val: JitType) -> Node: ...
    def tys(self, name: str) -> List[JitType]: ...
    def tys_(self, name: str, val: List[JitType]) -> Node: ...

# Defined in torch/torch/csrc/jit/ir/ir.h
class Graph:
    def inputs(self) -> Iterator[Value]: ...
    def outputs(self) -> Iterator[Value]: ...
    def nodes(self) -> Iterator[Node]: ...
    def param_node(self) -> Node: ...
    def return_node(self) -> Node: ...
    def addInput(self, name: str = "") -> Value: ...
    def eraseInput(self, i: _int) -> None: ...
    def registerOutput(self, n: Value) -> _int: ...
    def eraseOutput(self, i: _int) -> None: ...
    def create(self, name: str, args, num_outputs: _int) -> Node: ...
    def appendNode(self, n: Node) -> Node: ...
    def prependNode(self, n: Node) -> Node: ...
    def insertNode(self, n: Node) -> Node: ...
    def block(self) -> Block: ...
    def lint(self) -> None: ...
    def alias_db(self) -> AliasDb: ...
    def setInsertPoint(self, n: Union[Block, Node]) -> None: ...
    def insert_point_guard(self, n: Union[Block, Node]) -> _InsertPoint: ...
    def insertPoint(self) -> Node: ...
    def insertGraph(self, callee: Graph, inputs: List[Value]) -> List[Value]: ...
    def makeMultiOutputIntoTuple(self) -> None: ...
    def copy(self) -> Graph: ...

# Defined in torch/aten/src/ATen/core/alias_info.h
class AliasInfo:
    is_write: _bool
    before_set: Set[str]
    after_set: Set[str]

# Defined in torch/aten/src/ATen/core/function_schema.h
class Argument:
    name: str
    type: JitType
    default_value: Optional[Any]
    def has_default_value(self) -> _bool: ...
    kwarg_only: _bool
    is_out: _bool
    alias_info: Optional[AliasInfo]

class FunctionSchema:
    arguments: List[Argument]
    returns: List[Argument]
    name: str
    overload_name: str
    is_mutable: _bool

class _UpgraderEntry:
    bumped_at_version: _int
    upgrader_name: str
    old_schema: str
    def __init__(
        self,
        bumped_at_version: _int,
        upgrader_name: str,
        old_schema: str,
    ) -> None: ...

class _UpgraderRange:
    min_version: _int
    max_version: _int

def _get_max_operator_version() -> _int: ...
def _get_operator_version_map() -> Dict[str, List[_UpgraderEntry]]: ...
def _get_upgrader_ranges(name: str) -> List[_UpgraderRange]: ...
def _test_only_add_entry_to_op_version(op_name: str, entry: _UpgraderEntry) -> None: ...
def _test_only_remove_entry_to_op_version(op_name: str) -> None: ...

# Defined in torch/csrc/jit/python/script_init.cpp
class ScriptModuleSerializer:
    def __init__(self, export_writer: PyTorchFileWriter) -> None: ...
    def serialize(self, model: ScriptModule, script_module_id: _int) -> None: ...
    def write_files(self) -> None: ...
    def storage_context(self) -> SerializationStorageContext: ...

# Defined in torch/csrc/jit/python/script_init.cpp
class SerializationStorageContext:
    def __init__(self) -> None: ...
    def has_storage(self, storage: Storage) -> _bool: ...
    def get_or_add_storage(self, storage: Storage) -> _int: ...

# Defined in torch/csrc/jit/python/script_init.cpp
class DeserializationStorageContext:
    def __init__(self) -> None: ...
    def get_storage(self, name: str, dtype: _dtype) -> Tensor: ...
    def has_storage(self, name: str) -> _bool: ...
    def add_storage(self, name: str, tensor: Tensor) -> _int: ...

# Defined in torch/csrc/jit/python/script_init.cpp
class ConcreteModuleTypeBuilder:
    def __init__(self, obj: Any) -> None: ...
    def set_module_dict(self): ...
    def set_module_list(self): ...
    def set_parameter_list(self): ...
    def set_parameter_dict(self): ...
    def add_attribute(
        self,
        name: str,
        ty: JitType,
        is_param: _bool,
        is_buffer: _bool,
    ): ...
    def add_module(self, name: str, meta: ConcreteModuleType): ...
    def add_constant(self, name: str, value: Any): ...
    def add_overload(self, method_name: str, overloaded_method_names: List[str]): ...
    def add_builtin_function(self, name: str, symbol_name: str): ...
    def add_failed_attribute(self, name: str, failure_reason: str): ...
    def add_function_attribute(
        self,
        name: str,
        ty: JitType,
        func: Callable[..., Any],
    ): ...
    def add_ignored_attribute(self, name: str): ...
    def add_ignored_attributes(self, names: List[str]): ...
    def add_forward_hook(self, hook: Callable[..., Any]): ...
    def add_forward_pre_hook(self, pre_hook: Callable[..., Any]): ...

class ConcreteModuleType:
    def get_constants(self) -> Dict[str, Any]: ...
    def equals(self, other: ConcreteModuleType) -> _bool: ...
    @staticmethod
    def from_jit_type(ty: JitType) -> ConcreteModuleType: ...

class CallStack:
    def __init__(self, name: str, range: SourceRange): ...

class ErrorReport:
    def __init__(self, range: SourceRange) -> None: ...
    def what(self) -> str: ...
    @staticmethod
    def call_stack() -> str: ...

class CompilationUnit:
    def __init__(self, lang: str = ..., _frames_up: _int = ...) -> None: ...
    def find_function(self, name: str) -> ScriptFunction: ...
    def __getattr__(self, name: str) -> ScriptFunction: ...
    def define(
        self,
        script: str,
        rcb: ResolutionCallback = ...,
        _frames_up: _int = ...,
    ): ...
    def get_interface(self, name: str) -> InterfaceType: ...
    def get_functions(self) -> List[ScriptFunction]: ...
    def create_function(
        self,
        name: str,
        graph: Graph,
        shouldMangle: _bool = ...,
    ) -> ScriptFunction: ...
    def get_class(self, name: str) -> ClassType: ...

class ScriptObject:
    def setattr(self, name: str, value: Any): ...

class ScriptModule(ScriptObject):
    def _method_names(self) -> List[str]: ...
    def _get_method(self, name: str) -> ScriptMethod: ...

class LiteScriptModule:
    def __call__(self, *input): ...
    def find_method(self, method_name: str): ...
    def forward(self, *input) -> List[str]: ...
    def run_method(self, method_name: str, *input): ...

# NOTE: switch to collections.abc.Callable in python 3.9
class ScriptFunction(Generic[P, ReturnVal]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> ReturnVal: ...
    def save(self, filename: str, _extra_files: Dict[str, bytes]) -> None: ...
    def save_to_buffer(self, _extra_files: Dict[str, bytes]) -> bytes: ...
    @property
    def graph(self) -> Graph: ...
    def inlined_graph(self) -> Graph: ...
    def schema(self) -> FunctionSchema: ...
    def code(self) -> str: ...
    def name(self) -> str: ...
    @property
    def qualified_name(self) -> str: ...

# NOTE: switch to collections.abc.Callable in python 3.9
class ScriptMethod(Generic[P, ReturnVal]):
    graph: Graph
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> ReturnVal: ...
    @property
    def owner(self) -> ScriptModule: ...
    @property
    def name(self) -> str: ...

class ScriptDict(Generic[K, T]):
    def __init__(self, dict: Dict[K, T]) -> None: ...
    def __len__(self) -> _int: ...
    def __contains__(self, key: K) -> _bool: ...
    def __getitem__(self, key: K) -> T: ...
    def __setitem__(self, key: K, value: T) -> None: ...
    def __delitem__(self, key: K) -> None: ...
    def __iter__(self) -> Iterator[K]: ...
    def items(self) -> Iterator[tuple[K, T]]: ...
    def keys(self) -> Iterator[K]: ...

class ScriptList(Generic[T]):
    def __init__(self, list: List[T]) -> None: ...
    def __len__(self) -> _int: ...
    def __contains__(self, item: T) -> _bool: ...
    @overload
    def __getitem__(self, idx: _int) -> T: ...
    @overload
    def __getitem__(self, idx: slice) -> ScriptList[T]: ...
    @overload
    def __setitem__(self, idx: _int, value: T) -> None: ...
    @overload
    def __setitem__(self, idx: slice, value: List[T]) -> None: ...
    def __delitem__(self, idx: _int) -> None: ...
    def __iter__(self) -> Iterator[T]: ...
    def count(self, value: T) -> _int: ...
    def remove(self, value: T) -> None: ...
    def append(self, value: T) -> None: ...
    def clear(self) -> None: ...
    @overload
    def extend(self, values: List[T]) -> None: ...
    @overload
    def extend(self, values: Iterable[T]) -> None: ...
    @overload
    def pop(self) -> T: ...
    @overload
    def pop(self, idx: _int) -> T: ...

class ModuleDict:
    def __init__(self, mod: ScriptModule) -> None: ...
    def items(self) -> List[Tuple[str, Any]]: ...

class ParameterDict:
    def __init__(self, mod: ScriptModule) -> None: ...

class BufferDict:
    def __init__(self, mod: ScriptModule) -> None: ...

# Defined in torch/csrc/jit/api/module.h
class Module: ...

# Defined in torch/csrc/Module.cpp
def _initExtension(shm_manager_path: str) -> None: ...  # THPModule_initExtension
def _autograd_init() -> _bool: ...  # THPAutograd_initExtension
def _add_docstr(obj: T, doc_obj: str) -> T: ...  # THPModule_addDocStr
def _init_names(arg: Sequence[Type]) -> None: ...  # THPModule_initNames
def _has_distributed() -> _bool: ...  # THPModule_hasDistributed
def _set_default_tensor_type(type) -> None: ...  # THPModule_setDefaultTensorType
def _set_default_dtype(d: _dtype) -> None: ...  # THPModule_setDefaultDtype
def _infer_size(arg1: Size, arg2: Size) -> Size: ...  # THPModule_inferSize
def _crash_if_csrc_asan() -> _int: ...  # THPModule_crashIfCsrcASAN
def _crash_if_csrc_ubsan() -> _int: ...  # THPModule_crashIfCsrcUBSAN
def _crash_if_aten_asan() -> _int: ...  # THPModule_crashIfATenASAN
def _show_config() -> str: ...  # THPModule_showConfig
def _cxx_flags() -> str: ...  # THPModule_cxxFlags
def _parallel_info() -> str: ...  # THPModule_parallelInfo
def _get_cpu_capability() -> str: ...  # THPModule_getCpuCapability
def _set_backcompat_broadcast_warn(
    arg: _bool,
) -> None: ...  # THPModule_setBackcompatBroadcastWarn
def _get_backcompat_broadcast_warn() -> _bool: ...  # THPModule_getBackcompatBroadcastWarn
def _set_backcompat_keepdim_warn(
    arg: _bool,
) -> None: ...  # THPModule_setBackcompatKeepdimWarn
def _get_backcompat_keepdim_warn() -> _bool: ...  # THPModule_getBackcompatKeepdimWarn
def get_num_thread() -> _int: ...  # THPModule_getNumThreads
def set_num_threads(nthreads: _int) -> None: ...  # THPModule_setNumThreads
def get_num_interop_threads() -> _int: ...  # THPModule_getNumInteropThreads
def set_num_interop_threads(
    nthreads: _int,
) -> None: ...  # THPModule_setNumInteropThreads
def _get_cudnn_enabled() -> _bool: ...  # THPModule_userEnabledCuDNN
def _set_cudnn_enabled(arg: _bool) -> None: ...  # THPModule_setUserEnabledCuDNN
def _get_flash_sdp_enabled() -> _bool: ...  # THPModule_userEnabledFusedSDP
def _set_sdp_use_flash(arg: _bool) -> None: ...  # THPModule_setSDPUseFlash
def _get_mem_efficient_sdp_enabled() -> _bool: ...  # THPModule_userEnabledMathSDP
def _set_sdp_use_mem_efficient(
    arg: _bool,
) -> None: ...  # THPModule_setSDPUseMemEfficient
def _get_math_sdp_enabled() -> _bool: ...  # THPModule_userEnabledMathSDP
def _set_sdp_use_math(arg: _bool) -> None: ...  # THPModule_setSDPUseMath
def _get_cudnn_sdp_enabled() -> _bool: ...  # THPModule_userEnabledMathSDP
def _set_sdp_use_cudnn(arg: _bool) -> None: ...  # THPModule_setSDPUseMath
def _get_mkldnn_enabled() -> _bool: ...  # THPModule_userEnabledMkldnn
def _set_mkldnn_enabled(arg: _bool) -> None: ...  # THPModule_setUserEnabledMkldnn
def _get_cudnn_benchmark() -> _bool: ...  # THPModule_benchmarkCuDNN
def _set_cudnn_benchmark(arg: _bool) -> None: ...  # THPModule_setBenchmarkCuDNN
def _get_cudnn_deterministic() -> _bool: ...  # THPModule_deterministicCuDNN
def _set_cudnn_deterministic(arg: _bool) -> None: ...  # THPModule_setDeterministicCuDNN
def _get_deterministic_algorithms() -> _bool: ...  # THPModule_deterministicAlgorithms
def _get_deterministic_algorithms_warn_only() -> _bool: ...  # THPModule_deterministicAlgorithmsWarnOnly
def _set_deterministic_algorithms(
    mode: _bool,
    *,
    warn_only: _bool = ...,
) -> None: ...  # THPModule_setDeterministicAlgorithms
def _get_deterministic_fill_uninitialized_memory() -> _bool: ...  # THPModule_deterministicFillUninitializedMemory
def _set_deterministic_fill_uninitialized_memory(arg: _bool) -> None: ...  # THPModule_setDeterministicFillUninitializedMemory
def _get_nnpack_enabled() -> _bool: ...  # THPModule_userEnabledNNPACK
def _set_nnpack_enabled(arg: _bool) -> None: ...  # THPModule_setUserEnabledNNPACK
def _get_warnAlways() -> _bool: ...  # THPModule_warnAlways
def _set_warnAlways(arg: _bool) -> None: ...  # THPModule_setWarnAlways
def _get_cudnn_allow_tf32() -> _bool: ...  # THPModule_allowTF32CuDNN
def _set_cudnn_allow_tf32(arg: _bool) -> None: ...  # THPModule_setAllowTF32CuDNN
def _get_cublas_allow_tf32() -> _bool: ...  # THPModule_allowTF32CuBLAS
def _set_cublas_allow_tf32(arg: _bool) -> None: ...  # THPModule_setAllowTF32CuBLAS
def _get_float32_matmul_precision() -> str: ...  # THPModule_float32MatmulPrecision
def _set_float32_matmul_precision(
    arg: str,
) -> None: ...  # THPModule_setFloat32MatmulPrecision
def _get_cublas_allow_fp16_reduced_precision_reduction() -> _bool: ...  # THPModule_allowFP16ReductionCuBLAS
def _set_cublas_allow_fp16_reduced_precision_reduction(
    arg: _bool,
) -> None: ...  # THPModule_setAllowFP16ReductionCuBLAS
def _get_cublas_allow_bf16_reduced_precision_reduction() -> _bool: ...  # THPModule_allowBF16ReductionCuBLAS
def _set_cublas_allow_bf16_reduced_precision_reduction(
    arg: _bool,
) -> None: ...  # THPModule_setAllowBF16ReductionCuBLAS
def _set_conj(x: Tensor, conj: _bool) -> None: ...
def _set_neg(x: Tensor, neg: _bool) -> None: ...
def _set_meta_in_tls_dispatch_include(meta_in_tls: _bool) -> None: ...
def _meta_in_tls_dispatch_include() -> _bool: ...
def _stash_obj_in_tls(key: str, arg: Any) -> None: ...
def _get_obj_in_tls(key: str) -> Any: ...
def _is_key_in_tls(key: str) -> _bool: ...
def _select_batch_norm_backend(*args, **kwargs) -> BatchNormBackend: ...
def _select_conv_backend(*args, **kwargs) -> ConvBackend: ...
def _conv_determine_backend_memory_format(
    input: Tensor,
    weight: Tensor,
    backend: ConvBackend,
) -> memory_format: ...
def _has_storage(x: Tensor) -> _bool: ...
def _construct_storage_from_data_pointer(data_ptr: _int, device: torch.device, size: _int) -> Storage: ...
def _should_allow_numbers_as_tensors(func_name: str) -> _bool: ...
def _group_tensors_by_device_and_dtype(nested_tensorlists: List[List[Optional[Tensor]]], with_indices: _bool = False) -> Dict[Tuple[torch.device, str], Tuple[List[List[Optional[Tensor]]], List[_int]]]: ...
def _check_tp_alloc_is_default(cls: Type) -> bool: ...

# NB: There is no Capsule type in typing, see
# https://code.activestate.com/lists/python-dev/139675/
def _to_dlpack(data: Tensor) -> Any: ...  # THPModule_toDLPack
def _from_dlpack(data: Any) -> Tensor: ...  # THPModule_fromDLPack
def _get_cpp_backtrace(
    frames_to_skip: _int,
    maximum_number_of_frames: _int,
) -> str: ...  # THPModule_getCppBacktrace
def set_flush_denormal(arg: _bool) -> _bool: ...  # THPModule_setFlushDenormal
def get_default_dtype() -> _dtype: ...  # THPModule_getDefaultDtype
def _get_default_device() -> str: ...  # THPModule_getDefaultDevice
def _get_qengine() -> _int: ...  # THPModule_qEngine
def _set_qengine(qengine: _int) -> None: ...  # THPModule_setQEngine
def _supported_qengines() -> List[_int]: ...  # THPModule_supportedQEngines
def _is_xnnpack_enabled() -> _bool: ...  # THPModule_isEnabledXNNPACK
def _check_sparse_tensor_invariants() -> _bool: ...  # THPModule_checkSparseTensorInvariants
def _set_check_sparse_tensor_invariants(
    arg: _bool,
) -> None: ...  # THPModule_setCheckSparseTensorInvariants
def _set_default_mobile_cpu_allocator() -> None: ...  # THPModule_setDefaultMobileCPUAllocator
def _unset_default_mobile_cpu_allocator() -> None: ...  # THPModule_unsetDefaultMobileCPUAllocator
def _is_torch_function_enabled() -> _bool: ...  # THPModule_isEnabledTorchFunction
def _has_torch_function(
    args: Iterable[Any],
) -> _bool: ...  # THPModule_has_torch_function
def _has_torch_function_unary(Any) -> _bool: ...  # THPModule_has_torch_function_unary
def _has_torch_function_variadic(
    *args: Any,
) -> _bool: ...  # THPModule_has_torch_function_variadic
def _vmapmode_increment_nesting() -> _int: ...  # THPModule_vmapmode_increment_nesting
def _vmapmode_decrement_nesting() -> _int: ...  # THPModule_vmapmode_decrement_nesting
def _log_api_usage_once(str) -> None: ...  # LogAPIUsageOnceFromPython
def _log_api_usage_metadata(event: str, metadata_map: Dict[str, str]) -> None: ...  # LogAPIUsageMetadataFromPython
def _demangle(str) -> str: ...  # c10::demangle
def _disabled_torch_function_impl(
    func: Callable,
    types: Iterable[Type],
    args: Tuple,
    kwargs: Dict,
) -> Any: ...  # THPModule_disable_torch_function
def _disabled_torch_dispatch_impl(
    func: Callable,
    types: Iterable[Type],
    args: Tuple,
    kwargs: Dict,
) -> Any: ...  # THPModule_disable_dispatch_function
def _get_linalg_preferred_backend() -> torch._C._LinalgBackend: ...
def _set_linalg_preferred_backend(arg: torch._C._LinalgBackend): ...

class _LinalgBackend:
    Default: _LinalgBackend
    Cusolver: _LinalgBackend
    Magma: _LinalgBackend

class BatchNormBackend(Enum): ...

def _get_blas_preferred_backend() -> torch._C._BlasBackend: ...
def _set_blas_preferred_backend(arg: torch._C._BlasBackend): ...

class _BlasBackend:
    Cublas: _BlasBackend
    Cublaslt: _BlasBackend

class ConvBackend(Enum): ...

class Tag(Enum):
    core: _int = 0
    data_dependent_output: _int = 1
    dynamic_output_shape: _int = 2
    generated: _int = 3
    inplace_view: _int = 4
    needs_fixed_stride_order: _int = 5
    nondeterministic_bitwise: _int = 6
    nondeterministic_seeded: _int = 7
    pointwise: _int = 8
    pt2_compliant_tag: _int = 9
    view_copy: _int = 10

# Defined in `valgrind.h` and `callgrind.h` respectively.
def _valgrind_supported_platform() -> _bool: ...  # NVALGRIND
def _valgrind_toggle() -> None: ...  # CALLGRIND_TOGGLE_COLLECT
def _valgrind_toggle_and_dump_stats() -> None: ...  # CALLGRIND_TOGGLE_COLLECT and CALLGRIND_DUMP_STATS

has_openmp: _bool
has_mkl: _bool
_has_mps: _bool
has_lapack: _bool
_has_cuda: _bool
_has_magma: _bool
_has_xpu: _bool
_has_mkldnn: _bool
_has_cudnn: _bool
has_spectral: _bool
_GLIBCXX_USE_CXX11_ABI: _bool
default_generator: Generator

# Defined in torch/csrc/autograd/init.cpp
def _set_grad_enabled(enabled: _bool) -> None: ...
def is_grad_enabled() -> _bool: ...
def _set_fwd_grad_enabled(enabled: _bool) -> None: ...
def _is_fwd_grad_enabled() -> _bool: ...
def is_inference_mode_enabled() -> _bool: ...
def set_autocast_enabled(enabled: _bool) -> None: ...
def is_autocast_enabled() -> _bool: ...
def clear_autocast_cache() -> None: ...
def set_autocast_cpu_enabled(enabled: _bool) -> None: ...
def is_autocast_cpu_enabled() -> _bool: ...
def _is_any_autocast_enabled() -> _bool: ...
def set_autocast_cpu_dtype(dtype: _dtype) -> None: ...
def set_autocast_gpu_dtype(dtype: _dtype) -> None: ...
def get_autocast_cpu_dtype() -> _dtype: ...
def get_autocast_gpu_dtype() -> _dtype: ...
def autocast_increment_nesting() -> _int: ...
def autocast_decrement_nesting() -> _int: ...
def is_autocast_cache_enabled() -> _bool: ...
def set_autocast_cache_enabled(enabled: _bool) -> None: ...
def _increment_version(tensor: Tensor) -> None: ...
def set_anomaly_enabled(enabled: _bool, check_nan: _bool = True) -> None: ...
def is_anomaly_enabled() -> _bool: ...
def is_anomaly_check_nan_enabled() -> _bool: ...
def _is_multithreading_enabled() -> _bool: ...
def _set_multithreading_enabled(enabled: _bool) -> None: ...
def _set_view_replay_enabled(enabled: _bool) -> None: ...
def _is_view_replay_enabled() -> _bool: ...
def _enter_dual_level() -> _int: ...
def _exit_dual_level(level: _int) -> None: ...
def _make_dual(tensor: Tensor, tangent: Tensor, level: _int) -> Tensor: ...
def _unpack_dual(tensor: Tensor, level: _int) -> Tensor: ...
def __set_forward_AD_enabled(enabled: _bool) -> None: ...
def __is_forward_AD_enabled() -> _bool: ...
def _register_default_hooks(pack_hook: Callable, unpack_hook: Callable) -> None: ...
def _reset_default_hooks() -> None: ...
def _is_torch_function_mode_enabled() -> _bool: ...
def _set_torch_function_mode(cls: Any) -> None: ...
def _push_on_torch_function_stack(cls: Any) -> None: ...
def _pop_torch_function_stack() -> Any: ...
def _get_function_stack_at(idx: _int) -> Any: ...
def _len_torch_function_stack() -> _int: ...
def _set_torch_dispatch_mode(cls: Any) -> None: ...
def _push_on_torch_dispatch_stack(cls: Any) -> None: ...
def _pop_torch_dispatch_stack(mode_key: Optional[torch._C._TorchDispatchModeKey] = None) -> Any: ...
def _get_dispatch_mode(mode_key: Optional[torch._C._TorchDispatchModeKey]) -> Any: ...
def _unset_dispatch_mode(mode: torch._C._TorchDispatchModeKey) -> Any: ...
def _set_dispatch_mode(mode: Any) -> None: ...
def _get_dispatch_stack_at(idx: _int) -> Any: ...
def _len_torch_dispatch_stack() -> _int: ...
def _activate_gpu_trace() -> None: ...

class _DisableTorchDispatch:
    def __init__(self): ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_value, traceback): ...

class _EnableTorchFunction:
    def __init__(self): ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_value, traceback): ...

class _EnablePythonDispatcher:
    def __init__(self): ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_value, traceback): ...

class _DisablePythonDispatcher:
    def __init__(self): ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_value, traceback): ...

class _EnablePreDispatch:
    def __init__(self): ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_value, traceback): ...

class _DisableFuncTorch:
    def __init__(self): ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_value, traceback): ...

class _DisableAutocast:
    def __init__(self): ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_value, traceback): ...

class _InferenceMode:
    def __init__(self, enabled: _bool): ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_value, traceback): ...

def _set_autograd_fallback_mode(mode: str) -> None: ...
def _get_autograd_fallback_mode() -> str: ...

# Defined in torch/csrc/jit/python/script_init.cpp
class LoggerBase: ...
class NoopLogger(LoggerBase): ...
class LockingLogger(LoggerBase): ...

class AggregationType(Enum):
    SUM = 0
    AVG = 1

class FileCheck:
    def run(self, test_string: str) -> None: ...
    def check(self, test_string: str) -> FileCheck: ...
    def check_not(self, test_string: str) -> FileCheck: ...
    def check_same(self, test_string: str) -> FileCheck: ...
    def check_next(self, test_string: str) -> FileCheck: ...
    def check_count(
        self,
        test_string: str,
        count: _int,
        exactly: _bool = False,
    ) -> FileCheck: ...
    def check_dag(self, test_string: str) -> FileCheck: ...
    def check_source_highlighted(self, test_string: str) -> FileCheck: ...
    def check_regex(self, test_string: str) -> FileCheck: ...

# Defined in torch/csrc/jit/python/init.cpp
class PyTorchFileReader:
    @overload
    def __init__(self, name: str) -> None: ...
    @overload
    def __init__(self, buffer: BinaryIO) -> None: ...
    def get_record(self, name: str) -> bytes: ...
    def serialization_id(self) -> str: ...

class PyTorchFileWriter:
    @overload
    def __init__(self, name: str) -> None: ...
    @overload
    def __init__(self, buffer: BinaryIO) -> None: ...
    def write_record(self, name: str, data: Union[Storage, bytes, _int], size: _int) -> None: ...
    def write_end_of_file(self) -> None: ...
    def set_min_version(self, version: _int) -> None: ...
    def get_all_written_records(self) -> List[str]: ...
    def archive_name(self) -> str: ...
    def serialization_id(self) -> str: ...

def _jit_get_inline_everything_mode() -> _bool: ...
def _jit_set_inline_everything_mode(enabled: _bool) -> None: ...
def _jit_get_logging_option() -> str: ...
def _jit_set_logging_option(option: str) -> None: ...
def _jit_set_logging_stream(stream_name: str) -> None: ...
def _jit_pass_cse(Graph) -> _bool: ...
def _jit_pass_dce(Graph) -> None: ...
def _jit_pass_lint(Graph) -> None: ...

# Defined in torch/csrc/jit/python/python_custom_class.cpp
def _get_custom_class_python_wrapper(name: str, attr: str) -> Any: ...

# Defined in torch/csrc/Module.cpp
def _rename_privateuse1_backend(backend: str) -> None: ...
def _get_privateuse1_backend_name() -> str: ...

# Defined in torch/csrc/Generator.cpp
class Generator:
    device: _device
    def __init__(self, device: Optional[DeviceLikeType] = None) -> None: ...
    def get_state(self) -> Tensor: ...
    def set_state(self, _new_state: Tensor) -> Generator: ...
    def clone_state(self) -> Generator: ...
    def graphsafe_get_state(self) -> Generator: ...
    def graphsafe_set_state(self, _new_state: Generator) -> Generator: ...
    def set_offset(self, offset: _int) -> Generator: ...
    def get_offset(self) -> _int: ...
    def manual_seed(self, seed: _int) -> Generator: ...
    def seed(self) -> _int: ...
    def initial_seed(self) -> _int: ...

# Defined in torch/csrc/utils/python_dispatch.cpp

class _DispatchOperatorHandle:
    def schema(self) -> FunctionSchema: ...
    def debug(self) -> str: ...

class _DispatchModule:
    def def_(self, schema: str, alias: str = "") -> _DispatchModule: ...
    def def_legacy(self, schema: str) -> _DispatchModule: ...
    def def_name_t_t(
        self,
        name: str,
        dispatch: str,
        debug: str = "default_def_name_t_t",
    ) -> _DispatchModule: ...
    def def_schema_t_t(
        self,
        schema: str,
        dispatch: str,
        alias: str,
        debug: str = "default_def_schema_t_t",
    ) -> _DispatchModule: ...
    def impl_t_t(
        self,
        name: str,
        dispatch: str,
        debug: str = "impl_t_t",
    ) -> _DispatchModule: ...
    def impl(self, name: str, dispatch: str, func: Callable) -> _DispatchModule: ...
    def define(self, schema: str, alias: str = "") -> _DispatchModule: ...
    def fallback_fallthrough(self, dispatch: str = "") -> _DispatchModule: ...

_after_ADInplaceOrView_keyset: DispatchKeySet
_after_autograd_keyset: DispatchKeySet

def _dispatch_library(
    kind: str,
    name: str,
    dispatch: str,
    file: str = "",
    linenum: Any = 0,
) -> _DispatchModule: ...
def _dispatch_dump(name: str) -> str: ...
def _dispatch_dump_table(name: str) -> str: ...
def _dispatch_check_invariants(name: str) -> None: ...
def _dispatch_check_all_invariants() -> None: ...
def _dispatch_call_boxed(handle: _DispatchOperatorHandle, *args, **kwargs) -> Any: ...
def _dispatch_find_schema_or_throw(name: str, overload_name: str) -> _DispatchOperatorHandle: ...
def _dispatch_set_report_error_callback(handle: _DispatchOperatorHandle, callback: Callable) -> None: ...
def _dispatch_has_kernel(name: str) -> _bool: ...
def _dispatch_has_kernel_for_dispatch_key(
    name: str,
    dispatch: _dispatchkey,
) -> _bool: ...
def _dispatch_has_kernel_for_any_dispatch_key(
    name: str,
    dispatch_key_set: DispatchKeySet,
) -> _bool: ...
def _dispatch_kernel_for_dispatch_key_is_fallthrough(
    name: str,
    dispatch: _dispatchkey,
) -> _bool: ...
def _dispatch_has_computed_kernel_for_dispatch_key(
    name: str,
    dispatch: _dispatchkey,
) -> _bool: ...
def _dispatch_find_dangling_impls() -> List[str]: ...
def _dispatch_get_all_op_names() -> List[str]: ...
def _dispatch_tls_set_dispatch_key_excluded(
    dispatch: _dispatchkey,
    val: _bool,
) -> None: ...
def _dispatch_tls_is_dispatch_key_excluded(dispatch: _dispatchkey) -> _bool: ...
def _dispatch_tls_set_dispatch_key_included(
    dispatch: _dispatchkey,
    val: _bool,
) -> None: ...
def _dispatch_tls_is_dispatch_key_included(dispatch: _dispatchkey) -> _bool: ...
def _dispatch_isTensorSubclassLike(tensor: Tensor) -> _bool: ...
def _dispatch_key_name(dispatch: _dispatchkey) -> str: ...
def _dispatch_key_for_device(device_type: str) -> str: ...
def _parse_dispatch_key(key: str) -> Optional[DispatchKey]: ...
def _dispatch_key_parse(dispatch: _dispatchkey) -> DispatchKey: ...
def _dispatch_num_backends() -> _int: ...
def _dispatch_pystub(name: str, overload: str) -> Optional[Tuple[str, str]]: ...
def _dispatch_is_alias_key(dispatch: _dispatchkey) -> _bool: ...
def _functionality_to_backend_keys(dispatch: _dispatchkey) -> List[DispatchKey]: ...
def _functionalization_reapply_views_tls() -> _bool: ...
def _set_throw_on_mutable_data_ptr(tensor: Tensor) -> None: ...
def _set_warn_deprecated_on_mutable_data_ptr(tensor: Tensor) -> None: ...

class DispatchKey(Enum):
    Undefined: DispatchKey = ...
    FPGA: DispatchKey = ...
    ORT: DispatchKey = ...
    Vulkan: DispatchKey = ...
    Metal: DispatchKey = ...
    MKLDNN: DispatchKey = ...
    OpenGL: DispatchKey = ...
    OpenCL: DispatchKey = ...
    IDEEP: DispatchKey = ...
    CustomRNGKeyId: DispatchKey = ...
    MkldnnCPU: DispatchKey = ...
    Sparse: DispatchKey = ...
    SparseCsr: DispatchKey = ...
    NestedTensor: DispatchKey = ...
    Dense: DispatchKey = ...
    PythonTLSSnapshot: DispatchKey = ...
    PreDispatch: DispatchKey = ...
    PythonDispatcher: DispatchKey = ...
    Python: DispatchKey = ...
    FuncTorchDynamicLayerBackMode: DispatchKey = ...
    ZeroTensor: DispatchKey = ...
    Conjugate: DispatchKey = ...
    Negative: DispatchKey = ...
    BackendSelect: DispatchKey = ...
    Named: DispatchKey = ...
    AutogradOther: DispatchKey = ...
    AutogradFunctionality: DispatchKey = ...
    AutogradNestedTensor: DispatchKey = ...
    Tracer: DispatchKey = ...
    Autocast: DispatchKey = ...
    AutocastCPU: DispatchKey = ...
    AutocastCUDA: DispatchKey = ...
    Batched: DispatchKey = ...
    VmapMode: DispatchKey = ...
    FuncTorchGradWrapper: DispatchKey = ...
    FuncTorchBatched: DispatchKey = ...
    BatchedNestedTensor: DispatchKey = ...
    FuncTorchVmapMode: DispatchKey = ...
    FuncTorchDynamicLayerFrontMode: DispatchKey = ...
    Functionalize: DispatchKey = ...
    TESTING_ONLY_GenericWrapper: DispatchKey = ...
    TESTING_ONLY_GenericMode: DispatchKey = ...
    ADInplaceOrView: DispatchKey = ...
    Autograd: DispatchKey = ...
    CompositeImplicitAutograd: DispatchKey = ...
    CompositeImplicitAutogradNestedTensor: DispatchKey = ...
    CompositeExplicitAutograd: DispatchKey = ...
    CompositeExplicitAutogradNonFunctional: DispatchKey = ...
    FuncTorchBatchedDecomposition: DispatchKey = ...
    CPU: DispatchKey = ...
    CUDA: DispatchKey = ...
    HIP: DispatchKey = ...
    XLA: DispatchKey = ...
    MTIA: DispatchKey = ...
    MPS: DispatchKey = ...
    IPU: DispatchKey = ...
    XPU: DispatchKey = ...
    HPU: DispatchKey = ...
    VE: DispatchKey = ...
    Lazy: DispatchKey = ...
    Meta: DispatchKey = ...
    PrivateUse1: DispatchKey = ...
    PrivateUse2: DispatchKey = ...
    PrivateUse3: DispatchKey = ...
    QuantizedCPU: DispatchKey = ...
    QuantizedCUDA: DispatchKey = ...
    QuantizedHIP: DispatchKey = ...
    QuantizedXLA: DispatchKey = ...
    QuantizedMTIA: DispatchKey = ...
    QuantizedMPS: DispatchKey = ...
    QuantizedIPU: DispatchKey = ...
    QuantizedXPU: DispatchKey = ...
    QuantizedHPU: DispatchKey = ...
    QuantizedVE: DispatchKey = ...
    QuantizedLazy: DispatchKey = ...
    QuantizedMeta: DispatchKey = ...
    QuantizedPrivateUse1: DispatchKey = ...
    QuantizedPrivateUse2: DispatchKey = ...
    QuantizedPrivateUse3: DispatchKey = ...
    SparseCPU: DispatchKey = ...
    SparseCUDA: DispatchKey = ...
    SparseHIP: DispatchKey = ...
    SparseXLA: DispatchKey = ...
    SparseMTIA: DispatchKey = ...
    SparseMPS: DispatchKey = ...
    SparseIPU: DispatchKey = ...
    SparseXPU: DispatchKey = ...
    SparseHPU: DispatchKey = ...
    SparseVE: DispatchKey = ...
    SparseLazy: DispatchKey = ...
    SparseMeta: DispatchKey = ...
    SparsePrivateUse1: DispatchKey = ...
    SparsePrivateUse2: DispatchKey = ...
    SparsePrivateUse3: DispatchKey = ...
    SparseCsrCPU: DispatchKey = ...
    SparseCsrCUDA: DispatchKey = ...
    SparseCsrHIP: DispatchKey = ...
    SparseCsrXLA: DispatchKey = ...
    SparseCsrMTIA: DispatchKey = ...
    SparseCsrMPS: DispatchKey = ...
    SparseCsrIPU: DispatchKey = ...
    SparseCsrXPU: DispatchKey = ...
    SparseCsrHPU: DispatchKey = ...
    SparseCsrVE: DispatchKey = ...
    SparseCsrLazy: DispatchKey = ...
    SparseCsrMeta: DispatchKey = ...
    SparseCsrPrivateUse1: DispatchKey = ...
    SparseCsrPrivateUse2: DispatchKey = ...
    SparseCsrPrivateUse3: DispatchKey = ...
    NestedTensorCPU: DispatchKey = ...
    NestedTensorCUDA: DispatchKey = ...
    NestedTensorHIP: DispatchKey = ...
    NestedTensorXLA: DispatchKey = ...
    NestedTensorMTIA: DispatchKey = ...
    NestedTensorMPS: DispatchKey = ...
    NestedTensorIPU: DispatchKey = ...
    NestedTensorXPU: DispatchKey = ...
    NestedTensorHPU: DispatchKey = ...
    NestedTensorVE: DispatchKey = ...
    NestedTensorLazy: DispatchKey = ...
    NestedTensorMeta: DispatchKey = ...
    NestedTensorPrivateUse1: DispatchKey = ...
    NestedTensorPrivateUse2: DispatchKey = ...
    NestedTensorPrivateUse3: DispatchKey = ...
    AutogradCPU: DispatchKey = ...
    AutogradCUDA: DispatchKey = ...
    AutogradHIP: DispatchKey = ...
    AutogradXLA: DispatchKey = ...
    AutogradMTIA: DispatchKey = ...
    AutogradMPS: DispatchKey = ...
    AutogradIPU: DispatchKey = ...
    AutogradXPU: DispatchKey = ...
    AutogradHPU: DispatchKey = ...
    AutogradVE: DispatchKey = ...
    AutogradLazy: DispatchKey = ...
    AutogradMeta: DispatchKey = ...
    AutogradPrivateUse1: DispatchKey = ...
    AutogradPrivateUse2: DispatchKey = ...
    AutogradPrivateUse3: DispatchKey = ...

class DispatchKeySet:
    def __init__(self, key: DispatchKey) -> None: ...
    def __or__(self, other: DispatchKeySet) -> DispatchKeySet: ...
    def __sub__(self, other: DispatchKeySet) -> DispatchKeySet: ...
    def __and__(self, other: DispatchKeySet) -> DispatchKeySet: ...
    def highestPriorityTypeId(self) -> DispatchKey: ...
    def has(self, k: _dispatchkey) -> _bool: ...
    def add(self, k: _dispatchkey) -> DispatchKeySet: ...
    def remove(self, k: _dispatchkey) -> DispatchKeySet: ...
    def __repr__(self) -> str: ...

_dispatch_autogradother_backends: DispatchKeySet
_additional_keys_to_prop_for_wrapper_tensors: DispatchKeySet

def _dispatch_has_backend_fallback(dispatch: _dispatchkey) -> _bool: ...
def _dispatch_keyset_full_after(t: _dispatchkey) -> DispatchKeySet: ...
def _dispatch_keyset_full() -> DispatchKeySet: ...
def _dispatch_keyset_to_string(keyset: DispatchKeySet) -> str: ...
def _dispatch_get_backend_keyset_from_autograd(
    dispatch: _dispatchkey,
) -> DispatchKeySet: ...
def _dispatch_keys(tensor: Tensor) -> DispatchKeySet: ...
def _dispatch_tls_local_exclude_set() -> DispatchKeySet: ...
def _dispatch_tls_local_include_set() -> DispatchKeySet: ...
def _dispatch_is_included_in_alias(
    dispatch_a: _dispatchkey,
    dispatch_b: _dispatchkey,
) -> _bool: ...
def _propagate_xla_data(a: Tensor, b: Tensor) -> None: ...
def _replace_(a: Tensor, b: Tensor) -> None: ...
def _commit_update(a: Tensor) -> None: ...

class _ExcludeDispatchKeyGuard:
    def __init__(self, keyset: DispatchKeySet): ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_value, traceback): ...

class _IncludeDispatchKeyGuard:
    def __init__(self, k: DispatchKey): ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_value, traceback): ...

class _ForceDispatchKeyGuard:
    def __init__(self, include: DispatchKeySet, exclude: DispatchKeySet): ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_value, traceback): ...

class _PreserveDispatchKeyGuard:
    def __init__(self): ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_value, traceback): ...

class _AutoDispatchBelowAutograd:
    def __init__(self): ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_value, traceback): ...

class _AutoDispatchBelowADInplaceOrView:
    def __init__(self): ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_value, traceback): ...

def _dispatch_print_registrations_for_dispatch_key(dispatch_key: str = "") -> None: ...
def _dispatch_get_registrations_for_dispatch_key(
    dispatch_key: str = "",
) -> List[str]: ...
def _are_functorch_transforms_active() -> _bool: ...

# Define in torch/csrc/autograd/init.cpp
def _set_python_dispatcher(dispatcher: object) -> None: ...

def _get_nested_int(id: _int, coeff: _int) -> SymInt: ...

def _get_constant_bool_symnode(val: _bool) -> Any: ...

class _TorchDispatchModeKey(Enum):
    FAKE: _TorchDispatchModeKey = ...
    PROXY: _TorchDispatchModeKey = ...
    FUNCTIONAL: _TorchDispatchModeKey = ...

class _SetExcludeDispatchKeyGuard:
    def __init__(self, k: DispatchKey, enabled: _bool): ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_value, traceback): ...

# Defined in torch/csrc/utils/init.cpp
class BenchmarkConfig:
    num_calling_threads: _int
    num_worker_threads: _int
    num_warmup_iters: _int
    num_iters: _int
    profiler_output_path: str

class BenchmarkExecutionStats:
    latency_avg_ms: _float
    num_iters: _int

class ThroughputBenchmark:
    def __init__(self, module: Any) -> None: ...
    def add_input(self, *args: Any, **kwargs: Any) -> None: ...
    def run_once(self, *args: Any, **kwargs: Any) -> Any: ...
    def benchmark(self, config: BenchmarkConfig) -> BenchmarkExecutionStats: ...

# Defined in torch/csrc/Storage.cpp
class StorageBase(object): ...

# TODO: where
class DoubleTensor(Tensor): ...
class FloatTensor(Tensor): ...
class BFloat16Tensor(Tensor): ...
class LongTensor(Tensor): ...
class IntTensor(Tensor): ...
class ShortTensor(Tensor): ...
class HalfTensor(Tensor): ...
class CharTensor(Tensor): ...
class ByteTensor(Tensor): ...
class BoolTensor(Tensor): ...

# Defined in torch/csrc/autograd/python_engine.cpp
class _ImperativeEngine:
    def queue_callback(self, callback: Callable[[], None]) -> None: ...
    def run_backward(self, *args: Any, **kwargs: Any) -> Tuple[Tensor, ...]: ...
    def is_checkpoint_valid(self) -> _bool: ...

# Defined in torch/csrc/autograd/python_variable.cpp
class _TensorMeta(type): ...

# Defined in torch/csrc/autograd/python_variable.cpp
class TensorBase(metaclass=_TensorMeta):
    requires_grad: _bool
    retains_grad: _bool
    shape: Size
    data: Tensor
    names: List[str]
    device: _device
    dtype: _dtype
    layout: _layout
    real: Tensor
    imag: Tensor
    T: Tensor
    H: Tensor
    mT: Tensor
    mH: Tensor
    ndim: _int
    output_nr: _int
    _version: _int
    _base: Optional[Tensor]
    _cdata: _int
    grad_fn: Optional[_Node]
    _grad_fn: Any
    _grad: Optional[Tensor]
    grad: Optional[Tensor]
    _backward_hooks: Optional[Dict[_int, Callable[[Tensor], Optional[Tensor]]]]
    nbytes: _int
    itemsize: _int
    _has_symbolic_sizes_strides: _bool

    def _view_func_unsafe(
        self,
        new_base: Tensor,
        symint_visitor_fn: Optional[Callable[[_int], _int]] = None,
        tensor_visitor_fn: Optional[Callable[[Tensor], Tensor]] = None
    ):
        ...

    def __abs__(self) -> Tensor: ...
    def __add__(self, other: Any) -> Tensor: ...
    @overload
    def __and__(self, other: Tensor) -> Tensor: ...
    @overload
    def __and__(self, other: Union[Number, _complex]) -> Tensor: ...
    @overload
    def __and__(self, other: Any) -> Tensor: ...
    def __bool__(self) -> builtins.bool: ...
    def __complex__(self) -> builtins.complex: ...
    def __div__(self, other: Any) -> Tensor: ...
    def __eq__(self, other: Any) -> Tensor: ...  # type: ignore[override]
    def __float__(self) -> builtins.float: ...
    def __floordiv__(self, other: Any) -> Tensor: ...
    def __ge__(self, other: Any) -> Tensor: ...
    def __getitem__(self, indices: Union[Union[SupportsIndex, Union[None, _bool, _int, slice, ellipsis, Tensor], _NestedSequence[Union[None, _bool, _int, slice, ellipsis, Tensor]]], tuple[Union[SupportsIndex, Union[None, _bool, _int, slice, ellipsis, Tensor], _NestedSequence[Union[None, _bool, _int, slice, ellipsis, Tensor]]], ...]]) -> Tensor: ...
    def __gt__(self, other: Any) -> Tensor: ...
    def __iadd__(self, other: Any) -> Tensor: ...
    @overload
    def __iand__(self, other: Tensor) -> Tensor: ...
    @overload
    def __iand__(self, other: Union[Number, _complex]) -> Tensor: ...
    @overload
    def __iand__(self, other: Any) -> Tensor: ...
    def __idiv__(self, other: Any) -> Tensor: ...
    def __ifloordiv__(self, other: Any) -> Tensor: ...
    @overload
    def __ilshift__(self, other: Tensor) -> Tensor: ...
    @overload
    def __ilshift__(self, other: Union[Number, _complex]) -> Tensor: ...
    @overload
    def __ilshift__(self, other: Any) -> Tensor: ...
    def __imod__(self, other: Any) -> Tensor: ...
    def __imul__(self, other: Any) -> Tensor: ...
    def __index__(self) -> builtins.int: ...
    @overload
    def __init__(self, *args: Any, device: Optional[DeviceLikeType] = None) -> None: ...
    @overload
    def __init__(self, storage: Storage) -> None: ...
    @overload
    def __init__(self, other: Tensor) -> None: ...
    @overload
    def __init__(self, size: _size, *, device: Optional[DeviceLikeType] = None) -> None: ...
    def __int__(self) -> builtins.int: ...
    def __invert__(self) -> Tensor: ...
    @overload
    def __ior__(self, other: Tensor) -> Tensor: ...
    @overload
    def __ior__(self, other: Union[Number, _complex]) -> Tensor: ...
    @overload
    def __ior__(self, other: Any) -> Tensor: ...
    @overload
    def __irshift__(self, other: Tensor) -> Tensor: ...
    @overload
    def __irshift__(self, other: Union[Number, _complex]) -> Tensor: ...
    @overload
    def __irshift__(self, other: Any) -> Tensor: ...
    def __isub__(self, other: Any) -> Tensor: ...
    @overload
    def __ixor__(self, other: Tensor) -> Tensor: ...
    @overload
    def __ixor__(self, other: Union[Number, _complex]) -> Tensor: ...
    @overload
    def __ixor__(self, other: Any) -> Tensor: ...
    def __le__(self, other: Any) -> Tensor: ...
    def __long__(self) -> builtins.int: ...
    @overload
    def __lshift__(self, other: Tensor) -> Tensor: ...
    @overload
    def __lshift__(self, other: Union[Number, _complex]) -> Tensor: ...
    @overload
    def __lshift__(self, other: Any) -> Tensor: ...
    def __lt__(self, other: Any) -> Tensor: ...
    def __matmul__(self, other: Any) -> Tensor: ...
    def __mod__(self, other: Any) -> Tensor: ...
    def __mul__(self, other: Any) -> Tensor: ...
    def __ne__(self, other: Any) -> Tensor: ...  # type: ignore[override]
    def __neg__(self) -> Tensor: ...
    def __new__(self, *args, **kwargs) -> Tensor: ...
    def __nonzero__(self) -> builtins.bool: ...
    @overload
    def __or__(self, other: Tensor) -> Tensor: ...
    @overload
    def __or__(self, other: Union[Number, _complex]) -> Tensor: ...
    @overload
    def __or__(self, other: Any) -> Tensor: ...
    def __pow__(self, other: Any) -> Tensor: ...
    def __radd__(self, other: Any) -> Tensor: ...
    def __rand__(self, other: Any) -> Tensor: ...
    def __rfloordiv__(self, other: Any) -> Tensor: ...
    def __rmul__(self, other: Any) -> Tensor: ...
    def __ror__(self, other: Any) -> Tensor: ...
    def __rpow__(self, other: Any) -> Tensor: ...
    @overload
    def __rshift__(self, other: Tensor) -> Tensor: ...
    @overload
    def __rshift__(self, other: Union[Number, _complex]) -> Tensor: ...
    @overload
    def __rshift__(self, other: Any) -> Tensor: ...
    def __rsub__(self, other: Any) -> Tensor: ...
    def __rtruediv__(self, other: Any) -> Tensor: ...
    def __rxor__(self, other: Any) -> Tensor: ...
    def __setitem__(self, indices: Union[Union[SupportsIndex, Union[None, _bool, _int, slice, ellipsis, Tensor], _NestedSequence[Union[None, _bool, _int, slice, ellipsis, Tensor]]], tuple[Union[SupportsIndex, Union[None, _bool, _int, slice, ellipsis, Tensor], _NestedSequence[Union[None, _bool, _int, slice, ellipsis, Tensor]]], ...]], val: Union[Tensor, Number]) -> None: ...
    def __sub__(self, other: Any) -> Tensor: ...
    def __truediv__(self, other: Any) -> Tensor: ...
    @overload
    def __xor__(self, other: Tensor) -> Tensor: ...
    @overload
    def __xor__(self, other: Union[Number, _complex]) -> Tensor: ...
    @overload
    def __xor__(self, other: Any) -> Tensor: ...
    def _addmm_activation(self, mat1: Tensor, mat2: Tensor, *, beta: Union[Number, _complex] = 1, alpha: Union[Number, _complex] = 1, use_gelu: _bool = False) -> Tensor: ...
    def _autocast_to_full_precision(self, cuda_enabled: _bool, cpu_enabled: _bool) -> Tensor: ...
    def _autocast_to_reduced_precision(self, cuda_enabled: _bool, cpu_enabled: _bool, cuda_dtype: _dtype, cpu_dtype: _dtype) -> Tensor: ...
    def _coalesced_(self, coalesced: _bool) -> Tensor: ...
    def _conj(self) -> Tensor: ...
    def _conj_physical(self) -> Tensor: ...
    def _dimI(self) -> _int: ...
    def _dimV(self) -> _int: ...
    def _indices(self) -> Tensor: ...
    def _is_all_true(self) -> Tensor: ...
    def _is_any_true(self) -> Tensor: ...
    def _is_view(self) -> _bool: ...
    def _is_zerotensor(self) -> _bool: ...
    def _lazy_clone(self) -> Tensor: ...
    @staticmethod
    def _make_subclass(cls: Type[S], data: Tensor, require_grad: _bool = False, dispatch_strides: _bool = False, dispatch_device: _bool = False, device_for_backend_keys: Optional[_device] = None) -> S: ...
    def _neg_view(self) -> Tensor: ...
    def _nested_tensor_size(self) -> Tensor: ...
    def _nested_tensor_storage_offsets(self) -> Tensor: ...
    def _nested_tensor_strides(self) -> Tensor: ...
    def _nnz(self) -> _int: ...
    def _sparse_mask_projection(self, mask: Tensor, accumulate_matches: _bool = False) -> Tensor: ...
    def _to_dense(self, dtype: Optional[_dtype] = None, masked_grad: Optional[_bool] = None) -> Tensor: ...
    @overload
    def _to_sparse(self, *, layout: Optional[_layout] = None, blocksize: Optional[Union[_int, _size]] = None, dense_dim: Optional[_int] = None) -> Tensor: ...
    @overload
    def _to_sparse(self, sparse_dim: _int) -> Tensor: ...
    def _to_sparse_bsc(self, blocksize: Union[_int, _size], dense_dim: Optional[_int] = None) -> Tensor: ...
    def _to_sparse_bsr(self, blocksize: Union[_int, _size], dense_dim: Optional[_int] = None) -> Tensor: ...
    def _to_sparse_csc(self, dense_dim: Optional[_int] = None) -> Tensor: ...
    def _to_sparse_csr(self, dense_dim: Optional[_int] = None) -> Tensor: ...
    def _values(self) -> Tensor: ...
    def abs(self) -> Tensor:
        r"""
        abs() -> Tensor

        See :func:`torch.abs`
        """
        ...
    def abs_(self) -> Tensor:
        r"""
        abs_() -> Tensor

        In-place version of :meth:`~Tensor.abs`
        """
        ...
    def absolute(self) -> Tensor:
        r"""
        absolute() -> Tensor

        Alias for :func:`abs`
        """
        ...
    def absolute_(self) -> Tensor:
        r"""
        absolute_() -> Tensor

        In-place version of :meth:`~Tensor.absolute`
        Alias for :func:`abs_`
        """
        ...
    def acos(self) -> Tensor:
        r"""
        acos() -> Tensor

        See :func:`torch.acos`
        """
        ...
    def acos_(self) -> Tensor:
        r"""
        acos_() -> Tensor

        In-place version of :meth:`~Tensor.acos`
        """
        ...
    def acosh(self) -> Tensor:
        r"""
        acosh() -> Tensor

        See :func:`torch.acosh`
        """
        ...
    def acosh_(self) -> Tensor:
        r"""
        acosh_() -> Tensor

        In-place version of :meth:`~Tensor.acosh`
        """
        ...
    def add(self, other: Union[Tensor, Number, _complex, torch.SymInt, torch.SymFloat], *, alpha: Optional[Union[Number, _complex]] = 1, out: Optional[Tensor] = None) -> Tensor:
        r"""
        add(other, *, alpha=1) -> Tensor

        Add a scalar or tensor to :attr:`self` tensor. If both :attr:`alpha`
        and :attr:`other` are specified, each element of :attr:`other` is scaled by
        :attr:`alpha` before being used.

        When :attr:`other` is a tensor, the shape of :attr:`other` must be
        :ref:`broadcastable <broadcasting-semantics>` with the shape of the underlying
        tensor

        See :func:`torch.add`
        """
        ...
    def add_(self, other: Union[Tensor, Number, _complex, torch.SymInt, torch.SymFloat], *, alpha: Optional[Union[Number, _complex]] = 1) -> Tensor:
        r"""
        add_(other, *, alpha=1) -> Tensor

        In-place version of :meth:`~Tensor.add`
        """
        ...
    def addbmm(self, batch1: Tensor, batch2: Tensor, *, beta: Union[Number, _complex] = 1, alpha: Union[Number, _complex] = 1) -> Tensor:
        r"""
        addbmm(batch1, batch2, *, beta=1, alpha=1) -> Tensor

        See :func:`torch.addbmm`
        """
        ...
    def addbmm_(self, batch1: Tensor, batch2: Tensor, *, beta: Union[Number, _complex] = 1, alpha: Union[Number, _complex] = 1) -> Tensor:
        r"""
        addbmm_(batch1, batch2, *, beta=1, alpha=1) -> Tensor

        In-place version of :meth:`~Tensor.addbmm`
        """
        ...
    def addcdiv(self, tensor1: Tensor, tensor2: Tensor, *, value: Union[Number, _complex] = 1) -> Tensor:
        r"""
        addcdiv(tensor1, tensor2, *, value=1) -> Tensor

        See :func:`torch.addcdiv`
        """
        ...
    def addcdiv_(self, tensor1: Tensor, tensor2: Tensor, *, value: Union[Number, _complex] = 1) -> Tensor:
        r"""
        addcdiv_(tensor1, tensor2, *, value=1) -> Tensor

        In-place version of :meth:`~Tensor.addcdiv`
        """
        ...
    def addcmul(self, tensor1: Tensor, tensor2: Tensor, *, value: Union[Number, _complex] = 1) -> Tensor:
        r"""
        addcmul(tensor1, tensor2, *, value=1) -> Tensor

        See :func:`torch.addcmul`
        """
        ...
    def addcmul_(self, tensor1: Tensor, tensor2: Tensor, *, value: Union[Number, _complex] = 1) -> Tensor:
        r"""
        addcmul_(tensor1, tensor2, *, value=1) -> Tensor

        In-place version of :meth:`~Tensor.addcmul`
        """
        ...
    def addmm(self, mat1: Tensor, mat2: Tensor, *, beta: Union[Number, _complex] = 1, alpha: Union[Number, _complex] = 1) -> Tensor:
        r"""
        addmm(mat1, mat2, *, beta=1, alpha=1) -> Tensor

        See :func:`torch.addmm`
        """
        ...
    def addmm_(self, mat1: Tensor, mat2: Tensor, *, beta: Union[Number, _complex] = 1, alpha: Union[Number, _complex] = 1) -> Tensor:
        r"""
        addmm_(mat1, mat2, *, beta=1, alpha=1) -> Tensor

        In-place version of :meth:`~Tensor.addmm`
        """
        ...
    def addmv(self, mat: Tensor, vec: Tensor, *, beta: Union[Number, _complex] = 1, alpha: Union[Number, _complex] = 1) -> Tensor:
        r"""
        addmv(mat, vec, *, beta=1, alpha=1) -> Tensor

        See :func:`torch.addmv`
        """
        ...
    def addmv_(self, mat: Tensor, vec: Tensor, *, beta: Union[Number, _complex] = 1, alpha: Union[Number, _complex] = 1) -> Tensor:
        r"""
        addmv_(mat, vec, *, beta=1, alpha=1) -> Tensor

        In-place version of :meth:`~Tensor.addmv`
        """
        ...
    def addr(self, vec1: Tensor, vec2: Tensor, *, beta: Union[Number, _complex] = 1, alpha: Union[Number, _complex] = 1) -> Tensor:
        r"""
        addr(vec1, vec2, *, beta=1, alpha=1) -> Tensor

        See :func:`torch.addr`
        """
        ...
    def addr_(self, vec1: Tensor, vec2: Tensor, *, beta: Union[Number, _complex] = 1, alpha: Union[Number, _complex] = 1) -> Tensor:
        r"""
        addr_(vec1, vec2, *, beta=1, alpha=1) -> Tensor

        In-place version of :meth:`~Tensor.addr`
        """
        ...
    def adjoint(self) -> Tensor:
        r"""
        adjoint() -> Tensor

        Alias for :func:`adjoint`
        """
        ...
    def align_as(self, other: Tensor) -> Tensor:
        r"""
        align_as(other) -> Tensor

        Permutes the dimensions of the :attr:`self` tensor to match the dimension order
        in the :attr:`other` tensor, adding size-one dims for any new names.

        This operation is useful for explicit broadcasting by names (see examples).

        All of the dims of :attr:`self` must be named in order to use this method.
        The resulting tensor is a view on the original tensor.

        All dimension names of :attr:`self` must be present in ``other.names``.
        :attr:`other` may contain named dimensions that are not in ``self.names``;
        the output tensor has a size-one dimension for each of those new names.

        To align a tensor to a specific order, use :meth:`~Tensor.align_to`.

        Examples::

            # Example 1: Applying a mask
            >>> mask = torch.randint(2, [127, 128], dtype=torch.bool).refine_names('W', 'H')
            >>> imgs = torch.randn(32, 128, 127, 3, names=('N', 'H', 'W', 'C'))
            >>> imgs.masked_fill_(mask.align_as(imgs), 0)


            # Example 2: Applying a per-channel-scale
            >>> def scale_channels(input, scale):
            >>>    scale = scale.refine_names('C')
            >>>    return input * scale.align_as(input)

            >>> num_channels = 3
            >>> scale = torch.randn(num_channels, names=('C',))
            >>> imgs = torch.rand(32, 128, 128, num_channels, names=('N', 'H', 'W', 'C'))
            >>> more_imgs = torch.rand(32, num_channels, 128, 128, names=('N', 'C', 'H', 'W'))
            >>> videos = torch.randn(3, num_channels, 128, 128, 128, names=('N', 'C', 'H', 'W', 'D'))

            # scale_channels is agnostic to the dimension order of the input
            >>> scale_channels(imgs, scale)
            >>> scale_channels(more_imgs, scale)
            >>> scale_channels(videos, scale)

        .. warning::
            The named tensor API is experimental and subject to change.
        """
        ...
    @overload
    def align_to(self, order: Sequence[Union[str, ellipsis, None]], ellipsis_idx: _int) -> Tensor: ...
    @overload
    def align_to(self, names: Sequence[Union[str, ellipsis, None]]) -> Tensor: ...
    @overload
    def all(self) -> Tensor:
        r"""
        all(dim=None, keepdim=False) -> Tensor

        See :func:`torch.all`
        """
        ...
    @overload
    def all(self, dim: Optional[_size] = None, keepdim: _bool = False) -> Tensor:
        r"""
        all(dim=None, keepdim=False) -> Tensor

        See :func:`torch.all`
        """
        ...
    @overload
    def all(self, dim: _int, keepdim: _bool = False) -> Tensor:
        r"""
        all(dim=None, keepdim=False) -> Tensor

        See :func:`torch.all`
        """
        ...
    @overload
    def all(self, dim: Union[str, ellipsis, None], keepdim: _bool = False) -> Tensor:
        r"""
        all(dim=None, keepdim=False) -> Tensor

        See :func:`torch.all`
        """
        ...
    def allclose(self, other: Tensor, rtol: _float = 1e-05, atol: _float = 1e-08, equal_nan: _bool = False) -> _bool:
        r"""
        allclose(other, rtol=1e-05, atol=1e-08, equal_nan=False) -> Tensor

        See :func:`torch.allclose`
        """
        ...
    def amax(self, dim: Union[_int, _size] = (), keepdim: _bool = False) -> Tensor:
        r"""
        amax(dim=None, keepdim=False) -> Tensor

        See :func:`torch.amax`
        """
        ...
    def amin(self, dim: Union[_int, _size] = (), keepdim: _bool = False) -> Tensor:
        r"""
        amin(dim=None, keepdim=False) -> Tensor

        See :func:`torch.amin`
        """
        ...
    def aminmax(self, *, dim: Optional[_int] = None, keepdim: _bool = False) -> torch.return_types.aminmax:
        r"""
        aminmax(*, dim=None, keepdim=False) -> (Tensor min, Tensor max)

        See :func:`torch.aminmax`
        """
        ...
    def angle(self) -> Tensor:
        r"""
        angle() -> Tensor

        See :func:`torch.angle`
        """
        ...
    @overload
    def any(self) -> Tensor:
        r"""
        any(dim=None, keepdim=False) -> Tensor

        See :func:`torch.any`
        """
        ...
    @overload
    def any(self, dim: Optional[_size] = None, keepdim: _bool = False) -> Tensor:
        r"""
        any(dim=None, keepdim=False) -> Tensor

        See :func:`torch.any`
        """
        ...
    @overload
    def any(self, dim: _int, keepdim: _bool = False) -> Tensor:
        r"""
        any(dim=None, keepdim=False) -> Tensor

        See :func:`torch.any`
        """
        ...
    @overload
    def any(self, dim: Union[str, ellipsis, None], keepdim: _bool = False) -> Tensor:
        r"""
        any(dim=None, keepdim=False) -> Tensor

        See :func:`torch.any`
        """
        ...
    def apply_(self, callable: Callable) -> Tensor:
        r"""
        apply_(callable) -> Tensor

        Applies the function :attr:`callable` to each element in the tensor, replacing
        each element with the value returned by :attr:`callable`.

        .. note::

            This function only works with CPU tensors and should not be used in code
            sections that require high performance.
        """
        ...
    def arccos(self) -> Tensor:
        r"""
        arccos() -> Tensor

        See :func:`torch.arccos`
        """
        ...
    def arccos_(self) -> Tensor:
        r"""
        arccos_() -> Tensor

        In-place version of :meth:`~Tensor.arccos`
        """
        ...
    def arccosh(self) -> Tensor:
        r"""
        acosh() -> Tensor

        See :func:`torch.arccosh`
        """
        ...
    def arccosh_(self) -> Tensor:
        r"""
        acosh_() -> Tensor

        In-place version of :meth:`~Tensor.arccosh`
        """
        ...
    def arcsin(self) -> Tensor:
        r"""
        arcsin() -> Tensor

        See :func:`torch.arcsin`
        """
        ...
    def arcsin_(self) -> Tensor:
        r"""
        arcsin_() -> Tensor

        In-place version of :meth:`~Tensor.arcsin`
        """
        ...
    def arcsinh(self) -> Tensor:
        r"""
        arcsinh() -> Tensor

        See :func:`torch.arcsinh`
        """
        ...
    def arcsinh_(self) -> Tensor:
        r"""
        arcsinh_() -> Tensor

        In-place version of :meth:`~Tensor.arcsinh`
        """
        ...
    def arctan(self) -> Tensor:
        r"""
        arctan() -> Tensor

        See :func:`torch.arctan`
        """
        ...
    def arctan2(self, other: Tensor) -> Tensor:
        r"""
        arctan2(other) -> Tensor

        See :func:`torch.arctan2`
        """
        ...
    def arctan2_(self, other: Tensor) -> Tensor:
        r"""
        atan2_(other) -> Tensor

        In-place version of :meth:`~Tensor.arctan2`
        """
        ...
    def arctan_(self) -> Tensor:
        r"""
        arctan_() -> Tensor

        In-place version of :meth:`~Tensor.arctan`
        """
        ...
    def arctanh(self) -> Tensor:
        r"""
        arctanh() -> Tensor

        See :func:`torch.arctanh`
        """
        ...
    def arctanh_(self) -> Tensor:
        r"""
        arctanh_(other) -> Tensor

        In-place version of :meth:`~Tensor.arctanh`
        """
        ...
    def argmax(self, dim: Optional[_int] = None, keepdim: _bool = False) -> Tensor:
        r"""
        argmax(dim=None, keepdim=False) -> LongTensor

        See :func:`torch.argmax`
        """
        ...
    def argmin(self, dim: Optional[_int] = None, keepdim: _bool = False) -> Tensor:
        r"""
        argmin(dim=None, keepdim=False) -> LongTensor

        See :func:`torch.argmin`
        """
        ...
    @overload
    def argsort(self, *, stable: _bool, dim: _int = -1, descending: _bool = False) -> Tensor:
        r"""
        argsort(dim=-1, descending=False) -> LongTensor

        See :func:`torch.argsort`
        """
        ...
    @overload
    def argsort(self, dim: _int = -1, descending: _bool = False) -> Tensor:
        r"""
        argsort(dim=-1, descending=False) -> LongTensor

        See :func:`torch.argsort`
        """
        ...
    @overload
    def argsort(self, dim: Union[str, ellipsis, None], descending: _bool = False) -> Tensor:
        r"""
        argsort(dim=-1, descending=False) -> LongTensor

        See :func:`torch.argsort`
        """
        ...
    def argwhere(self) -> Tensor:
        r"""
        argwhere() -> Tensor

        See :func:`torch.argwhere`
        """
        ...
    def as_strided(self, size: Sequence[Union[_int, SymInt]], stride: Sequence[Union[_int, SymInt]], storage_offset: Optional[Union[_int, SymInt]] = None) -> Tensor:
        r"""
        as_strided(size, stride, storage_offset=None) -> Tensor

        See :func:`torch.as_strided`
        """
        ...
    def as_strided_(self, size: Sequence[Union[_int, SymInt]], stride: Sequence[Union[_int, SymInt]], storage_offset: Optional[Union[_int, SymInt]] = None) -> Tensor:
        r"""
        as_strided_(size, stride, storage_offset=None) -> Tensor

        In-place version of :meth:`~Tensor.as_strided`
        """
        ...
    def as_strided_scatter(self, src: Tensor, size: Sequence[Union[_int, SymInt]], stride: Sequence[Union[_int, SymInt]], storage_offset: Optional[Union[_int, SymInt]] = None) -> Tensor:
        r"""
        as_strided_scatter(src, size, stride, storage_offset=None) -> Tensor

        See :func:`torch.as_strided_scatter`
        """
        ...
    def as_subclass(self, cls: Type[S]) -> S:
        r"""
        as_subclass(cls) -> Tensor

        Makes a ``cls`` instance with the same data pointer as ``self``. Changes
        in the output mirror changes in ``self``, and the output stays attached
        to the autograd graph. ``cls`` must be a subclass of ``Tensor``.
        """
        ...
    def asin(self) -> Tensor:
        r"""
        asin() -> Tensor

        See :func:`torch.asin`
        """
        ...
    def asin_(self) -> Tensor:
        r"""
        asin_() -> Tensor

        In-place version of :meth:`~Tensor.asin`
        """
        ...
    def asinh(self) -> Tensor:
        r"""
        asinh() -> Tensor

        See :func:`torch.asinh`
        """
        ...
    def asinh_(self) -> Tensor:
        r"""
        asinh_() -> Tensor

        In-place version of :meth:`~Tensor.asinh`
        """
        ...
    def atan(self) -> Tensor:
        r"""
        atan() -> Tensor

        See :func:`torch.atan`
        """
        ...
    def atan2(self, other: Tensor) -> Tensor:
        r"""
        atan2(other) -> Tensor

        See :func:`torch.atan2`
        """
        ...
    def atan2_(self, other: Tensor) -> Tensor:
        r"""
        atan2_(other) -> Tensor

        In-place version of :meth:`~Tensor.atan2`
        """
        ...
    def atan_(self) -> Tensor:
        r"""
        atan_() -> Tensor

        In-place version of :meth:`~Tensor.atan`
        """
        ...
    def atanh(self) -> Tensor:
        r"""
        atanh() -> Tensor

        See :func:`torch.atanh`
        """
        ...
    def atanh_(self) -> Tensor:
        r"""
        atanh_(other) -> Tensor

        In-place version of :meth:`~Tensor.atanh`
        """
        ...
    def baddbmm(self, batch1: Tensor, batch2: Tensor, *, beta: Union[Number, _complex] = 1, alpha: Union[Number, _complex] = 1) -> Tensor:
        r"""
        baddbmm(batch1, batch2, *, beta=1, alpha=1) -> Tensor

        See :func:`torch.baddbmm`
        """
        ...
    def baddbmm_(self, batch1: Tensor, batch2: Tensor, *, beta: Union[Number, _complex] = 1, alpha: Union[Number, _complex] = 1) -> Tensor:
        r"""
        baddbmm_(batch1, batch2, *, beta=1, alpha=1) -> Tensor

        In-place version of :meth:`~Tensor.baddbmm`
        """
        ...
    @overload
    def bernoulli(self, *, generator: Optional[Generator] = None) -> Tensor:
        r"""
        bernoulli(*, generator=None) -> Tensor

        Returns a result tensor where each :math:`\texttt{result[i]}` is independently
        sampled from :math:`\text{Bernoulli}(\texttt{self[i]})`. :attr:`self` must have
        floating point ``dtype``, and the result will have the same ``dtype``.

        See :func:`torch.bernoulli`
        """
        ...
    @overload
    def bernoulli(self, p: _float, *, generator: Optional[Generator] = None) -> Tensor:
        r"""
        bernoulli(*, generator=None) -> Tensor

        Returns a result tensor where each :math:`\texttt{result[i]}` is independently
        sampled from :math:`\text{Bernoulli}(\texttt{self[i]})`. :attr:`self` must have
        floating point ``dtype``, and the result will have the same ``dtype``.

        See :func:`torch.bernoulli`
        """
        ...
    @overload
    def bernoulli_(self, p: Tensor, *, generator: Optional[Generator] = None) -> Tensor:
        r"""
        bernoulli_(p=0.5, *, generator=None) -> Tensor

        Fills each location of :attr:`self` with an independent sample from
        :math:`\text{Bernoulli}(\texttt{p})`. :attr:`self` can have integral
        ``dtype``.

        :attr:`p` should either be a scalar or tensor containing probabilities to be
        used for drawing the binary random number.

        If it is a tensor, the :math:`\text{i}^{th}` element of :attr:`self` tensor
        will be set to a value sampled from
        :math:`\text{Bernoulli}(\texttt{p\_tensor[i]})`. In this case `p` must have
        floating point ``dtype``.

        See also :meth:`~Tensor.bernoulli` and :func:`torch.bernoulli`
        """
        ...
    @overload
    def bernoulli_(self, p: _float = 0.5, *, generator: Optional[Generator] = None) -> Tensor:
        r"""
        bernoulli_(p=0.5, *, generator=None) -> Tensor

        Fills each location of :attr:`self` with an independent sample from
        :math:`\text{Bernoulli}(\texttt{p})`. :attr:`self` can have integral
        ``dtype``.

        :attr:`p` should either be a scalar or tensor containing probabilities to be
        used for drawing the binary random number.

        If it is a tensor, the :math:`\text{i}^{th}` element of :attr:`self` tensor
        will be set to a value sampled from
        :math:`\text{Bernoulli}(\texttt{p\_tensor[i]})`. In this case `p` must have
        floating point ``dtype``.

        See also :meth:`~Tensor.bernoulli` and :func:`torch.bernoulli`
        """
        ...
    def bfloat16(self) -> Tensor:
        r"""
        bfloat16(memory_format=torch.preserve_format) -> Tensor
        ``self.bfloat16()`` is equivalent to ``self.to(torch.bfloat16)``. See :func:`to`.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.
        """
        ...
    def bincount(self, weights: Optional[Tensor] = None, minlength: _int = 0) -> Tensor:
        r"""
        bincount(weights=None, minlength=0) -> Tensor

        See :func:`torch.bincount`
        """
        ...
    @overload
    def bitwise_and(self, other: Tensor) -> Tensor:
        r"""
        bitwise_and() -> Tensor

        See :func:`torch.bitwise_and`
        """
        ...
    @overload
    def bitwise_and(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        bitwise_and() -> Tensor

        See :func:`torch.bitwise_and`
        """
        ...
    @overload
    def bitwise_and_(self, other: Tensor) -> Tensor:
        r"""
        bitwise_and_() -> Tensor

        In-place version of :meth:`~Tensor.bitwise_and`
        """
        ...
    @overload
    def bitwise_and_(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        bitwise_and_() -> Tensor

        In-place version of :meth:`~Tensor.bitwise_and`
        """
        ...
    @overload
    def bitwise_left_shift(self, other: Tensor) -> Tensor:
        r"""
        bitwise_left_shift(other) -> Tensor

        See :func:`torch.bitwise_left_shift`
        """
        ...
    @overload
    def bitwise_left_shift(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        bitwise_left_shift(other) -> Tensor

        See :func:`torch.bitwise_left_shift`
        """
        ...
    @overload
    def bitwise_left_shift_(self, other: Tensor) -> Tensor:
        r"""
        bitwise_left_shift_(other) -> Tensor

        In-place version of :meth:`~Tensor.bitwise_left_shift`
        """
        ...
    @overload
    def bitwise_left_shift_(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        bitwise_left_shift_(other) -> Tensor

        In-place version of :meth:`~Tensor.bitwise_left_shift`
        """
        ...
    def bitwise_not(self) -> Tensor:
        r"""
        bitwise_not() -> Tensor

        See :func:`torch.bitwise_not`
        """
        ...
    def bitwise_not_(self) -> Tensor:
        r"""
        bitwise_not_() -> Tensor

        In-place version of :meth:`~Tensor.bitwise_not`
        """
        ...
    @overload
    def bitwise_or(self, other: Tensor) -> Tensor:
        r"""
        bitwise_or() -> Tensor

        See :func:`torch.bitwise_or`
        """
        ...
    @overload
    def bitwise_or(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        bitwise_or() -> Tensor

        See :func:`torch.bitwise_or`
        """
        ...
    @overload
    def bitwise_or_(self, other: Tensor) -> Tensor:
        r"""
        bitwise_or_() -> Tensor

        In-place version of :meth:`~Tensor.bitwise_or`
        """
        ...
    @overload
    def bitwise_or_(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        bitwise_or_() -> Tensor

        In-place version of :meth:`~Tensor.bitwise_or`
        """
        ...
    @overload
    def bitwise_right_shift(self, other: Tensor) -> Tensor:
        r"""
        bitwise_right_shift(other) -> Tensor

        See :func:`torch.bitwise_right_shift`
        """
        ...
    @overload
    def bitwise_right_shift(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        bitwise_right_shift(other) -> Tensor

        See :func:`torch.bitwise_right_shift`
        """
        ...
    @overload
    def bitwise_right_shift_(self, other: Tensor) -> Tensor:
        r"""
        bitwise_right_shift_(other) -> Tensor

        In-place version of :meth:`~Tensor.bitwise_right_shift`
        """
        ...
    @overload
    def bitwise_right_shift_(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        bitwise_right_shift_(other) -> Tensor

        In-place version of :meth:`~Tensor.bitwise_right_shift`
        """
        ...
    @overload
    def bitwise_xor(self, other: Tensor) -> Tensor:
        r"""
        bitwise_xor() -> Tensor

        See :func:`torch.bitwise_xor`
        """
        ...
    @overload
    def bitwise_xor(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        bitwise_xor() -> Tensor

        See :func:`torch.bitwise_xor`
        """
        ...
    @overload
    def bitwise_xor_(self, other: Tensor) -> Tensor:
        r"""
        bitwise_xor_() -> Tensor

        In-place version of :meth:`~Tensor.bitwise_xor`
        """
        ...
    @overload
    def bitwise_xor_(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        bitwise_xor_() -> Tensor

        In-place version of :meth:`~Tensor.bitwise_xor`
        """
        ...
    def bmm(self, mat2: Tensor) -> Tensor:
        r"""
        bmm(batch2) -> Tensor

        See :func:`torch.bmm`
        """
        ...
    def bool(self) -> Tensor:
        r"""
        bool(memory_format=torch.preserve_format) -> Tensor

        ``self.bool()`` is equivalent to ``self.to(torch.bool)``. See :func:`to`.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.
        """
        ...
    @overload
    def broadcast_to(self, size: Sequence[Union[_int, SymInt]]) -> Tensor:
        r"""
        broadcast_to(shape) -> Tensor

        See :func:`torch.broadcast_to`.
        """
        ...
    @overload
    def broadcast_to(self, *size: _int) -> Tensor:
        r"""
        broadcast_to(shape) -> Tensor

        See :func:`torch.broadcast_to`.
        """
        ...
    def byte(self) -> Tensor:
        r"""
        byte(memory_format=torch.preserve_format) -> Tensor

        ``self.byte()`` is equivalent to ``self.to(torch.uint8)``. See :func:`to`.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.
        """
        ...
    def cauchy_(self, median: _float = 0, sigma: _float = 1, *, generator: Optional[Generator] = None) -> Tensor:
        r"""
        cauchy_(median=0, sigma=1, *, generator=None) -> Tensor

        Fills the tensor with numbers drawn from the Cauchy distribution:

        .. math::

            f(x) = \dfrac{1}{\pi} \dfrac{\sigma}{(x - \text{median})^2 + \sigma^2}

        .. note::
          Sigma (:math:`\sigma`) is used to denote the scale parameter in Cauchy distribution.
        """
        ...
    def ccol_indices(self) -> Tensor: ...
    def ceil(self) -> Tensor:
        r"""
        ceil() -> Tensor

        See :func:`torch.ceil`
        """
        ...
    def ceil_(self) -> Tensor:
        r"""
        ceil_() -> Tensor

        In-place version of :meth:`~Tensor.ceil`
        """
        ...
    def chalf(self, *, memory_format: Optional[memory_format] = None) -> Tensor:
        r"""
        chalf(memory_format=torch.preserve_format) -> Tensor

        ``self.chalf()`` is equivalent to ``self.to(torch.complex32)``. See :func:`to`.

        Args:
             memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.
        """
        ...
    def char(self) -> Tensor:
        r"""
        char(memory_format=torch.preserve_format) -> Tensor

        ``self.char()`` is equivalent to ``self.to(torch.int8)``. See :func:`to`.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.
        """
        ...
    def cholesky(self, upper: _bool = False) -> Tensor:
        r"""
        cholesky(upper=False) -> Tensor

        See :func:`torch.cholesky`
        """
        ...
    def cholesky_inverse(self, upper: _bool = False) -> Tensor:
        r"""
        cholesky_inverse(upper=False) -> Tensor

        See :func:`torch.cholesky_inverse`
        """
        ...
    def cholesky_solve(self, input2: Tensor, upper: _bool = False) -> Tensor:
        r"""
        cholesky_solve(input2, upper=False) -> Tensor

        See :func:`torch.cholesky_solve`
        """
        ...
    def chunk(self, chunks: _int, dim: _int = 0) -> Tuple[Tensor, ...]:
        r"""
        chunk(chunks, dim=0) -> List of Tensors

        See :func:`torch.chunk`
        """
        ...
    @overload
    def clamp(self, min: Optional[Tensor] = None, max: Optional[Tensor] = None) -> Tensor:
        r"""
        clamp(min=None, max=None) -> Tensor

        See :func:`torch.clamp`
        """
        ...
    @overload
    def clamp(self, min: Optional[Union[Number, _complex]] = None, max: Optional[Union[Number, _complex]] = None) -> Tensor:
        r"""
        clamp(min=None, max=None) -> Tensor

        See :func:`torch.clamp`
        """
        ...
    @overload
    def clamp_(self, min: Optional[Tensor] = None, max: Optional[Tensor] = None) -> Tensor:
        r"""
        clamp_(min=None, max=None) -> Tensor

        In-place version of :meth:`~Tensor.clamp`
        """
        ...
    @overload
    def clamp_(self, min: Optional[Union[Number, _complex]] = None, max: Optional[Union[Number, _complex]] = None) -> Tensor:
        r"""
        clamp_(min=None, max=None) -> Tensor

        In-place version of :meth:`~Tensor.clamp`
        """
        ...
    @overload
    def clamp_max(self, max: Tensor) -> Tensor: ...
    @overload
    def clamp_max(self, max: Union[Number, _complex]) -> Tensor: ...
    @overload
    def clamp_max_(self, max: Tensor) -> Tensor: ...
    @overload
    def clamp_max_(self, max: Union[Number, _complex]) -> Tensor: ...
    @overload
    def clamp_min(self, min: Tensor) -> Tensor: ...
    @overload
    def clamp_min(self, min: Union[Number, _complex]) -> Tensor: ...
    @overload
    def clamp_min_(self, min: Tensor) -> Tensor: ...
    @overload
    def clamp_min_(self, min: Union[Number, _complex]) -> Tensor: ...
    @overload
    def clip(self, min: Optional[Tensor] = None, max: Optional[Tensor] = None) -> Tensor:
        r"""
        clip(min=None, max=None) -> Tensor

        Alias for :meth:`~Tensor.clamp`.
        """
        ...
    @overload
    def clip(self, min: Optional[Union[Number, _complex]] = None, max: Optional[Union[Number, _complex]] = None) -> Tensor:
        r"""
        clip(min=None, max=None) -> Tensor

        Alias for :meth:`~Tensor.clamp`.
        """
        ...
    @overload
    def clip_(self, min: Optional[Tensor] = None, max: Optional[Tensor] = None) -> Tensor:
        r"""
        clip_(min=None, max=None) -> Tensor

        Alias for :meth:`~Tensor.clamp_`.
        """
        ...
    @overload
    def clip_(self, min: Optional[Union[Number, _complex]] = None, max: Optional[Union[Number, _complex]] = None) -> Tensor:
        r"""
        clip_(min=None, max=None) -> Tensor

        Alias for :meth:`~Tensor.clamp_`.
        """
        ...
    def clone(self, *, memory_format: Optional[memory_format] = None) -> Tensor:
        r"""
        clone(*, memory_format=torch.preserve_format) -> Tensor

        See :func:`torch.clone`
        """
        ...
    def coalesce(self) -> Tensor:
        r"""
        coalesce() -> Tensor

        Returns a coalesced copy of :attr:`self` if :attr:`self` is an
        :ref:`uncoalesced tensor <sparse-uncoalesced-coo-docs>`.

        Returns :attr:`self` if :attr:`self` is a coalesced tensor.

        .. warning::
          Throws an error if :attr:`self` is not a sparse COO tensor.
        """
        ...
    def col_indices(self) -> Tensor:
        r"""
        col_indices() -> IntTensor

        Returns the tensor containing the column indices of the :attr:`self`
        tensor when :attr:`self` is a sparse CSR tensor of layout ``sparse_csr``.
        The ``col_indices`` tensor is strictly of shape (:attr:`self`.nnz())
        and of type ``int32`` or ``int64``.  When using MKL routines such as sparse
        matrix multiplication, it is necessary to use ``int32`` indexing in order
        to avoid downcasting and potentially losing information.

        Example::
            >>> csr = torch.eye(5,5).to_sparse_csr()
            >>> csr.col_indices()
            tensor([0, 1, 2, 3, 4], dtype=torch.int32)
        """
        ...
    def conj(self) -> Tensor:
        r"""
        conj() -> Tensor

        See :func:`torch.conj`
        """
        ...
    def conj_physical(self) -> Tensor:
        r"""
        conj_physical() -> Tensor

        See :func:`torch.conj_physical`
        """
        ...
    def conj_physical_(self) -> Tensor:
        r"""
        conj_physical_() -> Tensor

        In-place version of :meth:`~Tensor.conj_physical`
        """
        ...
    def contiguous(self, memory_format=torch.contiguous_format) -> Tensor:
        r"""
        contiguous(memory_format=torch.contiguous_format) -> Tensor

        Returns a contiguous in memory tensor containing the same data as :attr:`self` tensor. If
        :attr:`self` tensor is already in the specified memory format, this function returns the
        :attr:`self` tensor.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.contiguous_format``.
        """
        ...
    def copy_(self, src: Tensor, non_blocking: _bool = False) -> Tensor:
        r"""
        copy_(src, non_blocking=False) -> Tensor

        Copies the elements from :attr:`src` into :attr:`self` tensor and returns
        :attr:`self`.

        The :attr:`src` tensor must be :ref:`broadcastable <broadcasting-semantics>`
        with the :attr:`self` tensor. It may be of a different data type or reside on a
        different device.

        Args:
            src (Tensor): the source tensor to copy from
            non_blocking (bool): if ``True`` and this copy is between CPU and GPU,
                the copy may occur asynchronously with respect to the host. For other
                cases, this argument has no effect.
        """
        ...
    @overload
    def copysign(self, other: Tensor) -> Tensor:
        r"""
        copysign(other) -> Tensor

        See :func:`torch.copysign`
        """
        ...
    @overload
    def copysign(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        copysign(other) -> Tensor

        See :func:`torch.copysign`
        """
        ...
    @overload
    def copysign_(self, other: Tensor) -> Tensor:
        r"""
        copysign_(other) -> Tensor

        In-place version of :meth:`~Tensor.copysign`
        """
        ...
    @overload
    def copysign_(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        copysign_(other) -> Tensor

        In-place version of :meth:`~Tensor.copysign`
        """
        ...
    def corrcoef(self) -> Tensor:
        r"""
        corrcoef() -> Tensor

        See :func:`torch.corrcoef`
        """
        ...
    def cos(self) -> Tensor:
        r"""
        cos() -> Tensor

        See :func:`torch.cos`
        """
        ...
    def cos_(self) -> Tensor:
        r"""
        cos_() -> Tensor

        In-place version of :meth:`~Tensor.cos`
        """
        ...
    def cosh(self) -> Tensor:
        r"""
        cosh() -> Tensor

        See :func:`torch.cosh`
        """
        ...
    def cosh_(self) -> Tensor:
        r"""
        cosh_() -> Tensor

        In-place version of :meth:`~Tensor.cosh`
        """
        ...
    @overload
    def count_nonzero(self, dim: Optional[_int] = None) -> Tensor:
        r"""
        count_nonzero(dim=None) -> Tensor

        See :func:`torch.count_nonzero`
        """
        ...
    @overload
    def count_nonzero(self, dim: _size) -> Tensor:
        r"""
        count_nonzero(dim=None) -> Tensor

        See :func:`torch.count_nonzero`
        """
        ...
    @overload
    def count_nonzero(self, *dim: _int) -> Tensor:
        r"""
        count_nonzero(dim=None) -> Tensor

        See :func:`torch.count_nonzero`
        """
        ...
    def cov(self, *, correction: _int = 1, fweights: Optional[Tensor] = None, aweights: Optional[Tensor] = None) -> Tensor:
        r"""
        cov(*, correction=1, fweights=None, aweights=None) -> Tensor

        See :func:`torch.cov`
        """
        ...
    def cpu(self, memory_format: torch.memory_format = torch.preserve_format) -> Tensor:
        r"""
        cpu(memory_format=torch.preserve_format) -> Tensor

        Returns a copy of this object in CPU memory.

        If this object is already in CPU memory and on the correct device,
        then no copy is performed and the original object is returned.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.
        """
        ...
    def cross(self, other: Tensor, dim: Optional[_int] = None) -> Tensor:
        r"""
        cross(other, dim=None) -> Tensor

        See :func:`torch.cross`
        """
        ...
    def crow_indices(self) -> Tensor:
        r"""
        crow_indices() -> IntTensor

        Returns the tensor containing the compressed row indices of the :attr:`self`
        tensor when :attr:`self` is a sparse CSR tensor of layout ``sparse_csr``.
        The ``crow_indices`` tensor is strictly of shape (:attr:`self`.size(0) + 1)
        and of type ``int32`` or ``int64``. When using MKL routines such as sparse
        matrix multiplication, it is necessary to use ``int32`` indexing in order
        to avoid downcasting and potentially losing information.

        Example::
            >>> csr = torch.eye(5,5).to_sparse_csr()
            >>> csr.crow_indices()
            tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32)
        """
        ...
    def cuda(self, device: Optional[Union[_device, _int, str]] = None, non_blocking: _bool = False, memory_format: torch.memory_format = torch.preserve_format) -> Tensor:
        r"""
        cuda(device=None, non_blocking=False, memory_format=torch.preserve_format) -> Tensor

        Returns a copy of this object in CUDA memory.

        If this object is already in CUDA memory and on the correct device,
        then no copy is performed and the original object is returned.

        Args:
            device (:class:`torch.device`): The destination GPU device.
                Defaults to the current CUDA device.
            non_blocking (bool): If ``True`` and the source is in pinned memory,
                the copy will be asynchronous with respect to the host.
                Otherwise, the argument has no effect. Default: ``False``.
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.
        """
        ...
    @overload
    def cummax(self, dim: _int) -> torch.return_types.cummax:
        r"""
        cummax(dim) -> (Tensor, Tensor)

        See :func:`torch.cummax`
        """
        ...
    @overload
    def cummax(self, dim: Union[str, ellipsis, None]) -> torch.return_types.cummax:
        r"""
        cummax(dim) -> (Tensor, Tensor)

        See :func:`torch.cummax`
        """
        ...
    @overload
    def cummin(self, dim: _int) -> torch.return_types.cummin:
        r"""
        cummin(dim) -> (Tensor, Tensor)

        See :func:`torch.cummin`
        """
        ...
    @overload
    def cummin(self, dim: Union[str, ellipsis, None]) -> torch.return_types.cummin:
        r"""
        cummin(dim) -> (Tensor, Tensor)

        See :func:`torch.cummin`
        """
        ...
    @overload
    def cumprod(self, dim: _int, *, dtype: Optional[_dtype] = None) -> Tensor:
        r"""
        cumprod(dim, dtype=None) -> Tensor

        See :func:`torch.cumprod`
        """
        ...
    @overload
    def cumprod(self, dim: Union[str, ellipsis, None], *, dtype: Optional[_dtype] = None) -> Tensor:
        r"""
        cumprod(dim, dtype=None) -> Tensor

        See :func:`torch.cumprod`
        """
        ...
    @overload
    def cumprod_(self, dim: _int, *, dtype: Optional[_dtype] = None) -> Tensor:
        r"""
        cumprod_(dim, dtype=None) -> Tensor

        In-place version of :meth:`~Tensor.cumprod`
        """
        ...
    @overload
    def cumprod_(self, dim: Union[str, ellipsis, None], *, dtype: Optional[_dtype] = None) -> Tensor:
        r"""
        cumprod_(dim, dtype=None) -> Tensor

        In-place version of :meth:`~Tensor.cumprod`
        """
        ...
    @overload
    def cumsum(self, dim: _int, *, dtype: Optional[_dtype] = None) -> Tensor:
        r"""
        cumsum(dim, dtype=None) -> Tensor

        See :func:`torch.cumsum`
        """
        ...
    @overload
    def cumsum(self, dim: Union[str, ellipsis, None], *, dtype: Optional[_dtype] = None) -> Tensor:
        r"""
        cumsum(dim, dtype=None) -> Tensor

        See :func:`torch.cumsum`
        """
        ...
    @overload
    def cumsum_(self, dim: _int, *, dtype: Optional[_dtype] = None) -> Tensor:
        r"""
        cumsum_(dim, dtype=None) -> Tensor

        In-place version of :meth:`~Tensor.cumsum`
        """
        ...
    @overload
    def cumsum_(self, dim: Union[str, ellipsis, None], *, dtype: Optional[_dtype] = None) -> Tensor:
        r"""
        cumsum_(dim, dtype=None) -> Tensor

        In-place version of :meth:`~Tensor.cumsum`
        """
        ...
    def data_ptr(self) -> _int:
        r"""
        data_ptr() -> int

        Returns the address of the first element of :attr:`self` tensor.
        """
        ...
    def deg2rad(self) -> Tensor:
        r"""
        deg2rad() -> Tensor

        See :func:`torch.deg2rad`
        """
        ...
    def deg2rad_(self) -> Tensor:
        r"""
        deg2rad_() -> Tensor

        In-place version of :meth:`~Tensor.deg2rad`
        """
        ...
    def dense_dim(self) -> _int:
        r"""
        dense_dim() -> int

        Return the number of dense dimensions in a :ref:`sparse tensor <sparse-docs>` :attr:`self`.

        .. note::
          Returns ``len(self.shape)`` if :attr:`self` is not a sparse tensor.

        See also :meth:`Tensor.sparse_dim` and :ref:`hybrid tensors <sparse-hybrid-coo-docs>`.
        """
        ...
    def dequantize(self) -> Tensor:
        r"""
        dequantize() -> Tensor

        Given a quantized Tensor, dequantize it and return the dequantized float Tensor.
        """
        ...
    def det(self) -> Tensor:
        r"""
        det() -> Tensor

        See :func:`torch.det`
        """
        ...
    def detach(self) -> Tensor: ...
    def detach_(self) -> Tensor: ...
    def diag(self, diagonal: _int = 0) -> Tensor:
        r"""
        diag(diagonal=0) -> Tensor

        See :func:`torch.diag`
        """
        ...
    def diag_embed(self, offset: _int = 0, dim1: _int = -2, dim2: _int = -1) -> Tensor:
        r"""
        diag_embed(offset=0, dim1=-2, dim2=-1) -> Tensor

        See :func:`torch.diag_embed`
        """
        ...
    def diagflat(self, offset: _int = 0) -> Tensor:
        r"""
        diagflat(offset=0) -> Tensor

        See :func:`torch.diagflat`
        """
        ...
    @overload
    def diagonal(self, *, outdim: Union[str, ellipsis, None], dim1: Union[str, ellipsis, None], dim2: Union[str, ellipsis, None], offset: _int = 0) -> Tensor:
        r"""
        diagonal(offset=0, dim1=0, dim2=1) -> Tensor

        See :func:`torch.diagonal`
        """
        ...
    @overload
    def diagonal(self, offset: _int = 0, dim1: _int = 0, dim2: _int = 1) -> Tensor:
        r"""
        diagonal(offset=0, dim1=0, dim2=1) -> Tensor

        See :func:`torch.diagonal`
        """
        ...
    def diagonal_scatter(self, src: Tensor, offset: _int = 0, dim1: _int = 0, dim2: _int = 1) -> Tensor:
        r"""
        diagonal_scatter(src, offset=0, dim1=0, dim2=1) -> Tensor

        See :func:`torch.diagonal_scatter`
        """
        ...
    def diff(self, n: _int = 1, dim: _int = -1, prepend: Optional[Tensor] = None, append: Optional[Tensor] = None) -> Tensor:
        r"""
        diff(n=1, dim=-1, prepend=None, append=None) -> Tensor

        See :func:`torch.diff`
        """
        ...
    def digamma(self) -> Tensor:
        r"""
        digamma() -> Tensor

        See :func:`torch.digamma`
        """
        ...
    def digamma_(self) -> Tensor:
        r"""
        digamma_() -> Tensor

        In-place version of :meth:`~Tensor.digamma`
        """
        ...
    def dim(self) -> _int:
        r"""
        dim() -> int

        Returns the number of dimensions of :attr:`self` tensor.
        """
        ...
    def dist(self, other: Tensor, p: Union[Number, _complex] = 2) -> Tensor:
        r"""
        dist(other, p=2) -> Tensor

        See :func:`torch.dist`
        """
        ...
    def div(self, other: Union[Tensor, Number], *, rounding_mode: Optional[str] = None) -> Tensor:
        r"""
        div(value, *, rounding_mode=None) -> Tensor

        See :func:`torch.div`
        """
        ...
    def div_(self, other: Union[Tensor, Number], *, rounding_mode: Optional[str] = None) -> Tensor:
        r"""
        div_(value, *, rounding_mode=None) -> Tensor

        In-place version of :meth:`~Tensor.div`
        """
        ...
    @overload
    def divide(self, other: Tensor) -> Tensor:
        r"""
        divide(value, *, rounding_mode=None) -> Tensor

        See :func:`torch.divide`
        """
        ...
    @overload
    def divide(self, other: Tensor, *, rounding_mode: Optional[str]) -> Tensor:
        r"""
        divide(value, *, rounding_mode=None) -> Tensor

        See :func:`torch.divide`
        """
        ...
    @overload
    def divide(self, other: Union[Number, _complex], *, rounding_mode: Optional[str]) -> Tensor:
        r"""
        divide(value, *, rounding_mode=None) -> Tensor

        See :func:`torch.divide`
        """
        ...
    @overload
    def divide(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        divide(value, *, rounding_mode=None) -> Tensor

        See :func:`torch.divide`
        """
        ...
    @overload
    def divide_(self, other: Tensor) -> Tensor:
        r"""
        divide_(value, *, rounding_mode=None) -> Tensor

        In-place version of :meth:`~Tensor.divide`
        """
        ...
    @overload
    def divide_(self, other: Tensor, *, rounding_mode: Optional[str]) -> Tensor:
        r"""
        divide_(value, *, rounding_mode=None) -> Tensor

        In-place version of :meth:`~Tensor.divide`
        """
        ...
    @overload
    def divide_(self, other: Union[Number, _complex], *, rounding_mode: Optional[str]) -> Tensor:
        r"""
        divide_(value, *, rounding_mode=None) -> Tensor

        In-place version of :meth:`~Tensor.divide`
        """
        ...
    @overload
    def divide_(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        divide_(value, *, rounding_mode=None) -> Tensor

        In-place version of :meth:`~Tensor.divide`
        """
        ...
    def dot(self, tensor: Tensor) -> Tensor:
        r"""
        dot(other) -> Tensor

        See :func:`torch.dot`
        """
        ...
    def double(self) -> Tensor:
        r"""
        double(memory_format=torch.preserve_format) -> Tensor

        ``self.double()`` is equivalent to ``self.to(torch.float64)``. See :func:`to`.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.
        """
        ...
    @overload
    def dsplit(self, sections: _int) -> Tuple[Tensor, ...]:
        r"""
        dsplit(split_size_or_sections) -> List of Tensors

        See :func:`torch.dsplit`
        """
        ...
    @overload
    def dsplit(self, indices: _size) -> Tuple[Tensor, ...]:
        r"""
        dsplit(split_size_or_sections) -> List of Tensors

        See :func:`torch.dsplit`
        """
        ...
    @overload
    def dsplit(self, *indices: _int) -> Tuple[Tensor, ...]:
        r"""
        dsplit(split_size_or_sections) -> List of Tensors

        See :func:`torch.dsplit`
        """
        ...
    def element_size(self) -> _int:
        r"""
        element_size() -> int

        Returns the size in bytes of an individual element.

        Example::

            >>> torch.tensor([]).element_size()
            4
            >>> torch.tensor([], dtype=torch.uint8).element_size()
            1
        """
        ...
    @overload
    def eq(self, other: Tensor) -> Tensor:
        r"""
        eq(other) -> Tensor

        See :func:`torch.eq`
        """
        ...
    @overload
    def eq(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        eq(other) -> Tensor

        See :func:`torch.eq`
        """
        ...
    @overload
    def eq_(self, other: Tensor) -> Tensor:
        r"""
        eq_(other) -> Tensor

        In-place version of :meth:`~Tensor.eq`
        """
        ...
    @overload
    def eq_(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        eq_(other) -> Tensor

        In-place version of :meth:`~Tensor.eq`
        """
        ...
    def equal(self, other: Tensor) -> _bool:
        r"""
        equal(other) -> bool

        See :func:`torch.equal`
        """
        ...
    def erf(self) -> Tensor:
        r"""
        erf() -> Tensor

        See :func:`torch.erf`
        """
        ...
    def erf_(self) -> Tensor:
        r"""
        erf_() -> Tensor

        In-place version of :meth:`~Tensor.erf`
        """
        ...
    def erfc(self) -> Tensor:
        r"""
        erfc() -> Tensor

        See :func:`torch.erfc`
        """
        ...
    def erfc_(self) -> Tensor:
        r"""
        erfc_() -> Tensor

        In-place version of :meth:`~Tensor.erfc`
        """
        ...
    def erfinv(self) -> Tensor:
        r"""
        erfinv() -> Tensor

        See :func:`torch.erfinv`
        """
        ...
    def erfinv_(self) -> Tensor:
        r"""
        erfinv_() -> Tensor

        In-place version of :meth:`~Tensor.erfinv`
        """
        ...
    def exp(self) -> Tensor:
        r"""
        exp() -> Tensor

        See :func:`torch.exp`
        """
        ...
    def exp2(self) -> Tensor:
        r"""
        exp2() -> Tensor

        See :func:`torch.exp2`
        """
        ...
    def exp2_(self) -> Tensor:
        r"""
        exp2_() -> Tensor

        In-place version of :meth:`~Tensor.exp2`
        """
        ...
    def exp_(self) -> Tensor:
        r"""
        exp_() -> Tensor

        In-place version of :meth:`~Tensor.exp`
        """
        ...
    @overload
    def expand(self, size: Sequence[Union[_int, SymInt]], *, implicit: _bool = False) -> Tensor:
        r"""
        expand(*sizes) -> Tensor

        Returns a new view of the :attr:`self` tensor with singleton dimensions expanded
        to a larger size.

        Passing -1 as the size for a dimension means not changing the size of
        that dimension.

        Tensor can be also expanded to a larger number of dimensions, and the
        new ones will be appended at the front. For the new dimensions, the
        size cannot be set to -1.

        Expanding a tensor does not allocate new memory, but only creates a
        new view on the existing tensor where a dimension of size one is
        expanded to a larger size by setting the ``stride`` to 0. Any dimension
        of size 1 can be expanded to an arbitrary value without allocating new
        memory.

        Args:
            *sizes (torch.Size or int...): the desired expanded size

        .. warning::

            More than one element of an expanded tensor may refer to a single
            memory location. As a result, in-place operations (especially ones that
            are vectorized) may result in incorrect behavior. If you need to write
            to the tensors, please clone them first.

        Example::

            >>> x = torch.tensor([[1], [2], [3]])
            >>> x.size()
            torch.Size([3, 1])
            >>> x.expand(3, 4)
            tensor([[ 1,  1,  1,  1],
                    [ 2,  2,  2,  2],
                    [ 3,  3,  3,  3]])
            >>> x.expand(-1, 4)   # -1 means not changing the size of that dimension
            tensor([[ 1,  1,  1,  1],
                    [ 2,  2,  2,  2],
                    [ 3,  3,  3,  3]])
        """
        ...
    @overload
    def expand(self, *size: _int, implicit: _bool = False) -> Tensor:
        r"""
        expand(*sizes) -> Tensor

        Returns a new view of the :attr:`self` tensor with singleton dimensions expanded
        to a larger size.

        Passing -1 as the size for a dimension means not changing the size of
        that dimension.

        Tensor can be also expanded to a larger number of dimensions, and the
        new ones will be appended at the front. For the new dimensions, the
        size cannot be set to -1.

        Expanding a tensor does not allocate new memory, but only creates a
        new view on the existing tensor where a dimension of size one is
        expanded to a larger size by setting the ``stride`` to 0. Any dimension
        of size 1 can be expanded to an arbitrary value without allocating new
        memory.

        Args:
            *sizes (torch.Size or int...): the desired expanded size

        .. warning::

            More than one element of an expanded tensor may refer to a single
            memory location. As a result, in-place operations (especially ones that
            are vectorized) may result in incorrect behavior. If you need to write
            to the tensors, please clone them first.

        Example::

            >>> x = torch.tensor([[1], [2], [3]])
            >>> x.size()
            torch.Size([3, 1])
            >>> x.expand(3, 4)
            tensor([[ 1,  1,  1,  1],
                    [ 2,  2,  2,  2],
                    [ 3,  3,  3,  3]])
            >>> x.expand(-1, 4)   # -1 means not changing the size of that dimension
            tensor([[ 1,  1,  1,  1],
                    [ 2,  2,  2,  2],
                    [ 3,  3,  3,  3]])
        """
        ...
    def expand_as(self, other: Tensor) -> Tensor:
        r"""
        expand_as(other) -> Tensor

        Expand this tensor to the same size as :attr:`other`.
        ``self.expand_as(other)`` is equivalent to ``self.expand(other.size())``.

        Please see :meth:`~Tensor.expand` for more information about ``expand``.

        Args:
            other (:class:`torch.Tensor`): The result tensor has the same size
                as :attr:`other`.
        """
        ...
    def expm1(self) -> Tensor:
        r"""
        expm1() -> Tensor

        See :func:`torch.expm1`
        """
        ...
    def expm1_(self) -> Tensor:
        r"""
        expm1_() -> Tensor

        In-place version of :meth:`~Tensor.expm1`
        """
        ...
    def exponential_(self, lambd: _float = 1, *, generator: Optional[Generator] = None) -> Tensor:
        r"""
        exponential_(lambd=1, *, generator=None) -> Tensor

        Fills :attr:`self` tensor with elements drawn from the PDF (probability density function):

        .. math::

            f(x) = \lambda e^{-\lambda x}, x > 0

        .. note::
          In probability theory, exponential distribution is supported on interval [0, :math:`\inf`) (i.e., :math:`x >= 0`)
          implying that zero can be sampled from the exponential distribution.
          However, :func:`torch.Tensor.exponential_` does not sample zero,
          which means that its actual support is the interval (0, :math:`\inf`).

          Note that :func:`torch.distributions.exponential.Exponential` is supported on the interval [0, :math:`\inf`) and can sample zero.
        """
        ...
    @overload
    def fill_(self, value: Tensor) -> Tensor:
        r"""
        fill_(value) -> Tensor

        Fills :attr:`self` tensor with the specified value.
        """
        ...
    @overload
    def fill_(self, value: Union[Number, _complex]) -> Tensor:
        r"""
        fill_(value) -> Tensor

        Fills :attr:`self` tensor with the specified value.
        """
        ...
    def fill_diagonal_(self, fill_value: Union[Number, _complex], wrap: _bool = False) -> Tensor:
        r"""
        fill_diagonal_(fill_value, wrap=False) -> Tensor

        Fill the main diagonal of a tensor that has at least 2-dimensions.
        When dims>2, all dimensions of input must be of equal length.
        This function modifies the input tensor in-place, and returns the input tensor.

        Arguments:
            fill_value (Scalar): the fill value
            wrap (bool): the diagonal 'wrapped' after N columns for tall matrices.

        Example::

            >>> a = torch.zeros(3, 3)
            >>> a.fill_diagonal_(5)
            tensor([[5., 0., 0.],
                    [0., 5., 0.],
                    [0., 0., 5.]])
            >>> b = torch.zeros(7, 3)
            >>> b.fill_diagonal_(5)
            tensor([[5., 0., 0.],
                    [0., 5., 0.],
                    [0., 0., 5.],
                    [0., 0., 0.],
                    [0., 0., 0.],
                    [0., 0., 0.],
                    [0., 0., 0.]])
            >>> c = torch.zeros(7, 3)
            >>> c.fill_diagonal_(5, wrap=True)
            tensor([[5., 0., 0.],
                    [0., 5., 0.],
                    [0., 0., 5.],
                    [0., 0., 0.],
                    [5., 0., 0.],
                    [0., 5., 0.],
                    [0., 0., 5.]])
        """
        ...
    def fix(self) -> Tensor:
        r"""
        fix() -> Tensor

        See :func:`torch.fix`.
        """
        ...
    def fix_(self) -> Tensor:
        r"""
        fix_() -> Tensor

        In-place version of :meth:`~Tensor.fix`
        """
        ...
    @overload
    def flatten(self, start_dim: _int = 0, end_dim: _int = -1) -> Tensor:
        r"""
        flatten(start_dim=0, end_dim=-1) -> Tensor

        See :func:`torch.flatten`
        """
        ...
    @overload
    def flatten(self, start_dim: _int, end_dim: _int, out_dim: Union[str, ellipsis, None]) -> Tensor:
        r"""
        flatten(start_dim=0, end_dim=-1) -> Tensor

        See :func:`torch.flatten`
        """
        ...
    @overload
    def flatten(self, start_dim: Union[str, ellipsis, None], end_dim: Union[str, ellipsis, None], out_dim: Union[str, ellipsis, None]) -> Tensor:
        r"""
        flatten(start_dim=0, end_dim=-1) -> Tensor

        See :func:`torch.flatten`
        """
        ...
    @overload
    def flatten(self, dims: Sequence[Union[str, ellipsis, None]], out_dim: Union[str, ellipsis, None]) -> Tensor:
        r"""
        flatten(start_dim=0, end_dim=-1) -> Tensor

        See :func:`torch.flatten`
        """
        ...
    @overload
    def flip(self, dims: _size) -> Tensor:
        r"""
        flip(dims) -> Tensor

        See :func:`torch.flip`
        """
        ...
    @overload
    def flip(self, *dims: _int) -> Tensor:
        r"""
        flip(dims) -> Tensor

        See :func:`torch.flip`
        """
        ...
    def fliplr(self) -> Tensor:
        r"""
        fliplr() -> Tensor

        See :func:`torch.fliplr`
        """
        ...
    def flipud(self) -> Tensor:
        r"""
        flipud() -> Tensor

        See :func:`torch.flipud`
        """
        ...
    def float(self) -> Tensor:
        r"""
        float(memory_format=torch.preserve_format) -> Tensor

        ``self.float()`` is equivalent to ``self.to(torch.float32)``. See :func:`to`.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.
        """
        ...
    @overload
    def float_power(self, exponent: Tensor) -> Tensor:
        r"""
        float_power(exponent) -> Tensor

        See :func:`torch.float_power`
        """
        ...
    @overload
    def float_power(self, exponent: Union[Number, _complex]) -> Tensor:
        r"""
        float_power(exponent) -> Tensor

        See :func:`torch.float_power`
        """
        ...
    @overload
    def float_power_(self, exponent: Tensor) -> Tensor:
        r"""
        float_power_(exponent) -> Tensor

        In-place version of :meth:`~Tensor.float_power`
        """
        ...
    @overload
    def float_power_(self, exponent: Union[Number, _complex]) -> Tensor:
        r"""
        float_power_(exponent) -> Tensor

        In-place version of :meth:`~Tensor.float_power`
        """
        ...
    def floor(self) -> Tensor:
        r"""
        floor() -> Tensor

        See :func:`torch.floor`
        """
        ...
    def floor_(self) -> Tensor:
        r"""
        floor_() -> Tensor

        In-place version of :meth:`~Tensor.floor`
        """
        ...
    def floor_divide(self, other: Union[Tensor, Number, torch.SymInt, torch.SymFloat], *, out: Optional[Tensor] = None) -> Tensor:
        r"""
        floor_divide(value) -> Tensor

        See :func:`torch.floor_divide`
        """
        ...
    def floor_divide_(self, other: Union[Tensor, Number, torch.SymInt, torch.SymFloat]) -> Tensor:
        r"""
        floor_divide_(value) -> Tensor

        In-place version of :meth:`~Tensor.floor_divide`
        """
        ...
    def fmax(self, other: Tensor) -> Tensor:
        r"""
        fmax(other) -> Tensor

        See :func:`torch.fmax`
        """
        ...
    def fmin(self, other: Tensor) -> Tensor:
        r"""
        fmin(other) -> Tensor

        See :func:`torch.fmin`
        """
        ...
    @overload
    def fmod(self, other: Tensor) -> Tensor:
        r"""
        fmod(divisor) -> Tensor

        See :func:`torch.fmod`
        """
        ...
    @overload
    def fmod(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        fmod(divisor) -> Tensor

        See :func:`torch.fmod`
        """
        ...
    @overload
    def fmod_(self, other: Tensor) -> Tensor:
        r"""
        fmod_(divisor) -> Tensor

        In-place version of :meth:`~Tensor.fmod`
        """
        ...
    @overload
    def fmod_(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        fmod_(divisor) -> Tensor

        In-place version of :meth:`~Tensor.fmod`
        """
        ...
    def frac(self) -> Tensor:
        r"""
        frac() -> Tensor

        See :func:`torch.frac`
        """
        ...
    def frac_(self) -> Tensor:
        r"""
        frac_() -> Tensor

        In-place version of :meth:`~Tensor.frac`
        """
        ...
    def frexp(self) -> torch.return_types.frexp:
        r"""
        frexp(input) -> (Tensor mantissa, Tensor exponent)

        See :func:`torch.frexp`
        """
        ...
    @overload
    def gather(self, dim: _int, index: Tensor, *, sparse_grad: _bool = False) -> Tensor:
        r"""
        gather(dim, index) -> Tensor

        See :func:`torch.gather`
        """
        ...
    @overload
    def gather(self, dim: Union[str, ellipsis, None], index: Tensor, *, sparse_grad: _bool = False) -> Tensor:
        r"""
        gather(dim, index) -> Tensor

        See :func:`torch.gather`
        """
        ...
    def gcd(self, other: Tensor) -> Tensor:
        r"""
        gcd(other) -> Tensor

        See :func:`torch.gcd`
        """
        ...
    def gcd_(self, other: Tensor) -> Tensor:
        r"""
        gcd_(other) -> Tensor

        In-place version of :meth:`~Tensor.gcd`
        """
        ...
    @overload
    def ge(self, other: Tensor) -> Tensor:
        r"""
        ge(other) -> Tensor

        See :func:`torch.ge`.
        """
        ...
    @overload
    def ge(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        ge(other) -> Tensor

        See :func:`torch.ge`.
        """
        ...
    @overload
    def ge_(self, other: Tensor) -> Tensor:
        r"""
        ge_(other) -> Tensor

        In-place version of :meth:`~Tensor.ge`.
        """
        ...
    @overload
    def ge_(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        ge_(other) -> Tensor

        In-place version of :meth:`~Tensor.ge`.
        """
        ...
    def geometric_(self, p: _float, *, generator: Optional[Generator] = None) -> Tensor:
        r"""
        geometric_(p, *, generator=None) -> Tensor

        Fills :attr:`self` tensor with elements drawn from the geometric distribution:

        .. math::

            P(X=k) = (1 - p)^{k - 1} p, k = 1, 2, ...

        .. note::
          :func:`torch.Tensor.geometric_` `k`-th trial is the first success hence draws samples in :math:`\{1, 2, \ldots\}`, whereas
          :func:`torch.distributions.geometric.Geometric` :math:`(k+1)`-th trial is the first success
          hence draws samples in :math:`\{0, 1, \ldots\}`.
        """
        ...
    def geqrf(self) -> torch.return_types.geqrf:
        r"""
        geqrf() -> (Tensor, Tensor)

        See :func:`torch.geqrf`
        """
        ...
    def ger(self, vec2: Tensor) -> Tensor:
        r"""
        ger(vec2) -> Tensor

        See :func:`torch.ger`
        """
        ...
    def get_device(self) -> _int:
        r"""
        get_device() -> Device ordinal (Integer)

        For CUDA tensors, this function returns the device ordinal of the GPU on which the tensor resides.
        For CPU tensors, this function returns `-1`.

        Example::

            >>> x = torch.randn(3, 4, 5, device='cuda:0')
            >>> x.get_device()
            0
            >>> x.cpu().get_device()
            -1
        """
        ...
    @overload
    def greater(self, other: Tensor) -> Tensor:
        r"""
        greater(other) -> Tensor

        See :func:`torch.greater`.
        """
        ...
    @overload
    def greater(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        greater(other) -> Tensor

        See :func:`torch.greater`.
        """
        ...
    @overload
    def greater_(self, other: Tensor) -> Tensor:
        r"""
        greater_(other) -> Tensor

        In-place version of :meth:`~Tensor.greater`.
        """
        ...
    @overload
    def greater_(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        greater_(other) -> Tensor

        In-place version of :meth:`~Tensor.greater`.
        """
        ...
    @overload
    def greater_equal(self, other: Tensor) -> Tensor:
        r"""
        greater_equal(other) -> Tensor

        See :func:`torch.greater_equal`.
        """
        ...
    @overload
    def greater_equal(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        greater_equal(other) -> Tensor

        See :func:`torch.greater_equal`.
        """
        ...
    @overload
    def greater_equal_(self, other: Tensor) -> Tensor:
        r"""
        greater_equal_(other) -> Tensor

        In-place version of :meth:`~Tensor.greater_equal`.
        """
        ...
    @overload
    def greater_equal_(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        greater_equal_(other) -> Tensor

        In-place version of :meth:`~Tensor.greater_equal`.
        """
        ...
    @overload
    def gt(self, other: Tensor) -> Tensor:
        r"""
        gt(other) -> Tensor

        See :func:`torch.gt`.
        """
        ...
    @overload
    def gt(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        gt(other) -> Tensor

        See :func:`torch.gt`.
        """
        ...
    @overload
    def gt_(self, other: Tensor) -> Tensor:
        r"""
        gt_(other) -> Tensor

        In-place version of :meth:`~Tensor.gt`.
        """
        ...
    @overload
    def gt_(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        gt_(other) -> Tensor

        In-place version of :meth:`~Tensor.gt`.
        """
        ...
    def half(self) -> Tensor:
        r"""
        half(memory_format=torch.preserve_format) -> Tensor

        ``self.half()`` is equivalent to ``self.to(torch.float16)``. See :func:`to`.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.
        """
        ...
    def hardshrink(self, lambd: Union[Number, _complex] = 0.5) -> Tensor:
        r"""
        hardshrink(lambd=0.5) -> Tensor

        See :func:`torch.nn.functional.hardshrink`
        """
        ...
    def has_names(self) -> _bool:
        r"""
        Is ``True`` if any of this tensor's dimensions are named. Otherwise, is ``False``.
        """
        ...
    def heaviside(self, values: Tensor) -> Tensor:
        r"""
        heaviside(values) -> Tensor

        See :func:`torch.heaviside`
        """
        ...
    def heaviside_(self, values: Tensor) -> Tensor:
        r"""
        heaviside_(values) -> Tensor

        In-place version of :meth:`~Tensor.heaviside`
        """
        ...
    def histc(self, bins: _int = 100, min: Union[Number, _complex] = 0, max: Union[Number, _complex] = 0) -> Tensor:
        r"""
        histc(bins=100, min=0, max=0) -> Tensor

        See :func:`torch.histc`
        """
        ...
    @overload
    def histogram(self, bins: Tensor, *, weight: Optional[Tensor] = None, density: _bool = False) -> torch.return_types.histogram:
        r"""
        histogram(input, bins, *, range=None, weight=None, density=False) -> (Tensor, Tensor)

        See :func:`torch.histogram`
        """
        ...
    @overload
    def histogram(self, bins: _int = 100, *, range: Optional[Sequence[_float]] = None, weight: Optional[Tensor] = None, density: _bool = False) -> torch.return_types.histogram:
        r"""
        histogram(input, bins, *, range=None, weight=None, density=False) -> (Tensor, Tensor)

        See :func:`torch.histogram`
        """
        ...
    @overload
    def hsplit(self, sections: _int) -> Tuple[Tensor, ...]:
        r"""
        hsplit(split_size_or_sections) -> List of Tensors

        See :func:`torch.hsplit`
        """
        ...
    @overload
    def hsplit(self, indices: _size) -> Tuple[Tensor, ...]:
        r"""
        hsplit(split_size_or_sections) -> List of Tensors

        See :func:`torch.hsplit`
        """
        ...
    @overload
    def hsplit(self, *indices: _int) -> Tuple[Tensor, ...]:
        r"""
        hsplit(split_size_or_sections) -> List of Tensors

        See :func:`torch.hsplit`
        """
        ...
    def hypot(self, other: Tensor) -> Tensor:
        r"""
        hypot(other) -> Tensor

        See :func:`torch.hypot`
        """
        ...
    def hypot_(self, other: Tensor) -> Tensor:
        r"""
        hypot_(other) -> Tensor

        In-place version of :meth:`~Tensor.hypot`
        """
        ...
    def i0(self) -> Tensor:
        r"""
        i0() -> Tensor

        See :func:`torch.i0`
        """
        ...
    def i0_(self) -> Tensor:
        r"""
        i0_() -> Tensor

        In-place version of :meth:`~Tensor.i0`
        """
        ...
    def igamma(self, other: Tensor) -> Tensor:
        r"""
        igamma(other) -> Tensor

        See :func:`torch.igamma`
        """
        ...
    def igamma_(self, other: Tensor) -> Tensor:
        r"""
        igamma_(other) -> Tensor

        In-place version of :meth:`~Tensor.igamma`
        """
        ...
    def igammac(self, other: Tensor) -> Tensor:
        r"""
        igammac(other) -> Tensor
        See :func:`torch.igammac`
        """
        ...
    def igammac_(self, other: Tensor) -> Tensor:
        r"""
        igammac_(other) -> Tensor
        In-place version of :meth:`~Tensor.igammac`
        """
        ...
    @overload
    def index_add(self, dim: _int, index: Tensor, source: Tensor, *, alpha: Union[Number, _complex] = 1) -> Tensor:
        r"""
        index_add(dim, index, source, *, alpha=1) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.index_add_`.
        """
        ...
    @overload
    def index_add(self, dim: Union[str, ellipsis, None], index: Tensor, source: Tensor, *, alpha: Union[Number, _complex] = 1) -> Tensor:
        r"""
        index_add(dim, index, source, *, alpha=1) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.index_add_`.
        """
        ...
    def index_add_(self, dim: _int, index: Tensor, source: Tensor, *, alpha: Union[Number, _complex] = 1) -> Tensor:
        r"""
        index_add_(dim, index, source, *, alpha=1) -> Tensor

        Accumulate the elements of :attr:`alpha` times ``source`` into the :attr:`self`
        tensor by adding to the indices in the order given in :attr:`index`. For example,
        if ``dim == 0``, ``index[i] == j``, and ``alpha=-1``, then the ``i``\ th row of
        ``source`` is subtracted from the ``j``\ th row of :attr:`self`.

        The :attr:`dim`\ th dimension of ``source`` must have the same size as the
        length of :attr:`index` (which must be a vector), and all other dimensions must
        match :attr:`self`, or an error will be raised.

        For a 3-D tensor the output is given as::

            self[index[i], :, :] += alpha * src[i, :, :]  # if dim == 0
            self[:, index[i], :] += alpha * src[:, i, :]  # if dim == 1
            self[:, :, index[i]] += alpha * src[:, :, i]  # if dim == 2

        Note:
            This operation may behave nondeterministically when given tensors on a CUDA device. See :doc:`/notes/randomness` for more information.

        Args:
            dim (int): dimension along which to index
            index (Tensor): indices of ``source`` to select from,
                    should have dtype either `torch.int64` or `torch.int32`
            source (Tensor): the tensor containing values to add

        Keyword args:
            alpha (Number): the scalar multiplier for ``source``

        Example::

            >>> x = torch.ones(5, 3)
            >>> t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
            >>> index = torch.tensor([0, 4, 2])
            >>> x.index_add_(0, index, t)
            tensor([[  2.,   3.,   4.],
                    [  1.,   1.,   1.],
                    [  8.,   9.,  10.],
                    [  1.,   1.,   1.],
                    [  5.,   6.,   7.]])
            >>> x.index_add_(0, index, t, alpha=-1)
            tensor([[  1.,   1.,   1.],
                    [  1.,   1.,   1.],
                    [  1.,   1.,   1.],
                    [  1.,   1.,   1.],
                    [  1.,   1.,   1.]])
        """
        ...
    @overload
    def index_copy(self, dim: _int, index: Tensor, source: Tensor) -> Tensor:
        r"""
        index_copy(dim, index, tensor2) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.index_copy_`.
        """
        ...
    @overload
    def index_copy(self, dim: Union[str, ellipsis, None], index: Tensor, source: Tensor) -> Tensor:
        r"""
        index_copy(dim, index, tensor2) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.index_copy_`.
        """
        ...
    @overload
    def index_copy_(self, dim: _int, index: Tensor, source: Tensor) -> Tensor:
        r"""
        index_copy_(dim, index, tensor) -> Tensor

        Copies the elements of :attr:`tensor` into the :attr:`self` tensor by selecting
        the indices in the order given in :attr:`index`. For example, if ``dim == 0``
        and ``index[i] == j``, then the ``i``\ th row of :attr:`tensor` is copied to the
        ``j``\ th row of :attr:`self`.

        The :attr:`dim`\ th dimension of :attr:`tensor` must have the same size as the
        length of :attr:`index` (which must be a vector), and all other dimensions must
        match :attr:`self`, or an error will be raised.

        .. note::
            If :attr:`index` contains duplicate entries, multiple elements from
            :attr:`tensor` will be copied to the same index of :attr:`self`. The result
            is nondeterministic since it depends on which copy occurs last.

        Args:
            dim (int): dimension along which to index
            index (LongTensor): indices of :attr:`tensor` to select from
            tensor (Tensor): the tensor containing values to copy

        Example::

            >>> x = torch.zeros(5, 3)
            >>> t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
            >>> index = torch.tensor([0, 4, 2])
            >>> x.index_copy_(0, index, t)
            tensor([[ 1.,  2.,  3.],
                    [ 0.,  0.,  0.],
                    [ 7.,  8.,  9.],
                    [ 0.,  0.,  0.],
                    [ 4.,  5.,  6.]])
        """
        ...
    @overload
    def index_copy_(self, dim: Union[str, ellipsis, None], index: Tensor, source: Tensor) -> Tensor:
        r"""
        index_copy_(dim, index, tensor) -> Tensor

        Copies the elements of :attr:`tensor` into the :attr:`self` tensor by selecting
        the indices in the order given in :attr:`index`. For example, if ``dim == 0``
        and ``index[i] == j``, then the ``i``\ th row of :attr:`tensor` is copied to the
        ``j``\ th row of :attr:`self`.

        The :attr:`dim`\ th dimension of :attr:`tensor` must have the same size as the
        length of :attr:`index` (which must be a vector), and all other dimensions must
        match :attr:`self`, or an error will be raised.

        .. note::
            If :attr:`index` contains duplicate entries, multiple elements from
            :attr:`tensor` will be copied to the same index of :attr:`self`. The result
            is nondeterministic since it depends on which copy occurs last.

        Args:
            dim (int): dimension along which to index
            index (LongTensor): indices of :attr:`tensor` to select from
            tensor (Tensor): the tensor containing values to copy

        Example::

            >>> x = torch.zeros(5, 3)
            >>> t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
            >>> index = torch.tensor([0, 4, 2])
            >>> x.index_copy_(0, index, t)
            tensor([[ 1.,  2.,  3.],
                    [ 0.,  0.,  0.],
                    [ 7.,  8.,  9.],
                    [ 0.,  0.,  0.],
                    [ 4.,  5.,  6.]])
        """
        ...
    @overload
    def index_fill(self, dim: _int, index: Tensor, value: Tensor) -> Tensor:
        r"""
        index_fill(dim, index, value) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.index_fill_`.
        """
        ...
    @overload
    def index_fill(self, dim: Union[str, ellipsis, None], index: Tensor, value: Tensor) -> Tensor:
        r"""
        index_fill(dim, index, value) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.index_fill_`.
        """
        ...
    @overload
    def index_fill(self, dim: _int, index: Tensor, value: Union[Number, _complex]) -> Tensor:
        r"""
        index_fill(dim, index, value) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.index_fill_`.
        """
        ...
    @overload
    def index_fill(self, dim: Union[str, ellipsis, None], index: Tensor, value: Union[Number, _complex]) -> Tensor:
        r"""
        index_fill(dim, index, value) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.index_fill_`.
        """
        ...
    @overload
    def index_fill_(self, dim: _int, index: Tensor, value: Tensor) -> Tensor:
        r"""
        index_fill_(dim, index, value) -> Tensor

        Fills the elements of the :attr:`self` tensor with value :attr:`value` by
        selecting the indices in the order given in :attr:`index`.

        Args:
            dim (int): dimension along which to index
            index (LongTensor): indices of :attr:`self` tensor to fill in
            value (float): the value to fill with

        Example::
            >>> x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
            >>> index = torch.tensor([0, 2])
            >>> x.index_fill_(1, index, -1)
            tensor([[-1.,  2., -1.],
                    [-1.,  5., -1.],
                    [-1.,  8., -1.]])
        """
        ...
    @overload
    def index_fill_(self, dim: Union[str, ellipsis, None], index: Tensor, value: Tensor) -> Tensor:
        r"""
        index_fill_(dim, index, value) -> Tensor

        Fills the elements of the :attr:`self` tensor with value :attr:`value` by
        selecting the indices in the order given in :attr:`index`.

        Args:
            dim (int): dimension along which to index
            index (LongTensor): indices of :attr:`self` tensor to fill in
            value (float): the value to fill with

        Example::
            >>> x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
            >>> index = torch.tensor([0, 2])
            >>> x.index_fill_(1, index, -1)
            tensor([[-1.,  2., -1.],
                    [-1.,  5., -1.],
                    [-1.,  8., -1.]])
        """
        ...
    @overload
    def index_fill_(self, dim: _int, index: Tensor, value: Union[Number, _complex]) -> Tensor:
        r"""
        index_fill_(dim, index, value) -> Tensor

        Fills the elements of the :attr:`self` tensor with value :attr:`value` by
        selecting the indices in the order given in :attr:`index`.

        Args:
            dim (int): dimension along which to index
            index (LongTensor): indices of :attr:`self` tensor to fill in
            value (float): the value to fill with

        Example::
            >>> x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
            >>> index = torch.tensor([0, 2])
            >>> x.index_fill_(1, index, -1)
            tensor([[-1.,  2., -1.],
                    [-1.,  5., -1.],
                    [-1.,  8., -1.]])
        """
        ...
    @overload
    def index_fill_(self, dim: Union[str, ellipsis, None], index: Tensor, value: Union[Number, _complex]) -> Tensor:
        r"""
        index_fill_(dim, index, value) -> Tensor

        Fills the elements of the :attr:`self` tensor with value :attr:`value` by
        selecting the indices in the order given in :attr:`index`.

        Args:
            dim (int): dimension along which to index
            index (LongTensor): indices of :attr:`self` tensor to fill in
            value (float): the value to fill with

        Example::
            >>> x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
            >>> index = torch.tensor([0, 2])
            >>> x.index_fill_(1, index, -1)
            tensor([[-1.,  2., -1.],
                    [-1.,  5., -1.],
                    [-1.,  8., -1.]])
        """
        ...
    def index_put(self, indices: Optional[Union[Tuple[Tensor, ...], List[Tensor]]], values: Tensor, accumulate: _bool = False) -> Tensor:
        r"""
        index_put(indices, values, accumulate=False) -> Tensor

        Out-place version of :meth:`~Tensor.index_put_`.
        """
        ...
    def index_put_(self, indices: Optional[Union[Tuple[Tensor, ...], List[Tensor]]], values: Tensor, accumulate: _bool = False) -> Tensor:
        r"""
        index_put_(indices, values, accumulate=False) -> Tensor

        Puts values from the tensor :attr:`values` into the tensor :attr:`self` using
        the indices specified in :attr:`indices` (which is a tuple of Tensors). The
        expression ``tensor.index_put_(indices, values)`` is equivalent to
        ``tensor[indices] = values``. Returns :attr:`self`.

        If :attr:`accumulate` is ``True``, the elements in :attr:`values` are added to
        :attr:`self`. If accumulate is ``False``, the behavior is undefined if indices
        contain duplicate elements.

        Args:
            indices (tuple of LongTensor): tensors used to index into `self`.
            values (Tensor): tensor of same dtype as `self`.
            accumulate (bool): whether to accumulate into self
        """
        ...
    def index_reduce(self, dim: _int, index: Tensor, source: Tensor, reduce: str, *, include_self: _bool = True) -> Tensor: ...
    def index_reduce_(self, dim: _int, index: Tensor, source: Tensor, reduce: str, *, include_self: _bool = True) -> Tensor:
        r"""
        index_reduce_(dim, index, source, reduce, *, include_self=True) -> Tensor

        Accumulate the elements of ``source`` into the :attr:`self`
        tensor by accumulating to the indices in the order given in :attr:`index`
        using the reduction given by the ``reduce`` argument. For example, if ``dim == 0``,
        ``index[i] == j``, ``reduce == prod`` and ``include_self == True`` then the ``i``\ th
        row of ``source`` is multiplied by the ``j``\ th row of :attr:`self`. If
        :obj:`include_self="True"`, the values in the :attr:`self` tensor are included
        in the reduction, otherwise, rows in the :attr:`self` tensor that are accumulated
        to are treated as if they were filled with the reduction identites.

        The :attr:`dim`\ th dimension of ``source`` must have the same size as the
        length of :attr:`index` (which must be a vector), and all other dimensions must
        match :attr:`self`, or an error will be raised.

        For a 3-D tensor with :obj:`reduce="prod"` and :obj:`include_self=True` the
        output is given as::

            self[index[i], :, :] *= src[i, :, :]  # if dim == 0
            self[:, index[i], :] *= src[:, i, :]  # if dim == 1
            self[:, :, index[i]] *= src[:, :, i]  # if dim == 2

        Note:
            This operation may behave nondeterministically when given tensors on a CUDA device. See :doc:`/notes/randomness` for more information.

        .. note::

            This function only supports floating point tensors.

        .. warning::

            This function is in beta and may change in the near future.

        Args:
            dim (int): dimension along which to index
            index (Tensor): indices of ``source`` to select from,
                should have dtype either `torch.int64` or `torch.int32`
            source (FloatTensor): the tensor containing values to accumulate
            reduce (str): the reduction operation to apply
                (:obj:`"prod"`, :obj:`"mean"`, :obj:`"amax"`, :obj:`"amin"`)

        Keyword args:
            include_self (bool): whether the elements from the ``self`` tensor are
                included in the reduction

        Example::

            >>> x = torch.empty(5, 3).fill_(2)
            >>> t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=torch.float)
            >>> index = torch.tensor([0, 4, 2, 0])
            >>> x.index_reduce_(0, index, t, 'prod')
            tensor([[20., 44., 72.],
                    [ 2.,  2.,  2.],
                    [14., 16., 18.],
                    [ 2.,  2.,  2.],
                    [ 8., 10., 12.]])
            >>> x = torch.empty(5, 3).fill_(2)
            >>> x.index_reduce_(0, index, t, 'prod', include_self=False)
            tensor([[10., 22., 36.],
                    [ 2.,  2.,  2.],
                    [ 7.,  8.,  9.],
                    [ 2.,  2.,  2.],
                    [ 4.,  5.,  6.]])
        """
        ...
    @overload
    def index_select(self, dim: _int, index: Tensor) -> Tensor:
        r"""
        index_select(dim, index) -> Tensor

        See :func:`torch.index_select`
        """
        ...
    @overload
    def index_select(self, dim: Union[str, ellipsis, None], index: Tensor) -> Tensor:
        r"""
        index_select(dim, index) -> Tensor

        See :func:`torch.index_select`
        """
        ...
    def indices(self) -> Tensor:
        r"""
        indices() -> Tensor

        Return the indices tensor of a :ref:`sparse COO tensor <sparse-coo-docs>`.

        .. warning::
          Throws an error if :attr:`self` is not a sparse COO tensor.

        See also :meth:`Tensor.values`.

        .. note::
          This method can only be called on a coalesced sparse tensor. See
          :meth:`Tensor.coalesce` for details.
        """
        ...
    def inner(self, other: Tensor) -> Tensor:
        r"""
        inner(other) -> Tensor

        See :func:`torch.inner`.
        """
        ...
    def int(self) -> Tensor:
        r"""
        int(memory_format=torch.preserve_format) -> Tensor

        ``self.int()`` is equivalent to ``self.to(torch.int32)``. See :func:`to`.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.
        """
        ...
    def int_repr(self) -> Tensor:
        r"""
        int_repr() -> Tensor

        Given a quantized Tensor,
        ``self.int_repr()`` returns a CPU Tensor with uint8_t as data type that stores the
        underlying uint8_t values of the given Tensor.
        """
        ...
    def inverse(self) -> Tensor:
        r"""
        inverse() -> Tensor

        See :func:`torch.inverse`
        """
        ...
    def is_coalesced(self) -> _bool:
        r"""
        is_coalesced() -> bool

        Returns ``True`` if :attr:`self` is a :ref:`sparse COO tensor
        <sparse-coo-docs>` that is coalesced, ``False`` otherwise.

        .. warning::
          Throws an error if :attr:`self` is not a sparse COO tensor.

        See :meth:`coalesce` and :ref:`uncoalesced tensors <sparse-uncoalesced-coo-docs>`.
        """
        ...
    def is_complex(self) -> _bool:
        r"""
        is_complex() -> bool

        Returns True if the data type of :attr:`self` is a complex data type.
        """
        ...
    def is_conj(self) -> _bool:
        r"""
        is_conj() -> bool

        Returns True if the conjugate bit of :attr:`self` is set to true.
        """
        ...
    def is_contiguous(self, memory_format=torch.contiguous_format) -> _bool:
        r"""
        is_contiguous(memory_format=torch.contiguous_format) -> bool

        Returns True if :attr:`self` tensor is contiguous in memory in the order specified
        by memory format.

        Args:
            memory_format (:class:`torch.memory_format`, optional): Specifies memory allocation
                order. Default: ``torch.contiguous_format``.
        """
        ...
    is_cpu: _bool
    r"""Is ``True`` if the Tensor is stored on the CPU, ``False`` otherwise."""
    is_cuda: _bool
    r"""Is ``True`` if the Tensor is stored on the GPU, ``False`` otherwise."""
    def is_distributed(self) -> _bool: ...
    def is_floating_point(self) -> _bool:
        r"""
        is_floating_point() -> bool

        Returns True if the data type of :attr:`self` is a floating point data type.
        """
        ...
    def is_inference(self) -> _bool:
        r"""
        is_inference() -> bool

        See :func:`torch.is_inference`
        """
        ...
    is_ipu: _bool
    r"""Is ``True`` if the Tensor is stored on the IPU, ``False`` otherwise."""
    is_leaf: _bool
    r"""All Tensors that have :attr:`requires_grad` which is ``False`` will be leaf Tensors by convention.

    For Tensors that have :attr:`requires_grad` which is ``True``, they will be leaf Tensors if they were
    created by the user. This means that they are not the result of an operation and so
    :attr:`grad_fn` is None.

    Only leaf Tensors will have their :attr:`grad` populated during a call to :func:`backward`.
    To get :attr:`grad` populated for non-leaf Tensors, you can use :func:`retain_grad`.

    Example::

        >>> a = torch.rand(10, requires_grad=True)
        >>> a.is_leaf
        True
        >>> b = torch.rand(10, requires_grad=True).cuda()
        >>> b.is_leaf
        False
        # b was created by the operation that cast a cpu Tensor into a cuda Tensor
        >>> c = torch.rand(10, requires_grad=True) + 2
        >>> c.is_leaf
        False
        # c was created by the addition operation
        >>> d = torch.rand(10).cuda()
        >>> d.is_leaf
        True
        # d does not require gradients and so has no operation creating it (that is tracked by the autograd engine)
        >>> e = torch.rand(10).cuda().requires_grad_()
        >>> e.is_leaf
        True
        # e requires gradients and has no operations creating it
        >>> f = torch.rand(10, requires_grad=True, device="cuda")
        >>> f.is_leaf
        True
        # f requires grad, has no operation creating it"""
    is_meta: _bool
    r"""Is ``True`` if the Tensor is a meta tensor, ``False`` otherwise.  Meta tensors
    are like normal tensors, but they carry no data."""
    is_mkldnn: _bool
    is_mps: _bool
    r"""Is ``True`` if the Tensor is stored on the MPS device, ``False`` otherwise."""
    is_mtia: _bool
    def is_neg(self) -> _bool:
        r"""
        is_neg() -> bool

        Returns True if the negative bit of :attr:`self` is set to true.
        """
        ...
    is_nested: _bool
    def is_nonzero(self) -> _bool: ...
    is_ort: _bool
    def is_pinned(self, device: Optional[Optional[DeviceLikeType]] = None) -> _bool:
        r"""
        Returns true if this tensor resides in pinned memory.
        """
        ...
    is_quantized: _bool
    r"""Is ``True`` if the Tensor is quantized, ``False`` otherwise."""
    def is_same_size(self, other: Tensor) -> _bool: ...
    def is_set_to(self, tensor: Tensor) -> _bool:
        r"""
        is_set_to(tensor) -> bool

        Returns True if both tensors are pointing to the exact same memory (same
        storage, offset, size and stride).
        """
        ...
    def is_signed(self) -> _bool:
        r"""
        is_signed() -> bool

        Returns True if the data type of :attr:`self` is a signed data type.
        """
        ...
    is_sparse: _bool
    r"""Is ``True`` if the Tensor uses sparse COO storage layout, ``False`` otherwise."""
    is_sparse_csr: _bool
    r"""Is ``True`` if the Tensor uses sparse CSR storage layout, ``False`` otherwise."""
    is_vulkan: _bool
    def isclose(self, other: Tensor, rtol: _float = 1e-05, atol: _float = 1e-08, equal_nan: _bool = False) -> Tensor:
        r"""
        isclose(other, rtol=1e-05, atol=1e-08, equal_nan=False) -> Tensor

        See :func:`torch.isclose`
        """
        ...
    def isfinite(self) -> Tensor:
        r"""
        isfinite() -> Tensor

        See :func:`torch.isfinite`
        """
        ...
    def isinf(self) -> Tensor:
        r"""
        isinf() -> Tensor

        See :func:`torch.isinf`
        """
        ...
    def isnan(self) -> Tensor:
        r"""
        isnan() -> Tensor

        See :func:`torch.isnan`
        """
        ...
    def isneginf(self) -> Tensor:
        r"""
        isneginf() -> Tensor

        See :func:`torch.isneginf`
        """
        ...
    def isposinf(self) -> Tensor:
        r"""
        isposinf() -> Tensor

        See :func:`torch.isposinf`
        """
        ...
    def isreal(self) -> Tensor:
        r"""
        isreal() -> Tensor

        See :func:`torch.isreal`
        """
        ...
    def istft(self, n_fft: _int, hop_length: Optional[_int] = None, win_length: Optional[_int] = None, window: Optional[Tensor] = None, center: _bool = True, normalized: _bool = False, onesided: Optional[_bool] = None, length: Optional[_int] = None, return_complex: _bool = False) -> Tensor:
        r"""
        istft(n_fft, hop_length=None, win_length=None, window=None,
         center=True, normalized=False, onesided=True, length=None) -> Tensor

        See :func:`torch.istft`
        """
        ...
    def item(self) -> Number:
        r"""
        item() -> number

        Returns the value of this tensor as a standard Python number. This only works
        for tensors with one element. For other cases, see :meth:`~Tensor.tolist`.

        This operation is not differentiable.

        Example::

            >>> x = torch.tensor([1.0])
            >>> x.item()
            1.0
        """
        ...
    def kron(self, other: Tensor) -> Tensor:
        r"""
        kron(other) -> Tensor

        See :func:`torch.kron`
        """
        ...
    @overload
    def kthvalue(self, k: _int, dim: _int = -1, keepdim: _bool = False) -> torch.return_types.kthvalue:
        r"""
        kthvalue(k, dim=None, keepdim=False) -> (Tensor, LongTensor)

        See :func:`torch.kthvalue`
        """
        ...
    @overload
    def kthvalue(self, k: _int, dim: Union[str, ellipsis, None], keepdim: _bool = False) -> torch.return_types.kthvalue:
        r"""
        kthvalue(k, dim=None, keepdim=False) -> (Tensor, LongTensor)

        See :func:`torch.kthvalue`
        """
        ...
    def lcm(self, other: Tensor) -> Tensor:
        r"""
        lcm(other) -> Tensor

        See :func:`torch.lcm`
        """
        ...
    def lcm_(self, other: Tensor) -> Tensor:
        r"""
        lcm_(other) -> Tensor

        In-place version of :meth:`~Tensor.lcm`
        """
        ...
    def ldexp(self, other: Tensor) -> Tensor:
        r"""
        ldexp(other) -> Tensor

        See :func:`torch.ldexp`
        """
        ...
    def ldexp_(self, other: Tensor) -> Tensor:
        r"""
        ldexp_(other) -> Tensor

        In-place version of :meth:`~Tensor.ldexp`
        """
        ...
    @overload
    def le(self, other: Tensor) -> Tensor:
        r"""
        le(other) -> Tensor

        See :func:`torch.le`.
        """
        ...
    @overload
    def le(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        le(other) -> Tensor

        See :func:`torch.le`.
        """
        ...
    @overload
    def le_(self, other: Tensor) -> Tensor:
        r"""
        le_(other) -> Tensor

        In-place version of :meth:`~Tensor.le`.
        """
        ...
    @overload
    def le_(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        le_(other) -> Tensor

        In-place version of :meth:`~Tensor.le`.
        """
        ...
    @overload
    def lerp(self, end: Tensor, weight: Tensor) -> Tensor:
        r"""
        lerp(end, weight) -> Tensor

        See :func:`torch.lerp`
        """
        ...
    @overload
    def lerp(self, end: Tensor, weight: Union[Number, _complex]) -> Tensor:
        r"""
        lerp(end, weight) -> Tensor

        See :func:`torch.lerp`
        """
        ...
    @overload
    def lerp_(self, end: Tensor, weight: Tensor) -> Tensor:
        r"""
        lerp_(end, weight) -> Tensor

        In-place version of :meth:`~Tensor.lerp`
        """
        ...
    @overload
    def lerp_(self, end: Tensor, weight: Union[Number, _complex]) -> Tensor:
        r"""
        lerp_(end, weight) -> Tensor

        In-place version of :meth:`~Tensor.lerp`
        """
        ...
    @overload
    def less(self, other: Tensor) -> Tensor:
        r"""
        lt(other) -> Tensor

        See :func:`torch.less`.
        """
        ...
    @overload
    def less(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        lt(other) -> Tensor

        See :func:`torch.less`.
        """
        ...
    @overload
    def less_(self, other: Tensor) -> Tensor:
        r"""
        less_(other) -> Tensor

        In-place version of :meth:`~Tensor.less`.
        """
        ...
    @overload
    def less_(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        less_(other) -> Tensor

        In-place version of :meth:`~Tensor.less`.
        """
        ...
    @overload
    def less_equal(self, other: Tensor) -> Tensor:
        r"""
        less_equal(other) -> Tensor

        See :func:`torch.less_equal`.
        """
        ...
    @overload
    def less_equal(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        less_equal(other) -> Tensor

        See :func:`torch.less_equal`.
        """
        ...
    @overload
    def less_equal_(self, other: Tensor) -> Tensor:
        r"""
        less_equal_(other) -> Tensor

        In-place version of :meth:`~Tensor.less_equal`.
        """
        ...
    @overload
    def less_equal_(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        less_equal_(other) -> Tensor

        In-place version of :meth:`~Tensor.less_equal`.
        """
        ...
    def lgamma(self) -> Tensor:
        r"""
        lgamma() -> Tensor

        See :func:`torch.lgamma`
        """
        ...
    def lgamma_(self) -> Tensor:
        r"""
        lgamma_() -> Tensor

        In-place version of :meth:`~Tensor.lgamma`
        """
        ...
    def log(self) -> Tensor:
        r"""
        log() -> Tensor

        See :func:`torch.log`
        """
        ...
    def log10(self) -> Tensor:
        r"""
        log10() -> Tensor

        See :func:`torch.log10`
        """
        ...
    def log10_(self) -> Tensor:
        r"""
        log10_() -> Tensor

        In-place version of :meth:`~Tensor.log10`
        """
        ...
    def log1p(self) -> Tensor:
        r"""
        log1p() -> Tensor

        See :func:`torch.log1p`
        """
        ...
    def log1p_(self) -> Tensor:
        r"""
        log1p_() -> Tensor

        In-place version of :meth:`~Tensor.log1p`
        """
        ...
    def log2(self) -> Tensor:
        r"""
        log2() -> Tensor

        See :func:`torch.log2`
        """
        ...
    def log2_(self) -> Tensor:
        r"""
        log2_() -> Tensor

        In-place version of :meth:`~Tensor.log2`
        """
        ...
    def log_(self) -> Tensor:
        r"""
        log_() -> Tensor

        In-place version of :meth:`~Tensor.log`
        """
        ...
    def log_normal_(self, mean: _float = 1, std: _float = 2, *, generator: Optional[Generator] = None) -> Tensor:
        r"""
        log_normal_(mean=1, std=2, *, generator=None)

        Fills :attr:`self` tensor with numbers samples from the log-normal distribution
        parameterized by the given mean :math:`\mu` and standard deviation
        :math:`\sigma`. Note that :attr:`mean` and :attr:`std` are the mean and
        standard deviation of the underlying normal distribution, and not of the
        returned distribution:

        .. math::

            f(x) = \dfrac{1}{x \sigma \sqrt{2\pi}}\ e^{-\frac{(\ln x - \mu)^2}{2\sigma^2}}
        """
        ...
    @overload
    def log_softmax(self, dim: _int, dtype: Optional[_dtype] = None) -> Tensor: ...
    @overload
    def log_softmax(self, dim: Union[str, ellipsis, None], *, dtype: Optional[_dtype] = None) -> Tensor: ...
    def logaddexp(self, other: Tensor) -> Tensor:
        r"""
        logaddexp(other) -> Tensor

        See :func:`torch.logaddexp`
        """
        ...
    def logaddexp2(self, other: Tensor) -> Tensor:
        r"""
        logaddexp2(other) -> Tensor

        See :func:`torch.logaddexp2`
        """
        ...
    @overload
    def logcumsumexp(self, dim: _int) -> Tensor:
        r"""
        logcumsumexp(dim) -> Tensor

        See :func:`torch.logcumsumexp`
        """
        ...
    @overload
    def logcumsumexp(self, dim: Union[str, ellipsis, None]) -> Tensor:
        r"""
        logcumsumexp(dim) -> Tensor

        See :func:`torch.logcumsumexp`
        """
        ...
    def logdet(self) -> Tensor:
        r"""
        logdet() -> Tensor

        See :func:`torch.logdet`
        """
        ...
    def logical_and(self, other: Tensor) -> Tensor:
        r"""
        logical_and() -> Tensor

        See :func:`torch.logical_and`
        """
        ...
    def logical_and_(self, other: Tensor) -> Tensor:
        r"""
        logical_and_() -> Tensor

        In-place version of :meth:`~Tensor.logical_and`
        """
        ...
    def logical_not(self) -> Tensor:
        r"""
        logical_not() -> Tensor

        See :func:`torch.logical_not`
        """
        ...
    def logical_not_(self) -> Tensor:
        r"""
        logical_not_() -> Tensor

        In-place version of :meth:`~Tensor.logical_not`
        """
        ...
    def logical_or(self, other: Tensor) -> Tensor:
        r"""
        logical_or() -> Tensor

        See :func:`torch.logical_or`
        """
        ...
    def logical_or_(self, other: Tensor) -> Tensor:
        r"""
        logical_or_() -> Tensor

        In-place version of :meth:`~Tensor.logical_or`
        """
        ...
    def logical_xor(self, other: Tensor) -> Tensor:
        r"""
        logical_xor() -> Tensor

        See :func:`torch.logical_xor`
        """
        ...
    def logical_xor_(self, other: Tensor) -> Tensor:
        r"""
        logical_xor_() -> Tensor

        In-place version of :meth:`~Tensor.logical_xor`
        """
        ...
    def logit(self, eps: Optional[_float] = None) -> Tensor:
        r"""
        logit() -> Tensor

        See :func:`torch.logit`
        """
        ...
    def logit_(self, eps: Optional[_float] = None) -> Tensor:
        r"""
        logit_() -> Tensor

        In-place version of :meth:`~Tensor.logit`
        """
        ...
    @overload
    def logsumexp(self, dim: Union[_int, _size], keepdim: _bool = False) -> Tensor:
        r"""
        logsumexp(dim, keepdim=False) -> Tensor

        See :func:`torch.logsumexp`
        """
        ...
    @overload
    def logsumexp(self, dim: Sequence[Union[str, ellipsis, None]], keepdim: _bool = False) -> Tensor:
        r"""
        logsumexp(dim, keepdim=False) -> Tensor

        See :func:`torch.logsumexp`
        """
        ...
    def long(self) -> Tensor:
        r"""
        long(memory_format=torch.preserve_format) -> Tensor

        ``self.long()`` is equivalent to ``self.to(torch.int64)``. See :func:`to`.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.
        """
        ...
    @overload
    def lt(self, other: Tensor) -> Tensor:
        r"""
        lt(other) -> Tensor

        See :func:`torch.lt`.
        """
        ...
    @overload
    def lt(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        lt(other) -> Tensor

        See :func:`torch.lt`.
        """
        ...
    @overload
    def lt_(self, other: Tensor) -> Tensor:
        r"""
        lt_(other) -> Tensor

        In-place version of :meth:`~Tensor.lt`.
        """
        ...
    @overload
    def lt_(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        lt_(other) -> Tensor

        In-place version of :meth:`~Tensor.lt`.
        """
        ...
    def lu_solve(self, LU_data: Tensor, LU_pivots: Tensor) -> Tensor:
        r"""
        lu_solve(LU_data, LU_pivots) -> Tensor

        See :func:`torch.lu_solve`
        """
        ...
    def map2_(self, x: Tensor, y: Tensor, callable: Callable) -> Tensor: ...
    def map_(self, tensor: Tensor, callable: Callable) -> Tensor:
        r"""
        map_(tensor, callable)

        Applies :attr:`callable` for each element in :attr:`self` tensor and the given
        :attr:`tensor` and stores the results in :attr:`self` tensor. :attr:`self` tensor and
        the given :attr:`tensor` must be :ref:`broadcastable <broadcasting-semantics>`.

        The :attr:`callable` should have the signature::

            def callable(a, b) -> number
        """
        ...
    @overload
    def masked_fill(self, mask: Tensor, value: Tensor) -> Tensor:
        r"""
        masked_fill(mask, value) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.masked_fill_`
        """
        ...
    @overload
    def masked_fill(self, mask: Tensor, value: Union[Number, _complex]) -> Tensor:
        r"""
        masked_fill(mask, value) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.masked_fill_`
        """
        ...
    @overload
    def masked_fill_(self, mask: Tensor, value: Tensor) -> Tensor:
        r"""
        masked_fill_(mask, value)

        Fills elements of :attr:`self` tensor with :attr:`value` where :attr:`mask` is
        True. The shape of :attr:`mask` must be
        :ref:`broadcastable <broadcasting-semantics>` with the shape of the underlying
        tensor.

        Args:
            mask (BoolTensor): the boolean mask
            value (float): the value to fill in with
        """
        ...
    @overload
    def masked_fill_(self, mask: Tensor, value: Union[Number, _complex]) -> Tensor:
        r"""
        masked_fill_(mask, value)

        Fills elements of :attr:`self` tensor with :attr:`value` where :attr:`mask` is
        True. The shape of :attr:`mask` must be
        :ref:`broadcastable <broadcasting-semantics>` with the shape of the underlying
        tensor.

        Args:
            mask (BoolTensor): the boolean mask
            value (float): the value to fill in with
        """
        ...
    def masked_scatter(self, mask: Tensor, source: Tensor) -> Tensor:
        r"""
        masked_scatter(mask, tensor) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.masked_scatter_`

        .. note::

            The inputs :attr:`self` and :attr:`mask`
            :ref:`broadcast <broadcasting-semantics>`.

        Example:

            >>> self = torch.tensor([0, 0, 0, 0, 0])
            >>> mask = torch.tensor([[0, 0, 0, 1, 1], [1, 1, 0, 1, 1]], dtype=torch.bool)
            >>> source = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
            >>> self.masked_scatter(mask, source)
            tensor([[0, 0, 0, 0, 1],
                    [2, 3, 0, 4, 5]])
        """
        ...
    def masked_scatter_(self, mask: Tensor, source: Tensor) -> Tensor:
        r"""
        masked_scatter_(mask, source)

        Copies elements from :attr:`source` into :attr:`self` tensor at positions where
        the :attr:`mask` is True. Elements from :attr:`source` are copied into :attr:`self`
        starting at position 0 of :attr:`source` and continuing in order one-by-one for each
        occurrence of :attr:`mask` being True.
        The shape of :attr:`mask` must be :ref:`broadcastable <broadcasting-semantics>`
        with the shape of the underlying tensor. The :attr:`source` should have at least
        as many elements as the number of ones in :attr:`mask`.

        Args:
            mask (BoolTensor): the boolean mask
            source (Tensor): the tensor to copy from

        .. note::

            The :attr:`mask` operates on the :attr:`self` tensor, not on the given
            :attr:`source` tensor.

        Example:

            >>> self = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
            >>> mask = torch.tensor([[0, 0, 0, 1, 1], [1, 1, 0, 1, 1]], dtype=torch.bool)
            >>> source = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
            >>> self.masked_scatter_(mask, source)
            tensor([[0, 0, 0, 0, 1],
                    [2, 3, 0, 4, 5]])
        """
        ...
    def masked_select(self, mask: Tensor) -> Tensor:
        r"""
        masked_select(mask) -> Tensor

        See :func:`torch.masked_select`
        """
        ...
    def matmul(self, other: Tensor) -> Tensor:
        r"""
        matmul(tensor2) -> Tensor

        See :func:`torch.matmul`
        """
        ...
    def matrix_exp(self) -> Tensor:
        r"""
        matrix_exp() -> Tensor

        See :func:`torch.matrix_exp`
        """
        ...
    def matrix_power(self, n: _int) -> Tensor:
        r"""
        matrix_power(n) -> Tensor

        .. note:: :meth:`~Tensor.matrix_power` is deprecated, use :func:`torch.linalg.matrix_power` instead.

        Alias for :func:`torch.linalg.matrix_power`
        """
        ...
    @overload
    def max(self) -> Tensor:
        r"""
        max(dim=None, keepdim=False) -> Tensor or (Tensor, Tensor)

        See :func:`torch.max`
        """
        ...
    @overload
    def max(self, other: Tensor) -> Tensor:
        r"""
        max(dim=None, keepdim=False) -> Tensor or (Tensor, Tensor)

        See :func:`torch.max`
        """
        ...
    @overload
    def max(self, dim: _int, keepdim: _bool = False) -> torch.return_types.max:
        r"""
        max(dim=None, keepdim=False) -> Tensor or (Tensor, Tensor)

        See :func:`torch.max`
        """
        ...
    @overload
    def max(self, dim: Union[str, ellipsis, None], keepdim: _bool = False) -> torch.return_types.max:
        r"""
        max(dim=None, keepdim=False) -> Tensor or (Tensor, Tensor)

        See :func:`torch.max`
        """
        ...
    def maximum(self, other: Tensor) -> Tensor:
        r"""
        maximum(other) -> Tensor

        See :func:`torch.maximum`
        """
        ...
    @overload
    def mean(self, *, dtype: Optional[_dtype] = None) -> Tensor:
        r"""
        mean(dim=None, keepdim=False, *, dtype=None) -> Tensor

        See :func:`torch.mean`
        """
        ...
    @overload
    def mean(self, dim: Optional[Union[_int, _size]], keepdim: _bool = False, *, dtype: Optional[_dtype] = None) -> Tensor:
        r"""
        mean(dim=None, keepdim=False, *, dtype=None) -> Tensor

        See :func:`torch.mean`
        """
        ...
    @overload
    def mean(self, dim: Sequence[Union[str, ellipsis, None]], keepdim: _bool = False, *, dtype: Optional[_dtype] = None) -> Tensor:
        r"""
        mean(dim=None, keepdim=False, *, dtype=None) -> Tensor

        See :func:`torch.mean`
        """
        ...
    @overload
    def median(self) -> Tensor:
        r"""
        median(dim=None, keepdim=False) -> (Tensor, LongTensor)

        See :func:`torch.median`
        """
        ...
    @overload
    def median(self, dim: _int, keepdim: _bool = False) -> torch.return_types.median:
        r"""
        median(dim=None, keepdim=False) -> (Tensor, LongTensor)

        See :func:`torch.median`
        """
        ...
    @overload
    def median(self, dim: Union[str, ellipsis, None], keepdim: _bool = False) -> torch.return_types.median:
        r"""
        median(dim=None, keepdim=False) -> (Tensor, LongTensor)

        See :func:`torch.median`
        """
        ...
    @overload
    def min(self) -> Tensor:
        r"""
        min(dim=None, keepdim=False) -> Tensor or (Tensor, Tensor)

        See :func:`torch.min`
        """
        ...
    @overload
    def min(self, other: Tensor) -> Tensor:
        r"""
        min(dim=None, keepdim=False) -> Tensor or (Tensor, Tensor)

        See :func:`torch.min`
        """
        ...
    @overload
    def min(self, dim: _int, keepdim: _bool = False) -> torch.return_types.min:
        r"""
        min(dim=None, keepdim=False) -> Tensor or (Tensor, Tensor)

        See :func:`torch.min`
        """
        ...
    @overload
    def min(self, dim: Union[str, ellipsis, None], keepdim: _bool = False) -> torch.return_types.min:
        r"""
        min(dim=None, keepdim=False) -> Tensor or (Tensor, Tensor)

        See :func:`torch.min`
        """
        ...
    def minimum(self, other: Tensor) -> Tensor:
        r"""
        minimum(other) -> Tensor

        See :func:`torch.minimum`
        """
        ...
    def mm(self, mat2: Tensor) -> Tensor:
        r"""
        mm(mat2) -> Tensor

        See :func:`torch.mm`
        """
        ...
    @overload
    def mode(self, dim: _int = -1, keepdim: _bool = False) -> torch.return_types.mode:
        r"""
        mode(dim=None, keepdim=False) -> (Tensor, LongTensor)

        See :func:`torch.mode`
        """
        ...
    @overload
    def mode(self, dim: Union[str, ellipsis, None], keepdim: _bool = False) -> torch.return_types.mode:
        r"""
        mode(dim=None, keepdim=False) -> (Tensor, LongTensor)

        See :func:`torch.mode`
        """
        ...
    @overload
    def moveaxis(self, source: _int, destination: _int) -> Tensor:
        r"""
        moveaxis(source, destination) -> Tensor

        See :func:`torch.moveaxis`
        """
        ...
    @overload
    def moveaxis(self, source: _size, destination: _size) -> Tensor:
        r"""
        moveaxis(source, destination) -> Tensor

        See :func:`torch.moveaxis`
        """
        ...
    @overload
    def movedim(self, source: _int, destination: _int) -> Tensor:
        r"""
        movedim(source, destination) -> Tensor

        See :func:`torch.movedim`
        """
        ...
    @overload
    def movedim(self, source: _size, destination: _size) -> Tensor:
        r"""
        movedim(source, destination) -> Tensor

        See :func:`torch.movedim`
        """
        ...
    def msort(self) -> Tensor:
        r"""
        msort() -> Tensor

        See :func:`torch.msort`
        """
        ...
    def mul(self, other: Union[Tensor, Number, _complex, torch.SymInt, torch.SymFloat], *, out: Optional[Tensor] = None) -> Tensor:
        r"""
        mul(value) -> Tensor

        See :func:`torch.mul`.
        """
        ...
    def mul_(self, other: Union[Tensor, Number, _complex, torch.SymInt, torch.SymFloat]) -> Tensor:
        r"""
        mul_(value) -> Tensor

        In-place version of :meth:`~Tensor.mul`.
        """
        ...
    def multinomial(self, num_samples: _int, replacement: _bool = False, *, generator: Optional[Generator] = None) -> Tensor:
        r"""
        multinomial(num_samples, replacement=False, *, generator=None) -> Tensor

        See :func:`torch.multinomial`
        """
        ...
    @overload
    def multiply(self, other: Tensor) -> Tensor:
        r"""
        multiply(value) -> Tensor

        See :func:`torch.multiply`.
        """
        ...
    @overload
    def multiply(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        multiply(value) -> Tensor

        See :func:`torch.multiply`.
        """
        ...
    @overload
    def multiply_(self, other: Tensor) -> Tensor:
        r"""
        multiply_(value) -> Tensor

        In-place version of :meth:`~Tensor.multiply`.
        """
        ...
    @overload
    def multiply_(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        multiply_(value) -> Tensor

        In-place version of :meth:`~Tensor.multiply`.
        """
        ...
    def mv(self, vec: Tensor) -> Tensor:
        r"""
        mv(vec) -> Tensor

        See :func:`torch.mv`
        """
        ...
    def mvlgamma(self, p: _int) -> Tensor:
        r"""
        mvlgamma(p) -> Tensor

        See :func:`torch.mvlgamma`
        """
        ...
    def mvlgamma_(self, p: _int) -> Tensor:
        r"""
        mvlgamma_(p) -> Tensor

        In-place version of :meth:`~Tensor.mvlgamma`
        """
        ...
    def nan_to_num(self, nan: Optional[_float] = None, posinf: Optional[_float] = None, neginf: Optional[_float] = None) -> Tensor:
        r"""
        nan_to_num(nan=0.0, posinf=None, neginf=None) -> Tensor

        See :func:`torch.nan_to_num`.
        """
        ...
    def nan_to_num_(self, nan: Optional[_float] = None, posinf: Optional[_float] = None, neginf: Optional[_float] = None) -> Tensor:
        r"""
        nan_to_num_(nan=0.0, posinf=None, neginf=None) -> Tensor

        In-place version of :meth:`~Tensor.nan_to_num`.
        """
        ...
    def nanmean(self, dim: Optional[Union[_int, _size]] = None, keepdim: _bool = False, *, dtype: Optional[_dtype] = None) -> Tensor:
        r"""
        nanmean(dim=None, keepdim=False, *, dtype=None) -> Tensor

        See :func:`torch.nanmean`
        """
        ...
    @overload
    def nanmedian(self) -> Tensor:
        r"""
        nanmedian(dim=None, keepdim=False) -> (Tensor, LongTensor)

        See :func:`torch.nanmedian`
        """
        ...
    @overload
    def nanmedian(self, dim: _int, keepdim: _bool = False) -> torch.return_types.nanmedian:
        r"""
        nanmedian(dim=None, keepdim=False) -> (Tensor, LongTensor)

        See :func:`torch.nanmedian`
        """
        ...
    @overload
    def nanmedian(self, dim: Union[str, ellipsis, None], keepdim: _bool = False) -> torch.return_types.nanmedian:
        r"""
        nanmedian(dim=None, keepdim=False) -> (Tensor, LongTensor)

        See :func:`torch.nanmedian`
        """
        ...
    @overload
    def nanquantile(self, q: Tensor, dim: Optional[_int] = None, keepdim: _bool = False, *, interpolation: str = "linear") -> Tensor:
        r"""
        nanquantile(q, dim=None, keepdim=False, *, interpolation='linear') -> Tensor

        See :func:`torch.nanquantile`
        """
        ...
    @overload
    def nanquantile(self, q: _float, dim: Optional[_int] = None, keepdim: _bool = False, *, interpolation: str = "linear") -> Tensor:
        r"""
        nanquantile(q, dim=None, keepdim=False, *, interpolation='linear') -> Tensor

        See :func:`torch.nanquantile`
        """
        ...
    def nansum(self, dim: Optional[Union[_int, _size]] = None, keepdim: _bool = False, *, dtype: Optional[_dtype] = None) -> Tensor:
        r"""
        nansum(dim=None, keepdim=False, dtype=None) -> Tensor

        See :func:`torch.nansum`
        """
        ...
    @overload
    def narrow(self, dim: _int, start: Tensor, length: Union[_int, SymInt]) -> Tensor:
        r"""
        narrow(dimension, start, length) -> Tensor

        See :func:`torch.narrow`.
        """
        ...
    @overload
    def narrow(self, dim: _int, start: Union[_int, SymInt], length: Union[_int, SymInt]) -> Tensor:
        r"""
        narrow(dimension, start, length) -> Tensor

        See :func:`torch.narrow`.
        """
        ...
    def narrow_copy(self, dim: _int, start: Union[_int, SymInt], length: Union[_int, SymInt]) -> Tensor:
        r"""
        narrow_copy(dimension, start, length) -> Tensor

        See :func:`torch.narrow_copy`.
        """
        ...
    def ndimension(self) -> _int:
        r"""
        ndimension() -> int

        Alias for :meth:`~Tensor.dim()`
        """
        ...
    @overload
    def ne(self, other: Tensor) -> Tensor:
        r"""
        ne(other) -> Tensor

        See :func:`torch.ne`.
        """
        ...
    @overload
    def ne(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        ne(other) -> Tensor

        See :func:`torch.ne`.
        """
        ...
    @overload
    def ne_(self, other: Tensor) -> Tensor:
        r"""
        ne_(other) -> Tensor

        In-place version of :meth:`~Tensor.ne`.
        """
        ...
    @overload
    def ne_(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        ne_(other) -> Tensor

        In-place version of :meth:`~Tensor.ne`.
        """
        ...
    def neg(self) -> Tensor:
        r"""
        neg() -> Tensor

        See :func:`torch.neg`
        """
        ...
    def neg_(self) -> Tensor:
        r"""
        neg_() -> Tensor

        In-place version of :meth:`~Tensor.neg`
        """
        ...
    def negative(self) -> Tensor:
        r"""
        negative() -> Tensor

        See :func:`torch.negative`
        """
        ...
    def negative_(self) -> Tensor:
        r"""
        negative_() -> Tensor

        In-place version of :meth:`~Tensor.negative`
        """
        ...
    def nelement(self) -> _int:
        r"""
        nelement() -> int

        Alias for :meth:`~Tensor.numel`
        """
        ...
    @overload
    def new(self, *args: Any, device: Optional[DeviceLikeType] = None) -> Tensor: ...
    @overload
    def new(self, storage: Storage) -> Tensor: ...
    @overload
    def new(self, other: Tensor) -> Tensor: ...
    @overload
    def new(self, size: _size, *, device: Optional[DeviceLikeType] = None) -> Tensor: ...
    @overload
    def new_empty(self, size: Sequence[Union[_int, SymInt]], *, dtype: Optional[_dtype] = None, layout: Optional[_layout] = None, device: Optional[Optional[DeviceLikeType]] = None, pin_memory: Optional[_bool] = False, requires_grad: Optional[_bool] = False) -> Tensor:
        r"""
        new_empty(size, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, pin_memory=False) -> Tensor


        Returns a Tensor of size :attr:`size` filled with uninitialized data.
        By default, the returned Tensor has the same :class:`torch.dtype` and
        :class:`torch.device` as this tensor.

        Args:
            size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
                shape of the output tensor.

        Keyword args:
            dtype (:class:`torch.dtype`, optional): the desired type of returned tensor.
                Default: if None, same :class:`torch.dtype` as this tensor.
            device (:class:`torch.device`, optional): the desired device of returned tensor.
                Default: if None, same :class:`torch.device` as this tensor.
            requires_grad (bool, optional): If autograd should record operations on the
                returned tensor. Default: ``False``.
            layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
                Default: ``torch.strided``.
            pin_memory (bool, optional): If set, returned tensor would be allocated in
                the pinned memory. Works only for CPU tensors. Default: ``False``.

        Example::

            >>> tensor = torch.ones(())
            >>> tensor.new_empty((2, 3))
            tensor([[ 5.8182e-18,  4.5765e-41, -1.0545e+30],
                    [ 3.0949e-41,  4.4842e-44,  0.0000e+00]])
        """
        ...
    @overload
    def new_empty(self, *size: _int, dtype: Optional[_dtype] = None, layout: Optional[_layout] = None, device: Optional[Optional[DeviceLikeType]] = None, pin_memory: Optional[_bool] = False, requires_grad: Optional[_bool] = False) -> Tensor:
        r"""
        new_empty(size, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, pin_memory=False) -> Tensor


        Returns a Tensor of size :attr:`size` filled with uninitialized data.
        By default, the returned Tensor has the same :class:`torch.dtype` and
        :class:`torch.device` as this tensor.

        Args:
            size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
                shape of the output tensor.

        Keyword args:
            dtype (:class:`torch.dtype`, optional): the desired type of returned tensor.
                Default: if None, same :class:`torch.dtype` as this tensor.
            device (:class:`torch.device`, optional): the desired device of returned tensor.
                Default: if None, same :class:`torch.device` as this tensor.
            requires_grad (bool, optional): If autograd should record operations on the
                returned tensor. Default: ``False``.
            layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
                Default: ``torch.strided``.
            pin_memory (bool, optional): If set, returned tensor would be allocated in
                the pinned memory. Works only for CPU tensors. Default: ``False``.

        Example::

            >>> tensor = torch.ones(())
            >>> tensor.new_empty((2, 3))
            tensor([[ 5.8182e-18,  4.5765e-41, -1.0545e+30],
                    [ 3.0949e-41,  4.4842e-44,  0.0000e+00]])
        """
        ...
    def new_empty_strided(self, size: Sequence[Union[_int, SymInt]], stride: Sequence[Union[_int, SymInt]], *, dtype: Optional[_dtype] = None, layout: Optional[_layout] = None, device: Optional[Optional[DeviceLikeType]] = None, pin_memory: Optional[_bool] = False, requires_grad: Optional[_bool] = False) -> Tensor:
        r"""
        new_empty_strided(size, stride, dtype=None, device=None, requires_grad=False, layout=torch.strided, pin_memory=False) -> Tensor


        Returns a Tensor of size :attr:`size` and strides :attr:`stride` filled with
        uninitialized data. By default, the returned Tensor has the same
        :class:`torch.dtype` and :class:`torch.device` as this tensor.

        Args:
            size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
                shape of the output tensor.

        Keyword args:
            dtype (:class:`torch.dtype`, optional): the desired type of returned tensor.
                Default: if None, same :class:`torch.dtype` as this tensor.
            device (:class:`torch.device`, optional): the desired device of returned tensor.
                Default: if None, same :class:`torch.device` as this tensor.
            requires_grad (bool, optional): If autograd should record operations on the
                returned tensor. Default: ``False``.
            layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
                Default: ``torch.strided``.
            pin_memory (bool, optional): If set, returned tensor would be allocated in
                the pinned memory. Works only for CPU tensors. Default: ``False``.

        Example::

            >>> tensor = torch.ones(())
            >>> tensor.new_empty_strided((2, 3), (3, 1))
            tensor([[ 5.8182e-18,  4.5765e-41, -1.0545e+30],
                    [ 3.0949e-41,  4.4842e-44,  0.0000e+00]])
        """
        ...
    def new_full(self, size: Sequence[Union[_int, SymInt]], fill_value: Union[Number, _complex], *, dtype: Optional[_dtype] = None, layout: Optional[_layout] = None, device: Optional[Optional[DeviceLikeType]] = None, pin_memory: Optional[_bool] = False, requires_grad: Optional[_bool] = False) -> Tensor:
        r"""
        new_full(size, fill_value, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, pin_memory=False) -> Tensor


        Returns a Tensor of size :attr:`size` filled with :attr:`fill_value`.
        By default, the returned Tensor has the same :class:`torch.dtype` and
        :class:`torch.device` as this tensor.

        Args:
            fill_value (scalar): the number to fill the output tensor with.

        Keyword args:
            dtype (:class:`torch.dtype`, optional): the desired type of returned tensor.
                Default: if None, same :class:`torch.dtype` as this tensor.
            device (:class:`torch.device`, optional): the desired device of returned tensor.
                Default: if None, same :class:`torch.device` as this tensor.
            requires_grad (bool, optional): If autograd should record operations on the
                returned tensor. Default: ``False``.
            layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
                Default: ``torch.strided``.
            pin_memory (bool, optional): If set, returned tensor would be allocated in
                the pinned memory. Works only for CPU tensors. Default: ``False``.

        Example::

            >>> tensor = torch.ones((2,), dtype=torch.float64)
            >>> tensor.new_full((3, 4), 3.141592)
            tensor([[ 3.1416,  3.1416,  3.1416,  3.1416],
                    [ 3.1416,  3.1416,  3.1416,  3.1416],
                    [ 3.1416,  3.1416,  3.1416,  3.1416]], dtype=torch.float64)
        """
        ...
    @overload
    def new_ones(self, size: _size, dtype: Optional[_dtype] = None, device: Optional[DeviceLikeType] = None, requires_grad: _bool = False, pin_memory: _bool = False) -> Tensor:
        r"""
        new_ones(size, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, pin_memory=False) -> Tensor


        Returns a Tensor of size :attr:`size` filled with ``1``.
        By default, the returned Tensor has the same :class:`torch.dtype` and
        :class:`torch.device` as this tensor.

        Args:
            size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
                shape of the output tensor.

        Keyword args:
            dtype (:class:`torch.dtype`, optional): the desired type of returned tensor.
                Default: if None, same :class:`torch.dtype` as this tensor.
            device (:class:`torch.device`, optional): the desired device of returned tensor.
                Default: if None, same :class:`torch.device` as this tensor.
            requires_grad (bool, optional): If autograd should record operations on the
                returned tensor. Default: ``False``.
            layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
                Default: ``torch.strided``.
            pin_memory (bool, optional): If set, returned tensor would be allocated in
                the pinned memory. Works only for CPU tensors. Default: ``False``.

        Example::

            >>> tensor = torch.tensor((), dtype=torch.int32)
            >>> tensor.new_ones((2, 3))
            tensor([[ 1,  1,  1],
                    [ 1,  1,  1]], dtype=torch.int32)
        """
        ...
    @overload
    def new_ones(self, size: Sequence[Union[_int, SymInt]], *, dtype: Optional[_dtype] = None, layout: Optional[_layout] = None, device: Optional[Optional[DeviceLikeType]] = None, pin_memory: Optional[_bool] = False, requires_grad: Optional[_bool] = False) -> Tensor:
        r"""
        new_ones(size, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, pin_memory=False) -> Tensor


        Returns a Tensor of size :attr:`size` filled with ``1``.
        By default, the returned Tensor has the same :class:`torch.dtype` and
        :class:`torch.device` as this tensor.

        Args:
            size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
                shape of the output tensor.

        Keyword args:
            dtype (:class:`torch.dtype`, optional): the desired type of returned tensor.
                Default: if None, same :class:`torch.dtype` as this tensor.
            device (:class:`torch.device`, optional): the desired device of returned tensor.
                Default: if None, same :class:`torch.device` as this tensor.
            requires_grad (bool, optional): If autograd should record operations on the
                returned tensor. Default: ``False``.
            layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
                Default: ``torch.strided``.
            pin_memory (bool, optional): If set, returned tensor would be allocated in
                the pinned memory. Works only for CPU tensors. Default: ``False``.

        Example::

            >>> tensor = torch.tensor((), dtype=torch.int32)
            >>> tensor.new_ones((2, 3))
            tensor([[ 1,  1,  1],
                    [ 1,  1,  1]], dtype=torch.int32)
        """
        ...
    @overload
    def new_ones(self, *size: _int, dtype: Optional[_dtype] = None, layout: Optional[_layout] = None, device: Optional[Optional[DeviceLikeType]] = None, pin_memory: Optional[_bool] = False, requires_grad: Optional[_bool] = False) -> Tensor:
        r"""
        new_ones(size, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, pin_memory=False) -> Tensor


        Returns a Tensor of size :attr:`size` filled with ``1``.
        By default, the returned Tensor has the same :class:`torch.dtype` and
        :class:`torch.device` as this tensor.

        Args:
            size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
                shape of the output tensor.

        Keyword args:
            dtype (:class:`torch.dtype`, optional): the desired type of returned tensor.
                Default: if None, same :class:`torch.dtype` as this tensor.
            device (:class:`torch.device`, optional): the desired device of returned tensor.
                Default: if None, same :class:`torch.device` as this tensor.
            requires_grad (bool, optional): If autograd should record operations on the
                returned tensor. Default: ``False``.
            layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
                Default: ``torch.strided``.
            pin_memory (bool, optional): If set, returned tensor would be allocated in
                the pinned memory. Works only for CPU tensors. Default: ``False``.

        Example::

            >>> tensor = torch.tensor((), dtype=torch.int32)
            >>> tensor.new_ones((2, 3))
            tensor([[ 1,  1,  1],
                    [ 1,  1,  1]], dtype=torch.int32)
        """
        ...
    def new_tensor(self, data: Any, dtype: Optional[_dtype] = None, device: Optional[DeviceLikeType] = None, requires_grad: _bool = False, pin_memory: _bool = False) -> Tensor:
        r"""
        new_tensor(data, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, pin_memory=False) -> Tensor


        Returns a new Tensor with :attr:`data` as the tensor data.
        By default, the returned Tensor has the same :class:`torch.dtype` and
        :class:`torch.device` as this tensor.

        .. warning::

            :func:`new_tensor` always copies :attr:`data`. If you have a Tensor
            ``data`` and want to avoid a copy, use :func:`torch.Tensor.requires_grad_`
            or :func:`torch.Tensor.detach`.
            If you have a numpy array and want to avoid a copy, use
            :func:`torch.from_numpy`.

        .. warning::

            When data is a tensor `x`, :func:`new_tensor()` reads out 'the data' from whatever it is passed,
            and constructs a leaf variable. Therefore ``tensor.new_tensor(x)`` is equivalent to ``x.clone().detach()``
            and ``tensor.new_tensor(x, requires_grad=True)`` is equivalent to ``x.clone().detach().requires_grad_(True)``.
            The equivalents using ``clone()`` and ``detach()`` are recommended.

        Args:
            data (array_like): The returned Tensor copies :attr:`data`.

        Keyword args:
            dtype (:class:`torch.dtype`, optional): the desired type of returned tensor.
                Default: if None, same :class:`torch.dtype` as this tensor.
            device (:class:`torch.device`, optional): the desired device of returned tensor.
                Default: if None, same :class:`torch.device` as this tensor.
            requires_grad (bool, optional): If autograd should record operations on the
                returned tensor. Default: ``False``.
            layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
                Default: ``torch.strided``.
            pin_memory (bool, optional): If set, returned tensor would be allocated in
                the pinned memory. Works only for CPU tensors. Default: ``False``.

        Example::

            >>> tensor = torch.ones((2,), dtype=torch.int8)
            >>> data = [[0, 1], [2, 3]]
            >>> tensor.new_tensor(data)
            tensor([[ 0,  1],
                    [ 2,  3]], dtype=torch.int8)
        """
        ...
    @overload
    def new_zeros(self, size: Sequence[Union[_int, SymInt]], *, dtype: Optional[_dtype] = None, layout: Optional[_layout] = None, device: Optional[Optional[DeviceLikeType]] = None, pin_memory: Optional[_bool] = False, requires_grad: Optional[_bool] = False) -> Tensor:
        r"""
        new_zeros(size, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, pin_memory=False) -> Tensor


        Returns a Tensor of size :attr:`size` filled with ``0``.
        By default, the returned Tensor has the same :class:`torch.dtype` and
        :class:`torch.device` as this tensor.

        Args:
            size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
                shape of the output tensor.

        Keyword args:
            dtype (:class:`torch.dtype`, optional): the desired type of returned tensor.
                Default: if None, same :class:`torch.dtype` as this tensor.
            device (:class:`torch.device`, optional): the desired device of returned tensor.
                Default: if None, same :class:`torch.device` as this tensor.
            requires_grad (bool, optional): If autograd should record operations on the
                returned tensor. Default: ``False``.
            layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
                Default: ``torch.strided``.
            pin_memory (bool, optional): If set, returned tensor would be allocated in
                the pinned memory. Works only for CPU tensors. Default: ``False``.

        Example::

            >>> tensor = torch.tensor((), dtype=torch.float64)
            >>> tensor.new_zeros((2, 3))
            tensor([[ 0.,  0.,  0.],
                    [ 0.,  0.,  0.]], dtype=torch.float64)
        """
        ...
    @overload
    def new_zeros(self, *size: _int, dtype: Optional[_dtype] = None, layout: Optional[_layout] = None, device: Optional[Optional[DeviceLikeType]] = None, pin_memory: Optional[_bool] = False, requires_grad: Optional[_bool] = False) -> Tensor:
        r"""
        new_zeros(size, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, pin_memory=False) -> Tensor


        Returns a Tensor of size :attr:`size` filled with ``0``.
        By default, the returned Tensor has the same :class:`torch.dtype` and
        :class:`torch.device` as this tensor.

        Args:
            size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
                shape of the output tensor.

        Keyword args:
            dtype (:class:`torch.dtype`, optional): the desired type of returned tensor.
                Default: if None, same :class:`torch.dtype` as this tensor.
            device (:class:`torch.device`, optional): the desired device of returned tensor.
                Default: if None, same :class:`torch.device` as this tensor.
            requires_grad (bool, optional): If autograd should record operations on the
                returned tensor. Default: ``False``.
            layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
                Default: ``torch.strided``.
            pin_memory (bool, optional): If set, returned tensor would be allocated in
                the pinned memory. Works only for CPU tensors. Default: ``False``.

        Example::

            >>> tensor = torch.tensor((), dtype=torch.float64)
            >>> tensor.new_zeros((2, 3))
            tensor([[ 0.,  0.,  0.],
                    [ 0.,  0.,  0.]], dtype=torch.float64)
        """
        ...
    def nextafter(self, other: Tensor) -> Tensor:
        r"""
        nextafter(other) -> Tensor
        See :func:`torch.nextafter`
        """
        ...
    def nextafter_(self, other: Tensor) -> Tensor:
        r"""
        nextafter_(other) -> Tensor
        In-place version of :meth:`~Tensor.nextafter`
        """
        ...
    @overload
    def nonzero(self, *, as_tuple: Literal[False] = False) -> Tensor:
        r"""
        nonzero() -> LongTensor

        See :func:`torch.nonzero`
        """
        ...
    @overload
    def nonzero(self, *, as_tuple: Literal[True]) -> Tuple[Tensor, ...]:
        r"""
        nonzero() -> LongTensor

        See :func:`torch.nonzero`
        """
        ...
    def nonzero_static(self, *, size: _int, fill_value: _int = -1) -> Tensor:
        r"""
        nonzero_static(input, *, size, fill_value=-1) -> Tensor

        Returns a 2-D tensor where each row is the index for a non-zero value.
        The returned Tensor has the same `torch.dtype` as `torch.nonzero()`.

        Args:
            input (Tensor): the input tensor to count non-zero elements.

        Keyword args:
            size (int): the size of non-zero elements expected to be included in the out
                tensor. Pad the out tensor with `fill_value` if the `size` is larger
                than total number of non-zero elements, truncate out tensor if `size`
                is smaller. The size must be a non-negative integer.
            fill_value (int): the value to fill the output tensor with when `size` is larger
                than the total number of non-zero elements. Default is `-1` to represent
                invalid index.

        Example:

            # Example 1: Padding
            >>> input_tensor = torch.tensor([[1, 0], [3, 2]])
            >>> static_size = 4
            >>> t = torch.nonzero_static(input_tensor, size = static_size)
            tensor([[  0,   0],
                    [  1,   0],
                    [  1,   1],
                    [  -1, -1]], dtype=torch.int64)

            # Example 2: Truncating
            >>> input_tensor = torch.tensor([[1, 0], [3, 2]])
            >>> static_size = 2
            >>> t = torch.nonzero_static(input_tensor, size = static_size)
            tensor([[  0,   0],
                    [  1,   0]], dtype=torch.int64)

            # Example 3: 0 size
            >>> input_tensor = torch.tensor([10])
            >>> static_size = 0
            >>> t = torch.nonzero_static(input_tensor, size = static_size)
            tensor([], size=(0, 1), dtype=torch.int64)

            # Example 4: 0 rank input
            >>> input_tensor = torch.tensor(10)
            >>> static_size = 2
            >>> t = torch.nonzero_static(input_tensor, size = static_size)
            tensor([], size=(2, 0), dtype=torch.int64)
        """
        ...
    def normal_(self, mean: _float = 0, std: _float = 1, *, generator: Optional[Generator] = None) -> Tensor:
        r"""
        normal_(mean=0, std=1, *, generator=None) -> Tensor

        Fills :attr:`self` tensor with elements samples from the normal distribution
        parameterized by :attr:`mean` and :attr:`std`.
        """
        ...
    @overload
    def not_equal(self, other: Tensor) -> Tensor:
        r"""
        not_equal(other) -> Tensor

        See :func:`torch.not_equal`.
        """
        ...
    @overload
    def not_equal(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        not_equal(other) -> Tensor

        See :func:`torch.not_equal`.
        """
        ...
    @overload
    def not_equal_(self, other: Tensor) -> Tensor:
        r"""
        not_equal_(other) -> Tensor

        In-place version of :meth:`~Tensor.not_equal`.
        """
        ...
    @overload
    def not_equal_(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        not_equal_(other) -> Tensor

        In-place version of :meth:`~Tensor.not_equal`.
        """
        ...
    def numel(self) -> _int:
        r"""
        numel() -> int

        See :func:`torch.numel`
        """
        ...
    def numpy(self, *, force: _bool = False) -> numpy.ndarray:
        r"""
        numpy(*, force=False) -> numpy.ndarray

        Returns the tensor as a NumPy :class:`ndarray`.

        If :attr:`force` is ``False`` (the default), the conversion
        is performed only if the tensor is on the CPU, does not require grad,
        does not have its conjugate bit set, and is a dtype and layout that
        NumPy supports. The returned ndarray and the tensor will share their
        storage, so changes to the tensor will be reflected in the ndarray
        and vice versa.

        If :attr:`force` is ``True`` this is equivalent to
        calling ``t.detach().cpu().resolve_conj().resolve_neg().numpy()``.
        If the tensor isn't on the CPU or the conjugate or negative bit is set,
        the tensor won't share its storage with the returned ndarray.
        Setting :attr:`force` to ``True`` can be a useful shorthand.

        Args:
            force (bool): if ``True``, the ndarray may be a copy of the tensor
                       instead of always sharing memory, defaults to ``False``.
        """
        ...
    def orgqr(self, input2: Tensor) -> Tensor:
        r"""
        orgqr(input2) -> Tensor

        See :func:`torch.orgqr`
        """
        ...
    def ormqr(self, input2: Tensor, input3: Tensor, left: _bool = True, transpose: _bool = False) -> Tensor:
        r"""
        ormqr(input2, input3, left=True, transpose=False) -> Tensor

        See :func:`torch.ormqr`
        """
        ...
    def outer(self, vec2: Tensor) -> Tensor:
        r"""
        outer(vec2) -> Tensor

        See :func:`torch.outer`.
        """
        ...
    @overload
    def permute(self, dims: _size) -> Tensor:
        r"""
        permute(*dims) -> Tensor

        See :func:`torch.permute`
        """
        ...
    @overload
    def permute(self, *dims: _int) -> Tensor:
        r"""
        permute(*dims) -> Tensor

        See :func:`torch.permute`
        """
        ...
    def pin_memory(self, device: Optional[Optional[DeviceLikeType]] = None) -> Tensor:
        r"""
        pin_memory() -> Tensor

        Copies the tensor to pinned memory, if it's not already pinned.
        """
        ...
    def pinverse(self, rcond: _float = 1e-15) -> Tensor:
        r"""
        pinverse() -> Tensor

        See :func:`torch.pinverse`
        """
        ...
    def polygamma(self, n: _int) -> Tensor:
        r"""
        polygamma(n) -> Tensor

        See :func:`torch.polygamma`
        """
        ...
    def polygamma_(self, n: _int) -> Tensor:
        r"""
        polygamma_(n) -> Tensor

        In-place version of :meth:`~Tensor.polygamma`
        """
        ...
    def positive(self) -> Tensor:
        r"""
        positive() -> Tensor

        See :func:`torch.positive`
        """
        ...
    @overload
    def pow(self, exponent: Tensor) -> Tensor:
        r"""
        pow(exponent) -> Tensor

        See :func:`torch.pow`
        """
        ...
    @overload
    def pow(self, exponent: Union[Number, _complex]) -> Tensor:
        r"""
        pow(exponent) -> Tensor

        See :func:`torch.pow`
        """
        ...
    @overload
    def pow_(self, exponent: Tensor) -> Tensor:
        r"""
        pow_(exponent) -> Tensor

        In-place version of :meth:`~Tensor.pow`
        """
        ...
    @overload
    def pow_(self, exponent: Union[Number, _complex]) -> Tensor:
        r"""
        pow_(exponent) -> Tensor

        In-place version of :meth:`~Tensor.pow`
        """
        ...
    def prelu(self, weight: Tensor) -> Tensor: ...
    @overload
    def prod(self, *, dtype: Optional[_dtype] = None) -> Tensor:
        r"""
        prod(dim=None, keepdim=False, dtype=None) -> Tensor

        See :func:`torch.prod`
        """
        ...
    @overload
    def prod(self, dim: _int, keepdim: _bool = False, *, dtype: Optional[_dtype] = None) -> Tensor:
        r"""
        prod(dim=None, keepdim=False, dtype=None) -> Tensor

        See :func:`torch.prod`
        """
        ...
    @overload
    def prod(self, dim: Union[str, ellipsis, None], keepdim: _bool = False, *, dtype: Optional[_dtype] = None) -> Tensor:
        r"""
        prod(dim=None, keepdim=False, dtype=None) -> Tensor

        See :func:`torch.prod`
        """
        ...
    def put(self, index: Tensor, source: Tensor, accumulate: _bool = False) -> Tensor:
        r"""
        put(input, index, source, accumulate=False) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.put_`.
        `input` corresponds to `self` in :meth:`torch.Tensor.put_`.
        """
        ...
    def put_(self, index: Tensor, source: Tensor, accumulate: _bool = False) -> Tensor:
        r"""
        put_(index, source, accumulate=False) -> Tensor

        Copies the elements from :attr:`source` into the positions specified by
        :attr:`index`. For the purpose of indexing, the :attr:`self` tensor is treated as if
        it were a 1-D tensor.

        :attr:`index` and :attr:`source` need to have the same number of elements, but not necessarily
        the same shape.

        If :attr:`accumulate` is ``True``, the elements in :attr:`source` are added to
        :attr:`self`. If accumulate is ``False``, the behavior is undefined if :attr:`index`
        contain duplicate elements.

        Args:
            index (LongTensor): the indices into self
            source (Tensor): the tensor containing values to copy from
            accumulate (bool): whether to accumulate into self

        Example::

            >>> src = torch.tensor([[4, 3, 5],
            ...                     [6, 7, 8]])
            >>> src.put_(torch.tensor([1, 3]), torch.tensor([9, 10]))
            tensor([[  4,   9,   5],
                    [ 10,   7,   8]])
        """
        ...
    def q_per_channel_axis(self) -> _int:
        r"""
        q_per_channel_axis() -> int

        Given a Tensor quantized by linear (affine) per-channel quantization,
        returns the index of dimension on which per-channel quantization is applied.
        """
        ...
    def q_per_channel_scales(self) -> Tensor:
        r"""
        q_per_channel_scales() -> Tensor

        Given a Tensor quantized by linear (affine) per-channel quantization,
        returns a Tensor of scales of the underlying quantizer. It has the number of
        elements that matches the corresponding dimensions (from q_per_channel_axis) of
        the tensor.
        """
        ...
    def q_per_channel_zero_points(self) -> Tensor:
        r"""
        q_per_channel_zero_points() -> Tensor

        Given a Tensor quantized by linear (affine) per-channel quantization,
        returns a tensor of zero_points of the underlying quantizer. It has the number of
        elements that matches the corresponding dimensions (from q_per_channel_axis) of
        the tensor.
        """
        ...
    def q_scale(self) -> _float:
        r"""
        q_scale() -> float

        Given a Tensor quantized by linear(affine) quantization,
        returns the scale of the underlying quantizer().
        """
        ...
    def q_zero_point(self) -> _int:
        r"""
        q_zero_point() -> int

        Given a Tensor quantized by linear(affine) quantization,
        returns the zero_point of the underlying quantizer().
        """
        ...
    def qr(self, some: _bool = True) -> torch.return_types.qr:
        r"""
        qr(some=True) -> (Tensor, Tensor)

        See :func:`torch.qr`
        """
        ...
    def qscheme(self) -> _qscheme:
        r"""
        qscheme() -> torch.qscheme

        Returns the quantization scheme of a given QTensor.
        """
        ...
    @overload
    def quantile(self, q: Tensor, dim: Optional[_int] = None, keepdim: _bool = False, *, interpolation: str = "linear") -> Tensor:
        r"""
        quantile(q, dim=None, keepdim=False, *, interpolation='linear') -> Tensor

        See :func:`torch.quantile`
        """
        ...
    @overload
    def quantile(self, q: _float, dim: Optional[_int] = None, keepdim: _bool = False, *, interpolation: str = "linear") -> Tensor:
        r"""
        quantile(q, dim=None, keepdim=False, *, interpolation='linear') -> Tensor

        See :func:`torch.quantile`
        """
        ...
    def rad2deg(self) -> Tensor:
        r"""
        rad2deg() -> Tensor

        See :func:`torch.rad2deg`
        """
        ...
    def rad2deg_(self) -> Tensor:
        r"""
        rad2deg_() -> Tensor

        In-place version of :meth:`~Tensor.rad2deg`
        """
        ...
    @overload
    def random_(self, *, generator: Optional[Generator] = None) -> Tensor:
        r"""
        random_(from=0, to=None, *, generator=None) -> Tensor

        Fills :attr:`self` tensor with numbers sampled from the discrete uniform
        distribution over ``[from, to - 1]``. If not specified, the values are usually
        only bounded by :attr:`self` tensor's data type. However, for floating point
        types, if unspecified, range will be ``[0, 2^mantissa]`` to ensure that every
        value is representable. For example, `torch.tensor(1, dtype=torch.double).random_()`
        will be uniform in ``[0, 2^53]``.
        """
        ...
    @overload
    def random_(self, from_: _int, to: Optional[_int], *, generator: Optional[Generator] = None) -> Tensor:
        r"""
        random_(from=0, to=None, *, generator=None) -> Tensor

        Fills :attr:`self` tensor with numbers sampled from the discrete uniform
        distribution over ``[from, to - 1]``. If not specified, the values are usually
        only bounded by :attr:`self` tensor's data type. However, for floating point
        types, if unspecified, range will be ``[0, 2^mantissa]`` to ensure that every
        value is representable. For example, `torch.tensor(1, dtype=torch.double).random_()`
        will be uniform in ``[0, 2^53]``.
        """
        ...
    @overload
    def random_(self, to: _int, *, generator: Optional[Generator] = None) -> Tensor:
        r"""
        random_(from=0, to=None, *, generator=None) -> Tensor

        Fills :attr:`self` tensor with numbers sampled from the discrete uniform
        distribution over ``[from, to - 1]``. If not specified, the values are usually
        only bounded by :attr:`self` tensor's data type. However, for floating point
        types, if unspecified, range will be ``[0, 2^mantissa]`` to ensure that every
        value is representable. For example, `torch.tensor(1, dtype=torch.double).random_()`
        will be uniform in ``[0, 2^53]``.
        """
        ...
    def ravel(self) -> Tensor:
        r"""
        ravel() -> Tensor

        see :func:`torch.ravel`
        """
        ...
    def reciprocal(self) -> Tensor:
        r"""
        reciprocal() -> Tensor

        See :func:`torch.reciprocal`
        """
        ...
    def reciprocal_(self) -> Tensor:
        r"""
        reciprocal_() -> Tensor

        In-place version of :meth:`~Tensor.reciprocal`
        """
        ...
    def record_stream(self, s: Stream) -> None:
        r"""
        record_stream(stream)

        Marks the tensor as having been used by this stream.  When the tensor
        is deallocated, ensure the tensor memory is not reused for another tensor
        until all work queued on :attr:`stream` at the time of deallocation is
        complete.

        .. note::

            The caching allocator is aware of only the stream where a tensor was
            allocated. Due to the awareness, it already correctly manages the life
            cycle of tensors on only one stream. But if a tensor is used on a stream
            different from the stream of origin, the allocator might reuse the memory
            unexpectedly. Calling this method lets the allocator know which streams
            have used the tensor.

        .. warning::

            This method is most suitable for use cases where you are providing a
            function that created a tensor on a side stream, and want users to be able
            to make use of the tensor without having to think carefully about stream
            safety when making use of them.  These safety guarantees come at some
            performance and predictability cost (analogous to the tradeoff between GC
            and manual memory management), so if you are in a situation where
            you manage the full lifetime of your tensors, you may consider instead
            manually managing CUDA events so that calling this method is not necessary.
            In particular, when you call this method, on later allocations the
            allocator will poll the recorded stream to see if all operations have
            completed yet; you can potentially race with side stream computation and
            non-deterministically reuse or fail to reuse memory for an allocation.

            You can safely use tensors allocated on side streams without
            :meth:`~Tensor.record_stream`; you must manually ensure that
            any non-creation stream uses of a tensor are synced back to the creation
            stream before you deallocate the tensor.  As the CUDA caching allocator
            guarantees that the memory will only be reused with the same creation stream,
            this is sufficient to ensure that writes to future reallocations of the
            memory will be delayed until non-creation stream uses are done.
            (Counterintuitively, you may observe that on the CPU side we have already
            reallocated the tensor, even though CUDA kernels on the old tensor are
            still in progress.  This is fine, because CUDA operations on the new
            tensor will appropriately wait for the old operations to complete, as they
            are all on the same stream.)

            Concretely, this looks like this::

                with torch.cuda.stream(s0):
                    x = torch.zeros(N)

                s1.wait_stream(s0)
                with torch.cuda.stream(s1):
                    y = some_comm_op(x)

                ... some compute on s0 ...

                # synchronize creation stream s0 to side stream s1
                # before deallocating x
                s0.wait_stream(s1)
                del x

            Note that some discretion is required when deciding when to perform
            ``s0.wait_stream(s1)``.  In particular, if we were to wait immediately
            after ``some_comm_op``, there wouldn't be any point in having the side
            stream; it would be equivalent to have run ``some_comm_op`` on ``s0``.
            Instead, the synchronization must be placed at some appropriate, later
            point in time where you expect the side stream ``s1`` to have finished
            work.  This location is typically identified via profiling, e.g., using
            Chrome traces produced
            :meth:`torch.autograd.profiler.profile.export_chrome_trace`.  If you
            place the wait too early, work on s0 will block until ``s1`` has finished,
            preventing further overlapping of communication and computation.  If you
            place the wait too late, you will use more memory than is strictly
            necessary (as you are keeping ``x`` live for longer.)  For a concrete
            example of how this guidance can be applied in practice, see this post:
            `FSDP and CUDACachingAllocator
            <https://dev-discuss.pytorch.org/t/fsdp-cudacachingallocator-an-outsider-newb-perspective/1486>`_.
        """
        ...
    def refine_names(self, names: Sequence[Union[str, ellipsis, None]]) -> Tensor: ...
    def relu(self) -> Tensor: ...
    def relu_(self) -> Tensor: ...
    @overload
    def remainder(self, other: Tensor) -> Tensor:
        r"""
        remainder(divisor) -> Tensor

        See :func:`torch.remainder`
        """
        ...
    @overload
    def remainder(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        remainder(divisor) -> Tensor

        See :func:`torch.remainder`
        """
        ...
    @overload
    def remainder_(self, other: Tensor) -> Tensor:
        r"""
        remainder_(divisor) -> Tensor

        In-place version of :meth:`~Tensor.remainder`
        """
        ...
    @overload
    def remainder_(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        remainder_(divisor) -> Tensor

        In-place version of :meth:`~Tensor.remainder`
        """
        ...
    def rename(self, names: Optional[Sequence[Union[str, ellipsis, None]]]) -> Tensor: ...
    def rename_(self, names: Optional[Sequence[Union[str, ellipsis, None]]]) -> Tensor: ...
    def renorm(self, p: Union[Number, _complex], dim: _int, maxnorm: Union[Number, _complex]) -> Tensor:
        r"""
        renorm(p, dim, maxnorm) -> Tensor

        See :func:`torch.renorm`
        """
        ...
    def renorm_(self, p: Union[Number, _complex], dim: _int, maxnorm: Union[Number, _complex]) -> Tensor:
        r"""
        renorm_(p, dim, maxnorm) -> Tensor

        In-place version of :meth:`~Tensor.renorm`
        """
        ...
    @overload
    def repeat(self, repeats: Sequence[Union[_int, SymInt]]) -> Tensor:
        r"""
        repeat(*sizes) -> Tensor

        Repeats this tensor along the specified dimensions.

        Unlike :meth:`~Tensor.expand`, this function copies the tensor's data.

        .. warning::

            :meth:`~Tensor.repeat` behaves differently from
            `numpy.repeat <https://docs.scipy.org/doc/numpy/reference/generated/numpy.repeat.html>`_,
            but is more similar to
            `numpy.tile <https://docs.scipy.org/doc/numpy/reference/generated/numpy.tile.html>`_.
            For the operator similar to `numpy.repeat`, see :func:`torch.repeat_interleave`.

        Args:
            sizes (torch.Size or int...): The number of times to repeat this tensor along each
                dimension

        Example::

            >>> x = torch.tensor([1, 2, 3])
            >>> x.repeat(4, 2)
            tensor([[ 1,  2,  3,  1,  2,  3],
                    [ 1,  2,  3,  1,  2,  3],
                    [ 1,  2,  3,  1,  2,  3],
                    [ 1,  2,  3,  1,  2,  3]])
            >>> x.repeat(4, 2, 1).size()
            torch.Size([4, 2, 3])
        """
        ...
    @overload
    def repeat(self, *repeats: _int) -> Tensor:
        r"""
        repeat(*sizes) -> Tensor

        Repeats this tensor along the specified dimensions.

        Unlike :meth:`~Tensor.expand`, this function copies the tensor's data.

        .. warning::

            :meth:`~Tensor.repeat` behaves differently from
            `numpy.repeat <https://docs.scipy.org/doc/numpy/reference/generated/numpy.repeat.html>`_,
            but is more similar to
            `numpy.tile <https://docs.scipy.org/doc/numpy/reference/generated/numpy.tile.html>`_.
            For the operator similar to `numpy.repeat`, see :func:`torch.repeat_interleave`.

        Args:
            sizes (torch.Size or int...): The number of times to repeat this tensor along each
                dimension

        Example::

            >>> x = torch.tensor([1, 2, 3])
            >>> x.repeat(4, 2)
            tensor([[ 1,  2,  3,  1,  2,  3],
                    [ 1,  2,  3,  1,  2,  3],
                    [ 1,  2,  3,  1,  2,  3],
                    [ 1,  2,  3,  1,  2,  3]])
            >>> x.repeat(4, 2, 1).size()
            torch.Size([4, 2, 3])
        """
        ...
    @overload
    def repeat_interleave(self, repeats: Tensor, dim: Optional[_int] = None, *, output_size: Optional[Union[_int, SymInt]] = None) -> Tensor:
        r"""
        repeat_interleave(repeats, dim=None, *, output_size=None) -> Tensor

        See :func:`torch.repeat_interleave`.
        """
        ...
    @overload
    def repeat_interleave(self, repeats: Union[_int, SymInt], dim: Optional[_int] = None, *, output_size: Optional[Union[_int, SymInt]] = None) -> Tensor:
        r"""
        repeat_interleave(repeats, dim=None, *, output_size=None) -> Tensor

        See :func:`torch.repeat_interleave`.
        """
        ...
    def requires_grad_(self, mode: _bool = True) -> Tensor:
        r"""
        requires_grad_(requires_grad=True) -> Tensor

        Change if autograd should record operations on this tensor: sets this tensor's
        :attr:`requires_grad` attribute in-place. Returns this tensor.

        :func:`requires_grad_`'s main use case is to tell autograd to begin recording
        operations on a Tensor ``tensor``. If ``tensor`` has ``requires_grad=False``
        (because it was obtained through a DataLoader, or required preprocessing or
        initialization), ``tensor.requires_grad_()`` makes it so that autograd will
        begin to record operations on ``tensor``.

        Args:
            requires_grad (bool): If autograd should record operations on this tensor.
                Default: ``True``.

        Example::

            >>> # Let's say we want to preprocess some saved weights and use
            >>> # the result as new weights.
            >>> saved_weights = [0.1, 0.2, 0.3, 0.25]
            >>> loaded_weights = torch.tensor(saved_weights)
            >>> weights = preprocess(loaded_weights)  # some function
            >>> weights
            tensor([-0.5503,  0.4926, -2.1158, -0.8303])

            >>> # Now, start to record operations done to weights
            >>> weights.requires_grad_()
            >>> out = weights.pow(2).sum()
            >>> out.backward()
            >>> weights.grad
            tensor([-1.1007,  0.9853, -4.2316, -1.6606])
        """
        ...
    @overload
    def reshape(self, shape: Sequence[Union[_int, SymInt]]) -> Tensor:
        r"""
        reshape(*shape) -> Tensor

        Returns a tensor with the same data and number of elements as :attr:`self`
        but with the specified shape. This method returns a view if :attr:`shape` is
        compatible with the current shape. See :meth:`torch.Tensor.view` on when it is
        possible to return a view.

        See :func:`torch.reshape`

        Args:
            shape (tuple of ints or int...): the desired shape
        """
        ...
    @overload
    def reshape(self, *shape: _int) -> Tensor:
        r"""
        reshape(*shape) -> Tensor

        Returns a tensor with the same data and number of elements as :attr:`self`
        but with the specified shape. This method returns a view if :attr:`shape` is
        compatible with the current shape. See :meth:`torch.Tensor.view` on when it is
        possible to return a view.

        See :func:`torch.reshape`

        Args:
            shape (tuple of ints or int...): the desired shape
        """
        ...
    def reshape_as(self, other: Tensor) -> Tensor:
        r"""
        reshape_as(other) -> Tensor

        Returns this tensor as the same shape as :attr:`other`.
        ``self.reshape_as(other)`` is equivalent to ``self.reshape(other.sizes())``.
        This method returns a view if ``other.sizes()`` is compatible with the current
        shape. See :meth:`torch.Tensor.view` on when it is possible to return a view.

        Please see :meth:`reshape` for more information about ``reshape``.

        Args:
            other (:class:`torch.Tensor`): The result tensor has the same shape
                as :attr:`other`.
        """
        ...
    @overload
    def resize_(self, size: Sequence[Union[_int, SymInt]], *, memory_format: Optional[memory_format] = None) -> Tensor:
        r"""
        resize_(*sizes, memory_format=torch.contiguous_format) -> Tensor

        Resizes :attr:`self` tensor to the specified size. If the number of elements is
        larger than the current storage size, then the underlying storage is resized
        to fit the new number of elements. If the number of elements is smaller, the
        underlying storage is not changed. Existing elements are preserved but any new
        memory is uninitialized.

        .. warning::

            This is a low-level method. The storage is reinterpreted as C-contiguous,
            ignoring the current strides (unless the target size equals the current
            size, in which case the tensor is left unchanged). For most purposes, you
            will instead want to use :meth:`~Tensor.view()`, which checks for
            contiguity, or :meth:`~Tensor.reshape()`, which copies data if needed. To
            change the size in-place with custom strides, see :meth:`~Tensor.set_()`.

        .. note::

            If :func:`torch.use_deterministic_algorithms()` and
            :attr:`torch.utils.deterministic.fill_uninitialized_memory` are both set to
            ``True``, new elements are initialized to prevent nondeterministic behavior
            from using the result as an input to an operation. Floating point and
            complex values are set to NaN, and integer values are set to the maximum
            value.

        Args:
            sizes (torch.Size or int...): the desired size
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                Tensor. Default: ``torch.contiguous_format``. Note that memory format of
                :attr:`self` is going to be unaffected if ``self.size()`` matches ``sizes``.

        Example::

            >>> x = torch.tensor([[1, 2], [3, 4], [5, 6]])
            >>> x.resize_(2, 2)
            tensor([[ 1,  2],
                    [ 3,  4]])
        """
        ...
    @overload
    def resize_(self, *size: _int, memory_format: Optional[memory_format] = None) -> Tensor:
        r"""
        resize_(*sizes, memory_format=torch.contiguous_format) -> Tensor

        Resizes :attr:`self` tensor to the specified size. If the number of elements is
        larger than the current storage size, then the underlying storage is resized
        to fit the new number of elements. If the number of elements is smaller, the
        underlying storage is not changed. Existing elements are preserved but any new
        memory is uninitialized.

        .. warning::

            This is a low-level method. The storage is reinterpreted as C-contiguous,
            ignoring the current strides (unless the target size equals the current
            size, in which case the tensor is left unchanged). For most purposes, you
            will instead want to use :meth:`~Tensor.view()`, which checks for
            contiguity, or :meth:`~Tensor.reshape()`, which copies data if needed. To
            change the size in-place with custom strides, see :meth:`~Tensor.set_()`.

        .. note::

            If :func:`torch.use_deterministic_algorithms()` and
            :attr:`torch.utils.deterministic.fill_uninitialized_memory` are both set to
            ``True``, new elements are initialized to prevent nondeterministic behavior
            from using the result as an input to an operation. Floating point and
            complex values are set to NaN, and integer values are set to the maximum
            value.

        Args:
            sizes (torch.Size or int...): the desired size
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                Tensor. Default: ``torch.contiguous_format``. Note that memory format of
                :attr:`self` is going to be unaffected if ``self.size()`` matches ``sizes``.

        Example::

            >>> x = torch.tensor([[1, 2], [3, 4], [5, 6]])
            >>> x.resize_(2, 2)
            tensor([[ 1,  2],
                    [ 3,  4]])
        """
        ...
    def resize_as_(self, the_template: Tensor, *, memory_format: Optional[memory_format] = None) -> Tensor:
        r"""
        resize_as_(tensor, memory_format=torch.contiguous_format) -> Tensor

        Resizes the :attr:`self` tensor to be the same size as the specified
        :attr:`tensor`. This is equivalent to ``self.resize_(tensor.size())``.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                Tensor. Default: ``torch.contiguous_format``. Note that memory format of
                :attr:`self` is going to be unaffected if ``self.size()`` matches ``tensor.size()``.
        """
        ...
    def resize_as_sparse_(self, the_template: Tensor) -> Tensor: ...
    def resolve_conj(self) -> Tensor:
        r"""
        resolve_conj() -> Tensor

        See :func:`torch.resolve_conj`
        """
        ...
    def resolve_neg(self) -> Tensor:
        r"""
        resolve_neg() -> Tensor

        See :func:`torch.resolve_neg`
        """
        ...
    def retain_grad(self) -> None:
        r"""
        retain_grad() -> None

        Enables this Tensor to have their :attr:`grad` populated during
        :func:`backward`. This is a no-op for leaf tensors.
        """
        ...
    def roll(self, shifts: Union[Union[_int, SymInt], Sequence[Union[_int, SymInt]]], dims: Union[_int, _size] = ()) -> Tensor:
        r"""
        roll(shifts, dims) -> Tensor

        See :func:`torch.roll`
        """
        ...
    def rot90(self, k: _int = 1, dims: _size = (0,1)) -> Tensor:
        r"""
        rot90(k, dims) -> Tensor

        See :func:`torch.rot90`
        """
        ...
    @overload
    def round(self) -> Tensor:
        r"""
        round(decimals=0) -> Tensor

        See :func:`torch.round`
        """
        ...
    @overload
    def round(self, *, decimals: _int) -> Tensor:
        r"""
        round(decimals=0) -> Tensor

        See :func:`torch.round`
        """
        ...
    @overload
    def round_(self) -> Tensor:
        r"""
        round_(decimals=0) -> Tensor

        In-place version of :meth:`~Tensor.round`
        """
        ...
    @overload
    def round_(self, *, decimals: _int) -> Tensor:
        r"""
        round_(decimals=0) -> Tensor

        In-place version of :meth:`~Tensor.round`
        """
        ...
    def row_indices(self) -> Tensor: ...
    def rsqrt(self) -> Tensor:
        r"""
        rsqrt() -> Tensor

        See :func:`torch.rsqrt`
        """
        ...
    def rsqrt_(self) -> Tensor:
        r"""
        rsqrt_() -> Tensor

        In-place version of :meth:`~Tensor.rsqrt`
        """
        ...
    @overload
    def scatter(self, dim: _int, index: Tensor, src: Tensor) -> Tensor:
        r"""
        scatter(dim, index, src) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.scatter_`
        """
        ...
    @overload
    def scatter(self, dim: _int, index: Tensor, src: Tensor, *, reduce: str) -> Tensor:
        r"""
        scatter(dim, index, src) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.scatter_`
        """
        ...
    @overload
    def scatter(self, dim: _int, index: Tensor, value: Union[Number, _complex], *, reduce: str) -> Tensor:
        r"""
        scatter(dim, index, src) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.scatter_`
        """
        ...
    @overload
    def scatter(self, dim: Union[str, ellipsis, None], index: Tensor, src: Tensor) -> Tensor:
        r"""
        scatter(dim, index, src) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.scatter_`
        """
        ...
    @overload
    def scatter(self, dim: _int, index: Tensor, value: Union[Number, _complex]) -> Tensor:
        r"""
        scatter(dim, index, src) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.scatter_`
        """
        ...
    @overload
    def scatter(self, dim: Union[str, ellipsis, None], index: Tensor, value: Union[Number, _complex]) -> Tensor:
        r"""
        scatter(dim, index, src) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.scatter_`
        """
        ...
    @overload
    def scatter_(self, dim: _int, index: Tensor, src: Tensor) -> Tensor:
        r"""
        scatter_(dim, index, src, *, reduce=None) -> Tensor

        Writes all values from the tensor :attr:`src` into :attr:`self` at the indices
        specified in the :attr:`index` tensor. For each value in :attr:`src`, its output
        index is specified by its index in :attr:`src` for ``dimension != dim`` and by
        the corresponding value in :attr:`index` for ``dimension = dim``.

        For a 3-D tensor, :attr:`self` is updated as::

            self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
            self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
            self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2

        This is the reverse operation of the manner described in :meth:`~Tensor.gather`.

        :attr:`self`, :attr:`index` and :attr:`src` (if it is a Tensor) should all have
        the same number of dimensions. It is also required that
        ``index.size(d) <= src.size(d)`` for all dimensions ``d``, and that
        ``index.size(d) <= self.size(d)`` for all dimensions ``d != dim``.
        Note that ``index`` and ``src`` do not broadcast.

        Moreover, as for :meth:`~Tensor.gather`, the values of :attr:`index` must be
        between ``0`` and ``self.size(dim) - 1`` inclusive.

        .. warning::

            When indices are not unique, the behavior is non-deterministic (one of the
            values from ``src`` will be picked arbitrarily) and the gradient will be
            incorrect (it will be propagated to all locations in the source that
            correspond to the same index)!

        .. note::

            The backward pass is implemented only for ``src.shape == index.shape``.

        Additionally accepts an optional :attr:`reduce` argument that allows
        specification of an optional reduction operation, which is applied to all
        values in the tensor :attr:`src` into :attr:`self` at the indices
        specified in the :attr:`index`. For each value in :attr:`src`, the reduction
        operation is applied to an index in :attr:`self` which is specified by
        its index in :attr:`src` for ``dimension != dim`` and by the corresponding
        value in :attr:`index` for ``dimension = dim``.

        Given a 3-D tensor and reduction using the multiplication operation, :attr:`self`
        is updated as::

            self[index[i][j][k]][j][k] *= src[i][j][k]  # if dim == 0
            self[i][index[i][j][k]][k] *= src[i][j][k]  # if dim == 1
            self[i][j][index[i][j][k]] *= src[i][j][k]  # if dim == 2

        Reducing with the addition operation is the same as using
        :meth:`~torch.Tensor.scatter_add_`.

        .. warning::
            The reduce argument with Tensor ``src`` is deprecated and will be removed in
            a future PyTorch release. Please use :meth:`~torch.Tensor.scatter_reduce_`
            instead for more reduction options.

        Args:
            dim (int): the axis along which to index
            index (LongTensor): the indices of elements to scatter, can be either empty
                or of the same dimensionality as ``src``. When empty, the operation
                returns ``self`` unchanged.
            src (Tensor): the source element(s) to scatter.

        Keyword args:
            reduce (str, optional): reduction operation to apply, can be either
                ``'add'`` or ``'multiply'``.

        Example::

            >>> src = torch.arange(1, 11).reshape((2, 5))
            >>> src
            tensor([[ 1,  2,  3,  4,  5],
                    [ 6,  7,  8,  9, 10]])
            >>> index = torch.tensor([[0, 1, 2, 0]])
            >>> torch.zeros(3, 5, dtype=src.dtype).scatter_(0, index, src)
            tensor([[1, 0, 0, 4, 0],
                    [0, 2, 0, 0, 0],
                    [0, 0, 3, 0, 0]])
            >>> index = torch.tensor([[0, 1, 2], [0, 1, 4]])
            >>> torch.zeros(3, 5, dtype=src.dtype).scatter_(1, index, src)
            tensor([[1, 2, 3, 0, 0],
                    [6, 7, 0, 0, 8],
                    [0, 0, 0, 0, 0]])

            >>> torch.full((2, 4), 2.).scatter_(1, torch.tensor([[2], [3]]),
            ...            1.23, reduce='multiply')
            tensor([[2.0000, 2.0000, 2.4600, 2.0000],
                    [2.0000, 2.0000, 2.0000, 2.4600]])
            >>> torch.full((2, 4), 2.).scatter_(1, torch.tensor([[2], [3]]),
            ...            1.23, reduce='add')
            tensor([[2.0000, 2.0000, 3.2300, 2.0000],
                    [2.0000, 2.0000, 2.0000, 3.2300]])

        .. function:: scatter_(dim, index, value, *, reduce=None) -> Tensor:
           :noindex:

        Writes the value from :attr:`value` into :attr:`self` at the indices
        specified in the :attr:`index` tensor.  This operation is equivalent to the previous version,
        with the :attr:`src` tensor filled entirely with :attr:`value`.

        Args:
            dim (int): the axis along which to index
            index (LongTensor): the indices of elements to scatter, can be either empty
                or of the same dimensionality as ``src``. When empty, the operation
                returns ``self`` unchanged.
            value (Scalar): the value to scatter.

        Keyword args:
            reduce (str, optional): reduction operation to apply, can be either
                ``'add'`` or ``'multiply'``.

        Example::

            >>> index = torch.tensor([[0, 1]])
            >>> value = 2
            >>> torch.zeros(3, 5).scatter_(0, index, value)
            tensor([[2., 0., 0., 0., 0.],
                    [0., 2., 0., 0., 0.],
                    [0., 0., 0., 0., 0.]])
        """
        ...
    @overload
    def scatter_(self, dim: _int, index: Tensor, src: Tensor, *, reduce: str) -> Tensor:
        r"""
        scatter_(dim, index, src, *, reduce=None) -> Tensor

        Writes all values from the tensor :attr:`src` into :attr:`self` at the indices
        specified in the :attr:`index` tensor. For each value in :attr:`src`, its output
        index is specified by its index in :attr:`src` for ``dimension != dim`` and by
        the corresponding value in :attr:`index` for ``dimension = dim``.

        For a 3-D tensor, :attr:`self` is updated as::

            self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
            self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
            self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2

        This is the reverse operation of the manner described in :meth:`~Tensor.gather`.

        :attr:`self`, :attr:`index` and :attr:`src` (if it is a Tensor) should all have
        the same number of dimensions. It is also required that
        ``index.size(d) <= src.size(d)`` for all dimensions ``d``, and that
        ``index.size(d) <= self.size(d)`` for all dimensions ``d != dim``.
        Note that ``index`` and ``src`` do not broadcast.

        Moreover, as for :meth:`~Tensor.gather`, the values of :attr:`index` must be
        between ``0`` and ``self.size(dim) - 1`` inclusive.

        .. warning::

            When indices are not unique, the behavior is non-deterministic (one of the
            values from ``src`` will be picked arbitrarily) and the gradient will be
            incorrect (it will be propagated to all locations in the source that
            correspond to the same index)!

        .. note::

            The backward pass is implemented only for ``src.shape == index.shape``.

        Additionally accepts an optional :attr:`reduce` argument that allows
        specification of an optional reduction operation, which is applied to all
        values in the tensor :attr:`src` into :attr:`self` at the indices
        specified in the :attr:`index`. For each value in :attr:`src`, the reduction
        operation is applied to an index in :attr:`self` which is specified by
        its index in :attr:`src` for ``dimension != dim`` and by the corresponding
        value in :attr:`index` for ``dimension = dim``.

        Given a 3-D tensor and reduction using the multiplication operation, :attr:`self`
        is updated as::

            self[index[i][j][k]][j][k] *= src[i][j][k]  # if dim == 0
            self[i][index[i][j][k]][k] *= src[i][j][k]  # if dim == 1
            self[i][j][index[i][j][k]] *= src[i][j][k]  # if dim == 2

        Reducing with the addition operation is the same as using
        :meth:`~torch.Tensor.scatter_add_`.

        .. warning::
            The reduce argument with Tensor ``src`` is deprecated and will be removed in
            a future PyTorch release. Please use :meth:`~torch.Tensor.scatter_reduce_`
            instead for more reduction options.

        Args:
            dim (int): the axis along which to index
            index (LongTensor): the indices of elements to scatter, can be either empty
                or of the same dimensionality as ``src``. When empty, the operation
                returns ``self`` unchanged.
            src (Tensor): the source element(s) to scatter.

        Keyword args:
            reduce (str, optional): reduction operation to apply, can be either
                ``'add'`` or ``'multiply'``.

        Example::

            >>> src = torch.arange(1, 11).reshape((2, 5))
            >>> src
            tensor([[ 1,  2,  3,  4,  5],
                    [ 6,  7,  8,  9, 10]])
            >>> index = torch.tensor([[0, 1, 2, 0]])
            >>> torch.zeros(3, 5, dtype=src.dtype).scatter_(0, index, src)
            tensor([[1, 0, 0, 4, 0],
                    [0, 2, 0, 0, 0],
                    [0, 0, 3, 0, 0]])
            >>> index = torch.tensor([[0, 1, 2], [0, 1, 4]])
            >>> torch.zeros(3, 5, dtype=src.dtype).scatter_(1, index, src)
            tensor([[1, 2, 3, 0, 0],
                    [6, 7, 0, 0, 8],
                    [0, 0, 0, 0, 0]])

            >>> torch.full((2, 4), 2.).scatter_(1, torch.tensor([[2], [3]]),
            ...            1.23, reduce='multiply')
            tensor([[2.0000, 2.0000, 2.4600, 2.0000],
                    [2.0000, 2.0000, 2.0000, 2.4600]])
            >>> torch.full((2, 4), 2.).scatter_(1, torch.tensor([[2], [3]]),
            ...            1.23, reduce='add')
            tensor([[2.0000, 2.0000, 3.2300, 2.0000],
                    [2.0000, 2.0000, 2.0000, 3.2300]])

        .. function:: scatter_(dim, index, value, *, reduce=None) -> Tensor:
           :noindex:

        Writes the value from :attr:`value` into :attr:`self` at the indices
        specified in the :attr:`index` tensor.  This operation is equivalent to the previous version,
        with the :attr:`src` tensor filled entirely with :attr:`value`.

        Args:
            dim (int): the axis along which to index
            index (LongTensor): the indices of elements to scatter, can be either empty
                or of the same dimensionality as ``src``. When empty, the operation
                returns ``self`` unchanged.
            value (Scalar): the value to scatter.

        Keyword args:
            reduce (str, optional): reduction operation to apply, can be either
                ``'add'`` or ``'multiply'``.

        Example::

            >>> index = torch.tensor([[0, 1]])
            >>> value = 2
            >>> torch.zeros(3, 5).scatter_(0, index, value)
            tensor([[2., 0., 0., 0., 0.],
                    [0., 2., 0., 0., 0.],
                    [0., 0., 0., 0., 0.]])
        """
        ...
    @overload
    def scatter_(self, dim: _int, index: Tensor, value: Union[Number, _complex], *, reduce: str) -> Tensor:
        r"""
        scatter_(dim, index, src, *, reduce=None) -> Tensor

        Writes all values from the tensor :attr:`src` into :attr:`self` at the indices
        specified in the :attr:`index` tensor. For each value in :attr:`src`, its output
        index is specified by its index in :attr:`src` for ``dimension != dim`` and by
        the corresponding value in :attr:`index` for ``dimension = dim``.

        For a 3-D tensor, :attr:`self` is updated as::

            self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
            self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
            self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2

        This is the reverse operation of the manner described in :meth:`~Tensor.gather`.

        :attr:`self`, :attr:`index` and :attr:`src` (if it is a Tensor) should all have
        the same number of dimensions. It is also required that
        ``index.size(d) <= src.size(d)`` for all dimensions ``d``, and that
        ``index.size(d) <= self.size(d)`` for all dimensions ``d != dim``.
        Note that ``index`` and ``src`` do not broadcast.

        Moreover, as for :meth:`~Tensor.gather`, the values of :attr:`index` must be
        between ``0`` and ``self.size(dim) - 1`` inclusive.

        .. warning::

            When indices are not unique, the behavior is non-deterministic (one of the
            values from ``src`` will be picked arbitrarily) and the gradient will be
            incorrect (it will be propagated to all locations in the source that
            correspond to the same index)!

        .. note::

            The backward pass is implemented only for ``src.shape == index.shape``.

        Additionally accepts an optional :attr:`reduce` argument that allows
        specification of an optional reduction operation, which is applied to all
        values in the tensor :attr:`src` into :attr:`self` at the indices
        specified in the :attr:`index`. For each value in :attr:`src`, the reduction
        operation is applied to an index in :attr:`self` which is specified by
        its index in :attr:`src` for ``dimension != dim`` and by the corresponding
        value in :attr:`index` for ``dimension = dim``.

        Given a 3-D tensor and reduction using the multiplication operation, :attr:`self`
        is updated as::

            self[index[i][j][k]][j][k] *= src[i][j][k]  # if dim == 0
            self[i][index[i][j][k]][k] *= src[i][j][k]  # if dim == 1
            self[i][j][index[i][j][k]] *= src[i][j][k]  # if dim == 2

        Reducing with the addition operation is the same as using
        :meth:`~torch.Tensor.scatter_add_`.

        .. warning::
            The reduce argument with Tensor ``src`` is deprecated and will be removed in
            a future PyTorch release. Please use :meth:`~torch.Tensor.scatter_reduce_`
            instead for more reduction options.

        Args:
            dim (int): the axis along which to index
            index (LongTensor): the indices of elements to scatter, can be either empty
                or of the same dimensionality as ``src``. When empty, the operation
                returns ``self`` unchanged.
            src (Tensor): the source element(s) to scatter.

        Keyword args:
            reduce (str, optional): reduction operation to apply, can be either
                ``'add'`` or ``'multiply'``.

        Example::

            >>> src = torch.arange(1, 11).reshape((2, 5))
            >>> src
            tensor([[ 1,  2,  3,  4,  5],
                    [ 6,  7,  8,  9, 10]])
            >>> index = torch.tensor([[0, 1, 2, 0]])
            >>> torch.zeros(3, 5, dtype=src.dtype).scatter_(0, index, src)
            tensor([[1, 0, 0, 4, 0],
                    [0, 2, 0, 0, 0],
                    [0, 0, 3, 0, 0]])
            >>> index = torch.tensor([[0, 1, 2], [0, 1, 4]])
            >>> torch.zeros(3, 5, dtype=src.dtype).scatter_(1, index, src)
            tensor([[1, 2, 3, 0, 0],
                    [6, 7, 0, 0, 8],
                    [0, 0, 0, 0, 0]])

            >>> torch.full((2, 4), 2.).scatter_(1, torch.tensor([[2], [3]]),
            ...            1.23, reduce='multiply')
            tensor([[2.0000, 2.0000, 2.4600, 2.0000],
                    [2.0000, 2.0000, 2.0000, 2.4600]])
            >>> torch.full((2, 4), 2.).scatter_(1, torch.tensor([[2], [3]]),
            ...            1.23, reduce='add')
            tensor([[2.0000, 2.0000, 3.2300, 2.0000],
                    [2.0000, 2.0000, 2.0000, 3.2300]])

        .. function:: scatter_(dim, index, value, *, reduce=None) -> Tensor:
           :noindex:

        Writes the value from :attr:`value` into :attr:`self` at the indices
        specified in the :attr:`index` tensor.  This operation is equivalent to the previous version,
        with the :attr:`src` tensor filled entirely with :attr:`value`.

        Args:
            dim (int): the axis along which to index
            index (LongTensor): the indices of elements to scatter, can be either empty
                or of the same dimensionality as ``src``. When empty, the operation
                returns ``self`` unchanged.
            value (Scalar): the value to scatter.

        Keyword args:
            reduce (str, optional): reduction operation to apply, can be either
                ``'add'`` or ``'multiply'``.

        Example::

            >>> index = torch.tensor([[0, 1]])
            >>> value = 2
            >>> torch.zeros(3, 5).scatter_(0, index, value)
            tensor([[2., 0., 0., 0., 0.],
                    [0., 2., 0., 0., 0.],
                    [0., 0., 0., 0., 0.]])
        """
        ...
    @overload
    def scatter_(self, dim: _int, index: Tensor, value: Union[Number, _complex]) -> Tensor:
        r"""
        scatter_(dim, index, src, *, reduce=None) -> Tensor

        Writes all values from the tensor :attr:`src` into :attr:`self` at the indices
        specified in the :attr:`index` tensor. For each value in :attr:`src`, its output
        index is specified by its index in :attr:`src` for ``dimension != dim`` and by
        the corresponding value in :attr:`index` for ``dimension = dim``.

        For a 3-D tensor, :attr:`self` is updated as::

            self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
            self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
            self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2

        This is the reverse operation of the manner described in :meth:`~Tensor.gather`.

        :attr:`self`, :attr:`index` and :attr:`src` (if it is a Tensor) should all have
        the same number of dimensions. It is also required that
        ``index.size(d) <= src.size(d)`` for all dimensions ``d``, and that
        ``index.size(d) <= self.size(d)`` for all dimensions ``d != dim``.
        Note that ``index`` and ``src`` do not broadcast.

        Moreover, as for :meth:`~Tensor.gather`, the values of :attr:`index` must be
        between ``0`` and ``self.size(dim) - 1`` inclusive.

        .. warning::

            When indices are not unique, the behavior is non-deterministic (one of the
            values from ``src`` will be picked arbitrarily) and the gradient will be
            incorrect (it will be propagated to all locations in the source that
            correspond to the same index)!

        .. note::

            The backward pass is implemented only for ``src.shape == index.shape``.

        Additionally accepts an optional :attr:`reduce` argument that allows
        specification of an optional reduction operation, which is applied to all
        values in the tensor :attr:`src` into :attr:`self` at the indices
        specified in the :attr:`index`. For each value in :attr:`src`, the reduction
        operation is applied to an index in :attr:`self` which is specified by
        its index in :attr:`src` for ``dimension != dim`` and by the corresponding
        value in :attr:`index` for ``dimension = dim``.

        Given a 3-D tensor and reduction using the multiplication operation, :attr:`self`
        is updated as::

            self[index[i][j][k]][j][k] *= src[i][j][k]  # if dim == 0
            self[i][index[i][j][k]][k] *= src[i][j][k]  # if dim == 1
            self[i][j][index[i][j][k]] *= src[i][j][k]  # if dim == 2

        Reducing with the addition operation is the same as using
        :meth:`~torch.Tensor.scatter_add_`.

        .. warning::
            The reduce argument with Tensor ``src`` is deprecated and will be removed in
            a future PyTorch release. Please use :meth:`~torch.Tensor.scatter_reduce_`
            instead for more reduction options.

        Args:
            dim (int): the axis along which to index
            index (LongTensor): the indices of elements to scatter, can be either empty
                or of the same dimensionality as ``src``. When empty, the operation
                returns ``self`` unchanged.
            src (Tensor): the source element(s) to scatter.

        Keyword args:
            reduce (str, optional): reduction operation to apply, can be either
                ``'add'`` or ``'multiply'``.

        Example::

            >>> src = torch.arange(1, 11).reshape((2, 5))
            >>> src
            tensor([[ 1,  2,  3,  4,  5],
                    [ 6,  7,  8,  9, 10]])
            >>> index = torch.tensor([[0, 1, 2, 0]])
            >>> torch.zeros(3, 5, dtype=src.dtype).scatter_(0, index, src)
            tensor([[1, 0, 0, 4, 0],
                    [0, 2, 0, 0, 0],
                    [0, 0, 3, 0, 0]])
            >>> index = torch.tensor([[0, 1, 2], [0, 1, 4]])
            >>> torch.zeros(3, 5, dtype=src.dtype).scatter_(1, index, src)
            tensor([[1, 2, 3, 0, 0],
                    [6, 7, 0, 0, 8],
                    [0, 0, 0, 0, 0]])

            >>> torch.full((2, 4), 2.).scatter_(1, torch.tensor([[2], [3]]),
            ...            1.23, reduce='multiply')
            tensor([[2.0000, 2.0000, 2.4600, 2.0000],
                    [2.0000, 2.0000, 2.0000, 2.4600]])
            >>> torch.full((2, 4), 2.).scatter_(1, torch.tensor([[2], [3]]),
            ...            1.23, reduce='add')
            tensor([[2.0000, 2.0000, 3.2300, 2.0000],
                    [2.0000, 2.0000, 2.0000, 3.2300]])

        .. function:: scatter_(dim, index, value, *, reduce=None) -> Tensor:
           :noindex:

        Writes the value from :attr:`value` into :attr:`self` at the indices
        specified in the :attr:`index` tensor.  This operation is equivalent to the previous version,
        with the :attr:`src` tensor filled entirely with :attr:`value`.

        Args:
            dim (int): the axis along which to index
            index (LongTensor): the indices of elements to scatter, can be either empty
                or of the same dimensionality as ``src``. When empty, the operation
                returns ``self`` unchanged.
            value (Scalar): the value to scatter.

        Keyword args:
            reduce (str, optional): reduction operation to apply, can be either
                ``'add'`` or ``'multiply'``.

        Example::

            >>> index = torch.tensor([[0, 1]])
            >>> value = 2
            >>> torch.zeros(3, 5).scatter_(0, index, value)
            tensor([[2., 0., 0., 0., 0.],
                    [0., 2., 0., 0., 0.],
                    [0., 0., 0., 0., 0.]])
        """
        ...
    @overload
    def scatter_add(self, dim: _int, index: Tensor, src: Tensor) -> Tensor:
        r"""
        scatter_add(dim, index, src) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.scatter_add_`
        """
        ...
    @overload
    def scatter_add(self, dim: Union[str, ellipsis, None], index: Tensor, src: Tensor) -> Tensor:
        r"""
        scatter_add(dim, index, src) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.scatter_add_`
        """
        ...
    def scatter_add_(self, dim: _int, index: Tensor, src: Tensor) -> Tensor:
        r"""
        scatter_add_(dim, index, src) -> Tensor

        Adds all values from the tensor :attr:`src` into :attr:`self` at the indices
        specified in the :attr:`index` tensor in a similar fashion as
        :meth:`~torch.Tensor.scatter_`. For each value in :attr:`src`, it is added to
        an index in :attr:`self` which is specified by its index in :attr:`src`
        for ``dimension != dim`` and by the corresponding value in :attr:`index` for
        ``dimension = dim``.

        For a 3-D tensor, :attr:`self` is updated as::

            self[index[i][j][k]][j][k] += src[i][j][k]  # if dim == 0
            self[i][index[i][j][k]][k] += src[i][j][k]  # if dim == 1
            self[i][j][index[i][j][k]] += src[i][j][k]  # if dim == 2

        :attr:`self`, :attr:`index` and :attr:`src` should have same number of
        dimensions. It is also required that ``index.size(d) <= src.size(d)`` for all
        dimensions ``d``, and that ``index.size(d) <= self.size(d)`` for all dimensions
        ``d != dim``. Note that ``index`` and ``src`` do not broadcast.

        Note:
            This operation may behave nondeterministically when given tensors on a CUDA device. See :doc:`/notes/randomness` for more information.

        .. note::

            The backward pass is implemented only for ``src.shape == index.shape``.

        Args:
            dim (int): the axis along which to index
            index (LongTensor): the indices of elements to scatter and add, can be
                either empty or of the same dimensionality as ``src``. When empty, the
                operation returns ``self`` unchanged.
            src (Tensor): the source elements to scatter and add

        Example::

            >>> src = torch.ones((2, 5))
            >>> index = torch.tensor([[0, 1, 2, 0, 0]])
            >>> torch.zeros(3, 5, dtype=src.dtype).scatter_add_(0, index, src)
            tensor([[1., 0., 0., 1., 1.],
                    [0., 1., 0., 0., 0.],
                    [0., 0., 1., 0., 0.]])
            >>> index = torch.tensor([[0, 1, 2, 0, 0], [0, 1, 2, 2, 2]])
            >>> torch.zeros(3, 5, dtype=src.dtype).scatter_add_(0, index, src)
            tensor([[2., 0., 0., 1., 1.],
                    [0., 2., 0., 0., 0.],
                    [0., 0., 2., 1., 1.]])
        """
        ...
    def scatter_reduce(self, dim: _int, index: Tensor, src: Tensor, reduce: str, *, include_self: _bool = True) -> Tensor:
        r"""
        scatter_reduce(dim, index, src, reduce, *, include_self=True) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.scatter_reduce_`
        """
        ...
    def scatter_reduce_(self, dim: _int, index: Tensor, src: Tensor, reduce: str, *, include_self: _bool = True) -> Tensor:
        r"""
        scatter_reduce_(dim, index, src, reduce, *, include_self=True) -> Tensor

        Reduces all values from the :attr:`src` tensor to the indices specified in
        the :attr:`index` tensor in the :attr:`self` tensor using the applied reduction
        defined via the :attr:`reduce` argument (:obj:`"sum"`, :obj:`"prod"`, :obj:`"mean"`,
        :obj:`"amax"`, :obj:`"amin"`). For each value in :attr:`src`, it is reduced to an
        index in :attr:`self` which is specified by its index in :attr:`src` for
        ``dimension != dim`` and by the corresponding value in :attr:`index` for
        ``dimension = dim``. If :obj:`include_self="True"`, the values in the :attr:`self`
        tensor are included in the reduction.

        :attr:`self`, :attr:`index` and :attr:`src` should all have
        the same number of dimensions. It is also required that
        ``index.size(d) <= src.size(d)`` for all dimensions ``d``, and that
        ``index.size(d) <= self.size(d)`` for all dimensions ``d != dim``.
        Note that ``index`` and ``src`` do not broadcast.

        For a 3-D tensor with :obj:`reduce="sum"` and :obj:`include_self=True` the
        output is given as::

            self[index[i][j][k]][j][k] += src[i][j][k]  # if dim == 0
            self[i][index[i][j][k]][k] += src[i][j][k]  # if dim == 1
            self[i][j][index[i][j][k]] += src[i][j][k]  # if dim == 2

        Note:
            This operation may behave nondeterministically when given tensors on a CUDA device. See :doc:`/notes/randomness` for more information.

        .. note::

            The backward pass is implemented only for ``src.shape == index.shape``.

        .. warning::

            This function is in beta and may change in the near future.

        Args:
            dim (int): the axis along which to index
            index (LongTensor): the indices of elements to scatter and reduce.
            src (Tensor): the source elements to scatter and reduce
            reduce (str): the reduction operation to apply for non-unique indices
                (:obj:`"sum"`, :obj:`"prod"`, :obj:`"mean"`, :obj:`"amax"`, :obj:`"amin"`)
            include_self (bool): whether elements from the :attr:`self` tensor are
                included in the reduction

        Example::

            >>> src = torch.tensor([1., 2., 3., 4., 5., 6.])
            >>> index = torch.tensor([0, 1, 0, 1, 2, 1])
            >>> input = torch.tensor([1., 2., 3., 4.])
            >>> input.scatter_reduce(0, index, src, reduce="sum")
            tensor([5., 14., 8., 4.])
            >>> input.scatter_reduce(0, index, src, reduce="sum", include_self=False)
            tensor([4., 12., 5., 4.])
            >>> input2 = torch.tensor([5., 4., 3., 2.])
            >>> input2.scatter_reduce(0, index, src, reduce="amax")
            tensor([5., 6., 5., 2.])
            >>> input2.scatter_reduce(0, index, src, reduce="amax", include_self=False)
            tensor([3., 6., 5., 2.])
        """
        ...
    @overload
    def select(self, dim: _int, index: Union[_int, SymInt]) -> Tensor:
        r"""
        select(dim, index) -> Tensor

        See :func:`torch.select`
        """
        ...
    @overload
    def select(self, dim: Union[str, ellipsis, None], index: _int) -> Tensor:
        r"""
        select(dim, index) -> Tensor

        See :func:`torch.select`
        """
        ...
    def select_scatter(self, src: Tensor, dim: _int, index: Union[_int, SymInt]) -> Tensor:
        r"""
        select_scatter(src, dim, index) -> Tensor

        See :func:`torch.select_scatter`
        """
        ...
    @overload
    def set_(self, storage: Union[Storage, TypedStorage, UntypedStorage], offset: _int, size: _size, stride: _size) -> Tensor:
        r"""
        set_(source=None, storage_offset=0, size=None, stride=None) -> Tensor

        Sets the underlying storage, size, and strides. If :attr:`source` is a tensor,
        :attr:`self` tensor will share the same storage and have the same size and
        strides as :attr:`source`. Changes to elements in one tensor will be reflected
        in the other.

        If :attr:`source` is a :class:`~torch.Storage`, the method sets the underlying
        storage, offset, size, and stride.

        Args:
            source (Tensor or Storage): the tensor or storage to use
            storage_offset (int, optional): the offset in the storage
            size (torch.Size, optional): the desired size. Defaults to the size of the source.
            stride (tuple, optional): the desired stride. Defaults to C-contiguous strides.
        """
        ...
    @overload
    def set_(self, storage: Union[Storage, TypedStorage, UntypedStorage]) -> Tensor:
        r"""
        set_(source=None, storage_offset=0, size=None, stride=None) -> Tensor

        Sets the underlying storage, size, and strides. If :attr:`source` is a tensor,
        :attr:`self` tensor will share the same storage and have the same size and
        strides as :attr:`source`. Changes to elements in one tensor will be reflected
        in the other.

        If :attr:`source` is a :class:`~torch.Storage`, the method sets the underlying
        storage, offset, size, and stride.

        Args:
            source (Tensor or Storage): the tensor or storage to use
            storage_offset (int, optional): the offset in the storage
            size (torch.Size, optional): the desired size. Defaults to the size of the source.
            stride (tuple, optional): the desired stride. Defaults to C-contiguous strides.
        """
        ...
    def sgn(self) -> Tensor:
        r"""
        sgn() -> Tensor

        See :func:`torch.sgn`
        """
        ...
    def sgn_(self) -> Tensor:
        r"""
        sgn_() -> Tensor

        In-place version of :meth:`~Tensor.sgn`
        """
        ...
    def short(self) -> Tensor:
        r"""
        short(memory_format=torch.preserve_format) -> Tensor

        ``self.short()`` is equivalent to ``self.to(torch.int16)``. See :func:`to`.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.
        """
        ...
    def sigmoid(self) -> Tensor:
        r"""
        sigmoid() -> Tensor

        See :func:`torch.sigmoid`
        """
        ...
    def sigmoid_(self) -> Tensor:
        r"""
        sigmoid_() -> Tensor

        In-place version of :meth:`~Tensor.sigmoid`
        """
        ...
    def sign(self) -> Tensor:
        r"""
        sign() -> Tensor

        See :func:`torch.sign`
        """
        ...
    def sign_(self) -> Tensor:
        r"""
        sign_() -> Tensor

        In-place version of :meth:`~Tensor.sign`
        """
        ...
    def signbit(self) -> Tensor:
        r"""
        signbit() -> Tensor

        See :func:`torch.signbit`
        """
        ...
    def sin(self) -> Tensor:
        r"""
        sin() -> Tensor

        See :func:`torch.sin`
        """
        ...
    def sin_(self) -> Tensor:
        r"""
        sin_() -> Tensor

        In-place version of :meth:`~Tensor.sin`
        """
        ...
    def sinc(self) -> Tensor:
        r"""
        sinc() -> Tensor

        See :func:`torch.sinc`
        """
        ...
    def sinc_(self) -> Tensor:
        r"""
        sinc_() -> Tensor

        In-place version of :meth:`~Tensor.sinc`
        """
        ...
    def sinh(self) -> Tensor:
        r"""
        sinh() -> Tensor

        See :func:`torch.sinh`
        """
        ...
    def sinh_(self) -> Tensor:
        r"""
        sinh_() -> Tensor

        In-place version of :meth:`~Tensor.sinh`
        """
        ...
    @overload
    def size(self, dim: None = None) -> Size:
        r"""
        size(dim=None) -> torch.Size or int

        Returns the size of the :attr:`self` tensor. If ``dim`` is not specified,
        the returned value is a :class:`torch.Size`, a subclass of :class:`tuple`.
        If ``dim`` is specified, returns an int holding the size of that dimension.

        Args:
          dim (int, optional): The dimension for which to retrieve the size.

        Example::

            >>> t = torch.empty(3, 4, 5)
            >>> t.size()
            torch.Size([3, 4, 5])
            >>> t.size(dim=1)
            4
        """
        ...
    @overload
    def size(self, dim: _int) -> _int:
        r"""
        size(dim=None) -> torch.Size or int

        Returns the size of the :attr:`self` tensor. If ``dim`` is not specified,
        the returned value is a :class:`torch.Size`, a subclass of :class:`tuple`.
        If ``dim`` is specified, returns an int holding the size of that dimension.

        Args:
          dim (int, optional): The dimension for which to retrieve the size.

        Example::

            >>> t = torch.empty(3, 4, 5)
            >>> t.size()
            torch.Size([3, 4, 5])
            >>> t.size(dim=1)
            4
        """
        ...
    def slice_inverse(self, src: Tensor, dim: _int = 0, start: Optional[Union[_int, SymInt]] = None, end: Optional[Union[_int, SymInt]] = None, step: Union[_int, SymInt] = 1) -> Tensor: ...
    def slice_scatter(self, src: Tensor, dim: _int = 0, start: Optional[Union[_int, SymInt]] = None, end: Optional[Union[_int, SymInt]] = None, step: Union[_int, SymInt] = 1) -> Tensor:
        r"""
        slice_scatter(src, dim=0, start=None, end=None, step=1) -> Tensor

        See :func:`torch.slice_scatter`
        """
        ...
    def slogdet(self) -> torch.return_types.slogdet:
        r"""
        slogdet() -> (Tensor, Tensor)

        See :func:`torch.slogdet`
        """
        ...
    def smm(self, mat2: Tensor) -> Tensor:
        r"""
        smm(mat) -> Tensor

        See :func:`torch.smm`
        """
        ...
    @overload
    def softmax(self, dim: _int, dtype: Optional[_dtype] = None) -> Tensor:
        r"""
        softmax(dim) -> Tensor

        Alias for :func:`torch.nn.functional.softmax`.
        """
        ...
    @overload
    def softmax(self, dim: Union[str, ellipsis, None], *, dtype: Optional[_dtype] = None) -> Tensor:
        r"""
        softmax(dim) -> Tensor

        Alias for :func:`torch.nn.functional.softmax`.
        """
        ...
    @overload
    def sort(self, *, stable: Optional[_bool], dim: _int = -1, descending: _bool = False) -> torch.return_types.sort:
        r"""
        sort(dim=-1, descending=False) -> (Tensor, LongTensor)

        See :func:`torch.sort`
        """
        ...
    @overload
    def sort(self, dim: _int = -1, descending: _bool = False) -> torch.return_types.sort:
        r"""
        sort(dim=-1, descending=False) -> (Tensor, LongTensor)

        See :func:`torch.sort`
        """
        ...
    @overload
    def sort(self, *, stable: Optional[_bool], dim: Union[str, ellipsis, None], descending: _bool = False) -> torch.return_types.sort:
        r"""
        sort(dim=-1, descending=False) -> (Tensor, LongTensor)

        See :func:`torch.sort`
        """
        ...
    @overload
    def sort(self, dim: Union[str, ellipsis, None], descending: _bool = False) -> torch.return_types.sort:
        r"""
        sort(dim=-1, descending=False) -> (Tensor, LongTensor)

        See :func:`torch.sort`
        """
        ...
    def sparse_dim(self) -> _int:
        r"""
        sparse_dim() -> int

        Return the number of sparse dimensions in a :ref:`sparse tensor <sparse-docs>` :attr:`self`.

        .. note::
          Returns ``0`` if :attr:`self` is not a sparse tensor.

        See also :meth:`Tensor.dense_dim` and :ref:`hybrid tensors <sparse-hybrid-coo-docs>`.
        """
        ...
    def sparse_mask(self, mask: Tensor) -> Tensor:
        r"""
        sparse_mask(mask) -> Tensor

        Returns a new :ref:`sparse tensor <sparse-docs>` with values from a
        strided tensor :attr:`self` filtered by the indices of the sparse
        tensor :attr:`mask`. The values of :attr:`mask` sparse tensor are
        ignored. :attr:`self` and :attr:`mask` tensors must have the same
        shape.

        .. note::

          The returned sparse tensor might contain duplicate values if :attr:`mask`
          is not coalesced. It is therefore advisable to pass ``mask.coalesce()``
          if such behavior is not desired.

        .. note::

          The returned sparse tensor has the same indices as the sparse tensor
          :attr:`mask`, even when the corresponding values in :attr:`self` are
          zeros.

        Args:
            mask (Tensor): a sparse tensor whose indices are used as a filter

        Example::

            >>> nse = 5
            >>> dims = (5, 5, 2, 2)
            >>> I = torch.cat([torch.randint(0, dims[0], size=(nse,)),
            ...                torch.randint(0, dims[1], size=(nse,))], 0).reshape(2, nse)
            >>> V = torch.randn(nse, dims[2], dims[3])
            >>> S = torch.sparse_coo_tensor(I, V, dims).coalesce()
            >>> D = torch.randn(dims)
            >>> D.sparse_mask(S)
            tensor(indices=tensor([[0, 0, 0, 2],
                                   [0, 1, 4, 3]]),
                   values=tensor([[[ 1.6550,  0.2397],
                                   [-0.1611, -0.0779]],

                                  [[ 0.2326, -1.0558],
                                   [ 1.4711,  1.9678]],

                                  [[-0.5138, -0.0411],
                                   [ 1.9417,  0.5158]],

                                  [[ 0.0793,  0.0036],
                                   [-0.2569, -0.1055]]]),
                   size=(5, 5, 2, 2), nnz=4, layout=torch.sparse_coo)
        """
        ...
    def sparse_resize_(self, size: _size, sparse_dim: _int, dense_dim: _int) -> Tensor:
        r"""
        sparse_resize_(size, sparse_dim, dense_dim) -> Tensor

        Resizes :attr:`self` :ref:`sparse tensor <sparse-docs>` to the desired
        size and the number of sparse and dense dimensions.

        .. note::
          If the number of specified elements in :attr:`self` is zero, then
          :attr:`size`, :attr:`sparse_dim`, and :attr:`dense_dim` can be any
          size and positive integers such that ``len(size) == sparse_dim +
          dense_dim``.

          If :attr:`self` specifies one or more elements, however, then each
          dimension in :attr:`size` must not be smaller than the corresponding
          dimension of :attr:`self`, :attr:`sparse_dim` must equal the number
          of sparse dimensions in :attr:`self`, and :attr:`dense_dim` must
          equal the number of dense dimensions in :attr:`self`.

        .. warning::
          Throws an error if :attr:`self` is not a sparse tensor.

        Args:
            size (torch.Size): the desired size. If :attr:`self` is non-empty
              sparse tensor, the desired size cannot be smaller than the
              original size.
            sparse_dim (int): the number of sparse dimensions
            dense_dim (int): the number of dense dimensions
        """
        ...
    def sparse_resize_and_clear_(self, size: _size, sparse_dim: _int, dense_dim: _int) -> Tensor:
        r"""
        sparse_resize_and_clear_(size, sparse_dim, dense_dim) -> Tensor

        Removes all specified elements from a :ref:`sparse tensor
        <sparse-docs>` :attr:`self` and resizes :attr:`self` to the desired
        size and the number of sparse and dense dimensions.

        .. warning:
          Throws an error if :attr:`self` is not a sparse tensor.

        Args:
            size (torch.Size): the desired size.
            sparse_dim (int): the number of sparse dimensions
            dense_dim (int): the number of dense dimensions
        """
        ...
    @overload
    def split(self, split_size: _int, dim: _int = 0) -> Sequence[Tensor]: ...
    @overload
    def split(self, split_size: Tuple[_int, ...], dim: _int = 0) -> Sequence[Tensor]: ...
    def split_with_sizes(self, split_sizes: Sequence[Union[_int, SymInt]], dim: _int = 0) -> Tuple[Tensor, ...]: ...
    def sqrt(self) -> Tensor:
        r"""
        sqrt() -> Tensor

        See :func:`torch.sqrt`
        """
        ...
    def sqrt_(self) -> Tensor:
        r"""
        sqrt_() -> Tensor

        In-place version of :meth:`~Tensor.sqrt`
        """
        ...
    def square(self) -> Tensor:
        r"""
        square() -> Tensor

        See :func:`torch.square`
        """
        ...
    def square_(self) -> Tensor:
        r"""
        square_() -> Tensor

        In-place version of :meth:`~Tensor.square`
        """
        ...
    @overload
    def squeeze(self) -> Tensor:
        r"""
        squeeze(dim=None) -> Tensor

        See :func:`torch.squeeze`
        """
        ...
    @overload
    def squeeze(self, dim: _int) -> Tensor:
        r"""
        squeeze(dim=None) -> Tensor

        See :func:`torch.squeeze`
        """
        ...
    @overload
    def squeeze(self, dim: _size) -> Tensor:
        r"""
        squeeze(dim=None) -> Tensor

        See :func:`torch.squeeze`
        """
        ...
    @overload
    def squeeze(self, *dim: _int) -> Tensor:
        r"""
        squeeze(dim=None) -> Tensor

        See :func:`torch.squeeze`
        """
        ...
    @overload
    def squeeze(self, dim: Union[str, ellipsis, None]) -> Tensor:
        r"""
        squeeze(dim=None) -> Tensor

        See :func:`torch.squeeze`
        """
        ...
    @overload
    def squeeze_(self) -> Tensor:
        r"""
        squeeze_(dim=None) -> Tensor

        In-place version of :meth:`~Tensor.squeeze`
        """
        ...
    @overload
    def squeeze_(self, dim: _int) -> Tensor:
        r"""
        squeeze_(dim=None) -> Tensor

        In-place version of :meth:`~Tensor.squeeze`
        """
        ...
    @overload
    def squeeze_(self, dim: _size) -> Tensor:
        r"""
        squeeze_(dim=None) -> Tensor

        In-place version of :meth:`~Tensor.squeeze`
        """
        ...
    @overload
    def squeeze_(self, *dim: _int) -> Tensor:
        r"""
        squeeze_(dim=None) -> Tensor

        In-place version of :meth:`~Tensor.squeeze`
        """
        ...
    @overload
    def squeeze_(self, dim: Union[str, ellipsis, None]) -> Tensor:
        r"""
        squeeze_(dim=None) -> Tensor

        In-place version of :meth:`~Tensor.squeeze`
        """
        ...
    def sspaddmm(self, mat1: Tensor, mat2: Tensor, *, beta: Union[Number, _complex] = 1, alpha: Union[Number, _complex] = 1) -> Tensor:
        r"""
        sspaddmm(mat1, mat2, *, beta=1, alpha=1) -> Tensor

        See :func:`torch.sspaddmm`
        """
        ...
    @overload
    def std(self, dim: Optional[Union[_int, _size]], unbiased: _bool = True, keepdim: _bool = False) -> Tensor:
        r"""
        std(dim=None, *, correction=1, keepdim=False) -> Tensor

        See :func:`torch.std`
        """
        ...
    @overload
    def std(self, dim: Optional[Union[_int, _size]] = None, *, correction: Optional[Union[Number, _complex]] = None, keepdim: _bool = False) -> Tensor:
        r"""
        std(dim=None, *, correction=1, keepdim=False) -> Tensor

        See :func:`torch.std`
        """
        ...
    @overload
    def std(self, unbiased: _bool = True) -> Tensor:
        r"""
        std(dim=None, *, correction=1, keepdim=False) -> Tensor

        See :func:`torch.std`
        """
        ...
    @overload
    def std(self, dim: Sequence[Union[str, ellipsis, None]], unbiased: _bool = True, keepdim: _bool = False) -> Tensor:
        r"""
        std(dim=None, *, correction=1, keepdim=False) -> Tensor

        See :func:`torch.std`
        """
        ...
    @overload
    def std(self, dim: Sequence[Union[str, ellipsis, None]], *, correction: Optional[Union[Number, _complex]] = None, keepdim: _bool = False) -> Tensor:
        r"""
        std(dim=None, *, correction=1, keepdim=False) -> Tensor

        See :func:`torch.std`
        """
        ...
    def untyped_storage(self) -> UntypedStorage: ...
    def storage_offset(self) -> _int:
        r"""
        storage_offset() -> int

        Returns :attr:`self` tensor's offset in the underlying storage in terms of
        number of storage elements (not bytes).

        Example::

            >>> x = torch.tensor([1, 2, 3, 4, 5])
            >>> x.storage_offset()
            0
            >>> x[3:].storage_offset()
            3
        """
        ...
    def storage_type(self) -> Storage: ...
    @overload
    def stride(self, dim: None = None) -> Tuple[_int, ...]:
        r"""
        stride(dim) -> tuple or int

        Returns the stride of :attr:`self` tensor.

        Stride is the jump necessary to go from one element to the next one in the
        specified dimension :attr:`dim`. A tuple of all strides is returned when no
        argument is passed in. Otherwise, an integer value is returned as the stride in
        the particular dimension :attr:`dim`.

        Args:
            dim (int, optional): the desired dimension in which stride is required

        Example::

            >>> x = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
            >>> x.stride()
            (5, 1)
            >>> x.stride(0)
            5
            >>> x.stride(-1)
            1
        """
        ...
    @overload
    def stride(self, dim: _int) -> _int:
        r"""
        stride(dim) -> tuple or int

        Returns the stride of :attr:`self` tensor.

        Stride is the jump necessary to go from one element to the next one in the
        specified dimension :attr:`dim`. A tuple of all strides is returned when no
        argument is passed in. Otherwise, an integer value is returned as the stride in
        the particular dimension :attr:`dim`.

        Args:
            dim (int, optional): the desired dimension in which stride is required

        Example::

            >>> x = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
            >>> x.stride()
            (5, 1)
            >>> x.stride(0)
            5
            >>> x.stride(-1)
            1
        """
        ...
    def sub(self, other: Union[Tensor, Number, _complex, torch.SymInt, torch.SymFloat], *, alpha: Optional[Union[Number, _complex]] = 1, out: Optional[Tensor] = None) -> Tensor:
        r"""
        sub(other, *, alpha=1) -> Tensor

        See :func:`torch.sub`.
        """
        ...
    def sub_(self, other: Union[Tensor, Number, _complex, torch.SymInt, torch.SymFloat], *, alpha: Optional[Union[Number, _complex]] = 1) -> Tensor:
        r"""
        sub_(other, *, alpha=1) -> Tensor

        In-place version of :meth:`~Tensor.sub`
        """
        ...
    @overload
    def subtract(self, other: Tensor, *, alpha: Union[Number, _complex] = 1) -> Tensor:
        r"""
        subtract(other, *, alpha=1) -> Tensor

        See :func:`torch.subtract`.
        """
        ...
    @overload
    def subtract(self, other: Union[Number, _complex], alpha: Union[Number, _complex] = 1) -> Tensor:
        r"""
        subtract(other, *, alpha=1) -> Tensor

        See :func:`torch.subtract`.
        """
        ...
    @overload
    def subtract_(self, other: Tensor, *, alpha: Union[Number, _complex] = 1) -> Tensor:
        r"""
        subtract_(other, *, alpha=1) -> Tensor

        In-place version of :meth:`~Tensor.subtract`.
        """
        ...
    @overload
    def subtract_(self, other: Union[Number, _complex], alpha: Union[Number, _complex] = 1) -> Tensor:
        r"""
        subtract_(other, *, alpha=1) -> Tensor

        In-place version of :meth:`~Tensor.subtract`.
        """
        ...
    @overload
    def sum(self, *, dtype: Optional[_dtype] = None) -> Tensor:
        r"""
        sum(dim=None, keepdim=False, dtype=None) -> Tensor

        See :func:`torch.sum`
        """
        ...
    @overload
    def sum(self, dim: Optional[Union[_int, _size]], keepdim: _bool = False, *, dtype: Optional[_dtype] = None) -> Tensor:
        r"""
        sum(dim=None, keepdim=False, dtype=None) -> Tensor

        See :func:`torch.sum`
        """
        ...
    @overload
    def sum(self, dim: Sequence[Union[str, ellipsis, None]], keepdim: _bool = False, *, dtype: Optional[_dtype] = None) -> Tensor:
        r"""
        sum(dim=None, keepdim=False, dtype=None) -> Tensor

        See :func:`torch.sum`
        """
        ...
    @overload
    def sum_to_size(self, size: Sequence[Union[_int, SymInt]]) -> Tensor:
        r"""
        sum_to_size(*size) -> Tensor

        Sum ``this`` tensor to :attr:`size`.
        :attr:`size` must be broadcastable to ``this`` tensor size.

        Args:
            size (int...): a sequence of integers defining the shape of the output tensor.
        """
        ...
    @overload
    def sum_to_size(self, *size: _int) -> Tensor:
        r"""
        sum_to_size(*size) -> Tensor

        Sum ``this`` tensor to :attr:`size`.
        :attr:`size` must be broadcastable to ``this`` tensor size.

        Args:
            size (int...): a sequence of integers defining the shape of the output tensor.
        """
        ...
    def svd(self, some: _bool = True, compute_uv: _bool = True) -> torch.return_types.svd:
        r"""
        svd(some=True, compute_uv=True) -> (Tensor, Tensor, Tensor)

        See :func:`torch.svd`
        """
        ...
    def swapaxes(self, axis0: _int, axis1: _int) -> Tensor:
        r"""
        swapaxes(axis0, axis1) -> Tensor

        See :func:`torch.swapaxes`
        """
        ...
    def swapaxes_(self, axis0: _int, axis1: _int) -> Tensor:
        r"""
        swapaxes_(axis0, axis1) -> Tensor

        In-place version of :meth:`~Tensor.swapaxes`
        """
        ...
    def swapdims(self, dim0: _int, dim1: _int) -> Tensor:
        r"""
        swapdims(dim0, dim1) -> Tensor

        See :func:`torch.swapdims`
        """
        ...
    def swapdims_(self, dim0: _int, dim1: _int) -> Tensor:
        r"""
        swapdims_(dim0, dim1) -> Tensor

        In-place version of :meth:`~Tensor.swapdims`
        """
        ...
    def t(self) -> Tensor:
        r"""
        t() -> Tensor

        See :func:`torch.t`
        """
        ...
    def t_(self) -> Tensor:
        r"""
        t_() -> Tensor

        In-place version of :meth:`~Tensor.t`
        """
        ...
    def take(self, index: Tensor) -> Tensor:
        r"""
        take(indices) -> Tensor

        See :func:`torch.take`
        """
        ...
    def take_along_dim(self, indices: Tensor, dim: Optional[_int] = None) -> Tensor:
        r"""
        take_along_dim(indices, dim) -> Tensor

        See :func:`torch.take_along_dim`
        """
        ...
    def tan(self) -> Tensor:
        r"""
        tan() -> Tensor

        See :func:`torch.tan`
        """
        ...
    def tan_(self) -> Tensor:
        r"""
        tan_() -> Tensor

        In-place version of :meth:`~Tensor.tan`
        """
        ...
    def tanh(self) -> Tensor:
        r"""
        tanh() -> Tensor

        See :func:`torch.tanh`
        """
        ...
    def tanh_(self) -> Tensor:
        r"""
        tanh_() -> Tensor

        In-place version of :meth:`~Tensor.tanh`
        """
        ...
    @overload
    def tensor_split(self, indices: Sequence[Union[_int, SymInt]], dim: _int = 0) -> Tuple[Tensor, ...]:
        r"""
        tensor_split(indices_or_sections, dim=0) -> List of Tensors

        See :func:`torch.tensor_split`
        """
        ...
    @overload
    def tensor_split(self, tensor_indices_or_sections: Tensor, dim: _int = 0) -> Tuple[Tensor, ...]:
        r"""
        tensor_split(indices_or_sections, dim=0) -> List of Tensors

        See :func:`torch.tensor_split`
        """
        ...
    @overload
    def tensor_split(self, sections: Union[_int, SymInt], dim: _int = 0) -> Tuple[Tensor, ...]:
        r"""
        tensor_split(indices_or_sections, dim=0) -> List of Tensors

        See :func:`torch.tensor_split`
        """
        ...
    @overload
    def tile(self, dims: Sequence[Union[_int, SymInt]]) -> Tensor:
        r"""
        tile(dims) -> Tensor

        See :func:`torch.tile`
        """
        ...
    @overload
    def tile(self, *dims: _int) -> Tensor:
        r"""
        tile(dims) -> Tensor

        See :func:`torch.tile`
        """
        ...
    @overload
    def to(self, dtype: _dtype, non_blocking: _bool = False, copy: _bool = False, *, memory_format: Optional[torch.memory_format] = None) -> Tensor:
        r"""
        to(*args, **kwargs) -> Tensor

        Performs Tensor dtype and/or device conversion. A :class:`torch.dtype` and :class:`torch.device` are
        inferred from the arguments of ``self.to(*args, **kwargs)``.

        .. note::

            If the ``self`` Tensor already
            has the correct :class:`torch.dtype` and :class:`torch.device`, then ``self`` is returned.
            Otherwise, the returned tensor is a copy of ``self`` with the desired
            :class:`torch.dtype` and :class:`torch.device`.

        Here are the ways to call ``to``:

        .. method:: to(dtype, non_blocking=False, copy=False, memory_format=torch.preserve_format) -> Tensor
           :noindex:

            Returns a Tensor with the specified :attr:`dtype`

            Args:
                memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.

        .. method:: to(device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format) -> Tensor
           :noindex:

            Returns a Tensor with the specified :attr:`device` and (optional)
            :attr:`dtype`. If :attr:`dtype` is ``None`` it is inferred to be ``self.dtype``.
            When :attr:`non_blocking`, tries to convert asynchronously with respect to
            the host if possible, e.g., converting a CPU Tensor with pinned memory to a
            CUDA Tensor.
            When :attr:`copy` is set, a new Tensor is created even when the Tensor
            already matches the desired conversion.

            Args:
                memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.

        .. method:: to(other, non_blocking=False, copy=False) -> Tensor
           :noindex:

            Returns a Tensor with same :class:`torch.dtype` and :class:`torch.device` as
            the Tensor :attr:`other`. When :attr:`non_blocking`, tries to convert
            asynchronously with respect to the host if possible, e.g., converting a CPU
            Tensor with pinned memory to a CUDA Tensor.
            When :attr:`copy` is set, a new Tensor is created even when the Tensor
            already matches the desired conversion.

        Example::

            >>> tensor = torch.randn(2, 2)  # Initially dtype=float32, device=cpu
            >>> tensor.to(torch.float64)
            tensor([[-0.5044,  0.0005],
                    [ 0.3310, -0.0584]], dtype=torch.float64)

            >>> cuda0 = torch.device('cuda:0')
            >>> tensor.to(cuda0)
            tensor([[-0.5044,  0.0005],
                    [ 0.3310, -0.0584]], device='cuda:0')

            >>> tensor.to(cuda0, dtype=torch.float64)
            tensor([[-0.5044,  0.0005],
                    [ 0.3310, -0.0584]], dtype=torch.float64, device='cuda:0')

            >>> other = torch.randn((), dtype=torch.float64, device=cuda0)
            >>> tensor.to(other, non_blocking=True)
            tensor([[-0.5044,  0.0005],
                    [ 0.3310, -0.0584]], dtype=torch.float64, device='cuda:0')
        """
        ...
    @overload
    def to(self, device: Optional[DeviceLikeType] = None, dtype: Optional[_dtype] = None, non_blocking: _bool = False, copy: _bool = False, *, memory_format: Optional[torch.memory_format] = None) -> Tensor:
        r"""
        to(*args, **kwargs) -> Tensor

        Performs Tensor dtype and/or device conversion. A :class:`torch.dtype` and :class:`torch.device` are
        inferred from the arguments of ``self.to(*args, **kwargs)``.

        .. note::

            If the ``self`` Tensor already
            has the correct :class:`torch.dtype` and :class:`torch.device`, then ``self`` is returned.
            Otherwise, the returned tensor is a copy of ``self`` with the desired
            :class:`torch.dtype` and :class:`torch.device`.

        Here are the ways to call ``to``:

        .. method:: to(dtype, non_blocking=False, copy=False, memory_format=torch.preserve_format) -> Tensor
           :noindex:

            Returns a Tensor with the specified :attr:`dtype`

            Args:
                memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.

        .. method:: to(device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format) -> Tensor
           :noindex:

            Returns a Tensor with the specified :attr:`device` and (optional)
            :attr:`dtype`. If :attr:`dtype` is ``None`` it is inferred to be ``self.dtype``.
            When :attr:`non_blocking`, tries to convert asynchronously with respect to
            the host if possible, e.g., converting a CPU Tensor with pinned memory to a
            CUDA Tensor.
            When :attr:`copy` is set, a new Tensor is created even when the Tensor
            already matches the desired conversion.

            Args:
                memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.

        .. method:: to(other, non_blocking=False, copy=False) -> Tensor
           :noindex:

            Returns a Tensor with same :class:`torch.dtype` and :class:`torch.device` as
            the Tensor :attr:`other`. When :attr:`non_blocking`, tries to convert
            asynchronously with respect to the host if possible, e.g., converting a CPU
            Tensor with pinned memory to a CUDA Tensor.
            When :attr:`copy` is set, a new Tensor is created even when the Tensor
            already matches the desired conversion.

        Example::

            >>> tensor = torch.randn(2, 2)  # Initially dtype=float32, device=cpu
            >>> tensor.to(torch.float64)
            tensor([[-0.5044,  0.0005],
                    [ 0.3310, -0.0584]], dtype=torch.float64)

            >>> cuda0 = torch.device('cuda:0')
            >>> tensor.to(cuda0)
            tensor([[-0.5044,  0.0005],
                    [ 0.3310, -0.0584]], device='cuda:0')

            >>> tensor.to(cuda0, dtype=torch.float64)
            tensor([[-0.5044,  0.0005],
                    [ 0.3310, -0.0584]], dtype=torch.float64, device='cuda:0')

            >>> other = torch.randn((), dtype=torch.float64, device=cuda0)
            >>> tensor.to(other, non_blocking=True)
            tensor([[-0.5044,  0.0005],
                    [ 0.3310, -0.0584]], dtype=torch.float64, device='cuda:0')
        """
        ...
    @overload
    def to(self, other: Tensor, non_blocking: _bool = False, copy: _bool = False, *, memory_format: Optional[torch.memory_format] = None) -> Tensor:
        r"""
        to(*args, **kwargs) -> Tensor

        Performs Tensor dtype and/or device conversion. A :class:`torch.dtype` and :class:`torch.device` are
        inferred from the arguments of ``self.to(*args, **kwargs)``.

        .. note::

            If the ``self`` Tensor already
            has the correct :class:`torch.dtype` and :class:`torch.device`, then ``self`` is returned.
            Otherwise, the returned tensor is a copy of ``self`` with the desired
            :class:`torch.dtype` and :class:`torch.device`.

        Here are the ways to call ``to``:

        .. method:: to(dtype, non_blocking=False, copy=False, memory_format=torch.preserve_format) -> Tensor
           :noindex:

            Returns a Tensor with the specified :attr:`dtype`

            Args:
                memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.

        .. method:: to(device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format) -> Tensor
           :noindex:

            Returns a Tensor with the specified :attr:`device` and (optional)
            :attr:`dtype`. If :attr:`dtype` is ``None`` it is inferred to be ``self.dtype``.
            When :attr:`non_blocking`, tries to convert asynchronously with respect to
            the host if possible, e.g., converting a CPU Tensor with pinned memory to a
            CUDA Tensor.
            When :attr:`copy` is set, a new Tensor is created even when the Tensor
            already matches the desired conversion.

            Args:
                memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.

        .. method:: to(other, non_blocking=False, copy=False) -> Tensor
           :noindex:

            Returns a Tensor with same :class:`torch.dtype` and :class:`torch.device` as
            the Tensor :attr:`other`. When :attr:`non_blocking`, tries to convert
            asynchronously with respect to the host if possible, e.g., converting a CPU
            Tensor with pinned memory to a CUDA Tensor.
            When :attr:`copy` is set, a new Tensor is created even when the Tensor
            already matches the desired conversion.

        Example::

            >>> tensor = torch.randn(2, 2)  # Initially dtype=float32, device=cpu
            >>> tensor.to(torch.float64)
            tensor([[-0.5044,  0.0005],
                    [ 0.3310, -0.0584]], dtype=torch.float64)

            >>> cuda0 = torch.device('cuda:0')
            >>> tensor.to(cuda0)
            tensor([[-0.5044,  0.0005],
                    [ 0.3310, -0.0584]], device='cuda:0')

            >>> tensor.to(cuda0, dtype=torch.float64)
            tensor([[-0.5044,  0.0005],
                    [ 0.3310, -0.0584]], dtype=torch.float64, device='cuda:0')

            >>> other = torch.randn((), dtype=torch.float64, device=cuda0)
            >>> tensor.to(other, non_blocking=True)
            tensor([[-0.5044,  0.0005],
                    [ 0.3310, -0.0584]], dtype=torch.float64, device='cuda:0')
        """
        ...
    def to_dense(self, dtype: Optional[_dtype] = None, *, masked_grad: Optional[_bool] = None) -> Tensor:
        r"""
        to_dense(dtype=None, *, masked_grad=True) -> Tensor

        Creates a strided copy of :attr:`self` if :attr:`self` is not a strided tensor, otherwise returns :attr:`self`.

        Keyword args:
            {dtype}
            masked_grad (bool, optional): If set to ``True`` (default) and
              :attr:`self` has a sparse layout then the backward of
              :meth:`to_dense` returns ``grad.sparse_mask(self)``.

        Example::

            >>> s = torch.sparse_coo_tensor(
            ...        torch.tensor([[1, 1],
            ...                      [0, 2]]),
            ...        torch.tensor([9, 10]),
            ...        size=(3, 3))
            >>> s.to_dense()
            tensor([[ 0,  0,  0],
                    [ 9,  0, 10],
                    [ 0,  0,  0]])
        """
        ...
    def to_mkldnn(self, dtype: Optional[_dtype] = None) -> Tensor:
        r"""
        to_mkldnn() -> Tensor
        Returns a copy of the tensor in ``torch.mkldnn`` layout.
        """
        ...
    def to_padded_tensor(self, padding: _float, output_size: Optional[Sequence[Union[_int, SymInt]]] = None) -> Tensor:
        r"""
        to_padded_tensor(padding, output_size=None) -> Tensor
        See :func:`to_padded_tensor`
        """
        ...
    @overload
    def to_sparse(self, *, layout: Optional[_layout] = None, blocksize: Optional[Union[_int, _size]] = None, dense_dim: Optional[_int] = None) -> Tensor:
        r"""
        to_sparse(sparseDims) -> Tensor

        Returns a sparse copy of the tensor.  PyTorch supports sparse tensors in
        :ref:`coordinate format <sparse-coo-docs>`.

        Args:
            sparseDims (int, optional): the number of sparse dimensions to include in the new sparse tensor

        Example::

            >>> d = torch.tensor([[0, 0, 0], [9, 0, 10], [0, 0, 0]])
            >>> d
            tensor([[ 0,  0,  0],
                    [ 9,  0, 10],
                    [ 0,  0,  0]])
            >>> d.to_sparse()
            tensor(indices=tensor([[1, 1],
                                   [0, 2]]),
                   values=tensor([ 9, 10]),
                   size=(3, 3), nnz=2, layout=torch.sparse_coo)
            >>> d.to_sparse(1)
            tensor(indices=tensor([[1]]),
                   values=tensor([[ 9,  0, 10]]),
                   size=(3, 3), nnz=1, layout=torch.sparse_coo)

        .. method:: to_sparse(*, layout=None, blocksize=None, dense_dim=None) -> Tensor
           :noindex:

        Returns a sparse tensor with the specified layout and blocksize.  If
        the :attr:`self` is strided, the number of dense dimensions could be
        specified, and a hybrid sparse tensor will be created, with
        `dense_dim` dense dimensions and `self.dim() - 2 - dense_dim` batch
        dimension.

        .. note:: If the :attr:`self` layout and blocksize parameters match
                  with the specified layout and blocksize, return
                  :attr:`self`. Otherwise, return a sparse tensor copy of
                  :attr:`self`.

        Args:

            layout (:class:`torch.layout`, optional): The desired sparse
              layout. One of ``torch.sparse_coo``, ``torch.sparse_csr``,
              ``torch.sparse_csc``, ``torch.sparse_bsr``, or
              ``torch.sparse_bsc``. Default: if ``None``,
              ``torch.sparse_coo``.

            blocksize (list, tuple, :class:`torch.Size`, optional): Block size
              of the resulting BSR or BSC tensor. For other layouts,
              specifying the block size that is not ``None`` will result in a
              RuntimeError exception.  A block size must be a tuple of length
              two such that its items evenly divide the two sparse dimensions.

            dense_dim (int, optional): Number of dense dimensions of the
              resulting CSR, CSC, BSR or BSC tensor.  This argument should be
              used only if :attr:`self` is a strided tensor, and must be a
              value between 0 and dimension of :attr:`self` tensor minus two.

        Example::

            >>> x = torch.tensor([[1, 0], [0, 0], [2, 3]])
            >>> x.to_sparse(layout=torch.sparse_coo)
            tensor(indices=tensor([[0, 2, 2],
                                   [0, 0, 1]]),
                   values=tensor([1, 2, 3]),
                   size=(3, 2), nnz=3, layout=torch.sparse_coo)
            >>> x.to_sparse(layout=torch.sparse_bsr, blocksize=(1, 2))
            tensor(crow_indices=tensor([0, 1, 1, 2]),
                   col_indices=tensor([0, 0]),
                   values=tensor([[[1, 0]],
                                  [[2, 3]]]), size=(3, 2), nnz=2, layout=torch.sparse_bsr)
            >>> x.to_sparse(layout=torch.sparse_bsr, blocksize=(2, 1))
            RuntimeError: Tensor size(-2) 3 needs to be divisible by blocksize[0] 2
            >>> x.to_sparse(layout=torch.sparse_csr, blocksize=(3, 1))
            RuntimeError: to_sparse for Strided to SparseCsr conversion does not use specified blocksize

            >>> x = torch.tensor([[[1], [0]], [[0], [0]], [[2], [3]]])
            >>> x.to_sparse(layout=torch.sparse_csr, dense_dim=1)
            tensor(crow_indices=tensor([0, 1, 1, 3]),
                   col_indices=tensor([0, 0, 1]),
                   values=tensor([[1],
                                  [2],
                                  [3]]), size=(3, 2, 1), nnz=3, layout=torch.sparse_csr)
        """
        ...
    @overload
    def to_sparse(self, sparse_dim: _int) -> Tensor:
        r"""
        to_sparse(sparseDims) -> Tensor

        Returns a sparse copy of the tensor.  PyTorch supports sparse tensors in
        :ref:`coordinate format <sparse-coo-docs>`.

        Args:
            sparseDims (int, optional): the number of sparse dimensions to include in the new sparse tensor

        Example::

            >>> d = torch.tensor([[0, 0, 0], [9, 0, 10], [0, 0, 0]])
            >>> d
            tensor([[ 0,  0,  0],
                    [ 9,  0, 10],
                    [ 0,  0,  0]])
            >>> d.to_sparse()
            tensor(indices=tensor([[1, 1],
                                   [0, 2]]),
                   values=tensor([ 9, 10]),
                   size=(3, 3), nnz=2, layout=torch.sparse_coo)
            >>> d.to_sparse(1)
            tensor(indices=tensor([[1]]),
                   values=tensor([[ 9,  0, 10]]),
                   size=(3, 3), nnz=1, layout=torch.sparse_coo)

        .. method:: to_sparse(*, layout=None, blocksize=None, dense_dim=None) -> Tensor
           :noindex:

        Returns a sparse tensor with the specified layout and blocksize.  If
        the :attr:`self` is strided, the number of dense dimensions could be
        specified, and a hybrid sparse tensor will be created, with
        `dense_dim` dense dimensions and `self.dim() - 2 - dense_dim` batch
        dimension.

        .. note:: If the :attr:`self` layout and blocksize parameters match
                  with the specified layout and blocksize, return
                  :attr:`self`. Otherwise, return a sparse tensor copy of
                  :attr:`self`.

        Args:

            layout (:class:`torch.layout`, optional): The desired sparse
              layout. One of ``torch.sparse_coo``, ``torch.sparse_csr``,
              ``torch.sparse_csc``, ``torch.sparse_bsr``, or
              ``torch.sparse_bsc``. Default: if ``None``,
              ``torch.sparse_coo``.

            blocksize (list, tuple, :class:`torch.Size`, optional): Block size
              of the resulting BSR or BSC tensor. For other layouts,
              specifying the block size that is not ``None`` will result in a
              RuntimeError exception.  A block size must be a tuple of length
              two such that its items evenly divide the two sparse dimensions.

            dense_dim (int, optional): Number of dense dimensions of the
              resulting CSR, CSC, BSR or BSC tensor.  This argument should be
              used only if :attr:`self` is a strided tensor, and must be a
              value between 0 and dimension of :attr:`self` tensor minus two.

        Example::

            >>> x = torch.tensor([[1, 0], [0, 0], [2, 3]])
            >>> x.to_sparse(layout=torch.sparse_coo)
            tensor(indices=tensor([[0, 2, 2],
                                   [0, 0, 1]]),
                   values=tensor([1, 2, 3]),
                   size=(3, 2), nnz=3, layout=torch.sparse_coo)
            >>> x.to_sparse(layout=torch.sparse_bsr, blocksize=(1, 2))
            tensor(crow_indices=tensor([0, 1, 1, 2]),
                   col_indices=tensor([0, 0]),
                   values=tensor([[[1, 0]],
                                  [[2, 3]]]), size=(3, 2), nnz=2, layout=torch.sparse_bsr)
            >>> x.to_sparse(layout=torch.sparse_bsr, blocksize=(2, 1))
            RuntimeError: Tensor size(-2) 3 needs to be divisible by blocksize[0] 2
            >>> x.to_sparse(layout=torch.sparse_csr, blocksize=(3, 1))
            RuntimeError: to_sparse for Strided to SparseCsr conversion does not use specified blocksize

            >>> x = torch.tensor([[[1], [0]], [[0], [0]], [[2], [3]]])
            >>> x.to_sparse(layout=torch.sparse_csr, dense_dim=1)
            tensor(crow_indices=tensor([0, 1, 1, 3]),
                   col_indices=tensor([0, 0, 1]),
                   values=tensor([[1],
                                  [2],
                                  [3]]), size=(3, 2, 1), nnz=3, layout=torch.sparse_csr)
        """
        ...
    def to_sparse_bsc(self, blocksize: Union[_int, _size], dense_dim: Optional[_int] = None) -> Tensor:
        r"""
        to_sparse_bsc(blocksize, dense_dim) -> Tensor

        Convert a tensor to a block sparse column (BSC) storage format of
        given blocksize.  If the :attr:`self` is strided, then the number of
        dense dimensions could be specified, and a hybrid BSC tensor will be
        created, with `dense_dim` dense dimensions and `self.dim() - 2 -
        dense_dim` batch dimension.

        Args:

            blocksize (list, tuple, :class:`torch.Size`, optional): Block size
              of the resulting BSC tensor. A block size must be a tuple of
              length two such that its items evenly divide the two sparse
              dimensions.

            dense_dim (int, optional): Number of dense dimensions of the
              resulting BSC tensor.  This argument should be used only if
              :attr:`self` is a strided tensor, and must be a value between 0
              and dimension of :attr:`self` tensor minus two.

        Example::

            >>> dense = torch.randn(10, 10)
            >>> sparse = dense.to_sparse_csr()
            >>> sparse_bsc = sparse.to_sparse_bsc((5, 5))
            >>> sparse_bsc.row_indices()
            tensor([0, 1, 0, 1])

            >>> dense = torch.zeros(4, 3, 1)
            >>> dense[0:2, 0] = dense[0:2, 2] = dense[2:4, 1] = 1
            >>> dense.to_sparse_bsc((2, 1), 1)
            tensor(ccol_indices=tensor([0, 1, 2, 3]),
                   row_indices=tensor([0, 1, 0]),
                   values=tensor([[[[1.]],

                                   [[1.]]],


                                  [[[1.]],

                                   [[1.]]],


                                  [[[1.]],

                                   [[1.]]]]), size=(4, 3, 1), nnz=3,
                   layout=torch.sparse_bsc)
        """
        ...
    def to_sparse_bsr(self, blocksize: Union[_int, _size], dense_dim: Optional[_int] = None) -> Tensor:
        r"""
        to_sparse_bsr(blocksize, dense_dim) -> Tensor

        Convert a tensor to a block sparse row (BSR) storage format of given
        blocksize.  If the :attr:`self` is strided, then the number of dense
        dimensions could be specified, and a hybrid BSR tensor will be
        created, with `dense_dim` dense dimensions and `self.dim() - 2 -
        dense_dim` batch dimension.

        Args:

            blocksize (list, tuple, :class:`torch.Size`, optional): Block size
              of the resulting BSR tensor. A block size must be a tuple of
              length two such that its items evenly divide the two sparse
              dimensions.

            dense_dim (int, optional): Number of dense dimensions of the
              resulting BSR tensor.  This argument should be used only if
              :attr:`self` is a strided tensor, and must be a value between 0
              and dimension of :attr:`self` tensor minus two.

        Example::

            >>> dense = torch.randn(10, 10)
            >>> sparse = dense.to_sparse_csr()
            >>> sparse_bsr = sparse.to_sparse_bsr((5, 5))
            >>> sparse_bsr.col_indices()
            tensor([0, 1, 0, 1])

            >>> dense = torch.zeros(4, 3, 1)
            >>> dense[0:2, 0] = dense[0:2, 2] = dense[2:4, 1] = 1
            >>> dense.to_sparse_bsr((2, 1), 1)
            tensor(crow_indices=tensor([0, 2, 3]),
                   col_indices=tensor([0, 2, 1]),
                   values=tensor([[[[1.]],

                                   [[1.]]],


                                  [[[1.]],

                                   [[1.]]],


                                  [[[1.]],

                                   [[1.]]]]), size=(4, 3, 1), nnz=3,
                   layout=torch.sparse_bsr)
        """
        ...
    def to_sparse_csc(self, dense_dim: Optional[_int] = None) -> Tensor:
        r"""
        to_sparse_csc() -> Tensor

        Convert a tensor to compressed column storage (CSC) format.  Except
        for strided tensors, only works with 2D tensors.  If the :attr:`self`
        is strided, then the number of dense dimensions could be specified,
        and a hybrid CSC tensor will be created, with `dense_dim` dense
        dimensions and `self.dim() - 2 - dense_dim` batch dimension.

        Args:

            dense_dim (int, optional): Number of dense dimensions of the
              resulting CSC tensor.  This argument should be used only if
              :attr:`self` is a strided tensor, and must be a value between 0
              and dimension of :attr:`self` tensor minus two.

        Example::

            >>> dense = torch.randn(5, 5)
            >>> sparse = dense.to_sparse_csc()
            >>> sparse._nnz()
            25

            >>> dense = torch.zeros(3, 3, 1, 1)
            >>> dense[0, 0] = dense[1, 2] = dense[2, 1] = 1
            >>> dense.to_sparse_csc(dense_dim=2)
            tensor(ccol_indices=tensor([0, 1, 2, 3]),
                   row_indices=tensor([0, 2, 1]),
                   values=tensor([[[1.]],

                                  [[1.]],

                                  [[1.]]]), size=(3, 3, 1, 1), nnz=3,
                   layout=torch.sparse_csc)
        """
        ...
    def to_sparse_csr(self, dense_dim: Optional[_int] = None) -> Tensor:
        r"""
        to_sparse_csr(dense_dim=None) -> Tensor

        Convert a tensor to compressed row storage format (CSR).  Except for
        strided tensors, only works with 2D tensors.  If the :attr:`self` is
        strided, then the number of dense dimensions could be specified, and a
        hybrid CSR tensor will be created, with `dense_dim` dense dimensions
        and `self.dim() - 2 - dense_dim` batch dimension.

        Args:

            dense_dim (int, optional): Number of dense dimensions of the
              resulting CSR tensor.  This argument should be used only if
              :attr:`self` is a strided tensor, and must be a value between 0
              and dimension of :attr:`self` tensor minus two.

        Example::

            >>> dense = torch.randn(5, 5)
            >>> sparse = dense.to_sparse_csr()
            >>> sparse._nnz()
            25

            >>> dense = torch.zeros(3, 3, 1, 1)
            >>> dense[0, 0] = dense[1, 2] = dense[2, 1] = 1
            >>> dense.to_sparse_csr(dense_dim=2)
            tensor(crow_indices=tensor([0, 1, 2, 3]),
                   col_indices=tensor([0, 2, 1]),
                   values=tensor([[[1.]],

                                  [[1.]],

                                  [[1.]]]), size=(3, 3, 1, 1), nnz=3,
                   layout=torch.sparse_csr)
        """
        ...
    def tolist(self) -> List:
        r"""
        tolist() -> list or number

        Returns the tensor as a (nested) list. For scalars, a standard
        Python number is returned, just like with :meth:`~Tensor.item`.
        Tensors are automatically moved to the CPU first if necessary.

        This operation is not differentiable.

        Examples::

            >>> a = torch.randn(2, 2)
            >>> a.tolist()
            [[0.012766935862600803, 0.5415473580360413],
             [-0.08909505605697632, 0.7729271650314331]]
            >>> a[0,0].tolist()
            0.012766935862600803
        """
        ...
    def topk(self, k: Union[_int, SymInt], dim: _int = -1, largest: _bool = True, sorted: _bool = True) -> torch.return_types.topk:
        r"""
        topk(k, dim=None, largest=True, sorted=True) -> (Tensor, LongTensor)

        See :func:`torch.topk`
        """
        ...
    def trace(self) -> Tensor:
        r"""
        trace() -> Tensor

        See :func:`torch.trace`
        """
        ...
    @overload
    def transpose(self, dim0: _int, dim1: _int) -> Tensor:
        r"""
        transpose(dim0, dim1) -> Tensor

        See :func:`torch.transpose`
        """
        ...
    @overload
    def transpose(self, dim0: Union[str, ellipsis, None], dim1: Union[str, ellipsis, None]) -> Tensor:
        r"""
        transpose(dim0, dim1) -> Tensor

        See :func:`torch.transpose`
        """
        ...
    def transpose_(self, dim0: _int, dim1: _int) -> Tensor:
        r"""
        transpose_(dim0, dim1) -> Tensor

        In-place version of :meth:`~Tensor.transpose`
        """
        ...
    def triangular_solve(self, A: Tensor, upper: _bool = True, transpose: _bool = False, unitriangular: _bool = False) -> torch.return_types.triangular_solve:
        r"""
        triangular_solve(A, upper=True, transpose=False, unitriangular=False) -> (Tensor, Tensor)

        See :func:`torch.triangular_solve`
        """
        ...
    def tril(self, diagonal: _int = 0) -> Tensor:
        r"""
        tril(diagonal=0) -> Tensor

        See :func:`torch.tril`
        """
        ...
    def tril_(self, diagonal: _int = 0) -> Tensor:
        r"""
        tril_(diagonal=0) -> Tensor

        In-place version of :meth:`~Tensor.tril`
        """
        ...
    def triu(self, diagonal: _int = 0) -> Tensor:
        r"""
        triu(diagonal=0) -> Tensor

        See :func:`torch.triu`
        """
        ...
    def triu_(self, diagonal: _int = 0) -> Tensor:
        r"""
        triu_(diagonal=0) -> Tensor

        In-place version of :meth:`~Tensor.triu`
        """
        ...
    def true_divide(self, other: Union[Tensor, Number, torch.SymInt, torch.SymFloat], *, out: Optional[Tensor] = None) -> Tensor:
        r"""
        true_divide(value) -> Tensor

        See :func:`torch.true_divide`
        """
        ...
    def true_divide_(self, other: Union[Tensor, Number, torch.SymInt, torch.SymFloat]) -> Tensor:
        r"""
        true_divide_(value) -> Tensor

        In-place version of :meth:`~Tensor.true_divide_`
        """
        ...
    def trunc(self) -> Tensor:
        r"""
        trunc() -> Tensor

        See :func:`torch.trunc`
        """
        ...
    def trunc_(self) -> Tensor:
        r"""
        trunc_() -> Tensor

        In-place version of :meth:`~Tensor.trunc`
        """
        ...
    @overload
    def type(self, dtype: None = None, non_blocking: _bool = False) -> str:
        r"""
        type(dtype=None, non_blocking=False, **kwargs) -> str or Tensor
        Returns the type if `dtype` is not provided, else casts this object to
        the specified type.

        If this is already of the correct type, no copy is performed and the
        original object is returned.

        Args:
            dtype (dtype or string): The desired type
            non_blocking (bool): If ``True``, and the source is in pinned memory
                and destination is on the GPU or vice versa, the copy is performed
                asynchronously with respect to the host. Otherwise, the argument
                has no effect.
            **kwargs: For compatibility, may contain the key ``async`` in place of
                the ``non_blocking`` argument. The ``async`` arg is deprecated.
        """
        ...
    @overload
    def type(self, dtype: Union[str, _dtype], non_blocking: _bool = False) -> Tensor:
        r"""
        type(dtype=None, non_blocking=False, **kwargs) -> str or Tensor
        Returns the type if `dtype` is not provided, else casts this object to
        the specified type.

        If this is already of the correct type, no copy is performed and the
        original object is returned.

        Args:
            dtype (dtype or string): The desired type
            non_blocking (bool): If ``True``, and the source is in pinned memory
                and destination is on the GPU or vice versa, the copy is performed
                asynchronously with respect to the host. Otherwise, the argument
                has no effect.
            **kwargs: For compatibility, may contain the key ``async`` in place of
                the ``non_blocking`` argument. The ``async`` arg is deprecated.
        """
        ...
    def type_as(self, other: Tensor) -> Tensor:
        r"""
        type_as(tensor) -> Tensor

        Returns this tensor cast to the type of the given tensor.

        This is a no-op if the tensor is already of the correct type. This is
        equivalent to ``self.type(tensor.type())``

        Args:
            tensor (Tensor): the tensor which has the desired type
        """
        ...
    @overload
    def unbind(self, dim: _int = 0) -> Tuple[Tensor, ...]:
        r"""
        unbind(dim=0) -> seq

        See :func:`torch.unbind`
        """
        ...
    @overload
    def unbind(self, dim: Union[str, ellipsis, None]) -> Tuple[Tensor, ...]:
        r"""
        unbind(dim=0) -> seq

        See :func:`torch.unbind`
        """
        ...
    @overload
    def unflatten(self, dim: Union[str, ellipsis, None], sizes: Sequence[Union[_int, SymInt]], names: Sequence[Union[str, ellipsis, None]]) -> Tensor: ...
    @overload
    def unflatten(self, dim: _int, sizes: Sequence[Union[_int, SymInt]]) -> Tensor: ...
    def unfold(self, dimension: _int, size: _int, step: _int) -> Tensor:
        r"""
        unfold(dimension, size, step) -> Tensor

        Returns a view of the original tensor which contains all slices of size :attr:`size` from
        :attr:`self` tensor in the dimension :attr:`dimension`.

        Step between two slices is given by :attr:`step`.

        If `sizedim` is the size of dimension :attr:`dimension` for :attr:`self`, the size of
        dimension :attr:`dimension` in the returned tensor will be
        `(sizedim - size) / step + 1`.

        An additional dimension of size :attr:`size` is appended in the returned tensor.

        Args:
            dimension (int): dimension in which unfolding happens
            size (int): the size of each slice that is unfolded
            step (int): the step between each slice

        Example::

            >>> x = torch.arange(1., 8)
            >>> x
            tensor([ 1.,  2.,  3.,  4.,  5.,  6.,  7.])
            >>> x.unfold(0, 2, 1)
            tensor([[ 1.,  2.],
                    [ 2.,  3.],
                    [ 3.,  4.],
                    [ 4.,  5.],
                    [ 5.,  6.],
                    [ 6.,  7.]])
            >>> x.unfold(0, 2, 2)
            tensor([[ 1.,  2.],
                    [ 3.,  4.],
                    [ 5.,  6.]])
        """
        ...
    def uniform_(self, from_: _float = 0, to: _float = 1, *, generator: Optional[Generator] = None) -> Tensor:
        r"""
        uniform_(from=0, to=1, *, generator=None) -> Tensor

        Fills :attr:`self` tensor with numbers sampled from the continuous uniform
        distribution:

        .. math::
            f(x) = \dfrac{1}{\text{to} - \text{from}}
        """
        ...
    def unsafe_chunk(self, chunks: _int, dim: _int = 0) -> Tuple[Tensor, ...]:
        r"""
        unsafe_chunk(chunks, dim=0) -> List of Tensors

        See :func:`torch.unsafe_chunk`
        """
        ...
    def unsafe_split(self, split_size: Union[_int, SymInt], dim: _int = 0) -> Tuple[Tensor, ...]:
        r"""
        unsafe_split(split_size, dim=0) -> List of Tensors

        See :func:`torch.unsafe_split`
        """
        ...
    def unsafe_split_with_sizes(self, split_sizes: Sequence[Union[_int, SymInt]], dim: _int = 0) -> Tuple[Tensor, ...]: ...
    def unsqueeze(self, dim: _int) -> Tensor:
        r"""
        unsqueeze(dim) -> Tensor

        See :func:`torch.unsqueeze`
        """
        ...
    def unsqueeze_(self, dim: _int) -> Tensor:
        r"""
        unsqueeze_(dim) -> Tensor

        In-place version of :meth:`~Tensor.unsqueeze`
        """
        ...
    def values(self) -> Tensor:
        r"""
        values() -> Tensor

        Return the values tensor of a :ref:`sparse COO tensor <sparse-coo-docs>`.

        .. warning::
          Throws an error if :attr:`self` is not a sparse COO tensor.

        See also :meth:`Tensor.indices`.

        .. note::
          This method can only be called on a coalesced sparse tensor. See
          :meth:`Tensor.coalesce` for details.
        """
        ...
    @overload
    def var(self, dim: Optional[Union[_int, _size]], unbiased: _bool = True, keepdim: _bool = False) -> Tensor:
        r"""
        var(dim=None, *, correction=1, keepdim=False) -> Tensor

        See :func:`torch.var`
        """
        ...
    @overload
    def var(self, dim: Optional[Union[_int, _size]] = None, *, correction: Optional[Union[Number, _complex]] = None, keepdim: _bool = False) -> Tensor:
        r"""
        var(dim=None, *, correction=1, keepdim=False) -> Tensor

        See :func:`torch.var`
        """
        ...
    @overload
    def var(self, unbiased: _bool = True) -> Tensor:
        r"""
        var(dim=None, *, correction=1, keepdim=False) -> Tensor

        See :func:`torch.var`
        """
        ...
    @overload
    def var(self, dim: Sequence[Union[str, ellipsis, None]], unbiased: _bool = True, keepdim: _bool = False) -> Tensor:
        r"""
        var(dim=None, *, correction=1, keepdim=False) -> Tensor

        See :func:`torch.var`
        """
        ...
    @overload
    def var(self, dim: Sequence[Union[str, ellipsis, None]], *, correction: Optional[Union[Number, _complex]] = None, keepdim: _bool = False) -> Tensor:
        r"""
        var(dim=None, *, correction=1, keepdim=False) -> Tensor

        See :func:`torch.var`
        """
        ...
    def vdot(self, other: Tensor) -> Tensor:
        r"""
        vdot(other) -> Tensor

        See :func:`torch.vdot`
        """
        ...
    @overload
    def view(self, dtype: _dtype) -> Tensor:
        r"""
        view(*shape) -> Tensor

        Returns a new tensor with the same data as the :attr:`self` tensor but of a
        different :attr:`shape`.

        The returned tensor shares the same data and must have the same number
        of elements, but may have a different size. For a tensor to be viewed, the new
        view size must be compatible with its original size and stride, i.e., each new
        view dimension must either be a subspace of an original dimension, or only span
        across original dimensions :math:`d, d+1, \dots, d+k` that satisfy the following
        contiguity-like condition that :math:`\forall i = d, \dots, d+k-1`,

        .. math::

          \text{stride}[i] = \text{stride}[i+1] \times \text{size}[i+1]

        Otherwise, it will not be possible to view :attr:`self` tensor as :attr:`shape`
        without copying it (e.g., via :meth:`contiguous`). When it is unclear whether a
        :meth:`view` can be performed, it is advisable to use :meth:`reshape`, which
        returns a view if the shapes are compatible, and copies (equivalent to calling
        :meth:`contiguous`) otherwise.

        Args:
            shape (torch.Size or int...): the desired size

        Example::

            >>> x = torch.randn(4, 4)
            >>> x.size()
            torch.Size([4, 4])
            >>> y = x.view(16)
            >>> y.size()
            torch.Size([16])
            >>> z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
            >>> z.size()
            torch.Size([2, 8])

            >>> a = torch.randn(1, 2, 3, 4)
            >>> a.size()
            torch.Size([1, 2, 3, 4])
            >>> b = a.transpose(1, 2)  # Swaps 2nd and 3rd dimension
            >>> b.size()
            torch.Size([1, 3, 2, 4])
            >>> c = a.view(1, 3, 2, 4)  # Does not change tensor layout in memory
            >>> c.size()
            torch.Size([1, 3, 2, 4])
            >>> torch.equal(b, c)
            False


        .. method:: view(dtype) -> Tensor
           :noindex:

        Returns a new tensor with the same data as the :attr:`self` tensor but of a
        different :attr:`dtype`.

        If the element size of :attr:`dtype` is different than that of ``self.dtype``,
        then the size of the last dimension of the output will be scaled
        proportionally.  For instance, if :attr:`dtype` element size is twice that of
        ``self.dtype``, then each pair of elements in the last dimension of
        :attr:`self` will be combined, and the size of the last dimension of the output
        will be half that of :attr:`self`. If :attr:`dtype` element size is half that
        of ``self.dtype``, then each element in the last dimension of :attr:`self` will
        be split in two, and the size of the last dimension of the output will be
        double that of :attr:`self`. For this to be possible, the following conditions
        must be true:

            * ``self.dim()`` must be greater than 0.
            * ``self.stride(-1)`` must be 1.

        Additionally, if the element size of :attr:`dtype` is greater than that of
        ``self.dtype``, the following conditions must be true as well:

            * ``self.size(-1)`` must be divisible by the ratio between the element
              sizes of the dtypes.
            * ``self.storage_offset()`` must be divisible by the ratio between the
              element sizes of the dtypes.
            * The strides of all dimensions, except the last dimension, must be
              divisible by the ratio between the element sizes of the dtypes.

        If any of the above conditions are not met, an error is thrown.

        .. warning::

            This overload is not supported by TorchScript, and using it in a Torchscript
            program will cause undefined behavior.


        Args:
            dtype (:class:`torch.dtype`): the desired dtype

        Example::

            >>> x = torch.randn(4, 4)
            >>> x
            tensor([[ 0.9482, -0.0310,  1.4999, -0.5316],
                    [-0.1520,  0.7472,  0.5617, -0.8649],
                    [-2.4724, -0.0334, -0.2976, -0.8499],
                    [-0.2109,  1.9913, -0.9607, -0.6123]])
            >>> x.dtype
            torch.float32

            >>> y = x.view(torch.int32)
            >>> y
            tensor([[ 1064483442, -1124191867,  1069546515, -1089989247],
                    [-1105482831,  1061112040,  1057999968, -1084397505],
                    [-1071760287, -1123489973, -1097310419, -1084649136],
                    [-1101533110,  1073668768, -1082790149, -1088634448]],
                dtype=torch.int32)
            >>> y[0, 0] = 1000000000
            >>> x
            tensor([[ 0.0047, -0.0310,  1.4999, -0.5316],
                    [-0.1520,  0.7472,  0.5617, -0.8649],
                    [-2.4724, -0.0334, -0.2976, -0.8499],
                    [-0.2109,  1.9913, -0.9607, -0.6123]])

            >>> x.view(torch.cfloat)
            tensor([[ 0.0047-0.0310j,  1.4999-0.5316j],
                    [-0.1520+0.7472j,  0.5617-0.8649j],
                    [-2.4724-0.0334j, -0.2976-0.8499j],
                    [-0.2109+1.9913j, -0.9607-0.6123j]])
            >>> x.view(torch.cfloat).size()
            torch.Size([4, 2])

            >>> x.view(torch.uint8)
            tensor([[  0, 202, 154,  59, 182, 243, 253, 188, 185, 252, 191,  63, 240,  22,
                       8, 191],
                    [227, 165,  27, 190, 128,  72,  63,  63, 146, 203,  15,  63,  22, 106,
                      93, 191],
                    [205,  59,  30, 192, 112, 206,   8, 189,   7,  95, 152, 190,  12, 147,
                      89, 191],
                    [ 43, 246,  87, 190, 235, 226, 254,  63, 111, 240, 117, 191, 177, 191,
                      28, 191]], dtype=torch.uint8)
            >>> x.view(torch.uint8).size()
            torch.Size([4, 16])
        """
        ...
    @overload
    def view(self, size: Sequence[Union[_int, SymInt]]) -> Tensor:
        r"""
        view(*shape) -> Tensor

        Returns a new tensor with the same data as the :attr:`self` tensor but of a
        different :attr:`shape`.

        The returned tensor shares the same data and must have the same number
        of elements, but may have a different size. For a tensor to be viewed, the new
        view size must be compatible with its original size and stride, i.e., each new
        view dimension must either be a subspace of an original dimension, or only span
        across original dimensions :math:`d, d+1, \dots, d+k` that satisfy the following
        contiguity-like condition that :math:`\forall i = d, \dots, d+k-1`,

        .. math::

          \text{stride}[i] = \text{stride}[i+1] \times \text{size}[i+1]

        Otherwise, it will not be possible to view :attr:`self` tensor as :attr:`shape`
        without copying it (e.g., via :meth:`contiguous`). When it is unclear whether a
        :meth:`view` can be performed, it is advisable to use :meth:`reshape`, which
        returns a view if the shapes are compatible, and copies (equivalent to calling
        :meth:`contiguous`) otherwise.

        Args:
            shape (torch.Size or int...): the desired size

        Example::

            >>> x = torch.randn(4, 4)
            >>> x.size()
            torch.Size([4, 4])
            >>> y = x.view(16)
            >>> y.size()
            torch.Size([16])
            >>> z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
            >>> z.size()
            torch.Size([2, 8])

            >>> a = torch.randn(1, 2, 3, 4)
            >>> a.size()
            torch.Size([1, 2, 3, 4])
            >>> b = a.transpose(1, 2)  # Swaps 2nd and 3rd dimension
            >>> b.size()
            torch.Size([1, 3, 2, 4])
            >>> c = a.view(1, 3, 2, 4)  # Does not change tensor layout in memory
            >>> c.size()
            torch.Size([1, 3, 2, 4])
            >>> torch.equal(b, c)
            False


        .. method:: view(dtype) -> Tensor
           :noindex:

        Returns a new tensor with the same data as the :attr:`self` tensor but of a
        different :attr:`dtype`.

        If the element size of :attr:`dtype` is different than that of ``self.dtype``,
        then the size of the last dimension of the output will be scaled
        proportionally.  For instance, if :attr:`dtype` element size is twice that of
        ``self.dtype``, then each pair of elements in the last dimension of
        :attr:`self` will be combined, and the size of the last dimension of the output
        will be half that of :attr:`self`. If :attr:`dtype` element size is half that
        of ``self.dtype``, then each element in the last dimension of :attr:`self` will
        be split in two, and the size of the last dimension of the output will be
        double that of :attr:`self`. For this to be possible, the following conditions
        must be true:

            * ``self.dim()`` must be greater than 0.
            * ``self.stride(-1)`` must be 1.

        Additionally, if the element size of :attr:`dtype` is greater than that of
        ``self.dtype``, the following conditions must be true as well:

            * ``self.size(-1)`` must be divisible by the ratio between the element
              sizes of the dtypes.
            * ``self.storage_offset()`` must be divisible by the ratio between the
              element sizes of the dtypes.
            * The strides of all dimensions, except the last dimension, must be
              divisible by the ratio between the element sizes of the dtypes.

        If any of the above conditions are not met, an error is thrown.

        .. warning::

            This overload is not supported by TorchScript, and using it in a Torchscript
            program will cause undefined behavior.


        Args:
            dtype (:class:`torch.dtype`): the desired dtype

        Example::

            >>> x = torch.randn(4, 4)
            >>> x
            tensor([[ 0.9482, -0.0310,  1.4999, -0.5316],
                    [-0.1520,  0.7472,  0.5617, -0.8649],
                    [-2.4724, -0.0334, -0.2976, -0.8499],
                    [-0.2109,  1.9913, -0.9607, -0.6123]])
            >>> x.dtype
            torch.float32

            >>> y = x.view(torch.int32)
            >>> y
            tensor([[ 1064483442, -1124191867,  1069546515, -1089989247],
                    [-1105482831,  1061112040,  1057999968, -1084397505],
                    [-1071760287, -1123489973, -1097310419, -1084649136],
                    [-1101533110,  1073668768, -1082790149, -1088634448]],
                dtype=torch.int32)
            >>> y[0, 0] = 1000000000
            >>> x
            tensor([[ 0.0047, -0.0310,  1.4999, -0.5316],
                    [-0.1520,  0.7472,  0.5617, -0.8649],
                    [-2.4724, -0.0334, -0.2976, -0.8499],
                    [-0.2109,  1.9913, -0.9607, -0.6123]])

            >>> x.view(torch.cfloat)
            tensor([[ 0.0047-0.0310j,  1.4999-0.5316j],
                    [-0.1520+0.7472j,  0.5617-0.8649j],
                    [-2.4724-0.0334j, -0.2976-0.8499j],
                    [-0.2109+1.9913j, -0.9607-0.6123j]])
            >>> x.view(torch.cfloat).size()
            torch.Size([4, 2])

            >>> x.view(torch.uint8)
            tensor([[  0, 202, 154,  59, 182, 243, 253, 188, 185, 252, 191,  63, 240,  22,
                       8, 191],
                    [227, 165,  27, 190, 128,  72,  63,  63, 146, 203,  15,  63,  22, 106,
                      93, 191],
                    [205,  59,  30, 192, 112, 206,   8, 189,   7,  95, 152, 190,  12, 147,
                      89, 191],
                    [ 43, 246,  87, 190, 235, 226, 254,  63, 111, 240, 117, 191, 177, 191,
                      28, 191]], dtype=torch.uint8)
            >>> x.view(torch.uint8).size()
            torch.Size([4, 16])
        """
        ...
    @overload
    def view(self, *size: _int) -> Tensor:
        r"""
        view(*shape) -> Tensor

        Returns a new tensor with the same data as the :attr:`self` tensor but of a
        different :attr:`shape`.

        The returned tensor shares the same data and must have the same number
        of elements, but may have a different size. For a tensor to be viewed, the new
        view size must be compatible with its original size and stride, i.e., each new
        view dimension must either be a subspace of an original dimension, or only span
        across original dimensions :math:`d, d+1, \dots, d+k` that satisfy the following
        contiguity-like condition that :math:`\forall i = d, \dots, d+k-1`,

        .. math::

          \text{stride}[i] = \text{stride}[i+1] \times \text{size}[i+1]

        Otherwise, it will not be possible to view :attr:`self` tensor as :attr:`shape`
        without copying it (e.g., via :meth:`contiguous`). When it is unclear whether a
        :meth:`view` can be performed, it is advisable to use :meth:`reshape`, which
        returns a view if the shapes are compatible, and copies (equivalent to calling
        :meth:`contiguous`) otherwise.

        Args:
            shape (torch.Size or int...): the desired size

        Example::

            >>> x = torch.randn(4, 4)
            >>> x.size()
            torch.Size([4, 4])
            >>> y = x.view(16)
            >>> y.size()
            torch.Size([16])
            >>> z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
            >>> z.size()
            torch.Size([2, 8])

            >>> a = torch.randn(1, 2, 3, 4)
            >>> a.size()
            torch.Size([1, 2, 3, 4])
            >>> b = a.transpose(1, 2)  # Swaps 2nd and 3rd dimension
            >>> b.size()
            torch.Size([1, 3, 2, 4])
            >>> c = a.view(1, 3, 2, 4)  # Does not change tensor layout in memory
            >>> c.size()
            torch.Size([1, 3, 2, 4])
            >>> torch.equal(b, c)
            False


        .. method:: view(dtype) -> Tensor
           :noindex:

        Returns a new tensor with the same data as the :attr:`self` tensor but of a
        different :attr:`dtype`.

        If the element size of :attr:`dtype` is different than that of ``self.dtype``,
        then the size of the last dimension of the output will be scaled
        proportionally.  For instance, if :attr:`dtype` element size is twice that of
        ``self.dtype``, then each pair of elements in the last dimension of
        :attr:`self` will be combined, and the size of the last dimension of the output
        will be half that of :attr:`self`. If :attr:`dtype` element size is half that
        of ``self.dtype``, then each element in the last dimension of :attr:`self` will
        be split in two, and the size of the last dimension of the output will be
        double that of :attr:`self`. For this to be possible, the following conditions
        must be true:

            * ``self.dim()`` must be greater than 0.
            * ``self.stride(-1)`` must be 1.

        Additionally, if the element size of :attr:`dtype` is greater than that of
        ``self.dtype``, the following conditions must be true as well:

            * ``self.size(-1)`` must be divisible by the ratio between the element
              sizes of the dtypes.
            * ``self.storage_offset()`` must be divisible by the ratio between the
              element sizes of the dtypes.
            * The strides of all dimensions, except the last dimension, must be
              divisible by the ratio between the element sizes of the dtypes.

        If any of the above conditions are not met, an error is thrown.

        .. warning::

            This overload is not supported by TorchScript, and using it in a Torchscript
            program will cause undefined behavior.


        Args:
            dtype (:class:`torch.dtype`): the desired dtype

        Example::

            >>> x = torch.randn(4, 4)
            >>> x
            tensor([[ 0.9482, -0.0310,  1.4999, -0.5316],
                    [-0.1520,  0.7472,  0.5617, -0.8649],
                    [-2.4724, -0.0334, -0.2976, -0.8499],
                    [-0.2109,  1.9913, -0.9607, -0.6123]])
            >>> x.dtype
            torch.float32

            >>> y = x.view(torch.int32)
            >>> y
            tensor([[ 1064483442, -1124191867,  1069546515, -1089989247],
                    [-1105482831,  1061112040,  1057999968, -1084397505],
                    [-1071760287, -1123489973, -1097310419, -1084649136],
                    [-1101533110,  1073668768, -1082790149, -1088634448]],
                dtype=torch.int32)
            >>> y[0, 0] = 1000000000
            >>> x
            tensor([[ 0.0047, -0.0310,  1.4999, -0.5316],
                    [-0.1520,  0.7472,  0.5617, -0.8649],
                    [-2.4724, -0.0334, -0.2976, -0.8499],
                    [-0.2109,  1.9913, -0.9607, -0.6123]])

            >>> x.view(torch.cfloat)
            tensor([[ 0.0047-0.0310j,  1.4999-0.5316j],
                    [-0.1520+0.7472j,  0.5617-0.8649j],
                    [-2.4724-0.0334j, -0.2976-0.8499j],
                    [-0.2109+1.9913j, -0.9607-0.6123j]])
            >>> x.view(torch.cfloat).size()
            torch.Size([4, 2])

            >>> x.view(torch.uint8)
            tensor([[  0, 202, 154,  59, 182, 243, 253, 188, 185, 252, 191,  63, 240,  22,
                       8, 191],
                    [227, 165,  27, 190, 128,  72,  63,  63, 146, 203,  15,  63,  22, 106,
                      93, 191],
                    [205,  59,  30, 192, 112, 206,   8, 189,   7,  95, 152, 190,  12, 147,
                      89, 191],
                    [ 43, 246,  87, 190, 235, 226, 254,  63, 111, 240, 117, 191, 177, 191,
                      28, 191]], dtype=torch.uint8)
            >>> x.view(torch.uint8).size()
            torch.Size([4, 16])
        """
        ...
    def view_as(self, other: Tensor) -> Tensor:
        r"""
        view_as(other) -> Tensor

        View this tensor as the same size as :attr:`other`.
        ``self.view_as(other)`` is equivalent to ``self.view(other.size())``.

        Please see :meth:`~Tensor.view` for more information about ``view``.

        Args:
            other (:class:`torch.Tensor`): The result tensor has the same size
                as :attr:`other`.
        """
        ...
    @overload
    def vsplit(self, sections: _int) -> Tuple[Tensor, ...]:
        r"""
        vsplit(split_size_or_sections) -> List of Tensors

        See :func:`torch.vsplit`
        """
        ...
    @overload
    def vsplit(self, indices: _size) -> Tuple[Tensor, ...]:
        r"""
        vsplit(split_size_or_sections) -> List of Tensors

        See :func:`torch.vsplit`
        """
        ...
    @overload
    def vsplit(self, *indices: _int) -> Tuple[Tensor, ...]:
        r"""
        vsplit(split_size_or_sections) -> List of Tensors

        See :func:`torch.vsplit`
        """
        ...
    @overload
    def where(self, condition: Tensor, other: Tensor) -> Tensor:
        r"""
        where(condition, y) -> Tensor

        ``self.where(condition, y)`` is equivalent to ``torch.where(condition, self, y)``.
        See :func:`torch.where`
        """
        ...
    @overload
    def where(self, condition: Tensor, other: Union[Number, _complex]) -> Tensor:
        r"""
        where(condition, y) -> Tensor

        ``self.where(condition, y)`` is equivalent to ``torch.where(condition, self, y)``.
        See :func:`torch.where`
        """
        ...
    @overload
    def xlogy(self, other: Tensor) -> Tensor:
        r"""
        xlogy(other) -> Tensor

        See :func:`torch.xlogy`
        """
        ...
    @overload
    def xlogy(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        xlogy(other) -> Tensor

        See :func:`torch.xlogy`
        """
        ...
    @overload
    def xlogy_(self, other: Tensor) -> Tensor:
        r"""
        xlogy_(other) -> Tensor

        In-place version of :meth:`~Tensor.xlogy`
        """
        ...
    @overload
    def xlogy_(self, other: Union[Number, _complex]) -> Tensor:
        r"""
        xlogy_(other) -> Tensor

        In-place version of :meth:`~Tensor.xlogy`
        """
        ...
    def zero_(self) -> Tensor:
        r"""
        zero_() -> Tensor

        Fills :attr:`self` tensor with zeros.
        """
        ...

_TensorBase = TensorBase

# Defined in torch/csrc/multiprocessing/init.cpp
def _multiprocessing_init() -> None: ...

# Defined in torch/csrc/mps/Module.cpp
def _mps_deviceSynchronize() -> None: ...
def _mps_get_default_generator() -> Generator: ...
def _mps_emptyCache() -> None: ...
def _mps_setMemoryFraction(fraction: _float) -> None: ...
def _mps_currentAllocatedMemory() -> _int: ...
def _mps_driverAllocatedMemory() -> _int: ...
def _mps_is_available() -> _bool: ...
def _mps_is_on_macos_or_newer(major: _int, minor: _int) -> _bool: ...
def _mps_profilerStartTrace(mode: str, wait_until_completed: _bool) -> None: ...
def _mps_profilerStopTrace() -> None: ...
def _mps_acquireEvent(enable_timing: _bool) -> _int: ...
def _mps_releaseEvent(event_id: _int) -> None: ...
def _mps_recordEvent(event_id: _int) -> None: ...
def _mps_waitForEvent(event_id: _int) -> None: ...
def _mps_synchronizeEvent(event_id: _int) -> None: ...
def _mps_queryEvent(event_id: _int) -> _bool: ...
def _mps_elapsedTimeOfEvents(start_event_id: _int, end_event_id: _int) -> _float: ...


# Defined in torch/csrc/cuda/Module.cpp
def _cuda_getCurrentStream(device: _int) -> Tuple: ...
def _cuda_getCurrentRawStream(device: _int) -> _int: ...
def _cuda_getDefaultStream(device: _int) -> Tuple: ...
def _cuda_getCurrentBlasHandle() -> _int: ...
def _cuda_clearCublasWorkspaces() -> None: ...
def _cuda_setDevice(device: _int) -> None: ...
def _cuda_exchangeDevice(device: _int) -> _int: ...
def _cuda_maybeExchangeDevice(device: _int) -> _int: ...
def _cuda_getDevice() -> _int: ...
def _cuda_getDeviceCount() -> _int: ...
def _cuda_set_sync_debug_mode(warn_level: Union[_int, str]) -> None: ...
def _cuda_get_sync_debug_mode() -> _int: ...
def _cuda_sleep(cycles: _int) -> None: ...
def _cuda_synchronize() -> None: ...
def _cuda_ipc_collect() -> None: ...
def _cuda_getArchFlags() -> Optional[str]: ...
def _cuda_init() -> None: ...
def _cuda_setStream(stream_id: _int, device_index: _int, device_type: _int) -> None: ...
def _cuda_getCompiledVersion() -> _int: ...
def _cuda_cudaHostAllocator() -> _int: ...
def _cuda_cudaCachingAllocator_raw_alloc(size: _int, cuda_stream: _int) -> _int: ...
def _cuda_cudaCachingAllocator_raw_delete(ptr: _int) -> None: ...
def _cuda_cudaCachingAllocator_set_allocator_settings(env: str) -> None: ...
def _cuda_beginAllocateCurrentStreamToPool(device: _int, mempool_id: Tuple[_int, _int]) -> None: ...
def _cuda_endAllocateCurrentStreamToPool(device: _int, mempool_id: Tuple[_int, _int]) -> None: ...
def _cuda_releasePool(device: _int, mempool_id: Tuple[_int, _int]) -> None: ...
def _cuda_checkPoolLiveAllocations(device: _int, mempool_id: Tuple[_int, _int], expected_live_allocations: Set) -> _bool: ...
def _cuda_setCheckpointPoolState(device: _int, state: _cuda_CUDAAllocator_AllocatorState,  stale_storages: List[_int], storages_to_add_deleters_to: List[_int]) -> None: ...
def _cuda_setMemoryFraction(fraction: _float, device: _int) -> None: ...
def _cuda_emptyCache() -> None: ...
def _cuda_memoryStats(device: _int) -> Dict[str, Any]: ...
def _cuda_resetAccumulatedMemoryStats(device: _int) -> None: ...
def _cuda_resetPeakMemoryStats(device: _int) -> None: ...
def _cuda_memorySnapshot() -> Dict[str, Any]: ...
def _cuda_record_memory_history_legacy(
    enabled: _bool,
    record_context: _bool,
    record_context_cpp: _bool,
    alloc_trace_max_entries: _int,
    alloc_trace_record_context: _bool,
) -> None: ...
def _cuda_record_memory_history(
    enabled: Optional[str],
    context: Optional[str],
    stacks: str,
    max_entries
) -> None: ...
def _cuda_isHistoryEnabled() -> _bool: ...

def _cuda_getAllocatorBackend() -> str: ...
class _cuda_CUDAAllocator_AllocatorState:
    pass
def _cuda_getCheckpointState(device: _int, mempool: Tuple[_int, _int]) -> _cuda_CUDAAllocator_AllocatorState: ...
def _set_cached_tensors_enabled(enabled: _bool) -> None: ...
def _add_cached_tensor(t: Tensor) -> None: ...
def _remove_cached_tensor(t: Tensor) -> None: ...
def _tensors_data_ptrs_at_indices_equal(tensors: List[Tensor], ptrs: List[Optional[_int]], indices: List[_int]) -> _bool: ...
def _construct_CUDA_Tensor_From_Storage_And_Metadata(metadata: dict, storage: Storage) -> Tensor: ...
def _storage_Use_Count(storage_ptr: _int) -> _int: ...
def _set_storage_access_error_msg(t: Tensor, s: str) -> None: ...
def _free_And_Remove_DeleterFn(storage_ptr: _int) -> None: ...
def _has_Standard_Deleter(storage_ptr: _int) -> _bool: ...

class _cuda_CUDAAllocator: ...

def _cuda_customAllocator(alloc_fn: _int, free_fn: _int) -> _cuda_CUDAAllocator: ...
def _cuda_changeCurrentAllocator(allocator: _cuda_CUDAAllocator) -> None: ...
def _cuda_getAllocator() -> _cuda_CUDAAllocator: ...
def _cuda_lock_mutex() -> None: ...
def _cuda_unlock_mutex() -> None: ...
def _cuda_canDeviceAccessPeer(device: _int, peer_device: _int) -> _bool: ...
def _cuda_jiterator_compile_and_launch_kernel(
    code_string: str,
    kernel_name: str,
    return_by_ref: _bool,
    num_outputs: _int,
    tensors: Tuple,
    kwargs: Dict[str, Union[_int, _float, _bool]],
) -> Tensor: ...
def _cuda_get_cudnn_benchmark_limit() -> _int: ...
def _cuda_set_cudnn_benchmark_limit(arg: _int) -> None: ...
def _cuda_get_conv_benchmark_empty_cache() -> _bool: ...
def _cudnn_set_conv_benchmark_empty_cache(enable: _bool) -> None: ...
def _nccl_version() -> _int: ...
def _nccl_version_suffix() -> bytes : ...
def _nccl_unique_id() -> bytes: ...
def _nccl_init_rank(nranks: _int, comm_id: bytes, rank: _int) -> object: ...
def _nccl_reduce(
    input: Sequence[Tensor],
    output: Tensor,
    root: _int,
    op: _int,
    streams: Optional[Sequence[_CudaStreamBase]],
    comms: Optional[Sequence[object]],
) -> None: ...
def _nccl_all_reduce(
    input: Sequence[Tensor],
    output: Sequence[Tensor],
    op: _int,
    streams: Optional[Sequence[_CudaStreamBase]],
    comms: Optional[Sequence[object]],
) -> None: ...
def _nccl_broadcast(
    input: Sequence[Tensor],
    root: _int,
    streams: Optional[Sequence[_CudaStreamBase]],
    comms: Optional[Sequence[object]],
) -> None: ...
def _nccl_all_gather(
    input: Sequence[Tensor],
    output: Sequence[Tensor],
    streams: Optional[Sequence[_CudaStreamBase]],
    comms: Optional[Sequence[object]],
) -> None: ...
def _nccl_reduce_scatter(
    input: Sequence[Tensor],
    output: Sequence[Tensor],
    op: _int,
    streams: Optional[Sequence[_CudaStreamBase]],
    comms: Optional[Sequence[object]],
) -> None: ...
def _rocm_is_backward_pass() -> _bool: ...

class _CudaDeviceProperties:
    name: str
    major: _int
    minor: _int
    multi_processor_count: _int
    total_memory: _int
    is_integrated: _int
    is_multi_gpu_board: _int
    max_threads_per_multi_processor: _int
    gcnArchName: str

# Functions related to SDPA
class _SDPAParams:
    query: Tensor
    key: Tensor
    value: Tensor
    attn_mask: Optional[Tensor]
    dropout: _float
    is_causal: _bool
    def __init__(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor],
        dropout: _float,
        is_causal: _bool) -> None: ...

class _SDPBackend(Enum):
    ERROR = -1
    MATH = 0
    FLASH_ATTENTION = 1
    EFFICIENT_ATTENTION = 2
    CUDNN_ATTENTION = 3

def _can_use_flash_attention(params: _SDPAParams, debug: _bool) -> _bool: ...
def _can_use_mem_efficient_attention(params: _SDPAParams, debug: _bool) -> _bool: ...

# Defined in torch/csrc/cuda/python_comm.cpp
def _broadcast(tensor: Tensor, devices: List[_int]) -> List[Tensor]: ...
def _broadcast_out(tensor: Tensor, out_tensors: List[Tensor]) -> List[Tensor]: ...
def _broadcast_coalesced(
    tensors: List[Tensor],
    devices: List[_int],
    buffer_size: _int,
) -> List[List[Tensor]]: ...
def _scatter(
    tensor: Tensor,
    devices: List[_int],
    chunk_sizes: Optional[List[_int]],
    dim: _int,
    streams: Optional[List[Stream]],
) -> List[Tensor]: ...
def _scatter_out(
    tensor: Tensor,
    out_tensors: List[Tensor],
    dim: _int,
    streams: Optional[List[Stream]],
) -> List[Tensor]: ...
def _gather(
    tensors: List[Tensor],
    dim: _int,
    destination_index: Optional[_int],
) -> Tensor: ...
def _gather_out(tensors: List[Tensor], out_tensor: Tensor, dim: _int) -> Tensor: ...

# Defined in torch/csrc/cuda/Stream.cpp
class _CudaStreamBase(Stream):
    stream_id: _int
    device_index: _int
    device_type: _int

    device: _device
    cuda_stream: _int
    priority: _int

    def __new__(
        self,
        priority: _int = 0,
        stream_id: _int = 0,
        device_index: _int = 0,
        stream_ptr: _int = 0,
    ) -> _CudaStreamBase: ...
    def query(self) -> _bool: ...
    def synchronize(self) -> None: ...
    def priority_range(self) -> Tuple[_int, _int]: ...

# Defined in torch/csrc/cuda/Event.cpp
class _CudaEventBase:
    device: _device
    cuda_event: _int

    def __new__(
        cls,
        enable_timing: _bool = False,
        blocking: _bool = False,
        interprocess: _bool = False,
    ) -> _CudaEventBase: ...
    @classmethod
    def from_ipc_handle(cls, device: _device, ipc_handle: bytes) -> _CudaEventBase: ...
    def record(self, stream: _CudaStreamBase) -> None: ...
    def wait(self, stream: _CudaStreamBase) -> None: ...
    def query(self) -> _bool: ...
    def elapsed_time(self, other: _CudaEventBase) -> _float: ...
    def synchronize(self) -> None: ...
    def ipc_handle(self) -> bytes: ...

# Defined in torch/csrc/cuda/Graph.cpp
class _CUDAGraph:
    def capture_begin(self, pool: Optional[Tuple[_int, _int]] = ..., capture_error_mode: str = "global") -> None: ...
    def capture_end(self) -> None: ...
    def register_generator_state(self, Generator) -> None: ...
    def replay(self) -> None: ...
    def reset(self) -> None: ...
    def pool(self) -> Tuple[_int, _int]: ...
    def enable_debug_mode(self) -> None: ...
    def debug_dump(self, debug_path: str) -> None: ...

def _cuda_isCurrentStreamCapturing() -> _bool: ...
def _graph_pool_handle() -> Tuple[_int, _int]: ...

# Defined in torch/csrc/xpu/Module.cpp
def _xpu_setDevice(device: _int) -> None: ...
def _xpu_exchangeDevice(device: _int) -> _int: ...
def _xpu_maybeExchangeDevice(device: _int) -> _int: ...
def _xpu_getDevice() -> _int: ...
def _xpu_getDeviceCount() -> _int: ...
def _xpu_init() -> None: ...
def _xpu_setStream(stream_id: _int, device_index: _int, device_type: _int) -> None: ...
def _xpu_getCurrentStream(device: _int) -> Tuple: ...
def _xpu_getCurrentRawStream(device: _int) -> _int: ...
def _xpu_synchronize(device: _int) -> None: ...
def _xpu_emptyCache() -> None: ...

class _XpuDeviceProperties:
    name: str
    platform_name: str
    vendor: str
    driver_version: str
    version: str
    total_memory: _int
    max_compute_units: _int
    gpu_eu_count: _int
    gpu_subslice_count: _int
    max_work_group_size: _int
    max_num_sub_groups: _int
    sub_group_sizes: List[_int]
    has_fp16: _bool
    has_fp64: _bool
    has_atomic64: _bool
    type: str

# Defined in torch/csrc/xpu/Stream.cpp
class _XpuStreamBase(Stream):
    stream_id: _int
    device_index: _int
    device_type: _int

    device: _device
    sycl_queue: _int
    priority: _int

    def __new__(
        cls,
        priority: _int = 0,
        stream_id: _int = 0,
        device_index: _int = 0,
        device_type: _int = 0,
    ) -> _XpuStreamBase: ...
    def query(self) -> _bool: ...
    def synchronize(self) -> None: ...
    @staticmethod
    def priority_range() -> Tuple: ...

# Defined in torch/csrc/xpu/Event.cpp
class _XpuEventBase:
    device: _device
    sycl_event: _int

    def __new__(cls, enable_timing: _bool = False) -> _XpuEventBase: ...
    def record(self, stream: _XpuEventBase) -> None: ...
    def wait(self, stream: _XpuStreamBase) -> None: ...
    def query(self) -> _bool: ...
    def elapsed_time(self, other: _XpuEventBase) -> _float: ...
    def synchronize(self) -> None: ...

# Defined in torch/csrc/DataLoader.cpp
def _set_worker_signal_handlers(
    *arg: Any,
) -> None: ...  # THPModule_setWorkerSignalHandlers
def _set_worker_pids(
    key: _int,
    child_pids: Tuple[_int, ...],
) -> None: ...  # THPModule_setWorkerPIDs
def _remove_worker_pids(loader_id: _int) -> None: ...  # THPModule_removeWorkerPIDs
def _error_if_any_worker_fails() -> None: ...  # THPModule_errorIfAnyWorkerFails

# Defined in torch/csrc/jit/python/python_tracer.cpp
class TracingState:
    def push_scope(self, scope_name: str) -> None: ...
    def pop_scope(self) -> None: ...
    def current_scope(self) -> str: ...
    def set_graph(self, graph: Graph) -> None: ...
    def graph(self) -> Graph: ...

def _create_graph_by_tracing(
    func: Callable[..., Any],
    inputs: Any,
    var_name_lookup_fn: Callable[[Tensor], str],
    strict: Any,
    force_outplace: Any,
    self: Any = None,
    argument_names: List[str] = [],
) -> Tuple[Graph, Stack]: ...
def _tracer_warn_use_python(): ...
def _get_tracing_state() -> TracingState: ...

# Defined in torch/csrc/jit/python/python_ir.cpp
# Not actually defined in python_ir.cpp, not sure where they are.
class IValue: ...

Stack = List[IValue]

class JitType:
    annotation_str: str
    def isSubtypeOf(self, other: JitType) -> _bool: ...
    def with_dtype(self, dtype: _dtype) -> JitType: ...
    def with_sizes(self, sizes: List[Optional[_int]]) -> JitType: ...
    def kind(self) -> str: ...
    def scalarType(self) -> Optional[str]: ...
    def getElementType(self) -> JitType: ...
    def dtype(self) -> Optional[_dtype]: ...

class InferredType:
    def __init__(self, arg: Union[JitType, str]): ...
    def type(self) -> JitType: ...
    def success(self) -> _bool: ...
    def reason(self) -> str: ...

R = TypeVar("R", bound=JitType)

class AnyType(JitType):
    @staticmethod
    def get() -> AnyType: ...

class NoneType(JitType):
    @staticmethod
    def get() -> NoneType: ...

class BoolType(JitType):
    @staticmethod
    def get() -> BoolType: ...

class FloatType(JitType):
    @staticmethod
    def get() -> FloatType: ...

class ComplexType(JitType):
    @staticmethod
    def get() -> ComplexType: ...

class IntType(JitType):
    @staticmethod
    def get() -> IntType: ...

class SymIntType(JitType):
    @staticmethod
    def get() -> SymIntType: ...

class SymBoolType(JitType):
    @staticmethod
    def get() -> SymBoolType: ...

class NumberType(JitType):
    @staticmethod
    def get() -> NumberType: ...

class StringType(JitType):
    @staticmethod
    def get() -> StringType: ...

class DeviceObjType(JitType):
    @staticmethod
    def get() -> DeviceObjType: ...

class _GeneratorType(JitType):
    @staticmethod
    def get() -> _GeneratorType: ...

class StreamObjType(JitType):
    @staticmethod
    def get() -> StreamObjType: ...

class ListType(JitType):
    def __init__(self, a: JitType) -> None: ...
    def getElementType(self) -> JitType: ...
    @staticmethod
    def ofInts() -> ListType: ...
    @staticmethod
    def ofTensors() -> ListType: ...
    @staticmethod
    def ofFloats() -> ListType: ...
    @staticmethod
    def ofComplexDoubles() -> ListType: ...
    @staticmethod
    def ofBools() -> ListType: ...
    @staticmethod
    def ofStrings() -> ListType: ...

class DictType(JitType):
    def __init__(self, key: JitType, value: JitType) -> None: ...
    def getKeyType(self) -> JitType: ...
    def getValueType(self) -> JitType: ...

class TupleType(JitType):
    def __init__(self, a: List[Optional[JitType]]) -> None: ...
    def elements(self) -> List[JitType]: ...

class UnionType(JitType):
    def __init__(self, a: List[JitType]) -> None: ...

class ClassType(JitType):
    def __init__(self, qualified_name: str) -> None: ...

class InterfaceType(JitType):
    def __init__(self, qualified_name: str) -> None: ...
    def getMethod(self, name: str) -> Optional[FunctionSchema]: ...
    def getMethodNames(self) -> List[str]: ...

class OptionalType(JitType, Generic[R]):
    def __init__(self, a: JitType) -> None: ...
    def getElementType(self) -> JitType: ...
    @staticmethod
    def ofTensor() -> OptionalType: ...

class FutureType(JitType):
    def __init__(self, a: JitType) -> None: ...
    def getElementType(self) -> JitType: ...

class AwaitType(JitType):
    def __init__(self, a: JitType) -> None: ...
    def getElementType(self) -> JitType: ...

class RRefType(JitType):
    def __init__(self, a: JitType) -> None: ...

class EnumType(JitType):
    def __init__(
        self,
        qualified_name: str,
        value_type: JitType,
        enum_names_values: List[Any],
    ) -> None: ...

class TensorType(JitType):
    @classmethod
    def get(cls) -> TensorType: ...
    @classmethod
    def getInferred(cls) -> TensorType: ...
    def with_sizes(self, other: Optional[List[Optional[_int]]]) -> TensorType: ...
    def sizes(self) -> Optional[List[_int]]: ...
    def varyingSizes(self) -> Optional[List[Optional[_int]]]: ...
    def strides(self) -> Optional[List[_int]]: ...
    def device(self) -> Optional[_device]: ...
    def dim(self) -> _int: ...
    def dtype(self) -> Optional[_dtype]: ...
    @staticmethod
    def create_from_tensor(t: Tensor) -> TensorType: ...

# Defined in torch/csrc/jit/python/python_tree_views.cpp
class SourceRange: ...
class TreeView: ...

class Ident(TreeView):
    @property
    def name(self) -> str: ...

class ClassDef(TreeView): ...

class Def(TreeView):
    def name(self) -> Ident: ...

class Decl(TreeView): ...

# Defined in torch/csrc/distributed/rpc/init.cpp
def _rpc_init() -> _bool: ...

# Defined in torch/csrc/distributed/autograd/init.cpp
def _dist_autograd_init() -> _bool: ...

# Defined in torch/csrc/distributed/c10d/init.cpp
def _c10d_init() -> _bool: ...

# Defined in torch/csrc/distributed/rpc/testing/init.cpp
def _faulty_agent_init() -> _bool: ...
def _register_py_class_for_device(device: str, cls: Any) -> None: ...

# Defined in torch/csrc/Module.cpp
def _current_graph_task_id() -> _int: ...
def _current_autograd_node() -> _Node: ...

# Defined in torch/csrc/Exceptions.cpp
class OutOfMemoryError(RuntimeError): ...
class _DistError(RuntimeError): ...
class _DistBackendError(RuntimeError): ...
class _DistStoreError(RuntimeError): ...
class _DistNetworkError(RuntimeError): ...

# Defined in torch/csrc/profiler/init.cpp
class CapturedTraceback:
    pass
def gather_traceback(python: _bool, script: _bool, cpp: _bool) -> CapturedTraceback: ...
def symbolize_tracebacks(tracebacks: List[CapturedTraceback]) -> List[Dict[str, Any]]: ...

def _load_mobile_module_from_file(filename: str): ...
def _load_mobile_module_from_bytes(bytes_: bytes): ...
def _load_jit_module_from_file(filename: str): ...
def _load_jit_module_from_bytes(bytes_: bytes): ...
def _save_mobile_module(m: LiteScriptModule, filename: str): ...
def _save_jit_module(m: ScriptModule, filename: str, extra_files: Dict[str, Any]): ...
def _save_mobile_module_to_bytes(m: LiteScriptModule) -> bytes: ...
def _save_jit_module_to_bytes(m: ScriptModule,  extra_files: Dict[str, Any]) -> bytes: ...
def _get_module_info_from_flatbuffer(data: bytes): ...
def _jit_resolve_packet(op_name: str, *args, **kwargs) -> str: ...
def _swap_tensor_impl(t1: Tensor, t2: Tensor): ...
def _save_pickle(obj: Any) -> bytes: ...

# Defined in torch/csrc/jit/runtime/static/init.cpp
def _jit_to_static_module(graph_or_module: Union[Graph,ScriptModule]) -> Any: ...
def _fuse_to_static_module(graph_or_module: Union[Graph,ScriptModule], min_size: _int) -> Any: ...
