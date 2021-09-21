import warnings
from typing import List, NamedTuple, Iterable, Any, Optional, Tuple, Sequence

import tensorrt as trt
import torch
import torch.fx
from torch.fx.node import _get_qualified_name


TRTInterpreterResult = Tuple[Any, Sequence[str], Sequence[str]]


# Borrowed from torch2trt
def torch_dtype_to_trt(dtype):
    if trt.__version__ >= "7.0" and dtype == torch.bool:
        return trt.bool
    elif dtype == torch.int8:
        return trt.int8
    elif dtype == torch.int32:
        return trt.int32
    elif dtype == torch.float16:
        return trt.float16
    elif dtype == torch.float32:
        return trt.float32
    else:
        raise TypeError("%s is not supported by tensorrt" % dtype)


def torch_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return torch.int8
    elif trt.__version__ >= "7.0" and dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError("%s is not supported by torch" % dtype)


class TRTModule(torch.nn.Module):
    def __init__(self, engine=None, input_names=None, output_names=None):
        super(TRTModule, self).__init__()
        self._register_state_dict_hook(TRTModule._on_state_dict)
        self.engine = engine
        self.input_names = input_names
        self.output_names = output_names
        self.initialized = False

        if engine:
            self._initialize()

    def _initialize(self):
        self.initialized = True
        self.context = self.engine.create_execution_context()

        # Indices of inputs/outputs in the trt engine bindings, in the order
        # as they are in the original PyTorch model.
        self.input_binding_indices_in_order: Sequence[int] = [
            self.engine.get_binding_index(name) for name in self.input_names
        ]
        self.output_binding_indices_in_order: Sequence[int] = [
            self.engine.get_binding_index(name) for name in self.output_names
        ]

        self.input_dtypes: Sequence[torch.dtype] = [
            torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            for idx in self.input_binding_indices_in_order
        ]
        self.input_shapes: Sequence[Sequence[int]] = [
            tuple(self.engine.get_binding_shape(idx))
            for idx in self.input_binding_indices_in_order
        ]
        self.output_dtypes: Sequence[torch.dtype] = [
            torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            for idx in self.output_binding_indices_in_order
        ]

    def _check_initialized(self):
        if not self.initialized:
            raise RuntimeError("TRTModule is not initialized.")

    def _on_state_dict(self, state_dict, prefix, local_metadata):
        self._check_initialized()
        state_dict[prefix + "engine"] = bytearray(self.engine.serialize())
        state_dict[prefix + "input_names"] = self.input_names
        state_dict[prefix + "output_names"] = self.output_names

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        engine_bytes = state_dict[prefix + "engine"]

        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(engine_bytes)

        self.input_names = state_dict[prefix + "input_names"]
        self.output_names = state_dict[prefix + "output_names"]
        self._initialize()

    def forward(self, *inputs):
        with torch.autograd.profiler.record_function("TRTModule:Forward"):
            self._check_initialized()

            with torch.autograd.profiler.record_function("TRTModule:ProcessInputs"):
                assert len(inputs) == len(
                    self.input_names
                ), f"Wrong number of inputs, expect {len(self.input_names)} get {len(inputs)}."

                # This is only used when the trt engine is using implicit batch dim.
                batch_size = inputs[0].shape[0]
                contiguous_inputs: List[torch.Tensor] = [i.contiguous() for i in inputs]
                bindings: List[Any] = [None] * (
                    len(self.input_names) + len(self.output_names)
                )

                for i, input_name in enumerate(self.input_names):
                    assert inputs[
                        i
                    ].is_cuda, f"{i}th input({input_name}) is not on cuda device."
                    assert (
                        inputs[i].dtype == self.input_dtypes[i]
                    ), f"Dtype mismatch for {i}th input({input_name}). Expect {self.input_dtypes[i]}, got {inputs[i].dtype}."

                    idx = self.input_binding_indices_in_order[i]
                    bindings[idx] = contiguous_inputs[i].data_ptr()

                    if not self.engine.has_implicit_batch_dimension:
                        self.context.set_binding_shape(
                            idx, tuple(contiguous_inputs[i].shape)
                        )
                    else:
                        assert (
                            inputs[i].size()[1:] == self.input_shapes[i]
                        ), f"Shape mismatch for {i}th input({input_name}). " \
                           f"Expect {self.input_shapes[i]}, got {inputs[i].size()[1:]}."

            with torch.autograd.profiler.record_function("TRTModule:ProcessOutputs"):
                # create output tensors
                outputs: List[torch.Tensor] = []
                for i, idx in enumerate(self.output_binding_indices_in_order):
                    if self.engine.has_implicit_batch_dimension:
                        shape = (batch_size,) + tuple(
                            self.engine.get_binding_shape(idx)
                        )
                    else:
                        shape = tuple(self.context.get_binding_shape(idx))

                    output = torch.empty(  # type: ignore[call-overload]
                        size=shape,
                        dtype=self.output_dtypes[i],
                        device=torch.cuda.current_device(),
                    )
                    outputs.append(output)
                    bindings[idx] = output.data_ptr()

            with torch.autograd.profiler.record_function("TRTModule:TensorRTRuntime"):
                if self.engine.has_implicit_batch_dimension:
                    self.context.execute_async(
                        batch_size, bindings, torch.cuda.current_stream().cuda_stream
                    )
                else:
                    self.context.execute_async_v2(
                        bindings, torch.cuda.current_stream().cuda_stream
                    )

            if len(outputs) == 1:
                return outputs[0]

            return tuple(outputs)

    def enable_profiling(self):
        """
        Enable TensorRT profiling. After calling this function, TensorRT will report
        time spent on each layer in stdout for each forward run.
        """
        self._check_initialized()

        if not self.context.profiler:
            self.context.profiler = trt.Profiler()


CONVERTERS = {}


def tensorrt_converter(key):
    def register_converter(converter):
        CONVERTERS[key] = converter
        return converter

    return register_converter


class InputTensorSpec(NamedTuple):
    """
    This class contains the information of a input tensor.

    shape: shape of the tensor.

    dtype: dtyep of the tensor.

    device: device of the tensor. This is only used to generate inputs to the given model
        in order to run shape prop. For TensorRT engine, inputs have to be on cuda device.

    shape_ranges: If dynamic shape is needed (shape has dimensions of -1), then this field
        has to be provided (default is empty list). Every shape_range is a tuple of three
        tuples ((min_input_shape), (optimized_input_shape), (max_input_shape)). Each shape_range
        is used to populate a TensorRT optimization profile.
        e.g. If the input shape varies from (1, 224) to (100, 224) and we want to optimize
        for (25, 224) because it's the most common input shape, then we set shape_ranges to
        ((1, 224), (25, 225), (100, 224)).

    has_batch_dim: Whether the shape includes batch dimension. Batch dimension has to be provided
        if the engine want to run with dynamic shape.
    """

    shape: torch.Size
    dtype: torch.dtype
    device: torch.device = torch.device("cpu")
    shape_ranges: List[Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]] = []
    has_batch_dim: bool = True

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor):
        return cls(tensor.shape, tensor.dtype, tensor.device)

    @classmethod
    def from_tensors(cls, tensors: Iterable[torch.Tensor]):
        return [cls.from_tensor(t) for t in tensors]


def get_dynamic_dims(shape):
    dynamic_dims = []

    for i, s in enumerate(shape):
        if s == -1:
            dynamic_dims.append(i)

    return dynamic_dims


def create_inputs_from_specs(input_specs):
    inputs = []

    for shape, dtype, device, shape_ranges, has_batch_dim in input_specs:
        if len(get_dynamic_dims(shape)):
            shape = shape_ranges[0][1]
        elif not has_batch_dim:
            shape = (1,) + tuple(shape)

        inputs.append(torch.randn(shape).to(dtype=dtype, device=device))

    return inputs


class TRTInterpreter(torch.fx.Interpreter):
    def __init__(
        self,
        module: torch.fx.GraphModule,
        input_specs: List[InputTensorSpec],
        explicit_batch_dimension: bool = False,
        explicit_precision: bool = False,
        logger_level=trt.Logger.WARNING,
    ):
        super().__init__(module)

        self.logger = trt.Logger(logger_level)
        self.builder = trt.Builder(self.logger)

        flag = 0
        if explicit_batch_dimension:
            EXPLICIT_BATCH = 1 << (int)(
                trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH
            )
            flag |= EXPLICIT_BATCH

        if explicit_precision:
            EXPLICIT_PRECISION = 1 << (int)(
                trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION
            )
            flag |= EXPLICIT_PRECISION
        self.network = self.builder.create_network(flag)

        missing_ops = self.validate_conversion()
        if missing_ops:
            warnings.warn(
                "Interpretation will fail due to missing operations \n"
                + "\n".join(f"{i}" for i in missing_ops)
            )

        self.optimization_profiles: Optional[List] = None
        self.input_specs = input_specs
        self.input_specs_iter = 0
        self.validate_input_specs()
        self._cur_node_name: Optional[str] = None
        self._input_names: List[str] = []
        self._output_names: List[str] = []

    def validate_input_specs(self):
        for shape, dtpe, _, shape_ranges, has_batch_dim in self.input_specs:
            if not self.network.has_implicit_batch_dimension:
                assert (
                    has_batch_dim
                ), "It's required to specify batch dimension when it's explicit in TensorRT network."

            dynamic_dims = get_dynamic_dims(shape)
            if len(dynamic_dims):
                assert not self.network.has_implicit_batch_dimension, (
                    "Can't have dynamic dim when "
                    f"batch dim is implicit, got {shape}."
                )
                assert len(
                    shape_ranges
                ), "shape_ranges must be provided when shape has dynamic dim."

                if self.optimization_profiles:
                    assert len(shape_ranges) == len(self.optimization_profiles), (
                        "Number of optimization "
                        f"profiles {len(self.optimization_profiles)} doesn't match with the number of shape_range"
                        f" {len(shape_ranges)} provided."
                    )
                else:
                    self.optimization_profiles = [
                        self.builder.create_optimization_profile()
                        for _ in range(len(shape_ranges))
                    ]

                for shape_range in shape_ranges:
                    assert (
                        len(shape_range) == 3
                    ), f"Expect three elements in shape_range, got {len(shape_range)}"
                    assert all(len(s) == len(shape) for s in shape_range), (
                        "Expect elements in shape_range"
                        f" {shape_range} have the same number of dimension as the provided shape {len(shape)}"
                    )

                    for i in range(len(shape)):
                        if i in dynamic_dims:
                            assert all(
                                shape_range[j][i] <= shape_range[j + 1][i]
                                for j in range(2)
                            ), (
                                "Expect dynamic dim"
                                f" {i} to have incremental value for shapes in shape_range {shape_range}."
                            )
                        else:
                            assert all(s[i] == shape[i] for s in shape_range), (
                                f"Expect non dynamic dim {i} to be the same"
                                f" for all shapes in shape_range {shape_range}."
                            )
            else:
                assert (
                    len(shape_ranges) == 0
                ), "shape_ranges are provided for input that doesn't have dynamic dim."

    def validate_conversion(self):
        missing_converter = set()

        for node in self.module.graph.nodes:
            if node.op == "call_function" and not CONVERTERS.get(node.target):
                missing_converter.add(f"{node.op} {_get_qualified_name(node.target)}")
            elif node.op == "call_method" and not CONVERTERS.get(node.target):
                missing_converter.add(f"{node.op} torch.Tensor.{node.target}")
            elif node.op == "call_module":
                submod = self.fetch_attr(node.target)
                submod_type = getattr(submod, "_base_class_origin", type(submod))
                if not CONVERTERS.get(submod_type):
                    missing_converter.add(f"{node.op} {torch.typename(submod_type)}")

        return missing_converter

    def run(
        self,
        max_batch_size=64,
        max_workspace_size=1 << 25,
        fp16_mode=True,
        int8_mode=False,
        strict_type_constraints=True,
    ) -> TRTInterpreterResult:
        # TODO hack, should check contents of args and remove fp16_mode probably
        self.fp16_mode = fp16_mode

        if int8_mode and not self.builder.platform_has_fast_int8:
            warnings.warn("Current platform doesn't support fast native int8!")

        if fp16_mode and not self.builder.platform_has_fast_fp16:
            warnings.warn("Current platform doesn't support fast native fp16!")

        self.input_specs_iter = 0
        super().run()

        self.builder.max_batch_size = max_batch_size
        builder_config = self.builder.create_builder_config()
        builder_config.max_workspace_size = max_workspace_size
        if fp16_mode:
            builder_config.set_flag(trt.BuilderFlag.FP16)

        if int8_mode:
            builder_config.set_flag(trt.BuilderFlag.INT8)

        if strict_type_constraints:
            builder_config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        if self.optimization_profiles:
            for optimization_profile in self.optimization_profiles:
                builder_config.add_optimization_profile(optimization_profile)

        engine = self.builder.build_engine(self.network, builder_config)
        assert engine
        return engine, self._input_names, self._output_names

    def run_node(self, n):
        self._cur_node_name = str(n)
        return super().run_node(n)

    def placeholder(self, target, args, kwargs):
        self._input_names.append(target)
        shape, dtype, _, shape_ranges, has_batch_dim = self.input_specs[
            self.input_specs_iter
        ]
        self.input_specs_iter += 1

        if self.network.has_implicit_batch_dimension:
            if has_batch_dim:
                shape = shape[1:]
        else:
            for i, shape_range in enumerate(shape_ranges):
                assert self.optimization_profiles
                self.optimization_profiles[i].set_shape(target, *shape_range)

        return self.network.add_input(
            name=target, shape=tuple(shape), dtype=torch_dtype_to_trt(dtype)
        )

    def call_module(self, target, args, kwargs):
        assert isinstance(target, str)
        submod = self.fetch_attr(target)
        submod_type = getattr(submod, "_base_class_origin", type(submod))
        converter = CONVERTERS.get(submod_type)

        if not converter:
            raise RuntimeError(
                f"Conversion of module of type {submod_type} not currently supported!"
            )

        return converter(self.network, submod, args, kwargs, self._cur_node_name)

    def call_function(self, target, args, kwargs):
        converter = CONVERTERS.get(target)

        if not converter:
            raise RuntimeError(
                f"Conversion of function {torch.typename(target)} not currently supported!"
            )

        return converter(self.network, target, args, kwargs, self._cur_node_name)

    def call_method(self, target, args, kwargs):
        assert isinstance(target, str)
        converter = CONVERTERS.get(target)

        if not converter:
            raise RuntimeError(
                f"Conversion of method {target} not currently supported!"
            )

        return converter(self.network, target, args, kwargs, self._cur_node_name)

    def output(self, target, args, kwargs):
        assert len(args) == 1
        outputs = args[0] if isinstance(args[0], tuple) else (args[0],)

        if not all(isinstance(output, trt.tensorrt.ITensor) for output in outputs):
            raise RuntimeError("TensorRT requires all outputs to be Tensor!")

        for i, output in enumerate(outputs):
            name = f"output{i}"
            output.name = name
            self.network.mark_output(output)
            if self.fp16_mode and output.dtype == trt.float32:
                output.dtype = trt.float16
            self._output_names.append(name)
