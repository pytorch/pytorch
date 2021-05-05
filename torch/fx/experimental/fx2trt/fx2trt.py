from typing import List, NamedTuple, Iterable, Any, Optional

import torch
import torch.fx
import tensorrt as trt
import copy
from torch.fx.experimental.normalize import NormalizeArgs


# Borrowed from torch2trt
def torch_dtype_to_trt(dtype):
    if trt.__version__ >= '7.0' and dtype == torch.bool:
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
    elif trt.__version__ >= '7.0' and dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError("%s is not supported by torch" % dtype)

def torch_device_to_trt(device):
    if device.type == torch.device("cuda").type:
        return trt.TensorLocation.DEVICE
    elif device.type == torch.device("cpu").type:
        return trt.TensorLocation.HOST
    else:
        return TypeError("%s is not supported by tensorrt" % device)


def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device("cuda")
    elif device == trt.TensorLocation.HOST:
        return torch.device("cpu")
    else:
        return TypeError("%s is not supported by torch" % device)


class TRTModule(torch.nn.Module):
    def __init__(self, engine=None, input_names=None, output_names=None):
        super(TRTModule, self).__init__()
        self._register_state_dict_hook(TRTModule._on_state_dict)
        self.engine = engine
        if self.engine is not None:
            self.context = self.engine.create_execution_context()
        self.input_names = input_names
        self.output_names = output_names

    def _on_state_dict(self, state_dict, prefix, local_metadata):
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
            self.context = self.engine.create_execution_context()

        self.input_names = state_dict[prefix + "input_names"]
        self.output_names = state_dict[prefix + "output_names"]

    def forward(self, *inputs):
        batch_size = inputs[0].shape[0]
        bindings: List[Any] = [None] * (len(self.input_names) + len(self.output_names))

        # create output tensors
        outputs: List[torch.Tensor] = []
        for i, output_name in enumerate(self.output_names):
            idx: int = self.engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            shape = (batch_size,) + tuple(self.engine.get_binding_shape(idx))
            device = torch_device_from_trt(self.engine.get_location(idx))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs.append(output)
            bindings[idx] = output.data_ptr()

        for i, input_name in enumerate(self.input_names):
            idx = self.engine.get_binding_index(input_name)
            bindings[idx] = inputs[i].contiguous().data_ptr()

        self.context.execute_async(
            batch_size, bindings, torch.cuda.current_stream().cuda_stream
        )

        if len(outputs) == 1:
            return outputs[0]

        return tuple(outputs)

    def enable_profiling(self):
        if not self.context.profiler:
            self.context.profiler = trt.Profiler()


CONVERTERS = {}


def tensorrt_converter(key):
    def register_converter(converter):
        CONVERTERS[key] = converter
        return converter
    return register_converter


class InputTensorSpec(NamedTuple):
    shape : torch.Size
    dtype : torch.dtype

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor):
        return cls(tensor.shape, tensor.dtype)

    @classmethod
    def from_tensors(cls, tensors: Iterable[torch.Tensor]):
        return [cls.from_tensor(t) for t in tensors]


class TRTInterpreter(torch.fx.Interpreter):
    def __init__(self, module : torch.fx.GraphModule, input_shapes : List[InputTensorSpec], logger_level=trt.Logger.WARNING):
        # Preprocess the model
        module = copy.copy(module)
        module = module.cpu()
        module = NormalizeArgs(module).transform()
        super().__init__(module)

        self.logger = trt.Logger(logger_level)
        self.builder = trt.Builder(self.logger)
        self.network = self.builder.create_network()

        self.input_shape_itr = iter(input_shapes)

        self._cur_node_name: Optional[str] = None

        self._input_names: List[str] = []
        self._output_names: List[str] = []

    def run(
        self,
        *args,
        max_batch_size=10,
        max_workspace_size=1 << 25,
        fp16_mode=False,
        int8_mode=False,
        strict_type_constraints=False
    ):
        super().run(*args)

        self.builder.max_batch_size = max_batch_size
        self.builder.max_workspace_size = max_workspace_size
        self.builder.strict_type_constraints = strict_type_constraints
        self.builder.fp16_mode = fp16_mode
        self.builder.int8_mode = int8_mode

        return self.builder.build_cuda_engine(self.network), self._input_names, self._output_names

    def run_node(self, n):
        self._cur_node_name = str(n)

        try:
            return super().run_node(n)
        finally:
            self._cur_node_metadata = None

    def placeholder(self, target, args, kwargs):
        shape, dtype = next(self.input_shape_itr)
        self._input_names.append(target)
        return self.network.add_input(name=target, shape=tuple(shape[1:]), dtype=torch_dtype_to_trt(dtype))

    def call_module(self, target, args, kwargs):
        assert isinstance(target, str)
        submod = self.fetch_attr(target)

        converter = CONVERTERS.get(type(submod))

        if not converter:
            raise RuntimeError(f'Conversion of module of type {type(submod)} not currently supported!')

        return converter(self.network, submod, args, kwargs, self._cur_node_name)

    def call_function(self, target, args, kwargs):
        converter = CONVERTERS.get(target)

        if not converter:
            raise RuntimeError(f'Conversion of function {torch.typename(target)} not currently supported!')

        return converter(self.network, target, args, kwargs, self._cur_node_name)

    def call_method(self, target, args, kwargs):
        assert isinstance(target, str)

        converter = CONVERTERS.get(target)

        if not converter:
            raise RuntimeError(f'Conversion of method {target} not currently supported!')

        return converter(self.network, target, args, kwargs, self._cur_node_name)

    def output(self, target, args, kwargs):
        assert len(args) == 1
        outputs = args[0] if isinstance(args[0], tuple) else (args[0],)
        if not all(isinstance(output, trt.tensorrt.ITensor) for output in outputs):
            raise RuntimeError('TensorRT requires all outputs to be Tensor!')

        for i, output in enumerate(outputs):
            # TODO: set location and dtype?
            name = f'output{i}'
            output.name = name
            self.network.mark_output(output)
            self._output_names.append(name)
