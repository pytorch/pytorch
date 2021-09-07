from typing import List, Any
import torch


class TestTRTModule(torch.nn.Module):
    def __init__(self, trt, engine=None, input_names=None, output_names=None, fp16_output=False):
        super(TestTRTModule, self).__init__()
        self.trt = trt
        self._register_state_dict_hook(TestTRTModule._on_state_dict)
        self.engine = engine
        if self.engine is not None:
            self.context = self.engine.create_execution_context()
        self.input_names = input_names
        self.output_names = output_names

        # Indicate output is in fp16
        self.fp16_output = fp16_output

    def _on_state_dict(self, state_dict, prefix, local_metadata):
        state_dict[prefix + "engine"] = bytearray(self.engine.serialize())
        state_dict[prefix + "input_names"] = self.input_names
        state_dict[prefix + "output_names"] = self.output_names
        state_dict[prefix + "fp16_output"] = self.fp16_output

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

        with self.trt.Logger() as logger, self.trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(engine_bytes)
            self.context = self.engine.create_execution_context()

        self.input_names = state_dict[prefix + "input_names"]
        self.output_names = state_dict[prefix + "output_names"]

    def forward(self, *inputs):
        batch_size = inputs[0].shape[0]
        contiguous_inputs: List[torch.Tensor] = [i.contiguous() for i in inputs]
        bindings: List[Any] = [None] * (len(self.input_names) + len(self.output_names))

        # create output tensors
        outputs: List[torch.Tensor] = []
        for _, output_name in enumerate(self.output_names):
            idx: int = self.engine.get_binding_index(output_name)
            shape = (batch_size,) + tuple(self.engine.get_binding_shape(idx))
            output = torch.empty(size=shape, dtype=torch.float32, device="cuda")
            outputs.append(output)
            bindings[idx] = output.data_ptr()

        for i, input_name in enumerate(self.input_names):
            idx = self.engine.get_binding_index(input_name)
            bindings[idx] = contiguous_inputs[i].data_ptr()

        self.context.execute_async(
            batch_size, bindings, torch.cuda.current_stream().cuda_stream
        )

        if len(outputs) == 1:
            return outputs[0]

        return tuple(outputs)

def tensorrt_example(tensor):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()

    x = network.add_input("x", shape=(1, 2, 3), dtype=trt.float32)
    layer = network.add_elementwise(x, x, trt.ElementWiseOperation.SUM)
    layer.name = "add"
    output = layer.get_output(0)
    output.name = "output"
    network.mark_output(output)
    output.dtype = trt.float32

    builder.max_batch_size = 1024
    builder_config = builder.create_builder_config()
    builder_config.max_workspace_size = 1 << 25
    engine = builder.build_engine(network, builder_config)

    mod = TestTRTModule(trt, engine, ["x"], ["output"])
    return mod(tensor)
