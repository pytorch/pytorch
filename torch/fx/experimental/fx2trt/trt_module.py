from typing import List, Any, Sequence

import tensorrt as trt
import torch

from .utils import torch_dtype_from_trt


class TRTModule(torch.nn.Module):
    def __init__(self, engine=None, input_names=None, output_names=None, cuda_graph_batch_size=-1):
        super(TRTModule, self).__init__()
        self._register_state_dict_hook(TRTModule._on_state_dict)
        self.engine = engine
        self.input_names = input_names
        self.output_names = output_names
        self.cuda_graph_batch_size = cuda_graph_batch_size
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
        primary_input_outputs = set()
        primary_input_outputs.update(self.input_binding_indices_in_order)
        primary_input_outputs.update(self.output_binding_indices_in_order)
        self.hidden_output_binding_indices_in_order: Sequence[int] = []
        self.hidden_output_names: Sequence[str] = []
        for i in range(self.engine.num_bindings):
            if i not in primary_input_outputs:
                self.hidden_output_binding_indices_in_order.append(i)
                self.hidden_output_names.append(self.engine.get_binding_name(i))

        assert self.engine.num_bindings == (len(self.input_names) + len(self.output_names) + len(self.hidden_output_names))

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
        self.output_shapes = [
            tuple(self.engine.get_binding_shape(idx)) if self.engine.has_implicit_batch_dimension else tuple()
            for idx in self.output_binding_indices_in_order
        ]
        self.hidden_output_dtypes: Sequence[torch.dtype] = [
            torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            for idx in self.hidden_output_binding_indices_in_order
        ]
        self.hidden_output_shapes = [
            tuple(self.engine.get_binding_shape(idx)) if self.engine.has_implicit_batch_dimension else tuple()
            for idx in self.hidden_output_binding_indices_in_order
        ]


    def _check_initialized(self):
        if not self.initialized:
            raise RuntimeError("TRTModule is not initialized.")

    def _on_state_dict(self, state_dict, prefix, local_metadata):
        self._check_initialized()
        state_dict[prefix + "engine"] = bytearray(self.engine.serialize())
        state_dict[prefix + "input_names"] = self.input_names
        state_dict[prefix + "output_names"] = self.output_names
        state_dict[prefix + "cuda_graph_batch_size"] = self.cuda_graph_batch_size

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

        logger = trt.Logger()
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)

        self.input_names = state_dict[prefix + "input_names"]
        self.output_names = state_dict[prefix + "output_names"]
        self._initialize()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["engine"] = bytearray(self.engine.serialize())
        state.pop('context', None)
        return state

    def __setstate__(self, state):
        logger = trt.Logger()
        runtime = trt.Runtime(logger)
        state["engine"] = runtime.deserialize_cuda_engine(state["engine"])
        self.__dict__.update(state)
        if self.engine:
            self.context = self.engine.create_execution_context()

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
                    len(self.input_names) + len(self.output_names) + len(self.hidden_output_names)
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
                        shape = (batch_size,) + self.output_shapes[i]
                    else:
                        shape = tuple(self.context.get_binding_shape(idx))

                    output = torch.empty(  # type: ignore[call-overload]
                        size=shape,
                        dtype=self.output_dtypes[i],
                        device=torch.cuda.current_device(),
                    )
                    outputs.append(output)
                    bindings[idx] = output.data_ptr()

                for i, idx in enumerate(self.hidden_output_binding_indices_in_order):
                    if self.engine.has_implicit_batch_dimension:
                        shape = (batch_size,) + self.hidden_output_shapes[i]
                    else:
                        shape = tuple(self.context.get_binding_shape(idx))

                    output = torch.empty(  # type: ignore[call-overload]
                        size=shape,
                        dtype=self.hidden_output_dtypes[i],
                        device=torch.cuda.current_device(),
                    )
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

    def disable_profiling(self):
        """
        Disable TensorRT profiling.
        """
        self._check_initialized()

        torch.cuda.synchronize()
        del self.context
        self.context = self.engine.create_execution_context()
