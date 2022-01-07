import warnings
from typing import List, NamedTuple, Any, Optional, Sequence, Dict

import numpy
import tensorrt as trt
import torch
import torch.fx
from torch.fx.node import _get_qualified_name
from torch.fx.passes.shape_prop import TensorMetadata

from .converter_registry import CONVERTERS
from .input_tensor_spec import InputTensorSpec
from .utils import torch_dtype_to_trt, get_dynamic_dims


class TRTInterpreterResult(NamedTuple):
    engine: Any
    input_names: Sequence[str]
    output_names: Sequence[str]
    serialized_cache: bytearray


class TRTInterpreter(torch.fx.Interpreter):
    def __init__(
        self,
        module: torch.fx.GraphModule,
        input_specs: List[InputTensorSpec],
        explicit_batch_dimension: bool = False,
        explicit_precision: bool = False,
        logger_level=None,
    ):
        super().__init__(module)

        self.logger = trt.Logger(logger_level or trt.Logger.WARNING)
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
        self._itensor_to_tensor_meta: Dict[trt.tensorrt.ITensor, TensorMetadata] = dict()

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
        sparse_weights=False,
        force_fp32_output=False,
        strict_type_constraints=False,
        algorithm_selector=None,
        timing_cache=None,
        profiling_verbosity=None,
    ) -> TRTInterpreterResult:
        # For float outputs, we set their dtype to fp16 only if fp16_mode=True and
        # force_fp32_output=False.
        self.output_fp16 = not force_fp32_output and fp16_mode

        if int8_mode and not self.builder.platform_has_fast_int8:
            warnings.warn("Current platform doesn't support fast native int8!")

        if fp16_mode and not self.builder.platform_has_fast_fp16:
            warnings.warn("Current platform doesn't support fast native fp16!")

        self.input_specs_iter = 0
        super().run()

        self.builder.max_batch_size = max_batch_size
        builder_config = self.builder.create_builder_config()
        builder_config.max_workspace_size = max_workspace_size

        cache = None
        if timing_cache:
            cache_file = numpy.array(timing_cache)
            cache = builder_config.create_timing_cache(cache_file.tobytes())
        else:
            cache = builder_config.create_timing_cache(b"")
        builder_config.set_timing_cache(cache, False)

        if trt.__version__ >= "8.2":
            builder_config.profiling_verbosity = profiling_verbosity \
                if profiling_verbosity else \
                trt.ProfilingVerbosity.LAYER_NAMES_ONLY
        if fp16_mode:
            builder_config.set_flag(trt.BuilderFlag.FP16)

        if int8_mode:
            builder_config.set_flag(trt.BuilderFlag.INT8)

        if sparse_weights:
            assert fp16_mode or int8_mode, "We can only enable sparsity in fp16 or int8 mode."
            builder_config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

        if strict_type_constraints:
            builder_config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        if self.optimization_profiles:
            for optimization_profile in self.optimization_profiles:
                builder_config.add_optimization_profile(optimization_profile)

        if algorithm_selector:
            builder_config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
            builder_config.algorithm_selector = algorithm_selector

        engine = self.builder.build_engine(self.network, builder_config)
        assert engine

        serialized_cache = bytearray(cache.serialize()) \
            if builder_config.get_timing_cache() else bytearray()

        return TRTInterpreterResult(engine, self._input_names, self._output_names, serialized_cache)

    def run_node(self, n):
        self._cur_node_name = str(n)
        # add "_itensor_to_tensor_meta"
        kwargs = dict(n.kwargs)
        kwargs["_itensor_to_tensor_meta"] = self._itensor_to_tensor_meta
        n.kwargs = kwargs

        # run the node
        trt_node = super().run_node(n)

        # remove "_itensor_to_tensor_meta"
        kwargs = dict(n.kwargs)
        del kwargs["_itensor_to_tensor_meta"]
        n.kwargs = kwargs

        if isinstance(trt_node, trt.tensorrt.ITensor):
            self._itensor_to_tensor_meta[trt_node] = n.meta.get("tensor_meta")

        return trt_node

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

        assert self._cur_node_name is not None
        return converter(self.network, submod, args, kwargs, self._cur_node_name)

    def call_function(self, target, args, kwargs):
        converter = CONVERTERS.get(target)

        if not converter:
            raise RuntimeError(
                f"Conversion of function {torch.typename(target)} not currently supported!"
            )

        assert self._cur_node_name is not None
        return converter(self.network, target, args, kwargs, self._cur_node_name)

    def call_method(self, target, args, kwargs):
        assert isinstance(target, str)
        converter = CONVERTERS.get(target)

        if not converter:
            raise RuntimeError(
                f"Conversion of method {target} not currently supported!"
            )

        assert self._cur_node_name is not None
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
            if self.output_fp16 and output.dtype == trt.float32:
                output.dtype = trt.float16
            self._output_names.append(name)
