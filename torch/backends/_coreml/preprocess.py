import hashlib
import json
from dataclasses import dataclass, astuple, field
from typing import Dict, Tuple, List

import coremltools as ct  # type: ignore[import]
import torch
from coremltools.converters.mil.input_types import TensorType  # type: ignore[import]
from coremltools.converters.mil.mil import types  # type: ignore[import]

CT_METADATA_VERSION = "com.github.apple.coremltools.version"
CT_METADATA_SOURCE = "com.github.apple.coremltools.source"


class ScalarType:
    Float = 0
    Double = 1
    Int = 2
    Long = 3
    Undefined = 4

# Supported Tensor types in coremltools:
# https://github.com/apple/coremltools/blob/main/coremltools/converters/mil/frontend/torch/converter.py#L28
torch_to_mil_types = {
    ScalarType.Float: types.fp32,
    ScalarType.Double: types.fp64,
    ScalarType.Int: types.int32,
    ScalarType.Long: types.int64,
}


class CoreMLComputeUnit:
    CPU = "cpuOnly"
    CPUAndGPU = "cpuAndGPU"
    ALL = "all"


@dataclass
class _TensorSpec:
    shape: List[int] = field(default_factory=List[int])
    dtype: int = ScalarType.Float


def TensorSpec(*args, **kwargs):
    """
    TensorSpec specifies the tensor information. The default dtype is float32
    Example:
    ts = TensorSpec(
        shape = [1, 3, 224, 224],
        dtype = ScalarType.Float
    )
    """
    return astuple(_TensorSpec(*args, **kwargs))


@dataclass
class _CompileSpec:
    inputs: Tuple[_TensorSpec] = ()  # type: ignore[assignment]
    outputs: Tuple[_TensorSpec] = ()  # type: ignore[assignment]
    backend: str = CoreMLComputeUnit.CPU
    allow_low_precision: bool = True


def CompileSpec(*args, **kwargs):
    """
    CompileSpec specifies the model information.
    Example:
    cs = CompileSpec(
            inputs=(
                TensorSpec(
                    shape=[1, 3, 224, 224],
                ),
            ),
            outputs=(
                TensorSpec(
                    shape=[1, 1000],
                ),
            ),
            backend=CoreMLComputeUnit.CPU,
            allow_low_precision=True,
    ),
    """
    return astuple(_CompileSpec(*args, **kwargs))


def _convert_to_mil_type(spec: _TensorSpec, name: str):
    ml_type = TensorType(shape=spec.shape, dtype=torch_to_mil_types[spec.dtype])
    ml_type.name = name
    return ml_type


def preprocess(script_module: torch._C.ScriptObject, compile_spec: Dict[str, Tuple]):
    spec = compile_spec["forward"]
    forward_spec = _CompileSpec(*spec)
    mil_inputs = []
    inputs = []
    for index, input_spec in enumerate(forward_spec.inputs):
        input_spec = _TensorSpec(*input_spec)  # type: ignore[misc]
        name = "input_" + str(index)
        inputs.append([name, str(input_spec.dtype), str(input_spec.shape)])
        ml_type = _convert_to_mil_type(input_spec, name)
        mil_inputs.append(ml_type)
    model = torch.jit.RecursiveScriptModule._construct(script_module, lambda x: None)
    mlmodel = ct.convert(model, inputs=mil_inputs)
    spec = mlmodel.get_spec()
    output_specs = forward_spec.outputs
    assert len(spec.description.output) == len(output_specs)  # type: ignore[attr-defined]
    outputs = []
    for index, output_spec in enumerate(output_specs):
        output_spec = _TensorSpec(*output_spec)  # type: ignore[misc]
        name = spec.description.output[index].name  # type: ignore[attr-defined]
        outputs.append([name, str(output_spec.dtype), str(output_spec.shape)])
    mlmodel = ct.models.model.MLModel(spec)
    config = {
        "spec_ver": str(spec.specificationVersion),  # type: ignore[attr-defined]
        "backend": forward_spec.backend,
        "allow_low_precision": str(forward_spec.allow_low_precision),
    }
    metadata = {
        "coremltool_ver": mlmodel.user_defined_metadata[CT_METADATA_VERSION],
        "torch_ver": mlmodel.user_defined_metadata[CT_METADATA_SOURCE],
    }
    coreml_compile_spec = {
        "inputs": inputs,
        "outputs": outputs,
        "config": config,
        "metadata": metadata,
    }
    mlmodel = spec.SerializeToString()  # type: ignore[attr-defined]

    return {
        "model": mlmodel,
        "hash": str(hashlib.sha256(mlmodel).hexdigest()),
        "extra": json.dumps(coreml_compile_spec),
    }
