# mypy: allow-untyped-defs
import hashlib
import json

import coremltools as ct  # type: ignore[import]
from coremltools.converters.mil.input_types import TensorType  # type: ignore[import]
from coremltools.converters.mil.mil import types  # type: ignore[import]
from coremltools.models.neural_network import quantization_utils  # type: ignore[import]

import torch


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


class CoreMLQuantizationMode:
    LINEAR = "linear"
    LINEAR_SYMMETRIC = "linear_symmetric"
    NONE = "none"


def TensorSpec(shape, dtype=ScalarType.Float):
    return (shape, dtype)


def CompileSpec(
    inputs,
    outputs,
    backend=CoreMLComputeUnit.CPU,
    allow_low_precision=True,
    quantization_mode=CoreMLQuantizationMode.NONE,
    mlmodel_export_path=None,
):
    return (
        inputs,
        outputs,
        backend,
        allow_low_precision,
        quantization_mode,
        mlmodel_export_path,
    )


def _check_enumerated_shape(shape):
    for s in shape:
        if not isinstance(s, (list, tuple)):
            return False
    return True


def _convert_to_mil_type(shape, dtype, name: str):
    mil_shape = shape
    if _check_enumerated_shape(shape):
        mil_shape = ct.EnumeratedShapes(shape)
    ml_type = TensorType(shape=mil_shape, dtype=torch_to_mil_types[dtype])
    ml_type.name = name
    return ml_type


def preprocess(script_module: torch._C.ScriptObject, compile_spec: dict[str, tuple]):
    spec = compile_spec["forward"]
    (
        input_specs,
        output_specs,
        backend,
        allow_low_precision,
        quantization_mode,
        mlmodel_export_path,
    ) = spec
    mil_inputs = []
    inputs = []
    for index, input in enumerate(input_specs):
        shape, dtype = input
        name = "input_" + str(index)
        inputs.append([name, str(dtype), str(shape)])
        ml_type = _convert_to_mil_type(shape, dtype, name)
        mil_inputs.append(ml_type)
    model = torch.jit.RecursiveScriptModule._construct(script_module, lambda x: None)
    mlmodel = ct.convert(model, inputs=mil_inputs)

    if quantization_mode != CoreMLQuantizationMode.NONE:
        quant_model_spec = quantization_utils.quantize_weights(
            mlmodel, nbits=8, quantization_mode=quantization_mode
        )
        mlmodel = ct.models.MLModel(quant_model_spec)

    spec = mlmodel.get_spec()
    assert len(spec.description.output) == len(output_specs)  # type: ignore[attr-defined]
    outputs = []
    for index, output in enumerate(output_specs):
        shape, dtype = output
        name = spec.description.output[index].name  # type: ignore[attr-defined]
        outputs.append([name, str(dtype), str(shape)])
    mlmodel = ct.models.model.MLModel(spec)
    print(mlmodel)

    if mlmodel_export_path is not None:
        print(f"Saving CoreML .mlmodel file to {mlmodel_export_path}")
        mlmodel.save(mlmodel_export_path)

    config = {
        "spec_ver": str(spec.specificationVersion),  # type: ignore[attr-defined]
        "backend": backend,
        "allow_low_precision": str(allow_low_precision),
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
