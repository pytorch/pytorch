import torch
import yaml
from builtin_ops import BuiltinOpsModule
from math_ops import (
    PointwiseOpsModule,
    ReductionOpsModule,
    ComparisonOpsModule,
    OtherMathOpsModule,
    SpectralOpsModule,
    BlasLapackOpsModule,
)
from nn_ops import (
    NNConvolutionModule,
    NNPoolingModule,
    NNPaddingModule,
    NNNormalizationModule,
    NNActivationModule,
    NNRecurrentModule,
    NNTransformerModule,
    NNLinearModule,
    NNDropoutModule,
    NNSparseModule,
    NNDistanceModule,
    NNLossFunctionModule,
    NNVisionModule,
    NNShuffleModule,
    NNUtilsModule,
)
from quantization_ops import (
    GeneralQuantModule,
    DynamicQuantModule,
    StaticQuantModule,
    FusedQuantModule,
)
from sampling_ops import SamplingOpsModule
from tensor_ops import (
    TensorOpsModule,
    TensorCreationOpsModule,
    TensorIndexingOpsModule,
    TensorTypingOpsModule,
    TensorViewOpsModule,
)
from torch.utils.mobile_optimizer import optimize_for_mobile


output_path = "ios/TestApp/models/"
production_ops_path = "test/mobile/model_test/model_ops.yaml"
coverage_out_path = "test/mobile/model_test/coverage.yaml"


def scriptAndSave(module, name):
    module = torch.jit.script(module)
    return save(module, name)


def traceAndSave(module, name):
    module = torch.jit.trace(module, [])
    return save(module, name)


def save(module, name):
    module._save_for_lite_interpreter(output_path + name + ".ptl")
    print("model saved to " + output_path + name + ".ptl")
    ops = torch.jit.export_opnames(module)
    print(ops)
    module()
    return ops


ops = [
    # math ops
    scriptAndSave(PointwiseOpsModule(), "pointwise_ops"),
    scriptAndSave(ReductionOpsModule(), "reduction_ops"),
    scriptAndSave(ComparisonOpsModule(), "comparison_ops"),
    scriptAndSave(OtherMathOpsModule(), "other_math_ops"),
    scriptAndSave(SpectralOpsModule(), "spectral_ops"),
    scriptAndSave(BlasLapackOpsModule(), "blas_lapack_ops"),
    # sampling
    scriptAndSave(SamplingOpsModule(), "sampling_ops"),
    # tensor ops
    scriptAndSave(TensorOpsModule(), "tensor_general_ops"),
    scriptAndSave(TensorCreationOpsModule(), "tensor_creation_ops"),
    scriptAndSave(TensorIndexingOpsModule(), "tensor_indexing_ops"),
    scriptAndSave(TensorTypingOpsModule(), "tensor_typing_ops"),
    scriptAndSave(TensorViewOpsModule(), "tensor_view_ops"),
    # nn ops
    scriptAndSave(NNConvolutionModule(), "convolution_ops"),
    scriptAndSave(NNPoolingModule(), "pooling_ops"),
    scriptAndSave(NNPaddingModule(), "padding_ops"),
    scriptAndSave(NNActivationModule(), "activation_ops"),
    scriptAndSave(NNNormalizationModule(), "normalization_ops"),
    scriptAndSave(NNRecurrentModule(), "recurrent_ops"),
    scriptAndSave(NNTransformerModule(), "transformer_ops"),
    scriptAndSave(NNLinearModule(), "linear_ops"),
    scriptAndSave(NNDropoutModule(), "dropout_ops"),
    scriptAndSave(NNSparseModule(), "sparse_ops"),
    scriptAndSave(NNDistanceModule(), "distance_function_ops"),
    scriptAndSave(NNLossFunctionModule(), "loss_function_ops"),
    scriptAndSave(NNVisionModule(), "vision_function_ops"),
    scriptAndSave(NNShuffleModule(), "shuffle_ops"),
    scriptAndSave(NNUtilsModule(), "nn_utils_ops"),
    # quantization ops
    scriptAndSave(GeneralQuantModule(), "general_quant_ops"),
    scriptAndSave(DynamicQuantModule().getModule(), "dynamic_quant_ops"),
    traceAndSave(StaticQuantModule().getModule(), "static_quant_ops"),
    scriptAndSave(FusedQuantModule().getModule(), "fused_quant_ops"),
    # TorchScript buildin ops
    scriptAndSave(BuiltinOpsModule(), "torchscript_builtin_ops"),
]


with open(production_ops_path) as input_yaml_file:
    production_ops = yaml.safe_load(input_yaml_file)

production_ops = set(production_ops["root_operators"])
all_generated_ops = set().union(*ops)
covered_ops = production_ops.intersection(all_generated_ops)
uncovered_ops = production_ops - covered_ops
coverage = 100 * len(covered_ops) / len(production_ops)
print(
    f"\nGenerated {len(all_generated_ops)} ops and covered {len(covered_ops)}/{len(production_ops)} ({round(coverage, 2)}%) production ops. \n"
)
with open(coverage_out_path, "w") as f:
    yaml.safe_dump(
        {
            "uncovered_ops": sorted(list(uncovered_ops)),
            "covered_ops": sorted(list(covered_ops)),
            "all_generated_ops": sorted(list(all_generated_ops)),
        },
        f,
    )
