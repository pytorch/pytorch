import torch
import yaml
from math_ops import (
    PointwiseOpsModule,
    ReductionOpsModule,
    ComparisonOpsModule,
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
)
from sampling_ops import SamplingOpsModule
from tensor_ops import (
    TensorOpsModule,
    TensorCreationOpsModule,
    TensorIndexingOpsModule,
    TensorTypingOpsModule,
    TensorViewOpsModule,
)

output_path = "ios/TestApp/models/"
production_ops_path = "test/mobile/model_test/model_ops.yaml"
coverage_out_path = "test/mobile/model_test/coverage.yaml"


def scriptAndSave(module, name):
    script_module = torch.jit.script(module)
    script_module._save_for_lite_interpreter(output_path + name)
    script_module()
    print("model saved to " + output_path + name)
    ops = torch.jit.export_opnames(script_module)
    return ops


ops = [
    # math ops
    scriptAndSave(PointwiseOpsModule(), "pointwise_ops.ptl"),
    scriptAndSave(ReductionOpsModule(), "reduction_ops.ptl"),
    scriptAndSave(ComparisonOpsModule(), "comparison_ops.ptl"),
    scriptAndSave(SpectralOpsModule(), "spectral_ops.ptl"),
    scriptAndSave(BlasLapackOpsModule(), "blas_lapack_ops.ptl"),
    # sampling
    scriptAndSave(SamplingOpsModule(), "sampling_ops.ptl"),
    # tensor ops
    scriptAndSave(TensorOpsModule(), "tensor_general_ops.ptl"),
    scriptAndSave(TensorCreationOpsModule(), "tensor_creation_ops.ptl"),
    scriptAndSave(TensorIndexingOpsModule(), "tensor_indexing_ops.ptl"),
    scriptAndSave(TensorTypingOpsModule(), "tensor_typing_ops.ptl"),
    scriptAndSave(TensorViewOpsModule(), "tensor_view_ops.ptl"),
    # nn ops
    scriptAndSave(NNConvolutionModule(), "convolution_ops.ptl"),
    scriptAndSave(NNPoolingModule(), "pooling_ops.ptl"),
    scriptAndSave(NNPaddingModule(), "padding_ops.ptl"),
    scriptAndSave(NNActivationModule(), "activation_ops.ptl"),
    scriptAndSave(NNNormalizationModule(), "normalization_ops.ptl"),
    scriptAndSave(NNRecurrentModule(), "recurrent_ops.ptl"),
    scriptAndSave(NNTransformerModule(), "transformer_ops.ptl"),
    scriptAndSave(NNLinearModule(), "linear_ops.ptl"),
    scriptAndSave(NNDropoutModule(), "dropout_ops.ptl"),
    scriptAndSave(NNSparseModule(), "sparse_ops.ptl"),
    scriptAndSave(NNDistanceModule(), "distance_function_ops.ptl"),
    scriptAndSave(NNLossFunctionModule(), "loss_function_ops.ptl"),
    scriptAndSave(NNVisionModule(), "vision_function_ops.ptl"),
    scriptAndSave(NNShuffleModule(), "shuffle_ops.ptl"),
]


with open(production_ops_path) as input_yaml_file:
    production_ops = yaml.safe_load(input_yaml_file)

production_ops = set(production_ops["root_operators"])
all_generated_ops = set().union(*ops)
covered_ops = production_ops.intersection(all_generated_ops)
uncovered_ops = production_ops - covered_ops
coverage = 100 * len(covered_ops) / len(production_ops)
print(
    f"\nThese generated models covered {len(covered_ops)} of {len(production_ops)} production ops. ({round(coverage, 2)}%)\n"
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
