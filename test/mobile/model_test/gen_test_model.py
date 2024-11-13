import io
import sys

import yaml
from android_api_module import AndroidAPIModule
from builtin_ops import TSBuiltinOpsModule, TSCollectionOpsModule
from math_ops import (
    BlasLapackOpsModule,
    ComparisonOpsModule,
    OtherMathOpsModule,
    PointwiseOpsModule,
    ReductionOpsModule,
    SpectralOpsModule,
)
from nn_ops import (
    NNActivationModule,
    NNConvolutionModule,
    NNDistanceModule,
    NNDropoutModule,
    NNLinearModule,
    NNLossFunctionModule,
    NNNormalizationModule,
    NNPaddingModule,
    NNPoolingModule,
    NNRecurrentModule,
    NNShuffleModule,
    NNSparseModule,
    NNTransformerModule,
    NNUtilsModule,
    NNVisionModule,
)
from quantization_ops import FusedQuantModule, GeneralQuantModule, StaticQuantModule
from sampling_ops import SamplingOpsModule
from tensor_ops import (
    TensorCreationOpsModule,
    TensorIndexingOpsModule,
    TensorOpsModule,
    TensorTypingOpsModule,
    TensorViewOpsModule,
)
from torchvision_models import (
    MobileNetV2Module,
    MobileNetV2VulkanModule,
    Resnet18Module,
)

import torch
from torch.jit.mobile import _load_for_lite_interpreter


test_path_ios = "ios/TestApp/models/"
test_path_android = "android/pytorch_android/src/androidTest/assets/"

production_ops_path = "test/mobile/model_test/model_ops.yaml"
coverage_out_path = "test/mobile/model_test/coverage.yaml"

all_modules = {
    # math ops
    "pointwise_ops": PointwiseOpsModule(),
    "reduction_ops": ReductionOpsModule(),
    "comparison_ops": ComparisonOpsModule(),
    "spectral_ops": SpectralOpsModule(),
    "other_math_ops": OtherMathOpsModule(),
    "blas_lapack_ops": BlasLapackOpsModule(),
    # sampling
    "sampling_ops": SamplingOpsModule(),
    # tensor ops
    "tensor_general_ops": TensorOpsModule(),
    "tensor_creation_ops": TensorCreationOpsModule(),
    "tensor_indexing_ops": TensorIndexingOpsModule(),
    "tensor_typing_ops": TensorTypingOpsModule(),
    "tensor_view_ops": TensorViewOpsModule(),
    # nn ops
    "convolution_ops": NNConvolutionModule(),
    "pooling_ops": NNPoolingModule(),
    "padding_ops": NNPaddingModule(),
    "activation_ops": NNActivationModule(),
    "normalization_ops": NNNormalizationModule(),
    "recurrent_ops": NNRecurrentModule(),
    "transformer_ops": NNTransformerModule(),
    "linear_ops": NNLinearModule(),
    "dropout_ops": NNDropoutModule(),
    "sparse_ops": NNSparseModule(),
    "distance_function_ops": NNDistanceModule(),
    "loss_function_ops": NNLossFunctionModule(),
    "vision_function_ops": NNVisionModule(),
    "shuffle_ops": NNShuffleModule(),
    "nn_utils_ops": NNUtilsModule(),
    # quantization ops
    "general_quant_ops": GeneralQuantModule(),
    # TODO(sdym@fb.com): fix and re-enable dynamic_quant_ops
    # "dynamic_quant_ops": DynamicQuantModule(),
    "static_quant_ops": StaticQuantModule(),
    "fused_quant_ops": FusedQuantModule(),
    # TorchScript buildin ops
    "torchscript_builtin_ops": TSBuiltinOpsModule(),
    "torchscript_collection_ops": TSCollectionOpsModule(),
    # vision
    "mobilenet_v2": MobileNetV2Module(),
    "mobilenet_v2_vulkan": MobileNetV2VulkanModule(),
    "resnet18": Resnet18Module(),
    # android api module
    "android_api_module": AndroidAPIModule(),
}

models_need_trace = [
    "static_quant_ops",
]


def calcOpsCoverage(ops):
    with open(production_ops_path) as input_yaml_file:
        production_ops_dict = yaml.safe_load(input_yaml_file)

    production_ops = set(production_ops_dict["root_operators"].keys())
    all_generated_ops = set(ops)
    covered_ops = production_ops.intersection(all_generated_ops)
    uncovered_ops = production_ops - covered_ops
    coverage = round(100 * len(covered_ops) / len(production_ops), 2)

    # weighted coverage (take op occurances into account)
    total_occurances = sum(production_ops_dict["root_operators"].values())
    covered_ops_dict = {
        op: production_ops_dict["root_operators"][op] for op in covered_ops
    }
    uncovered_ops_dict = {
        op: production_ops_dict["root_operators"][op] for op in uncovered_ops
    }
    covered_occurances = sum(covered_ops_dict.values())
    occurances_coverage = round(100 * covered_occurances / total_occurances, 2)

    print(f"\n{len(uncovered_ops)} uncovered ops: {uncovered_ops}\n")
    print(f"Generated {len(all_generated_ops)} ops")
    print(
        f"Covered {len(covered_ops)}/{len(production_ops)} ({coverage}%) production ops"
    )
    print(
        f"Covered {covered_occurances}/{total_occurances} ({occurances_coverage}%) occurances"
    )
    print(f"pytorch ver {torch.__version__}\n")

    with open(coverage_out_path, "w") as f:
        yaml.safe_dump(
            {
                "_covered_ops": len(covered_ops),
                "_production_ops": len(production_ops),
                "_generated_ops": len(all_generated_ops),
                "_uncovered_ops": len(uncovered_ops),
                "_coverage": round(coverage, 2),
                "uncovered_ops": uncovered_ops_dict,
                "covered_ops": covered_ops_dict,
                "all_generated_ops": sorted(all_generated_ops),
            },
            f,
        )


def getModuleFromName(model_name):
    if model_name not in all_modules:
        print("Cannot find test model for " + model_name)
        return None, []

    module = all_modules[model_name]
    if not isinstance(module, torch.nn.Module):
        module = module.getModule()

    has_bundled_inputs = False  # module.find_method("get_all_bundled_inputs")

    if model_name in models_need_trace:
        module = torch.jit.trace(module, [])
    else:
        module = torch.jit.script(module)

    ops = torch.jit.export_opnames(module)
    print(ops)

    # try to run the model
    runModule(module)

    return module, ops


def runModule(module):
    buffer = io.BytesIO(module._save_to_buffer_for_lite_interpreter())
    buffer.seek(0)
    lite_module = _load_for_lite_interpreter(buffer)
    if lite_module.find_method("get_all_bundled_inputs"):
        # run with the first bundled input
        input = lite_module.run_method("get_all_bundled_inputs")[0]
        lite_module.forward(*input)
    else:
        # assuming model has no input
        lite_module()


# generate all models in the given folder.
# If it's "on the fly" mode, add "_temp" suffix to the model file.
def generateAllModels(folder, on_the_fly=False):
    all_ops = []
    for name in all_modules:
        module, ops = getModuleFromName(name)
        all_ops = all_ops + ops
        path = folder + name + ("_temp.ptl" if on_the_fly else ".ptl")
        module._save_for_lite_interpreter(path)
        print("model saved to " + path)
    calcOpsCoverage(all_ops)


# generate/update a given model for storage
def generateModel(name):
    module, ops = getModuleFromName(name)
    if module is None:
        return
    path_ios = test_path_ios + name + ".ptl"
    path_android = test_path_android + name + ".ptl"
    module._save_for_lite_interpreter(path_ios)
    module._save_for_lite_interpreter(path_android)
    print("model saved to " + path_ios + " and " + path_android)


def main(argv):
    if argv is None or len(argv) != 1:
        print(
            """
This script generate models for mobile test. For each model we have a "storage" version
and an "on-the-fly" version. The "on-the-fly" version will be generated during test,and
should not be committed to the repo.
The "storage" version is for back compatibility # test (a model generated today should
run on master branch in the next 6 months). We can use this script to update a model that
is no longer supported.
- use 'python gen_test_model.py android-test' to generate on-the-fly models for android
- use 'python gen_test_model.py ios-test' to generate on-the-fly models for ios
- use 'python gen_test_model.py android' to generate checked-in models for android
- use 'python gen_test_model.py ios' to generate on-the-fly models for ios
- use 'python gen_test_model.py <model_name_no_suffix>' to update the given storage model
"""
        )
        return

    if argv[0] == "android":
        generateAllModels(test_path_android, on_the_fly=False)
    elif argv[0] == "ios":
        generateAllModels(test_path_ios, on_the_fly=False)
    elif argv[0] == "android-test":
        generateAllModels(test_path_android, on_the_fly=True)
    elif argv[0] == "ios-test":
        generateAllModels(test_path_ios, on_the_fly=True)
    else:
        generateModel(argv[0])


if __name__ == "__main__":
    main(sys.argv[1:])
