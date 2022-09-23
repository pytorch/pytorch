# This file keeps a list of PyTorch source files that are used for templated selective build.
# NB: as this is PyTorch Edge selective build, we assume only CPU targets are
# being built

load("@bazel_skylib//lib:paths.bzl", "paths")
load("//tools/build_defs:fbsource_utils.bzl", "is_arvr_mode")
load(":build_variables.bzl", "aten_native_source_list")
load(
    ":ufunc_defs.bzl",
    "aten_ufunc_generated_cpu_kernel_sources",
    "aten_ufunc_generated_cpu_sources",
)

# Files in this list are supposed to be built separately for each app,
# for different operator allow lists.
TEMPLATE_SOURCE_LIST = [
    "torch/csrc/jit/runtime/register_prim_ops.cpp",
    "torch/csrc/jit/runtime/register_special_ops.cpp",
] + aten_native_source_list

# For selective build, we can lump the CPU and CPU kernel sources altogether
# because there is only ever one vectorization variant that is compiled
def aten_ufunc_generated_all_cpu_sources(gencode_pattern = "{}"):
    return (
        aten_ufunc_generated_cpu_sources(gencode_pattern) +
        aten_ufunc_generated_cpu_kernel_sources(gencode_pattern)
    )

TEMPLATE_MASKRCNN_SOURCE_LIST = [
    "register_maskrcnn_ops.cpp",
]

TEMPLATE_BATCH_BOX_COX_SOURCE_LIST = [
    "register_batch_box_cox_ops.cpp",
]

METAL_SOURCE_LIST = [
    "aten/src/ATen/native/metal/MetalAten.mm",
    "aten/src/ATen/native/metal/MetalGuardImpl.cpp",
    "aten/src/ATen/native/metal/MetalPrepackOpRegister.cpp",
    "aten/src/ATen/native/metal/MetalCommandBuffer.mm",
    "aten/src/ATen/native/metal/MetalContext.mm",
    "aten/src/ATen/native/metal/MetalConvParams.mm",
    "aten/src/ATen/native/metal/MetalTensorImplStorage.mm",
    "aten/src/ATen/native/metal/MetalTensorUtils.mm",
    "aten/src/ATen/native/metal/mpscnn/MPSCNNClampOp.mm",
    "aten/src/ATen/native/metal/mpscnn/MPSCNNConvOp.mm",
    "aten/src/ATen/native/metal/mpscnn/MPSCNNFullyConnectedOp.mm",
    "aten/src/ATen/native/metal/mpscnn/MPSCNNNeuronOp.mm",
    "aten/src/ATen/native/metal/mpscnn/MPSCNNUtils.mm",
    "aten/src/ATen/native/metal/mpscnn/MPSImage+Tensor.mm",
    "aten/src/ATen/native/metal/mpscnn/MPSImageUtils.mm",
    "aten/src/ATen/native/metal/mpscnn/MPSImageWrapper.mm",
    "aten/src/ATen/native/metal/ops/MetalAddmm.mm",
    "aten/src/ATen/native/metal/ops/MetalBinaryElementwise.mm",
    "aten/src/ATen/native/metal/ops/MetalChunk.mm",
    "aten/src/ATen/native/metal/ops/MetalClamp.mm",
    "aten/src/ATen/native/metal/ops/MetalConcat.mm",
    "aten/src/ATen/native/metal/ops/MetalConvolution.mm",
    "aten/src/ATen/native/metal/ops/MetalCopy.mm",
    "aten/src/ATen/native/metal/ops/MetalHardswish.mm",
    "aten/src/ATen/native/metal/ops/MetalHardshrink.mm",
    "aten/src/ATen/native/metal/ops/MetalLeakyReLU.mm",
    "aten/src/ATen/native/metal/ops/MetalNeurons.mm",
    "aten/src/ATen/native/metal/ops/MetalPadding.mm",
    "aten/src/ATen/native/metal/ops/MetalPooling.mm",
    "aten/src/ATen/native/metal/ops/MetalReduce.mm",
    "aten/src/ATen/native/metal/ops/MetalReshape.mm",
    "aten/src/ATen/native/metal/ops/MetalSoftmax.mm",
    "aten/src/ATen/native/metal/ops/MetalTranspose.mm",
    "aten/src/ATen/native/metal/ops/MetalUpsamplingNearest.mm",
]

UNET_METAL_PREPACK_SOURCE_LIST = [
    "unet_metal_prepack.cpp",
    "unet_metal_prepack.mm",
]

METAL_MASKRCNN_SOURCE_LIST = [
    "maskrcnn/srcs/GenerateProposals.mm",
    "maskrcnn/srcs/RoIAlign.mm",
]

# The get_template_source_dict() returns a dict containing a path prefix
# and a list of .cpp source files containing operator definitions and
# registrations that should get selected via templated selective build.
# The file selected_mobile_ops.h has the list of selected top level
# operators.
# NB: doesn't include generated files; copy_template_registration_files
# handles those specially
def get_template_source_dict():
    ret = {}
    for file_path in TEMPLATE_SOURCE_LIST:
        path_prefix = paths.dirname(file_path)
        if path_prefix not in ret:
            ret[path_prefix] = []
        ret[path_prefix].append(file_path)
    return ret

def get_gen_oplist_outs():
    return {
        "SupportedMobileModelsRegistration.cpp": [
            "SupportedMobileModelsRegistration.cpp",
        ],
        "selected_mobile_ops.h": [
            "selected_mobile_ops.h",
        ],
        "selected_operators.yaml": [
            "selected_operators.yaml",
        ],
    }

def get_generate_code_bin_outs():
    outs = {
        "autograd/generated/ADInplaceOrViewTypeEverything.cpp": ["autograd/generated/ADInplaceOrViewTypeEverything.cpp"],
        "autograd/generated/ADInplaceOrViewType_0.cpp": ["autograd/generated/ADInplaceOrViewType_0.cpp"],
        "autograd/generated/ADInplaceOrViewType_1.cpp": ["autograd/generated/ADInplaceOrViewType_1.cpp"],
        "autograd/generated/Functions.cpp": ["autograd/generated/Functions.cpp"],
        "autograd/generated/Functions.h": ["autograd/generated/Functions.h"],
        "autograd/generated/TraceTypeEverything.cpp": ["autograd/generated/TraceTypeEverything.cpp"],
        "autograd/generated/TraceType_0.cpp": ["autograd/generated/TraceType_0.cpp"],
        "autograd/generated/TraceType_1.cpp": ["autograd/generated/TraceType_1.cpp"],
        "autograd/generated/TraceType_2.cpp": ["autograd/generated/TraceType_2.cpp"],
        "autograd/generated/TraceType_3.cpp": ["autograd/generated/TraceType_3.cpp"],
        "autograd/generated/TraceType_4.cpp": ["autograd/generated/TraceType_4.cpp"],
        "autograd/generated/VariableType.h": ["autograd/generated/VariableType.h"],
        "autograd/generated/VariableTypeEverything.cpp": ["autograd/generated/VariableTypeEverything.cpp"],
        "autograd/generated/VariableType_0.cpp": ["autograd/generated/VariableType_0.cpp"],
        "autograd/generated/VariableType_1.cpp": ["autograd/generated/VariableType_1.cpp"],
        "autograd/generated/VariableType_2.cpp": ["autograd/generated/VariableType_2.cpp"],
        "autograd/generated/VariableType_3.cpp": ["autograd/generated/VariableType_3.cpp"],
        "autograd/generated/VariableType_4.cpp": ["autograd/generated/VariableType_4.cpp"],
        "autograd/generated/variable_factories.h": ["autograd/generated/variable_factories.h"],
    }

    if is_arvr_mode():
        outs.update({
            "autograd/generated/python_enum_tag.cpp": ["autograd/generated/python_enum_tag.cpp"],
            "autograd/generated/python_fft_functions.cpp": ["autograd/generated/python_fft_functions.cpp"],
            "autograd/generated/python_functions.h": ["autograd/generated/python_functions.h"],
            "autograd/generated/python_functions_0.cpp": ["autograd/generated/python_functions_0.cpp"],
            "autograd/generated/python_functions_1.cpp": ["autograd/generated/python_functions_1.cpp"],
            "autograd/generated/python_functions_2.cpp": ["autograd/generated/python_functions_2.cpp"],
            "autograd/generated/python_functions_3.cpp": ["autograd/generated/python_functions_3.cpp"],
            "autograd/generated/python_functions_4.cpp": ["autograd/generated/python_functions_4.cpp"],
            "autograd/generated/python_linalg_functions.cpp": ["autograd/generated/python_linalg_functions.cpp"],
            "autograd/generated/python_nested_functions.cpp": ["autograd/generated/python_nested_functions.cpp"],
            "autograd/generated/python_nn_functions.cpp": ["autograd/generated/python_nn_functions.cpp"],
            "autograd/generated/python_return_types.cpp": ["autograd/generated/python_return_types.cpp"],
            "autograd/generated/python_sparse_functions.cpp": ["autograd/generated/python_sparse_functions.cpp"],
            "autograd/generated/python_special_functions.cpp": ["autograd/generated/python_special_functions.cpp"],
            "autograd/generated/python_torch_functions_0.cpp": ["autograd/generated/python_torch_functions_0.cpp"],
            "autograd/generated/python_torch_functions_1.cpp": ["autograd/generated/python_torch_functions_1.cpp"],
            "autograd/generated/python_torch_functions_2.cpp": ["autograd/generated/python_torch_functions_2.cpp"],
            "autograd/generated/python_variable_methods.cpp": ["autograd/generated/python_variable_methods.cpp"],
        })
    return outs

def get_template_registration_files_outs(is_oss = False):
    outs = {}
    if not is_oss:
        for file_path in TEMPLATE_MASKRCNN_SOURCE_LIST:
            outs[file_path] = [file_path]

        for file_path in TEMPLATE_BATCH_BOX_COX_SOURCE_LIST:
            outs[file_path] = [file_path]

    for file_path in TEMPLATE_SOURCE_LIST:
        outs[file_path] = [file_path]

    for base_name in aten_ufunc_generated_all_cpu_sources():
        file_path = "aten/src/ATen/{}".format(base_name)
        outs[file_path] = [file_path]

    return outs

def get_template_registration_file_rules(rule_name, is_oss = False):
    rules = []
    for file_path in TEMPLATE_SOURCE_LIST if is_oss else (TEMPLATE_SOURCE_LIST + TEMPLATE_MASKRCNN_SOURCE_LIST + TEMPLATE_BATCH_BOX_COX_SOURCE_LIST):
        rules.append(":{}[{}]".format(rule_name, file_path))
    for file_path in aten_ufunc_generated_all_cpu_sources():
        rules.append(":{}[aten/src/ATen/{}]".format(rule_name, file_path))

    return rules

# ---------------------METAL RULES---------------------
def get_metal_source_dict():
    ret = {}
    for file_path in METAL_SOURCE_LIST:
        path_prefix = paths.dirname(file_path)
        if path_prefix not in ret:
            ret[path_prefix] = []
        ret[path_prefix].append(file_path)
    return ret

def get_metal_registration_files_outs():
    outs = {}
    for file_path in METAL_SOURCE_LIST:
        outs[file_path] = [file_path]

    for file_path in UNET_METAL_PREPACK_SOURCE_LIST:
        outs[file_path] = [file_path]

    for file_path in METAL_MASKRCNN_SOURCE_LIST:
        outs[file_path] = [file_path]
    return outs

# There is a really weird issue with the arvr windows builds where
# the custom op files are breaking them. See https://fburl.com/za87443c
# The hack is just to not build them for that platform and pray they arent needed.
def get_metal_registration_files_outs_windows():
    outs = {}
    for file_path in METAL_SOURCE_LIST:
        outs[file_path] = [file_path]
    return outs

def get_metal_registration_files_rules(rule_name):
    ret = {}
    objc_rules = []
    cxx_rules = []

    for file_path in METAL_SOURCE_LIST + METAL_MASKRCNN_SOURCE_LIST + UNET_METAL_PREPACK_SOURCE_LIST:
        if ".cpp" not in file_path:
            objc_rules.append(":{}[{}]".format(rule_name, file_path))
        else:
            cxx_rules.append(":{}[{}]".format(rule_name, file_path))
    ret["objc"] = objc_rules
    ret["cxx"] = cxx_rules
    return ret

def get_metal_registration_files_rules_windows(rule_name):
    ret = {}
    objc_rules = []
    cxx_rules = []

    for file_path in METAL_SOURCE_LIST:
        if ".cpp" not in file_path:
            objc_rules.append(":{}[{}]".format(rule_name, file_path))
        else:
            cxx_rules.append(":{}[{}]".format(rule_name, file_path))
    ret["objc"] = objc_rules
    ret["cxx"] = cxx_rules
    return ret
