load("@bazel_skylib//lib:paths.bzl", "paths")
load("//tools/build_defs:fb_xplat_cxx_library.bzl", "fb_xplat_cxx_library")
load("//tools/build_defs:fb_xplat_genrule.bzl", "fb_xplat_genrule")
load("//tools/build_defs:type_defs.bzl", "is_list", "is_string")
load(":build_variables.bzl", "aten_native_source_list")
load(
    ":ufunc_defs.bzl",
    "aten_ufunc_generated_cpu_kernel_sources",
    "aten_ufunc_generated_cpu_sources",
    "aten_ufunc_generated_cuda_sources",
)

USED_PT_BACKENDS = [
    "CPU",
    "QuantizedCPU",
    "SparseCPU",  # brings ~20 kb size regression
]

# This needs to be kept in sync with https://github.com/pytorch/pytorch/blob/release/1.9/torchgen/gen.py#L892
PT_BACKEND_HEADERS = [
    "CPU",
    "CUDA",
    "CompositeExplicitAutograd",
    "CompositeExplicitAutogradNonFunctional",
    "CompositeImplicitAutograd",
    "Meta",
]

PT_BASE_OPS = [
    "aten::_coalesced_",
    "aten::_copy_from",
    "aten::_empty_affine_quantized",
    "aten::_empty_per_channel_affine_quantized",
    "aten::_indices",
    "aten::_nnz",
    "aten::_values",
    "aten::add",
    "aten::add_",
    "aten::arange",
    "aten::as_strided",
    "aten::as_strided_",
    "aten::cat",
    "aten::clone",
    "aten::coalesce",
    "aten::contiguous",
    "aten::copy_",
    "aten::copy_sparse_to_sparse_",
    "aten::dense_dim",
    "aten::dequantize",
    "aten::div",
    "aten::div_",
    "aten::empty",
    "aten::empty_like",
    "aten::empty_strided",
    "aten::empty.memory_format",
    "aten::eq",
    "aten::equal",
    "aten::expand",
    "aten::fill_",
    "aten::is_coalesced",
    "aten::is_complex",
    "aten::is_floating_point",
    "aten::is_leaf",
    "aten::is_nonzero",
    "aten::item",
    "aten::max",
    "aten::min",
    "aten::mul",
    "aten::mul_",
    "aten::narrow",
    "aten::ne",
    "aten::permute",
    "aten::q_per_channel_axis",
    "aten::q_per_channel_scales",
    "aten::q_per_channel_zero_points",
    "aten::q_scale",
    "aten::q_zero_point",
    "aten::qscheme",
    "aten::quantize_per_tensor",
    "aten::reshape",
    "aten::_reshape_alias",
    "aten::resize_",
    "aten::resize_as_",
    "aten::scalar_tensor",
    "aten::select",
    "aten::set_",
    "aten::size",
    "aten::slice",
    "aten::sparse_dim",
    "aten::sparse_resize_and_clear_",
    "aten::squeeze",
    "aten::squeeze_",
    "aten::stride",
    "aten::sub",
    "aten::sub_",
    "aten::sum",
    "aten::t",
    "aten::to",
    "aten::_to_copy",
    "aten::unsqueeze",
    "aten::view",
    "aten::zero_",
    "aten::zeros",
    "aten::zeros_like",
]

def get_aten_compiler_flags():
    return ATEN_COMPILER_FLAGS

def get_generate_code_bin_outs():
    return {
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

ATEN_COMPILER_FLAGS = [
    "-fexceptions",
    "-frtti",
    "-fPIC",
    "-Os",
    "-Wno-absolute-value",
    "-Wno-deprecated-declarations",
    "-Wno-macro-redefined",
    "-Wno-tautological-constant-out-of-range-compare",
    "-Wno-unknown-pragmas",
    "-Wno-unknown-warning-option",
    "-Wno-unused-function",
    "-Wno-unused-variable",
    "-Wno-pass-failed",
    "-Wno-shadow",
]

PT_COMPILER_FLAGS = [
    "-frtti",
    "-Os",
    "-Wno-unknown-pragmas",
    "-Wno-write-strings",
    "-Wno-unused-variable",
    "-Wno-unused-function",
    "-Wno-deprecated-declarations",
    "-Wno-shadow",
    "-Wno-global-constructors",
    "-Wno-missing-prototypes",
    "-std=gnu++17",  # to accommodate Eigen
]

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
        #"SupportedMobileModelsRegistration.cpp": [
        #    "SupportedMobileModelsRegistration.cpp",
        #],
        "selected_mobile_ops.h": [
            "selected_mobile_ops.h",
        ],
        "selected_operators.yaml": [
            "selected_operators.yaml",
        ],
    }

def get_pt_compiler_flags():
    return PT_COMPILER_FLAGS

def get_aten_preprocessor_flags():
    # read_config is not allowed outside of function in Starlark
    ATEN_PREPROCESSOR_FLAGS = [
        "-DC10_MOBILE",
        "-DCPU_CAPABILITY_DEFAULT",
        "-DCPU_CAPABILITY=DEFAULT",
        "-DCAFFE2_USE_LITE_PROTO",
        "-DATEN_CUDNN_ENABLED_FBXPLAT=0",
        "-DATEN_MKLDNN_ENABLED_FBXPLAT=0",
        "-DATEN_NNPACK_ENABLED_FBXPLAT=0",
        "-DATEN_MKL_ENABLED_FBXPLAT=0",
        "-DATEN_MKL_SEQUENTIAL_FBXPLAT=0",
        "-DUSE_PYTORCH_METAL",
        "-DUSE_PYTORCH_QNNPACK",
        "-DUSE_XNNPACK",
        "-DNO_EXPORT",
        "-DPYTORCH_QNNPACK_RUNTIME_QUANTIZATION",
        "-DAT_PARALLEL_OPENMP_FBXPLAT=0",
        "-DAT_PARALLEL_NATIVE_FBXPLAT=1",
        "-DAT_PARALLEL_NATIVE_TBB_FBXPLAT=0",
        "-DUSE_LAPACK_FBXPLAT=0",
        "-DAT_BLAS_F2C_FBXPLAT=0",
        "-DAT_BLAS_USE_CBLAS_DOT_FBXPLAT=0",
        "-DUSE_RUY_QMATMUL",  # need third_party:ruy
    ]

    # if get_disable_per_op_profiling():
    ATEN_PREPROCESSOR_FLAGS.append("-DPYTORCH_DISABLE_PER_OP_PROFILING")
    return ATEN_PREPROCESSOR_FLAGS

TEMPLATE_SOURCE_LIST = [
    "torch/csrc/jit/runtime/register_prim_ops.cpp",
    "torch/csrc/jit/runtime/register_special_ops.cpp",
] + aten_native_source_list

# the basic xplat library for OSS build
def pt_xplat_cxx_library(extra_flags = {}, **kwargs):
    fb_xplat_cxx_library(
        compiler_flags = get_pt_compiler_flags() + extra_flags.get("compiler_flags", []),
        exported_preprocessor_flags = get_pt_preprocessor_flags() + extra_flags.get("exported_preprocessor_flags", []),
        **kwargs
    )

def get_aten_default_args():
    return dict(
        compiler_flags = get_aten_compiler_flags(),
        exported_preprocessor_flags = get_aten_preprocessor_flags(),
    )

# For selective build, we can lump the CPU and CPU kernel sources altogether
# because there is only ever one vectorization variant that is compiled
def aten_ufunc_generated_all_cpu_sources(gencode_pattern = "{}"):
    return (
        aten_ufunc_generated_cpu_sources(gencode_pattern) +
        aten_ufunc_generated_cpu_kernel_sources(gencode_pattern)
    )

def get_template_registration_files_outs():
    outs = {}

    for file_path in TEMPLATE_SOURCE_LIST:
        outs[file_path] = [file_path]

    for base_name in aten_ufunc_generated_all_cpu_sources():
        file_path = "aten/src/ATen/{}".format(base_name)
        outs[file_path] = [file_path]

    return outs

def get_pt_preprocessor_flags():
    # read_config is not allowed outside of function in Starlark
    PT_PREPROCESSOR_FLAGS = [
        "-D_THP_CORE",
        "-DC10_MOBILE",
        "-DUSE_SCALARS",
        "-DNO_CUDNN_DESTROY_HANDLE",
        "-DNO_EXPORT",
        "-DBUILD_CAFFE2",
    ]
    return PT_PREPROCESSOR_FLAGS

def is_arvr_mode():
    return False

def get_build_from_deps_query():
    build_from_query = native.read_config("pt", "build_from_deps_query", "1")
    return bool(int(build_from_query))

def get_enable_lightweight_dispatch():
    enable_lightweight_dispatch = native.read_config("pt", "enable_lightweight_dispatch", "0")
    return bool(int(enable_lightweight_dispatch))

def get_static_dispatch_backend():
    static_dispatch_backend = native.read_config("pt", "static_dispatch_backend", None)
    if static_dispatch_backend == None:
        return []
    return static_dispatch_backend.split(";")

def get_aten_codegen_extra_params(backends):
    if get_build_from_deps_query():
        extra_params = {
            "force_schema_registration": True,
        }
        static_backends = get_static_dispatch_backend()
        if static_backends:
            extra_params["static_dispatch_backend"] = static_backends
            extra_params["enabled_backends"] = static_backends
        else:
            extra_params["enabled_backends"] = backends
        return extra_params
    else:
        return {}

def gen_aten_files(
        name,
        extra_flags = {},
        visibility = [],
        compatible_with = []):
    extra_params = []
    force_schema_registration = extra_flags.get("force_schema_registration", False)
    op_registration_allowlist = extra_flags.get("op_registration_allowlist", None)
    op_selection_yaml_path = extra_flags.get("op_selection_yaml_path", None)
    enabled_backends = extra_flags.get("enabled_backends", None)
    static_dispatch_backend = extra_flags.get("static_dispatch_backend", None)

    if force_schema_registration:
        extra_params.append("--force_schema_registration")
    if op_registration_allowlist != None and is_string(op_registration_allowlist):
        extra_params.append("--op_registration_whitelist")
        extra_params.append(op_registration_allowlist)
    if op_selection_yaml_path != None and is_string(op_selection_yaml_path):
        extra_params.append("--op_selection_yaml_path")
        extra_params.append(op_selection_yaml_path)
    if enabled_backends != None and is_list(enabled_backends):
        extra_params.append("--backend_whitelist")
        extra_params.extend(enabled_backends)
    if get_enable_lightweight_dispatch():
        extra_params.append("--skip_dispatcher_op_registration")
    if static_dispatch_backend:
        extra_params.append("--static_dispatch_backend")
        extra_params.extend(static_dispatch_backend)
        backends = static_dispatch_backend
    else:
        backends = enabled_backends
    fb_xplat_genrule(
        name = name,
        default_outs = ["."],
        outs = get_aten_generated_files(backends),
        cmd = "$(exe //torchgen:gen) " + " ".join([
            "--source-path $(location //:aten_src_path)/aten/src/ATen",
            "--install_dir $OUT",
        ] + extra_params),
        visibility = visibility,
        compatible_with = compatible_with,
    )

def get_aten_generated_files(enabled_backends):
    # NB: RegisterMeta counts as an optionally enabled backend,
    # and is intentionally omitted from here
    src_files = [
        "RegisterBackendSelect.cpp",
        "RegisterCompositeImplicitAutograd.cpp",
        "RegisterCompositeExplicitAutograd.cpp",
        "RegisterCompositeExplicitAutogradNonFunctional.cpp",
        "CompositeViewCopyKernels.cpp",
        "RegisterSchema.cpp",
        "Declarations.yaml",
        "Functions.cpp",
        "Functions.h",
        "RedispatchFunctions.h",
        "NativeFunctions.h",
        "NativeMetaFunctions.h",
        "MethodOperators.h",
        "FunctionalInverses.h",
        "Operators.h",
        "Operators_0.cpp",
        "Operators_1.cpp",
        "Operators_2.cpp",
        "Operators_3.cpp",
        "Operators_4.cpp",
        "CompositeImplicitAutogradFunctions.h",
        "CompositeImplicitAutogradFunctions_inl.h",
        "CompositeExplicitAutogradFunctions.h",
        "CompositeExplicitAutogradFunctions_inl.h",
        "CompositeExplicitAutogradNonFunctionalFunctions.h",
        "CompositeExplicitAutogradNonFunctionalFunctions_inl.h",
        "core/ATenOpList.cpp",
        "core/TensorBody.h",
        "core/TensorMethods.cpp",
        "core/aten_interned_strings.h",
        "core/enum_tag.h",
    ] + get_aten_derived_type_srcs(enabled_backends)

    # This is tiresome.  A better strategy would be to unconditionally
    # generate these files, and then only actually COMPILE them depended
    # on the generated set.  C'est la vie...
    if "CPU" in enabled_backends:
        src_files.extend(aten_ufunc_generated_cpu_sources())
        src_files.extend(aten_ufunc_generated_cpu_kernel_sources())
    if "CUDA" in enabled_backends:
        # Cannot unconditionally include this, because in the Edge selective
        # build CUDA is not enabled and thus the ufunc codegen for CUDA gets
        # skipped
        src_files.extend(aten_ufunc_generated_cuda_sources())

    res = {}
    for file_name in src_files:
        res[file_name] = [file_name]
    return res

def get_template_registration_file_rules(rule_name):
    rules = []
    for file_path in TEMPLATE_SOURCE_LIST:
        rules.append(":{}[{}]".format(rule_name, file_path))
    for file_path in aten_ufunc_generated_all_cpu_sources():
        rules.append(":{}[aten/src/ATen/{}]".format(rule_name, file_path))

    return rules

# Originally, there were two sets of codes in caffe2:aten_cpu, native codes and non-native.
# Now we have only non-naitve sources in aten_cpu. However, there are some aten related
# tests that may require both native and non-native codes. This rule is used to generate
# both aten_cpu and aten_native_cpu. They are using the same compilation setups.
def build_aten_cpu(name, srcs, deps = []):
    cxx_library(
        name = name,
        srcs = srcs,
        header_namespace = "",
        compiler_flags = get_pt_compiler_flags(),
        exported_preprocessor_flags = get_aten_preprocessor_flags(),
        link_whole = True,
        linker_flags = ["-Wl,--no-as-needed", "-ldl"],
        visibility = ["PUBLIC"],
        deps = [
            "//third_party:cpuinfo",
            "//third_party:glog",
            "//third_party:XNNPACK",
            #"//third_party/linker_lib:omp",
        ],
        exported_deps = [
            "//third_party:fmt",
            "//aten/src/ATen/native/quantized/cpu/qnnpack:pytorch_qnnpack",
            "//c10:c10",
            ":aten_header",
            ":caffe2_headers",
            ":common_core",
            ":generated_aten_config_header",
            ":generated_aten_headers_cpu",
            ":jit_core_headers",
            ":pthreadpool",
            "//third_party:ruy_lib",
        ],
    )

######### selective build #########

def get_pt_ops_deps(name, deps, train = False, enforce_traced_op_list = False, enable_flatbuffer = False, **kwargs):
    if not get_build_from_deps_query():
        return deps
    pt_operator_registry(
        name,
        deps,
        train = train,
        enforce_traced_op_list = enforce_traced_op_list,
        enable_flatbuffer = enable_flatbuffer,
        **kwargs
    )
    return deps + [":" + name]

# pt_operator_registry is the method that defines the fb_xplat_cxx_library that contains
# code for all selected PyTorch Operators and kernel functions. This also includes
# operator registration into the dispatcher.
#
# template_select: bool: Indicates if template based selective build is enabled.
#
# enforce_traced_op_list: bool: Enforces that only new-style operator
# lists based on the all_mobile_model_configs.yaml file and tracing based selective
# build are used in this library.
#
# train: bool: Build this library for training (True) or inference only (False).
# If built for training, codegen for VariableType is also included.
#
# pt_allow_forced_schema_registration: Manually disables forced schema registration when set to false, Default is true.
# Only does anything when train=True and the app requires full jit then force_schema_registration needs to occur.
# As Federated Learning migrates to lite interpreter
# we can slowly turn off forced schema registration as it is useless space and floods the compatibility api
#
def pt_operator_registry(
        name,
        deps = [],
        train = False,
        labels = [],
        env = [],
        template_select = True,
        enforce_traced_op_list = False,
        pt_allow_forced_schema_registration = True,
        enable_flatbuffer = False,
        **kwargs):
    compatible_with = kwargs.get("compatible_with", [])
    code_gen_files = pt_operator_query_codegen(name, deps = deps, train = train, enforce_traced_op_list = enforce_traced_op_list, pt_allow_forced_schema_registration = pt_allow_forced_schema_registration, compatible_with = compatible_with)
    code_gen_srcs = code_gen_files["srcs"]

    lib_deps = [
        ":aten_cpu",
        ":torch_mobile_core",
        "//c10:c10",
        "//third_party:glog",
    ]

    #if train:
    #    lib_deps = lib_deps + ["fbsource//xplat/caffe2:torch_mobile_train"]

    exported_preprocessor_flags = get_aten_preprocessor_flags()
    exported_preprocessor_flags += kwargs.pop("exported_preprocessor_flags", [])
    if template_select:
        # In addition to the
        # original code-gen select, this option further filter more operators based on
        # compile-time calculation. Examples include prim ops and any other ops that were
        # not filtered out before. The purpose of this option is to reduce the production
        # size further. However, it may have less flexibility, especially for tests from
        # python, where the used operator list is not explicitly generated. If the tests
        # are for functionality but not for size, and it's difficult to maintain an explicit
        # operator list, it's suggested to turn this option off.
        exported_preprocessor_flags.append("-DTEMPLATE_SELECTIVE_BUILD")
    kwargs.pop("exported_headers", [])
    cxx_library(
        name = name,
        srcs = code_gen_srcs,
        linker_flags = [
            "-Wl,--no-as-needed",
            "-ldl",
        ],
        link_whole = True,
        soname = "libtorch-code-gen.$(ext)",
        compiler_flags = get_aten_compiler_flags(),
        platform_compiler_flags = get_cpukernel_avx2_flags(),
        platform_deps = get_cpukernel_avx2_deps(),
        header_namespace = "ATen",
        exported_headers = code_gen_files["headers"],
        exported_preprocessor_flags = exported_preprocessor_flags,
        headers = kwargs.pop("headers", []),
        deps = lib_deps + [
            "//third_party:XNNPACK",
        ],
        **kwargs
    )

def get_aten_derived_type_src_rules(aten_rule_name, enabled_backends):
    return [
        ":{}[{}]".format(aten_rule_name, "Register" + backend + ".cpp")
        for backend in enabled_backends
    ]

def get_aten_selective_cpp_rules(aten_rule_name, enabled_backends):
    return [
        ":{}[{}]".format(aten_rule_name, f)
        for f in ["RegisterCompositeImplicitAutograd.cpp", "RegisterCompositeExplicitAutograd.cpp", "RegisterCompositeExplicitAutogradNonFunctional.cpp", "RegisterSchema.cpp", "RegisterBackendSelect.cpp", "CompositeViewCopyKernels.cpp"]
    ] + get_aten_derived_type_src_rules(aten_rule_name, enabled_backends)

def get_aten_derived_type_srcs(enabled_backends):
    return [
        "Register" + derived_type + ".cpp"
        for derived_type in enabled_backends
    ] + [
        derived_type + "Functions.h"
        for derived_type in enabled_backends
        if derived_type in PT_BACKEND_HEADERS or derived_type in get_static_dispatch_backend()
    ] + [
        derived_type + "Functions_inl.h"
        for derived_type in enabled_backends
        if derived_type in PT_BACKEND_HEADERS or derived_type in get_static_dispatch_backend()
    ]

def pt_operator_query_codegen(name, deps = [], train = False, enforce_traced_op_list = False, pt_allow_forced_schema_registration = True, compatible_with = []):
    oplist_dir_name = name + "_pt_oplist"

    # @lint-ignore BUCKLINT
    fb_xplat_genrule(
        name = oplist_dir_name,
        cmd = ("$(exe //:gen_oplist) " +
               "--model_file_list_path $(@query_outputs 'attrfilter(labels, pt_operator_library, deps(set({deps})))') " +
               ("" if enforce_traced_op_list else "--allow_include_all_overloads ") +
               "--output_dir $OUT ").format(deps = " ".join(["\"{}\"".format(d) for d in deps])),
        outs = get_gen_oplist_outs(),
        default_outs = ["."],
        compatible_with = compatible_with,
    )

    # Aten files
    aten_genrule = name + "_aten"
    extra_flags = {
        "enabled_backends": USED_PT_BACKENDS,
        "op_selection_yaml_path": "$(location :{}[selected_operators.yaml])".format(oplist_dir_name),
    }

    if train and pt_allow_forced_schema_registration:
        extra_flags["force_schema_registration"] = True

    # if get_enable_lightweight_dispatch():
    #    unboxing_genrule = name + "_unboxing"
    #    gen_aten_unboxing_files(
    #        unboxing_genrule,
    #        extra_flags = extra_flags,
    #    )

    static_dispatch_backend = get_static_dispatch_backend()
    if static_dispatch_backend:
        extra_flags["static_dispatch_backend"] = static_dispatch_backend

    gen_aten_files(
        aten_genrule,
        extra_flags = extra_flags,
        compatible_with = compatible_with,
    )

    # unboxing_wrappers files
    extra_params = [
        "--operators_yaml_path",
        "$(location :" + oplist_dir_name + "[selected_operators.yaml])",
    ]
    unboxing_and_autograd_genrule = name + "_unboxing_and_autograd"
    gen_aten_libtorch_files(unboxing_and_autograd_genrule, extra_params, compatible_with)

    # Template runtime files (prim ops, etc)
    template_registration_genrule = name + "_template_registration"
    copy_template_registration_files(template_registration_genrule)

    srcs = get_aten_selective_cpp_rules(
        aten_genrule,
        static_dispatch_backend if static_dispatch_backend else USED_PT_BACKENDS,
    ) + get_template_registration_file_rules(
        template_registration_genrule,
    ) + ([
        ":{}[autograd/generated/VariableType_0.cpp]".format(unboxing_and_autograd_genrule),
        ":{}[autograd/generated/VariableType_1.cpp]".format(unboxing_and_autograd_genrule),
        ":{}[autograd/generated/VariableType_2.cpp]".format(unboxing_and_autograd_genrule),
        ":{}[autograd/generated/VariableType_3.cpp]".format(unboxing_and_autograd_genrule),
        ":{}[autograd/generated/VariableType_4.cpp]".format(unboxing_and_autograd_genrule),
        ":{}[autograd/generated/ADInplaceOrViewType_0.cpp]".format(unboxing_and_autograd_genrule),
        ":{}[autograd/generated/ADInplaceOrViewType_1.cpp]".format(unboxing_and_autograd_genrule),
    ] if train else []) + ([
        #":{}[SupportedMobileModelsRegistration.cpp]".format(oplist_dir_name),
    ])

    headers = {
        "selected_mobile_ops.h": ":{}[selected_mobile_ops.h]".format(oplist_dir_name),
    }

    # if get_enable_lightweight_dispatch():
    #     srcs.extend([
    #         ":{}[UnboxingFunctions_0.cpp]".format(unboxing_genrule),
    #         ":{}[UnboxingFunctions_1.cpp]".format(unboxing_genrule),
    #         ":{}[UnboxingFunctions_2.cpp]".format(unboxing_genrule),
    #         ":{}[UnboxingFunctions_3.cpp]".format(unboxing_genrule),
    #         ":{}[UnboxingFunctions_4.cpp]".format(unboxing_genrule),
    #         ":{}[RegisterCodegenUnboxedKernels_0.cpp]".format(unboxing_genrule),
    #         ":{}[RegisterCodegenUnboxedKernels_1.cpp]".format(unboxing_genrule),
    #         ":{}[RegisterCodegenUnboxedKernels_2.cpp]".format(unboxing_genrule),
    #         ":{}[RegisterCodegenUnboxedKernels_3.cpp]".format(unboxing_genrule),
    #         ":{}[RegisterCodegenUnboxedKernels_4.cpp]".format(unboxing_genrule),
    #         ":{}[RegisterCodegenUnboxedKernels_5.cpp]".format(unboxing_genrule),
    #         ":{}[RegisterCodegenUnboxedKernels_6.cpp]".format(unboxing_genrule),
    #         ":{}[RegisterCodegenUnboxedKernels_7.cpp]".format(unboxing_genrule),
    #         ":{}[RegisterCodegenUnboxedKernels_8.cpp]".format(unboxing_genrule),
    #         ":{}[RegisterCodegenUnboxedKernels_9.cpp]".format(unboxing_genrule),
    #     ])
    #     headers["UnboxingFunctions.h"] = ":{}[UnboxingFunctions.h]".format(unboxing_genrule)
    return {"headers": headers, "srcs": srcs}

def gen_aten_libtorch_files(name, extra_params = [], compatible_with = []):
    fb_xplat_genrule(
        name = name,
        outs = get_generate_code_bin_outs(),
        default_outs = ["."],
        cmd = "mkdir -p tools && " +
              "$(exe //tools/setup_helpers:generate_code_bin) " + " ".join(
            # Mobile build only needs libtorch - skip python bindings for now, except
            # for ovrsource, which needs Python bindings.
            (["--subset libtorch"] if not is_arvr_mode() else []) + [
                "--native-functions-path $(location :aten_src_path)/aten/src/ATen/native/native_functions.yaml",
                "--tags-path $(location :aten_src_path)/aten/src/ATen/native/tags.yaml",  # todo D35992309
                "--install_dir $OUT",
            ] + extra_params,
        ),
        cmd_exe = "@powershell -Command New-Item -Path tools -ItemType Directory -Force; " +
                  "$(exe //tools/setup_helpers:generate_code_bin) " + " ".join(
            # Mobile build only needs libtorch - skip python bindings for now, except
            # for ovrsource, which needs Python bindings.
            (["--subset libtorch"] if not is_arvr_mode() else []) + [
                "--native-functions-path $(location :aten_src_path)/aten/src/ATen/native/native_functions.yaml",
                "--tags-path $(location :aten_src_path)/aten/src/ATen/native/tags.yaml",
                "--install_dir $OUT",
            ] + extra_params,
        ),
        compatible_with = compatible_with,
    )

def copy_template_registration_files(name):
    cmd = []
    cmd_exe = []

    template_source_dict = get_template_source_dict()

    # Ideally, we would run one copy command for a single source directory along
    # with all its child directories, but it's somewhat hard to know if a directory
    # is a child of another just bu looking at the metadata (directory relative
    # path) that we currently have since 1 directory could look like a parent of
    # another and yet come from a different filegroup() rule.
    #
    for (path_prefix, file_paths) in template_source_dict.items():
        cmd.append("mkdir -p $OUT/{}".format(path_prefix))
        cmd_exe.append("md $OUT/{}".format(path_prefix))

        # Adding *.cpp is a workaround to prevent cp from thrown an error when it
        # encounters a directory (since -r was not specified). If files with an
        # extension other than .cpp need to be copied, then the command below
        # will not work and will need to be updated.
        #
        cmd.append("cp -f {0}/{1}/*.cpp $OUT/{1}/".format("$(location :templated_selective_build_srcs)", path_prefix))
        cmd_exe.append("robocopy /E {0}/{1} $OUT/{1}".format("$(location :templated_selective_build_srcs)", path_prefix))

    cmd.append("mkdir -p $OUT/aten/src/ATen")
    cmd_exe.append("md $OUT/aten/src/ATen")

    # NB: CUDA is skipped here because this is selective build and CUDA is not
    # supported for selective build
    for ufunc_file in aten_ufunc_generated_all_cpu_sources("$(location :gen_aten[{}])"):
        cmd.append("cp -f " + ufunc_file + " $OUT/aten/src/ATen")
        cmd_exe.append("copy " + ufunc_file + " $OUT/aten/src/ATen")

    fb_xplat_genrule(
        name = name,
        cmd = " && ".join(cmd),
        cmd_exe = "@powershell -Command " + ("; ".join(cmd_exe)),
        outs = get_template_registration_files_outs(),
        default_outs = ["."],
    )

def pt_operator_library(
        name,
        ops = [],
        exported_deps = [],
        check_decl = True,
        train = False,
        model = None,
        include_all_operators = False,
        **kwargs):
    model_name = name

    if get_build_from_deps_query():
        ops = [op.strip() for op in ops]

        # If ops are specified, then we are in static selective build mode, so we append
        # base ops to this list to avoid additional special case logic in subsequent code.
        if len(ops) > 0:
            ops.extend(PT_BASE_OPS)

        visibility = kwargs.pop("visibility", ["PUBLIC"])

        fb_xplat_genrule(
            name = name,
            out = "model_operators.yaml",
            cmd = (
                "$(exe :gen_operators_yaml) " +
                "{optionally_root_ops} " +
                "{optionally_training_root_ops} " +
                "--rule_name {rule_name} " +
                "--output_path \"${{OUT}}\" " +
                "--model_name {model_name} " +
                "--dep_graph_yaml_path pytorch_op_deps.yaml " +
                "--models_yaml_path all_mobile_model_configs.yaml " +
                #"{optionally_model_versions} " +
                #"{optionally_model_assets} " +
                #"{optionally_model_traced_backends} " +
                "{optionally_include_all_operators}"
            ).format(
                rule_name = name,
                model_name = model_name,
                optionally_root_ops = "--root_ops " + (",".join(ops)) if len(ops) > 0 else "",
                optionally_training_root_ops = "--training_root_ops " + (",".join(ops)) if len(ops) > 0 and train else "",
                #optionally_model_versions = "--model_versions " + (",".join(model_versions)) if model_versions != None else "",
                #optionally_model_assets = "--model_assets " + (",".join(model_assets)) if model_assets != None else "",
                #optionally_model_traced_backends = "--model_traced_backends " + (",".join(model_traced_backends)) if model_traced_backends != None else "",
                optionally_include_all_operators = "--include_all_operators " if include_all_operators else "",
            ),
            labels = ["pt_operator_library"],  # for pt_operator_query_codegen query
            visibility = visibility,
            **kwargs
        )
    else:
        if check_decl:
            pass
            # ensure_ops_are_declared(ops)

        cxx_library(
            name = name,
            compiler_flags = get_pt_compiler_flags(),
            cxx_platform_compiler_flags = get_cpukernel_avx2_flags(),
            exported_deps = exported_deps,
            **kwargs
        )

def compose_platform_setting_list(settings):
    """Settings object:
    os/cpu pair: should be valid key, or at most one part can be wildcard.
    flags: the values added to the compiler flags
    """
    result = []
    for setting in settings:
        result = result.append([
            "^{}-{}$".format(setting["os"], setting["cpu"]),
            setting["flags"],
        ])
    return result

def get_cpukernel_avx2_flags():
    # flags = compose_platform_setting_list([
    #     {
    #         "cpu": "x86_64",
    #         "flags": ["-DHAVE_AVX2_CPU_DEFINITION"],
    #         "os": "macosx",
    #     },
    # ]) if build_cpukernel_avx2() else []
    return []

def build_cpukernel_avx2():
    return not is_arvr_mode()

def get_cpukernel_avx2_deps():
    # flags = compose_platform_setting_list([
    #     {
    #         "cpu": "x86_64",
    #         "flags": ["fbsource//xplat/caffe2:cpukernel_avx2"],
    #         "os": "macosx",
    #     },
    # ]) if build_cpukernel_avx2() else []
    return []
