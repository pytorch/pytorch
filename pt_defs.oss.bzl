load("//tools/build_defs:fb_xplat_cxx_library.bzl", "fb_xplat_cxx_library")
load("//tools/build_defs:fb_xplat_genrule.bzl", "fb_xplat_genrule")
load(
    ":buckbuild.bzl",
    "get_pt_operator_registry_dict",
)

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

######### selective build #########

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
    args = get_pt_operator_registry_dict(
        name,
        deps,
        train,
        labels,
        env,
        template_select,
        enforce_traced_op_list,
        pt_allow_forced_schema_registration,
        enable_flatbuffer = True,
        **kwargs
    )

    fb_xplat_cxx_library(
        name = name,
        **args
    )

def get_pt_ops_deps(name, deps, train = False, enforce_traced_op_list = False, enable_flatbuffer = False, **kwargs):
    pt_operator_registry(
        name,
        deps,
        train = train,
        enforce_traced_op_list = enforce_traced_op_list,
        enable_flatbuffer = enable_flatbuffer,
        **kwargs
    )
    return deps + [":" + name]

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
