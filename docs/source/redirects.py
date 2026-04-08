redirects = {
    # Redirects for deprecated TorchScript documentation
    "jit": "torch.compiler_api.html",
    "jit_language_reference": "torch.compiler_api.html",
    "jit_language_reference_v2": "torch.compiler_api.html",
    "jit_python_reference": "torch.compiler_api.html",
    "jit_unsupported": "torch.compiler_api.html",
    "jit_builtin_functions": "torch.compiler_api.html",
    # Redirects for documents moved from source/ to source/user_guide/torch_compiler/
    "torch.compiler": "user_guide/torch_compiler/torch.compiler.html",
    "torch.compiler.config": "user_guide/torch_compiler/torch.compiler.config.html",
    "torch.compiler_aot_inductor": "user_guide/torch_compiler/torch.compiler_aot_inductor.html",
    "torch.compiler_aot_inductor_debugging_guide": (
        "user_guide/torch_compiler/torch.compiler_aot_inductor_debugging_guide.html"
    ),
    "torch.compiler_aot_inductor_minifier": (
        "user_guide/torch_compiler/torch.compiler_aot_inductor_minifier.html"
    ),
    "torch.compiler_backward": "user_guide/torch_compiler/torch.compiler_backward.html",
    "torch.compiler_cudagraph_trees": (
        "user_guide/torch_compiler/torch.compiler_cudagraph_trees.html"
    ),
    "torch.compiler_custom_backends": (
        "user_guide/torch_compiler/torch.compiler_custom_backends.html"
    ),
    "torch.compiler_dynamic_shapes": (
        "user_guide/torch_compiler/torch.compiler_dynamic_shapes.html"
    ),
    "torch.compiler_dynamo_deepdive": (
        "user_guide/torch_compiler/torch.compiler_dynamo_deepdive.html"
    ),
    "torch.compiler_dynamo_overview": (
        "user_guide/torch_compiler/torch.compiler_dynamo_overview.html"
    ),
    "torch.compiler_fake_tensor": (
        "user_guide/torch_compiler/torch.compiler_fake_tensor.html"
    ),
    "torch.compiler_faq": "user_guide/torch_compiler/torch.compiler_faq.html",
    "torch.compiler_fine_grain_apis": (
        "user_guide/torch_compiler/torch.compiler_fine_grain_apis.html"
    ),
    "torch.compiler_get_started": (
        "user_guide/torch_compiler/torch.compiler_get_started.html"
    ),
    "torch.compiler_inductor_profiling": (
        "user_guide/torch_compiler/torch.compiler_inductor_profiling.html"
    ),
    "torch.compiler_inductor_provenance": (
        "user_guide/torch_compiler/torch.compiler_inductor_provenance.html"
    ),
    "torch.compiler_ir": "user_guide/torch_compiler/torch.compiler_ir.html",
    "torch.compiler_nn_module": "user_guide/torch_compiler/torch.compiler_nn_module.html",
    "torch.compiler_performance_dashboard": (
        "user_guide/torch_compiler/torch.compiler_performance_dashboard.html"
    ),
    "torch.compiler_profiling_torch_compile": (
        "user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html"
    ),
    "torch.compiler_transformations": (
        "user_guide/torch_compiler/torch.compiler_transformations.html"
    ),
    "torch.compiler_troubleshooting": (
        "user_guide/torch_compiler/torch.compiler_troubleshooting.html"
    ),
    # Redirects for export documents moved from source/ to source/user_guide/torch_compiler/
    "export": "user_guide/torch_compiler/export.html",
    "export/api_reference": "user_guide/torch_compiler/export/api_reference.html",
    "export/draft_export": "user_guide/torch_compiler/export/draft_export.html",
    "export/ir_spec": "user_guide/torch_compiler/export/ir_spec.html",
    "export/joint_with_descriptors": (
        "user_guide/torch_compiler/export/joint_with_descriptors.html"
    ),
    "export/programming_model": (
        "user_guide/torch_compiler/export/programming_model.html"
    ),
    "export/pt2_archive": "user_guide/torch_compiler/export/pt2_archive.html",
    "cond": "higher_order_ops/cond.html",
    # Redirects for compile documents moved from source/compile/ to
    # source/user_guide/torch_compiler/compile/
    "compile/dynamic_shapes_advanced_control_options": (
        "user_guide/torch_compiler/compile/dynamic_shapes_advanced_control_options.html"
    ),
    "compile/dynamic_shapes_backed_unbacked": (
        "user_guide/torch_compiler/compile/dynamic_shapes_backed_unbacked.html"
    ),
    "compile/dynamic_shapes_beyond_the_basics": (
        "user_guide/torch_compiler/compile/dynamic_shapes_beyond_the_basics.html"
    ),
    "compile/dynamic_shapes_core_concepts": (
        "user_guide/torch_compiler/compile/dynamic_shapes_core_concepts.html"
    ),
    "compile/dynamic_shapes_debugging_tlparse_torch_logs": (
        "user_guide/torch_compiler/compile/"
        "dynamic_shapes_debugging_tlparse_torch_logs.html"
    ),
    "compile/dynamic_shapes_troubleshooting": (
        "user_guide/torch_compiler/compile/dynamic_shapes_troubleshooting.html"
    ),
    "compile/dynamic_shapes_troubleshooting_guardon_errors": (
        "user_guide/torch_compiler/compile/"
        "dynamic_shapes_troubleshooting_guardon_errors.html"
    ),
    "compile/dynamic_shapes_zero_one_specialization": (
        "user_guide/torch_compiler/compile/dynamic_shapes_zero_one_specialization.html"
    ),
    "compile/programming_model": (
        "user_guide/torch_compiler/compile/programming_model.html"
    ),
    "compile/programming_model.common_graph_breaks": (
        "user_guide/torch_compiler/compile/programming_model.common_graph_breaks.html"
    ),
    "compile/programming_model.compiler_disable": (
        "user_guide/torch_compiler/compile/programming_model.compiler_disable.html"
    ),
    "compile/programming_model.custom_ops": (
        "user_guide/torch_compiler/compile/programming_model.custom_ops.html"
    ),
    "compile/programming_model.dynamo_core_concepts": (
        "user_guide/torch_compiler/compile/programming_model.dynamo_core_concepts.html"
    ),
    "compile/programming_model.dynamo_nonstrict_trace": (
        "user_guide/torch_compiler/compile/programming_model.dynamo_nonstrict_trace.html"
    ),
    "compile/programming_model.error_on_graph_break": (
        "user_guide/torch_compiler/compile/programming_model.error_on_graph_break.html"
    ),
    "compile/programming_model.fullgraph_false": (
        "user_guide/torch_compiler/compile/programming_model.fullgraph_false.html"
    ),
    "compile/programming_model.fullgraph_true": (
        "user_guide/torch_compiler/compile/programming_model.fullgraph_true.html"
    ),
    "compile/programming_model.graph_breaks_index": (
        "user_guide/torch_compiler/compile/programming_model.graph_breaks_index.html"
    ),
    "compile/programming_model.nested_graph_breaks": (
        "user_guide/torch_compiler/compile/programming_model.nested_graph_breaks.html"
    ),
    "compile/programming_model.non_strict_tracing_model": (
        "user_guide/torch_compiler/compile/"
        "programming_model.non_strict_tracing_model.html"
    ),
    "compile/programming_model.observability": (
        "user_guide/torch_compiler/compile/programming_model.observability.html"
    ),
    "compile/programming_model.recompilation": (
        "user_guide/torch_compiler/compile/programming_model.recompilation.html"
    ),
    "compile/programming_model.reporting_issues": (
        "user_guide/torch_compiler/compile/programming_model.reporting_issues.html"
    ),
    "compile/programming_model.skipped_functions": (
        "user_guide/torch_compiler/compile/programming_model.skipped_functions.html"
    ),
    "compile/programming_model.where_to_apply_compile": (
        "user_guide/torch_compiler/compile/"
        "programming_model.where_to_apply_compile.html"
    ),
}
