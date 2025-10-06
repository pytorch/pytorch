#include <c10/macros/Macros.h>

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/op_registration/op_allowlist.h>
#include <ATen/core/op_registration/op_registration.h>
#if !defined(CAFFE2_IS_XPLAT_BUILD)
#include <torch/csrc/jit/frontend/function_schema_parser.h>
#endif

namespace c10 {
namespace impl {
void build_feature_required_feature_not_available(const char* feature) {
  TORCH_CHECK(
      false,
      "Required feature '" + std::string(feature) + "' is not available");
}
} // namespace impl

static_assert(std::is_nothrow_move_constructible_v<
              std::optional<RegistrationHandleRAII>>);
static_assert(std::is_nothrow_move_assignable_v<
              std::optional<RegistrationHandleRAII>>);

void RegisterOperators::checkSchemaAndRegisterOp_(Options&& options) {
  TORCH_CHECK(
      options.schemaOrName_.has_value(),
      "In operator registration: Tried to register an operator without specifying a schema or operator name.");
  if (options.schemaOrName_->index() == 1) {
    // schema was explicitly specified.

    checkNoDuplicateKernels_(options);

    registerOp_(std::move(options));
  } else {
    // schema wasn't explicitly specified. Take the inferred schema for
    // registering the op.

    OperatorName name =
        std::get<OperatorName>(std::move(*options.schemaOrName_));
    FunctionSchema inferred_schema = inferSchemaFromKernels_(name, options);

    options.schemaOrName_ = FunctionSchema(
        std::move(name.name),
        std::move(name.overload_name),
        inferred_schema.arguments(),
        inferred_schema.returns(),
        inferred_schema.is_vararg(),
        inferred_schema.is_varret());

    checkNoDuplicateKernels_(options);

    // This would have unexpected behavior since an inferred schema will not
    // have aliasing annotations.
    TORCH_CHECK(
        options.aliasAnalysisKind_ != AliasAnalysisKind::FROM_SCHEMA,
        "In operator registration: Tried to register operator ",
        std::get<FunctionSchema>(options.schemaOrName_.value()),
        " with AliasAnalysisKind::FROM_SCHEMA, but the schema is inferred.");

    // Register all kernels with the schema we inferred
    registerOp_(std::move(options));
  }
}

c10::FunctionSchema RegisterOperators::inferSchemaFromKernels_(
    const OperatorName& opName,
    const RegisterOperators::Options& options) {
  TORCH_CHECK(
      !options.kernels.empty(),
      "Cannot infer operator schema in registration of operator ",
      opName,
      " because there is no kernel specified.");

  std::optional<FunctionSchema> inferred_schema = std::nullopt;
  for (const auto& kernel : options.kernels) {
    if (nullptr != kernel.inferred_function_schema) {
      if (!inferred_schema.has_value()) {
        inferred_schema = *kernel.inferred_function_schema;
        break;
      }
    }
  }
  TORCH_CHECK(
      inferred_schema.has_value(),
      "Cannot infer operator schema for this kind of kernel in registration of operator ",
      opName,
      ". Please explicitly specify the operator schema or specify at least one kernel for which we can infer the schema.");

  return *inferred_schema;
}

void RegisterOperators::checkNoDuplicateKernels_(const Options& options) {
  std::unordered_set<DispatchKey> dispatch_keys;
  bool has_catchall_kernel = false;

  for (const auto& kernel : options.kernels) {
    if (kernel.dispatch_key.has_value()) {
      TORCH_CHECK(
          0 == dispatch_keys.count(*kernel.dispatch_key),
          "In operator registration: Tried to register multiple kernels with same dispatch key ",
          *kernel.dispatch_key,
          " for operator schema ",
          toString(std::get<FunctionSchema>(options.schemaOrName_.value())));
      dispatch_keys.insert(*kernel.dispatch_key);
    } else {
      TORCH_CHECK(
          !has_catchall_kernel,
          "In operator registration: Tried to register multiple catch-all kernels for operator schema ",
          toString(std::get<FunctionSchema>(options.schemaOrName_.value())));
      has_catchall_kernel = true;
    }
  }
}

void RegisterOperators::registerOp_(Options&& options) {
  FunctionSchema schema =
      std::get<FunctionSchema>(std::move(options.schemaOrName_.value()));

  // HACK: bong in the alias analysis kind from the legacy API directly
  // into schema
  if (options.aliasAnalysisKind_.has_value()) {
    schema.setAliasAnalysis(*options.aliasAnalysisKind_);
  }

  OperatorName op_name = schema.operator_name();

  registrars_.emplace_back(Dispatcher::singleton().registerDef(
      std::move(schema), "registered by RegisterOperators"));

  for (auto& kernel : options.kernels) {
    registrars_.emplace_back(Dispatcher::singleton().registerImpl(
        op_name,
        kernel.dispatch_key,
        std::move(kernel.func),
        kernel.cpp_signature,
        std::move(kernel.inferred_function_schema),
        "registered by RegisterOperators"));
  }
}

} // namespace c10
