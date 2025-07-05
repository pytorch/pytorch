#include <c10/macros/Macros.h>

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/op_registration/op_allowlist.h>
#include <ATen/core/op_registration/op_registration.h>
#include <regex>
#include <algorithm>
#include <string>
#include <c10/util/env.h>
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
    if (nullptr != kernel.inferred_function_schema.get()) {
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

// This map is used to enable users to use PYTORCH_CPU_FALLBACK_OPS=all. There are
// some ops that cannot fallback to CPU, these ops are listed here to prevent cases
// where such ops are un-registered.
// TODO: Find programmatic way to generate this list
static std::unordered_set<std::string> mustRegisterOpNames{
  // View Ops //
  "view",
  "alias",
  "_reshape_alias",
  "_unsafe_view",
  "as_strided",
  "unfold",
  "slice.Tensor",
  // Tensor Ops //
  "slice_backward",
  "slice_copy.Tensor",
  "resize_",
  "set_.source_Tensor",
  "set_.source_Storage",
  "set_.source_Storage_storage_offset",
  "empty_strided",
  "empty.memory_format",
  "eye.out",
  "eye.m_out",
  "_copy_from",
  "_copy_from_and_resize",
  "record_stream",
  "_to_cpu"
  // Misc Ops //
  "abs.out",
  "logical_not.out",
  "native_layer_norm",};

static bool matchRegexList(
  const std::string& str,
  const std::vector<std::string>& regexList) {
  auto anyOfCondition = [&str](const std::string& regex) {
    try {
      std::regex filterRegex(regex);
      return std::regex_search(str, filterRegex);
    } catch (const std::regex_error& e) {
      std::cerr << "Regex error: " << e.what() << " for pattern: " << regex << "\n";
      return false;
    }
  };

  return std::any_of(regexList.begin(), regexList.end(), anyOfCondition);
}
bool registerOp(const std::string& opName) {
  auto fallbackOpsStr = c10::utils::get_env("PYTORCH_CPU_FALLBACK_OPS");
  if (!fallbackOpsStr.has_value()) {
    return true;
  }

  bool must_register = mustRegisterOpNames.find(opName) != mustRegisterOpNames.end();

  static std::vector<std::string> fallbackOps; // Declare fallbackOps
  if (fallbackOpsStr.has_value() && fallbackOps.empty()) {
    std::string str = fallbackOpsStr.value();
    std::stringstream ss(str);
    std::string token;

    while (std::getline(ss, token, ',')) {
      // Trim whitespace
      token.erase(0, token.find_first_not_of(" \t\n\r\f\v"));
      token.erase(token.find_last_not_of(" \t\n\r\f\v") + 1);
      if (!token.empty()) {
        fallbackOps.push_back(token);
      }
    }
  }
  static bool fallbackAllOps =
      std::find(fallbackOps.begin(), fallbackOps.end(), "all") !=
      fallbackOps.end();
  if (fallbackAllOps) {
    return must_register;
  }
  return must_register || !matchRegexList(opName, fallbackOps);
}

} // namespace c10
