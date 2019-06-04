#include <ATen/core/op_registration/op_registration.h>
#if !defined(CAFFE2_IS_XPLAT_BUILD)
#include <torch/csrc/jit/script/function_schema_parser.h>
#endif

namespace c10 {

static_assert(std::is_nothrow_move_constructible<c10::optional<RegistrationHandleRAII>>::value, "");
static_assert(std::is_nothrow_move_assignable<c10::optional<RegistrationHandleRAII>>::value, "");

// OperatorRegistrar in its constructor registers an operator in the dispatch
// table deregisters it in the destructor.
class RegisterOperators::OperatorRegistrar final {
public:
  explicit OperatorRegistrar(FunctionSchema&& schema, OperatorOptions&& operatorOptions, c10::optional<TensorTypeId> dispatch_key, KernelFunction* kernel, KernelCacheCreatorFunction&& cache_creator)
  : op_(Dispatcher::singleton().registerSchema(std::move(schema), std::move(operatorOptions))), kernel_registration_handle_(c10::nullopt) {
    // either both, kernel and cache_creator, or none must be set.
    TORCH_INTERNAL_ASSERT((kernel != nullptr) == static_cast<bool>(cache_creator));

    if (kernel != nullptr) {
      if (dispatch_key.has_value()) {
        kernel_registration_handle_ = Dispatcher::singleton().registerKernel(op_.opHandle(), *dispatch_key, kernel, std::move(cache_creator));
      } else {
        kernel_registration_handle_ = Dispatcher::singleton().registerCatchallKernel(op_.opHandle(), kernel, std::move(cache_creator));
      }
    }
  }

  OperatorRegistrar(OperatorRegistrar&& rhs) noexcept = default;
  OperatorRegistrar& operator=(OperatorRegistrar&& rhs) noexcept = default;

  // not needed and would break RAII if defaulted.
  OperatorRegistrar(const OperatorRegistrar& rhs) = delete;
  OperatorRegistrar& operator=(const OperatorRegistrar& rhs) = delete;

private:
  c10::SchemaRegistrationHandleRAII op_;
  c10::optional<RegistrationHandleRAII> kernel_registration_handle_;
};

void RegisterOperators::checkSchemaAndRegisterOp_(const std::string& schemaOrNameStr, Options&& options) {
  #if defined(CAFFE2_IS_XPLAT_BUILD)
    throw std::logic_error("Tried to register operator " + schemaOrNameStr + ". We don't support registering c10 ops on mobile yet because the function schema parser isn't present in the mobile build.");
  #else
    either<OperatorName, FunctionSchema> schemaOrName = torch::jit::parseSchemaOrName(schemaOrNameStr);
    if (schemaOrName.is_right()) {
      // schema was explicitly specified. Check it matches the inferred one and register the op.
      checkSchemaAndRegisterOp_(std::move(schemaOrName).right(), std::move(options));
    } else {
      // schema wasn't explicitly specified. Take the inferred schema for registering the op.

      FunctionSchema inferred_schema = inferSchemaFromKernels_(schemaOrNameStr, options);
      OperatorName name = std::move(schemaOrName).left();
      FunctionSchema inferred_schema_with_name(
        std::move(name.name),
        std::move(name.overload_name),
        inferred_schema.arguments(),
        inferred_schema.returns(),
        inferred_schema.is_vararg(),
        inferred_schema.is_varret()
      );

      checkNoDuplicateKernels_(inferred_schema_with_name, options);

      // Register all kernels with the schema we inferred
      registerOp_(std::move(inferred_schema_with_name), std::move(options));
    }
  #endif
}

c10::FunctionSchema RegisterOperators::inferSchemaFromKernels_(const std::string& opNameStr, const RegisterOperators::Options& options) {
  TORCH_CHECK(options.kernels.size() > 0, "Cannot infer operator schema in registration of operator ", opNameStr, " because there is no kernel specified.");

  c10::optional<FunctionSchema> inferred_schema = c10::nullopt;
  for (const auto& kernel : options.kernels) {
    if (nullptr != kernel.inferred_function_schema.get()) {
      if (inferred_schema.has_value()) {
        c10::optional<std::string> schema_difference = findSchemaDifferences(*inferred_schema, *kernel.inferred_function_schema);
        if (schema_difference.has_value()) {
          TORCH_CHECK(false, "In operator registration: Tried to register kernels for same operator that infer a different function schema: [", toString(*inferred_schema), "] ",
                   "doesn't match [", toString(*kernel.inferred_function_schema), "]. ",
                   *schema_difference);
        }
      } else {
        inferred_schema = *kernel.inferred_function_schema;
      }
    }
  }
  TORCH_CHECK(inferred_schema.has_value(), "Cannot infer operator schema for this kind of kernel in registration of operator ", opNameStr,". Please explicitly specify the operator schema or specify at least one kernel for which we can infer the schema.");

  return *inferred_schema;
}

void RegisterOperators::checkSchemaAndRegisterOp_(FunctionSchema schema, Options&& options) {
  for (auto& kernel : options.kernels) {
    if (nullptr != kernel.inferred_function_schema.get()) {
      c10::optional<std::string> schema_difference = findSchemaDifferences(schema, *kernel.inferred_function_schema);
      if (schema_difference.has_value()) {
        TORCH_CHECK(false, "In operator registration: Specified function schema [", toString(schema), "] ",
                 "doesn't match inferred function schema [", toString(*kernel.inferred_function_schema), "]. ",
                 *schema_difference);
      }
    }
  }

  checkNoDuplicateKernels_(schema, options);

  registerOp_(std::move(schema), std::move(options));
}

void RegisterOperators::checkNoDuplicateKernels_(const FunctionSchema& schema, const Options& options) {
  std::unordered_set<TensorTypeId> dispatch_keys;
  bool has_catchall_kernel = false;

  for (const auto& kernel : options.kernels) {
    if (kernel.dispatch_key.has_value()) {
      TORCH_CHECK(0 == dispatch_keys.count(*kernel.dispatch_key), "In operator registration: Tried to register multiple kernels with same dispatch key " + detail::dispatch_key_to_string(*kernel.dispatch_key) + " for operator schema " + toString(schema));
      dispatch_keys.insert(*kernel.dispatch_key);
    } else {
      TORCH_CHECK(!has_catchall_kernel, "In operator registration: Tried to register multiple catch-all kernels for operator schema " + toString(schema));
      has_catchall_kernel = true;
    }
  }
}

void RegisterOperators::registerOp_(FunctionSchema&& schema, Options&& options) {
  std::string op_name = schema.name();
  std::string overload_name = schema.overload_name();

  auto operatorOptions = makeOperatorOptions_(options);

  if (0 == options.kernels.size()) {
    registerSchemaOnly_(std::move(schema), std::move(operatorOptions));
  } else {
    for (auto& kernel : options.kernels) {
      registerSchemaAndKernel_(schema, std::move(kernel), std::move(operatorOptions));
    }
  }

  auto op_handle = c10::Dispatcher::singleton().findSchema(op_name.c_str(), overload_name.c_str()).value();
}

OperatorOptions RegisterOperators::makeOperatorOptions_(const RegisterOperators::Options& options) {
  OperatorOptions result;
  if (options.aliasAnalysisKind_.has_value()) {
    result.setAliasAnalysis(*options.aliasAnalysisKind_);
  }
  return result;
}

void RegisterOperators::registerSchemaAndKernel_(FunctionSchema schema, Options::KernelRegistrationConfig&& kernel, OperatorOptions&& operatorOptions) {
  TORCH_INTERNAL_ASSERT(kernel.kernel_func != nullptr && static_cast<bool>(kernel.cache_creator_func), "Kernel must be set");

  registrars_.emplace_back(std::move(schema), std::move(operatorOptions), kernel.dispatch_key, kernel.kernel_func, std::move(kernel.cache_creator_func));
}

void RegisterOperators::registerSchemaOnly_(FunctionSchema&& schema, OperatorOptions&& operatorOptions) {
  registrars_.emplace_back(std::move(schema), std::move(operatorOptions), c10::nullopt, nullptr, nullptr);
}

RegisterOperators::RegisterOperators() = default;
RegisterOperators::~RegisterOperators() = default;
RegisterOperators::RegisterOperators(RegisterOperators&&) noexcept = default;
RegisterOperators& RegisterOperators::operator=(RegisterOperators&&) noexcept = default;

}
