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
  explicit OperatorRegistrar(FunctionSchema&& schema, OperatorOptions&& operatorOptions, c10::optional<TensorTypeId> dispatch_key, c10::optional<KernelFunction> kernel)
  : op_(Dispatcher::singleton().registerSchema(std::move(schema), std::move(operatorOptions))), kernel_registration_handle_(c10::nullopt) {
    if (kernel.has_value()) {
      TORCH_INTERNAL_ASSERT(kernel->isValid());
      if (dispatch_key.has_value()) {
        kernel_registration_handle_ = Dispatcher::singleton().registerKernel(op_.opHandle(), *dispatch_key, std::move(*kernel));
      } else {
        kernel_registration_handle_ = Dispatcher::singleton().registerCatchallKernel(op_.opHandle(), std::move(*kernel));
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

void RegisterOperators::checkSchemaAndRegisterOp_(Options&& options) {
  if (options.legacyATenSchema_.has_value()) {
    // Ignore legacy aten operators, don't add them to c10
    return;
  }

  TORCH_CHECK(options.schemaOrName_.has_value(), "In operator registration: Tried to register an operator without specifying a schema or operator name.");
  if (options.schemaOrName_->is_right()) {
    // schema was explicitly specified. Check it matches the inferred one and register the op.

    const FunctionSchema& schema = options.schemaOrName_->right();
    TORCH_CHECK(
        options.aliasAnalysisKind_ == AliasAnalysisKind::FROM_SCHEMA ||
            !schema.hasAnyAliasInfo(),
        "In operator registration: Tried to register operator ",
        options.schemaOrName_->right(),
        " with aliasing information in the schema but without AliasAnalysisKind::FROM_SCHEMA.");

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

    checkNoDuplicateKernels_(options);

    registerOp_(std::move(options));
  } else {
    // schema wasn't explicitly specified. Take the inferred schema for registering the op.

    OperatorName name = std::move(*options.schemaOrName_).left();
    FunctionSchema inferred_schema = inferSchemaFromKernels_(name, options);

    options.schemaOrName_ = c10::make_right<OperatorName, FunctionSchema>(
      std::move(name.name),
      std::move(name.overload_name),
      inferred_schema.arguments(),
      inferred_schema.returns(),
      inferred_schema.is_vararg(),
      inferred_schema.is_varret()
    );

    checkNoDuplicateKernels_(options);

    // This would have unexpected behavior since an inferred schema will not
    // have aliasing annotations.
    TORCH_CHECK(
        options.aliasAnalysisKind_ != AliasAnalysisKind::FROM_SCHEMA,
        "In operator registration: Tried to register operator ",
        options.schemaOrName_->right(),
        " with AliasAnalysisKind::FROM_SCHEMA, but the schema is inferred.");

    // Register all kernels with the schema we inferred
    registerOp_(std::move(options));
  }
}

c10::FunctionSchema RegisterOperators::inferSchemaFromKernels_(const OperatorName& opName, const RegisterOperators::Options& options) {
  TORCH_CHECK(options.kernels.size() > 0, "Cannot infer operator schema in registration of operator ", toString(opName), " because there is no kernel specified.");

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
  TORCH_CHECK(inferred_schema.has_value(), "Cannot infer operator schema for this kind of kernel in registration of operator ", toString(opName), ". Please explicitly specify the operator schema or specify at least one kernel for which we can infer the schema.");

  return *inferred_schema;
}

void RegisterOperators::checkNoDuplicateKernels_(const Options& options) {
  std::unordered_set<TensorTypeId> dispatch_keys;
  bool has_catchall_kernel = false;

  for (const auto& kernel : options.kernels) {
    if (kernel.dispatch_key.has_value()) {
      TORCH_CHECK(0 == dispatch_keys.count(*kernel.dispatch_key), "In operator registration: Tried to register multiple kernels with same dispatch key ", toString(*kernel.dispatch_key), " for operator schema ", toString(options.schemaOrName_->right()));
      dispatch_keys.insert(*kernel.dispatch_key);
    } else {
      TORCH_CHECK(!has_catchall_kernel, "In operator registration: Tried to register multiple catch-all kernels for operator schema " + toString(options.schemaOrName_->right()));
      has_catchall_kernel = true;
    }
  }
}

void RegisterOperators::registerOp_(Options&& options) {
  FunctionSchema schema = std::move(*options.schemaOrName_).right();
  OperatorName op_name = schema.operator_name();

  auto operatorOptions = makeOperatorOptions_(options);

  if (0 == options.kernels.size()) {
    registerSchemaOnly_(std::move(schema), std::move(operatorOptions));
  } else {
    for (auto& kernel : options.kernels) {
      registerSchemaAndKernel_(schema, std::move(kernel), std::move(operatorOptions));
    }
  }

  TORCH_INTERNAL_ASSERT(c10::Dispatcher::singleton().findSchema(op_name).has_value());
}

OperatorOptions RegisterOperators::makeOperatorOptions_(const RegisterOperators::Options& options) {
  OperatorOptions result;
  if (options.aliasAnalysisKind_.has_value()) {
    result.setAliasAnalysis(*options.aliasAnalysisKind_);
  }
  return result;
}

void RegisterOperators::registerSchemaAndKernel_(FunctionSchema schema, Options::KernelRegistrationConfig&& kernel, OperatorOptions&& operatorOptions) {
  TORCH_INTERNAL_ASSERT(kernel.func.isValid(), "Kernel must be set");

  registrars_.emplace_back(std::move(schema), std::move(operatorOptions), kernel.dispatch_key, std::move(kernel.func));
}

void RegisterOperators::registerSchemaOnly_(FunctionSchema&& schema, OperatorOptions&& operatorOptions) {
  registrars_.emplace_back(std::move(schema), std::move(operatorOptions), c10::nullopt, c10::nullopt);
}

RegisterOperators::RegisterOperators() = default;
RegisterOperators::~RegisterOperators() = default;
RegisterOperators::RegisterOperators(RegisterOperators&&) noexcept = default;
RegisterOperators& RegisterOperators::operator=(RegisterOperators&&) noexcept = default;

}
