#include <ATen/core/op_registration/op_registration.h>
#if !defined(CAFFE2_IS_XPLAT_BUILD)
#include <torch/csrc/jit/frontend/function_schema_parser.h>
#endif

namespace c10 {

static_assert(std::is_nothrow_move_constructible<c10::optional<RegistrationHandleRAII>>::value, "");
static_assert(std::is_nothrow_move_assignable<c10::optional<RegistrationHandleRAII>>::value, "");

// OperatorRegistrar in its constructor registers an operator in the dispatch
// table deregisters it in the destructor.
class RegisterOperators::OperatorRegistrar final {
public:
  explicit OperatorRegistrar(FunctionSchema&& schema, c10::optional<DispatchKey> dispatch_key, c10::optional<KernelFunction> kernel)
  : op_(Dispatcher::singleton().registerSchema(std::move(schema))), kernel_registration_handle_(c10::nullopt) {
    if (kernel.has_value()) {
      TORCH_INTERNAL_ASSERT(kernel->isValid());
      kernel_registration_handle_ = Dispatcher::singleton().registerKernel(op_.second, dispatch_key, std::move(*kernel));
    }
  }

  OperatorRegistrar(OperatorRegistrar&& rhs) noexcept = default;
  OperatorRegistrar& operator=(OperatorRegistrar&& rhs) noexcept = default;

  // not needed and would break RAII if defaulted.
  OperatorRegistrar(const OperatorRegistrar& rhs) = delete;
  OperatorRegistrar& operator=(const OperatorRegistrar& rhs) = delete;

private:
  std::pair<RegistrationHandleRAII, OperatorHandle> op_;
  c10::optional<RegistrationHandleRAII> kernel_registration_handle_;
};

void RegisterOperators::checkSchemaAndRegisterOp_(Options&& options) {
  TORCH_CHECK(options.schemaOrName_.has_value(), "In operator registration: Tried to register an operator without specifying a schema or operator name.");
  if (options.schemaOrName_->is_right()) {
    // schema was explicitly specified. Check it matches the inferred one and register the op.

    const FunctionSchema& schema = options.schemaOrName_->right();

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
  std::unordered_set<DispatchKey> dispatch_keys;
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

  // HACK: bong in the alias analysis kind from the legacy API directly
  // into schema
  if (options.aliasAnalysisKind_.has_value()) {
    schema.setAliasAnalysis(*options.aliasAnalysisKind_);
  }

  OperatorName op_name = schema.operator_name();

  if (0 == options.kernels.size()) {
    registerSchemaOnly_(std::move(schema));
  } else {
    for (auto& kernel : options.kernels) {
      registerSchemaAndKernel_(schema, std::move(kernel));
    }
  }

  TORCH_INTERNAL_ASSERT(c10::Dispatcher::singleton().findSchema(op_name).has_value());
}

void RegisterOperators::registerSchemaAndKernel_(FunctionSchema schema, Options::KernelRegistrationConfig&& kernel) {
  TORCH_INTERNAL_ASSERT(kernel.func.isValid(), "Kernel must be set");

  registrars_.emplace_back(std::move(schema), kernel.dispatch_key, std::move(kernel.func));
}

void RegisterOperators::registerSchemaOnly_(FunctionSchema&& schema) {
  registrars_.emplace_back(std::move(schema), c10::nullopt, c10::nullopt);
}

RegisterOperators::RegisterOperators() = default;
RegisterOperators::~RegisterOperators() = default;
RegisterOperators::RegisterOperators(RegisterOperators&&) noexcept = default;
RegisterOperators& RegisterOperators::operator=(RegisterOperators&&) noexcept = default;


CppFunction::CppFunction(KernelFunction func, std::unique_ptr<c10::FunctionSchema> schema)
  : func_(std::move(func))
  , schema_(std::move(schema))
  {}

Module::Module(const char* ns)
  : ns_(ns)
  {}

Module::Module(Module&&) noexcept = default;
Module& Module::operator=(Module&&) noexcept = default;

// TODO: Error if an operator is def'ed multiple times.  Right now we just
// merge everything

namespace {
  std::string addNamespace(const char* ns, const char* name_or_schema) { if (ns) {
      // TODO: slow!  Fix internal data structures so I don't have to paste the
      // names together
      std::ostringstream oss;
      oss << ns << "::" << name_or_schema;
      return oss.str();
    } else {
      return name_or_schema;
    }
  }
}

Module&& Module::def(const char* schema) && {
  register_.op(c10::RegisterOperators::options()
    .schema(addNamespace(ns_, schema))
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
  return std::move(*this);
}

Module&& Module::def(const char* name_or_schema, CppFunction&& f) && {
  register_.op(c10::RegisterOperators::options()
    .schema(addNamespace(ns_, name_or_schema))
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA)
    .kernel(f.dispatch_key_, std::move(f.func_), std::move(f.schema_)));
  return std::move(*this);
}

Module&& Module::impl(const char* name_or_schema, CppFunction&& f) && {
  register_.op(c10::RegisterOperators::options()
    .schema(addNamespace(ns_, name_or_schema))
    // NB: Don't specify AliasAnalysis; the def() is expected to provide
    // this
    .kernel(f.dispatch_key_, std::move(f.func_), std::move(f.schema_)));
  return std::move(*this);
}

}
