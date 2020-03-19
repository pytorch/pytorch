#include <ATen/core/op_registration/op_registration.h>
#if !defined(CAFFE2_IS_XPLAT_BUILD)
#include <torch/csrc/jit/frontend/function_schema_parser.h>
#endif

namespace c10 {

static_assert(std::is_nothrow_move_constructible<c10::optional<RegistrationHandleRAII>>::value, "");
static_assert(std::is_nothrow_move_assignable<c10::optional<RegistrationHandleRAII>>::value, "");

void RegisterOperators::checkSchemaAndRegisterOp_(Options&& options) {
  TORCH_CHECK(options.schemaOrName_.has_value(), "In operator registration: Tried to register an operator without specifying a schema or operator name.");
  if (options.schemaOrName_->is_right()) {
    // schema was explicitly specified. Check it matches the inferred one and register the op.

    const FunctionSchema& schema = options.schemaOrName_->right();

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
  TORCH_CHECK(options.kernels.size() > 0, "Cannot infer operator schema in registration of operator ", opName, " because there is no kernel specified.");

  c10::optional<FunctionSchema> inferred_schema = c10::nullopt;
  for (const auto& kernel : options.kernels) {
    if (nullptr != kernel.inferred_function_schema.get()) {
      if (!inferred_schema.has_value()) {
        inferred_schema = *kernel.inferred_function_schema;
        break;
      }
    }
  }
  TORCH_CHECK(inferred_schema.has_value(), "Cannot infer operator schema for this kind of kernel in registration of operator ", opName, ". Please explicitly specify the operator schema or specify at least one kernel for which we can infer the schema.");

  return *inferred_schema;
}

void RegisterOperators::checkNoDuplicateKernels_(const Options& options) {
  std::unordered_set<DispatchKey> dispatch_keys;
  bool has_catchall_kernel = false;

  for (const auto& kernel : options.kernels) {
    if (kernel.dispatch_key.has_value()) {
      TORCH_CHECK(0 == dispatch_keys.count(*kernel.dispatch_key), "In operator registration: Tried to register multiple kernels with same dispatch key ", *kernel.dispatch_key, " for operator schema ", toString(options.schemaOrName_->right()));
      dispatch_keys.insert(*kernel.dispatch_key);
    } else {
      TORCH_CHECK(!has_catchall_kernel, "In operator registration: Tried to register multiple catch-all kernels for operator schema ", toString(options.schemaOrName_->right()));
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

  registrars_.emplace_back(
    Dispatcher::singleton().registerDef(std::move(schema))
  );

  for (auto& kernel : options.kernels) {
    registrars_.emplace_back(
      Dispatcher::singleton().registerImpl(op_name, kernel.dispatch_key, std::move(kernel.func), std::move(kernel.inferred_function_schema), "legacy kernel from RegisterOperators")
    );
  }
}

RegisterOperators::RegisterOperators() = default;
RegisterOperators::~RegisterOperators() = default;
RegisterOperators::RegisterOperators(RegisterOperators&&) noexcept = default;
RegisterOperators& RegisterOperators::operator=(RegisterOperators&&) noexcept = default;


CppFunction::CppFunction(KernelFunction func, std::unique_ptr<c10::FunctionSchema> schema, std::string debug)
  : func_(std::move(func))
  , schema_(std::move(schema))
  , debug_(std::move(debug))
  {}

Module::Module(std::string ns)
  : ns_(std::move(ns))
  {}

Module::Module()
  : ns_(c10::nullopt)
  {}

Module::Module(Module&&) = default;
Module& Module::operator=(Module&&) = default;

// TODO: Error if an operator is def'ed multiple times.  Right now we just
// merge everything

Module& Module::def(FunctionSchema&& schema) & {
  if (ns_.has_value()) schema.setNamespaceIfNotSet(ns_->c_str());
  registrars_.emplace_back(Dispatcher::singleton().registerDef(std::move(schema)));
  return *this;
}
Module&& Module::def(FunctionSchema&& schema) && {
  def(std::move(schema));
  return std::move(*this);
}

Module& Module::def(c10::either<OperatorName, FunctionSchema>&& name_or_schema, CppFunction&& f) & {
  FunctionSchema schema = [&] {
    if (name_or_schema.is_right()) {
      return std::move(name_or_schema).right();
    } else {
      // it's a name; use the inferred schema
      TORCH_CHECK(f.schema_, "Module::def(): schema was not specified, and we "
          "couldn't infer schema either.  Please explicitly provide schema.");
      OperatorName name = std::move(name_or_schema).left();
      FunctionSchema s = f.schema_->cloneWithName(std::move(name.name), std::move(name.overload_name));
      s.setAliasAnalysis(c10::AliasAnalysisKind::CONSERVATIVE);
      return s;
    }
  }();
  if (ns_.has_value()) schema.setNamespaceIfNotSet(ns_->c_str());
  // Retain the OperatorName for Impl call
  OperatorName name = schema.operator_name();
  registrars_.emplace_back(Dispatcher::singleton().registerDef(std::move(schema)));
  registrars_.emplace_back(Dispatcher::singleton().registerImpl(name, f.dispatch_key_, std::move(f.func_), std::move(f.schema_), std::move(f.debug_)));
  return *this;
}
Module&& Module::def(c10::either<OperatorName, FunctionSchema>&& name_or_schema, CppFunction&& f) && {
  def(std::move(name_or_schema), std::move(f));
  return std::move(*this);
}

Module& Module::impl(const char* name_str, CppFunction&& f) & {
  auto name = torch::jit::parseName(name_str);
  if (ns_.has_value()) name.setNamespaceIfNotSet(ns_->c_str());
  registrars_.emplace_back(
    Dispatcher::singleton().registerImpl(
      std::move(name),
      f.dispatch_key_,
      std::move(f.func_),
      std::move(f.schema_),
      std::move(f.debug_)
    )
  );
  return *this;
}
Module&& Module::impl(const char* name_str, CppFunction&& f) && {
  impl(name_str, std::move(f));
  return std::move(*this);
}

Module& Module::fallback(CppFunction&& f) & {
  TORCH_CHECK(!ns_, "Cannot define fallbacks from namespaces, use c10::import().fallback() instead");
  TORCH_CHECK(f.dispatch_key_, "Fallback for catch all function not supported");
  registrars_.emplace_back(Dispatcher::singleton().registerFallback(*f.dispatch_key_, std::move(f.func_)));
  return *this;
}
Module&& Module::fallback(CppFunction&& f) && {
  fallback(std::move(f));
  return std::move(*this);
}

}
