#include <ATen/core/op_registration/op_registration.h>
#if !defined(CAFFE2_IS_XPLAT_BUILD)
#include <torch/csrc/jit/frontend/function_schema_parser.h>
#endif

namespace c10 {

namespace {
  // TODO: Consider representing debug info as a struct instead so you
  // don't have to allocate strings all the time
  std::string debugString(std::string debug, const char* file, uint32_t line) {
    if (debug.empty()) {
      return c10::str("registered at ", file, ":", line);
    } else {
      return std::move(debug);
    }
  }
}

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
    Dispatcher::singleton().registerDef(std::move(schema), "registered by RegisterOperators")
  );

  for (auto& kernel : options.kernels) {
    registrars_.emplace_back(
      Dispatcher::singleton().registerImpl(op_name, kernel.dispatch_key, std::move(kernel.func), std::move(kernel.inferred_function_schema), "registered by RegisterOperators")
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

Library::Library(std::string ns, const char* file, uint32_t line)
  : ns_(ns == "_" ? c10::nullopt : c10::make_optional(std::move(ns)))
  , dispatch_key_(c10::nullopt)
  , file_(file)
  , line_(line)
  {}

Library::Library(std::string ns, DispatchKey k, const char* file, uint32_t line)
  : ns_(ns == "_" ? c10::nullopt : c10::make_optional(std::move(ns)))
  , dispatch_key_(k == DispatchKey::CatchAll ? c10::nullopt : c10::make_optional(k))
  , file_(file)
  , line_(line)
  {}

// TODO: Error if an operator is def'ed multiple times.  Right now we just
// merge everything

Library& Library::_def(FunctionSchema&& schema) & {
  if (ns_.has_value()) {
    TORCH_CHECK(schema.setNamespaceIfNotSet(ns_->c_str()), "Attempted to def ", toString(schema.operator_name()), " which is explicitly qualified with a namespace inside a TORCH_LIBRARY, which is not allowed.  If TORCH_LIBRARY's namespace matches the explicitly given namespace, remove the qualifier; otherwise, move your def into a TORCH_LIBRARY block with the correct namespace, or give it a different namespace.  Registration site was ", file_, ":", line_);
  }
  registrars_.emplace_back(Dispatcher::singleton().registerDef(std::move(schema), debugString("", file_, line_)));
  return *this;
}

Library& Library::_def(c10::either<OperatorName, FunctionSchema>&& name_or_schema, CppFunction&& f) & {
  FunctionSchema schema = [&] {
    if (name_or_schema.is_right()) {
      return std::move(name_or_schema).right();
    } else {
      // it's a name; use the inferred schema
      TORCH_CHECK(f.schema_, "Library::def(): schema was not specified, and we "
          "couldn't infer schema either.  Please explicitly provide schema.  Registration site was ", file_, ":", line_);
      OperatorName name = std::move(name_or_schema).left();
      FunctionSchema s = f.schema_->cloneWithName(std::move(name.name), std::move(name.overload_name));
      s.setAliasAnalysis(c10::AliasAnalysisKind::CONSERVATIVE);
      return s;
    }
  }();
  if (ns_.has_value()) {
    TORCH_CHECK(schema.setNamespaceIfNotSet(ns_->c_str()), "Attempted to def ", toString(schema.operator_name()), " which is explicitly qualified with a namespace inside a TORCH_LIBRARY, which is not allowed.  If TORCH_LIBRARY's namespace matches the explicitly given namespace, remove the qualifier; otherwise, please move your def into a TORCH_LIBRARY block with the correct namespace, or give it a different namespace.  Registration site was ", file_, ":", line_);
  }
  TORCH_CHECK(!(f.dispatch_key_.has_value() && dispatch_key_.has_value()), "Cannot specify a different dispatch key inside a TORCH_LIBRARY_IMPL; please declare a separate TORCH_LIBRARY_IMPL for your dispatch key.  Registration site was ", file_, ":", line_);
  auto dispatch_key = f.dispatch_key_.has_value() ? f.dispatch_key_ : dispatch_key_;
  // Retain the OperatorName for Impl call
  OperatorName name = schema.operator_name();
  registrars_.emplace_back(Dispatcher::singleton().registerDef(std::move(schema), debugString("", file_, line_)));
  registrars_.emplace_back(Dispatcher::singleton().registerImpl(name, dispatch_key, std::move(f.func_), std::move(f.schema_), debugString(std::move(f.debug_), file_, line_)));
  return *this;
}

Library& Library::_impl(const char* name_str, CppFunction&& f) & {
  auto name = torch::jit::parseName(name_str);
  if (ns_.has_value()) {
    TORCH_CHECK(name.setNamespaceIfNotSet(ns_->c_str()), "Attempted to impl ", toString(name), " which is explicitly qualified with a namespace inside a TORCH_LIBRARY, which is not allowed.  If TORCH_LIBRARY's namespace matches the explicitly given namespace, remove the qualifier; otherwise, please place it in an separate TORCH_LIBRARY_IMPL block to make it clear that you are overriding behavior for an operator in a different library.  Registration site was ", file_, ":", line_);
  }
  TORCH_CHECK(!(f.dispatch_key_.has_value() && dispatch_key_.has_value()), "Cannot specify a different dispatch key inside a TORCH_LIBRARY_IMPL; please declare a separate TORCH_LIBRARY_IMPL for your dispatch key.  Registration site was ", file_, ":", line_);
  auto dispatch_key = f.dispatch_key_.has_value() ? f.dispatch_key_ : dispatch_key_;
  registrars_.emplace_back(
    Dispatcher::singleton().registerImpl(
      std::move(name),
      dispatch_key,
      std::move(f.func_),
      std::move(f.schema_),
      debugString(std::move(f.debug_), file_, line_)
    )
  );
  return *this;
}

Library& Library::_fallback(CppFunction&& f) & {
  TORCH_CHECK(!ns_.has_value(), "Cannot define a fallback in a namespaced TORCH_LIBRARY (fallbacks always affect operators outside of your library).  Instead, use TORCH_LIBRARY_IMPL(_, Backend, m) { m.fallback(...); }.  Registration site was ", file_, ":", line_);
  auto dispatch_key = f.dispatch_key_.has_value() ? f.dispatch_key_ : dispatch_key_;
  TORCH_CHECK(dispatch_key.has_value(), "Fallback must be defined for a specific backend, e.g., inside a TORCH_LIBRARY_IMPL.  Registration site was", file_, ":", line_);
  registrars_.emplace_back(
    Dispatcher::singleton().registerFallback(
      *dispatch_key,
      std::move(f.func_),
      debugString(std::move(f.debug_), file_, line_)
    )
  );
  return *this;
}

}
