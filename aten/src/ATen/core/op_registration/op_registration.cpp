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
  explicit OperatorRegistrar(FunctionSchema&& schema, c10::optional<TensorTypeId> dispatch_key, KernelFunction* kernel, KernelCacheCreatorFunction&& cache_creator)
  : op_(Dispatcher::singleton().registerSchema(std::move(schema))), kernel_registration_handle_(c10::nullopt) {
    // either both, kernel and cache_creator, or none must be set.
    AT_ASSERT((kernel != nullptr) == static_cast<bool>(cache_creator));

    if (kernel != nullptr) {
      if (dispatch_key.has_value()) {
        kernel_registration_handle_ = Dispatcher::singleton().registerKernel(op_.opHandle(), *dispatch_key, kernel, std::move(cache_creator));
      } else {
        kernel_registration_handle_ = Dispatcher::singleton().registerFallbackKernel(op_.opHandle(), kernel, std::move(cache_creator));
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
    throw std::logic_error("We don't support registering c10 ops on mobile yet because the function schema parser isn't present in the mobile build.");
  #else
    either<OperatorName, FunctionSchema> schemaOrName = torch::jit::parseSchemaOrName(schemaOrNameStr);
    if (schemaOrName.is_right()) {
      // schema was explicitly specified. Check it matches the inferred one and register the op.
      checkSchemaAndRegisterOp_(std::move(schemaOrName).right(), std::move(options));
    } else {
      // schema wasn't explicitly specified. Take the inferred schema for registering the op.
      AT_ASSERTM(nullptr != options.config.inferred_function_schema.get(), "Cannot infer schema from this kernel function. Please explicitly specify the operator schema.");
      OperatorName name = std::move(schemaOrName).left();
      FunctionSchema inferredSchema(
        std::move(name.name),
        std::move(name.overload_name),
        options.config.inferred_function_schema->arguments(),
        options.config.inferred_function_schema->returns(),
        options.config.inferred_function_schema->is_vararg(),
        options.config.inferred_function_schema->is_varret()
      );
      registerOp_(std::move(inferredSchema), std::move(options));
    }
  #endif
}

void RegisterOperators::checkSchemaAndRegisterOp_(FunctionSchema&& schema, Options&& options) {
  if (options.config.inferred_function_schema.get() != nullptr) {
    assertSchemasHaveSameSignature(*options.config.inferred_function_schema, schema);
  }

  registerOp_(std::move(schema), std::move(options));
}

void RegisterOperators::registerOp_(FunctionSchema&& schema, Options&& options) {
  TORCH_CHECK(!options.config.dispatch_key.has_value() || options.config.kernel_func != nullptr,
    "Tried to register an operator with a dispatch key but without a kernel. "
    "Please either specify a kernel or omit the dispatch key to only register the schema.");

  // if kernel_func is set, so must be cache_creator_func, the API shouldn't allow anything else.
  AT_ASSERT((options.config.kernel_func != nullptr) == static_cast<bool>(options.config.cache_creator_func));

  registrars_.emplace_back(std::move(schema), options.config.dispatch_key, options.config.kernel_func, std::move(options.config.cache_creator_func));
}

RegisterOperators::RegisterOperators() = default;
RegisterOperators::~RegisterOperators() = default;
RegisterOperators::RegisterOperators(RegisterOperators&&) noexcept = default;
RegisterOperators& RegisterOperators::operator=(RegisterOperators&&) noexcept = default;

}
