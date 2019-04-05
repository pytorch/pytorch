#include <ATen/core/op_registration/op_registration.h>
#include <torch/csrc/jit/script/function_schema_parser.h>

namespace c10 {

RegisterOperators::RegisterOperators() = default;
RegisterOperators::~RegisterOperators() = default;
RegisterOperators::RegisterOperators(RegisterOperators&&) noexcept = default;
RegisterOperators& RegisterOperators::operator=(RegisterOperators&&) = default;

// OperatorRegistrar in its constructor registers an operator in the dispatch
// table deregisters it in the destructor.
class RegisterOperators::OperatorRegistrar final {
public:
  explicit OperatorRegistrar(FunctionSchema&& schema, c10::optional<TensorTypeId> dispatch_key, KernelFunction* kernel, KernelCacheCreatorFunction&& cache_creator)
  : op_(Dispatcher::singleton().registerSchema(std::move(schema))), dispatch_key_(std::move(dispatch_key)), has_kernel_(kernel != nullptr), owns_registration_(true) {
    // either both, kernel and cache_creator, or none must be set.
    AT_ASSERT((kernel != nullptr) == static_cast<bool>(cache_creator));

    if (has_kernel_) {
      if (dispatch_key_.has_value()) {
        Dispatcher::singleton().registerKernel(op_, *dispatch_key_, kernel, std::move(cache_creator));
      } else {
        Dispatcher::singleton().registerFallbackKernel(op_, kernel, std::move(cache_creator));
      }
    }
  }

  OperatorRegistrar(OperatorRegistrar&& rhs) noexcept
  :  op_(std::move(rhs.op_)), dispatch_key_(std::move(rhs.dispatch_key_)), has_kernel_(rhs.has_kernel_), owns_registration_(rhs.owns_registration_) {
    rhs.owns_registration_ = false;
  }

  // not needed and would break RAII if defaulted.
  OperatorRegistrar& operator=(OperatorRegistrar&& rhs) noexcept = delete;
  OperatorRegistrar(const OperatorRegistrar& rhs) = delete;
  OperatorRegistrar& operator=(const OperatorRegistrar& rhs) = delete;

  ~OperatorRegistrar() {
    if (owns_registration_) {
      if (has_kernel_) {
        if (dispatch_key_.has_value()) {
          Dispatcher::singleton().deregisterKernel(op_, *dispatch_key_);
        } else {
          Dispatcher::singleton().deregisterFallbackKernel(op_);
        }
      }
      Dispatcher::singleton().deregisterSchema(op_);
    }
  }

private:
  const OperatorHandle op_;
  const c10::optional<TensorTypeId> dispatch_key_;
  bool has_kernel_;
  bool owns_registration_;
};

void RegisterOperators::checkSchemaAndRegisterOp_(const std::string& schemaOrNameStr, detail::KernelRegistrationConfig&& config) {
  either<OperatorName, FunctionSchema> schemaOrName = torch::jit::parseSchemaOrName(schemaOrNameStr);
  if (schemaOrName.is_right()) {
    // schema was explicitly specified. Check it matches the inferred one and register the op.
    checkSchemaAndRegisterOp_(std::move(schemaOrName).right(), std::move(config));
  } else {
    // schema wasn't explicitly specified. Take the inferred schema for registering the op.
    AT_ASSERTM(nullptr != config.inferred_function_schema.get(), "Cannot infer schema from this kernel function. Please explicitly specify the operator schema.");
    OperatorName name = std::move(schemaOrName).left();
    FunctionSchema inferredSchema(
      std::move(name.name),
      std::move(name.overload_name),
      config.inferred_function_schema->arguments(),
      config.inferred_function_schema->returns(),
      config.inferred_function_schema->is_vararg(),
      config.inferred_function_schema->is_varret()
    );
    registerOp_(std::move(inferredSchema), std::move(config));
  }
}

void RegisterOperators::checkSchemaAndRegisterOp_(FunctionSchema&& schema, detail::KernelRegistrationConfig&& config) {
  if (config.inferred_function_schema.get() != nullptr) {
    assertSchemasHaveSameSignature(*config.inferred_function_schema, schema);
  }

  registerOp_(std::move(schema), std::move(config));
}

void RegisterOperators::registerOp_(FunctionSchema&& schema, detail::KernelRegistrationConfig&& config) {
  AT_CHECK(!config.dispatch_key.has_value() || config.kernel_func != nullptr,
    "Tried to register an operator with a dispatch key but without a kernel. "
    "Please either specify a kernel or omit the dispatch key to only register the schema.");

  // if kernel_func is set, so must be cache_creator_func, the API shouldn't allow anything else.
  AT_ASSERT((config.kernel_func != nullptr) == static_cast<bool>(config.cache_creator_func));

  registrars_.emplace_back(std::move(schema), config.dispatch_key, config.kernel_func, std::move(config.cache_creator_func));
}

}
