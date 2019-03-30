#include <ATen/core/op_registration/op_registration.h>

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
  : op_(Dispatcher::singleton().registerSchema(std::move(schema))), dispatch_key_(std::move(dispatch_key)), owns_registration_(true) {
    if (dispatch_key_.has_value()) {
      Dispatcher::singleton().registerKernel(op_, *dispatch_key_, kernel, std::move(cache_creator));
    } else {
      Dispatcher::singleton().registerFallbackKernel(op_, kernel, std::move(cache_creator));
    }
  }

  OperatorRegistrar(OperatorRegistrar&& rhs) noexcept
  :  op_(std::move(rhs.op_)), dispatch_key_(std::move(rhs.dispatch_key_)), owns_registration_(rhs.owns_registration_) {
    rhs.owns_registration_ = false;
  }

  // not needed and would break RAII if defaulted.
  OperatorRegistrar& operator=(OperatorRegistrar&& rhs) noexcept = delete;
  OperatorRegistrar(const OperatorRegistrar& rhs) = delete;
  OperatorRegistrar& operator=(const OperatorRegistrar& rhs) = delete;

  ~OperatorRegistrar() {
    if (owns_registration_) {
      if (dispatch_key_.has_value()) {
        Dispatcher::singleton().deregisterKernel(op_, *dispatch_key_);
      } else {
        Dispatcher::singleton().deregisterFallbackKernel(op_);
      }
      Dispatcher::singleton().deregisterSchema(op_);
    }
  }

private:
  const OperatorHandle op_;
  const c10::optional<TensorTypeId> dispatch_key_;
  bool owns_registration_;
};

void RegisterOperators::registerOp_(FunctionSchema&& schema, detail::KernelRegistrationConfig&& config) {
  // TODO Should we allow this and only register a schema without a kernel?
  AT_CHECK(config.kernel_func != nullptr,
      "Tried to register an operator with function schema ", toString(schema),
      ", but didn't specify a kernel. Please add a c10::kernel<...>(...) parameter to the registration call.");
  // if kernel_func is set, so must be cache_creator_func, the API shouldn't allow anything else.
  AT_ASSERT(static_cast<bool>(config.cache_creator_func));

  if (config.inferred_function_schema.get() != nullptr) {
    assertSchemasHaveSameSignature(*config.inferred_function_schema, schema);
  }

  registrars_.emplace_back(std::move(schema), config.dispatch_key, config.kernel_func, std::move(config.cache_creator_func));
}

}
