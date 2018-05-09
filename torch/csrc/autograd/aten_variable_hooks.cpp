#include <ATen/detail/VariableHooksInterface.h>
#include <torch/csrc/autograd/generated/VariableType.h>

namespace torch { namespace autograd {

struct VariableHooks : public at::VariableHooksInterface {
  VariableHooks(at::VariableHooksArgs) {}
  void registerVariableTypeFor(at::Context*, at::Backend, at::ScalarType) const override;
};

// Sigh, the registry doesn't support namespaces :(
using at::RegistererVariableHooksRegistry;
using at::VariableHooksRegistry;

REGISTER_VARIABLE_HOOKS(VariableHooks)

// Pre-condition: backend/scalar_type is a valid type in the type_registry
void VariableHooks::registerVariableTypeFor(at::Context* context, at::Backend backend, at::ScalarType scalar_type) const {
  register_variable_type_for(context, backend, scalar_type);
}

}} // torch::autograd
