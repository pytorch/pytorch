#include <ATen/ATen.h>
#include "torch/csrc/jit/runtime/operator.h"
#include "torch/csrc/jit/runtime/custom_operator.h"
#include <ATen/core/op_registration/op_registration.h>

#include <torch/csrc/distributed/autograd/context/container.h>
#include <torch/csrc/distributed/autograd/engine/dist_engine.h>

#include <torch/csrc/distributed/rpc/rref_impl.h>

using at::Scalar;
using at::Tensor;
namespace dist_autograd = torch::distributed::autograd;
namespace dist_rpc = torch::distributed::rpc;

namespace torch {
namespace jit {

namespace {
at::Tensor toOptionalTensor(const c10::IValue& v) {
  if (v.isNone()) {
    return at::Tensor();
  }
  return v.toTensor();
}

at::Tensor optional_to_tensor(c10::optional<at::Tensor> v) {
  return v.has_value() ? *v : at::Tensor();
}

c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return c10::AliasAnalysisKind::FROM_SCHEMA;
}

RegisterOperators reg_rpc_ops({
    Operator(
        "aten::to_here(RRef(t) self) -> t",
        [](Stack& stack) {
          auto rref = pop(stack).toRRef();
          IValue res;
          if (rref->isOwner()) {
            res = c10::dynamic_intrusive_pointer_cast<dist_rpc::OwnerRRef>(rref)
                      ->getValue();
          } else {
            res = c10::dynamic_intrusive_pointer_cast<dist_rpc::UserRRef>(rref)
                      ->toHere();
          }
          push(stack, std::move(res));
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::is_owner(RRef(t) self) -> bool",
        [](Stack& stack) {
          auto rref = pop(stack).toRRef();
          push(stack, rref->isOwner());
          return 0;
        },
        aliasAnalysisFromSchema()),
});

auto reg_distributed_ops =
    torch::RegisterOperators()
        .op("aten::get_gradients(int context_id) -> Dict(Tensor, Tensor)",
            torch::RegisterOperators::options()
                .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
                .catchAllKernel([](int64_t context_id) {
                  const auto& autogradContext =
                      dist_autograd::DistAutogradContainer::getInstance()
                          .retrieveContext(context_id);
                  return autogradContext->getGradients();
                }));

} // namespace
} // namespace jit
} // namespace torch
