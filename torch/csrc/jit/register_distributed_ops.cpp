#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/core/stack.h>

#include <torch/csrc/distributed/autograd/context/container.h>
#include <torch/csrc/distributed/autograd/engine/dist_engine.h>
#include <torch/csrc/jit/custom_operator.h>
#include <torch/csrc/jit/operator.h>

using Stack = std::vector<c10::IValue>;
using at::Scalar;
using at::Tensor;
using torch::jit::drop;
using torch::jit::pack;
using torch::jit::peek;
using torch::jit::pop;
using torch::jit::push;

namespace dist_autograd = torch::distributed::autograd;

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
