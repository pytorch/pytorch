#include "simple_ops.h"

#include <iostream>

#include <c10/core/TensorOptions.h>
#include <ATen/core/op_registration/op_registration.h>

#include "utils.h"

namespace at {

// AA -> BB
Tensor AA_op(const Tensor& self) {
  std::cout << "AA op" << std::endl;
  return call_BB_op(self);
}

// BB -> AA
Tensor BB_op(const Tensor& self) {
  std::cout << "BB op" << std::endl;
  return global_helper_call_AA_op_1(self);
}

// CC -> (AA -> BB)
Tensor CC_op(const Tensor& self) {
  std::cout << "CC op" << std::endl;
  return global_helper_call_AA_op_2(self);
}

// DD -> (AA -> BB) / (EE -> FF)
Tensor DD_op(const Tensor& self) {
  std::cout << "DD op" << std::endl;
  if (self.sizes().size() < 4) {
    return global_helper_call_AA_op_3(self);
  }
  return call_EE_op(self);
}

// EE -> FF
Tensor EE_op(const Tensor& self) {
  std::cout << "EE op" << std::endl;
  return call_FF_op(self);
}

// FF -> EE
Tensor FF_op(const Tensor& self) {
  std::cout << "FF op" << std::endl;
  return call_EE_op(self);
}

namespace {

auto registerer = torch::RegisterOperators()
  .op(torch::RegisterOperators::options()
    .schema("aten::AA(Tensor self) -> Tensor")
    .kernel<decltype(AA_op), &AA_op>(TensorTypeId::CPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::BB(Tensor self) -> Tensor")
    .catchAllKernel<decltype(BB_op), &BB_op>()
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::CC(Tensor self) -> Tensor")
    .kernel(TensorTypeId::CPUTensorId, &CC_op)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::DD(Tensor self) -> Tensor")
    .catchAllKernel(&DD_op)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::EE(Tensor self) -> Tensor")
    .impl_unboxedOnlyKernel<decltype(EE_op), &EE_op>(TensorTypeId::CPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::FF(Tensor self) -> Tensor")
    .impl_unboxedOnlyCatchAllKernel<decltype(FF_op), &FF_op>()
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::GG(Tensor self) -> Tensor")
    .kernel(TensorTypeId::CPUTensorId, [] (Tensor a) -> Tensor {
      return call_FF_op(a);
    }))
  .op(torch::RegisterOperators::options()
    .schema("aten::HH(Tensor self) -> Tensor")
    .catchAllKernel([] (Tensor a) -> Tensor {
      return a;
    }));

} // namespace

} // namespace at
