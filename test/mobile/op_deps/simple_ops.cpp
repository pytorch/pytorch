#include "simple_ops.h"

#include <iostream>

#include <c10/core/TensorOptions.h>
#include <ATen/core/op_registration/op_registration.h>

#include "utils.h"

namespace at {

// AA -> BB
Tensor AA_op(const Tensor& self) {
  std::cout << "AA op" << std::endl;
  if (self.ndimension() >= 4) {
    return call_BB_op(self);
  }
  return self;
}

// BB -> AA
Tensor BB_op(const Tensor& self) {
  std::cout << "BB op" << std::endl;
  if (self.ndimension() < 4) {
    return global_helper_call_AA_op_1(self);
  }
  return self;
}

// CC -> (AA -> BB)
Tensor CC_op(const Tensor& self) {
  std::cout << "CC op" << std::endl;
  return global_helper_call_AA_op_2(self);
}

// DD -> (AA -> BB) / (EE -> FF)
Tensor DD_op(const Tensor& self) {
  std::cout << "DD op" << std::endl;
  if (self.ndimension() < 4) {
    return global_helper_call_AA_op_3(self);
  }
  return call_EE_op(self);
}

// EE -> FF
Tensor EE_op(const Tensor& self) {
  std::cout << "EE op" << std::endl;
  if (self.ndimension() >= 4) {
    return call_FF_op(self);
  }
  return self;
}

// FF -> EE
Tensor FF_op(const Tensor& self) {
  std::cout << "FF op" << std::endl;
  if (self.ndimension() < 4) {
    return call_EE_op(self);
  }
  return self;
}

namespace {

auto registerer = torch::import()
  .def("aten::AA(Tensor self) -> Tensor",
    torch::dispatch(DispatchKey::CPUTensorId, &AA_op))
  .def("aten::BB(Tensor self) -> Tensor", &BB_op)
  .impl("aten::CC(Tensor self) -> Tensor",
    torch::dispatch(DispatchKey::CPUTensorId, &CC_op))
  .impl("aten::DD(Tensor self) -> Tensor", &DD_op)
  .def("aten::EE(Tensor self) -> Tensor", torch::dispatch(
    DispatchKey::CPUTensorId,
    CppFunction::makeUnboxedOnly(EE_op)))
  .def("aten::FF(Tensor self) -> Tensor",
    CppFunction::makeUnboxedOnly(FF_op))
  .impl("aten::GG(Tensor self) -> Tensor", torch::dispatch(
    DispatchKey::CPUTensorId, [] (Tensor a) -> Tensor {
      return call_FF_op(a);
    }))
  .impl("aten::HH(Tensor self) -> Tensor",
    [] (Tensor a) -> Tensor {
      return a;
    });

} // namespace

} // namespace at
