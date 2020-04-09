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

// NB: Some of these registrations (AA, EE) are not what you
// actually expect to see in practice, but we cover them here
// as they are technically "valid" API calls and we want to
// make sure the analyzer catches them.  (The analyzer is very
// generic, so actually there isn't any reason it shouldn't work,
// but it's good to test them!)
//
// Additionally, the code in this file is not really runnable; for
// example we are missing schemas for all of the impl registrations
// here.  The analyzer doesn't really care, as it only really
// cares about the name
auto registerer = torch::import()
  .def("aten::AA(Tensor self) -> Tensor", torch::dispatch(kCPU, &AA_op))
  .def("aten::BB(Tensor self) -> Tensor", &BB_op)
  .impl("aten::CC", kCPU, &CC_op)
  .impl("aten::DD", &DD_op)
  .impl_UNBOXED("aten::EE", kCPU, EE_op)
  .def("aten::FF(Tensor self) -> Tensor", CppFunction::makeUnboxedOnly(FF_op))
  .impl("aten::GG",
    kCPU, [] (Tensor a) -> Tensor {
      return call_FF_op(a);
    })
  .impl("aten::HH",
    [] (Tensor a) -> Tensor {
      return a;
    });

} // namespace

} // namespace at
