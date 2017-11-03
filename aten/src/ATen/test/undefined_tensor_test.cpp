#include "ATen/ATen.h"
#include <string>

using namespace at;

#define ASSERT_THROWS(fn, message)                                  \
try {                                                               \
  fn;                                                               \
  assert(false);                                                    \
} catch(std::runtime_error &e) {                                    \
  assert(std::string(e.what()).find(message) != std::string::npos); \
}


int main() {
  // mainly test ops on undefined tensors don't segfault and give a reasonable errror message.
  Tensor und;
  Tensor ft = CPU(kFloat).ones({1});

  std::cout << und << std::endl;
  assert(!und.defined());
  assert(std::string("UndefinedTensor") == und.toString());

  ASSERT_THROWS(und.strides(), "strides");
  ASSERT_THROWS(und.dim(), "dim");
  ASSERT_THROWS(und.assign_(Scalar(5)), "assign");
  ASSERT_THROWS(und.unsafeGetTH(true), "unsafeGetTH");
  ASSERT_THROWS(und.add(und), "add");
  ASSERT_THROWS(und.add(ft), "add");
  ASSERT_THROWS(ft.add(und), "add");
  ASSERT_THROWS(und.add(5), "add");
  ASSERT_THROWS(und.mm(und), "mm");

  und.toType(und.type());
  ASSERT_THROWS(und.toType(ft.type()), ""); // FIXME: UNKNOWN_BACKENDFloatType is not enabled
  ASSERT_THROWS(ft.toType(und.type()), ""); // FIXME: tensor is not implemented for type UndefinedType
  und.toType(ScalarType::Undefined);
  ASSERT_THROWS(und.toType(ScalarType::Float), "toScalarType");
  ASSERT_THROWS(ft.toType(ScalarType::Undefined), ""); // FIXME: CPUUNKNOWN_SCALAR_TYPEType is not enabled.

  // copy_
  ASSERT_THROWS(und.copy_(und), "copy");
  ASSERT_THROWS(und.copy_(ft), "copy");
  ASSERT_THROWS(ft.copy_(und), "copy");

  und.toBackend(Backend::Undefined);
  ASSERT_THROWS(und.toBackend(Backend::CPU), "toBackend");
  ASSERT_THROWS(ft.toBackend(Backend::Undefined), ""); // UNKNOWN_BACKENDFloatType is not enabled.

  return 0;
}

