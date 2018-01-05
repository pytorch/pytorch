#include "ATen/ATen.h"
#include "ATen/UndefinedTensor.h"
#include <string>
#include "test_assert.h"

using namespace at;

int main() {
  // mainly test ops on undefined tensors don't segfault and give a reasonable errror message.
  Tensor und;
  Tensor ft = CPU(kFloat).ones({1});

  std::cout << und << std::endl;
  ASSERT(!und.defined());
  ASSERT(std::string("UndefinedTensor") == und.toString());

  ASSERT_THROWSM(und.strides(), "strides");
  ASSERT_THROWSM(und.dim(), "dim");
  ASSERT_THROWSM([]() {return Tensor();}() = Scalar(5), "UndefinedType");
  ASSERT_THROWSM(und.unsafeGetTH(true), "unsafeGetTH");
  ASSERT_THROWSM(und.add(und), "add");
  ASSERT_THROWSM(und.add(ft), "add");
  ASSERT_THROWSM(ft.add(und), "add");
  ASSERT_THROWSM(und.add(5), "add");
  ASSERT_THROWSM(und.mm(und), "mm");

  und.toType(und.type());
  ASSERT_THROWSM(und.toType(ft.type()), "attempt to copy an undefined tensor");
  ASSERT_THROWSM(ft.toType(und.type()), "UndefinedType");
  und.toType(ScalarType::Undefined);
  ASSERT_THROWSM(und.toType(ScalarType::Float), "toScalarType");
  ASSERT_THROWSM(ft.toType(ScalarType::Undefined), "UndefinedType");

  // copy_
  ASSERT_THROWSM(und.copy_(und), "copy");
  ASSERT_THROWSM(und.copy_(ft), "copy");
  ASSERT_THROWSM(ft.copy_(und), "copy");

  und.toBackend(Backend::Undefined);
  ASSERT_THROWSM(und.toBackend(Backend::CPU), "toBackend");
  ASSERT_THROWSM(ft.toBackend(Backend::Undefined), "UndefinedType");

  Tensor to_move = CPU(kFloat).ones({1});
  Tensor m(std::move(to_move));
  ASSERT(!to_move.defined());
  ASSERT(to_move.get() == UndefinedTensor::singleton());

  return 0;
}

