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

  ASSERT_THROWS(und.strides(), "strides");
  ASSERT_THROWS(und.dim(), "dim");
  ASSERT_THROWS([]() {return Tensor();}() = Scalar(5), "UndefinedType");
  ASSERT_THROWS(und.unsafeGetTH(true), "unsafeGetTH");
  ASSERT_THROWS(und.add(und), "add");
  ASSERT_THROWS(und.add(ft), "add");
  ASSERT_THROWS(ft.add(und), "add");
  ASSERT_THROWS(und.add(5), "add");
  ASSERT_THROWS(und.mm(und), "mm");

  und.toType(und.type());
  ASSERT_THROWS(und.toType(ft.type()), "attempt to copy an undefined tensor");
  ASSERT_THROWS(ft.toType(und.type()), "UndefinedType");
  und.toType(ScalarType::Undefined);
  ASSERT_THROWS(und.toType(ScalarType::Float), "toScalarType");
  ASSERT_THROWS(ft.toType(ScalarType::Undefined), "UndefinedType");

  // copy_
  ASSERT_THROWS(und.copy_(und), "copy");
  ASSERT_THROWS(und.copy_(ft), "copy");
  ASSERT_THROWS(ft.copy_(und), "copy");

  und.toBackend(Backend::Undefined);
  ASSERT_THROWS(und.toBackend(Backend::CPU), "toBackend");
  ASSERT_THROWS(ft.toBackend(Backend::Undefined), "UndefinedType");

  Tensor to_move = CPU(kFloat).ones({1});
  Tensor m(std::move(to_move));
  ASSERT(!to_move.defined());
  ASSERT(to_move.get() == UndefinedTensor::singleton());

  return 0;
}

