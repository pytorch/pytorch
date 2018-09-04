#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "ATen/ATen.h"
#include "ATen/core/UndefinedTensorImpl.h"
#include <string>
#include "test_seed.h"

using namespace at;

TEST_CASE( "undefined tensor test", "[]" ) {
  manual_seed(123, at::kCPU);

  // mainly test ops on undefined tensors don't segfault and give a reasonable errror message.
  Tensor und;
  Tensor ft = ones({1}, CPU(kFloat));

  std::stringstream ss;
  ss << und << std::endl;
  REQUIRE(!und.defined());
  REQUIRE(std::string("UndefinedType") == und.toString());

  REQUIRE_THROWS(und.strides());
  REQUIRE_THROWS(und.dim());
  REQUIRE_THROWS([]() {return Tensor();}() = Scalar(5));
  REQUIRE_THROWS(und.add(und));
  REQUIRE_THROWS(und.add(ft));
  REQUIRE_THROWS(ft.add(und));
  REQUIRE_THROWS(und.add(5));
  REQUIRE_THROWS(und.mm(und));

  und.toType(und.type());
  REQUIRE_THROWS(und.toType(ft.type()));
  REQUIRE_THROWS(ft.toType(und.type()));
  und.toType(ScalarType::Undefined);
  REQUIRE_THROWS(und.toType(ScalarType::Float));
  REQUIRE_THROWS(ft.toType(ScalarType::Undefined));

  // copy_
  REQUIRE_THROWS(und.copy_(und));
  REQUIRE_THROWS(und.copy_(ft));
  REQUIRE_THROWS(ft.copy_(und));

  und.toBackend(Backend::Undefined);
  REQUIRE_THROWS(und.toBackend(Backend::CPU));
  REQUIRE_THROWS(ft.toBackend(Backend::Undefined));

  Tensor to_move = ones({1}, CPU(kFloat));
  Tensor m(std::move(to_move));
  REQUIRE(!to_move.defined());
  REQUIRE(to_move.unsafeGetTensorImpl() == UndefinedTensorImpl::singleton());
}
