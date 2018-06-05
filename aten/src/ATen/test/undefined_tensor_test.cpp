#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "ATen/ATen.h"
#include "ATen/UndefinedTensor.h"
#include <string>
#include "test_seed.h"

using namespace at;

TEST_CASE( "undefined tensor test", "[]" ) {
  manual_seed(123, at::Backend::CPU);

  // mainly test ops on undefined tensors don't segfault and give a reasonable errror message.
  Tensor und;
  Tensor ft = ones(CPU(kFloat), {1});

  std::stringstream ss;
  ss << und << std::endl;
  REQUIRE(!und.defined());
  REQUIRE(std::string("UndefinedTensor") == und.toString());

  REQUIRE_THROWS_WITH(und.strides(), Catch::Contains("strides"));
  REQUIRE_THROWS_WITH(und.dim(), Catch::Contains("dim"));
  REQUIRE_THROWS_WITH([]() {return Tensor();}() = Scalar(5), Catch::Contains("UndefinedType"));
  REQUIRE_THROWS_WITH(und.unsafeGetTH(true), Catch::Contains("unsafeGetTH"));
  REQUIRE_THROWS_WITH(und.add(und), Catch::Contains("add"));
  REQUIRE_THROWS_WITH(und.add(ft), Catch::Contains("add"));
  REQUIRE_THROWS_WITH(ft.add(und), Catch::Contains("add"));
  REQUIRE_THROWS_WITH(und.add(5), Catch::Contains("add"));
  REQUIRE_THROWS_WITH(und.mm(und), Catch::Contains("mm"));

  und.toType(und.type());
  REQUIRE_THROWS_WITH(und.toType(ft.type()), Catch::Contains("attempt to copy an undefined tensor"));
  REQUIRE_THROWS_WITH(ft.toType(und.type()), Catch::Contains("UndefinedType"));
  und.toType(ScalarType::Undefined);
  REQUIRE_THROWS_WITH(und.toType(ScalarType::Float), Catch::Contains("toScalarType"));
  REQUIRE_THROWS_WITH(ft.toType(ScalarType::Undefined), Catch::Contains("UndefinedType"));

  // copy_
  REQUIRE_THROWS_WITH(und.copy_(und), Catch::Contains("copy"));
  REQUIRE_THROWS_WITH(und.copy_(ft), Catch::Contains("copy"));
  REQUIRE_THROWS_WITH(ft.copy_(und), Catch::Contains("copy"));

  und.toBackend(Backend::Undefined);
  REQUIRE_THROWS_WITH(und.toBackend(Backend::CPU), Catch::Contains("toBackend"));
  REQUIRE_THROWS_WITH(ft.toBackend(Backend::Undefined), Catch::Contains("UndefinedType"));

  Tensor to_move = ones(CPU(kFloat), {1});
  Tensor m(std::move(to_move));
  REQUIRE(!to_move.defined());
  REQUIRE(to_move.get() == UndefinedTensor::singleton());
}

