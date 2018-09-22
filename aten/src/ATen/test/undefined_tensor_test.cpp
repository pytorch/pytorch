#define CATCH_CONFIG_MAIN
#include "catch_utils.hpp"

#include "ATen/ATen.h"
#include "ATen/core/UndefinedTensorImpl.h"
#include <string>
#include "test_seed.h"

using namespace at;

CATCH_TEST_CASE( "undefined tensor test", "[]" ) {
  manual_seed(123, at::kCPU);

  // mainly test ops on undefined tensors don't segfault and give a reasonable errror message.
  Tensor und;
  Tensor ft = ones({1}, CPU(kFloat));

  std::stringstream ss;
  ss << und << std::endl;
  CATCH_REQUIRE(!und.defined());
  CATCH_REQUIRE(std::string("UndefinedType") == und.toString());

  _CATCH_REQUIRE_THROWS(und.strides());
  _CATCH_REQUIRE_THROWS(und.dim());
  _CATCH_REQUIRE_THROWS([]() {return Tensor();}() = Scalar(5));
  _CATCH_REQUIRE_THROWS(und.add(und));
  _CATCH_REQUIRE_THROWS(und.add(ft));
  _CATCH_REQUIRE_THROWS(ft.add(und));
  _CATCH_REQUIRE_THROWS(und.add(5));
  _CATCH_REQUIRE_THROWS(und.mm(und));

  und.toType(und.type());
  _CATCH_REQUIRE_THROWS(und.toType(ft.type()));
  _CATCH_REQUIRE_THROWS(ft.toType(und.type()));
  und.toType(ScalarType::Undefined);
  _CATCH_REQUIRE_THROWS(und.toType(ScalarType::Float));
  _CATCH_REQUIRE_THROWS(ft.toType(ScalarType::Undefined));

  // copy_
  _CATCH_REQUIRE_THROWS(und.copy_(und));
  _CATCH_REQUIRE_THROWS(und.copy_(ft));
  _CATCH_REQUIRE_THROWS(ft.copy_(und));

  und.toBackend(Backend::Undefined);
  _CATCH_REQUIRE_THROWS(und.toBackend(Backend::CPU));
  _CATCH_REQUIRE_THROWS(ft.toBackend(Backend::Undefined));

  Tensor to_move = ones({1}, CPU(kFloat));
  Tensor m(std::move(to_move));
  CATCH_REQUIRE(!to_move.defined());
  CATCH_REQUIRE(to_move.unsafeGetTensorImpl() == UndefinedTensorImpl::singleton());
}
