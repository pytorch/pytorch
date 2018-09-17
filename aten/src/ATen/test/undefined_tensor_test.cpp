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

  CATCH_REQUIRE_THROWS(und.strides());
  CATCH_REQUIRE_THROWS(und.dim());
  CATCH_REQUIRE_THROWS([]() {return Tensor();}() = Scalar(5));
  CATCH_REQUIRE_THROWS(und.add(und));
  CATCH_REQUIRE_THROWS(und.add(ft));
  CATCH_REQUIRE_THROWS(ft.add(und));
  CATCH_REQUIRE_THROWS(und.add(5));
  CATCH_REQUIRE_THROWS(und.mm(und));

  und.toType(und.type());
  CATCH_REQUIRE_THROWS(und.toType(ft.type()));
  CATCH_REQUIRE_THROWS(ft.toType(und.type()));
  und.toType(ScalarType::Undefined);
  CATCH_REQUIRE_THROWS(und.toType(ScalarType::Float));
  CATCH_REQUIRE_THROWS(ft.toType(ScalarType::Undefined));

  // copy_
  CATCH_REQUIRE_THROWS(und.copy_(und));
  CATCH_REQUIRE_THROWS(und.copy_(ft));
  CATCH_REQUIRE_THROWS(ft.copy_(und));

  und.toBackend(Backend::Undefined);
  CATCH_REQUIRE_THROWS(und.toBackend(Backend::CPU));
  CATCH_REQUIRE_THROWS(ft.toBackend(Backend::Undefined));

  Tensor to_move = ones({1}, CPU(kFloat));
  Tensor m(std::move(to_move));
  CATCH_REQUIRE(!to_move.defined());
  CATCH_REQUIRE(to_move.unsafeGetTensorImpl() == UndefinedTensorImpl::singleton());
}
