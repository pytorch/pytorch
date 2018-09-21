#define CATCH_CONFIG_MAIN
#include "catch_utils.hpp"

#include "ATen/ATen.h"
#include "ATen/optional.h"

#include <assert.h>

using namespace at;

CATCH_TEST_CASE( "optional in cuda files", "[cuda]" ) {
  at::optional<int64_t> trivially_destructible;
  at::optional<std::vector<int64_t>> non_trivially_destructible;
  CATCH_REQUIRE(!trivially_destructible.has_value());
  CATCH_REQUIRE(!non_trivially_destructible.has_value());

  trivially_destructible = {5};
  non_trivially_destructible = std::vector<int64_t>{5, 10};
  CATCH_REQUIRE(trivially_destructible.has_value());
  CATCH_REQUIRE(non_trivially_destructible.has_value());
}

