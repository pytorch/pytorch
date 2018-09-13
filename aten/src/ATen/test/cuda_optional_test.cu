#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "ATen/ATen.h"
#include "ATen/optional.h"

#include <assert.h>

using namespace at;

TEST_CASE( "optional in cuda files", "[cuda]" ) {
  at::optional<int64_t> trivially_destructible;
  at::optional<std::vector<int64_t>> non_trivially_destructible;
  REQUIRE(!trivially_destructible.has_value());
  REQUIRE(!non_trivially_destructible.has_value());

  trivially_destructible = {5};
  non_trivially_destructible = std::vector<int64_t>{5, 10};
  REQUIRE(trivially_destructible.has_value());
  REQUIRE(non_trivially_destructible.has_value());
}

