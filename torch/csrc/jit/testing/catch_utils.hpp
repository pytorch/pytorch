#pragma once

#define CATCH_CONFIG_PREFIX_ALL
#include <catch.hpp>

// CATCH_REQUIRE_THROWS is not defined identically to REQUIRE_THROWS and causes
// warning; define our own version that doesn't warn.
#define _CATCH_REQUIRE_THROWS(...) \
  INTERNAL_CATCH_THROWS(           \
      "CATCH_REQUIRE_THROWS", Catch::ResultDisposition::Normal, __VA_ARGS__)
