#define CATCH_CONFIG_MAIN
#include "catch_utils.hpp"

#include "ATen/ATen.h"

#include <iostream>
#include <chrono>
#include <sstream>

using at::Tensor;
using at::WeakTensor;

CATCH_TEST_CASE( "Weak pointer tests", "" ) {
  CATCH_SECTION("gets invalidated") {
    Tensor a = at::ones({2, 2});
    WeakTensor b = a;
    a.reset();
    CATCH_REQUIRE_FALSE(b.lock().defined());
  }

  CATCH_SECTION("can successfully lock") {
    Tensor a = at::ones({2, 2});
    WeakTensor b = a;
    auto c = b.lock();
    CATCH_REQUIRE(c.defined());

    a.reset();
    CATCH_REQUIRE(b.lock().defined());
    c.reset();
    CATCH_REQUIRE_FALSE(b.lock().defined());
  }

  CATCH_SECTION("updates refcounts correctly") {
    Tensor a = at::ones({2, 2});
    CATCH_REQUIRE(a.use_count() == 1);
    CATCH_REQUIRE(a.weak_use_count() == 1);
    {
      WeakTensor b = a;
      CATCH_REQUIRE(a.use_count() == 1);
      CATCH_REQUIRE(a.weak_use_count() == 2);
    }
    CATCH_REQUIRE(a.use_count() == 1);
    CATCH_REQUIRE(a.weak_use_count() == 1);
    {
      WeakTensor b = a;
      CATCH_REQUIRE(a.use_count() == 1);
      auto locked = b.lock();
      CATCH_REQUIRE(locked.defined());
      CATCH_REQUIRE(a.use_count() == 2);
    }
    CATCH_REQUIRE(a.use_count() == 1);
    CATCH_REQUIRE(a.weak_use_count() == 1);
    {
      WeakTensor b = a;
      CATCH_REQUIRE(a.use_count() == 1);
      CATCH_REQUIRE(a.weak_use_count() == 2);
      a.reset();
      CATCH_REQUIRE(b.use_count() == 0);
      CATCH_REQUIRE(b.weak_use_count() == 1);
    }
  }
}
