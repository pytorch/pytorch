#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "ATen/ATen.h"

#include <iostream>
#include <chrono>
#include <sstream>

using at::Tensor;
using at::WeakTensor;

TEST_CASE( "Weak pointer tests", "" ) {
  SECTION("gets invalidated") {
    Tensor a = at::ones({2, 2});
    WeakTensor b = a;
    a.reset();
    REQUIRE_FALSE(b.lock().defined());
  }

  SECTION("can successfully lock") {
    Tensor a = at::ones({2, 2});
    WeakTensor b = a;
    auto c = b.lock();
    REQUIRE(c.defined());

    a.reset();
    REQUIRE(b.lock().defined());
    c.reset();
    REQUIRE_FALSE(b.lock().defined());
  }

  SECTION("updates refcounts correctly") {
    Tensor a = at::ones({2, 2});
    auto ai = a.unsafeGetTensorImpl();
    REQUIRE(ai->use_count() == 1);
    REQUIRE(ai->weak_use_count() == 1);
    {
      WeakTensor b = a;
      REQUIRE(ai->use_count() == 1);
      REQUIRE(ai->weak_use_count() == 2);
    }
    REQUIRE(ai->use_count() == 1);
    REQUIRE(ai->weak_use_count() == 1);
    {
      WeakTensor b = a;
      REQUIRE(ai->use_count() == 1);
      auto locked = b.lock();
      REQUIRE(locked.defined());
      REQUIRE(ai->use_count() == 2);
    }
    REQUIRE(ai->use_count() == 1);
    REQUIRE(ai->weak_use_count() == 1);
    {
      WeakTensor b = a;
      REQUIRE(ai->use_count() == 1);
      REQUIRE(ai->weak_use_count() == 2);
      a.reset();
      auto bi = b.unsafeGetTensorImpl();
      REQUIRE(bi->use_count() == 0);
      REQUIRE(bi->weak_use_count() == 1);
    }
  }
}
