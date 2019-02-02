#pragma once

// This file defines assertion macros that work in both gtest and non-gtest
// builds, and has some common includes.
//
// To add a new test file:
// 1. Add a test_foo.h file in this directory
// 2. include test_base.h
// 3. Write your tests as pure functions
// 4. Include test_foo.h in gtest.cpp and no-gtest.cpp and register the tests
//    there.
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/operator.h"

#if defined(USE_GTEST)
#include <gtest/gtest.h>
#include <test/cpp/common/support.h>
#else
#include "c10/util/Exception.h"
#define ASSERT_EQ(x, y) AT_ASSERT((x) == (y))
#define ASSERT_NE(x, y) AT_ASSERT((x) != (y))
#define ASSERT_TRUE AT_ASSERT
#define ASSERT_FALSE(x) ASSERT_TRUE(!(x))
#define ASSERT_THROWS_WITH(statement, substring)                         \
  try {                                                                  \
    (void)statement;                                                     \
    ASSERT_TRUE(false);                                                  \
  } catch (const std::exception& e) {                                    \
    ASSERT_NE(std::string(e.what()).find(substring), std::string::npos); \
  }
#define ASSERT_ANY_THROW(statement)   \
  bool threw = false;                 \
  try {                               \
    (void)statement;                  \
  } catch (const std::exception& e) { \
    threw = true;                     \
  }                                   \
  ASSERT_TRUE(threw);

#endif // defined(USE_GTEST)

bool isSandcastle() {
  return ((std::getenv("SANDCASTLE")) || \
    (std::getenv("TW_JOB_USER") && std::string(std::getenv("TW_JOB_USER")) == "sandcastle"));
}
