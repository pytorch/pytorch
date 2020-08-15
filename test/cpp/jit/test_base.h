#pragma once

// This file defines assertion macros that work in both gtest and non-gtest
// builds, and has some common includes.
#include "torch/csrc/jit/ir/ir.h"
#include "torch/csrc/jit/runtime/operator.h"

#if defined(USE_GTEST)
#include <gtest/gtest.h>
#include <test/cpp/common/support.h>
#else
#include "c10/util/Exception.h"
#define ASSERT_EQ(x, y) TORCH_INTERNAL_ASSERT((x) == (y))
#define ASSERT_NE(x, y) TORCH_INTERNAL_ASSERT((x) != (y))
#define ASSERT_TRUE TORCH_INTERNAL_ASSERT
#define ASSERT_FALSE(x) ASSERT_TRUE(!(x))
#define ASSERT_THROWS_WITH(statement, substring)                         \
  try {                                                                  \
    (void)statement;                                                     \
    ASSERT_TRUE(false);                                                  \
  } catch (const std::exception& e) {                                    \
    ASSERT_NE(std::string(e.what()).find(substring), std::string::npos); \
  }
#define ASSERT_ANY_THROW(statement)     \
  {                                     \
    bool threw = false;                 \
    try {                               \
      (void)statement;                  \
    } catch (const std::exception& e) { \
      threw = true;                     \
    }                                   \
    ASSERT_TRUE(threw);                 \
  }

#endif // defined(USE_GTEST)

static inline bool isSandcastle() {
  return (
      (std::getenv("SANDCASTLE")) ||
      (std::getenv("TW_JOB_USER") &&
       std::string(std::getenv("TW_JOB_USER")) == "sandcastle"));
}
