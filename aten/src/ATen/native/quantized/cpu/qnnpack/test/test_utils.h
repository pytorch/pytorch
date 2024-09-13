/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <gtest/gtest.h>

namespace qnnpack {
namespace testing {

enum class Mode {
  Static,
  Runtime,
};

#define _MAKE_TEST(TestClass, test_name, test_body, ...)  \
  TEST(TestClass, test_name) {                            \
    test_body.testQ8(__VA_ARGS__);                        \
  }

#define _STATIC_TEST(TestClass, test_name, test_body)                   \
  _MAKE_TEST(TestClass, test_name##_static, test_body, Mode::Static)

#define _RUNTIME_TEST(TestClass, test_name, test_body)                  \
  _MAKE_TEST(TestClass, test_name##_runtime, test_body, Mode::Runtime)

#define _STATIC_AND_RUNTIME_TEST(TestClass, test_name, test_body) \
  _STATIC_TEST(TestClass, test_name, test_body)                   \
  _RUNTIME_TEST(TestClass, test_name, test_body)

}}  // namespace qnnpack::testing
