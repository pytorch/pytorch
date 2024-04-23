// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#include "opentelemetry/nostd/function_ref.h"

#include <gtest/gtest.h>
using namespace opentelemetry::nostd;

int Call(function_ref<int()> f)
{
  return f();
}

int Return3()
{
  return 3;
}

TEST(FunctionRefTest, Call)
{
  int x = 9;

  auto f = [&] { return x; };
  EXPECT_EQ(Call(f), 9);

  EXPECT_EQ(Call(Return3), 3);
}

TEST(FunctionRefTest, BoolConversion)
{
  auto f = [] { return 0; };
  function_ref<int()> fref1{nullptr};
  function_ref<int()> fref2{f};
  EXPECT_TRUE(!static_cast<bool>(fref1));
  EXPECT_TRUE(static_cast<bool>(fref2));
}
