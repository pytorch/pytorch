// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#include "opentelemetry/version.h"

#include <gtest/gtest.h>

TEST(VersionTest, Consistency)
{
  char expected[10];
  snprintf(expected, sizeof(expected), "%d.%d.%d", OPENTELEMETRY_VERSION_MAJOR,
           OPENTELEMETRY_VERSION_MINOR, OPENTELEMETRY_VERSION_PATCH);

  std::string actual = OPENTELEMETRY_VERSION;
  EXPECT_EQ(actual, expected);
}
