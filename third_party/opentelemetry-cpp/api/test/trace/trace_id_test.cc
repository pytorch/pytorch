// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#include "opentelemetry/trace/trace_id.h"

#include <cstring>
#include <string>

#include <gtest/gtest.h>

namespace
{

using opentelemetry::trace::TraceId;

std::string Hex(const opentelemetry::trace::TraceId &trace)
{
  char buf[32];
  trace.ToLowerBase16(buf);
  return std::string(buf, sizeof(buf));
}

TEST(TraceIdTest, DefaultConstruction)
{
  TraceId id;
  EXPECT_FALSE(id.IsValid());
  EXPECT_EQ("00000000000000000000000000000000", Hex(id));
}

TEST(TraceIdTest, ValidId)
{
  constexpr uint8_t buf[] = {1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1};
  TraceId id(buf);
  EXPECT_TRUE(id.IsValid());
  EXPECT_EQ("01020304050607080807060504030201", Hex(id));
  EXPECT_NE(TraceId(), id);
  EXPECT_EQ(TraceId(buf), id);
}

TEST(TraceIdTest, LowercaseBase16)
{
  constexpr uint8_t buf[] = {1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff};
  TraceId id(buf);
  EXPECT_TRUE(id.IsValid());
  EXPECT_EQ("01020304050607080807aabbccddeeff", Hex(id));
  EXPECT_NE(TraceId(), id);
  EXPECT_EQ(TraceId(buf), id);
}

TEST(TraceIdTest, CopyBytesTo)
{
  constexpr uint8_t src[] = {1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1};
  TraceId id(src);
  uint8_t buf[TraceId::kSize];
  id.CopyBytesTo(buf);
  EXPECT_TRUE(memcmp(src, buf, sizeof(buf)) == 0);
}
}  // namespace
