// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#include "opentelemetry/trace/span_id.h"

#include <cstring>
#include <string>

#include <gtest/gtest.h>

namespace
{

using opentelemetry::trace::SpanId;

std::string Hex(const opentelemetry::trace::SpanId &span)
{
  char buf[16];
  span.ToLowerBase16(buf);
  return std::string(buf, sizeof(buf));
}

TEST(SpanIdTest, DefaultConstruction)
{
  SpanId id;
  EXPECT_FALSE(id.IsValid());
  EXPECT_EQ("0000000000000000", Hex(id));
}

TEST(SpanIdTest, ValidId)
{
  constexpr uint8_t buf[] = {1, 2, 3, 4, 5, 6, 7, 8};
  SpanId id(buf);
  EXPECT_TRUE(id.IsValid());
  EXPECT_EQ("0102030405060708", Hex(id));
  EXPECT_NE(SpanId(), id);
  EXPECT_EQ(SpanId(buf), id);
}

TEST(SpanIdTest, LowercaseBase16)
{
  constexpr uint8_t buf[] = {1, 2, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff};
  SpanId id(buf);
  EXPECT_TRUE(id.IsValid());
  EXPECT_EQ("0102aabbccddeeff", Hex(id));
  EXPECT_NE(SpanId(), id);
  EXPECT_EQ(SpanId(buf), id);
}

TEST(SpanIdTest, CopyBytesTo)
{
  constexpr uint8_t src[] = {1, 2, 3, 4, 5, 6, 7, 8};
  SpanId id(src);
  uint8_t buf[8];
  id.CopyBytesTo(buf);
  EXPECT_TRUE(memcmp(src, buf, 8) == 0);
}
}  // namespace
