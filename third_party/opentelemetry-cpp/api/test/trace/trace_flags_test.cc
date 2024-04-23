// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#include "opentelemetry/trace/trace_flags.h"

#include <cstring>
#include <string>

#include <gtest/gtest.h>

namespace
{

using opentelemetry::trace::TraceFlags;

std::string Hex(const TraceFlags &flags)
{
  char buf[2];
  flags.ToLowerBase16(buf);
  return std::string(buf, sizeof(buf));
}

TEST(TraceFlagsTest, DefaultConstruction)
{
  TraceFlags flags;
  EXPECT_FALSE(flags.IsSampled());
  EXPECT_EQ(0, flags.flags());
  EXPECT_EQ("00", Hex(flags));
}

TEST(TraceFlagsTest, Sampled)
{
  TraceFlags flags{TraceFlags::kIsSampled};
  EXPECT_TRUE(flags.IsSampled());
  EXPECT_EQ(1, flags.flags());
  EXPECT_EQ("01", Hex(flags));

  uint8_t buf[1];
  flags.CopyBytesTo(buf);
  EXPECT_EQ(1, buf[0]);
}

}  // namespace
