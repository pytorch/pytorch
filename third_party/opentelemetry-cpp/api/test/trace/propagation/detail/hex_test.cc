// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#include "opentelemetry/trace/propagation/detail/hex.h"

#include <map>

#include <gtest/gtest.h>

using namespace opentelemetry;

TEST(HexTest, ConvertOddLength)
{
  const int kLength        = 16;
  std::string trace_id_hex = "78cfcfec62ae9e9";
  uint8_t trace_id[kLength];
  trace::propagation::detail::HexToBinary(trace_id_hex, trace_id, sizeof(trace_id));

  const uint8_t expected_trace_id[kLength] = {0x0, 0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
                                              0x7, 0x8c, 0xfc, 0xfe, 0xc6, 0x2a, 0xe9, 0xe9};

  for (int i = 0; i < kLength; ++i)
  {
    EXPECT_EQ(trace_id[i], expected_trace_id[i]);
  }
}

TEST(HexTest, ConvertEvenLength)
{
  const int kLength        = 16;
  std::string trace_id_hex = "078cfcfec62ae9e9";
  uint8_t trace_id[kLength];
  trace::propagation::detail::HexToBinary(trace_id_hex, trace_id, sizeof(trace_id));

  const uint8_t expected_trace_id[kLength] = {0x0, 0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
                                              0x7, 0x8c, 0xfc, 0xfe, 0xc6, 0x2a, 0xe9, 0xe9};

  for (int i = 0; i < kLength; ++i)
  {
    EXPECT_EQ(trace_id[i], expected_trace_id[i]);
  }
}
