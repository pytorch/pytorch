#include <gtest/gtest.h>

#include <torch/headeronly/core/MemoryFormat.h>

TEST(TestMemoryFormat, TestMemoryFormat) {
  using torch::headeronly::MemoryFormat;
  constexpr MemoryFormat expected_memory_formats[] = {
      MemoryFormat::Contiguous,
      MemoryFormat::Preserve,
      MemoryFormat::ChannelsLast,
      MemoryFormat::ChannelsLast3d,
  };
  for (int8_t i = 0; i < static_cast<int8_t>(MemoryFormat::NumOptions); i++) {
    EXPECT_EQ(static_cast<MemoryFormat>(i), expected_memory_formats[i]);
  }
}

TEST(TestMemoryFormat, get_contiguous_memory_format) {
  using torch::headeronly::get_contiguous_memory_format;
  using torch::headeronly::MemoryFormat;

  EXPECT_EQ(get_contiguous_memory_format(), MemoryFormat::Contiguous);
}
