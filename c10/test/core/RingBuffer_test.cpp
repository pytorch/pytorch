#include <c10/core/RingBuffer.h>

#include <gtest/gtest.h>

#include <vector>

namespace {

TEST(RingBufferTest, PreservesChronologicalOrderWhenWrapping) {
  c10::RingBuffer<int> buffer;
  buffer.setMaxEntries(3);

  for (int value = 1; value <= 5; ++value) {
    buffer.insertEntries(value);
  }

  std::vector<int> entries;
  buffer.getEntries(entries);
  EXPECT_EQ(entries, (std::vector<int>{3, 4, 5}));
}

TEST(RingBufferTest, PreservesOrderWhenCapacityChanges) {
  c10::RingBuffer<int> buffer;
  buffer.setMaxEntries(3);
  for (int value = 1; value <= 5; ++value) {
    buffer.insertEntries(value);
  }

  buffer.setMaxEntries(5);
  buffer.insertEntries(6);
  std::vector<int> entries;
  buffer.getEntries(entries);
  EXPECT_EQ(entries, (std::vector<int>{3, 4, 5, 6}));

  buffer.setMaxEntries(2);
  entries.clear();
  buffer.getEntries(entries);
  EXPECT_EQ(entries, (std::vector<int>{5, 6}));

  buffer.insertEntries(7);
  entries.clear();
  buffer.getEntries(entries);
  EXPECT_EQ(entries, (std::vector<int>{6, 7}));
}

TEST(RingBufferTest, ClearAllowsBufferToBeReused) {
  c10::RingBuffer<int> buffer;
  buffer.setMaxEntries(2);
  buffer.insertEntries(1);
  buffer.insertEntries(2);

  buffer.clear();
  buffer.insertEntries(3);

  std::vector<int> entries;
  buffer.getEntries(entries);
  EXPECT_EQ(entries, (std::vector<int>{3}));
}

} // namespace
