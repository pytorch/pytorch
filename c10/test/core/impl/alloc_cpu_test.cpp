#include <c10/core/impl/alloc_cpu.h>

#include <gtest/gtest.h>

#include <cstdint>

TEST(AllocCpuTest, ReleaseUnusedCpuMemoryPreservesLiveAllocations) {
  constexpr size_t size = 1024;
  auto* data = static_cast<uint8_t*>(c10::alloc_cpu(size));
  data[0] = 0x12;
  data[size - 1] = 0x34;

  const bool collection_supported = c10::release_unused_cpu_memory();
#if defined(USE_MIMALLOC) && !defined(__ANDROID__)
  EXPECT_TRUE(collection_supported);
#else
  EXPECT_FALSE(collection_supported);
#endif

  EXPECT_EQ(data[0], 0x12);
  EXPECT_EQ(data[size - 1], 0x34);
  c10::free_cpu(data);

  // Repeated collection after freeing allocations must also be safe.
  c10::release_unused_cpu_memory();
  c10::release_unused_cpu_memory();
}
