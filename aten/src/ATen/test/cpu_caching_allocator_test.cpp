#include <gtest/gtest.h>

#include <ATen/cpu/vec256/vec256.h>
#include <ATen/ATen.h>

#include <c10/core/CPUCachingAllocator.h>

TEST(CPUCachingAllocatorTest, check_alloc_free) {
  c10::WithCPUCachingAllocatorGuard cachine_allocator_guard;
  at::Tensor a = at::rand({23, 23});
  float* data_ptr = a.data_ptr<float>();
  a.reset();
  a = at::rand({23, 23});
  ASSERT_TRUE(data_ptr == a.data_ptr<float>());
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  at::manual_seed(42);
  return RUN_ALL_TESTS();
}
