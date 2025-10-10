#include <gtest/gtest.h>

#include <ATen/cpu/vec/vec.h>
#include <ATen/ATen.h>

#include <c10/mobile/CPUCachingAllocator.h>

// At the moment caching allocator is only exposed to mobile cpu allocator.
#ifdef C10_MOBILE

TEST(CPUCachingAllocatorTest, check_alloc_free) {
  c10::CPUCachingAllocator caching_allocator;
  c10::WithCPUCachingAllocatorGuard cachine_allocator_guard(
      &caching_allocator);
  at::Tensor a = at::rand({23, 23});
  float* data_ptr = a.data_ptr<float>();
  a.reset();
  a = at::rand({23, 23});
  ASSERT_TRUE(data_ptr == a.data_ptr<float>());
}

// This should just free the pointer correctly.
TEST(CPUCachingAllocatorTest, check_alloc_outside_free_inside) {
  c10::CPUCachingAllocator caching_allocator;
  at::Tensor a = at::rand({23, 23});
  {
    c10::WithCPUCachingAllocatorGuard cachine_allocator_guard(
        &caching_allocator);
    [[maybe_unused]] float* data_ptr = a.data_ptr<float>();
    a.reset();
    a = at::rand({23, 23});
  }
}

TEST(CPUCachingAllocatorTest, check_alloc_inside_free_outside) {
  c10::CPUCachingAllocator caching_allocator;
  at::Tensor a;
  {
    c10::WithCPUCachingAllocatorGuard cachine_allocator_guard(
        &caching_allocator);
    a = at::rand({23, 23});
  }
  a.reset();
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  at::manual_seed(42);
  return RUN_ALL_TESTS();
}

#endif /* C10_Mobile */
