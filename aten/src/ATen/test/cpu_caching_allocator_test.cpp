#include <gtest/gtest.h>

#include <ATen/cpu/vec256/vec256.h>
#include <ATen/ATen.h>

#include <c10/core/CPUCachingAllocator.h>

TEST(CPUCachingAllocatorTest, check_alloc_free) {
  std::unique_ptr<c10::CPUCachingAllocator> caching_allocator =
    std::make_unique<c10::CPUCachingAllocator>();
  c10::WithCPUCachingAllocatorGuard cachine_allocator_guard(
      caching_allocator.get());
  at::Tensor a = at::rand({23, 23});
  float* data_ptr = a.data_ptr<float>();
  a.reset();
  a = at::rand({23, 23});
  ASSERT_TRUE(data_ptr == a.data_ptr<float>());
}

// This should just free the pointer correctly.
TEST(CPUCachingAllocatorTest, check_alloc_outside_free_inside) {
  std::unique_ptr<c10::CPUCachingAllocator> caching_allocator =
    std::make_unique<c10::CPUCachingAllocator>();
  at::Tensor a = at::rand({23, 23});
  {
    c10::WithCPUCachingAllocatorGuard cachine_allocator_guard(
        caching_allocator.get());
    float* data_ptr = a.data_ptr<float>();
    a.reset();
    a = at::rand({23, 23});
  }
}

TEST(CPUCachingAllocatorTest, check_alloc_inside_free_outside) {
  std::unique_ptr<c10::CPUCachingAllocator> caching_allocator =
    std::make_unique<c10::CPUCachingAllocator>();
  at::Tensor a;
  {
    c10::WithCPUCachingAllocatorGuard cachine_allocator_guard(
        caching_allocator.get());
    a = at::rand({23, 23});
  }
  a.reset();
}

TEST(CPUCachingAllocatorTest, enable_without_allocator) {
  std::function<void()> tmp_lambda = [](){
    at::Tensor a;
    {
      c10::GetThreadLocalCachingAllocatorInfo().enable();
      a = at::rand({23, 23});
    }
    a.reset();
    a = at::rand({23, 23});
  };
  ASSERT_THROW(tmp_lambda(), c10::Error);
}

TEST(CPUCachingAllocatorTest, enable_disable_blocks) {
  std::unique_ptr<c10::CPUCachingAllocator> caching_allocator =
    std::make_unique<c10::CPUCachingAllocator>();
  c10::WithCPUCachingAllocatorGuard cachine_allocator_guard(
      caching_allocator.get(), false);
  at::Tensor a;
  {
    c10::GetThreadLocalCachingAllocatorInfo().enable();
    a = at::rand({23, 23});
    float* data_ptr = a.data_ptr<float>();
    a.reset();
    a = at::rand({23, 23});
    ASSERT_TRUE(data_ptr == a.data_ptr<float>());
    // Cache the same pointer again.
    a.reset();
    // Disable caching.
    c10::GetThreadLocalCachingAllocatorInfo().disable();
    a = at::rand({23, 23});
    // This time we should get a new pointer.
    ASSERT_TRUE(data_ptr != a.data_ptr<float>());
  }
  a.reset();
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  at::manual_seed(42);
  return RUN_ALL_TESTS();
}
