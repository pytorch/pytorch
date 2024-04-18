#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <ATen/test/allocator_clone_test.h>

TEST(AllocatorTestCUDA, test_clone) {
  test_allocator_clone(c10::cuda::CUDACachingAllocator::get());
}
