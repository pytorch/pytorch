#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <ATen/test/allocator_copy_data_test.h>

TEST(AllocatorTestCUDA, test_copy_data) {
  test_allocator_copy_data(c10::cuda::CUDACachingAllocator::get());
}
