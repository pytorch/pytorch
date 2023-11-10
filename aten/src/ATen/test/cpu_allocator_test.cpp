#include <gtest/gtest.h>

#include <c10/core/CPUAllocator.h>
#include <ATen/ATen.h>

#include <ATen/test/allocator_copy_data_test.h>

TEST(AllocatorTestCPU, test_copy_data) {
  test_allocator_copy_data(c10::GetDefaultCPUAllocator());
}
