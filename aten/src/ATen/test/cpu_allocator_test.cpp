#include <gtest/gtest.h>

#include <c10/core/CPUAllocator.h>
#include <ATen/ATen.h>

#include <ATen/test/allocator_clone_test.h>

TEST(AllocatorTestCPU, test_clone) {
  test_allocator_clone(c10::GetDefaultCPUAllocator());
}
