#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <ATen/test/allocator_clone_test.h>

#include <torch/csrc/cuda/CUDAPluggableAllocator.h>

TEST(AllocatorTestCUDA, test_clone) {
  test_allocator_clone(c10::cuda::CUDACachingAllocator::get());
}

void* dummy_alloc_0(size_t size, int device, void* stream) {return nullptr;}
void dummy_free_0(void* ptr) {}
void dummy_free_1(void* ptr) {}

// Tests that data_ptrs have their respective deleters
// when mixing allocators
TEST(AllocatorTestCUDA, test_pluggable_allocator) {
  // Create a tensor with dummy_allocator_0, where dummy_free_0 is the deleter
  auto dummy_allocator_0 = torch::cuda::CUDAPluggableAllocator::createCustomAllocator(dummy_alloc_0, dummy_free_0);
  c10::cuda::CUDACachingAllocator::allocator.store(dummy_allocator_0.get());
  at::Tensor a = at::empty({0}, at::TensorOptions().device(at::kCUDA));

  // Create a tensor with dummy_allocator_1, where dummy_free_1 is the deleter
  auto dummy_allocator_1 = torch::cuda::CUDAPluggableAllocator::createCustomAllocator(dummy_alloc_0, dummy_free_1);
  c10::cuda::CUDACachingAllocator::allocator.store(dummy_allocator_1.get());
  at::Tensor b = at::empty({0}, at::TensorOptions().device(at::kCUDA));

  // a should have dummy_free_0 as the deleter
  // b should have dummy_free_1 as the deleter
  // a and b's deleters are not the same function pointers
  ASSERT_TRUE(a.storage().data_ptr().get_deleter() == &dummy_free_0);
  ASSERT_TRUE(b.storage().data_ptr().get_deleter() == &dummy_free_1);
  ASSERT_TRUE(a.storage().data_ptr().get_deleter() != b.storage().data_ptr().get_deleter());
}
