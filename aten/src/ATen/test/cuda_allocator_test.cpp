#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <ATen/test/allocator_clone_test.h>

#include <torch/csrc/cuda/CUDAPluggableAllocator.h>

TEST(AllocatorTestCUDA, test_clone) {
  test_allocator_clone(c10::cuda::CUDACachingAllocator::get());
}

static int called_dummy_free_0 = 0;
static int called_dummy_free_1 = 0;

void* dummy_alloc_0(size_t size, int device, void* stream) {return nullptr;}
void dummy_free_0(void* data, size_t size, int device, void* stream) {
  called_dummy_free_0++;
}
void dummy_free_1(void* data, size_t size, int device, void* stream) {
  called_dummy_free_1++;
}

// Tests that data_ptrs have their respective deleters
// when mixing allocators
TEST(AllocatorTestCUDA, test_pluggable_allocator_deleters) {
  // Create a tensor with dummy_allocator_0, where dummy_free_0 is the deleter
  auto dummy_allocator_0 = torch::cuda::CUDAPluggableAllocator::createCustomAllocator(dummy_alloc_0, dummy_free_0);
  c10::cuda::CUDACachingAllocator::allocator.store(dummy_allocator_0.get());
  at::Tensor a = at::empty({0}, at::TensorOptions().device(at::kCUDA));

  // Create a tensor with dummy_allocator_1, where dummy_free_1 is the deleter
  auto dummy_allocator_1 = torch::cuda::CUDAPluggableAllocator::createCustomAllocator(dummy_alloc_0, dummy_free_1);
  c10::cuda::CUDACachingAllocator::allocator.store(dummy_allocator_1.get());
  at::Tensor b = at::empty({0}, at::TensorOptions().device(at::kCUDA));

  // Manually use a's deleter
  auto* ctx = a.storage().data_ptr().get_context();
  a.storage().data_ptr().get_deleter()(ctx);
  a.storage().mutable_data_ptr().release_context();

  // a's deleter is dummy_free_0
  // dummy_free_0 should be called above, so called_dummy_free_0 should be 1
  ASSERT_TRUE(called_dummy_free_0 == 1);

  // Manually use b's deleter
  ctx = b.storage().data_ptr().get_context();
  b.storage().data_ptr().get_deleter()(ctx);
  b.storage().mutable_data_ptr().release_context();

  // b's deleter is dummy_free_1
  // dummy_free_1 should be called above, so called_dummy_free_1 should be 1
  ASSERT_TRUE(called_dummy_free_1 == 1);
}
