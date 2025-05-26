#pragma once
#include <gtest/gtest.h>
#include <ATen/ATen.h>

void test_allocator_clone(c10::Allocator* allocator) {
  ASSERT_TRUE(allocator != nullptr);

  c10::Storage a_storage(c10::make_intrusive<c10::StorageImpl>(
    c10::StorageImpl::use_byte_size_t(),
    0,
    allocator,
    /*resizable=*/true));

  c10::Storage b_storage(c10::make_intrusive<c10::StorageImpl>(
    c10::StorageImpl::use_byte_size_t(),
    0,
    allocator,
    /*resizable=*/true));

  at::Tensor a = at::empty({0}, at::TensorOptions().device(a_storage.device())).set_(a_storage);
  at::Tensor b = at::empty({0}, at::TensorOptions().device(b_storage.device())).set_(b_storage);

  std::vector<int64_t> sizes({13, 4, 5});

  at::rand_out(a, sizes);
  at::rand_out(b, sizes);

  ASSERT_TRUE(a_storage.nbytes() == static_cast<size_t>(a.numel() * a.element_size()));
  ASSERT_TRUE(a_storage.nbytes() == b_storage.nbytes());

  void* a_data_ptr = a_storage.mutable_data();
  b_storage.set_data_ptr(allocator->clone(a_data_ptr, a_storage.nbytes()));

  ASSERT_TRUE((a == b).all().item<bool>());
}
