#pragma once

#include <ATen/ATen.h>

#include <torch/csrc/distributed/c10d/Store.hpp>

namespace c10d::nvshmem_extension {

void initialize_nvshmem_with_store(
    c10::intrusive_ptr<c10d::Store> store,
    int rank,
    int world_size);

void* nvshmem_malloc(size_t size);

void* nvshmem_ptr(const void* dest, int pe);

at::Tensor nvshmem_hello(at::Tensor& input);

} // namespace c10d::nvshmem_extension
