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

at::Tensor nvshmem_broadcast(at::Tensor& input, const std::string& group_name);

at::Tensor nvshmem_reduce_scatter_out(
    at::Tensor& input,
    std::string group_name,
    at::Tensor& out);

at::Tensor nvshmem_sendrecv(
    at::Tensor& input,
    at::Tensor& out,
    std::string group_name);


at::Tensor nvshmem_all_to_all(
    at::Tensor& input,
    at::Tensor& out,
    std::string group_name);

} // namespace c10d::nvshmem_extension
