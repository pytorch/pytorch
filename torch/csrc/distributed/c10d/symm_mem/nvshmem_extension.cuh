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

// Intializes the device state in CUmodule so that it’s able to perform NVSHMEM
// operations.
TORCH_API void nvshmemx_cumodule_init(uintptr_t module);

at::Tensor nvshmem_broadcast(at::Tensor& input, const std::string& group_name);

at::Tensor nvshmem_all_to_all(
    at::Tensor& input,
    at::Tensor& out,
    std::string group_name);

at::Tensor nvshmem_all_to_all_vdev(
    at::Tensor& input,
    at::Tensor& out,
    at::Tensor& in_out_splits,
    std::string group_name);

at::Tensor nvshmem_all_to_all_vdev_2d(
    at::Tensor& input,
    at::Tensor& out,
    at::Tensor& in_out_splits,
    std::string group_name,
    std::optional<int64_t> major_align = std::nullopt);

} // namespace c10d::nvshmem_extension
