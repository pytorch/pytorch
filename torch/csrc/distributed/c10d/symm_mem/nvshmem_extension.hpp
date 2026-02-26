#pragma once

#include <ATen/ATen.h>
#include <c10/macros/Macros.h>

#define NVSHMEM_CHECK(stmt, msg)                                             \
  do {                                                                       \
    int result = (stmt);                                                     \
    TORCH_CHECK(                                                             \
        result == 0,                                                         \
        std::string(__FILE__) + ":" + std::to_string(__LINE__) + " " + msg + \
            ". Error code: " + std::to_string(result));                      \
  } while (0)

namespace c10d::nvshmem_extension {

// Check if NVSHMEM is available
TORCH_API bool is_nvshmem_available();

// Initializes the device state in CUmodule so that it’s able to perform NVSHMEM
// operations.
TORCH_API void nvshmemx_cumodule_init(uintptr_t module);

TORCH_API void nvshmem_put(at::Tensor& tensor, const int64_t peer);

TORCH_API void nvshmem_get(at::Tensor& tensor, const int64_t peer);

at::Tensor nvshmem_broadcast(
    at::Tensor& input,
    const int64_t root,
    const std::string& group_name);

TORCH_API void nvshmem_wait_for_signal(
    at::Tensor& sigpad,
    int64_t signal,
    int64_t peer);

TORCH_API void nvshmem_put_with_signal(
    at::Tensor& tensor,
    at::Tensor& sigpad,
    int64_t signal,
    int64_t peer);

at::Tensor nvshmem_all_to_all(
    at::Tensor& input,
    at::Tensor& out,
    std::string group_name);

void all_to_all_vdev(
    at::Tensor& input,
    at::Tensor& out,
    at::Tensor& in_splits,
    at::Tensor& out_splits_offsets,
    std::string group_name);

void all_to_all_vdev_2d(
    at::Tensor& input,
    at::Tensor& out,
    at::Tensor& in_splits,
    at::Tensor& out_splits_offsets,
    std::string group_name,
    std::optional<int64_t> major_align = std::nullopt);

void all_to_all_vdev_2d_offset(
    at::Tensor& input,
    at::Tensor& out,
    at::Tensor& in_splits_offsets,
    at::Tensor& out_splits_offsets,
    std::string group_name);

void tile_reduce(
    at::Tensor& in_tile,
    at::Tensor& out_tile,
    int64_t root,
    std::string group_name,
    std::string reduce_op = "sum");

void multi_root_tile_reduce(
    at::ArrayRef<at::Tensor> in_tiles,
    at::Tensor& out_tile,
    at::ArrayRef<int64_t> roots,
    std::string group_name,
    std::string reduce_op = "sum");

} // namespace c10d::nvshmem_extension
