#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <cstdint>

namespace torch {
namespace jit {
namespace tensorexpr {

TORCH_API void nnc_aten_conv2d(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args);

TORCH_API void nnc_aten_matmul(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
