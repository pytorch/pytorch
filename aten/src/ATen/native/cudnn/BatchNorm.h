#pragma once

#if AT_CUDNN_ENABLED()

#include <ATen/cudnn/Descriptors.h>

namespace at::native {

namespace {

cudnnBatchNormMode_t getCudnnBatchNormMode(
    bool training,
    at::MemoryFormat memory_format,
    int64_t dim);

}

TORCH_API size_t _get_cudnn_batch_norm_reserve_space_size(const Tensor& input_t);

}

#endif

