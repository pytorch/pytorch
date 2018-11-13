#pragma once

#include "fbgemm/Fbgemm.h"

namespace caffe2 {

/**
 * If there's an existing packed matrix for the same matrix, reuse it.
 * Create a new one otherwise. This can save memory usage if many threads are
 * sharing the same weight.
 */
template <typename ACC_T>
std::shared_ptr<fbgemm::PackBMatrix<int8_t, ACC_T>>
GetOrCreateFbgemmPackBMatrix(
    fbgemm::matrix_op_t trans,
    std::int32_t m,
    std::int32_t n,
    const void* orig_data,
    const std::int8_t* quantized_data,
    std::int32_t ld,
    std::int32_t zero_point);

} // namespace caffe2
