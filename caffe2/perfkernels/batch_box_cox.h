// Impmenets BoxCox operator for CPU
#pragma once
#include <cstdint>

namespace caffe2 {

template <typename T>
void compute_batch_box_cox(
    std::size_t N,
    std::size_t D,
    std::size_t block_size,
    const T* self_data,
    const T* lambda1_data,
    const T* lambda2_data,
    T* output_data);

extern template void compute_batch_box_cox<float>(
    std::size_t N,
    std::size_t D,
    std::size_t block_size,
    const float* data,
    const float* lambda1_data,
    const float* lambda2_data,
    float* output_data);

extern template void compute_batch_box_cox<double>(
    std::size_t N,
    std::size_t D,
    std::size_t block_size,
    const double* data,
    const double* lambda1_data,
    const double* lambda2_data,
    double* output_data);

} // namespace caffe2
