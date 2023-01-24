#pragma once
#include <cstdint>

namespace caffe2 {
namespace perfkernels {

template <typename T>
void LstmUnitImpl__avx2_fma(
    const int N,
    const int D,
    const int t,
    const T* H_prev,
    const T* C_prev,
    const T* X,
    const int32_t* seqLengths,
    const bool drop_states,
    T* C,
    T* H,
    const float forget_bias);

template <typename T>
void LstmUnitGradientImpl__avx2_fma(
    int N,
    int D,
    int t,
    const T* C_prev,
    const T* X,
    const int32_t* seqLengths,
    const T* C,
    const T* H,
    const T* C_diff,
    const T* H_diff,
    bool drop_states,
    T* H_prev_diff,
    T* C_prev_diff,
    T* X_diff,
    const float forget_bias);

// Forward declaration of specialized functions
extern template void LstmUnitImpl__avx2_fma(
    const int N,
    const int D,
    const int t,
    const float* H_prev,
    const float* C_prev,
    const float* X,
    const int32_t* seqLengths,
    const bool drop_states,
    float* C,
    float* H,
    const float forget_bias);

extern template void LstmUnitGradientImpl__avx2_fma(
    int N,
    int D,
    int t,
    const float* C_prev,
    const float* X,
    const int32_t* seqLengths,
    const float* C,
    const float* H,
    const float* C_diff,
    const float* H_diff,
    bool drop_states,
    float* H_prev_diff,
    float* C_prev_diff,
    float* X_diff,
    const float forget_bias);

} // namespace perfkernels
} // namespace caffe2
